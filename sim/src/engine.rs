//! Action-phase engine. Frame order EMPIRICAL (replay-confirmed; the rules page lists
//! shields first but the engine grants at post-move positions): movement (incl.
//! breach-on-next-attempt + self-destruct) -> shields -> attacks (creation order,
//! sequential damage, 0-health untargetable) -> deaths (repath on structure death).

use crate::config::{self, Config};
use crate::geo;
use crate::path::{self, NavField};
use crate::state::{gi, Cmd, State, EMPTY};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------- events

#[derive(Clone, Debug, Default)]
pub struct Events {
    /// (owner, kind, x, y) — includes REMOVE(6)/UPGRADE(7) marks
    pub spawn: Vec<(u8, u8, i8, i8)>,
    /// (owner, kind, fx, fy, tx, ty)
    pub mv: Vec<(u8, u8, i8, i8, i8, i8)>,
    /// (giver_owner, giver_kind, sx, sy, tx, ty, amount) — field [3] of the engine's
    /// shield event is the GIVER's unit type (API doc says target type; replay says giver).
    pub shield: Vec<(u8, u8, i8, i8, i8, i8, f32)>,
    /// (attacker_owner, attacker_kind, ax, ay, tx, ty, damage)
    pub attack: Vec<(u8, u8, i8, i8, i8, i8, f32)>,
    /// (victim_owner, victim_kind, x, y, amount)
    pub damage: Vec<(u8, u8, i8, i8, f32)>,
    /// (owner, kind, x, y, was_player_removal)
    pub death: Vec<(u8, u8, i8, i8, bool)>,
    /// (owner, kind, x, y, breach_damage)
    pub breach: Vec<(u8, u8, i8, i8, f32)>,
    /// (owner, kind, x, y, damage, affected_locations sorted)
    pub self_destruct: Vec<(u8, u8, i8, i8, f32, Vec<(i8, i8)>)>,
}

impl Events {
    /// Canonical, serializer-tolerant form for diffing: floats go through display01 (the
    /// engine writes ceil-tenth displays of raw f64s), tuples become sorted strings.
    pub fn canon(&self) -> Vec<(&'static str, Vec<String>)> {
        // event floats are f32; shortest {:?} repr is identical on both sides
        #[inline]
        fn d(x: f32) -> f32 {
            x
        }
        let mut out: Vec<(&'static str, Vec<String>)> = Vec::with_capacity(8);
        macro_rules! push {
            ($name:expr, $it:expr) => {{
                let mut v: Vec<String> = $it;
                v.sort();
                out.push(($name, v));
            }};
        }
        push!("spawn", self.spawn.iter().map(|e| format!("{:?}", e)).collect());
        push!("move", self.mv.iter().map(|e| format!("{:?}", e)).collect());
        push!(
            "shield",
            self.shield
                .iter()
                .map(|e| format!("{:?}", (e.0, e.1, e.2, e.3, e.4, e.5, d(e.6))))
                .collect()
        );
        push!(
            "attack",
            self.attack
                .iter()
                .map(|e| format!("{:?}", (e.0, e.1, e.2, e.3, e.4, e.5, d(e.6))))
                .collect()
        );
        push!(
            "damage",
            self.damage.iter().map(|e| format!("{:?}", (e.0, e.1, e.2, e.3, d(e.4)))).collect()
        );
        push!("death", self.death.iter().map(|e| format!("{:?}", e)).collect());
        push!(
            "breach",
            self.breach.iter().map(|e| format!("{:?}", (e.0, e.1, e.2, e.3, d(e.4)))).collect()
        );
        push!(
            "selfDestruct",
            self.self_destruct
                .iter()
                .map(|e| format!("{:?}", (e.0, e.1, e.2, e.3, d(e.4), &e.5)))
                .collect()
        );
        out
    }
}

#[derive(Clone, Debug)]
pub struct UnitSnap {
    pub owner: u8,
    pub kind: u8,
    pub x: i8,
    pub y: i8,
    pub health: f32,
    pub upgraded: bool,
    pub pending: bool,
    pub remove_countdown: u32,
}

#[derive(Clone, Debug)]
pub struct FrameRec {
    /// [player][hp, sp, mp]
    pub stats: [[f32; 3]; 2],
    pub units: Vec<UnitSnap>,
    pub ev: Events,
}

#[derive(Clone, Debug, Default)]
pub struct TurnSummary {
    pub breaches: [f32; 2], // damage dealt BY player i
    pub structure_damage_dealt: [f32; 2],
    pub frames: u32,
}

// ------------------------------------------------------------------------ navigation

struct Nav {
    layout_version: u64,
    pocket: Vec<u16>,
    fields: HashMap<(u8, u16), Arc<NavField>>,
}

impl Nav {
    fn new() -> Nav {
        Nav { layout_version: u64::MAX, pocket: vec![0; 28 * 28], fields: HashMap::new() }
    }

    fn refresh(&mut self, st: &State) {
        if self.layout_version == st.layout_version {
            return;
        }
        self.layout_version = st.layout_version;
        self.fields.clear();
        // flood-fill pocket labels over unblocked in-bounds tiles
        for v in self.pocket.iter_mut() {
            *v = u16::MAX;
        }
        let mut next_label = 0u16;
        let mut stack: Vec<(i32, i32)> = Vec::new();
        for x in 0..28i32 {
            for y in 0..28i32 {
                if !geo::in_bounds(x, y)
                    || st.grid[gi(x as i8, y as i8)] != EMPTY
                    || self.pocket[(x * 28 + y) as usize] != u16::MAX
                {
                    continue;
                }
                stack.push((x, y));
                self.pocket[(x * 28 + y) as usize] = next_label;
                while let Some((cx, cy)) = stack.pop() {
                    for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                        let (nx, ny) = (cx + dx, cy + dy);
                        if geo::in_bounds(nx, ny)
                            && st.grid[gi(nx as i8, ny as i8)] == EMPTY
                            && self.pocket[(nx * 28 + ny) as usize] == u16::MAX
                        {
                            self.pocket[(nx * 28 + ny) as usize] = next_label;
                            stack.push((nx, ny));
                        }
                    }
                }
                next_label += 1;
            }
        }
    }

    fn field(&mut self, st: &State, x: i8, y: i8, edge: u8) -> Arc<NavField> {
        self.refresh(st);
        let key = (edge, self.pocket[(x as i32 * 28 + y as i32) as usize]);
        if let Some(f) = self.fields.get(&key) {
            return f.clone();
        }
        let blocked = |bx: i32, by: i32| st.grid[gi(bx as i8, by as i8)] != EMPTY;
        let f = Arc::new(path::compute_field(&blocked, x as i32, y as i32, edge));
        self.fields.insert(key, f.clone());
        f
    }
}

// ---------------------------------------------------------------------------- deploy

/// One player's commands for a turn: builds (walls/supports/turrets/removes/upgrades, in
/// submitted order) and deploys (mobiles, in submitted order). Optional forced engine ids
/// (replay mode).
#[derive(Clone, Debug, Default)]
pub struct TurnCommands {
    pub build: Vec<Cmd>,
    pub build_ids: Option<Vec<u32>>,
    pub deploy: Vec<Cmd>,
    pub deploy_ids: Option<Vec<u32>>,
}

fn spawn_kind_of(cmd: &Cmd) -> u8 {
    match *cmd {
        Cmd::Build { kind, .. } => kind,
        Cmd::Upgrade { .. } => config::UPGRADE,
        Cmd::Remove { .. } => config::REMOVE,
        Cmd::Deploy { kind, .. } => kind,
    }
}

fn cmd_loc(cmd: &Cmd) -> (i8, i8) {
    match *cmd {
        Cmd::Build { x, y, .. }
        | Cmd::Upgrade { x, y }
        | Cmd::Remove { x, y }
        | Cmd::Deploy { x, y, .. } => (x, y),
    }
}

// ------------------------------------------------------------------------- the engine

pub fn play_turn(
    st: &mut State,
    mut cmds: [TurnCommands; 2],
    strict: bool,
    frames: &mut Vec<FrameRec>,
) -> TurnSummary {
    let cfg = st.cfg.clone();
    let mut summary = TurnSummary::default();
    let mut ev0 = Events::default();

    // Deploy order: P1 builds, P2 builds, P1 deploys, P2 deploys (replay-confirmed).
    let mut accepted: Vec<(Cmd, u32)> = Vec::new();
    for p in 0..2u8 {
        let c = std::mem::take(&mut cmds[p as usize]);
        st.apply_commands(p, &c.build, c.build_ids.as_deref(), strict, &mut accepted);
        for (cmd, _id) in accepted.drain(..) {
            let (x, y) = cmd_loc(&cmd);
            ev0.spawn.push((p, spawn_kind_of(&cmd), x, y));
        }
        cmds[p as usize] = c;
    }
    for p in 0..2u8 {
        let c = std::mem::take(&mut cmds[p as usize]);
        st.apply_commands(p, &c.deploy, c.deploy_ids.as_deref(), strict, &mut accepted);
        for (cmd, _id) in accepted.drain(..) {
            let (x, y) = cmd_loc(&cmd);
            ev0.spawn.push((p, spawn_kind_of(&cmd), x, y));
        }
        cmds[p as usize] = c;
    }

    // Attack iteration order: creation order across structures and mobiles.
    // Structures from previous turns have smaller seq than anything new this turn.
    #[derive(Clone, Copy)]
    enum Ref {
        S(usize),
        M(usize),
    }
    let mut order: Vec<(u32, Ref)> = Vec::with_capacity(st.structures.len() + st.mobiles.len());
    for (i, s) in st.structures.iter().enumerate() {
        if s.alive {
            order.push((s.seq, Ref::S(i)));
        }
    }
    for (i, m) in st.mobiles.iter().enumerate() {
        order.push((m.seq, Ref::M(i)));
    }
    order.sort_by_key(|&(seq, _)| seq);

    let mut nav = Nav::new();
    let mut mobile_axis: Vec<u8> = vec![0; st.mobiles.len()]; // 0 none, 1 horiz, 2 vert

    let mut f: u32 = 0;
    loop {
        let mut ev = if f == 0 { std::mem::take(&mut ev0) } else { Events::default() };

        // ---- 1. movement (creation order)
        for mi in 0..st.mobiles.len() {
            let (alive, fpm, kind, owner, x, y, edge, steps, health) = {
                let m = &st.mobiles[mi];
                (m.alive, m.frames_per_move, m.kind, m.owner, m.x, m.y, m.target_edge, m.steps, m.health)
            };
            if !alive || health <= 0.0 {
                continue; // self-destructed earlier this frame chain? (dies at step 4)
            }
            if (f + 1) % fpm != 0 {
                continue;
            }
            let field = nav.field(st, x, y, edge);
            match path::step(&field, x as i32, y as i32, mobile_axis[mi]) {
                Some((nx, ny)) => {
                    let (nx, ny) = (nx as i8, ny as i8);
                    mobile_axis[mi] = if nx == x { 2 } else { 1 };
                    {
                        let m = &mut st.mobiles[mi];
                        m.x = nx;
                        m.y = ny;
                        m.steps = steps + 1;
                    }
                    ev.mv.push((owner, kind, x, y, nx, ny));
                }
                None if geo::is_on_edge(edge, x as i32, y as i32) => {
                    // Standing on the target edge and attempting to move again = breach
                    // (replay-confirmed: scoring happens on the attempt AFTER arriving, so
                    // the unit spends one frame attacking from the edge tile first, and can
                    // be killed there before it scores).
                    let stats = cfg.stats(kind, false);
                    let enemy = 1 - owner as usize;
                    st.hp[enemy] -= stats.breach_damage;
                    st.sp[owner as usize] += stats.metal_for_breach;
                    summary.breaches[owner as usize] += stats.breach_damage;
                    ev.breach.push((owner, kind, x, y, stats.breach_damage));
                    ev.death.push((owner, kind, x, y, false));
                    st.mobiles[mi].alive = false;
                }
                None => {
                    // nowhere to move -> self-destruct (health becomes 0; still attacks
                    // this frame; removed at step 4)
                    let stats = cfg.stats(kind, false);
                    let mut affected: Vec<(i8, i8)> = Vec::new();
                    if steps >= stats.sd_steps {
                        let thr = cfg.sd_d2[kind as usize][0];
                        // enemy mobiles
                        for oi in 0..st.mobiles.len() {
                            if oi == mi {
                                continue;
                            }
                            let o = &st.mobiles[oi];
                            // ≤0-health units still take SD damage (they are only
                            // removed at step 4) — ladder-replay-confirmed
                            if !o.alive || o.owner == owner {
                                continue;
                            }
                            if geo::dist2(x as i32, y as i32, o.x as i32, o.y as i32) <= thr {
                                let (ox, oy, okind, oowner) = (o.x, o.y, o.kind, o.owner);
                                damage_mobile(&mut st.mobiles[oi], stats.sd_walker);
                                ev.damage.push((oowner, okind, ox, oy, stats.sd_walker));
                                affected.push((ox, oy));
                            }
                        }
                        // enemy structures
                        for dx in -2i32..=2 {
                            for dy in -2i32..=2 {
                                let (tx, ty) = (x as i32 + dx, y as i32 + dy);
                                if dx * dx + dy * dy > thr || !geo::in_bounds(tx, ty) {
                                    continue;
                                }
                                if let Some(sidx) = st.structure_at(tx as i8, ty as i8) {
                                    let s = &st.structures[sidx];
                                    if s.owner != owner && s.alive {
                                        let (sx2, sy2, skind, sowner) = (s.x, s.y, s.kind, s.owner);
                                        st.structures[sidx].health -= stats.sd_tower;
                                        summary.structure_damage_dealt[owner as usize] +=
                                            stats.sd_tower;
                                        ev.damage.push((sowner, skind, sx2, sy2, stats.sd_tower));
                                        affected.push((sx2, sy2));
                                    }
                                }
                            }
                        }
                    }
                    affected.sort();
                    // engine emits a selfDestruct event only when the unit qualified for
                    // SD damage (corpus: trapped 0/1-step spawns die with no event)
                    if steps >= stats.sd_steps {
                        ev.self_destruct.push((owner, kind, x, y, stats.sd_walker, affected));
                    }
                    st.mobiles[mi].health = 0.0;
                    st.mobiles[mi].shield = 0.0;
                }
            }
        }

        // ---- 2. shields (post-movement positions; replay-confirmed ordering)
        for si in 0..st.structures.len() {
            let s = &st.structures[si];
            if !s.alive || s.kind != config::SUPPORT {
                continue;
            }
            let stats = cfg.stats(config::SUPPORT, s.upgraded);
            let thr = cfg.shield_d2[config::SUPPORT as usize][s.upgraded as usize];
            let own_y = if s.owner == 0 { s.y as f64 } else { (27 - s.y) as f64 };
            // engine computes the grant amount in double, then narrows once
            // (platform replay: stacked pools match f64-then-f32, not an f32 chain)
            let amount =
                (stats.shield_per_unit as f64 + stats.shield_bonus_per_y as f64 * own_y) as f32;
            let (sx, sy, sowner) = (s.x, s.y, s.owner);
            for mi in 0..st.mobiles.len() {
                let m = &st.mobiles[mi];
                // health <= 0.0: a unit that self-destructed this frame must not be
                // resurrected by a shield
                if !m.alive || m.owner != sowner || m.health <= 0.0 {
                    continue;
                }
                let d2 = geo::dist2(sx as i32, sy as i32, m.x as i32, m.y as i32);
                if d2 > thr {
                    continue;
                }
                if st.structures[si].granted.contains(&st.mobiles[mi].seq) {
                    continue;
                }
                let seq = st.mobiles[mi].seq;
                st.structures[si].granted.push(seq);
                let m = &mut st.mobiles[mi];
                m.shield += amount;
                ev.shield.push((sowner, config::SUPPORT, sx, sy, m.x, m.y, amount));
            }
        }

        // ---- 3. attacks (creation order, sequential damage; 0-health untargetable;
        //         units reduced to 0 this frame still attack — see MECHANICS.md ⚠)
        for &(_, r) in order.iter() {
            let (owner, kind, upgraded, ax, ay, attacker_alive) = match r {
                Ref::S(i) => {
                    let s = &st.structures[i];
                    (s.owner, s.kind, s.upgraded, s.x, s.y, s.alive)
                }
                Ref::M(i) => {
                    let m = &st.mobiles[i];
                    (m.owner, m.kind, false, m.x, m.y, m.alive)
                }
            };
            if !attacker_alive {
                continue; // breached (gone) or structure destroyed in an earlier turn
            }
            let stats = cfg.stats(kind, upgraded);
            if stats.dmg_walker <= 0.0 && stats.dmg_tower <= 0.0 {
                continue;
            }
            if let Some(t) = pick_target(st, &cfg, owner, kind, upgraded, ax, ay) {
                match t {
                    Target::Mobile(ti, d) => {
                        let (tx, ty, tkind, towner) = {
                            let m = &st.mobiles[ti];
                            (m.x, m.y, m.kind, m.owner)
                        };
                        damage_mobile(&mut st.mobiles[ti], d);
                        ev.attack.push((owner, kind, ax, ay, tx, ty, d));
                        ev.damage.push((towner, tkind, tx, ty, d));
                    }
                    Target::Structure(ti, d) => {
                        let (tx, ty, tkind, towner) = {
                            let s = &st.structures[ti];
                            (s.x, s.y, s.kind, s.owner)
                        };
                        st.structures[ti].health -= d;
                        summary.structure_damage_dealt[owner as usize] += d;
                        ev.attack.push((owner, kind, ax, ay, tx, ty, d));
                        ev.damage.push((towner, tkind, tx, ty, d));
                    }
                }
            }
        }

        // ---- 4. deaths
        for i in 0..st.structures.len() {
            let s = &st.structures[i];
            if s.alive && s.health <= 0.0 {
                let (owner, kind, x, y) = (s.owner, s.kind, s.x, s.y);
                ev.death.push((owner, kind, x, y, false));
                st.structures[i].alive = false;
                st.grid[gi(x, y)] = EMPTY;
                st.layout_version += 1;
            }
        }
        for i in 0..st.mobiles.len() {
            let m = &st.mobiles[i];
            if m.alive && m.health <= 0.0 {
                let (owner, kind, x, y) = (m.owner, m.kind, m.x, m.y);
                ev.death.push((owner, kind, x, y, false));
                st.mobiles[i].alive = false;
            }
        }

        frames.push(FrameRec { stats: snapshot_stats(st), units: snapshot_units(st), ev });

        f += 1;
        if !st.mobiles.iter().any(|m| m.alive) {
            break;
        }
        if f > 2000 {
            panic!("action phase runaway (frame > 2000)");
        }
    }
    summary.frames = f;

    // NOTE: player-initiated removals do NOT execute here — marked structures stay on the
    // board (pending) through the whole action phase; execute_removals() runs at the next
    // turn's restore (replay-confirmed), and its death events surface in the next turn's
    // frame 0 via State::pending_removal_deaths.

    st.mobiles.clear();
    for s in st.structures.iter_mut() {
        s.granted.clear();
    }
    summary
}

fn snapshot_stats(st: &State) -> [[f32; 3]; 2] {
    [
        [st.hp[0], st.sp[0], st.mp[0]],
        [st.hp[1], st.sp[1], st.mp[1]],
    ]
}

fn snapshot_units(st: &State) -> Vec<UnitSnap> {
    let mut out = Vec::with_capacity(st.structures.len() + st.mobiles.len());
    for s in st.structures.iter().filter(|s| s.alive) {
        out.push(UnitSnap {
            owner: s.owner,
            kind: s.kind,
            x: s.x,
            y: s.y,
            health: s.health,
            upgraded: s.upgraded,
            pending: s.pending_removal,
            remove_countdown: s.remove_countdown,
        });
    }
    for m in st.mobiles.iter().filter(|m| m.alive) {
        out.push(UnitSnap {
            owner: m.owner,
            kind: m.kind,
            x: m.x,
            y: m.y,
            health: m.health + m.shield,
            upgraded: false,
            pending: false,
            remove_countdown: 0,
        });
    }
    out
}

/// Damage drains the shield pool before base health (engine float path: with pool p and
/// damage d > p, base -= d - p).
fn damage_mobile(m: &mut crate::state::Mobile, d: f32) {
    if m.shield >= d {
        m.shield -= d;
    } else {
        m.health -= d - m.shield;
        m.shield = 0.0;
    }
}

enum Target {
    Mobile(usize, f32),
    Structure(usize, f32),
}

/// Targeting per rules: mobile > structure; nearest; lowest health; furthest toward the
/// attacker's side; closest to an edge; residual tie -> most recently created.
fn pick_target(
    st: &State,
    cfg: &Config,
    owner: u8,
    kind: u8,
    upgraded: bool,
    ax: i8,
    ay: i8,
) -> Option<Target> {
    let stats = cfg.stats(kind, upgraded);
    let thr_row = &cfg.attack_d2[kind as usize][upgraded as usize];
    // key: (is_structure, d2, health, y_directional, -edge_dist, -seq)
    let mut best_key: Option<(u8, i32, f32, i32, i32, i64)> = None;
    let mut best: Option<Target> = None;

    if stats.dmg_walker > 0.0 {
        for (i, m) in st.mobiles.iter().enumerate() {
            if !m.alive || m.owner == owner || m.health <= 0.0 {
                continue;
            }
            let d2 = geo::dist2(ax as i32, ay as i32, m.x as i32, m.y as i32);
            if d2 > thr_row[m.kind as usize] {
                continue;
            }
            let ydir = if owner == 0 { m.y as i32 } else { -(m.y as i32) };
            let edge = -((2 * m.x as i32 - 27).abs());
            let key = (0u8, d2, m.health + m.shield, ydir, edge, -(m.seq as i64));
            if best_key.map_or(true, |bk| key_lt(&key, &bk)) {
                best_key = Some(key);
                best = Some(Target::Mobile(i, stats.dmg_walker));
            }
        }
    }
    if stats.dmg_tower > 0.0 {
        let reach = (stats.attack_range + 1.0) as i32; // window certainly covers thr
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                let (tx, ty) = (ax as i32 + dx, ay as i32 + dy);
                if !geo::in_bounds(tx, ty) {
                    continue;
                }
                let sidx = match st.structure_at(tx as i8, ty as i8) {
                    Some(i) => i,
                    None => continue,
                };
                let s = &st.structures[sidx];
                if !s.alive || s.owner == owner || s.health <= 0.0 {
                    continue;
                }
                let d2 = geo::dist2(ax as i32, ay as i32, tx, ty);
                if d2 > thr_row[s.kind as usize] {
                    continue;
                }
                let ydir = if owner == 0 { s.y as i32 } else { -(s.y as i32) };
                let edge = -((2 * s.x as i32 - 27).abs());
                let key = (1u8, d2, s.health, ydir, edge, -(s.seq as i64));
                if best_key.map_or(true, |bk| key_lt(&key, &bk)) {
                    best_key = Some(key);
                    best = Some(Target::Structure(sidx, stats.dmg_tower));
                }
            }
        }
    }
    best
}

#[inline]
fn key_lt(a: &(u8, i32, f32, i32, i32, i64), b: &(u8, i32, f32, i32, i32, i64)) -> bool {
    if a.0 != b.0 {
        return a.0 < b.0;
    }
    if a.1 != b.1 {
        return a.1 < b.1;
    }
    if a.2 != b.2 {
        return a.2 < b.2;
    }
    if a.3 != b.3 {
        return a.3 < b.3;
    }
    if a.4 != b.4 {
        return a.4 < b.4;
    }
    a.5 < b.5
}
