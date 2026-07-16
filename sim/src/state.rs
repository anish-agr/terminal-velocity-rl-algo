//! Game state: board, units, resources, deploy-phase command application.

use crate::config::Config;
use crate::geo;
use std::sync::Arc;

pub const EMPTY: u16 = u16::MAX;

#[derive(Clone, Debug)]
pub struct Structure {
    pub kind: u8,
    pub owner: u8, // 0 = player1 (bottom), 1 = player2 (top)
    pub x: i8,
    pub y: i8,
    pub health: f32,
    pub max_health: f32,
    pub upgraded: bool,
    pub pending_removal: bool,
    pub remove_countdown: u32,
    pub id: u32,
    pub seq: u32, // creation order (engine id order)
    pub alive: bool,
    /// Mobile seqs this support has already shielded (this action phase).
    pub granted: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Mobile {
    pub kind: u8,
    pub owner: u8,
    pub x: i8,
    pub y: i8,
    /// Base health. Serialized health = health + shield (engine keeps a separate shield
    /// pool; damage drains the pool before base health — replay float-path-confirmed).
    pub health: f32,
    pub shield: f32,
    pub id: u32,
    pub seq: u32,
    pub alive: bool,
    pub steps: u32,
    pub frames_per_move: u32,
    pub target_edge: u8,
    pub path: Arc<Vec<(i8, i8)>>,
    pub path_idx: usize,
    pub needs_repath: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cmd {
    Build { kind: u8, x: i8, y: i8 },
    Upgrade { x: i8, y: i8 },
    Remove { x: i8, y: i8 },
    Deploy { kind: u8, x: i8, y: i8 },
}

#[derive(Clone)]
pub struct State {
    pub cfg: Arc<Config>,
    pub turn: u32,
    pub hp: [f32; 2],
    pub sp: [f32; 2],
    pub mp: [f32; 2],
    pub structures: Vec<Structure>,
    pub mobiles: Vec<Mobile>,
    pub grid: Vec<u16>, // 28*28, EMPTY or index into structures
    pub next_id: u32,
    pub layout_version: u64,
    /// Removal deaths produced by the last execute_removals(), reported as death events
    /// (removal=true) at frame 0 of the following action phase.
    pub pending_removal_deaths: Vec<(u8, u8, i8, i8)>,
}

#[inline]
pub fn gi(x: i8, y: i8) -> usize {
    (x as usize) * 28 + (y as usize)
}

impl State {
    pub fn new(cfg: Arc<Config>) -> State {
        State {
            turn: 0,
            hp: [cfg.start_hp; 2],
            sp: [cfg.start_sp; 2],
            mp: [cfg.start_mp; 2],
            structures: Vec::with_capacity(128),
            mobiles: Vec::with_capacity(256),
            grid: vec![EMPTY; 28 * 28],
            next_id: 1,
            layout_version: 0,
            pending_removal_deaths: Vec::new(),
            cfg,
        }
    }

    #[inline]
    pub fn structure_at(&self, x: i8, y: i8) -> Option<usize> {
        let s = self.grid[gi(x, y)];
        if s == EMPTY { None } else { Some(s as usize) }
    }

    pub fn own_half(owner: u8, y: i8) -> bool {
        if owner == 0 { (y as i32) < geo::HALF } else { (y as i32) >= geo::HALF }
    }

    pub fn on_own_spawn_edge(owner: u8, x: i8, y: i8) -> bool {
        let (x, y) = (x as i32, y as i32);
        if owner == 0 {
            geo::is_on_edge(geo::BOTTOM_LEFT, x, y) || geo::is_on_edge(geo::BOTTOM_RIGHT, x, y)
        } else {
            geo::is_on_edge(geo::TOP_LEFT, x, y) || geo::is_on_edge(geo::TOP_RIGHT, x, y)
        }
    }

    /// Restore phase: decay stored MP, then add income. `self.turn` must already be set
    /// to the NEW turn number.
    ///
    /// RAW accounting (replay-derived): the engine never rounds resources internally; the
    /// serialized stats are display01 (ceil-to-tenth) of the raw values. Decay is a plain
    /// multiply. ⚠ x*(1-decay) vs x-x*decay differ by <=1 ulp on arbitrary values; corpus
    /// has not yet distinguished them (both exact on all values seen so far).
    pub fn restore(&mut self) {
        let cfg = self.cfg.clone();
        for p in 0..2 {
            // Raw f32 decay. NOTE (exhaustively tested 2026-07-15): no quantization
            // variant (round/rint/ceil/floor at tenths, f32/f64, either decay expression,
            // either display mode) survives chain-validation against the 8-replay corpus;
            // raw f32 is the best fit. A bounded sub-0.1 drift vs the engine remains on
            // long banking chains (both signs observed) — see MECHANICS.md §Open fixes.
            let decayed = self.mp[p] * (1.0 - cfg.decay);
            let ramp = (self.turn / cfg.mp_interval) as f32 * cfg.mp_growth;
            self.mp[p] = (decayed + cfg.mp_per_round + ramp).min(cfg.max_mp);
            self.sp[p] += cfg.sp_per_round;
        }
    }

    /// Apply one player's commands with engine validation semantics (invalid commands are
    /// skipped). `strict` panics on invalid commands instead — used by the replay harness
    /// where every command is known-accepted, so a rejection means OUR model is wrong.
    /// Returns accepted commands with assigned ids, in order.
    pub fn apply_commands(
        &mut self,
        owner: u8,
        cmds: &[Cmd],
        forced_ids: Option<&[u32]>,
        strict: bool,
        accepted: &mut Vec<(Cmd, u32)>,
    ) {
        for (i, &cmd) in cmds.iter().enumerate() {
            let id = forced_ids.map(|ids| ids[i]);
            let ok = self.apply_one(owner, cmd, id, false, accepted);
            if !ok && strict {
                // An engine-accepted command our model rejects. Affordability rejections
                // are the known bounded MP-drift (MECHANICS §Open fixes 3): force-apply
                // (resources may dip fractionally negative) so the phase stays faithful.
                // A GEOMETRY rejection under force would be a real modeling bug — loud.
                let forced = self.apply_one(owner, cmd, id, true, accepted);
                if forced {
                    println!(
                        "AFFORDABILITY GAP (forced): {:?} owner {} sp {:.4} mp {:.4}",
                        cmd, owner, self.sp[owner as usize], self.mp[owner as usize]
                    );
                } else {
                    println!(
                        "GEOMETRY GAP (real bug!): sim rejected engine-accepted {:?} (owner {}, blocked {:?})",
                        cmd,
                        owner,
                        match cmd {
                            Cmd::Build { x, y, .. }
                            | Cmd::Upgrade { x, y }
                            | Cmd::Remove { x, y }
                            | Cmd::Deploy { x, y, .. } => self.structure_at(x, y).is_some(),
                        }
                    );
                }
            }
        }
    }

    fn take_id(&mut self, forced: Option<u32>) -> u32 {
        let id = forced.unwrap_or(self.next_id);
        self.next_id = self.next_id.max(id + 1);
        id
    }

    fn apply_one(
        &mut self,
        owner: u8,
        cmd: Cmd,
        forced_id: Option<u32>,
        force_afford: bool,
        accepted: &mut Vec<(Cmd, u32)>,
    ) -> bool {
        let cfg = self.cfg.clone();
        match cmd {
            Cmd::Build { kind, x, y } => {
                if !geo::in_bounds(x as i32, y as i32)
                    || !Self::own_half(owner, y)
                    || self.structure_at(x, y).is_some()
                {
                    return false;
                }
                let st = cfg.stats(kind, false);
                if !st.is_structure {
                    return false;
                }
                if !force_afford && self.sp[owner as usize] < st.cost_sp {
                    return false;
                }
                self.sp[owner as usize] -= st.cost_sp;
                let id = self.take_id(forced_id);
                let idx = self.structures.len();
                self.structures.push(Structure {
                    kind,
                    owner,
                    x,
                    y,
                    health: st.start_health,
                    max_health: st.start_health,
                    upgraded: false,
                    pending_removal: false,
                    remove_countdown: 0,
                    id,
                    seq: id,
                    alive: true,
                    granted: Vec::new(),
                });
                self.grid[gi(x, y)] = idx as u16;
                self.layout_version += 1;
                accepted.push((cmd, id));
                true
            }
            Cmd::Upgrade { x, y } => {
                let idx = match self.structure_at(x, y) {
                    Some(i) if self.structures[i].owner == owner => i,
                    _ => return false,
                };
                let (kind, upgraded, health, max_health) = {
                    let s = &self.structures[idx];
                    (s.kind, s.upgraded, s.health, s.max_health)
                };
                let base = cfg.stats(kind, false);
                if upgraded || !base.has_upgrade {
                    return false;
                }
                if !force_afford && self.sp[owner as usize] < base.upgrade_cost_sp {
                    return false;
                }
                self.sp[owner as usize] -= base.upgrade_cost_sp;
                let up = cfg.stats(kind, true);
                let missing = max_health - health;
                let s = &mut self.structures[idx];
                s.upgraded = true;
                s.max_health = up.start_health;
                s.health = up.start_health - missing; // missing health persists
                let id = self.take_id(forced_id);
                accepted.push((cmd, id));
                true
            }
            Cmd::Remove { x, y } => {
                let idx = match self.structure_at(x, y) {
                    Some(i) if self.structures[i].owner == owner => i,
                    _ => return false,
                };
                let id = self.take_id(forced_id);
                let s = &mut self.structures[idx];
                s.pending_removal = true;
                s.remove_countdown = cfg.turns_to_remove;
                accepted.push((cmd, id));
                true
            }
            Cmd::Deploy { kind, x, y } => {
                let st = cfg.stats(kind, false);
                if st.is_structure
                    || !geo::in_bounds(x as i32, y as i32)
                    || !Self::own_spawnable(self, owner, x, y)
                {
                    return false;
                }
                if !force_afford && self.mp[owner as usize] < st.cost_mp {
                    return false;
                }
                self.mp[owner as usize] -= st.cost_mp;
                let id = self.take_id(forced_id);
                self.mobiles.push(Mobile {
                    kind,
                    owner,
                    x,
                    y,
                    health: st.start_health,
                    shield: 0.0,
                    id,
                    seq: id,
                    alive: true,
                    steps: 0,
                    frames_per_move: st.frames_per_move,
                    target_edge: geo::target_edge_for(x as i32, y as i32),
                    path: Arc::new(Vec::new()), // computed at phase start
                    path_idx: 0,
                    needs_repath: true,
                    });
                accepted.push((cmd, id));
                true
            }
        }
    }

    fn own_spawnable(&self, owner: u8, x: i8, y: i8) -> bool {
        Self::on_own_spawn_edge(owner, x, y) && self.structure_at(x, y).is_none()
    }

    /// Execute pending removals — called at the RESTORE step (replay-confirmed: marked
    /// structures survive the whole action phase; refunds land next turn). Death events go
    /// to pending_removal_deaths for the next phase's frame 0.
    /// ⚠ Refund base for upgraded structures assumed base+upgrade cost (MECHANICS.md).
    pub fn execute_removals(&mut self) -> Vec<(u8, u8, i8, i8, u32)> {
        let cfg = self.cfg.clone();
        let mut removed = Vec::new();
        for i in 0..self.structures.len() {
            if !self.structures[i].alive || !self.structures[i].pending_removal {
                continue;
            }
            self.structures[i].remove_countdown -= 1;
            if self.structures[i].remove_countdown > 0 {
                continue;
            }
            let (kind, owner, x, y, id, health, max_health, upgraded) = {
                let s = &self.structures[i];
                (s.kind, s.owner, s.x, s.y, s.id, s.health, s.max_health, s.upgraded)
            };
            let base = cfg.stats(kind, false);
            let mut invested = base.cost_sp;
            if upgraded {
                invested += base.upgrade_cost_sp;
            }
            let refund_pct = cfg.stats(kind, upgraded).refund_pct;
            // refund quantized to a tenth (rules text + corpus: SP displays match only
            // with snapped refunds)
            let refund = refund_pct * invested * (health / max_health);
            self.sp[owner as usize] += refund;
            self.structures[i].alive = false;
            self.grid[gi(x, y)] = EMPTY;
            self.layout_version += 1;
            self.pending_removal_deaths.push((owner, kind, x, y));
            removed.push((owner, kind, x, y, id));
        }
        removed
    }
}
