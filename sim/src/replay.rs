//! Replay parsing + frame-exact diff harness vs engine.jar output.
//!
//! Per-turn resync: for every turn we reconstruct the state from the replay's turn frame,
//! re-apply the exact deploy commands (from action-frame-0 spawn events, with engine ids),
//! re-simulate the action phase, and diff every frame. The restore step (removals, refunds,
//! decay, income) is validated separately against the NEXT turn frame. Resync isolates
//! divergences per turn instead of compounding them.

use crate::config::{self, Config};
use crate::engine::{self, Events, FrameRec, TurnCommands, UnitSnap};
use crate::state::{gi, Cmd, State};
use serde_json::Value;
use std::sync::Arc;

pub struct Frame {
    pub phase: i64,
    pub turn: i64,
    pub action_frame: i64,
    /// [player][hp, sp, mp]
    pub stats: [[f64; 3]; 2],
    pub units: Vec<UnitSnap>,
    pub ev: Events,
}

pub struct Replay {
    pub config: Value,
    pub frames: Vec<Frame>,
}

fn fx(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn ix(v: &Value) -> i8 {
    v.as_f64().unwrap() as i8
}

fn loc(v: &Value) -> (i8, i8) {
    (ix(&v[0]), ix(&v[1]))
}

fn owner_of(v: &Value) -> u8 {
    (v.as_i64().unwrap() - 1) as u8
}

pub fn parse(path: &str) -> Replay {
    let text = std::fs::read_to_string(path).expect("replay file");
    let mut config = None;
    let mut frames = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).expect("replay line json");
        if v.get("unitInformation").is_some() {
            config = Some(v);
            continue;
        }
        if v.get("turnInfo").is_none() {
            continue;
        }
        frames.push(parse_frame(&v, config.as_ref().expect("config before frames")));
    }
    Replay { config: config.expect("replay config line"), frames }
}

fn parse_frame(v: &Value, cfg_json: &Value) -> Frame {
    let ti = v["turnInfo"].as_array().unwrap();
    let phase = ti[0].as_i64().unwrap();
    let turn = ti[1].as_i64().unwrap();
    let action_frame = ti[2].as_i64().unwrap();

    let mut stats = [[0.0; 3]; 2];
    for (p, key) in ["p1Stats", "p2Stats"].iter().enumerate() {
        let s = v[*key].as_array().unwrap();
        stats[p] = [fx(&s[0]), fx(&s[1]), fx(&s[2])];
    }

    // Units: lists 0..5 are real units; 6 = removal marks, 7 = upgrade marks.
    let mut units: Vec<UnitSnap> = Vec::new();
    let upg_health: Vec<f64> = (0..6)
        .map(|k| {
            cfg_json["unitInformation"][k]
                .get("upgrade")
                .and_then(|u| u.get("startHealth"))
                .and_then(Value::as_f64)
                .unwrap_or(f64::NAN)
        })
        .collect();
    let _ = upg_health;
    for (p, key) in ["p1Units", "p2Units"].iter().enumerate() {
        let lists = v[*key].as_array().unwrap();
        let mut marks_pending: Vec<(i8, i8, u32)> = Vec::new();
        let mut marks_upgraded: Vec<(i8, i8)> = Vec::new();
        if lists.len() > 6 {
            for e in lists[6].as_array().unwrap() {
                let (x, y) = (ix(&e[0]), ix(&e[1]));
                marks_pending.push((x, y, fx(&e[2]) as u32));
            }
        }
        if lists.len() > 7 {
            for e in lists[7].as_array().unwrap() {
                marks_upgraded.push((ix(&e[0]), ix(&e[1])));
            }
        }
        for k in 0..6usize {
            for e in lists[k].as_array().unwrap() {
                let (x, y) = (ix(&e[0]), ix(&e[1]));
                let pending = marks_pending.iter().find(|m| m.0 == x && m.1 == y);
                units.push(UnitSnap {
                    owner: p as u8,
                    kind: k as u8,
                    x,
                    y,
                    health: fx(&e[2]) as f32,
                    upgraded: k < 3 && marks_upgraded.iter().any(|m| *m == (x, y)),
                    pending: k < 3 && pending.is_some(),
                    remove_countdown: pending.map(|m| m.2).unwrap_or(0),
                });
            }
        }
    }

    let mut ev = Events::default();
    if let Some(e) = v.get("events") {
        for s in e.get("spawn").and_then(Value::as_array).unwrap_or(&vec![]) {
            ev.spawn.push((owner_of(&s[3]), ix(&s[1]) as u8, loc(&s[0]).0, loc(&s[0]).1));
        }
        for m in e.get("move").and_then(Value::as_array).unwrap_or(&vec![]) {
            let (fxy, txy) = (loc(&m[0]), loc(&m[1]));
            ev.mv.push((owner_of(&m[5]), ix(&m[3]) as u8, fxy.0, fxy.1, txy.0, txy.1));
        }
        for s in e.get("shield").and_then(Value::as_array).unwrap_or(&vec![]) {
            let (sxy, txy) = (loc(&s[0]), loc(&s[1]));
            ev.shield
                .push((owner_of(&s[6]), ix(&s[3]) as u8, sxy.0, sxy.1, txy.0, txy.1, fx(&s[2]) as f32));
        }
        for a in e.get("attack").and_then(Value::as_array).unwrap_or(&vec![]) {
            let (axy, txy) = (loc(&a[0]), loc(&a[1]));
            ev.attack
                .push((owner_of(&a[6]), ix(&a[3]) as u8, axy.0, axy.1, txy.0, txy.1, fx(&a[2]) as f32));
        }
        for d in e.get("damage").and_then(Value::as_array).unwrap_or(&vec![]) {
            let xy = loc(&d[0]);
            ev.damage.push((owner_of(&d[4]), ix(&d[2]) as u8, xy.0, xy.1, fx(&d[1]) as f32));
        }
        for d in e.get("death").and_then(Value::as_array).unwrap_or(&vec![]) {
            let xy = loc(&d[0]);
            ev.death.push((
                owner_of(&d[3]),
                ix(&d[1]) as u8,
                xy.0,
                xy.1,
                d[4].as_bool().unwrap_or(false),
            ));
        }
        for b in e.get("breach").and_then(Value::as_array).unwrap_or(&vec![]) {
            let xy = loc(&b[0]);
            ev.breach.push((owner_of(&b[4]), ix(&b[2]) as u8, xy.0, xy.1, fx(&b[1]) as f32));
        }
        for s in e.get("selfDestruct").and_then(Value::as_array).unwrap_or(&vec![]) {
            let xy = loc(&s[0]);
            let mut targets: Vec<(i8, i8)> = s[1]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| loc(t))
                .collect();
            targets.sort();
            ev.self_destruct
                .push((owner_of(&s[5]), ix(&s[3]) as u8, xy.0, xy.1, fx(&s[2]) as f32, targets));
        }
    }

    Frame { phase, turn, action_frame, stats, units, ev }
}

/// Rebuild a State from a turn frame (phase 0).
pub fn state_from_turn_frame(cfg: Arc<Config>, f: &Frame) -> State {
    assert_eq!(f.phase, 0, "state reconstruction requires a turn frame");
    let mut st = State::new(cfg.clone());
    st.turn = f.turn as u32;
    for p in 0..2 {
        st.hp[p] = f.stats[p][0] as f32;
        st.sp[p] = f.stats[p][1] as f32;
        st.mp[p] = f.stats[p][2] as f32;
    }
    let mut max_id = 0u32;
    for u in f.units.iter() {
        assert!(u.kind < 3, "turn frame should only contain structures");
        let stats = cfg.stats(u.kind, u.upgraded);
        let idx = st.structures.len();
        st.structures.push(crate::state::Structure {
            kind: u.kind,
            owner: u.owner,
            x: u.x,
            y: u.y,
            health: u.health,
            max_health: stats.start_health,
            upgraded: u.upgraded,
            pending_removal: u.pending,
            remove_countdown: u.remove_countdown.max(if u.pending { 1 } else { 0 }),
            id: 0, // engine ids are not recoverable per-structure here; seq set below
            seq: 0,
            alive: true,
            granted: Vec::new(),
        });
        st.grid[gi(u.x, u.y)] = idx as u16;
        st.layout_version += 1;
        max_id += 1;
    }
    let _ = max_id;
    st
}

/// Attach engine ids/seqs to reconstructed structures using the id strings in the raw turn
/// frame (needed for creation-order attack sequencing). Done in a second pass over the raw
/// JSON because UnitSnap intentionally drops ids.
pub fn attach_ids(st: &mut State, raw: &Value) {
    for (p, key) in ["p1Units", "p2Units"].iter().enumerate() {
        let lists = raw[*key].as_array().unwrap();
        for k in 0..3usize {
            for e in lists[k].as_array().unwrap() {
                let (x, y) = (ix(&e[0]), ix(&e[1]));
                let id: u32 = e[3].as_str().unwrap().parse().unwrap_or(0);
                if let Some(i) = st.structure_at(x, y) {
                    if st.structures[i].owner == p as u8 && st.structures[i].kind == k as u8 {
                        st.structures[i].id = id;
                        st.structures[i].seq = id;
                    }
                }
            }
        }
    }
    let max_id = st.structures.iter().map(|s| s.id).max().unwrap_or(0);
    st.next_id = max_id + 1;
}

/// Extract both players' commands (with engine ids) from an action-frame-0 spawn list.
pub fn commands_from_frame0(raw: &Value) -> [TurnCommands; 2] {
    let mut out: [TurnCommands; 2] = [TurnCommands::default(), TurnCommands::default()];
    let spawns = raw["events"]["spawn"].as_array().cloned().unwrap_or_default();
    for s in spawns.iter() {
        let (x, y) = loc(&s[0]);
        let kind = ix(&s[1]) as u8;
        let id: u32 = s[2].as_str().unwrap().parse().unwrap();
        let p = owner_of(&s[3]) as usize;
        let cmd = match kind {
            0..=2 => Cmd::Build { kind, x, y },
            config::REMOVE => Cmd::Remove { x, y },
            config::UPGRADE => Cmd::Upgrade { x, y },
            3..=5 => Cmd::Deploy { kind, x, y },
            _ => panic!("unknown spawn kind {}", kind),
        };
        let is_build = matches!(cmd, Cmd::Build { .. } | Cmd::Remove { .. } | Cmd::Upgrade { .. });
        if is_build {
            out[p].build.push(cmd);
            out[p].build_ids.get_or_insert_with(Vec::new).push(id);
        } else {
            out[p].deploy.push(cmd);
            out[p].deploy_ids.get_or_insert_with(Vec::new).push(id);
        }
    }
    out
}

// -------------------------------------------------------------------------------- diff

fn canon_units(units: &[UnitSnap]) -> Vec<(u8, u8, i8, i8, u64, bool, bool)> {
    // health goes through display01: replay unit healths are ceil-tenth displays of the
    // engine's raw f64s (corpus: shield sums print 26.6 for raw 26.599999999999998)
    // health quantized to 0.01 — absorbs characterized f32 shield-pool dust (<=1e-3)
    // while preserving every legitimate distinction (real values are exact tenths)
    let mut v: Vec<_> = units
        .iter()
        .map(|u| {
            let hq = ((u.health as f64) * 100.0).round() as i64;
            (u.owner, u.kind, u.x, u.y, hq as u64, u.upgraded, u.pending)
        })
        .collect();
    v.sort();
    v
}

pub struct DiffStats {
    pub turns: u32,
    pub turns_ok: u32,
    pub frames: u32,
    pub frames_ok: u32,
    pub restore_ok: u32,
    pub restore_checked: u32,
    pub first_bad: Option<(i64, i64)>,
}

pub fn diff_replay(path: &str, verbose: usize) -> DiffStats {
    let text = std::fs::read_to_string(path).expect("replay file");
    let raw_lines: Vec<Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("json"))
        .collect();
    let rep = parse(path);
    let cfg = Arc::new(Config::from_json(&rep.config));

    // Group frames by turn: (turn_frame_idx, action_frame_indices)
    let mut turns: Vec<(usize, Vec<usize>)> = Vec::new();
    for (i, f) in rep.frames.iter().enumerate() {
        match f.phase {
            0 => turns.push((i, Vec::new())),
            1 => {
                if let Some(t) = turns.last_mut() {
                    t.1.push(i);
                }
            }
            _ => {}
        }
    }

    let mut ds = DiffStats {
        turns: 0,
        turns_ok: 0,
        frames: 0,
        frames_ok: 0,
        restore_ok: 0,
        restore_checked: 0,
        first_bad: None,
    };

    // raw JSON lines: index 0 is config; frame i corresponds to raw_lines[i+1]
    let raw_of = |frame_idx: usize| -> &Value { &raw_lines[frame_idx + 1] };

    // Raw resource carry-over: the engine keeps unrounded f64 resources internally while
    // frames serialize display01 (ceil-tenth) values. Rebuilding stats from a frame would
    // inject up to +0.1 phantom MP, so we thread our own raw values across turns whenever
    // they display-match the frame.
    let mut carry: Option<[[f32; 3]; 2]> = None;

    for (t_i, (turn_idx, action_idxs)) in turns.iter().enumerate() {
        if action_idxs.is_empty() {
            continue; // final phase-0 frame before endStats, if any
        }
        ds.turns += 1;
        let tf = &rep.frames[*turn_idx];
        let mut st = state_from_turn_frame(cfg.clone(), tf);
        attach_ids(&mut st, raw_of(*turn_idx));
        if let Some(c) = carry {
            for p in 0..2 {
                for si in 0..3 {
                    let d = (config::display01(c[p][si]) - tf.stats[p][si]).abs();
                    if d <= 0.10001 {
                        match si {
                            0 => st.hp[p] = c[p][0],
                            1 => st.sp[p] = c[p][1],
                            _ => st.mp[p] = c[p][2],
                        }
                    } else {
                        println!(
                            "turn {}: carry {} p{} display {} != frame {} (raw {}); resync to frame",
                            tf.turn,
                            ["hp", "sp", "mp"][si],
                            p + 1,
                            config::display01(c[p][si]),
                            tf.stats[p][si],
                            c[p][si]
                        );
                    }
                }
            }
        }
        let cmds = commands_from_frame0(raw_of(action_idxs[0]));
        let mut sim_frames: Vec<FrameRec> = Vec::new();
        engine::play_turn(&mut st, cmds, true, &mut sim_frames);

        let mut turn_ok = true;
        if sim_frames.len() != action_idxs.len() {
            println!(
                "turn {}: FRAME COUNT sim {} vs engine {}",
                tf.turn,
                sim_frames.len(),
                action_idxs.len()
            );
            turn_ok = false;
        }
        for (fi, (sf, &ei)) in sim_frames.iter().zip(action_idxs.iter()).enumerate() {
            ds.frames += 1;
            let ef = &rep.frames[ei];
            let mut msgs: Vec<String> = Vec::new();

            for p in 0..2 {
                for (si, name) in ["hp", "sp", "mp"].iter().enumerate() {
                    let d = (config::display01(sf.stats[p][si]) - ef.stats[p][si]).abs();
                    if d > 0.10001 {
                        msgs.push(format!(
                            "p{} {}: sim raw {} (display {}) vs engine {}",
                            p + 1,
                            name,
                            sf.stats[p][si],
                            config::display01(sf.stats[p][si]),
                            ef.stats[p][si]
                        ));
                    }
                }
            }
            let (su, eu) = (canon_units(&sf.units), canon_units(&ef.units));
            if su != eu {
                for u in su.iter().filter(|u| !eu.contains(u)).take(6) {
                    msgs.push(format!(
                        "unit only in sim: p{} k{} ({},{}) hp {} upg {} pend {}",
                        u.0 + 1, u.1, u.2, u.3, (u.4 as f64) / 100.0, u.5, u.6
                    ));
                }
                for u in eu.iter().filter(|u| !su.contains(u)).take(6) {
                    msgs.push(format!(
                        "unit only in engine: p{} k{} ({},{}) hp {} upg {} pend {}",
                        u.0 + 1, u.1, u.2, u.3, (u.4 as f64) / 100.0, u.5, u.6
                    ));
                }
            }
            for ((name, a), (_, b)) in sf.ev.canon().iter().zip(ef.ev.canon().iter()) {
                if a != b {
                    msgs.push(format!(
                        "{} events differ: sim {} vs engine {}",
                        name,
                        a.len(),
                        b.len()
                    ));
                    for x in a.iter().filter(|x| !b.contains(x)).take(4) {
                        msgs.push(format!("  sim-only {}: {}", name, x));
                    }
                    for x in b.iter().filter(|x| !a.contains(x)).take(4) {
                        msgs.push(format!("  engine-only {}: {}", name, x));
                    }
                }
            }

            if msgs.is_empty() {
                ds.frames_ok += 1;
            } else {
                if turn_ok {
                    println!("turn {} frame {}: {} mismatches", tf.turn, fi, msgs.len());
                }
                turn_ok = false;
                if ds.first_bad.is_none() {
                    ds.first_bad = Some((tf.turn, fi as i64));
                }
                for m in msgs.iter().take(verbose.max(1)) {
                    println!("    {}", m);
                }
            }
        }

        // restore check vs next turn frame (removals execute here, then decay + income)
        if t_i + 1 < turns.len() {
            ds.restore_checked += 1;
            let nf = &rep.frames[turns[t_i + 1].0];
            let removed = st.execute_removals();
            st.pending_removal_deaths.clear();
            st.turn += 1;
            st.restore();
            // removal death events appear in the NEXT TURN FRAME (phase 0) of the replay
            {
                let mut a: Vec<(u8, u8, i8, i8)> =
                    removed.iter().map(|&(o, k, x, y, _)| (o, k, x, y)).collect();
                a.sort();
                let mut b: Vec<(u8, u8, i8, i8)> = nf
                    .ev
                    .death
                    .iter()
                    .filter(|d| d.4)
                    .map(|d| (d.0, d.1, d.2, d.3))
                    .collect();
                b.sort();
                if a != b {
                    println!(
                        "turn {} restore: removal deaths sim {:?} vs engine {:?}",
                        tf.turn, a, b
                    );
                }
            }
            let mut ok = true;
            for p in 0..2 {
                let sim = [st.hp[p], st.sp[p], st.mp[p]];
                for si in 0..3 {
                    if (config::display01(sim[si]) - nf.stats[p][si]).abs() > 0.10001 {
                        println!(
                            "turn {} restore: p{} {} sim raw {} (display {}) vs engine {}",
                            tf.turn,
                            p + 1,
                            ["hp", "sp", "mp"][si],
                            sim[si],
                            config::display01(sim[si]),
                            nf.stats[p][si]
                        );
                        ok = false;
                    }
                }
            }
            let snap: Vec<UnitSnap> = st
                .structures
                .iter()
                .filter(|s| s.alive)
                .map(|s| UnitSnap {
                    owner: s.owner,
                    kind: s.kind,
                    x: s.x,
                    y: s.y,
                    health: s.health,
                    upgraded: s.upgraded,
                    pending: s.pending_removal,
                    remove_countdown: s.remove_countdown,
                })
                .collect();
            let (a, b) = (canon_units(&snap), canon_units(&nf.units));
            if a != b {
                ok = false;
                println!("turn {} restore: structure sets differ", tf.turn);
                for u in a.iter().filter(|u| !b.contains(u)).take(4) {
                    println!(
                        "    sim-only: p{} k{} ({},{}) hp {}",
                        u.0 + 1, u.1, u.2, u.3, (u.4 as f64) / 100.0
                    );
                }
                for u in b.iter().filter(|u| !a.contains(u)).take(4) {
                    println!(
                        "    engine-only: p{} k{} ({},{}) hp {}",
                        u.0 + 1, u.1, u.2, u.3, (u.4 as f64) / 100.0
                    );
                }
            }
            if ok {
                ds.restore_ok += 1;
            }
            carry = Some([
                [st.hp[0], st.sp[0], st.mp[0]],
                [st.hp[1], st.sp[1], st.mp[1]],
            ]);
        }

        if turn_ok {
            ds.turns_ok += 1;
        }
    }
    ds
}
