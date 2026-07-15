//! Python bindings (abi3, py3.8+). One class: `Game` — used by self-play actors, the
//! training feature pipeline, and the in-match deployment search (via `fork`).
//!
//! Command encoding: (kind, x, y) tuples. kind 0..2 build, 3..5 mobile deploy,
//! 6 remove-mark, 7 upgrade. Invalid commands are silently skipped (engine semantics).
//! `play_turn` runs deploy + full action phase + removals + restore, leaving the state
//! ready for the next turn's decisions.

use crate::config::{display01, Config};
use crate::engine::{self, TurnCommands};
use crate::geo;
use crate::state::{Cmd, State};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;

fn decode(cmds: &[(u8, i8, i8)]) -> TurnCommands {
    let mut tc = TurnCommands::default();
    for &(kind, x, y) in cmds {
        match kind {
            0..=2 => tc.build.push(Cmd::Build { kind, x, y }),
            6 => tc.build.push(Cmd::Remove { x, y }),
            7 => tc.build.push(Cmd::Upgrade { x, y }),
            3..=5 => tc.deploy.push(Cmd::Deploy { kind, x, y }),
            _ => {} // unknown kind: skip
        }
    }
    tc
}

/// Number of board feature planes exported by `board_planes`.
pub const PLANES: usize = 12;

#[pyclass]
pub struct Game {
    cfg: Arc<Config>,
    st: State,
    /// (frames, p1_breach_dealt, p2_breach_dealt, p1_struct_dmg, p2_struct_dmg) of last turn
    last: (u32, f32, f32, f32, f32),
}

#[pymethods]
impl Game {
    #[new]
    fn new(config_json: &str) -> PyResult<Self> {
        let v: serde_json::Value = serde_json::from_str(config_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("config: {e}")))?;
        let cfg = Arc::new(Config::from_json(&v));
        let st = State::new(cfg.clone());
        Ok(Game { cfg, st, last: (0, 0.0, 0.0, 0.0, 0.0) })
    }

    fn reset(&mut self) {
        self.st = State::new(self.cfg.clone());
        self.last = (0, 0.0, 0.0, 0.0, 0.0);
    }

    /// Deep copy for search branching.
    fn fork(&self) -> Game {
        Game { cfg: self.cfg.clone(), st: self.st.clone(), last: self.last }
    }

    #[getter]
    fn turn(&self) -> u32 {
        self.st.turn
    }

    /// Raw f32 resources [hp, sp, mp] for a player (affordability uses RAW, not display).
    fn stats(&self, player: usize) -> (f32, f32, f32) {
        (self.st.hp[player], self.st.sp[player], self.st.mp[player])
    }

    /// Display (ceil-tenth) stats as shown by the engine/UI.
    fn stats_display(&self, player: usize) -> (f64, f64, f64) {
        (
            display01(self.st.hp[player]),
            display01(self.st.sp[player]),
            display01(self.st.mp[player]),
        )
    }

    /// Play a full turn. Returns (frames, p1_breach, p2_breach, p1_struct_dmg, p2_struct_dmg).
    fn play_turn(
        &mut self,
        p1: Vec<(u8, i8, i8)>,
        p2: Vec<(u8, i8, i8)>,
    ) -> (u32, f32, f32, f32, f32) {
        let cmds = [decode(&p1), decode(&p2)];
        let mut frames = Vec::new();
        let summary = engine::play_turn(&mut self.st, cmds, false, &mut frames);
        self.st.execute_removals();
        self.st.pending_removal_deaths.clear();
        self.st.turn += 1;
        self.st.restore();
        self.last = (
            summary.frames,
            summary.breaches[0],
            summary.breaches[1],
            summary.structure_damage_dealt[0],
            summary.structure_damage_dealt[1],
        );
        self.last
    }

    fn game_over(&self) -> bool {
        self.st.hp[0] <= 0.0 || self.st.hp[1] <= 0.0 || self.st.turn >= 100
    }

    /// -1 ongoing, 0/1 winner, 2 tie (engine tiebreak = lower compute time, not modeled).
    fn winner(&self) -> i32 {
        let (a, b) = (self.st.hp[0], self.st.hp[1]);
        if a <= 0.0 && b <= 0.0 {
            return 2;
        }
        if a <= 0.0 {
            return 1;
        }
        if b <= 0.0 {
            return 0;
        }
        if self.st.turn >= 100 {
            if a > b {
                return 0;
            }
            if b > a {
                return 1;
            }
            return 2;
        }
        -1
    }

    /// Alive structures: (kind, owner, x, y, health, upgraded, pending_removal).
    fn structures(&self) -> Vec<(u8, u8, i8, i8, f32, bool, bool)> {
        self.st
            .structures
            .iter()
            .filter(|s| s.alive)
            .map(|s| (s.kind, s.owner, s.x, s.y, s.health, s.upgraded, s.pending_removal))
            .collect()
    }

    /// Board feature planes from `player`'s perspective (board flipped for player 1 so the
    /// own side is always y<14). PLANES x 28 x 28 f32, little-endian bytes.
    /// Planes: 0..3 own wall/support/turret hp (normalized by upgraded max) + own upgraded
    /// mask; 4 own pending-removal; 5..9 same for enemy; 10 in-arena mask; 11 own-half mask.
    fn board_planes<'py>(&self, py: Python<'py>, player: u8) -> Bound<'py, PyBytes> {
        let mut planes = vec![0f32; PLANES * 28 * 28];
        let norm = [120.0f32, 30.0, 75.0];
        let flip = player == 1;
        let mut set = |p: usize, x: i8, y: i8, v: f32| {
            let (fx, fy) = if flip { (27 - x as usize, 27 - y as usize) } else { (x as usize, y as usize) };
            planes[p * 784 + fx * 28 + fy] = v;
        };
        for s in self.st.structures.iter().filter(|s| s.alive) {
            let own = (s.owner == player) as usize;
            let base = if own == 1 { 0 } else { 5 };
            let k = s.kind as usize;
            set(base + k, s.x, s.y, s.health / norm[k]);
            if s.upgraded {
                set(base + 3, s.x, s.y, 1.0);
            }
            if s.pending_removal {
                set(base + 4, s.x, s.y, 1.0);
            }
        }
        for x in 0..28i32 {
            for y in 0..28i32 {
                if geo::in_bounds(x, y) {
                    planes[10 * 784 + (x as usize) * 28 + y as usize] = 1.0;
                    if y < 14 {
                        planes[11 * 784 + (x as usize) * 28 + y as usize] = 1.0;
                    }
                }
            }
        }
        let mut bytes = Vec::with_capacity(planes.len() * 4);
        for v in planes {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        PyBytes::new(py, &bytes)
    }

    /// Scalar features from `player`'s perspective:
    /// [own hp, own sp, own mp, enemy hp, enemy sp, enemy mp, turn, mp_income_now]
    fn scalar_features(&self, player: usize) -> Vec<f32> {
        let o = player;
        let e = 1 - player;
        let income =
            self.cfg.mp_per_round + (self.st.turn / self.cfg.mp_interval) as f32 * self.cfg.mp_growth;
        vec![
            self.st.hp[o],
            self.st.sp[o],
            self.st.mp[o],
            self.st.hp[e],
            self.st.sp[e],
            self.st.mp[e],
            self.st.turn as f32,
            income,
        ]
    }

    /// Path a mobile unit would take if spawned at (x, y) right now (empty if blocked).
    fn pathfind(&self, x: i32, y: i32) -> Vec<(i8, i8)> {
        if !geo::in_bounds(x, y) || self.st.structure_at(x as i8, y as i8).is_some() {
            return Vec::new();
        }
        let blocked =
            |bx: i32, by: i32| self.st.structure_at(bx as i8, by as i8).is_some();
        crate::path::pathfind(&blocked, x, y, geo::target_edge_for(x, y))
    }

    /// Units of `kind` affordable for `player` right now (raw resources).
    fn affordable(&self, player: usize, kind: u8) -> u32 {
        let stats = self.cfg.stats(kind, false);
        if stats.is_structure {
            if stats.cost_sp <= 0.0 {
                return 0;
            }
            (self.st.sp[player] / stats.cost_sp).floor() as u32
        } else {
            if stats.cost_mp <= 0.0 {
                return 0;
            }
            (self.st.mp[player] / stats.cost_mp).floor() as u32
        }
    }
}

#[pymodule]
fn terminal_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Game>()?;
    m.add("PLANES", PLANES)?;
    Ok(())
}
