//! Config resolution: game-configs.json -> flat per-(kind, upgraded) unit stats.
//! Upgrade semantics (API reference): the `upgrade{}` block overrides listed fields, all
//! others inherit from base; `upgrade.cost1` (when present) is the upgrade price, otherwise
//! the upgrade price equals the base cost (replay-confirmed: wall upgrade cost 1).

use serde_json::Value;

pub const WALL: u8 = 0;
pub const SUPPORT: u8 = 1;
pub const TURRET: u8 = 2;
pub const SCOUT: u8 = 3;
pub const DEMOLISHER: u8 = 4;
pub const INTERCEPTOR: u8 = 5;
pub const REMOVE: u8 = 6;
pub const UPGRADE: u8 = 7;

/// Gameplay scalars are 32-bit floats: the Java engine uses `float` for health, shields,
/// damage and resources (replay-proven: literal "6.6000004" = f32(26.6) - 20).
#[derive(Clone, Debug, Default)]
pub struct UnitStats {
    pub cost_sp: f32,
    pub cost_mp: f32,
    pub start_health: f32,
    pub dmg_walker: f32,
    pub dmg_tower: f32,
    pub attack_range: f64,
    pub hit_radius: f64,
    pub frames_per_move: u32, // 0 for structures
    pub shield_per_unit: f32,
    pub shield_range: f64,
    pub shield_bonus_per_y: f32,
    pub sd_walker: f32,
    pub sd_tower: f32,
    pub sd_range: f64,
    pub sd_steps: u32,
    pub breach_damage: f32,
    pub metal_for_breach: f32,
    pub refund_pct: f32,
    pub is_structure: bool,
    /// SP price of upgrading (meaningful on the base variant).
    pub upgrade_cost_sp: f32,
    pub has_upgrade: bool,
}

#[derive(Clone, Debug)]
pub struct Config {
    /// [kind 0..6][upgraded as usize]
    pub units: [[UnitStats; 2]; 6],
    pub start_hp: f32,
    pub start_sp: f32,
    pub start_mp: f32,
    pub sp_per_round: f32,
    pub mp_per_round: f32,
    pub mp_growth: f32,
    pub mp_interval: u32,
    pub decay: f32,
    pub max_mp: f32,
    pub turns_to_remove: u32,
    /// Attack range check thresholds: dist2 (integer) must be strictly below
    /// (range + target_hit_radius)^2. Precomputed as the max admissible integer d2:
    /// thr[attacker_kind][upgraded][target_kind].
    pub attack_d2: [[[i32; 6]; 2]; 6],
    pub shield_d2: [[i32; 2]; 6],
    pub sd_d2: [[i32; 2]; 6],
}

fn f(v: &Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(default)
}

fn stats_from(unit: &Value, upgrade_over: Option<&Value>) -> UnitStats {
    // Build a getter that prefers the upgrade block, falling back to base.
    let get = |key: &str, default: f64| -> f64 {
        if let Some(up) = upgrade_over {
            if let Some(x) = up.get(key).and_then(Value::as_f64) {
                return x;
            }
        }
        f(unit, key, default)
    };
    let speed = get("speed", 0.0);
    let frames_per_move = if speed > 0.0 { (1.0 / speed).round() as u32 } else { 0 };
    let is_structure = f(unit, "unitCategory", 0.0) as i64 == 0;
    let base_cost_sp = f(unit, "cost1", 0.0) as f32;
    let upgrade_cost_sp = unit
        .get("upgrade")
        .and_then(|u| u.get("cost1"))
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(base_cost_sp);
    UnitStats {
        cost_sp: base_cost_sp,
        cost_mp: f(unit, "cost2", 0.0) as f32,
        start_health: get("startHealth", 0.0) as f32,
        dmg_walker: get("attackDamageWalker", 0.0) as f32,
        dmg_tower: get("attackDamageTower", 0.0) as f32,
        attack_range: get("attackRange", 0.0),
        hit_radius: get("getHitRadius", 0.0),
        frames_per_move,
        shield_per_unit: get("shieldPerUnit", 0.0) as f32,
        shield_range: get("shieldRange", 0.0),
        shield_bonus_per_y: get("shieldBonusPerY", 0.0) as f32,
        sd_walker: get("selfDestructDamageWalker", 0.0) as f32,
        sd_tower: get("selfDestructDamageTower", 0.0) as f32,
        sd_range: get("selfDestructRange", 0.0),
        sd_steps: get("selfDestructStepsRequired", 0.0) as u32,
        breach_damage: get("playerBreachDamage", 0.0) as f32,
        metal_for_breach: get("metalForBreach", 0.0) as f32,
        refund_pct: get("refundPercentage", 0.0) as f32,
        is_structure,
        upgrade_cost_sp,
        has_upgrade: unit.get("upgrade").is_some(),
    }
}

/// Max integer d2 strictly below (range + hit)^2. Because no sqrt(integer) lies within
/// (range+0.01, range+0.03) for any range this season, this is exact (see MECHANICS.md).
fn d2_threshold(range: f64, hit: f64) -> i32 {
    if range <= 0.0 {
        return -1;
    }
    let lim = (range + hit) * (range + hit);
    let mut t = lim.floor() as i32;
    if (t as f64) >= lim {
        // boundary exactly on an integer: strict '<' excludes it
        t -= 1;
    }
    t
}

impl Config {
    pub fn from_json(root: &Value) -> Config {
        let ui = root
            .get("unitInformation")
            .and_then(Value::as_array)
            .expect("config: unitInformation");
        let mut units: [[UnitStats; 2]; 6] = Default::default();
        for k in 0..6 {
            let u = &ui[k];
            units[k][0] = stats_from(u, None);
            units[k][1] = stats_from(u, u.get("upgrade"));
            for s in units[k].iter() {
                assert!(
                    u.get("generatesResource1").is_none() && u.get("generatesResource2").is_none(),
                    "resource-generating structures are not implemented (old-season config?)"
                );
                let _ = s;
            }
        }
        let res = root.get("resources").expect("config: resources");
        let turns_to_remove = ui[0]
            .get("turnsRequiredToRemove")
            .and_then(Value::as_f64)
            .unwrap_or(1.0) as u32;

        let mut attack_d2 = [[[-1i32; 6]; 2]; 6];
        let mut shield_d2 = [[-1i32; 2]; 6];
        let mut sd_d2 = [[-1i32; 2]; 6];
        for a in 0..6 {
            for up in 0..2 {
                let st = &units[a][up];
                for t in 0..6 {
                    // hit radius belongs to the target unit (base variant radius; upgrades
                    // don't override getHitRadius this season).
                    attack_d2[a][up][t] = d2_threshold(st.attack_range, units[t][0].hit_radius);
                }
                // shields include the exact-distance boundary (dist <= range + hit):
                // upgraded support range 7.0 DOES grant at d2 = 49 — confirmed by the
                // previously-exact corpus regressing under a strict-< experiment.
                shield_d2[a][up] = d2_threshold(st.shield_range, units[SCOUT as usize][0].hit_radius);
                sd_d2[a][up] = d2_threshold(st.sd_range, units[SCOUT as usize][0].hit_radius);
            }
        }

        Config {
            units,
            start_hp: f(res, "startingHP", 30.0) as f32,
            start_sp: f(res, "startingCores", 40.0) as f32,
            start_mp: f(res, "startingBits", 5.0) as f32,
            sp_per_round: f(res, "coresPerRound", 5.0) as f32,
            mp_per_round: f(res, "bitsPerRound", 5.0) as f32,
            mp_interval: f(res, "turnIntervalForBitSchedule", 10.0) as u32,
            decay: f(res, "bitDecayPerRound", 0.25) as f32,
            max_mp: f(res, "maxBits", 999_999.0) as f32,
            mp_growth: f(res, "bitGrowthRate", 1.0) as f32,
            turns_to_remove,
            attack_d2,
            shield_d2,
            sd_d2,
        }
    }

    #[inline]
    pub fn stats(&self, kind: u8, upgraded: bool) -> &UnitStats {
        &self.units[kind as usize][upgraded as usize]
    }
}

/// STATS serialization: the engine keeps resources as raw f32 (no rounding anywhere;
/// affordability uses the raw value) and serializes hp/sp/mp as ceil-to-the-next-tenth
/// computed in f64 after widening (replay-derived: raw 6.12890625 displays 6.2, raw 0.75
/// displays 0.8). UNIT HEALTH serializes as raw f32 shortest-repr instead (literal
/// "6.6000004" observed in replays) — compare unit healths by f32 bits, not display.
#[inline]
pub fn display01(x: f32) -> f64 {
    ((x as f64) * 10.0).ceil() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn season_config() -> Config {
        let text = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game-configs.json"),
        )
        .expect("game-configs.json at repo root");
        Config::from_json(&serde_json::from_str(&text).unwrap())
    }

    #[test]
    fn season_stats_resolve() {
        let c = season_config();
        assert_eq!(c.stats(WALL, false).start_health, 40.0);
        assert_eq!(c.stats(WALL, true).start_health, 120.0);
        assert_eq!(c.stats(WALL, false).upgrade_cost_sp, 1.0); // inherits base cost
        assert_eq!(c.stats(WALL, false).refund_pct, 0.8);
        assert_eq!(c.stats(TURRET, false).attack_range, 4.5);
        assert_eq!(c.stats(TURRET, true).attack_range, 3.5); // upgrade DROPS range
        assert_eq!(c.stats(TURRET, true).dmg_walker, 16.0);
        assert_eq!(c.stats(TURRET, false).upgrade_cost_sp, 4.0);
        assert_eq!(c.stats(SUPPORT, true).shield_range, 7.0);
        assert_eq!(c.stats(SUPPORT, true).shield_bonus_per_y, 0.3);
        assert_eq!(c.stats(DEMOLISHER, false).cost_mp, 2.0);
        assert_eq!(c.stats(DEMOLISHER, false).frames_per_move, 2);
        assert_eq!(c.stats(INTERCEPTOR, false).frames_per_move, 4);
        assert_eq!(c.stats(INTERCEPTOR, false).dmg_tower, 0.0);
        assert_eq!(c.stats(SCOUT, false).frames_per_move, 1);
    }

    #[test]
    fn range_thresholds() {
        let c = season_config();
        // turret base 4.5: (4.5+0.03)^2 = 20.52 -> d2 <= 20; vs scout radius applies per
        // target, but no sqrt(n) in the gap, so both 0.01 and 0.03 land at 20.
        assert_eq!(c.attack_d2[TURRET as usize][0][SCOUT as usize], 20);
        assert_eq!(c.attack_d2[TURRET as usize][1][SCOUT as usize], 12); // 3.5 -> 12.32
        assert_eq!(c.attack_d2[SCOUT as usize][0][WALL as usize], 12); // 3.5
        assert_eq!(c.attack_d2[DEMOLISHER as usize][0][TURRET as usize], 20); // 4.5
        assert_eq!(c.sd_d2[SCOUT as usize][0], 2); // 1.5 -> 2.34
        assert_eq!(c.shield_d2[SUPPORT as usize][0], 6); // 2.5 -> 6.40
        assert_eq!(c.shield_d2[SUPPORT as usize][1], 49); // 7.0 -> 49.42 (49 included!)
    }

    #[test]
    fn display_semantics() {
        assert_eq!(display01(8.75), 8.8);
        assert_eq!(display01(6.12890625), 6.2);
        assert_eq!(display01(0.75), 0.8);
        assert_eq!(display01(5.0), 5.0);
        // f32 gameplay arithmetic: the replay-observed artifact
        let hp: f32 = 26.6;
        assert_eq!(format!("{:?}", hp - 20.0f32), "6.6000004");
    }
}
