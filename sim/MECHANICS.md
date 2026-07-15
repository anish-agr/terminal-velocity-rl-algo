# Terminal Engine Mechanics — VERIFIED SPEC (this competition's ruleset)

Status: **frame-exact vs engine.jar across the local corpus** (4 matches, 2,719 action
frames). **Production-server replays** (pulled via
`https://terminal.c1games.com/api/game/replayexpanded/<id>` — public, no auth) confirm the
server runs the same engine and a gameplay-identical config (diffs are icon fields only).
Two gaps surfaced by server replays, both characterized (see §Open fixes). Verification
tool: `sim/target/release/tsim diff <replay>`. Re-run on every sim change and every new
replay.

## Open fixes (root-caused, exact fix logic specified — implement in this order)

1. **Self-destruct AoE must hit 0-health units too.** The engine's SD damages every enemy
   unit in range whose removal hasn't happened yet (step 4), INCLUDING units already at
   ≤0 health from earlier SDs this frame; our sim filters `health > 0`, producing fewer
   damage events and shorter SD target lists (ladder replay turns 7/12/14/19: sim damage
   15 vs engine 20, etc.). FIX: in the SD branch of the movement step, drop the
   `health > 0` filters on both mobile and structure targets (keep `alive`); the 0-health
   exclusion applies to ATTACK targeting only. Gate: both ladder replays reach the same
   pass-rate profile as the platform-probe replay.
2. **Shield-pool micro-dust (±4e-6 HP).** Platform 100-turn replay: units with many
   stacked grants (8 observed) differ from the engine by ~4e-6 (sim 11.600002 vs engine
   11.599998) — the engine computes the grant amount along a different float path than
   our f32 chain (likely f64 `shieldPerUnit + bonus*y` cast to f32 once). FIX (two-part):
   (a) compute the amount in f64, cast to f32 at grant time; if residue persists, (b) add
   a relative epsilon of 1e-4 to the diff's unit-health comparison ONLY (never to
   positions/events), documented as characterized dust. Materiality: damage values are
   integers; a 4e-6 health offset changes a kill threshold only if effective health sits
   within 4e-6 of an exact damage multiple — negligible, but keep exactness where free.
3. **Bounded MP micro-drift on long banking chains (BOTH signs, ≤ ~0.1).** Exhaustive
   model search (32 variants: f32/f64 × mul/sub × {none, round, rint, ceil, floor}-at-
   tenths × ceil/round display) — NO variant survives chain validation across the corpus;
   the pure-banking chain in ladder-15330187 requires engine-internal values that cannot
   arise from ANY function of the displayed state (t2 must exceed 8.7627 while exact
   arithmetic from turn 0 gives 8.75; yet at t17 the engine sits BELOW the raw chain).
   Model kept: raw f32 (best fit, simplest). Consequences + mitigations:
   - replay harness: per-turn resync + one-display-tick stats tolerance (in place);
   - deployment: read own resources from the server state every turn; plan spends with a
     0.1 margin at integer boundaries; deploys the engine can't afford are SILENTLY
     SKIPPED, so optimistic attempts are free — attempt the marginal unit, never rely on it;
   - self-play training: unaffected (the sim is its own engine, exactly self-consistent).

## External bug-report audit (2026-07-15) — verify before "fixing"

Three reported "critical bugs" were tested against engine data. Verdicts:
- **"Shields are capped / max-overwrite instead of additive" — REFUTED.** The platform
  100-turn replay shows one unit receiving 8 separate shield grants; an interceptor
  reached 56.6 HP (40 + 16.6) and a demolisher 46.6 (5 + 41.6). Pure summation passes
  4,094/4,114 frames on that replay. No 44.1-HP unit exists anywhere in the corpus. The
  rules page also states "no limit to the amount of shielding". Do NOT change.
- **"Frame order must be shields → movement → attacks" — REFUTED.** Dispositive event
  evidence: scouts spawned at (13,0) inside a support's range receive their grant with
  target location (13,1) — their POST-move tile (turn-5/7 frame-0 shield events, local
  corpus). Movement→shields→attacks passes 7,500+ frames including production. Reverting
  to the rules-page order would itself introduce the 1-frame lag the report describes.
- **"Engine quantizes MP to tenth/hundredth after decay; spawn accepted at 1.0 vs sim
  0.9977" — MECHANISM REFUTED, underlying issue REAL.** All quantization variants fail
  chain validation (see Open fix 3, which also covers the affordability symptom: engine
  accepted deploys our chain under-affords by <0.1). Handled by Open fix 3 mitigations,
  not by quantizing.

## Numerics (hard-won, do not regress)

- **The engine uses 32-bit floats (Java `float`) for all gameplay scalars** — health,
  shields, damage, resources. Proof: replays contain literal `6.6000004` = f32(26.6)−20.
  The Rust sim mirrors f32 everywhere.
- **No internal rounding of resources.** Raw f32 accounting; affordability uses raw values
  (a displayed 6.2 MP may only afford 6 scouts — plan with ≥0.05 margin).
- Mobile units carry a **separate shield pool**: serialized health = base + pool (single
  f32 add); damage drains pool first, spill = `base -= d - pool`. Shield amount =
  `shieldPerUnit + shieldBonusPerY * own_side_y(support)` in f32 (own_side_y = y for P1,
  27−y for P2; y-formula uses the SUPPORT's position, verified at y=3 → 4.9 and y=9 → 6.7).

## Serialization (replay/diff contract)

- **Stats (HP/SP/MP)**: ceil-to-next-tenth of raw f32, computed in double
  (raw 6.12890625 → "6.2", raw 0.75 → "0.8"). Engine-internal add ORDER inside restore is
  unobservable at ±1 ulp; the diff tolerates one display tick (0.1) on stats only.
- **Unit health**: raw f32 shortest-repr ("6.6000004"). Compare by f32 bits.
- Shield event field [3] = the GIVER's unit type (API doc wrong).
- IDs: global counter in creation order; upgrades consume IDs and the upgraded structure is
  re-listed under its upgrade's ID.
- Removal-flag death events appear in the TURN frame (phase 0), not action frames.

## Turn cycle (all replay-verified)

1. **Restore**: pending removals execute (marked structures survive the ENTIRE previous
   action phase; refund = `refundPct × invested × health/maxHealth` in f32, credited now;
   ⚠ upgraded-structure refund uses base+upgrade cost — assumed, exercise pending) →
   MP decay `mp *= (1 - 0.25)` raw f32 → income +5 SP, +(5 + floor(turn/10)) MP, MP capped
   at 150.
2. **Deploy**: engine applies P1 builds+upgrades, P2 builds+upgrades, P1 mobiles, P2
   mobiles. Wall upgrade costs base cost (1). Upgrades keep missing health
   (health += upgMax − oldMax). Turret upgrade: dmg 5→16, range 4.5→**3.5** (drops!).
3. **Action phase**, frames until no mobile units alive (frame 0 always runs):
   1. **Movement** (creation order; cadence: unit moves on frame f iff
      `(f+1) % framesPerMove == 0`; scout 1, demolisher 2, interceptor 4):
      - Step per pathfinder (below). Stepping onto the target edge does NOT score yet.
      - A unit standing ON its target edge that attempts to move **breaches**: −1 enemy HP,
        +1 SP, disappears. (It spends ≥1 frame on the edge attacking first, and CAN be
        killed there before scoring — corner turrets matter.)
      - A unit due to move with nowhere to go (trapped or at a non-edge destination)
        **self-destructs**: health & shield → 0; if steps ≥ 5, deals its selfDestruct
        damage (walker/tower values) to ENEMY units within d² ≤ 2; selfDestruct event only
        when steps ≥ 5; the unit still attacks this frame; removed at step 4.
   2. **Shields**: every support × every alive friendly mobile with base health > 0, in
      creation order, once per (support, unit) pair, d² ≤ 6 base / 49 upgraded, at
      POST-MOVE positions. (Rules page lists shields first; engine grants after movement.)
   3. **Attacks**: every unit in creation order, sequential damage. Target priority:
      mobile > structure; min d²; min effective health (base+shield); deepest toward
      attacker's side (P1: min y, P2: max y); max |x−13.5|; then most-recently-created.
      0-health units are untargetable. Overkill does not spill. Interceptors and turrets
      cannot hit structures. Range: d² ≤ floor((range + hitRadius)²) — integer thresholds,
      no boundary cases this season (turret 4.5→20, upgraded 3.5→12, scout 3.5→12,
      demo/interceptor 4.5→20). ⚠ unit reduced to ≤0 earlier in the same frame still
      attacks when its turn comes (default; no distinguishing case seen yet).
   4. **Deaths**: health ≤ 0 removed; ANY structure death ⇒ all mobiles repath from their
      current tile (units keep their previous-move-axis memory across repaths ⚠ default).
4. Game end: HP ≤ 0; after round 100 higher HP wins; ties → lower total computation time.
   Turn timing: >5s soft = 1 HP/sec penalty; 35s = skipped turn ("timeout death").

## Pathfinder (port of official kit, all cases verified incl. double-backs)

Idealness BFS over the pocket (idealness = 28·depth + lateral toward target; target-edge
tile = ∞ short-circuit → endpoints = whole target edge; else endpoint = single most-ideal
tile). Validation BFS pathlengths from endpoints. Step = min-pathlength neighbor;
ties: first-ever move prefers vertical → prefer axis change vs previous move → prefer
toward target edge. Implementation: `NavField` cached per (edge, pocket, layout_version);
units walk fields with persistent axis memory (sim/src/path.rs).

## Config confirmations (rules page + replay)

Special ruleset diffs all match our game-configs.json: Wall 40 HP (60 base), Support
shield 2 @ 2.5 (3 @ 3.5), Demolisher 6 dmg @ 2 MP (8 @ 3), Interceptor 15 dmg (20).
Verify engine.jar version matches the portal's before trusting new mechanics conclusions.

## Sim performance

Single core (laptop): ~5,500 turns/sec, ~480K frames/sec release build. Path fields cached
per layout; targeting via integer-d² thresholds.
