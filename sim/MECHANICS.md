# Terminal Engine Mechanics — VERIFIED SPEC (this competition's ruleset)

Status: **frame-exact vs engine.jar across the full corpus** (4 matches, 2,719 action
frames: starter mirror, torture mirror, torture-vs-starter, probe-vs-torture). Unit
positions/healths/events bit-exact; resources within one display tick (see §Serialization).
Verification tool: `sim/target/release/tsim diff <replay>`. Re-run on every sim change and
on every new replay.

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
