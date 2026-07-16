# Terminal Engine Mechanics — VERIFIED SPEC (this competition's ruleset)

Status: **frame-exact vs engine.jar across the local corpus** (4 matches, 2,719 action
frames). **Production-server replays** (pulled via
`https://terminal.c1games.com/api/game/replayexpanded/<id>` — public, no auth) confirm the
server runs the same engine and a gameplay-identical config (diffs are icon fields only).
Two gaps surfaced by server replays, both characterized (see §Open fixes). Verification
tool: `sim/target/release/tsim diff <replay>`. Re-run on every sim change and every new
replay.

## Fix status (2026-07-16, round 3 — full corpus revalidation, 3,847 replays)

Full re-validation via `scripts/batch_diff.sh` against `replays/scraped/*.replay` (all
3,847 files, using the corrected `tsim.exe` — see below for the corrupted-binary incident
this superseded): **3,489/3,847 full-replay PASS (90.7%), frame-exact 3,446,733/3,451,316
(99.8672%)**.

**Corrupted-binary incident (caught before being reported as fact):** an intervening
`cargo build --release --features python` (run to refresh `terminal_sim.pyd`) silently
rebuilt the standalone `tsim` CLI binary under PyO3's `extension-module` ABI too — Cargo
applies `--features` package-wide by default, and that ABI makes the CLI binary exit 0 with
zero output when run outside a Python host process. A batch run against that broken binary
produced a nonsensical "243/3103 PASS, 99.9967% frame-exact" result (internally
inconsistent — frame count crashed to ~241K from an expected ~3.86M, several worst-10
entries were literal all-zero rows, a crash signature). Fixed by rebuilding the plain
feature-free binary (`cargo build --release`, no features) before any fidelity gate, and
the same ordering hazard was fixed in `train/setup_runpod.sh` (which built the python wheel
*before* its fidelity gate — would have failed the gate on every pod run). Repo-wide grep
confirmed no other script has this hazard.

**Root cause of the dominant remaining divergence category (diagnosed, NOT a sim bug):**
triaged the worst-10 replays by frame-mismatch count (594, 256, 182, 178, 153, 150, 110, 77,
77, 77 — a handful of replays account for a disproportionate share of the corpus's frame
mismatches because one divergence cascades forward for the rest of that replay). Every one
of the 5 checked has the identical signature: the FIRST divergence is a small (1-3 point)
player-HP mismatch at a turn-boundary frame, with **zero** breach/damage/selfDestruct event
anywhere in that frame's event log, and no enemy mobile unit anywhere near the breaching
edge. Confirmed by exhaustively scanning one full replay (15330159) for every breach event
of either owner across the whole game: player 1's HP always drops in perfect lock-step with
a logged owner=2 breach event in the same frame (correct, expected) — but player 2's HP
drops twice (turn 26, turn 30) with **no owner=1 breach event anywhere in the entire
replay**. Replay frames carry no compute-time metadata at all.

This matches the competition's own turn-timer rule (5s soft limit, **1 HP/sec penalty for
time over the limit**, 35s = skipped turn) — a real bot on the ladder occasionally thinking
too long and eating an HP penalty that is a pure server-side artifact of that bot's
wall-clock compute time, invisible in the replay JSON. A rules-only simulator cannot
reconstruct this from replay data under any circumstances — it isn't a state-transition bug,
it's missing input. **Do not chase this further** — it inflates the "divergent replay" count
without reflecting any real engine inaccuracy. True engine fidelity (excluding
compute-time-penalty replays) is materially higher than the raw 90.7%/99.87% headline
numbers. If a specific replay needs to be confirmed as a timeout case vs. a real bug, look
for the same signature (isolated small HP delta, zero causal event, no enemy unit near the
edge) before assuming it's fixable.

## Fix status (2026-07-16, round 2 — externally reported drift bugs)

Four externally-reported "critical drift bugs" were verified against fresh replay evidence
before touching anything (per the audit-first policy below). Two were real; the sim's
overall fidelity on a fixed 443-replay sample (deterministic subset, every 7th scraped
replay by ID) moved from **99.13% -> 99.93% frame-exact** and **98.61% -> 99.73%
restore-exact** as a direct result — a ~13x and ~5x reduction in residual error rate,
respectively, with zero regression on any metric. Full 3,103-replay corpus re-run in
progress; see `replays/scraped/diff_results.tsv` for the latest numbers.

1. **CONFIRMED — SP refund quantized PER STRUCTURE to the nearest tenth, round-half-up,
   before being summed/credited.** Root-caused against `scraped/15327732.replay` turn 16
   (P2 removes 4 walls + 2 upgraded walls): raw refund sum = 5.146667, engine-observed
   refund = 5.100000. Then validated at scale: scanned every scraped replay for turns
   where a player's ONLY frame0 commands were REMOVEs (isolates the SP delta cleanly),
   found 229 such turns across 400 replays, and checked the observed SP delta against
   raw-sum / floor-sum / round-sum / ceil-sum of the individually-computed refunds (using
   each structure's health at the END of that turn's action phase, matching our
   restore-time execute_removals). Result: **round-sum matches 229/229; raw only 184/229;
   floor 206/229; ceil 199/229.** Fix applied in `state.rs::execute_removals`:
   `refund = (raw*10.0 + 0.5).floor() / 10.0` per structure, summed after.
2. **CONFIRMED — upgrading a structure assigns it a NEW engine id, and that id becomes its
   new attack-order tie-break seq.** Root-caused against the same replay, turn 8: a wall
   built at (22,15) gets id 143; upgrading it the same turn logs a SEPARATE spawn event,
   id 158, same tile. Our sim already advanced the global id counter on upgrade (so
   replay-mode id matching was fine) but never wrote the new id back onto the structure's
   own `seq` field — which is what attack ordering actually sorts by. Almost certainly the
   dominant contributor to the "rare targeting near-ties" residual documented below. Fix:
   `state.rs::apply_one` (Upgrade arm) now sets `s.id = id; s.seq = id;` from the same
   `take_id()` call already in use.
3. **REFUTED — "build over an existing structure."** Claimed evidence: P1 builds at
   (0,13)/(1,13)/(2,13) on turn 24 of `15327732.replay`, tiles that held a wall+2 turrets
   at turn 23. Traced directly: those three structures show ordinary COMBAT deaths
   (`removal_flag=False`) at turn 23 frame 26 — destroyed by enemy fire, not replaced.
   By turn 24's deploy the tiles are already empty; these are unremarkable builds onto
   empty ground, which the engine already handles correctly (Deaths step clears the grid
   entry same-frame). No such mechanic exists; it would also contradict the rules text
   ("no two Structures can occupy the same location"). `Cmd::Build`'s occupancy check is
   unchanged.
4. **REFUTED as a distinct bug — `upgrade_cost_sp` JSON fallback.** Claimed the config
   parser falls back to 0.0 for a missing `upgrade.cost1`, producing a 0.4 SP variance at
   turn 21 of the same replay. The parser already falls back to the BASE unit's cost (not
   0.0) — `unwrap_or(base_cost_sp)` — independently verified correct earlier via turn-0 SP
   accounting (season config: wall upgrade omits cost1, engine charges 1 SP = base cost).
   The claimed 0.4 SP variance is fully explained by bug #1 above (unrounded refund sum);
   a single upgraded-wall refund's raw-vs-rounded gap is easily that large. config.rs
   unchanged.

## Fix status (2026-07-16, round 1)

APPLIED & validated: (1) SD AoE hits <=0-health units — ladder replays now frame-exact;
(2) action phase ends immediately when a player reaches 0 HP mid-phase (engine stops
emitting frames — ladder-15330187 turn 27); (3) replay harness force-applies
engine-accepted commands past affordability (bounded MP drift, §3 below) and screams
"GEOMETRY GAP" if a forced command is structurally impossible (= real bug); (4) unit
health comparison quantized to 0.01 (absorbs shield-pool f32 dust <=1e-3).
CONFIRMED-CORRECT after a false fix attempt: shield grant amount is the PURE f32 chain
`shieldPerUnit + shieldBonusPerY * own_y` (local engine grants print "6.7" = exactly what
the f32 chain yields; widening through the f32-stored config to f64 lands one ulp HIGH —
reverted; see engine.rs comment).

KNOWN RESIDUALS (characterized, deliberately not chased):
- platform-probe-match turns 44 & 55 (3 frames of 4,114): rare attacker/target
  disagreement under near-ties (two turrets, equal-range equal-health targets) —
  reproduction pointers preserved in that replay; revisit only if scaled validation shows
  the pattern is common.
- SHIELD ATTRIBUTION in deep stacks (scraped/15327711 turn 15 frame 26): with ~18 scouts
  stacked, TWO units ended the phase missing exactly two specific supports' grants
  (6.4 + 4.6 = the observed 11.0 hp delta) in the engine while our sim granted uniformly;
  event multisets match (grants differ only in receiving unit ID). Hypothesis space:
  engine's once-per-pair bookkeeping interacts with unit iteration order when stacks
  split. NOTE: a strict dist<range shield-boundary experiment (excluding d2=49 for range
  7.0) was tested and REFUTED — it regressed the previously-exact corpus; inclusive
  d2<=49 is correct. Frequency quantified by the corpus batch run (diff_results.tsv).
- resource micro-drift on long banking chains (§Open fix 3 below) — bounded < 0.1,
  mitigations in place, mathematically shown unresolvable from replay data alone.
- REFUND of marked-then-destroyed structures (scraped/15329386 turn 10): a marked wall
  destroyed mid-phase appears to yield a partial refund (~hp-correlated, +0.22 observed)
  that we don't model; our refunds (survivors at end-of-phase health) match the other 7
  removals exactly. Magnitude <=0.8 SP, frequency ~0.3% of turns (mass-marking bots).
  Candidate mechanisms tested and failed: mark-time valuation (5.68), per-refund
  rounding modes (4.8/4.9/5.5), all-8-at-end (n/a). Deployment unaffected (server SP is
  read each turn); training impact negligible.
- Engine's own shield event amounts confirm the pure-f32 amount chain: production replay
  15327711 serializes a grant amount as literal "6.1000004" = f32 chain of 4 + 0.3f*7.

## Historical fix specs (superseded by the section above)

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
