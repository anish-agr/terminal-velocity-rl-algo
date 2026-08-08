# Terminal Velocity — Engineering Report

*Citadel's Terminal competition, July 2026. A Rust game-engine reimplementation, a league-self-play RL system, and a deployment stack for a hostile container, built in 13 days. This report goes below the README's headline numbers: how the engine was rebuilt and verified to 99.87% frame fidelity, the game mechanics reverse-engineered from production replay data, what the training league actually did, and how the shipped algo plans a turn on the ranked server.*

---

## At a glance

| | |
|---|---|
| **Simulator** | From-scratch Rust reimplementation of the Terminal engine: 2,989 lines, one runtime dependency, ~5,500 turns/sec/core. Reproduces **99.87% of 3,451,316 real engine frames exactly** across 3,847 scraped ranked replays; 3,489 of them replay end-to-end without a single divergent frame. |
| **Training** | Behavior cloning over ~310K positions from 3,887 config-matched ranked replays, then league self-play (PFSP snapshots + seven scripted archetypes) on an H100, with every move chosen by a security-weighted K×M search — the same search code that runs on the ranked server. |
| **Deployment** | A four-rung strategy ladder inside a restricted container: pure-numpy inference (torch's 2.5 s import is unaffordable at match time), a reconstructed "mirror" simulator of the live opponent, an anytime search under a 5-second watchdog, and complete scripted strategies beneath it that always submit a legal turn. |
| **Validation** | Everything measured against the real engine: a deterministic ten-bot sparring arena gating every change, a frame-diff harness re-run on every simulator edit, numpy and Rust forward passes parity-tested to <1e-4 on every weights export, and a probe algo that measured the competition container from the inside before the deployment design was fixed. |
| **Final build** | 10–0 on the sparring panel with margins of +13 to +41; ~1400–1500 on the ranked ladder. Ships the full inference stack behind a single switch, with the arena-proven scripted game plan playing primary. |

Everything below is traceable to the repo: the mechanics spec ([sim/MECHANICS.md](sim/MECHANICS.md)), the training spec ([train/ARCHITECTURE.md](train/ARCHITECTURE.md)), the commit history, and the corpus manifests in `replays/scraped/`.

---

## 1. The problem

Terminal is a simultaneous-commit tower-defense game on a 28×28 diamond. Each turn, both players secretly commit a full plan — structures built, upgraded, or marked for removal; mobile units launched — and the turn then resolves as ~30 deterministic action frames of pathing, targeting, shields, and breaches. The action space is a combinatorial multiset, the physics are deterministic but deeply stateful, and the ranked ladder is a zoo of sharply tuned archetypes (scout floods, demolisher grinders, funnel turtles, corner snipes), each punishing a different structural weakness.

Three properties drove every design decision:

1. **Evaluating a plan requires simulating it.** There is no useful static evaluation of a Terminal position; whether a 14-scout wave breaches depends on exact pathing through exact structures with exact shield arithmetic.
2. **Moves are simultaneous**, so planning needs a distribution over what the opponent is about to do, not just a value function.
3. **The server is restrictive**: effectively one core, a 5-second soft cap per turn (1 HP/sec penalty for overruns), a 50 MB unpacked limit, and no local access to ranked conditions — so every capability has to be built and proven off-platform first.

## 2. The simulator

### 2.1 Why reimplement the engine

The official engine is `engine.jar` — a black-box Java process that plays one match at a time over a socket protocol. Three consumers needed something it cannot provide:

- **The search** forks a game state for every (own plan, opponent plan) pair — K×M ≈ 72 simulator forks per decision in training, 128 on the server — inside a per-turn budget. That requires an in-process, copyable state, not a subprocess.
- **Self-play training** needs throughput: a 50-turn game is ~100 decisions ≈ 7,000 simulations, at a target of 60–150 games/min on a CPU pod.
- **The replay pipeline** needs to re-execute scraped games to extract training tensors and to verify its own parsing.

So the engine was reimplemented in Rust ([sim/src/](sim/src/)): 2,989 lines across ten modules, one mandatory dependency (`serde_json`), PyO3 bindings behind a feature flag, compiled with fat LTO. Single-core throughput on a laptop: **~5,500 turns/sec, ~480K action frames/sec**.

### 2.2 Inside the engine

The interesting parts of the implementation are the ones the rules page doesn't tell you:

- **Frame resolution order is empirical, not documented.** The implemented order — movement (including breach-on-next-attempt and self-destruct arming) → shield grants → attacks in unit-creation order with sequential damage and zero-health targets immediately untargetable → deaths, with pathing recomputed when a structure dies — was derived from event evidence and holds across 7,500+ verified frames. The rules page implies a different order; production data settles it (§2.4).
- **Pathing is an exact port of the official navigation algorithm**, with a `NavField` cached per (target edge, pocket, wall layout) so repeated queries against an unchanged board are free. Targeting uses integer squared-distance thresholds throughout — no floating-point range comparisons — which is both faster and immune to epsilon drift.
- **Resource arithmetic reproduces the engine's f32 chains bit-for-bit** rather than computing "mathematically correct" values. One production replay serializes a shield grant as the literal `6.1000004` — exactly the f32 evaluation of 4 + 0.3ₓ×7 — and the sim matches it because it runs the same chain. An f64-widened version was tested and lands one ulp high; exactness at this level is what lets three million frames diff clean.
- **The Python surface is small and search-shaped**: a `Game` class with `fork()` (deep copy for search branching), `play_turn(p1_cmds, p2_cmds)` returning frame count, breach totals, and structure damage, `board_planes()` emitting feature tensors as raw bytes, and a tuple command ABI in which invalid commands are silently skipped — the engine's own semantics, so a plan legal in the sim is a plan the server accepts.
- The same crate embeds a hand-written forward pass of the trained network ([sim/src/nn.rs](sim/src/nn.rs), 477 lines) as the fastest rung of the deployment inference ladder — GRU cell, pointer attention, and all — parity-tested against PyTorch primitive by primitive.

### 2.3 How fidelity was measured

The measuring instrument is `tsim diff` ([sim/src/replay.rs](sim/src/replay.rs)): for every turn of a real replay, it reconstructs the state from the replay's turn frame, re-applies the exact deploy commands (recovered from spawn events, with engine unit IDs), re-simulates the action phase, and diffs **every frame** — unit positions, health values, event streams. The per-turn *resync* design is deliberate: it isolates divergences per turn instead of letting one early mismatch invalidate the rest of the match, so each divergence is an independent, attributable data point. The end-of-turn restore step (removals, refunds, decay, income) is validated separately against the *next* turn frame. The CLI exits nonzero on any divergence, which makes it a drop-in gate: the training-pod bootstrap refuses to proceed unless every bundled replay passes.

The corpus grew in rounds. A resumable parallel scraper ([scripts/scrape_replays.py](scripts/scrape_replays.py)) harvested the public replay API and kept games whose embedded config matched this competition's exact ruleset — 19 gameplay fields per unit type (normalized to effective-upgrade stats, since upgrade configs inherit missing fields from the base unit) plus 9 resource-schedule fields, compared field-by-field with cosmetic fields ignored. Final corpus: **4,041 IDs scanned, 3,887 config-matched matches, 3,847 diffed against the sim.**

| Measurement round | Corpus | Frame-exact |
|---|---|---|
| Round 2, before the last two fixes | 443-replay deterministic subset | 99.13% |
| Round 2, after | same subset | **99.93%** |
| Round 3, full corpus | 3,847 replays / 3,451,316 frames | **99.87%**, 3,489 replays exact end-to-end |

The jump from 99.13% to 99.93% — a ~13× reduction in residual error — came from **two one-line fixes**, each a genuine reverse-engineering find backed by recorded evidence:

- **SP refunds are quantized per structure** to the nearest tenth, round-half-up, *before* summing. Root-caused on a single turn where the raw refund sum was 5.146667 and the engine paid 5.100000; then validated at scale: across 229 removal-only turns in 400 replays, per-structure rounding matches **229/229**, while raw summation matches only 184, floor 206, ceil 199.
- **Upgrading a structure assigns it a new engine ID**, and that ID becomes its new attack-order tie-break. The replay shows it plainly — a wall built as id 143 logs a second spawn event as id 158 on upgrade — and the tie-break follows the new ID.

Neither behavior appears in any documentation. Both are now permanent, evidenced entries in the mechanics spec.

### 2.4 An evidence-first mechanics spec

[sim/MECHANICS.md](sim/MECHANICS.md) does something most specs don't: it records the hypotheses that were **tested against production data and rejected**, so verified mechanics are protected from plausible-sounding "fixes" later. During the competition, external reports claimed several critical sim bugs; each was adjudicated with replay evidence before any code changed:

- **"Shields must be capped."** Production replays show a single unit receiving 8 separate shield grants; an interceptor reached 56.6 HP. Pure summation passes 4,094/4,114 frames, and no capped-value unit exists anywhere in the corpus. Rejected.
- **"Frame order must be shields → movement → attacks"** (as the rules page implies). Dispositive event evidence: scouts spawned inside a support's range receive their shield grant tagged with their *post-move* tile. The implemented order passes 7,500+ frames; the proposed one would introduce the exact 1-frame lag it claimed to remove. Rejected.
- **"The engine quantizes MP after decay."** All 32 candidate models (f32/f64 × mul/sub × five rounding modes × two display conventions) were enumerated; none survives chain validation — one turn requires the bank above what exact arithmetic gives, a later turn requires it below. The raw f32 chain was kept as best-fit. Rejected.

The remaining 0.13% of divergent frames is *characterized*, not hand-waved: a deep-stack shield-attribution quirk (~18 units stacked, two specific grants land on different unit IDs than the sim's — event multisets still match), targeting near-ties (3 frames in 4,114 in the most affected replay), and sub-0.1 resource micro-drift on long banking chains that is provably unresolvable from replay data alone.

The forensic capstone: the *largest* remaining divergence category turned out not to be simulator error at all. The worst-diverging replays share one signature — a 1–3 point player-HP drop at a turn boundary with no breach, damage, or self-destruct event anywhere near it; in one fully scanned replay, a player's HP falls twice with no opposing breach event in the entire file. Diagnosis: the competition's **compute-time penalty** (1 HP per second over the 5-second cap), a server-side function of the opposing bot's wall-clock time that is simply absent from replay JSON — missing input, not a state-transition bug. Excluding those replays, true engine fidelity is materially *higher* than the 99.87% headline. The analysis also quantified exactly how expensive turn-time overruns are on the ranked server, which sized the deployment watchdog.

## 3. The learning system

### 3.1 State, and an action space that can't emit illegal plans

The board is 18 feature planes over the 28×28 grid (12 from the sim bridge — occupancy, health, upgrade state, pending removals — plus 6 deploy-history planes: recent deployment counts and per-turn EMAs for both players) with 14 normalized scalars covering health, both resource banks, the income schedule, and running breach and structure-damage flows.

A turn's plan is a token sequence: up to 24 tokens of (action type, board location, count bucket), with Fibonacci-spaced count buckets plus a spend-everything bucket, closed by an END token. The decoder is constrained by `PlanScratch` ([train/tokens.py](train/tokens.py)), an incremental legality mask maintained *as the sequence grows*: affordability with a safety margin, placement rules, and pruning of provably-null deployments. Masking is deliberately minimal — only provably useless actions are removed, so sacrifices, baits, and slow-plays remain expressible and the search, not a heuristic, judges them. The masks are too large to store with trajectories (9 type + 784 location + 8 count booleans per token, per candidate), so the learner rebuilds them exactly at batch time from stored scratch ingredients — a design that keeps training and acting provably mask-consistent.

### 3.2 A network built to be ported

`TerminalNet` ([train/model.py](train/model.py)) is compact by design: **795,654 parameters (~3.2 MB of f32)**. A 3×3 stem into six norm-free residual blocks at 64 channels — no batch or layer norm anywhere, which keeps CPU inference deterministic and made the numpy and Rust ports line-for-line transcriptions — with the scalar vector injected as a FiLM-style channel bias. Four heads share the torso:

- **value** — a tanh scalar, the only head that scores decisions, trained exclusively on final game outcomes (no shaped rewards, so reward hacking through the value channel is structurally impossible);
- **aux** — three self-supervised 3-turn-horizon deltas (health, board net-worth, resources) for representation shaping, never consulted at play time;
- **policy** — an autoregressive plan decoder: a GRU over token embeddings with spatial pointer attention over the 28×28 feature map for location selection;
- **predict** — a second, independently parameterized decoder run over the *opponent-perspective* encoding, trained to predict the opponent's next plan. Simultaneous-move games need an opponent model, not just an evaluation; this head is it.

### 3.3 The search is the policy

Every decision — in training actors and on the ranked server, byte-for-byte the same code — runs a **security-weighted K×M joint evaluation**: sample K own plans from the policy head (always including the greedy plan and the all-defense plan), M opponent plans from the prediction head (always including the opponent's literal previous plan, re-legalized, and the empty plan), fork the simulator for all K×M pairings, score every resulting position with the value head in one batch, and rank own plans by

> score(i) = λ · Σⱼ wⱼ·v(i,j) + (1−λ) · minⱼ v(i,j)

an expected-case/worst-case blend with temperature-flattened prediction weights (wⱼ ∝ pⱼ^0.5). λ = 0.75 was selected by self-play duels over {0.65…0.95}: high enough that the worst-case term can't collapse the policy into pure defense, low enough to respect rush lines. The search-improved distribution also becomes the policy's training target — the expert-iteration loop that lets a small network punch above its parameter count.

### 3.4 From raw replays to training tensors

Replay files don't contain commands — they contain observations. The corpus pipeline ([train/replays.py](train/replays.py)) reconstructs each player's actual decisions: builds and unit deployments from action-frame spawn events; upgrades from the diff of per-structure marker lists between consecutive snapshots; removals from `was_removed` death events in the *next* turn frame, attributed back to the turn the mark was issued. Health-tied games fall back to the engine's compute-time tiebreak to determine the winner.

Behavior-cloning targets are winners' moves only; losing sides are still ingested for value, aux, and prediction signal, and the prediction head trains on *all* players' moves as a population model. Two de-biasing mechanisms matter: an **opening-fingerprint cap** (SHA-1 of the turn 0–3 build sequence; no single fingerprint may exceed 25% of the BC dataset, so the ladder's dominant meta opening can't monopolize the prior) and **x-mirror augmentation** applied stochastically at sample time — the mirror is implemented in absolute coordinates so it commutes with the perspective flip and one transform serves both seats, with an involution test pinning it.

### 3.5 What the league did

Pure self-play has a known failure mode: strategies that stop appearing in the pool stop being defended against. The league ([train/league.py](train/league.py)) is the counter. Opponents are sampled 35% current parameters / 40% snapshots / 15% scripted bots / 10% frozen BC anchor. Snapshots are drawn by **prioritized fictitious self-play**: sampling weight f(w) = w·(1−w) + ε over a rolling 50-game win rate against each snapshot — mass concentrates on opponents the current agent goes ~50/50 with, unplayed snapshots get an optimistic prior so fresh entries are explored, the ε keeps solved opponents from vanishing entirely, and snapshots beaten >90% continuously for two hours are evicted.

The seven scripted league bots ([train/scripted.py](train/scripted.py)) are pure functions of the visible board — deterministic, stateless across turns, written in perspective space so each plays both seats identically. Five span the ladder's canonical archetypes (rush, funnel, demolisher line, interceptor turtle, plus a mechanics-torture bot that keeps the league honest about weird-but-legal play). The other two were **distilled from ladder study during the competition**: `corner_hammer`, from a replay analysis of the visible #1 seed's nineteen games — full turret line with corner anchors, MP banked to 0.875× the income/decay equilibrium, whole-bank waves aimed each turn at whichever enemy corner scores weaker on upgrade-aware turret coverage — and `line_grinder`, from a study of ranked losses: a solid 27–35-turret line with no walls, cracked only by banked 9–16-demolisher waves that outrange upgraded turrets. Distilling live opponents into league members is the mechanism that keeps a self-play pool anchored to the meta it will actually face.

### 3.6 Training infrastructure

Training ran on a single H100 with a pool of CPU actors playing full-search self-play. The plumbing details are where the throughput lives:

- **Actors own no GPU memory.** Every network call from every actor crosses a **shared-memory inference server** with a 3 ms batching window and batches up to 512, so dozens of self-play processes share one model instance at full batch efficiency alongside the learner. The server wraps the model in the same client class the deployment driver uses, which makes train/deploy consistency structural rather than aspirational.
- **The learner** consumes a 500K-position FIFO buffer (mirror augmentation applied at sample time, p=0.5) and optimizes four losses jointly — search-improved policy NLL, outcome-only value regression, opponent-prediction NLL, annealed aux losses, plus a decaying entropy bonus — AdamW at 3e-4 with cosine decay, batch 1024, with sequence NLL micro-batched under gradient accumulation inside each optimizer step.
- **Fresh weights broadcast to actors every ~2 minutes** via atomic export and hot-reload; actors re-read the league roster from disk every game, tolerating races with the learner's atomic rewrites; a transient failure costs one game, never an actor.
- **Resignation** ends decided games early (three consecutive value readings below −0.97), with a 10% exemption fraction always played to completion so the value head stays calibrated on late-game states; a symmetric early-exit stops widening the search once a candidate's worst case clears +0.98.
- **Actor count derives from the cgroup CPU quota** — the only number a container actually enforces — rather than `cpu_count` or CPU affinity, both of which report the host's 128 cores on shared pods.

**Export is gated on parity.** `train/export.py` serializes weights to a custom binary format and refuses to package unless the pure-numpy forward pass reproduces PyTorch outputs to <1e-4. The forward pass ultimately exists in three implementations — PyTorch (training), numpy (deployment), Rust (embedded in the sim) — all fed by one export, all parity-tested against each other.

## 4. Deployment

### 4.1 Measure the container before designing for it

Before the deployment design was fixed, a diagnostic probe algo was uploaded to the platform to measure the environment from the inside: Python 3.10.12 on glibc 2.35, numpy import at 120 ms, torch import at 2.5 s (fatal at match time — hence the numpy inference path), 20 iterations of 256³ matmul in 9 ms, the algo executing from filesystem root, four visible cores with ranked possibly enforcing one. Two direct consequences: the training pod was pinned to Ubuntu 22.04 so its glibc matches the container exactly and the compiled `terminal_sim` module transfers unmodified, and the whole stack is designed for one core with extras treated as bonus.

### 4.2 The strategy ladder

The shipped algo is a ladder, each rung falling back to the next, selected by a single switch:

```
1. K×M anytime search   (terminal_sim + weights.bin + numpy forward pass)
2. CornerHammerBot      (complete scripted game plan — the shipped primary)
3. AntiRushBot          (adaptive rush counter behind a Schmitt-trigger detector)
4. FallbackBot          (a static plan that always submits a legal turn)
```

Rung isolation is deliberate: the scripted modules import nothing at module level and their imports are exception-wrapped in the driver, so no layer can be a source of import-time failure for the layers below it. BLAS thread pools are capped through five environment variables *before* numpy loads, keeping the process inside the container's limits.

### 4.3 The mirror: keeping a live simulator of a game you can only observe

The server never shows a player the opponent's command log — only observations — but the search needs a live simulator state. The driver reconstructs the opponent's commands every turn from action-frame spawn events and turn-frame structure diffs, replays both command logs into a fresh sim (the *mirror*), and cross-checks the mirror's structures against the server frame.

Reconstruction cannot recover the opponent's *submission order*, so combat tie-breaks can occasionally drift the mirror. When the exact-match check fails, the driver rebuilds a simulator **from the server frame itself** — and since the sim API deliberately has no state injection, the rebuild is composed entirely of legal simulated turns:

1. **Health**: on this ruleset every mobile unit breaches for exactly 1, so exact player health is reproduced by marching scouts across an *empty* board, alternating sides so waves never meet.
2. **Structures**: every structure and upgrade the frame shows is re-issued each catch-up turn until cumulative income makes it affordable; with no mobiles in play, nothing fights.
3. **Clock**: empty turns pad the game to the current turn number so the search's lookahead sees the correct income schedule.

The rebuild runs under a hard iteration budget and is accepted only if the structure set matches the server exactly and both players' health lands within 0.5 — an *exact* board, verified before use. Planning then runs through a thin view that overrides the sim's resource banks with the server's true values, so every plan the search scores is a plan the engine will execute at full size.

### 4.4 Planning a turn under the watchdog

The search runs under an **anytime budget**: it opens with a K×M floor sized to provably finish in time, then doubles K and M while budget remains, keeping the best completed round's answer. The first candidate scored is always the all-defense plan, so a legal submission exists milliseconds into the turn; a worker-thread watchdog stages the strongest scripted layer's turn on any deadline miss — a missed deadline is treated as a timing problem, never as evidence the position needs a weaker plan. Search widths and budgets (k12/m6 over a k10/m5 floor, 1.6 s soft budget under a 4.5 s watchdog) were retuned from ranked-server telemetry, not dev-machine timings; measured on-server search compute came to 0.06–0.42 s per turn.

Committed plans pass through a **two-pass staging step**: structures, removals, and upgrades are issued first, then mobile deployments are routed against the board *as it will stand after this turn's builds* — each wave's lane scored by summing prospective turret attackers along the actual computed path, rerouted to the safest lane that provably reaches the enemy's half, and withheld entirely (bank preserved) if no lane reaches.

### 4.5 The scripted stack is a strategy system, not a stub

The scripted layers were developed and measured on the same arena as the learned agent, and the primary is a complete game plan. `CornerHammerBot` ([deploy/corner_hammer_bot.py](deploy/corner_hammer_bot.py)) plays an upgraded corner-anchored wall line with layered corner defense and sealed deep-edge diagonals; a normally-closed center gate that opens for exactly one turn per banked attack wave; **launch-size learning** that times waves around the opponent's observed commit level; **breach-heat tracking** across both flanks that reinforces whichever side is taking damage, with emergency interceptor screens that respond immediately while breaches are live; and a wave composer that switches from scout floods to demolisher-led attacks with scouts following through the opening when the opponent's front line is turret-dense.

`AntiRushBot` ([deploy/fallback.py](deploy/fallback.py)) wraps a funnel-and-trap layout in an income-scaled **Schmitt-trigger rush detector** — entry on genuine floods, breach evidence, or a proven flooder's reloading bank; exit only after consecutive clean turns, so the state can't oscillate — plus a counterattack cycled through a one-turn sally gate, lane-scored by real pathing, with a projected-damage check that banks the wave rather than sending it into a defended funnel.

Every configuration that could reach the ranked server was measured on the arena first, and ladder telemetry drove the final selection: the shipped build runs CornerHammerBot for the whole game, with the complete search stack resident behind the `NET_PRIMARY` switch and the watchdog degrading any failure to the strongest scripted layer.

## 5. Validation and tooling

**The arena** ([scripts/arena.py](scripts/arena.py), [sparring/](sparring/)) runs real `engine.jar` matches against ten frozen archetype bots — scout rush, shielded push, demolisher line, interceptor wall, static maze, corner gun, turret-wall flood, alpha-strike banker, a never-attacks "punching bag" that isolates whether the challenger can close out a game, and the frozen starter. Panel bots are deterministic and non-adaptive *by rule* (no randomness, no cross-game adaptation), so one match per pairing is meaningful and **margins are comparable across builds** — a win shrinking from 30–0 to 30–27 is a regression the W/L column can't show. The panel validates itself: a determinism check plays a panel to a canonical digest of replay *content* — SHA-256 over states, units, and events, deliberately excluding every timing field, because two runs of the same deterministic pairing produce byte-different replay files.

**The torture bot** ([bots/torture/](bots/torture/)) is a corpus generator: a deterministic per-turn script that exercises every engine mechanic ladder replays rarely cover — trapped-spawn self-destructs, same-turn build-then-upgrade (the sequence that exposed the engine's ID-reassignment behavior), mass removals of damaged upgraded structures (the sequence that exposed per-structure refund quantization), exact-decay-boundary banking, mutual mid-board kills. Its matches against the real engine feed the fidelity corpus; an in-sim twin plays in the league so the sim faces the same gauntlet.

**The test suite** — 50 Python tests plus the Rust parity suite — leans on properties rather than fixtures: closed-form arithmetic checks on the security scoring, an overfit smoke test on the learner, involution tests on mirror augmentation, a regression pinning the prediction head to opponent-perspective features (perturbing the opponent's board must move the prediction loss and nothing else), a decoder test asserting no reachable sampling path can emit an illegal token, and end-to-end parity of the numpy and Rust forward passes against PyTorch to <1e-4.

**The replay pipeline** owns its format in one place: a single parser ([scripts/replay_utils.py](scripts/replay_utils.py)) with frame taxonomy and per-event field layouts verified against real engine output feeds the fidelity harness, the training corpus, and match analysis, so a format discovery only ever has to be fixed once.

## 6. Results

**Fidelity** — 99.87% of 3,451,316 real engine frames exact; 3,489 of 3,847 ranked replays exact end-to-end; every residual divergence characterized and bounded in the spec, with the dominant category traced to server-side compute-time penalties that no rules-only simulator can reconstruct.

**Speed** — ~5,500 turns/sec/core puts a 72-fork search decision at ~0.5 ms of simulator time, which is what makes both league self-play and in-match joint search feasible at all.

**Final build** — 10–0 on the deterministic sparring panel with margins from +13 to +41, ~1400–1500 on the ranked ladder, inside the platform's 50 MB unpacked limit, submitting a legal turn within milliseconds of every turn start.

**Code** — roughly 10K lines of load-bearing source in 13 days: 2,989 of Rust (engine + native forward pass) with a parity suite, 3,706 of training system plus 1,450 of tests, ~1,940 of deployment stack, plus the arena, scraper, probe, and diff tooling.

| Date (2026) | Milestone |
|---|---|
| Jul 14 | Competition config captured and fingerprinted |
| Jul 15 | Rust engine lands (one ~2,600-line commit); PyO3 bridge; sparring arena; replay scraper |
| Jul 16 | Full RL system per spec; Rust forward pass + parity gate; fidelity round 3 → 99.87% |
| Jul 17 | H100 league run underway; deploy budgets retuned from ranked telemetry |
| Jul 18 | Anti-rush layer; league bots distilled from ladder study; search widened on-server |
| Jul 19 | Frame-grounded mirror rebuild; final build selection and hardening; ship |

The through-line of the project is that one measurement standard connects every component. The scraped corpus that seeded behavior cloning is the same corpus that proves the simulator. The simulator that powers self-play is the same binary that drives the mirror on the ranked server. The search that trains the network is byte-for-byte the search that plans ranked turns. The arena that gated every scripted change is the arena that selected the final build, and the ladder-forensics loop that tuned the deployment stack is the one that distilled the meta into the league. Thirteen days doesn't allow building things twice — every artifact here earns its keep in at least two roles.

---

*Numbers in this report are reproducible from the repo: `scripts/batch_diff.sh` regenerates the fidelity table from the scraped corpus, `scripts/arena.py` replays the sparring panel, `pytest train/tests/` and `cargo test` cover the training system and parity gates, and the commit history carries the timeline.*
