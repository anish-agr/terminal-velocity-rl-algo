# Terminal Velocity — Engineering Report

*Citadel's Terminal competition, July 2026. A Rust game-engine reimplementation, a league-self-play RL system, and a scripted strategy stack, built in 13 days. This report covers what the README's numbers don't: why the engine was rebuilt from scratch, how 99.87% fidelity was actually measured, what the training league did, what broke on the way to the ranked ladder — and the honest answer to which part of the system won games.*

---

## The 90-second version

| | |
|---|---|
| **Simulator** | From-scratch Rust reimplementation of the Terminal engine: 2,989 lines, one runtime dependency, ~5,500 turns/sec/core. Reproduces **99.87% of 3,451,316 real engine frames exactly** across 3,847 scraped ranked replays; 3,489 of them replay end-to-end without a single divergent frame. |
| **Training** | Behavior cloning over ~310K positions from 3,887 config-matched ranked replays, then league self-play (PFSP snapshots + seven scripted archetypes) on an H100, with every move chosen by a security-weighted K×M search — the same search code that runs on the ranked server. |
| **Deployment** | A four-rung strategy ladder inside a hostile container: pure-numpy inference (torch's 2.5 s import is unaffordable), a reconstructed "mirror" simulator of the live opponent, an anytime search under a 5-second watchdog, and scripted strategies beneath it that always submit a legal turn. |
| **The honest result** | The neural net went **0–11 on the ranked ladder** and was benched 90 minutes before the deadline. The scripted `CornerHammerBot` — built *with* the pipeline's replay corpus, arena, and ladder forensics — played the final submission and finished **10–0 on the sparring panel with margins of +13 to +41**, holding ~1400–1500 on the ladder. |
| **The lesson** | Every component that touched real engine output before ranked play worked. The one component that couldn't be validated against the real engine until ranked play — the net — failed there. The model lost; the measurement infrastructure around it is what won. |

Everything below is traceable to the repo: the mechanics spec ([sim/MECHANICS.md](sim/MECHANICS.md)), the training spec ([train/ARCHITECTURE.md](train/ARCHITECTURE.md)), the commit history, and the scraped-corpus manifests in `replays/scraped/`.

---

## 1. The problem

Terminal is a simultaneous-commit tower-defense game on a 28×28 diamond. Each turn, both players secretly commit a full plan — structures built, upgraded, or marked for removal; mobile units launched — and the turn then resolves as ~30 deterministic action frames of pathing, targeting, shields, and breaches. The action space is a combinatorial multiset, the physics are deterministic but deeply stateful, and the ranked ladder is a zoo of sharply tuned archetypes (scout floods, demolisher grinders, funnel turtles, corner snipes), each punishing a different structural weakness.

Three properties drove every design decision:

1. **Evaluating a plan requires simulating it.** There is no useful static evaluation of a Terminal position; whether a 14-scout wave breaches depends on exact pathing through exact structures with exact shield arithmetic.
2. **Moves are simultaneous**, so planning needs a distribution over what the opponent is about to do, not just a value function.
3. **The server is restrictive**: effectively one core, a 5-second soft cap per turn (1 HP/sec penalty for overruns), a 50 MB unpacked limit, and no way to test a full build except by playing ranked games.

## 2. The simulator: rebuilding the engine so that search could exist

### 2.1 Why reimplement

The official engine is `engine.jar` — a black-box Java process that plays one match at a time over a socket protocol. Three consumers needed something it cannot provide:

- **The search** needs to fork a game state hundreds of times per decision. The chosen move policy evaluates every (own plan, opponent plan) pair — K×M ≈ 72 simulator forks per decision in training, 128 on the server — inside a per-turn budget. That requires an in-process, copyable state, not a subprocess.
- **Self-play training** needs throughput. A 50-turn game is ~100 decisions ≈ 7,000 simulations; the pilot target was 60–150 games/min on a 20-vCPU pod. No JVM subprocess fleet approaches that.
- **The replay pipeline** needs to re-execute scraped games to extract training tensors and verify its own parsing.

So the engine was reimplemented in Rust ([sim/src/](sim/src/)): pathing, targeting, combat, economies, shields, upgrades, self-destructs — 2,989 lines across ten files, exposed to Python via PyO3 as `terminal_sim`, with exactly one mandatory dependency (`serde_json`). The core primitive is `Game.fork()`, a deep copy for search branching. Single-core throughput on a laptop: **~5,500 turns/sec, ~480K action frames/sec** in release build. The same crate embeds a hand-written forward pass of the trained network ([sim/src/nn.rs](sim/src/nn.rs)) as the fastest rung of the deployment inference ladder, parity-tested against PyTorch to <1e-4.

A reimplemented engine is only useful if it is *right*, which turns fidelity into the project's first-class metric.

### 2.2 How fidelity was measured

The measuring instrument is `tsim diff` ([sim/src/replay.rs](sim/src/replay.rs)): for every turn of a real replay, it reconstructs the state from the replay's turn frame, re-applies the exact deploy commands (recovered from spawn events, with engine unit IDs), re-simulates the action phase, and diffs **every frame** — unit positions, health values, event streams. The per-turn *resync* design is deliberate: it isolates divergences per turn instead of letting one early mismatch invalidate the rest of the match. The end-of-turn restore step (removals, refunds, decay, income) is validated separately against the *next* turn frame.

The corpus grew in rounds. A resumable parallel scraper ([scripts/scrape_replays.py](scripts/scrape_replays.py)) harvested the public replay API and kept games whose embedded config matched this competition's exact ruleset — 19 gameplay fields per unit type plus 9 resource-schedule fields, compared field-by-field with icon fields ignored. Final corpus: **4,041 IDs scanned, 3,887 config-matched matches, 3,847 diffed against the sim.**

| Measurement round | Corpus | Frame-exact |
|---|---|---|
| Round 2, before the last two fixes | 443-replay deterministic subset | 99.13% |
| Round 2, after | same subset | **99.93%** |
| Round 3, full corpus | 3,847 replays / 3,451,316 frames | **99.87%**, 3,489 replays fully exact |

The jump from 99.13% to 99.93% — a ~13× reduction in residual error — came from **two one-line fixes** backed by ~95 lines of recorded evidence:

- **SP refunds are quantized per structure** to the nearest tenth, round-half-up, *before* summing. Root-caused on a single turn where the raw refund sum was 5.146667 and the engine paid 5.100000; then validated at scale: across 229 removal-only turns in 400 replays, per-structure rounding matches **229/229**, while raw summation matches only 184, floor 206, ceil 199.
- **Upgrading a structure assigns it a new engine ID**, and that ID becomes its new attack-order tie-break. The replay showed it plainly — a wall built as id 143 logs a second spawn event as id 158 on upgrade — and the sim was advancing its ID counter without writing the new ID back.

### 2.3 The discipline: a spec that records rejected hypotheses

The most unusual thing about [sim/MECHANICS.md](sim/MECHANICS.md) is that it documents the fixes that were *refuted*, so verified mechanics can't be broken by plausible-sounding corrections later. During the competition, external bug reports claimed several "critical" sim bugs. Each was tested against production replay data before any code changed. Among the rejected:

- **"Shields must be capped."** Production replays show a single unit receiving 8 separate shield grants; an interceptor reached 56.6 HP. Pure summation passes 4,094/4,114 frames. No capped-value unit exists anywhere in the corpus. Rejected.
- **"Frame order must be shields → movement → attacks"** (as the rules page implies). Dispositive event evidence: scouts spawned inside a support's range receive their shield grant tagged with their *post-move* tile. The implemented order (movement → shields → attacks) passes 7,500+ frames; the "fix" would have introduced the exact 1-frame lag it claimed to remove. Rejected.
- **"The engine quantizes MP after decay."** All 32 model variants (f32/f64 × mul/sub × five rounding modes × two display modes) were enumerated; none survives chain validation — one turn requires the bank to sit *above* what exact arithmetic gives while a later turn requires it *below*. The raw f32 chain was kept as best-fit. Rejected.
- A **strict shield-range boundary** experiment and an **f64-widened shield formula** were both tried and reverted — the second because the engine demonstrably runs the pure f32 chain (a production replay serializes a grant as literal `6.1000004`, which is exactly f32 of 4 + 0.3ₓ×7).

The remaining 0.13% of divergent frames is characterized rather than chased: a deep-stack shield-attribution quirk (~18 units stacked, two specific grants land on different unit IDs than the sim's), targeting near-ties (3 frames in 4,114 in the worst affected replay), and sub-0.1 resource micro-drift on long banking chains that is provably unresolvable from replay data alone.

### 2.4 The biggest divergence wasn't a bug

The worst-diverging replays (594, 256, 182 mismatched frames…) all share one signature: the first divergence is a 1–3 point player-HP drop at a turn boundary with **no breach, damage, or self-destruct event anywhere near it** — in one fully scanned replay, a player's HP falls twice with no opposing breach event in the entire file. The diagnosis: the competition's **compute-time penalty** (1 HP per second over the 5-second cap), a server-side artifact of the *opposing bot's wall-clock time* that is simply absent from replay JSON. A rules-only simulator cannot reconstruct missing input; the spec marks it "do not chase this further." Excluding those replays, true engine fidelity is materially *higher* than the 99.87% headline.

That analysis paid for itself twice: it stopped a hunt for a bug that didn't exist, and it flagged how expensive turn-time overruns are on the ranked server — which shaped the deployment watchdog.

One measurement-hygiene war story worth recording: a Cargo feature flag (`--features python` applies package-wide) once silently rebuilt the standalone `tsim` CLI under PyO3's extension-module ABI, making it exit 0 with no output. The resulting batch run reported a nonsensical "243/3103 PASS, 99.9967% frame-exact" — caught because the numbers were internally inconsistent (frame count had collapsed 16×), at the cost of one wasted 3,103-replay validation run. The fix ships in [train/setup_runpod.sh](train/setup_runpod.sh): the feature-free binary is rebuilt *before* any fidelity gate runs.

## 3. Training: behavior cloning into league self-play

### 3.1 Representation and network

The board is 18 feature planes over the 28×28 grid (12 from the sim bridge — occupancy, health, upgrades, pending removals — plus 6 deploy-history planes maintained in Python) with 14 normalized scalars. A turn's plan is a token sequence: up to 24 tokens of (action type, board location, Fibonacci-bucketed count), closed by an END token, with an incremental legality mask (`PlanScratch`) that makes illegal plans unrepresentable — affordability with a safety margin, placement rules, and provably-null deployment pruning. Masking is deliberately minimal: only *provably useless* actions are removed, so sacrifices, baits, and slow-plays stay in the policy's reach and the search — not a heuristic — judges them.

`TerminalNet` ([train/model.py](train/model.py)) is small on purpose: **795,654 parameters (~3.2 MB of f32)**. A 3×3 stem into six norm-free residual blocks at 64 channels — no batch or layer norm anywhere, which keeps CPU inference deterministic and made the numpy and Rust ports line-for-line transcriptions — with scalars injected as a FiLM-style channel bias. Four heads share the torso: a tanh **value** (the only head that scores decisions), an **aux** head predicting 3-turn resource/health/net-worth deltas (representation shaping only, never consulted at play time, so reward hacking through it is structurally impossible), an autoregressive **policy** decoder (GRU over token embeddings with pointer attention over the spatial map), and an independently parameterized **predict** decoder that models the *opponent's* next plan from the opponent-perspective encoding.

### 3.2 The search is the policy

Raw policy samples don't play the game; the search does. Every decision — in training actors and on the ranked server, same code — runs a **security-weighted K×M joint evaluation**: sample K own plans (always including the greedy and the all-defense plan), M opponent plans from the prediction head (always including the opponent's literal previous plan re-legalized, and the empty plan), fork the simulator for all K×M pairs, score the resulting positions with the value head in one batch, and pick

> score(own plan *i*) = λ · Σⱼ wⱼ·v(i,j) + (1−λ) · minⱼ v(i,j)

an expected-case/worst-case blend. λ = 0.75 was not guessed: it was selected by self-play duels over {0.65…0.95}, high enough that the worst-case term can't collapse the policy into pure defense, low enough to respect rush lines.

### 3.3 What the league actually did

Pure self-play has a known failure mode: strategies that stop appearing in the pool stop being defended against. The league ([train/league.py](train/league.py)) is the counter. Opponents are sampled 35% current parameters / 40% snapshots / 15% scripted bots / 10% frozen BC anchor. Snapshots are drawn by **prioritized fictitious self-play**: sampling weight f(w) = w·(1−w) + ε over the rolling 50-game win rate w against each snapshot — mass concentrates on opponents the current agent goes ~50/50 with, the ε keeps solved opponents from vanishing entirely, and snapshots beaten >90% for two continuous hours are evicted.

The seven scripted league bots ([train/scripted.py](train/scripted.py)) were the anti-forgetting anchors, and two were *distilled from ladder forensics mid-competition*:

- `corner_hammer` — from a replay study of the visible #1 seed's 19 games: full turret line with corner anchors, MP banked to 0.875× the income/decay equilibrium, whole-bank scout waves aimed at whichever enemy corner scores weaker each turn.
- `line_grinder` — from the three ranked opponents that beat the deployed net with the identical plan: a solid 27–35-turret line with no walls at all, cracked only by banked 9–16-demolisher waves that outrange upgraded turrets. It was added to the league specifically because the net wasn't learning that answer (§5.4).

Training ran in two phases on a single H100. **Phase one, behavior cloning**: ~310K positions from the 3,887 config-matched replays — policy targets from winners' moves only, an opening-fingerprint cap so no single meta opening exceeds 25% of the dataset, x-mirror augmentation applied at sample time, and the prediction head trained on *all* players' moves as a population model. **Phase two, league self-play**: CPU actors playing full-search games, trajectories flowing through a 500K-position FIFO buffer into a four-loss objective (search-improved policy targets, outcome-only value regression, opponent-prediction NLL, annealed aux losses), fresh weights broadcast to actors every ~2 minutes. Actors own no GPU memory — every network call from every actor is batched through a shared-memory inference server with a 3 ms batching window, which is also why a "run actors locally, GPU in the cloud" split was rejected on paper: internet RTT is 10–30× that window.

**Export** gates on parity: `weights.bin` ships only if the pure-numpy forward pass reproduces PyTorch to <1e-4. The forward pass ultimately existed in three implementations — PyTorch, numpy, Rust — all parity-tested, all fed by the same export.

## 4. Deployment: planning inside someone else's container

A diagnostic probe algo was uploaded to the platform *first*, and the deployment design is built on its measurements rather than assumptions: Python 3.10.12, glibc 2.35 (matched exactly by the training pod's Ubuntu 22.04 so the compiled `.so` transfers unmodified), numpy import 120 ms, torch import 2.5 s (fatal at match time), 4 visible cores but designed-for-1, and the algo executing from filesystem root.

The shipped algo is a ladder, each rung falling back to the next:

```
1. K×M anytime search   (terminal_sim + weights.bin + numpy forward pass)
2. CornerHammerBot      (complete scripted game plan — the final primary)
3. AntiRushBot          (adaptive rush counter behind a Schmitt-trigger detector)
4. FallbackBot          (a static plan that always submits a legal turn)
```

Two mechanisms are worth explaining because they solve problems the platform creates:

**The mirror.** The server never shows you the opponent's command log — only observations — but the search needs a live simulator state. The driver reconstructs the opponent's commands each turn from spawn events and structure diffs, replays both logs into a fresh sim, and cross-checks the result against the server frame. Reconstruction cannot recover the opponent's *submission order*, so combat tie-breaks can drift the mirror out of sync. The cure (after a failed detour — §5.3) is **frame-grounding**: when the exact-match check fails, the driver rebuilds a simulator *from the server frame itself*. The sim API has no state-injection, so the rebuild is composed entirely of legal simulated turns: exact player health is reproduced by marching scouts across an empty board (every unit breaches for exactly 1 on this config), structures are re-issued each catch-up turn until income makes them affordable, and empty turns pad the clock so the income schedule aligns. The result is accepted only if the structure set matches the server exactly and both healths land within 0.5.

**The watchdog.** The search opens with a floor-sized K×M round that provably finishes in time, then doubles K and M while budget remains, keeping the best completed round. The first candidate scored is always the all-defense plan, so a legal submission exists milliseconds into the turn; a worker-thread watchdog stages the strongest *scripted* layer's turn on any deadline miss — a missed deadline is a timing problem, not evidence the position needs a weaker plan.

## 5. What failed

This project's failures are better documented than most projects' successes, because the repo's rule was that diagnoses live next to fixes. In chronological order:

### 5.1 Infrastructure fires on the training pod (July 16–17)

- **The BC corpus silently vanished from the replay buffer.** Bootstrap seeded 7,500 scripted games (~750K positions) *after* corpus ingestion — into a 500K FIFO. Every behavior-cloning step ran with `loss_policy = 0.0` and nobody had a reason to look. Fixed by reordering and streaming ingestion; the symptom is now called out in the code as the thing to check first.
- **A redundant config filter rejected every replay.** A second field-by-field config comparison at ingestion (subtly different from the scraper's canonical fingerprint) config-skipped the entire corpus — a zero-position warm start. The fix deleted the duplicate: one fingerprint function owns config equivalence, everywhere.
- **libgomp killed the pilot.** RunPod containers report all 128 host cores to `os.cpu_count()` *and* `sched_getaffinity`, so the system launched 126 actors, each of whose BLAS spawned a 128-thread pool. Thread caps were added — then had to be *hoisted to module import* when the bootstrap learner (which missed the phase-scoped cap) died at its first training step. The real actor count finally came from the cgroup CPU quota, the only number the container actually enforces.
- **The prediction head was trained on the wrong perspective** — own-view features rather than opponent-view — a silent train/inference skew. Fixed and pinned with a regression test asserting that perturbing the opponent's board moves the prediction loss and nothing else.

### 5.2 Ranked play was silently FallbackBot (July 17)

The first ranked games looked like the net was playing badly. It wasn't playing at all. Two independent causes produced the same symptom — the bottom-rung fallback playing entire matches with no visible error:

1. Uncapped OpenBLAS tried to spawn one thread per host core inside the container's process limits; the first matmul raised, the search worker died every turn — and the same build looked fine in the playground, which has more headroom.
2. The k16/m8 opening search round, sized on a dev machine, overran the watchdog every turn on the shared ranked box (~2–4× slower per thread).

The immediate fix capped five BLAS env vars before any numpy import and shrank the search. The satisfying epilogue: once thread caps were in place, ranked replays showed actual search compute of **0.06–0.42 s per turn** against a 5-second budget — the "overrun" had been the thread explosion all along, and the search was re-widened to k12/m6 with a raised anytime floor.

### 5.3 The mirror desync, and the wrong fix before the right one (July 18–19)

A ladder audit found the mirror desyncing between turns 9 and 26 in most ranked games — one order-dependent combat tie-break and the exact-match test fails permanently, dropping the driver to a scripted layer for the rest of the match. The audit's rating trace made the cost concrete: **1500 → 1295**, with FallbackBot getting farmed by scout bankers.

The first fix was to *tolerate* small drift (≤10 mismatched cells) and keep the net playing. It shipped, and it was wrong: **"the net paths against walls that are not there."** A plan that threads a wave through a gap that doesn't exist is worse than a scripted plan that sees the board correctly. Tolerance was deleted and replaced with the frame-grounded rebuild (§4), which restores an *exact* board at the cost of approximate structure health — and the interim triage (degrade to AntiRushBot, the layer that had actually carried the rating, rather than FallbackBot) is its own small lesson about knowing which fallback is load-bearing.

### 5.4 The net lost on the ladder, and was benched (July 19)

With the mirror fixed and the infrastructure honest, the net finally played real ranked games under its own name. It lost them in instructive ways, each diagnosed from ladder replays within hours:

- **Undersized waves**: the search planned against the mirror's approximated resource bank; encode-time clamping then shrank its waves to 3–4 units. One replay: 110 spawns at a single cell, 5 total damage dealt. Fix: a thin view that overrides sim banks with the server's real values, so every plan the search scores is a plan the engine will accept at full size.
- **Self-trapping offense**: 123 scouts dead at y=12 — in its own half — for zero damage. Fix: a driver-level router that re-lanes offense onto the lowest-danger path that provably reaches the enemy, pathing against the board as it will stand *after* this turn's builds.
- **Structural monoculture**: ~30 turrets and almost no walls, no funnel geometry, ~0 breach damage across four consecutive audited games.

A sticky mid-match handover was added — if the net is clearly failing by turn 14, CornerHammer takes over — and it wasn't enough: the games were already poisoned by the opening (SP spent, no wall geometry to inherit). The final ledger, preserved in the commit history: **0–11 on the ladder, ~0 breach dealt**, including a 30:30 tie-break loss and a 4:30 blowout. Ninety minutes of iteration later, the decisive commit set `NET_PRIMARY = False` for good: *CornerHammer plays the whole game; it is the only plan that provably wins on the real engine.*

Why did a net that went 9–0 on the local panel go 0–11 on the ladder? The recorded evidence supports three overlapping answers rather than one clean villain. First, **a train/deploy evaluation gap**: the full net stack (compiled sim + weights + numpy path) could not play real-engine matches on the machines that had `engine.jar` — its first genuine full-stack games *were* ranked games, so weeks of compounding errors were discovered in hours, live. Second, **distribution shift the league only partially closed**: the ladder's dominant counter-shape (solid turret lines broken only by demolisher waves) was distilled into `line_grinder` and added to the league late; the net's replays show it deployed zero demolishers in 9 of the 10 games it lost to that archetype. Third, **known training gaps shipped as-is** under the deadline: the actor temperature anneal was specified but never wired (self-play always sampled at τ=1.0), the promotion gauntlet existed but never ran in-loop, and a deploy-time sampling temperature of 0.9 was found *sharpening into a degenerate mode* — hundreds of demolishers per game, all at one cell — and had to be flattened to 1.1. None of these is exotic; all of them are the tax of compressing an AlphaStar-shaped system into thirteen days.

### 5.5 Dead ends, tabulated

| Tried | Outcome |
|---|---|
| Tolerating mirror drift to keep the net playing | Reverted — the net paths against phantom walls; exact rebuild instead |
| Shield cap, frame reorder, MP quantization, f64 shield math (external bug reports) | All refuted against production replays before touching code; spec records the evidence |
| "Only open the gate when the enemy bank is low" (scripted bot) | Reverted — vs any banking opponent the gate never opens and the bot goes fully passive |
| Local CPU actors + cloud GPU split | Rejected on paper: 3 ms batching window vs 30–100 ms RTT is a ~50× throughput collapse |
| Mid-match handover as a safety net for the net | Kept, but insufficient — turn-14 rescue can't fix a poisoned opening |
| Chasing the largest fidelity divergences | Correctly abandoned — they were the server's compute-time HP penalty, missing input, not sim error |

## 6. Results

**Fidelity** — 99.87% of 3,451,316 real engine frames exact; 3,489/3,847 ranked replays exact end-to-end; residuals characterized and bounded in the spec. **Speed** — ~5,500 turns/sec/core, which put 72-fork search decisions at ~0.5 ms of sim time and made both training and in-match search feasible.

**Final build** — 10–0 on the deterministic ten-bot sparring panel with margins +13 to +41 (the panel is deterministic and non-adaptive *by rule*, so one game per pairing is meaningful and margins are comparable across builds; the panel's own determinism is verified via timing-independent replay digests). Ladder rating ~1400–1500 as a standalone strategy, versus the 1295 trough during the desync era. The net's numpy inference path, mirror reconstruction, and watchdog all shipped and ran; the plan they served came from the scripted layer.

**Code** — roughly 10K lines of load-bearing source in six days of building and seven of firefighting: 2,989 Rust (sim + native NN) with a 149-line parity suite, 3,706 Python of training system plus 1,450 lines of tests (50 tests: token legality, mirror involution, search arithmetic closed-form, an overfit smoke test, a train/infer-skew regression), ~1,940 of deployment stack, plus the arena, scraper, and probe tooling.

| Date (2026) | Milestone |
|---|---|
| Jul 14 | Competition config captured and fingerprinted |
| Jul 15 | Rust engine lands (one ~2,600-line commit); PyO3 bridge; sparring arena; replay scraper |
| Jul 16 | Full RL system per spec; Rust forward pass + parity gate; fidelity round 3 (99.87%); first pod fires |
| Jul 17 | Thread-cap and cgroup fixes; H100 run underway; ranked FallbackBot incident diagnosed |
| Jul 18 | Search retuned from ranked telemetry; anti-rush layer; league bots distilled from ladder audits |
| Jul 19 | Frame-grounded mirror rebuild (03:40); net's ranked forensics (08:00–12:30); net benched (13:28); final hardening (13:51); ship |

## 7. What I'd do differently

**Close the reality gap first.** The single structural mistake was that the learned agent's first full-stack games on the real engine were ranked games. The arena gated every scripted change on real `engine.jar` matches from day two; the net deserved the same harness from the day it had weights, even at the cost of a day of packaging work.

**Run the gauntlet in-loop.** The promotion rule (win rates vs anchors, zero crashes, zero timeouts) existed as tested code and never ran during training. A strength trend line during the 72-hour run would have surfaced the demolisher blind spot while there was still GPU time to spend on it.

**Treat sampling temperatures as deploy config, not training config.** Two of the net's worst behaviors — the one-cell demolisher monoculture and the never-annealed self-play temperature — were single scalar values. Cheap to sweep, expensive to discover on a ladder.

**Keep the forensic comments.** The pre-ship cleanup pass that sanitized ladder IDs and ratings out of source comments made the code tidier and the history poorer; this report had to recover its best material from `git show`. The diagnosis-next-to-fix convention was the right one.

The system's final irony is worth stating plainly, because it is the honest summary of the whole project: the machine-learning agent lost, and the machine-learning *infrastructure* won. The replay corpus that fed behavior cloning also proved the simulator. The simulator that powered self-play also drove the mirror on the ranked server. The arena built to promote checkpoints instead promoted the scripted bot that shipped. And the ladder-forensics loop that was supposed to tune the net distilled the opponents' play into the strategy that actually held the rating. Every hour spent on measurement paid out; the hours the deadline stole were the ones the model needed.

---

*Numbers in this report are reproducible from the repo: `scripts/batch_diff.sh` regenerates the fidelity table from the scraped corpus, `scripts/arena.py` replays the sparring panel, `pytest train/tests/` and `cargo test` cover the training system and parity gates, and the commit history carries the deployment timeline.*
