# Terminal Velocity

**A machine-learning game agent for [Terminal](https://terminal.c1games.com/rules), built for Citadel's Terminal competition.**

Terminal Velocity is a complete competitive-AI system built in under two weeks: a Rust reimplementation of the Terminal engine that reproduces 99.87% of real engine frames exactly, a PyTorch policy/value network trained with roughly 72 hours of league self-play on an H100, a security-weighted game-theoretic search that runs the same code in training and on the ranked server, a dependency-free numpy inference path for the restricted competition container, and a ladder of scripted strategies beneath the learned agent that guarantees a legal, competitive turn under every condition.

---

## Headline numbers

| Component | Result |
|---|---|
| Simulator fidelity | 99.87% of 3,451,316 real engine frames reproduced exactly; 3,489 of 3,847 ladder replays replay end-to-end without divergence |
| Training run | ~72 hours of league self-play on a single H100, seeded by behavior cloning over a scraped corpus of ~4,000 ranked replays |
| numpy forward-pass parity | < 1e-4 vs PyTorch, checked on every weights export |
| Sparring-panel record (final build) | 10-0 with margins of +13 to +41 against ten deterministic archetype bots |
| Per-turn planning | anytime search under the engine's 5-second cap, watchdog-guarded, submittable within milliseconds |
| Deployment footprint | pure python + numpy + one prebuilt native module, inside the platform's 50 MB unpacked limit |

---

## The game

Terminal is a head-to-head tower-defense strategy game played on a 28×28 diamond grid. Each player owns the half of the diamond nearest them and, every turn, both players simultaneously commit a complete plan before any of it resolves.

A plan has two halves. With **structure points (SP)** you place and upgrade static defenses on your own half: **walls**, which shape enemy pathing; **turrets**, which fire on enemy units in range; and **supports**, which project shields onto friendly units passing nearby, scaling with board depth. Structures can be marked for removal, refunding a fraction of their cost, so rebuilding and re-shaping the maze is a core mechanic. With **mobile points (MP)** you launch units from your edge of the diamond: **scouts**, fast and cheap; **demolishers**, slower but outranging turrets; and **interceptors**, hard-hitting defensive units that operate in your own half. Both currencies accrue on a schedule that grows over the match, and held MP decays each turn, so banking a large attack carries a real cost.

After the simultaneous deploy phase, the turn resolves as a sequence of action frames. Mobile units path toward the opposite edge, recomputing routes as walls and turrets are destroyed around them; units and turrets select targets by a strict priority rule; supports grant shields to units passing through their range; units that reach the far edge **breach**, removing one of the defender's 30 health; and units that become boxed in self-destruct, dealing area damage if they have traveled far enough. A player who reaches zero health loses immediately. If both players survive 100 turns, the higher health wins, and exact ties are decided by total compute time used.

As an AI problem, Terminal combines three difficulties. The per-turn action space is combinatorial: any affordable multiset of builds, upgrades, removals, and unit deployments is legal, and plans are committed simultaneously with the opponent's. The physics are deterministic but deeply stateful, so evaluating a plan requires simulating it. And ranked play consists of sharply tuned archetypes (mass-scout floods, funnel turtles, demolisher grinders, corner snipes), each of which punishes a different structural weakness, so an agent must be robust to all of them at once.

---

## System architecture

```
                ┌──────────────────────────────────────────────┐
                │  GPU learner (PyTorch, H100)                 │
                │  replay buffer ← self-play trajectories      │
                │  L = policy + value + opponent-pred + aux    │
                └────────▲───────────────────────┬─────────────┘
        trajectories     │                       │ weights (~2 min cadence)
                ┌────────┴───────────────────────▼─────────────┐
                │  actor pool (CPU)                            │
                │  terminal_sim self-play, K×M search moves    │
                │  net calls batched via shared-mem server     │
                └────────▲─────────────────────────────────────┘
                         │ opponents sampled from a league:
                         │ current θ · PFSP snapshots · scripted bots · BC anchor
   scraped ladder replays┴─→ BC warm start · prediction targets · eval anchors

   ──────────────────────────  deployment  ──────────────────────────
   engine frames → mirror reconstruction → K×M anytime search → commands
                      │ (desync?) rebuild the mirror from the server frame
                      │ (rush detected?) AntiRushBot override
                      │ (any failure?) scripted game plan takes the turn
```

### 1. The simulator (`sim/`, Rust)

The foundation of the system is a from-scratch Rust reimplementation of the Terminal engine covering pathing, targeting, combat, economies, shields, upgrades, and self-destructs, exposed to Python via PyO3 as `terminal_sim`, with a standalone `tsim` CLI for replay diffing.

The mechanics spec (`sim/MECHANICS.md`) was verified mechanic-by-mechanic against real `engine.jar` output and continuously re-validated against the scraped ladder corpus with `scripts/batch_diff.sh`: **3,489 of 3,847 real ladder replays reproduce end-to-end without divergence, and 99.87% of all 3.45 million action frames match exactly.** The remaining divergences are characterized in the spec (deep-stack shield attribution and rare targeting near-ties, about 0.13% of mid-phase frames, non-compounding for self-play). The spec also records hypotheses that were tested against production data and rejected, so verified mechanics are protected from speculative changes.

The simulator is fast enough to be forked per candidate plan inside the per-turn search budget, which is what makes the search architecture viable. It also embeds a native forward pass (`sim/src/nn.rs`, parity-tested in `sim/tests/nn_parity.rs`) as the fastest rung of the inference ladder.

### 2. State and action representation (`train/features.py`, `train/tokens.py`)

The board is encoded as **18 feature planes over the 28×28 grid**: structure occupancy, health, upgrade state, and pending removals from the sim bridge, plus deploy-history planes tracking recent deployment counts and per-turn EMAs for both players. Alongside the planes are **14 normalized scalars**: health totals, both resource banks, the income schedule, turn number, and running breach and structure-damage flows. Training doubles every sample with x-mirror augmentation, using the game's left-right symmetry.

A turn's plan is a **token sequence**: up to 24 tokens of (action type, board location, count bucket). Types cover the three builds, upgrade, removal, and the three unit deployments; counts use Fibonacci-spaced buckets plus a spend-everything bucket; an END token closes the plan. A scratch structure (`PlanScratch`) maintains incremental legality masks as the sequence grows, covering affordability with a safety margin, placement rules, and provably-null deployment pruning, so the decoder can only emit plans the engine will accept.

### 3. The network (`train/model.py`)

`TerminalNet` is a compact architecture designed for CPU deployment: a 3×3 convolutional stem into a 64-channel, 6-block **norm-free residual torso** (no batch or layer norm, which keeps CPU inference deterministic and makes the numpy and Rust ports direct), with the scalar vector injected as a FiLM-style channel bias. Four heads share the torso:

- **value**: a tanh scalar, the only head that scores decisions;
- **aux**: three self-supervised targets (health delta, net-worth delta, resource delta at a 3-turn horizon) used only for representation shaping;
- **policy**: an autoregressive plan decoder, a GRU over token embeddings with spatial pointer attention over the 28×28 feature map for location selection;
- **predict**: a second, independently-parameterized decoder run over the opponent-perspective encoding, trained to predict the opponent's next plan.

The prediction head exists because Terminal is a simultaneous-move game: move selection needs a distribution over what the opponent is about to do, grounded in everything observed so far, in addition to an evaluation of the current position.

### 4. The search (`train/search.py`)

Move selection is a **K×M security-weighted joint evaluation**, and the same code runs in the training actors and on the ranked server:

1. Sample **K** own plans from the policy head, always including the greedy plan and the all-defense plan.
2. Sample **M** opponent plans from the prediction head, always including the opponent's literal previous plan (re-legalized) and the empty plan.
3. Fork the simulator for every (own, opponent) pair, play out the joint turn, and score every resulting position with the value head in a single batch.
4. Score each own plan as `λ · Σⱼ wⱼ vᵢⱼ + (1−λ) · minⱼ vᵢⱼ`, an expected-case / worst-case blend with temperature-flattened prediction weights, balancing exploitation of likely opponent plans against robustness to unlikely ones.

On the ranked server the search runs under an **anytime budget**: it opens with a small K×M floor sized to finish under the watchdog, then doubles K and M while time remains, keeping the best completed round's answer. The first candidate scored is always the all-defense plan, so a legal submission exists within milliseconds of the turn starting.

### 5. Training (`train/learner.py`, `train/actor.py`, `train/league.py`)

Training ran for roughly 72 hours on a single H100 in two phases.

**Phase one: behavior cloning.** The scraped ladder corpus (~4,000 ranked games, config-fingerprinted to this competition's exact ruleset) is converted into per-decision tensors, and the network is warm-started on the winners' moves with teacher-forced token-level cross-entropy while the prediction head trains on both players' moves. This phase is gated on held-out metrics (top-1 token accuracy on winner moves, decreasing opponent-prediction perplexity) before self-play begins, so reinforcement learning starts from a policy that already plays coherent Terminal.

**Phase two: league self-play.** A pool of CPU actors plays continuous `terminal_sim` self-play, choosing every move with the full K×M search. Opponents are sampled from a **league**: the current parameters, prioritized-fictitious-self-play snapshots of past versions (weighted toward opponents the current agent scores worst against), the frozen BC anchor, and seven deterministic scripted bots (`train/scripted.py`) spanning the ladder's archetypes, including `corner_hammer`, distilled from a replay study of top-rated play, and `line_grinder`, distilled from a study of ranked losses. The league counteracts the standard failure mode of pure self-play, in which strategies that stop appearing in the pool stop being defended against.

The learner consumes trajectories through a replay buffer and optimizes four losses jointly: policy NLL against **search-improved targets** (the K×M evaluation produces a sharper distribution than the raw policy, and the policy is trained toward it), value regression on game outcomes, opponent-prediction NLL, and the auxiliary representation losses. Fresh weights broadcast to the actors on a ~2 minute cadence. Actor-side sampling temperature anneals over the run; resignation ends decided games early, with an exemption fraction always played to completion to keep the value head calibrated on late-game states; and an **evaluation gauntlet** measures win-rate against the scripted anchors throughout the run.

Actors do not own GPU memory: all network calls from all actors are batched through a **shared-memory inference server** on the GPU, so dozens of CPU self-play processes share one model instance at full batch efficiency alongside the learner.

**Export.** `train/export.py` serializes the weights to a custom `weights.bin` format and gates the export on a numpy parity check: the pure-numpy forward pass must reproduce PyTorch outputs to under 1e-4 before a package ships.

### 6. Deployment (`deploy/`)

The competition container is restricted: effectively one core, a 5-second soft turn cap, a 50 MB unpacked size limit, numpy available but torch too slow to import at match time. Deployment is structured as an **inference ladder**, with each rung falling back to the next:

1. **Full search**: `terminal_sim` + `weights.bin` + `npforward.py`, a pure-numpy reimplementation of the exact forward pass (im2col convolutions, GRU cell, pointer attention).
2. **CornerHammerBot**: a complete scripted game plan.
3. **AntiRushBot**: a scripted rush counter with an adaptive detector.
4. **FallbackBot**: a minimal static plan that always submits a legal turn.

The engine gives each player only observations, never the opponent's command log, while the search needs a live simulator state. The driver therefore **reconstructs the opponent's commands** every turn from action-frame spawn events and turn-frame structure diffs, replays both command logs into a fresh sim (the *mirror*), and cross-checks the mirror's structures against the server frame. Because reconstruction cannot recover the opponent's submission order, combat tie-breaks can occasionally drift the mirror; when the exact-match check fails, the driver rebuilds a simulator **from the server frame itself**, reproducing exact structure positions and exact player health through a sequence of synthetic catch-up turns (the sim API has no state injection, so the rebuild is composed entirely of legal simulated turns and verified exact before use). The search plans through a thin view that overrides the sim's resource banks with the server's real values, so every generated plan is affordable, and a two-pass staging step routes attack waves onto the lowest-danger lane that reaches the enemy, pathing against the board as it will stand after the current turn's builds.

The search runs in a worker thread under a watchdog that stages a scripted turn on any miss, and BLAS thread pools are capped before numpy loads to fit the container's process limits.

### 7. The scripted stack (`deploy/corner_hammer_bot.py`, `deploy/fallback.py`)

The scripted layers are full strategies in their own right, developed and measured on the same arena as the learned agent.

`CornerHammerBot` is a complete game plan distilled from ladder study: an upgraded corner-anchored wall line with layered corner defense, sealed deep-edge diagonals, a normally-closed center gate that opens for exactly one turn per banked attack wave, launch-size learning that times waves around the opponent's observed commit level, breach-heat tracking that reinforces whichever flank is taking damage, interceptor screens that respond immediately while breaches are live, and a wave composer that switches from scout floods to demolisher-led attacks with scouts following through the opening when the opponent's front line is turret-dense.

`AntiRushBot` wraps a funnel-and-trap layout in an income-scaled, Schmitt-trigger rush detector (entry on genuine floods, breach evidence, or a proven flooder's reloading bank; exit only after consecutive clean turns) plus a counterattack cycled through a one-turn sally gate, lane-scored by real pathing, with a projected-damage check that banks the wave rather than sending it into a defended funnel. The module imports nothing at module level, so it cannot be a source of import-time failure.

The driver can run the net as primary with a sticky mid-game handover, or run the scripted plan for the whole match, via a single switch (`NET_PRIMARY`). Every configuration that could reach the ranked server was measured on the same arena first.

---

## Validation and tooling

**Deterministic sparring arena** (`scripts/arena.py`, `sparring/`). Ten frozen archetype bots: scout rush, shielded push, demolisher line, interceptor wall, static maze, corner gun, turret-wall flood, and others. Each is deterministic and non-adaptive by rule, so one match per pairing is meaningful and margins are comparable across builds; a determinism self-check validates the panel itself via timing-independent canonical digests of replay content. The harness runs real `engine.jar` matches, attributes replays, and flags crashes. Every change to the deployed bot was gated on this panel; the final build is 10-0 with margins from +13 to +41.

**Replay pipeline** (`scripts/scrape_replays.py`, `scripts/replay_utils.py`). A resumable, parallel scraper harvests ladder replays from the public API and keeps those whose embedded config matches this competition's exact ruleset, comparing unit stats and resource schedules field-by-field while ignoring cosmetic fields. A single shared parser owns the replay format, with frame taxonomy and event layouts verified against real engine output, and feeds the fidelity harness, the training corpus, and match analysis.

**Fidelity harness** (`sim/target/release/tsim diff`, `scripts/batch_diff.sh`). Frame-by-frame comparison of the Rust simulator against every scraped replay, reporting per-replay pass/fail and corpus-level exactness on every simulator change.

**Container probe** (`bots/probe/`). A diagnostic algo uploaded to the real platform to measure the deployment environment from the inside: python version, installed libraries, import timings, matmul throughput, core counts, and filesystem layout. The deployment design (numpy inference, thread caps, budget sizing) is based on numbers measured on the actual competition hardware.

---

## Repository guide

```
sim/                 Rust engine reimplementation
  src/               engine, pathing, state, config, replay, PyO3 bindings, native NN
  MECHANICS.md       verified mechanics spec and fidelity status
  tests/             NN forward-pass parity tests
train/               the full RL system
  ARCHITECTURE.md    implementation spec: tensors, losses, schedules, budgets
  model.py           TerminalNet (torso + value/aux/policy/predict heads)
  tokens.py          action tokenization + incremental legality (PlanScratch)
  features.py        board planes, scalars, deploy history
  search.py          K×M security-weighted anytime search (train == deploy)
  actor.py           self-play actors
  learner.py         GPU learner: replay buffer, four-loss optimization
  league.py          PFSP opponent pool and snapshot management
  scripted.py        seven deterministic league bots
  infer_server.py    shared-memory GPU inference server
  export.py          weights.bin export + numpy parity gate
  evaluate.py        scripted-anchor evaluation gauntlet
  replays.py         replay corpus ingestion to training tensors
  config.yaml        every hyperparameter, keyed to ARCHITECTURE.md sections
  tests/             50 tests over tokens, features, model, search, league, export
deploy/              the shipped algo
  algo_strategy.py   driver: mirror reconstruction, watchdog, staging, strategy ladder
  npforward.py       pure-numpy TerminalNet forward pass
  corner_hammer_bot.py  full scripted game plan
  fallback.py        AntiRushBot + FallbackBot
scripts/             arena.py, scrape_replays.py, replay_utils.py, batch_diff.sh,
                     run_match.* single-match runners, gen_nn_fixtures.py
sparring/            ten deterministic archetype opponents for the arena
bots/                probe/ (platform diagnostics), torture/ (mechanics gauntlet)
python-algo/         C1 starter kit (the gamelib interface the driver builds on)
replays/             local match replays + the scraped ladder corpus manifest
engine.jar           the official game engine, used by the arena and match runners
game-configs.json    the competition ruleset the whole stack is keyed to
```

---

## Running it

**A local match** (Java 10+ required):

```bash
java -jar engine.jar work python-algo/run.sh python-algo/run.sh
```

**The sparring panel** against any algo directory:

```bash
python scripts/arena.py                      # full panel
python scripts/arena.py --only scout_rush    # one opponent
python scripts/arena.py --check-determinism  # verify panel determinism
```

**Tests:**

```bash
python -m pytest train/tests/                # tokens, features/model, search, league, export
cargo test --manifest-path sim/Cargo.toml    # engine + nn parity
```

**Training** (GPU host): see `train/ARCHITECTURE.md` and `train/setup_runpod.sh`; `python -m train.run` drives behavior cloning, league self-play, evaluation, and packaging phases from `train/config.yaml`.

**Scraping replays:**

```bash
python scripts/scrape_replays.py scan <lo_id> <hi_id> [workers]
```

---

## Acknowledgments

Built on the [C1GamesStarterKit](https://github.com/correlation-one/C1GamesStarterKit): `python-algo/`, `engine.jar`, and the game itself are Correlation One's (see `License.md`). The simulator (`sim/`), training system (`train/`), deployment stack (`deploy/`), and tooling (`scripts/`, `sparring/`, `bots/`) were built by the team during the competition.
