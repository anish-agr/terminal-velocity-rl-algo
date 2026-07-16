# Training & Deployment Architecture — Implementation Spec

This is the complete blueprint for the RL system. It is written to be implemented as-is:
every tensor shape, loss term, schedule, and budget is specified. Where a judgment call
remains, the DEFAULT is stated and marked (tunable). Read sim/MECHANICS.md first — the
simulator's contract is the foundation, and its two "Open fixes" are implementation tasks
#1 and #2 below, BEFORE any training code.

Deployment environment (PROBE-VERIFIED on the platform, 2026-07-15): Python 3.10.12,
linux x86_64, glibc 2.35, **numpy 2.2.6 + scipy 1.15.3 + torch 2.11 ARE installed**
(numpy import 120 ms, 20x 256^3 matmuls in 9 ms; torch import 2.5 s — usable but heavy),
no gcc, ctypes/.so loading WORKS, the algo executes at filesystem root `//` (always
resolve files via os.path.dirname(os.path.abspath(__file__))), test container showed
4 cores and no cgroup CPU/memory caps (ranked may still enforce 1 CPU / 3.5 GB — design
for 1 core, treat extras as bonus). Constraints: 5 s/turn soft (1 HP/s over), **50 MB
UNPACKED folder**, compiled binaries allowed. Inference ladder: (1) our terminal_sim .so
(sim + native forward pass; wheel target abi3-py38, built on Ubuntu 22.04 = glibc match),
(2) numpy forward pass (viable — numpy is present in the container), (3) pure-python
heuristic bot. Turn-100 ties break on LOWER total compute time — exit search early when
winning.

---

## 0. Implementation order (each step has a hard verification gate)

1. SIM STATUS (2026-07-16): all planned fixes are APPLIED and validated at scale — SD
   AoE vs 0-health units, game-over phase termination, force-afford replay
   reconstruction, health comparison quantization. The simulator is production-validated
   against 3,100+ scraped server replays (see replays/scraped/diff_results.tsv for the
   definitive numbers; end-of-turn "restore" states — the RL decision points — pass at
   a higher rate than mid-phase frames). Remaining residuals are characterized in
   MECHANICS §Fix status (deep-stack shield/damage attribution + rare targeting
   near-ties, ~0.5% of mid-phase frames, non-compounding for self-play because the sim
   is its own engine on both sides). Read MECHANICS §External bug-report audit before
   changing ANYTHING in the sim — several plausible-sounding "fixes" (shield cap, phase
   reorder, MP quantization, strict shield range, f64 shield amounts) were tested and
   REFUTED against production data; the audit trail is there so nobody re-breaks it.
2. Sim: `Game.legal_mask` + incremental plan-scratch support in py.rs (spec §3.4). Gate:
   mask agrees with `apply_commands` acceptance on 10K random commands.
3. Replay ingestion (§7). Gate: parses every scraped replay; per-position tensors
   round-trip; winner labels match endStats.
4. Net (§4) + losses (§6) on BC data only. Gate: top-1 token accuracy on held-out winner
   moves > 30%; opponent-prediction perplexity decreasing.
5. Actor loop (§5) at K=4, M=2, single process. Gate: completes 100 self-play games with
   zero exceptions; mean game length 20–60 turns.
6. Parallel actors + GPU inference server (§5.5). Gate: >50 games/min on the pod.
7. League + eval gauntlet (§6.4, §8). Gate: pilot run shows monotone gauntlet win-rate
   over 4 hours.
8. Export + Rust forward pass (§9). Gate: Rust forward output matches PyTorch to 1e-4;
   full turn under 1.0 s on ONE core with K=16, M=8.
9. Docker-equivalent rehearsal (§9.4). Gate: 50 games, zero timeouts/crashes.

---

## 1. System overview

```
                    ┌────────────────────────────────────────────┐
                    │ GPU learner (PyTorch, 1×H100)               │
                    │  replay buffer ← trajectories               │
                    │  L = L_policy + L_value + L_predict + aux   │
                    └───────▲──────────────────────┬──────────────┘
             trajectories   │                      │ weights (every ~2 min)
                    ┌───────┴──────────────────────▼──────────────┐
                    │ N actor processes (CPU, ~2 per vCPU)        │
                    │  each: terminal_sim.Game self-play          │
                    │  move choice = K plans × M opponent plans   │
                    │  evaluated through sim forks + value net    │
                    │  (NN calls batched via shared-mem inference │
                    │   server on the GPU)                        │
                    └───────▲─────────────────────────────────────┘
                            │ opponents sampled from LEAGUE:
                            │  current θ / PFSP snapshots / scripted bots / BC anchor
   scraped ladder replays ──┴─→ BC warm start + prediction-head data + eval anchors
```

---

## 2. State representation

All tensors from player-perspective (board flipped for player 1 so "own" side is always
y < 14; the sim bridge already does this in `board_planes`).

### 2.1 Board planes — f32 [C=18, 28, 28]

| idx | content | normalization |
|---|---|---|
| 0–2 | own wall / support / turret health | ÷ upgraded max (120/30/75) |
| 3 | own upgraded mask | 0/1 |
| 4 | own pending-removal mask | 0/1 |
| 5–7 | enemy wall / support / turret health | same |
| 8 | enemy upgraded mask | 0/1 |
| 9 | enemy pending-removal mask | 0/1 |
| 10 | in-arena mask | 0/1 |
| 11 | own-half mask | 0/1 |
| 12–14 | enemy deploys LAST turn: scout/demolisher/interceptor counts at spawn tiles | count/10, clamp 1 |
| 15–17 | enemy deploys EMA over match (per kind), decay 0.7/turn | clamp 1 |

Planes 0–11 come from `Game.board_planes` (12 planes, already implemented); 12–17 are
maintained by the actor/bot from observed spawn events (bridge addition: expose last-turn
enemy deploy list, or track python-side from the commands we feed in self-play).

### 2.2 Scalar vector — f32 [S=14]

[own hp/30, own sp/40, own mp/15, enemy hp/30, enemy sp/40, enemy mp/15, turn/100,
mp_income/10, own next-turn-mp-if-banked/15, enemy same/15, own breach dealt last turn/5,
taken last turn/5, own structure-damage dealt last turn/50, taken/50].

### 2.3 Symmetry augmentation

Every training sample is duplicated with x-mirror (x → 27−x): board planes flipped on the
x axis, plan location tokens remapped, scalars unchanged. Exact symmetry of the rules
(edges/targeting mirror cleanly). Free 2× data.

---

## 3. Action space: a turn as a token sequence

### 3.1 Token vocabulary

A plan is a sequence of ≤ T_max = 24 tokens, each token = (type, loc, count):

- type ∈ {BUILD_WALL, BUILD_SUPPORT, BUILD_TURRET, UPGRADE, REMOVE, DEP_SCOUT,
  DEP_DEMOLISHER, DEP_INTERCEPTOR, END} (9 types)
- loc ∈ 28×28 grid (pointer over 784 cells; masked to legal cells per type)
- count: only for DEP_* tokens, bucket ∈ {1, 2, 3, 5, 8, 13, 21, ALL} (8 buckets;
  ALL = spend-remaining at this tile; repeated tokens at the same tile are allowed, so any
  integer is reachable)

Execution order = token order (matters: builds before upgrades of the same tile; the sim
consumes them in submitted order, deploys are split out automatically by the bridge).

### 3.2 Legality masking (mandatory, exact)

Maintained incrementally over a plan-scratch copy of the state:
- BUILD_k: tile in own half ∧ in-arena ∧ no structure ∧ scratch-SP ≥ cost.
- UPGRADE: own structure at tile ∧ ¬upgraded ∧ scratch-SP ≥ upgrade cost.
- REMOVE: own structure at tile (marking pending twice is legal-but-null → mask repeats).
- DEP_k: tile on own spawn edges ∧ no structure ∧ scratch-MP ≥ cost·bucket_min.
- END: always legal.
- **Affordability margin** (MECHANICS §Open fixes 3): plan-scratch resources start from
  the SERVER-provided values each turn; a plan must not depend on the last 0.1 of MP/SP.
  At deployment the marginal unit is still ATTEMPTED (the engine silently skips
  unaffordable commands — optimistic attempts are free), but the search scores the plan
  as if that unit may not spawn.

### 3.3 Provably-null masks ONLY (no judgment masks)

Per the "don't over-mask" concern — the search itself is the arbiter of "dumb" moves
(sacrifices, baits, slow-plays all stay available). We mask only actions with provably
zero game effect:
- DEP_k at a tile whose pocket (a) cannot reach the enemy half AND (b) has max path length
  < 5 from the tile (self-destructs with no damage — pure MP burn). Computed with one
  cached pathfind per edge tile per layout.
- UPGRADE/REMOVE on nothing (covered by legality).
Nothing else. (tunable: OFF switch for ablation.)

### 3.4 Bridge additions needed (implementation task #2)

`Game.legal_mask(player, scratch) → bytes` per token type, and
`Game.plan_scratch_apply(token)` maintaining scratch SP/MP/occupancy — OR implement the
scratch purely python-side from `structures()` + raw stats (acceptable; ~100 µs/turn).

---

## 4. Network

One shared torso; four heads. ~2.6 M parameters, f32 (~10.5 MB exported — fits 50 MB with
huge margin).

### 4.1 Torso

- Input: board [18,28,28]; scalars [14] → MLP(14→64→64, ReLU) → broadcast-add as bias to
  every spatial cell after the stem (FiLM-lite: added to channel dims 0–63).
- Stem: 3×3 conv, 18→64, ReLU.
- 6 residual blocks: (3×3 conv 64→64, ReLU, 3×3 conv 64→64) + skip, ReLU. NO batch/layer
  norm (keeps the Rust port trivial and CPU inference deterministic; net is shallow enough
  to train without norm at lr 3e-4 with grad clip).
- Output feature map F ∈ [64, 28, 28]; pooled vector g = concat(GAP(F), GMP(F)) ∈ [128].

### 4.2 Value head (the ONLY head used for decision scoring)

g → FC(128→256, ReLU) → FC(256→1) → tanh. Trained exclusively on final result z ∈ {−1,+1}
(tie = 0). NO shaped rewards in this head — reward hacking is structurally impossible.

### 4.3 Auxiliary dense head (cold-start helper; answer to "value bootstrapping")

g → FC(128→128, ReLU) → FC(128→3): predicts, for t+3 (3 turns ahead, clipped at game end):
Δ(hp_own−hp_opp)/10, Δ(board SP net-worth own−opp)/50, Δ(resource total own−opp)/20.
Net-worth of a structure = invested SP × health%. Pure representation-shaping auxiliary —
its outputs are NEVER used in decisions, so it accelerates the cold start (dense signal
from the very first random games) with zero hacking risk. Weight annealed §6.3.

### 4.4 Policy decoder (autoregressive plan head)

- Committed-plan state: c_i ∈ [128], GRU cell; c_0 = FC(g). Each chosen token is embedded
  as e(token) = type_emb[16] ⊕ F[:, loc] (64) ⊕ count_emb[8→16] → FC(96→128) and fed to
  the GRU.
- Per step i: type logits = FC(c_i →9) + mask; loc pointer = softmax over
  ⟨W_q c_i, W_k F[:, x, y]⟩ (W_q: 128→64, W_k: 1×1 conv 64→64) + mask; count logits =
  FC([c_i ⊕ F[:,loc]] → 8) + mask.
- Token log-prob = log p(type) + log p(loc|type) + log p(count|type,loc) (loc/count terms
  zero for END/UPGRADE/REMOVE-style tokens where absent).

### 4.5 Opponent-prediction head

Identical decoder structure with its OWN GRU/pointer parameters, applied to the
OPPONENT-perspective torso output (flip the board and run the shared torso a second time —
1 extra torso pass per decision, acceptable). Predicts the opponent's next plan token
sequence. Trained on all observed opponent plans (self-play and scraped replays).

---

## 5. Decision procedure (identical in training actors and deployment)

### 5.1 Candidate generation

- Own plans: K sequences sampled from the policy decoder at temperature τ_act (training:
  1.0 → 0.7 linear over the run; deployment: 0.5), PLUS the greedy (argmax) plan, PLUS the
  "all-defense" plan (greedy with DEP_* masked), deduped → K_eff ≤ K+2. K default 12
  (training) / 16 (deployment, budget-adaptive §9.3).
- Opponent plans: M samples from the prediction head at τ = 1.0, PLUS the opponent's
  literal previous plan (re-legalized), PLUS the empty plan. M default 6 / 8.

### 5.2 Scoring — likelihood-weighted approximate security

For each (i, j) ∈ K_eff × M_eff: fork the sim, `play_turn(plan_i, opp_plan_j)`, evaluate
v_ij = value head on the resulting state (from our perspective). Batch all K·M value calls
into one NN batch. With prediction-head plan probabilities p_j (normalized over M_eff,
temperature-flattened: w_j ∝ p_j^0.5):

    score_i = λ · Σ_j w_j v_ij  +  (1−λ) · min_j v_ij        λ = 0.7 (tunable)

This is the "approximate security" from the CMU-13 postmortem: expected-case play weighted
toward the likely opponent, with a worst-case floor so we never open a one-shot loss.
Chosen action a* = argmax_i score_i.

### 5.3 Expert target (ExIt)

π*(i) = softmax(score_i / τ_tgt), τ_tgt = 0.25, support = the K_eff candidates. Stored with
the position for the policy loss (§6.1).

### 5.4 Early exit / compute tiebreaker

If max_i min_j v_ij > 0.98 → play a* immediately without widening (deployment: also skip
optional extra K widening). Self-play resign: if v(s) < −0.97 for 3 consecutive turns →
resign (label z accordingly); 10% of games exempt from resignation (value-blind-spot
insurance).

### 5.5 Actor/inference topology on the pod

- N_actor = 2 × vCPU processes, each running whole games sequentially.
- NN calls go through a shared-memory batching server on the GPU (collect up to 512
  requests or 3 ms). Three request types: torso+policy sample, torso+prediction sample,
  value-batch (K·M states). States serialized as the [18,28,28]+[14] tensors.
- Throughput estimate: per decision ≈ K·M = 72 sim forks (~0.5 ms total) + 1 value batch +
  ~20 decoder steps. A 50-turn game ≈ 100 decisions ≈ 7K sims + 100 GPU batches.
  Expect 60–150 games/min at 20 vCPUs; buffer fills fast.

---

## 6. Learning

### 6.1 Losses

Per position (state s, expert π*, sampled plans {a_i}, outcome z, opponent plan b):

- L_policy = − Σ_i π*(i) · log p_θ(a_i | s) (sequence NLL over tokens of each candidate).
- L_value = (v_θ(s) − z)².
- L_aux = ‖aux_θ(s) − Δ_3(s)‖² (targets from the trajectory).
- L_predict = − log p_φ(b | s_opp) (token NLL, teacher forcing).
- Entropy bonus on decoder type/loc distributions: −c_e · H, c_e = 1e-3 (prevents
  premature collapse; decays to 0 by mid-run).

L = L_policy + 1.0·L_value + c_aux·L_aux + 0.5·L_predict + entropy term.

### 6.2 Optimization

AdamW, lr 3e-4 cosine-decayed to 3e-5 over the planned run length, weight decay 1e-4,
batch 1024 positions (with mirror augmentation applied at sample time), grad-norm clip
1.0. Replay buffer: FIFO 500K positions, uniform sampling, min-fill 20K before training
steps begin. Learner:actor ratio ~4 gradient steps per 1K new positions (tunable; watch
for overfitting via eval).

### 6.3 Cold start & annealing (answer to "value bootstrapping")

- Hour 0: buffer seeded with (a) BC positions from scraped winner moves (§7), (b) 5–10K
  games of scripted-vs-scripted and random-policy-vs-scripted play (cheap, CPU-only, no
  GPU search needed — pure sim). Value + aux heads train immediately on real outcomes.
- c_aux = 0.5 for the first 25% of the run, linear → 0.1 by 50%, then fixed. (The main
  value target is z from step one; the aux head only shapes features. No shaped rewards
  ever enter the decision path — this is the reward-hack-proof version of the idea.)

### 6.4 League / opponent sampling (answer to "fictitious play") — INCLUDED

Each self-play game samples the opponent controller:
- 35% current θ (mirror self-play),
- 40% snapshot pool (checkpoint every 30 min): PFSP weighting f(w) = w·(1−w) over the
  snapshot's rolling win-rate w vs current (prioritizes near-peers; keeps beating-the-old
  pressure without wasting compute on solved opponents),
- 15% scripted archetypes (rush, funnel, demolisher-line, interceptor-turtle, torture) —
  the "don't forget the basics" guarantee,
- 10% frozen BC-anchor policy (the field's meta, embodied).
Snapshot pool capped at 20 (evict lowest-information: w > 0.9 for 2 h).

---

## 7. Ladder replays: ingestion (verdict: useful, but NOT a self-play substitute)

Honest assessment: with O(10–100) replays of mid-strength bots, imitation cannot replace
self-play (too little data, wrong ceiling). Their real value, in order:
1. **Server-fidelity corpus**: run `tsim diff` on every scraped replay (already caught two
   engine subtleties local play never produced).
2. **Opponent-prediction pretraining**: train the prediction head on ALL moves by ALL
   players (we're modeling the population, not imitating skill). This is the
   research-PDF's "underexploited edge", fitted to the ACTUAL current field.
3. **BC warm start of the policy** (~1 GPU-hour): winners' moves only (z=+1 side of each
   replay), so we never imitate losing play. De-biasing: anonymized names → fingerprint
   each side by its turn-0–3 build-sequence hash; cap any fingerprint at 25% of the BC
   dataset. Stop BC as soon as self-play starts — it is an initialization, not a target.
4. **Eval anchors + opening statistics**: the BC-anchor bot joins the league (§6.4) and
   the gauntlet (§8); turn-0–5 build histograms parameterize scripted exploiter variants.

Scraping: `GET https://terminal.c1games.com/api/game/replayexpanded/<match_id>` — public,
no auth (verified: pulled 1.2 MB directly). Match IDs are sequential across ALL of
Terminal, so harvesting needs NO manual labeling: scan ID ranges anchored at 2-3 known
IDs from this competition (have: 15330187), download each, keep replays whose embedded
config matches ours on gameplay fields (unit stats + resources; icon fields differ
harmlessly). Rate-limit ~1 req/s, resumable. Player names are anonymized ("algo1/algo2")
— de-biasing uses the opening-fingerprint scheme above; no team labels required.
Target: every same-config replay in the scanned ranges (likely 50-500).

---

## 8. Evaluation & promotion

- Gauntlet: 200 games (100 as P1, 100 as P2) vs {each scripted bot, BC anchor, previous
  best checkpoint}. Metrics per opponent: win-rate, loss-rate, timeout count, mean margin.
- Promote to `best` iff: ≥55% vs previous best AND ≥85% vs EVERY scripted bot AND zero
  crashes/timeouts. (Min-based, not mean — single-elim logic.)
- Ladder testing protocol: at most 1–2 uploads/day of MID-strength checkpoints; never
  upload current-best (counter-intelligence: don't teach the field our final bot).
  Scrape our own ladder replays → feed prediction head + weakness review.

## 9. Deployment build

### 9.1 Inference engine

Rust-native forward pass inside `terminal_sim` (no numpy, no onnxruntime): conv3×3 via
im2col + matmul f32 (single core, autovectorized), GRU/FC trivial. Weights: single flat
binary (`weights.bin`, header + f32 arrays, ~10.5 MB) parsed by the .so; a build-time
export script serializes from the PyTorch checkpoint. Gate: max |rust − torch| < 1e-4 on
1K random states.

### 9.2 Folder layout (≤ 50 MB unpacked; expected ~15 MB)

python-algo/: algo_strategy.py (thin driver), gamelib/, terminal_sim.so (manylinux abi3),
weights.bin, fallback.py (pure-python funnel bot, zero deps).

### 9.3 Turn budget (5 s soft; target p99 < 2.5 s)

Parse+features ~5 ms; torso ~10–20 ms ×2 (own+opp perspective); decode K=16 plans ~30 ms;
K·M = 128 sim forks ~50 ms; value batch of 128 ~150–300 ms; overhead → ~0.6–1.0 s typical.
Anytime loop: start K=8,M=4; double K then M while elapsed < 2.5 s; watchdog thread
submits best-so-far at 4.0 s wall. First candidate scored is ALWAYS the all-defense plan
(a submittable answer exists from ~0.3 s). If .so import fails at game start →
fallback.py runs the whole match.

### 9.4 Rehearsal

On the pod (no Docker-in-Docker): `taskset -c 0` + `ulimit -v 3500000` + `nice` running
50 full matches vs the gauntlet with per-turn wall-clock logging. Any timeout = blocker.

---

## 10. Compute plan (RunPod), run schedule

Pod: Secure Cloud on-demand, **1× H100 80 GB SXM** (PCIe acceptable), the listing with the
MOST vCPUs (typically 20–26 vCPU / 200+ GB RAM), PyTorch 2.x CUDA 12 template, 30 GB
container disk + 100 GB network volume. Fallback: A100 80 GB (~60% price, ~75% throughput
here — the workload is actor-CPU-bound; do NOT pay for 2 GPUs, pay for cores).

| Phase | wall hours | purpose |
|---|---|---|
| P0 bootstrap | 2–3 | setup_runpod.sh, fidelity gate, BC warm start, seed games |
| P1 pilot | 6–8 | full loop small (K=8,M=4); gate: gauntlet win-rate rising hour/hour |
| P2 main | 12–18 | full scale; snapshots every 30 min; gauntlet every 2 h |
| P3 finish | 4–6 | optional exploiter, distill/export, Rust-parity gate, rehearsal |

Total ≈ 25–35 GPU-hours ≈ $80–120 at ~$3/h (H100) or ~$55–75 (A100). Buy ~10 h now for
P0+P1; extend only if the pilot gate passes. Checkpoint to the network volume every
10 min (spot-interruption safe; spot OK for P1 only).

## 11. Additional efficiency measures (included above, listed for visibility)

Mirror augmentation (§2.3); scripted-game value pretraining (§6.3); early resignation
with exemption quota (§5.4); PFSP snapshot eviction (§6.4); identical search machinery in
training and deployment (zero train/deploy skew); early-exit compute cap when winning
(§5.4 — also serves the turn-100 compute-time tiebreaker); position-seeded self-play
(start 15% of games from buffer-sampled mid-game states for coverage) (tunable, P2 only).

## 12. What Anish must provide

1. PROBE| log lines from the platform match (gates: python version, .so loadability).
2. Organizer answer on native .so files (if prohibited → §9.1 switches to a pure-python
   fixed-point int8 forward pass — 10× slower, K=6/M=4 budget, still viable).
3. 50–200 ladder replay IDs with team labels (for §7 dedupe).
4. RunPod pod per §10 + repo access on it (private GitHub token or rsync).


---

## 13. Curriculum: cold start → self-play (the two-stage story, explicit)

**Stage A — supervised warm start (GPU hours 0–2, all data already on disk):**
1. Parse the scraped corpus (~2,000–3,000 competition replays in replays/scraped/) into
   (state tensors, executed plan, outcome) triples for BOTH sides of every game.
2. Train, simultaneously, from scratch:
   - policy decoder on WINNERS' plans only (loser plans excluded), fingerprint-capped
     (§7.3) — many of these teams are weak, so this is only a sane-opening prior, not a
     target: 1 epoch cap, stop at plateau, expect ~30–40% token top-1;
   - opponent-prediction head on ALL plans by ALL players (the population model — weak
     players are IN the population we must predict, so their data is genuinely useful);
   - value + aux heads on outcomes z of the same games PLUS 5–10K scripted-vs-scripted
     sim games generated on the CPU while the GPU trains (§6.3).
3. Freeze a copy = **BC-anchor** (league member + gauntlet opponent forever).
   Gate: anchor beats the starter algo ≥90% when driven by the K×M search.

**Stage B — Expert Iteration self-play (the main engine, GPU hours 2–25):**
Actors play league games (§6.4 mix); every decision runs the K×M search (§5); the search
result π* becomes the policy target, game outcomes become value targets, opponents' actual
plans feed the prediction head — the network is forever chasing its own search, and the
search is forever sharpened by the better network (standard ExIt loop). The BC prior
washes out automatically as self-play data floods the buffer (FIFO 500K). No phase switch,
no reward change, no schedule cliff: Stage B is one continuous run with snapshots every
30 min and the gauntlet every 2 h.

**Stage C — selection & packaging (last 4–6 h):** pick best-by-gauntlet (min-based
promotion rule §8), export weights.bin, build dist/python-algo/, Rust-parity gate,
1-core rehearsal, ship.

## 14. Implementation contract for the code-writing pass

Produce exactly this tree (names are binding; each module lists its public surface):

```
train/
  config.yaml          # every hyperparameter in this doc, under the section names used here
  features.py          # build_planes(game, player, history) -> (board[18,28,28] f32, scalars[14])
                       #   thin wrapper over Game.board_planes + deploy-history planes 12-17
  tokens.py            # Token = (type:int, loc:int, count_bucket:int); encode/decode plan<->
                       #   sim command tuples; legality mask builder over a plan-scratch;
                       #   mirror-augmentation remapping
  model.py             # TerminalNet(nn.Module): torso/value/aux/policy-GRU/prediction-GRU
                       #   per §4; forward_torso, decode_step, value, aux, predict_step
  search.py            # choose(game, net_client, K, M, tau, budget_s) -> (plan, pi_star,
                       #   diagnostics); §5 exactly; identical code path used by actors AND
                       #   the deployment driver (train/deploy split happens at net_client)
  replays.py           # scraped-corpus reader -> BC/prediction/value datasets; winner
                       #   filter; opening-fingerprint dedupe caps; mirror augmentation
  scripted.py          # the 5 scripted bots (rush/funnel/demolisher-line/turtle/torture)
                       #   as pure functions state->commands, used by league + gauntlet
  league.py            # snapshot pool, PFSP sampling, eviction (§6.4)
  actor.py             # one process: plays games, streams trajectories to the learner
                       #   over a socket/shared-mem queue; resignation rule §5.4
  infer_server.py      # GPU batching server: torso/decode/value/predict request types,
                       #   batch<=512 or 3ms; weight hot-reload every ~2min
  learner.py           # replay buffer, losses §6.1, optimizer §6.2, checkpointing
  evaluate.py          # gauntlet runner + promotion rule §8; writes eval/report.json
  export.py            # torch checkpoint -> weights.bin (flat f32, headered) + parity test
  run.py               # CLI: --phase {bootstrap,pilot,main,package} orchestrating all of
                       #   the above; resumable; tmux-friendly logging + tensorboard
deploy/
  algo_strategy.py     # thin driver: gamelib parse -> Game mirror -> search.choose with
                       #   anytime budget §9.3 -> submit; watchdog; fallback ladder
  fallback.py          # zero-dependency scripted funnel bot
```

Definition of done (the "write the code" prompt should end with all of these green):
1. `cargo test` + `tsim diff` on every replay in replays/ AND a 200-replay sample of
   replays/scraped/ → PASS rate ≥ 99% of frames (documented residuals only).
2. `python -m train.run --phase bootstrap` on the pod: builds everything, Stage A
   completes, BC-anchor gate passes.
3. `python -m train.run --phase pilot` (K=8, M=4, 2 h): gauntlet win-rate strictly
   increasing across the run; zero actor crashes.
4. `python -m train.run --phase main` runs unattended (only GPU-hours limit it).
5. `python -m train.run --phase package`: dist/python-algo/ < 50 MB unpacked, parity
   test < 1e-4, 50-game 1-core rehearsal zero timeouts, folder uploads and wins vs the
   probe algo on the platform.

Human effort after code lands: buy pod → run 4 commands → watch TensorBoard →
download dist/python-algo/ → upload. Nothing else.
