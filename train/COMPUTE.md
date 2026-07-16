# Compute: The Buying Decision & Runbook

Companion to GPU_SETUP.md (same conclusions; this doc adds the *reasoning* and the
step-by-step runbook so the whole team can execute the purchase and setup without
guessing). Read this before renting anything.

## What to buy — one machine, exactly this profile

| Component | Spec | Why |
|---|---|---|
| GPU | **1× H100 80 GB** (SXM or PCIe; NVLink irrelevant) | Learner + batched inference server. The net is 2.6 M params — one GPU is nowhere near saturated. **Never buy a second GPU; spend that money on cores.** |
| vCPUs | **Target 64, floor 32** — take the listing with the most cores | The actual bottleneck; see below. |
| RAM | **≥ 128 GB** | 500 K-position replay buffer (~20 GB) + ~2 actor processes per vCPU + replay parsing. |
| Disk | 30 GB container + **100 GB network volume** | Checkpoints every 10 min + scraped corpus + trajectory spool. The volume survives pod death — that is the spot-interruption insurance. |
| Image | **Ubuntu 22.04** PyTorch 2.x CUDA 12 template | A hard constraint, not a preference: glibc 2.35 matches the competition container exactly (probe-verified), so the .so built on the pod runs on the platform unmodified. |
| Billing | On-demand for the main run; spot only for the pilot | Checkpoints make spot survivable, but a spot kill mid-final-run costs hours we don't have before Sunday. |

**Why vCPU count matters as much as the GPU:** this workload is inverted from normal
deep learning. Every self-play decision runs K×M ≈ 72 Rust-sim forks **on CPU**
(~5,500 turns/sec/core) and sends one small batch to the GPU. The GPU idles waiting
for actors; games/min scales **linearly with cores** (20 vCPU ≈ 60–150 games/min;
64 vCPU ≈ 2–3×). The wall-clock schedule is fixed (~25–35 GPU-hours before Sunday),
so cores directly convert into how much of the ExIt curve we climb before freeze.
Hence the rule: **A100 + 64 vCPU beats H100 + 16 vCPU** — the A100 costs ~25%
training throughput, the missing cores cost ~70%.

**Budget:** buy ~10 hours first (bootstrap + pilot ≈ $30). Extend to the full
~$100–140 **only if** the pilot gate passes (gauntlet win-rate rising
hour-over-hour). Never prepay the whole run before the pilot proves the loop.

## Where to buy

1. **RunPod** (Secure Cloud, on-demand) — first choice; per-listing vCPU counts
   visible, network volumes, `setup_runpod.sh` targets it.
2. **Lambda Labs** — reliable SSH boxes; their 1×H100 is 26 vCPU (acceptable).
3. **Vast.ai** — cheapest, variable trust; pilot only, hosts with ≥99% reliability
   and DEDICATED cores.

## What is banned, and the real reasons

- **Colab / Kaggle / notebook services.** This is a 15+ hour system of ~50
  cooperating processes (actors + shared-memory inference server + learner +
  TensorBoard) that must survive disconnects. Notebooks give session time limits,
  preemption, throttled background execution, and no persistent daemons. The run
  dies when the tab does. (Notebooks are fine for analyzing results — never as host.)
- **Serverless GPU endpoints (Modal/Replicate-style).** The architecture needs a
  stateful learner holding a 20 GB buffer and an inference server answering
  micro-batches every ~3 ms. Per-call serverless has neither state nor latency, and
  you would pay per-invocation for millions of tiny calls.
- **Split environments (local CPU actors ↔ cloud GPU) — the one that sounds clever.**
  The inference server's batching window is **3 milliseconds**; internet RTT is
  30–100 ms — 10–30× the entire window — and each decision needs ~20 *sequential*
  decoder steps plus a value batch. Throughput collapses ~50× while the GPU idles.
  Shared-memory IPC cannot cross a network. And Windows laptops cannot build the
  glibc-matched .so the submission needs. Everything lives on one box — an
  architectural requirement, not a preference.

## Console runbook (node live → training)

```bash
# 0. Provider UI: create the 100 GB network volume FIRST (same region as the pod),
#    then deploy the pod with the volume attached, PyTorch/Ubuntu 22.04 template.

# 1. SSH in (key from the provider dashboard)
ssh root@<pod-ip> -p <port>

# 2. Get the repo (fine-grained GitHub token for the private repo)
git clone https://<TOKEN>@github.com/anish-agr/terminal-velocity.git
cd terminal-velocity

# 3. Ship the scraped replay corpus up from the machine that has it (NOT in git)
rsync -avz -e "ssh -p <port>" replays/scraped/ root@<pod-ip>:~/terminal-velocity/replays/scraped/

# 4. One-command bootstrap: Rust, sim build+tests, Linux wheel, deps, fidelity gate
bash train/setup_runpod.sh
#    HARD STOP unless it prints: "ALL REPLAYS PASS" and "bridge OK".
#    A fidelity failure means DO NOT TRAIN — ping the team instead.

# 5. Everything below runs inside tmux (SSH drop != dead run)
tmux new -s train
python -m train.run --phase bootstrap      # Stage A: BC warm start + seed games
python -m train.run --phase pilot          # 2 h small-scale; watch the gate
#   TensorBoard (2nd tmux window): tensorboard --logdir runs --port 6006 --bind_all
python -m train.run --phase main           # the long run; checkpoints -> volume
python -m train.run --phase package        # export weights.bin, build dist/, rehearse

# 6. Pull the finished submission folder down; upload to the portal
rsync -avz -e "ssh -p <port>" root@<pod-ip>:~/terminal-velocity/dist/python-algo/ ./submit/
```
