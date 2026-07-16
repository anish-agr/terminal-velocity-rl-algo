# GPU Box: What to Buy and How to Set It Up

## Exact spec to request

| Item | Requirement | Why |
|---|---|---|
| GPU | **1× H100 80 GB** (SXM or PCIe — either is fine; we don't need NVLink) | Learner + batched actor inference. One GPU is enough; the workload is CPU-bound. |
| vCPUs | **64 (min 32)** dedicated cores | Self-play actors run the Rust simulator on CPU; games/min scales ~linearly with cores. This is the single most important line after the GPU itself. |
| RAM | **≥ 128 GB** (200+ ideal) | Replay buffer (500K positions ≈ 20 GB) + 64 actor processes + parsing 3K replays. |
| Disk | **≥ 200 GB NVMe** attached to the pod, plus **100 GB persistent volume** if the provider separates them | Repo + 3.1K scraped replays (~8 GB) + checkpoints every 10 min (~11 MB each → ~3 GB/day) + trajectory spool. 200 GB is comfortable, 100 GB is survivable. |
| OS / image | **Ubuntu 22.04** with CUDA 12.x + PyTorch 2.x preinstalled (any "PyTorch" template) | glibc 2.35 matches the competition container EXACTLY → the .so we build here runs there unmodified. This is why 22.04 specifically. |
| Network | SSH access; outbound HTTPS | git clone + replay scraping from the box if needed. |
| Billing | On-demand for main runs; spot/interruptible acceptable ONLY for the pilot (we checkpoint every 10 min) | A spot kill mid-final-run costs hours we don't have. |

Hours: **buy ~10 h first** (bootstrap + pilot). Extend to ~35–40 h total only after the
pilot gate passes (gauntlet win-rate rising hour-over-hour). Ballpark $2.5–3.5/hr for
H100 on the marketplaces below → **$100–140 total**. If only A100 80 GB is available:
fully acceptable (~75% of throughput, ~60% of price) — prefer A100+64vCPU over H100+16vCPU.

## Where to buy — and where NOT to

**Good (in order):**
1. **RunPod** (runpod.io) — Secure Cloud, on-demand. In the deploy screen the vCPU/RAM
   count is listed per offering — pick the H100 listing with the most vCPUs. Use a
   "Network Volume" for persistence.
2. **Lambda Labs** (lambda.ai) — simple SSH boxes, reliable, 1×H100 = 26 vCPU (fine if
   64 isn't available anywhere).
3. **Vast.ai** — cheapest, variable reliability; pilot runs only, verify the host shows
   ≥99% reliability and the advertised cores are DEDICATED.
4. Voltage Park / Nebius / Crusoe — fine if the teammate has access; same spec sheet.

**Do NOT use:**
- **Google Colab / Kaggle notebooks** — session time limits (hrs), preemption, no
  persistent daemons, no real multi-process CPU actors, throttled background execution.
  Our run is a 15+ hour multi-process system, not a notebook.
- **Paperspace free/Gradient notebooks** — same problems.
- **Serverless GPU endpoints** (Modal/Replicate/Banana-style) — per-call model doesn't
  fit a persistent learner + 64 actor processes + shared-memory inference server.
- **Your own laptops** — no CUDA H100, and Windows breaks the glibc-matched .so build.
- Anything without SSH/root: we install Rust and run tmux daemons.

## Setup (15 minutes, copy-paste)

```bash
# 1. clone (use a GitHub fine-grained token if the repo is private)
git clone https://github.com/anish-agr/terminal-velocity.git && cd terminal-velocity

# 2. one-command bootstrap: installs Rust, builds + tests the simulator, builds the
#    Linux python wheel, installs deps, and runs the replay fidelity gate
bash train/setup_runpod.sh
# REQUIRED OUTPUT: "ALL REPLAYS PASS" and "bridge OK". If any replay fails, STOP — do
# not train on an unverified simulator. (Ship the replays/ dir with the repo or rsync it.)

# 3. run everything inside tmux so SSH drops don't kill training
tmux new -s train
```

Training itself (once the train/ package is implemented) is designed to be two commands:
`python -m train.run --phase pilot` then `--phase main`; checkpoints land in
checkpoints/ on the volume every 10 min, TensorBoard on port 6006, and the current-best
deployment folder is continuously exported to `dist/python-algo/` — download it, upload
to the portal, done. No babysitting beyond checking the gauntlet chart every few hours.

## Getting files on/off the box

- Up: `git pull` (code) + `rsync -avz replays/ pod:~/terminal-velocity/replays/` (corpus).
- Down: `rsync -avz pod:~/terminal-velocity/dist/python-algo/ ./python-algo-submit/`
  then upload that folder to the portal. Nothing else needs to move.
