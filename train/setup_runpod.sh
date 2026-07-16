#!/bin/bash
# RunPod training-box bootstrap. Target: 1x H100 (or 2x A100) pod with 32-64 vCPUs,
# a PyTorch template image (torch preinstalled), Ubuntu 22+.
#
# Usage: clone the repo onto the pod (private GitHub token or scp), then:
#   bash train/setup_runpod.sh
# Everything is idempotent. The fidelity gate at the end MUST print PASS for every replay
# before any training run starts — never train on an unverified sim build.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== system =="
nproc; free -g | head -2; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "== rust toolchain =="
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"

echo "== build simulator (CLI + tests) =="
cargo test --manifest-path sim/Cargo.toml --release
cargo build --manifest-path sim/Cargo.toml --release

echo "== build python extension (native linux, abi3) =="
pip install -q maturin
(cd sim && maturin build --release --features python)
pip install -q --force-reinstall sim/target/wheels/terminal_sim-*.whl

echo "== python deps =="
pip install -q numpy tensorboard pyyaml

# CRITICAL: `cargo build --features python` (which the maturin step above runs
# internally) rebuilds the WHOLE package under that feature, including the `tsim` CLI
# binary — pyo3's extension-module ABI makes that binary silently non-functional when
# run standalone (exits 0, prints nothing; verified 2026-07-16, cost a wasted
# 3,103-replay validation run before being caught). Rebuild the plain, feature-free
# binary again here so the fidelity gate below tests the actual deployed sim, not a
# python-linked one.
echo "== rebuild plain CLI binary (undo any python-feature contamination) =="
cargo build --manifest-path sim/Cargo.toml --release

echo "== fidelity gate: sim must be frame-exact vs engine.jar replays =="
FAIL=0
for r in replays/*.replay; do
  if ! ./sim/target/release/tsim diff "$r" 1 | tail -1 | grep -q PASS; then
    echo "FIDELITY FAILURE: $r"
    FAIL=1
  fi
done
[ "$FAIL" = "0" ] && echo "ALL REPLAYS PASS"

echo "== bridge smoke test =="
python - <<'EOF'
import terminal_sim, time
g = terminal_sim.Game(open('game-configs.json').read())
g.play_turn([(2,3,12),(0,13,11)] + [(3,13,0)]*5, [(3,14,27)]*5)
assert not g.game_over()
t0 = time.time(); n = 0
g.reset()
while not g.game_over():
    g.play_turn([(3,13,0)]*int(g.stats(0)[2]), [(3,14,27)]*int(g.stats(1)[2])); n += 1
print(f"bridge OK: {n}-turn game in {time.time()-t0:.4f}s; planes={terminal_sim.PLANES}")
EOF

echo "== READY. Launch training with: python -m train.run (once train/ lands) =="
