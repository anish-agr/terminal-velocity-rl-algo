"""Generate parity fixtures for the Rust forward pass (sim/src/nn.rs).

Builds a randomly-initialized TerminalNet, writes a REAL weights.bin through
train/export.py::write_weights (the production format, not a re-implementation),
computes PyTorch ground-truth outputs on random states, and dumps everything
into sim/target/nn_fixtures/ for sim/tests/nn_parity.rs to check to < 1e-4.

target/ is cargo-ignored, so the ~10 MB of fixtures never enter git. Re-run
any time; after training, run it against the real checkpoint instead:

    python scripts/gen_nn_fixtures.py                     # random-init net
    python scripts/gen_nn_fixtures.py runs/r1/weights_current.pt   # trained

io.bin layout (little-endian; must match nn_parity.rs exactly):
    magic  b"TVF1"
    u32    n_states
    f32    boards   [n*18*784]
    f32    scalars  [n*14]
    f32    feat_ref [n*64*784]
    f32    g_ref    [n*128]
    f32    v_ref    [n]
    per head, "policy" then "predict":
        f32 c0_ref     [n*128]
        f32 keys_ref   [n*64*784]
        f32 type_ref   [n*9]
        f32 loc_ref    [n*784]     (exercises fc_q, not covered by keys)
        u32 locs       [n]
        f32 count_ref  [n*8]
        u32 ttypes     [n]
        u32 counts     [n]
        f32 adv_ref    [n*128]
"""

import os
import struct
import sys
import time

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from train.export import write_weights  # noqa: E402
from train.model import TerminalNet  # noqa: E402

N_STATES = 4
OUT_DIR = os.path.join(_REPO, "sim", "target", "nn_fixtures")


def main():
    torch.manual_seed(7)
    net = TerminalNet()
    if len(sys.argv) > 1:
        net.load_state_dict(torch.load(sys.argv[1], map_location="cpu"))
        print("loaded trained checkpoint:", sys.argv[1])
    net.eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    write_weights(net.state_dict(), os.path.join(OUT_DIR, "weights.bin"))

    rng = np.random.default_rng(0)
    boards = rng.standard_normal((N_STATES, 18, 28, 28)).astype(np.float32)
    scalars = rng.standard_normal((N_STATES, 14)).astype(np.float32)
    locs = np.array([5, 100, 400, 783], dtype=np.uint32)
    ttypes = np.array([0, 2, 5, 8], dtype=np.uint32)
    counts = np.array([0, 3, 7, 1], dtype=np.uint32)

    with torch.no_grad():
        feat, g = net.forward_torso(torch.from_numpy(boards),
                                    torch.from_numpy(scalars))
        v = net.value(g)
        t0 = time.perf_counter()
        for _ in range(10):
            f2, g2 = net.forward_torso(torch.from_numpy(boards),
                                       torch.from_numpy(scalars))
            net.value(g2)
        torch_ms = (time.perf_counter() - t0) / (10 * N_STATES) * 1e3

        heads = {}
        for head in ("policy", "predict"):
            dec = getattr(net, head)
            c0, keys = dec.init(feat, g)
            heads[head] = {
                "c0": c0.numpy(),
                "keys": keys.numpy(),
                "type": dec.type_logits(c0).numpy(),
                "count": dec.count_logits(
                    c0, feat, torch.from_numpy(locs.astype(np.int64))).numpy(),
                "adv": dec.advance(
                    c0, feat,
                    torch.from_numpy(ttypes.astype(np.int64)),
                    torch.from_numpy(locs.astype(np.int64)),
                    torch.from_numpy(counts.astype(np.int64))).numpy(),
                "loc": dec.loc_logits(c0, keys).numpy(),
            }

    path = os.path.join(OUT_DIR, "io.bin")
    with open(path, "wb") as fh:
        fh.write(b"TVF1")
        fh.write(struct.pack("<I", N_STATES))
        fh.write(boards.astype("<f4").tobytes())
        fh.write(scalars.astype("<f4").tobytes())
        fh.write(feat.numpy().astype("<f4").tobytes())
        fh.write(g.numpy().astype("<f4").tobytes())
        fh.write(v.numpy().astype("<f4").tobytes())
        for head in ("policy", "predict"):
            h = heads[head]
            fh.write(h["c0"].astype("<f4").tobytes())
            fh.write(h["keys"].astype("<f4").tobytes())
            fh.write(h["type"].astype("<f4").tobytes())
            fh.write(h["loc"].astype("<f4").tobytes())
            fh.write(locs.astype("<u4").tobytes())
            fh.write(h["count"].astype("<f4").tobytes())
            fh.write(ttypes.astype("<u4").tobytes())
            fh.write(counts.astype("<u4").tobytes())
            fh.write(h["adv"].astype("<f4").tobytes())

    # numpy-reference timing on the same machine, for the rung-1-vs-rung-2 call
    sys.path.insert(0, os.path.join(_REPO, "deploy"))
    from npforward import NumpyNet  # noqa: E402
    npnet = NumpyNet(os.path.join(OUT_DIR, "weights.bin"))
    t0 = time.perf_counter()
    for _ in range(10):
        _, gn = npnet.forward_torso(boards, scalars)
        npnet.value(gn)
    numpy_ms = (time.perf_counter() - t0) / (10 * N_STATES) * 1e3
    worst = float(np.abs(gn - g.numpy()).max())
    assert worst < 1e-4, worst

    print("fixtures -> {}".format(OUT_DIR))
    print("torso+value per state: torch {:.1f} ms | numpy {:.1f} ms "
          "(npforward parity vs torch: {:.2e})".format(torch_ms, numpy_ms, worst))


if __name__ == "__main__":
    main()
