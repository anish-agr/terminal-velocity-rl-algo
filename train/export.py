"""weights.bin export + parity gate (ARCHITECTURE §9.1).

Format (all little-endian) — the contract for BOTH consumers (the numpy
forward pass in deploy/npforward.py and the Rust port inside terminal_sim):

    magic   4 bytes  b"TVW1"
    count   u32      number of tensors
    per tensor, in state_dict order:
        name_len u16, name utf-8, ndim u8, dims u32 x ndim, payload f32 LE

Gate: max |reference - torch| < 1e-4 on random states. Until the Rust forward
lands, the reference implementation is deploy/npforward.py — the parity test
here runs LOCALLY and the Rust port must later match the same file to the same
tolerance (§0 gate 8).
"""

from __future__ import annotations

import struct
from typing import Dict

import numpy as np

MAGIC = b"TVW1"


def write_weights(state_dict, path: str) -> None:
    """Serialize a TerminalNet state_dict (torch tensors) to weights.bin."""
    items = [(k, v.detach().cpu().numpy().astype("<f4")) for k, v in
             state_dict.items()]
    with open(path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(items)))
        for name, arr in items:
            nb = name.encode("utf-8")
            fh.write(struct.pack("<H", len(nb)))
            fh.write(nb)
            fh.write(struct.pack("<B", arr.ndim))
            for d in arr.shape:
                fh.write(struct.pack("<I", d))
            fh.write(arr.tobytes(order="C"))


def read_weights(path: str) -> Dict[str, np.ndarray]:
    """Parse weights.bin -> {name: f32 ndarray}. Pure numpy/stdlib."""
    out: Dict[str, np.ndarray] = {}
    with open(path, "rb") as fh:
        if fh.read(4) != MAGIC:
            raise ValueError("bad magic in " + path)
        (count,) = struct.unpack("<I", fh.read(4))
        for _ in range(count):
            (nlen,) = struct.unpack("<H", fh.read(2))
            name = fh.read(nlen).decode("utf-8")
            (ndim,) = struct.unpack("<B", fh.read(1))
            dims = struct.unpack("<{}I".format(ndim), fh.read(4 * ndim))
            n = int(np.prod(dims)) if dims else 1
            arr = np.frombuffer(fh.read(4 * n), dtype="<f4").reshape(dims)
            out[name] = arr.copy()
    return out


def export_checkpoint(checkpoint_or_net, out_path: str) -> None:
    """torch checkpoint path / state_dict / TerminalNet -> weights.bin."""
    import torch

    obj = checkpoint_or_net
    if isinstance(obj, str):
        obj = torch.load(obj, map_location="cpu")
    if isinstance(obj, dict) and "net" in obj:
        obj = obj["net"]
    if hasattr(obj, "state_dict") and not isinstance(obj, dict):
        obj = obj.state_dict()
    write_weights(obj, out_path)


def parity_check(net, weights_path: str, n_states: int = 64,
                 tol: float = 1e-4, seed: int = 0) -> float:
    """Max |numpy forward - torch forward| over random states; raises if > tol.

    Covers the torso, value head, and every decoder primitive of both heads —
    the same surface the Rust port must match (§0 gate 8).
    """
    import torch

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "deploy"))
    from npforward import NumpyNet

    rng = np.random.default_rng(seed)
    npnet = NumpyNet(weights_path)
    net = net.eval()
    worst = 0.0

    boards = rng.standard_normal((n_states, 18, 28, 28)).astype(np.float32)
    scalars = rng.standard_normal((n_states, 14)).astype(np.float32)

    with torch.no_grad():
        feat_t, g_t = net.forward_torso(torch.from_numpy(boards),
                                        torch.from_numpy(scalars))
        v_t = net.value(g_t).numpy()
    feat_n, g_n = npnet.forward_torso(boards, scalars)
    v_n = npnet.value(g_n)
    worst = max(worst, float(np.abs(feat_n - feat_t.numpy()).max()))
    worst = max(worst, float(np.abs(g_n - g_t.numpy()).max()))
    worst = max(worst, float(np.abs(v_n - v_t).max()))

    for head in ("policy", "predict"):
        dec_t = getattr(net, head)
        with torch.no_grad():
            c_t, k_t = dec_t.init(feat_t[:4], torch.from_numpy(g_n[:4]))
        c_n, k_n = npnet.decoder_init(head, feat_n[:4], g_n[:4])
        worst = max(worst, float(np.abs(c_n - c_t.numpy()).max()))
        worst = max(worst, float(np.abs(k_n - k_t.numpy()).max()))

        loc = np.array([5, 100, 400, 783])
        ttype = np.array([0, 2, 5, 8])
        count = np.array([0, 3, 7, 1])
        with torch.no_grad():
            tl_t = dec_t.type_logits(c_t).numpy()
            ll_t = dec_t.loc_logits(c_t, k_t).numpy()
            cl_t = dec_t.count_logits(c_t, feat_t[:4],
                                      torch.from_numpy(loc)).numpy()
            adv_t = dec_t.advance(
                c_t, feat_t[:4], torch.from_numpy(ttype),
                torch.from_numpy(loc), torch.from_numpy(count)).numpy()
        worst = max(worst, float(np.abs(npnet.type_logits(head, c_n) - tl_t).max()))
        worst = max(worst, float(np.abs(npnet.loc_logits(head, c_n, k_n) - ll_t).max()))
        worst = max(worst, float(np.abs(
            npnet.count_logits(head, c_n, feat_n[:4], loc) - cl_t).max()))
        worst = max(worst, float(np.abs(
            npnet.advance(head, c_n, feat_n[:4], ttype, loc, count) - adv_t).max()))

    if worst > tol:
        raise AssertionError("parity {} > tol {}".format(worst, tol))
    return worst
