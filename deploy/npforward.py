"""Pure-numpy TerminalNet forward pass + NetClient (inference ladder rung 2).

Loads weights.bin (train/export.py format) and reproduces model.py exactly:
torch's Conv2d (3x3 pad 1, via batched im2col), Linear, GRUCell (gate order
r, z, n), the FiLM-lite scalar bias, and both decoders. Verified against
PyTorch to < 1e-4 by train/export.py::parity_check — which also makes this
module the executable reference the Rust port (rung 1) must match.

No torch anywhere: the competition container has numpy (probe-verified) and
torch's 2.5 s import is unaffordable at match time. NumpyNetClient implements
the same NetClient protocol as search.LocalNetClient, so search.choose() runs
unmodified on top of it.
"""

from __future__ import annotations

import os
import sys

# Cap BLAS/OpenMP threads before numpy loads (defensive: algo_strategy sets these
# too, but npforward may be imported first). Uncapped OpenBLAS spawns one thread
# per host core and dies on the competition container's process limit.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from typing import Dict, List, Tuple

import numpy as np

# dist layout puts train/ next to this file's dir; repo layout has it one up
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train.search import NetClient  # noqa: E402  (numpy-only import chain)
from train.tokens import DEPLOY_TYPES, END, END_TOKEN, Token  # noqa: E402


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _relu(x):
    return np.maximum(x, 0.0)


def _linear(w: Dict[str, np.ndarray], prefix: str, x: np.ndarray) -> np.ndarray:
    return x @ w[prefix + ".weight"].T + w[prefix + ".bias"]


def _conv3x3(weight: np.ndarray, bias: np.ndarray, x: np.ndarray) -> np.ndarray:
    """x [B,C,28,28], weight [O,C,3,3] -> [B,O,28,28]. Batched im2col."""
    b, c, h, wd = x.shape
    o = weight.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    # cols [B, C*9, H*W]
    cols = np.empty((b, c * 9, h * wd), dtype=np.float32)
    idx = 0
    for dy in range(3):
        for dx in range(3):
            patch = xp[:, :, dy:dy + h, dx:dx + wd]
            cols[:, idx * c:(idx + 1) * c] = patch.reshape(b, c, h * wd)
            idx += 1
    # torch layout is weight[o, c, ky, kx] -> flatten to [o, ky*kx*c] matching
    # our col order (ky, kx, c)
    wmat = weight.transpose(2, 3, 1, 0).reshape(9 * c, o)
    out = np.einsum("bkp,ko->bop", cols, wmat, optimize=True)
    return (out + bias[None, :, None]).reshape(b, o, h, wd).astype(np.float32)


class NumpyNet:
    def __init__(self, weights_path: str):
        from train.export import read_weights

        self.w = read_weights(weights_path)

    # -- torso + heads --------------------------------------------------------

    def forward_torso(self, board: np.ndarray, scalars: np.ndarray):
        """board [B,18,28,28], scalars [B,14] -> (F [B,64,28,28], g [B,128])."""
        w = self.w
        s = _relu(_linear(w, "scalar_mlp.0", scalars))
        s = _linear(w, "scalar_mlp.2", s)
        x = _conv3x3(w["stem.weight"], w["stem.bias"], board)
        x = _relu(x + s[:, :, None, None])
        for i in range(6):
            y = _relu(_conv3x3(w["blocks.{}.c1.weight".format(i)],
                               w["blocks.{}.c1.bias".format(i)], x))
            y = _conv3x3(w["blocks.{}.c2.weight".format(i)],
                         w["blocks.{}.c2.bias".format(i)], y)
            x = _relu(x + y)
        g = np.concatenate([x.mean(axis=(2, 3)), x.max(axis=(2, 3))], axis=1)
        return x, g.astype(np.float32)

    def value(self, g: np.ndarray) -> np.ndarray:
        h = _relu(_linear(self.w, "fc_value.0", g))
        return np.tanh(_linear(self.w, "fc_value.2", h)).reshape(-1)

    # -- decoder primitives (head in {"policy", "predict"}) --------------------

    def _gru(self, head: str, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        w = self.w
        gi = x @ w[head + ".gru.weight_ih"].T + w[head + ".gru.bias_ih"]
        gh = h @ w[head + ".gru.weight_hh"].T + w[head + ".gru.bias_hh"]
        hid = h.shape[1]
        i_r, i_z, i_n = gi[:, :hid], gi[:, hid:2 * hid], gi[:, 2 * hid:]
        h_r, h_z, h_n = gh[:, :hid], gh[:, hid:2 * hid], gh[:, 2 * hid:]
        r = _sigmoid(i_r + h_r)
        z = _sigmoid(i_z + h_z)
        n = np.tanh(i_n + r * h_n)
        return ((1.0 - z) * n + z * h).astype(np.float32)

    @staticmethod
    def _cell(feat: np.ndarray, loc: np.ndarray) -> np.ndarray:
        b, c = feat.shape[0], feat.shape[1]
        flat = feat.reshape(b, c, -1)
        return flat[np.arange(b), :, loc].astype(np.float32)

    def decoder_init(self, head: str, feat: np.ndarray, g: np.ndarray):
        c0 = np.tanh(_linear(self.w, head + ".fc_init", g)).astype(np.float32)
        wk = self.w[head + ".wk.weight"].reshape(
            self.w[head + ".wk.weight"].shape[0], -1)
        b, ch = feat.shape[0], feat.shape[1]
        keys = (wk @ feat.reshape(b, ch, -1)) + \
            self.w[head + ".wk.bias"][None, :, None]
        return c0, keys.reshape(b, wk.shape[0], 28, 28).astype(np.float32)

    def type_logits(self, head: str, c: np.ndarray) -> np.ndarray:
        return _linear(self.w, head + ".fc_type", c)

    def loc_logits(self, head: str, c: np.ndarray, keys: np.ndarray) -> np.ndarray:
        q = _linear(self.w, head + ".fc_q", c)
        return np.einsum("bq,bqxy->bxy", q, keys,
                         optimize=True).reshape(c.shape[0], -1)

    def count_logits(self, head: str, c: np.ndarray, feat: np.ndarray,
                     loc: np.ndarray) -> np.ndarray:
        return _linear(self.w, head + ".fc_count",
                       np.concatenate([c, self._cell(feat, loc)], axis=1))

    def advance(self, head: str, c: np.ndarray, feat: np.ndarray,
                ttype: np.ndarray, loc: np.ndarray, count: np.ndarray):
        w = self.w
        e = np.concatenate([
            w[head + ".type_emb.weight"][ttype],
            self._cell(feat, loc),
            w[head + ".count_emb.weight"][count],
        ], axis=1)
        return self._gru(head, np.tanh(_linear(w, head + ".fc_tok", e)), c)


# ---------------------------------------------------------------------------
# NetClient over NumpyNet — the deployment inference path
# ---------------------------------------------------------------------------

def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    return z - np.log(np.exp(z).sum())


def _pick(logits: np.ndarray, mask: np.ndarray, tau: float, greedy: bool,
          rng: np.random.Generator) -> Tuple[int, float]:
    masked = np.where(mask, logits, -1e9)
    lp = _log_softmax(masked)
    if greedy:
        idx = int(masked.argmax())
    else:
        p = np.exp(_log_softmax(masked / tau))
        idx = int(rng.choice(len(p), p=p / p.sum()))
    return idx, float(lp[idx])


class NumpyNetClient(NetClient):
    """search.NetClient implemented on NumpyNet — no torch at match time."""

    def __init__(self, net: NumpyNet, seed: int = 0):
        self.net = net
        self.rng = np.random.default_rng(seed)

    def _torso1(self, board, scalars):
        return self.net.forward_torso(board[None], scalars[None])

    def sample_plans(self, board, scalars, scratch_factory, k, tau, head,
                     greedy_extra=False, mask_deploys_extra=False):
        feat, g = self._torso1(board, scalars)
        out = []
        for _ in range(k):
            out.append(self._sample(head, feat, g, scratch_factory(), tau, False, False))
        if greedy_extra:
            out.append(self._sample(head, feat, g, scratch_factory(), tau, True, False))
        if mask_deploys_extra:
            out.append(self._sample(head, feat, g, scratch_factory(), tau, True, True))
        return out

    def _sample(self, head, feat, g, scratch, tau, greedy, mask_deploys):
        net = self.net
        c, keys = net.decoder_init(head, feat, g)
        plan: List[Token] = []
        logp = 0.0
        for _ in range(23):
            tmask = scratch.type_mask()
            if mask_deploys:
                for t in DEPLOY_TYPES:
                    tmask[t] = False
            ttype, lp = _pick(net.type_logits(head, c)[0], tmask, tau, greedy, self.rng)
            logp += lp
            if ttype == END:
                break
            loc, lp = _pick(net.loc_logits(head, c, keys)[0],
                            scratch.loc_mask(ttype), tau, greedy, self.rng)
            logp += lp
            count, lp = _pick(
                net.count_logits(head, c, feat, np.array([loc]))[0],
                scratch.count_mask(ttype), tau, greedy, self.rng)
            if ttype in DEPLOY_TYPES:
                logp += lp
            else:
                count = 0
            tok = Token(ttype, loc, count)
            scratch.apply(tok)
            plan.append(tok)
            c = net.advance(head, c, feat, np.array([ttype]), np.array([loc]),
                            np.array([count]))
        plan.append(END_TOKEN)
        return tuple(plan), logp

    def score_plans(self, board, scalars, scratch_factory, plans, head):
        feat, g = self._torso1(board, scalars)
        net = self.net
        out = []
        for plan in plans:
            scratch = scratch_factory()
            c, keys = net.decoder_init(head, feat, g)
            logp = 0.0
            for tok in plan:
                tmask = scratch.type_mask()
                if not tmask[tok.type]:
                    continue
                logp += _log_softmax(
                    np.where(tmask, net.type_logits(head, c)[0], -1e9))[tok.type]
                if tok.type == END:
                    break
                lmask = scratch.loc_mask(tok.type)
                if not lmask[tok.loc]:
                    continue
                logp += _log_softmax(
                    np.where(lmask, net.loc_logits(head, c, keys)[0], -1e9))[tok.loc]
                if tok.type in DEPLOY_TYPES:
                    cmask = scratch.count_mask(tok.type)
                    if not cmask[tok.count]:
                        continue
                    logp += _log_softmax(np.where(
                        cmask, net.count_logits(head, c, feat,
                                                np.array([tok.loc]))[0],
                        -1e9))[tok.count]
                scratch.apply(tok)
                c = net.advance(head, c, feat, np.array([tok.type]),
                                np.array([tok.loc]), np.array([tok.count]))
            out.append(float(logp))
        return out

    def values(self, boards, scalars):
        _, g = self.net.forward_torso(boards, scalars)
        return self.net.value(g)
