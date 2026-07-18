"""TerminalNet (ARCHITECTURE §4): shared torso, four heads.

Torso: 3x3 stem (18->64) + FiLM-lite scalar bias + 6 norm-free residual blocks.
No batch/layer norm anywhere — keeps the Rust forward-pass port trivial and CPU
inference deterministic (§4.1); shallow enough to train with grad clipping.

Heads:
  value    g -> 256 -> 1, tanh. The ONLY head used for decision scoring (§4.2).
  aux      g -> 128 -> 3. Representation shaping only, never used in decisions.
  policy   autoregressive plan decoder (§4.4) — GRU + spatial pointer.
  predict  a second PolicyDecoder with its OWN parameters (§4.5), run over the
           opponent-perspective torso output.

The decoder exposes step-level primitives (init/type/loc/count/advance) used by
search.py for sampling, plus a teacher-forced `plan_nll` for the learner. All
legality masks are SUPPLIED by callers (built via tokens.PlanScratch) — this
module is pure tensor math and never inspects game state.

Convention: feature maps are [B, C, 28, 28] with the same [x, y] axis order as
features.py; a flattened loc index is x*28+y everywhere.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F_t

from .tokens import DEPLOY_TYPES, END, GRID, N_BUCKETS, N_LOCS, N_TYPES, Token

_NEG = -1e9  # mask fill; large-negative beats -inf for f32 softmax stability


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F_t.relu(x + self.c2(F_t.relu(self.c1(x))))


class PolicyDecoder(nn.Module):
    """Autoregressive (type, loc, count) plan decoder over a torso feature map."""

    TYPE_EMB = 16
    COUNT_EMB = 16
    QK = 64

    def __init__(self, ch: int = 64, hidden: int = 128):
        super().__init__()
        self.hidden = hidden
        self.fc_init = nn.Linear(2 * ch, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.fc_type = nn.Linear(hidden, N_TYPES)
        self.fc_q = nn.Linear(hidden, self.QK)
        self.wk = nn.Conv2d(ch, self.QK, 1)
        self.fc_count = nn.Linear(hidden + ch, N_BUCKETS)
        self.type_emb = nn.Embedding(N_TYPES, self.TYPE_EMB)
        self.count_emb = nn.Embedding(N_BUCKETS, self.COUNT_EMB)
        self.fc_tok = nn.Linear(self.TYPE_EMB + ch + self.COUNT_EMB, hidden)

    # -- step-level API (used by search.py sampling) -------------------------

    def init(self, feat: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> (c0 [B,H], K [B,QK,28,28] pointer keys, precomputed once)."""
        return torch.tanh(self.fc_init(g)), self.wk(feat)

    def type_logits(self, c: torch.Tensor) -> torch.Tensor:
        return self.fc_type(c)

    def loc_logits(self, c: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """[B, 784] pointer scores <W_q c, K[x,y]>."""
        q = self.fc_q(c)
        return torch.einsum("bq,bqxy->bxy", q, keys).reshape(c.shape[0], N_LOCS)

    def count_logits(
        self, c: torch.Tensor, feat: torch.Tensor, loc: torch.Tensor
    ) -> torch.Tensor:
        """[B, 8]; loc is a [B] long tensor of flattened cells."""
        cell = _gather_cell(feat, loc)
        return self.fc_count(torch.cat([c, cell], dim=1))

    def advance(
        self,
        c: torch.Tensor,
        feat: torch.Tensor,
        ttype: torch.Tensor,
        loc: torch.Tensor,
        count: torch.Tensor,
    ) -> torch.Tensor:
        """Feed the chosen token through the GRU -> next committed-plan state."""
        e = torch.cat(
            [self.type_emb(ttype), _gather_cell(feat, loc), self.count_emb(count)],
            dim=1,
        )
        return self.gru(torch.tanh(self.fc_tok(e)), c)

    # -- training API ---------------------------------------------------------

    def plan_nll(
        self,
        feat: torch.Tensor,
        g: torch.Tensor,
        plans: torch.Tensor,
        lengths: torch.Tensor,
        type_masks: torch.Tensor,
        loc_masks: torch.Tensor,
        count_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced sequence NLL (§6.1) and summed masked entropy.

        plans        [B, T, 3] long — (type, loc, count) per step, END-padded
        lengths      [B] long — tokens incl. the END step
        type_masks   [B, T, 9] bool  (True = legal)
        loc_masks    [B, T, 784] bool
        count_masks  [B, T, 8] bool
        Returns (nll [B], entropy [B]). Loc terms are counted for every type
        except END; count terms only for deploy types (§4.4).
        """
        bsz, t_max, _ = plans.shape
        c, keys = self.init(feat, g)
        nll = feat.new_zeros(bsz)
        ent = feat.new_zeros(bsz)
        deploy = torch.zeros(N_TYPES, dtype=torch.bool, device=feat.device)
        for t in DEPLOY_TYPES:
            deploy[t] = True

        for t in range(t_max):
            live = (lengths > t).float()
            ttype, loc, count = plans[:, t, 0], plans[:, t, 1], plans[:, t, 2]

            lt = self.type_logits(c).masked_fill(~type_masks[:, t], _NEG)
            lp = F_t.log_softmax(lt, dim=1)
            nll = nll - lp.gather(1, ttype[:, None]).squeeze(1) * live
            ent = ent + _masked_entropy(lp) * live

            has_loc = (ttype != END).float() * live
            ll = self.loc_logits(c, keys).masked_fill(~loc_masks[:, t], _NEG)
            lpl = F_t.log_softmax(ll, dim=1)
            nll = nll - lpl.gather(1, loc[:, None]).squeeze(1) * has_loc
            ent = ent + _masked_entropy(lpl) * has_loc

            has_count = deploy[ttype].float() * live
            lc = self.count_logits(c, feat, loc).masked_fill(~count_masks[:, t], _NEG)
            lpc = F_t.log_softmax(lc, dim=1)
            nll = nll - lpc.gather(1, count[:, None]).squeeze(1) * has_count

            c = torch.where(
                live[:, None].bool(), self.advance(c, feat, ttype, loc, count), c
            )
        return nll, ent


class TerminalNet(nn.Module):
    def __init__(self, planes: int = 18, scalars: int = 14, ch: int = 64, blocks: int = 6):
        super().__init__()
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalars, 64), nn.ReLU(), nn.Linear(64, ch)
        )
        self.stem = nn.Conv2d(planes, ch, 3, padding=1)
        self.blocks = nn.ModuleList(_ResBlock(ch) for _ in range(blocks))
        self.fc_value = nn.Sequential(nn.Linear(2 * ch, 256), nn.ReLU(), nn.Linear(256, 1))
        self.fc_aux = nn.Sequential(nn.Linear(2 * ch, 128), nn.ReLU(), nn.Linear(128, 3))
        self.policy = PolicyDecoder(ch)
        self.predict = PolicyDecoder(ch)

    def forward_torso(
        self, board: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(F [B,64,28,28], g [B,128]). FiLM-lite: scalar encoding added as a
        per-channel bias to every spatial cell after the stem (§4.1)."""
        s = self.scalar_mlp(scalars)
        x = F_t.relu(self.stem(board) + s[:, :, None, None])
        for blk in self.blocks:
            x = blk(x)
        g = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return x, g

    def value(self, g: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc_value(g)).squeeze(-1)

    def aux(self, g: torch.Tensor) -> torch.Tensor:
        return self.fc_aux(g)


def _gather_cell(feat: torch.Tensor, loc: torch.Tensor) -> torch.Tensor:
    """feat [B,C,28,28], loc [B] flattened x*28+y -> [B,C] cell features."""
    bsz, ch = feat.shape[0], feat.shape[1]
    flat = feat.reshape(bsz, ch, N_LOCS)
    return flat.gather(2, loc[:, None, None].expand(bsz, ch, 1)).squeeze(2)


def _masked_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Entropy of a masked categorical given its log-probs ([B,K] -> [B]).
    Cells masked to ~-1e9 carry ~0 probability and contribute ~0."""
    p = log_probs.exp()
    return -(p * log_probs.clamp(min=_NEG)).sum(dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
