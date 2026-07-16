"""Learner (ARCHITECTURE §6): replay buffer, losses, optimization, checkpoints.

Losses per position (§6.1):
    L_policy   = - sum_i pi*(i) log p_theta(a_i | s)   (sequence NLL, K_eff cands)
    L_value    = (v_theta(s) - z)^2
    L_aux      = || aux_theta(s) - Delta_3(s) ||^2      (weight annealed §6.3)
    L_predict  = - log p_phi(b | s_opp)                 (opponent's actual plan)
    entropy    = - c_e * H(decoder dists)               (decays to 0 by mid-run)

Legality masks for every NLL are REBUILT at batch time from each position's
stored scratch ingredients (structures + sp/mp + flip) — they cannot be stored
(9+784+8 bools x T x K_eff per position) and must match sampling exactly.

Mirror augmentation happens at sample time with p=0.5: board planes x-flipped,
every plan token loc-remapped, structures x-mirrored in absolute coords (the
perspective flip commutes with the x-mirror, so one absolute mirror serves both
seats). Scalars are mirror-invariant.

Memory note: policy NLL touches B x K_eff sequences whose gathered feature maps
would OOM in one shot at B=1024 — sequences are processed in micro-batches with
gradient accumulation inside one optimizer step.
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import mirror_board
from .league import League
from .tokens import (
    Costs, DEPLOY_TYPES, END, N_BUCKETS, N_LOCS, N_TYPES, T_MAX, Token,
    PlanScratch, mirror_plan,
)


# ---------------------------------------------------------------------------
# Mask building (shared by policy + prediction losses)
# ---------------------------------------------------------------------------

def plan_masks(
    plan: Sequence[Token], scratch: PlanScratch, t_max: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-step legality masks for teacher forcing, replaying the scratch.

    Returns (type[T,9], loc[T,784], count[T,8], length). Steps beyond the plan
    are all-True padding (gated off by length in plan_nll). A token illegal
    under the rebuilt scratch (possible for opponent plans whose executed spend
    the 0.1 affordability margin rejects) ends the sequence at that step —
    length counts only the LEGAL prefix and is 0 when the very first token is
    illegal. Callers must gate such rows out (lengths=0 makes plan_nll emit
    exactly 0 for the row); feeding the illegal token itself would gather a
    -1e9-masked logit and poison the loss.
    """
    tm = np.ones((t_max, N_TYPES), dtype=bool)
    lm = np.ones((t_max, N_LOCS), dtype=bool)
    cm = np.ones((t_max, N_BUCKETS), dtype=bool)
    length = 0
    for tok in plan[:t_max]:
        t = length
        tm[t] = scratch.type_mask()
        if not tm[t][tok.type]:
            break
        if tok.type == END:
            length += 1
            break
        lm[t] = scratch.loc_mask(tok.type)
        cm[t] = scratch.count_mask(tok.type)
        if not lm[t][tok.loc] or not cm[t][tok.count]:
            break
        scratch.apply(tok)
        length += 1
    return tm, lm, cm, length


def _mirror_structures(structures):
    return tuple((k, o, 27 - x, y, hp, up, pend)
                 for (k, o, x, y, hp, up, pend) in structures)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, min_fill: int):
        self.buf: deque = deque(maxlen=capacity)
        self.min_fill = min_fill
        self.total_added = 0

    def add_many(self, positions: List[dict]) -> None:
        self.buf.extend(positions)
        self.total_added += len(positions)

    def ready(self) -> bool:
        return len(self.buf) >= self.min_fill

    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, batch: int, rng: np.random.Generator) -> List[dict]:
        idx = rng.integers(len(self.buf), size=batch)
        out = []
        for i in idx:
            pos = self.buf[int(i)]
            if rng.random() < 0.5:
                pos = self._mirror(pos)
            out.append(pos)
        return out

    @staticmethod
    def _mirror(pos: dict) -> dict:
        m = dict(pos)
        m["board"] = mirror_board(pos["board"])
        m["opp_board"] = mirror_board(pos["opp_board"])
        m["candidates"] = [tuple(mirror_plan(p)) for p in pos["candidates"]]
        m["opp_plan"] = tuple(mirror_plan(pos["opp_plan"]))
        m["structures"] = _mirror_structures(pos["structures"])
        m["opp_structures"] = _mirror_structures(pos["opp_structures"])
        return m


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------

class Learner:
    def __init__(self, cfg: dict, config: dict, device: str = "cpu",
                 seed: int = 0):
        import torch

        from .model import TerminalNet

        self.torch = torch
        self.cfg = cfg
        self.config = config
        self.costs = Costs(config)
        self.device = device
        self.rng = np.random.default_rng(seed)

        lc = cfg["learning"]
        self.net = TerminalNet().to(device)
        self.opt = torch.optim.AdamW(
            self.net.parameters(), lr=float(lc["lr"]),
            weight_decay=float(lc["weight_decay"]),
        )
        self.lr0 = float(lc["lr"])
        self.lr1 = float(lc["lr_end"])
        self.total_steps = int(lc["total_steps"])
        self.batch_size = int(lc["batch_size"])
        self.micro_seq = int(lc["micro_batch_sequences"])
        self.grad_clip = float(lc["grad_clip"])
        self.w_value = float(lc["loss_weights"]["value"])
        self.w_aux0 = float(lc["loss_weights"]["aux_start"])
        self.w_aux1 = float(lc["loss_weights"]["aux_end"])
        self.w_pred = float(lc["loss_weights"]["predict"])
        self.c_e0 = float(lc["entropy_coef"])
        self.ent_end = float(lc["entropy_end_frac"])
        # policy/prediction gradients flow through the torso (correct per the
        # ExIt formulation); flip OFF only if pod memory profiling forces it
        self.policy_through_torso = bool(lc.get("policy_through_torso", True))

        self.buffer = ReplayBuffer(int(lc["buffer_capacity"]),
                                   int(lc["buffer_min_fill"]))
        self.league = League(cfg)
        self.step_count = 0

    # -- schedules (§6.2, §6.3) ------------------------------------------------

    def _progress(self) -> float:
        return min(1.0, self.step_count / max(1, self.total_steps))

    def lr_now(self) -> float:
        p = self._progress()
        return self.lr1 + 0.5 * (self.lr0 - self.lr1) * (1 + math.cos(math.pi * p))

    def aux_weight(self) -> float:
        p = self._progress()
        if p < 0.25:
            return self.w_aux0
        if p < 0.50:
            frac = (p - 0.25) / 0.25
            return self.w_aux0 + frac * (self.w_aux1 - self.w_aux0)
        return self.w_aux1

    def entropy_coef(self) -> float:
        p = self._progress()
        return self.c_e0 * max(0.0, 1.0 - p / self.ent_end)

    # -- ingestion ----------------------------------------------------------------

    def ingest(self, meta: dict, positions: List[dict]) -> None:
        self.buffer.add_many(positions)
        if meta.get("opponent_kind") == "snapshot" and meta.get("winner", -1) >= 0:
            current_won = meta["winner"] == meta.get("me", 0)
            self.league.report_result(meta["opponent"], current_won)

    # -- one gradient step -----------------------------------------------------

    def train_step(self) -> Optional[Dict[str, float]]:
        if not self.buffer.ready():
            return None
        torch = self.torch
        batch = self.buffer.sample(self.batch_size, self.rng)

        for g in self.opt.param_groups:
            g["lr"] = self.lr_now()
        self.opt.zero_grad(set_to_none=True)

        boards = torch.from_numpy(np.stack([p["board"] for p in batch])).to(self.device)
        scalars = torch.from_numpy(np.stack([p["scalars"] for p in batch])).to(self.device)
        z = torch.tensor([p["z"] for p in batch], dtype=torch.float32,
                         device=self.device)
        aux_t = torch.from_numpy(np.stack([p["aux"] for p in batch])).to(self.device)

        feat, g_vec = self.net.forward_torso(boards, scalars)
        v = self.net.value(g_vec)
        loss_value = ((v - z) ** 2).mean()
        loss_aux = ((self.net.aux(g_vec) - aux_t) ** 2).sum(dim=1).mean()

        # scalar heads contribute once; the torso graph stays alive so the NLL
        # chunks below can push policy/prediction gradients through it too
        head_loss = self.w_value * loss_value + self.aux_weight() * loss_aux
        head_loss.backward(retain_graph=self.policy_through_torso)

        # ---- policy NLL over (position, candidate) sequences ----------------
        seqs = []
        for b_idx, pos in enumerate(batch):
            flip = pos["side"] == 1
            for plan, weight in zip(pos["candidates"], pos["pi"]):
                if weight < 1e-4:
                    continue
                seqs.append((b_idx, plan, weight, "policy", flip,
                             pos["structures"], pos["sp"], pos["mp"], pos["side"]))
        for b_idx, pos in enumerate(batch):
            opp = 1 - pos["side"]
            seqs.append((b_idx, pos["opp_plan"], 1.0, "predict", opp == 1,
                         pos["opp_structures"], pos["opp_sp"], pos["opp_mp"], opp))

        c_e = self.entropy_coef()
        pol_sum = pred_sum = ent_sum = 0.0
        n_pol = n_pred = 0

        for lo in range(0, len(seqs), self.micro_seq):
            chunk = seqs[lo: lo + self.micro_seq]
            t_len = max(2, max(len(s[1]) for s in chunk))
            t_len = min(t_len, T_MAX)

            plans_t = torch.zeros(len(chunk), t_len, 3, dtype=torch.long)
            lengths = torch.zeros(len(chunk), dtype=torch.long)
            tm = torch.ones(len(chunk), t_len, N_TYPES, dtype=torch.bool)
            lm = torch.ones(len(chunk), t_len, N_LOCS, dtype=torch.bool)
            cm = torch.ones(len(chunk), t_len, N_BUCKETS, dtype=torch.bool)
            pos_idx = torch.zeros(len(chunk), dtype=torch.long)
            weights = torch.zeros(len(chunk), dtype=torch.float32)
            is_policy = torch.zeros(len(chunk), dtype=torch.bool)

            for i, (b_idx, plan, w, head, flip, structures, sp, mp, owner) in \
                    enumerate(chunk):
                scratch = PlanScratch(self.costs, sp, mp, structures,
                                      flip=flip, own_player=owner)
                tmk, lmk, cmk, length = plan_masks(plan, scratch, t_len)
                tm[i] = torch.from_numpy(tmk)
                lm[i] = torch.from_numpy(lmk)
                cm[i] = torch.from_numpy(cmk)
                # length 0 = plan illegal from token 0 under the rebuilt
                # scratch; lengths=0 makes plan_nll contribute exactly 0 for
                # this row (never gather the -1e9-masked illegal token)
                lengths[i] = length
                for t, tok in enumerate(plan[:length]):
                    plans_t[i, t, 0] = tok.type
                    plans_t[i, t, 1] = tok.loc
                    plans_t[i, t, 2] = tok.count
                pos_idx[i] = b_idx
                weights[i] = w
                is_policy[i] = head == "policy"

            dev = self.device
            src_f = feat if self.policy_through_torso else feat.detach()
            src_g = g_vec if self.policy_through_torso else g_vec.detach()
            feat_c = src_f[pos_idx.to(self.device)]
            g_c = src_g[pos_idx.to(self.device)]

            nll_p, ent_p = self.net.policy.plan_nll(
                feat_c, g_c, plans_t.to(dev), lengths.to(dev),
                tm.to(dev), lm.to(dev), cm.to(dev),
            )
            nll_q, _ = self.net.predict.plan_nll(
                feat_c, g_c, plans_t.to(dev), lengths.to(dev),
                tm.to(dev), lm.to(dev), cm.to(dev),
            )
            w_t = weights.to(dev)
            pol_mask = is_policy.to(dev)
            loss_pol = (nll_p * w_t * pol_mask).sum() / max(1, int(pol_mask.sum()))
            loss_pred = (nll_q * (~pol_mask).float()).sum() / max(
                1, int((~pol_mask).sum()))
            loss_ent = -(ent_p * pol_mask).sum() / max(1, int(pol_mask.sum()))
            last_chunk = lo + self.micro_seq >= len(seqs)
            (loss_pol + self.w_pred * loss_pred + c_e * loss_ent).backward(
                retain_graph=self.policy_through_torso and not last_chunk
            )

            pol_sum += float(loss_pol.detach()) * int(pol_mask.sum())
            pred_sum += float(loss_pred.detach()) * int((~pol_mask).sum())
            ent_sum += float(-loss_ent.detach()) * int(pol_mask.sum())
            n_pol += int(pol_mask.sum())
            n_pred += int((~pol_mask).sum())

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.opt.step()
        self.step_count += 1

        return {
            "step": self.step_count,
            "lr": self.lr_now(),
            "loss_value": float(loss_value.detach()),
            "loss_aux": float(loss_aux.detach()),
            "loss_policy": pol_sum / max(1, n_pol),
            "loss_predict": pred_sum / max(1, n_pred),
            "entropy": ent_sum / max(1, n_pol),
            "buffer": len(self.buffer),
        }

    # -- persistence -------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        torch = self.torch
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        torch.save(
            {
                "net": self.net.state_dict(),
                "opt": self.opt.state_dict(),
                "step": self.step_count,
                "buffer_total": self.buffer.total_added,
            },
            tmp,
        )
        os.replace(tmp, path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = self.torch.load(path, map_location="cpu")
        self.net.load_state_dict(ckpt["net"])
        self.opt.load_state_dict(ckpt["opt"])
        self.step_count = int(ckpt["step"])

    def export_weights(self, path: str) -> None:
        """Weights-only snapshot for inference-server hot reload / league."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        self.torch.save(self.net.state_dict(), tmp)
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Learner process
# ---------------------------------------------------------------------------

def run_learner(
    trajectory_q,
    control_q,
    cfg: dict,
    config: dict,
    run_dir: str,
    device: str = "cpu",
    max_steps: Optional[int] = None,
    steps_per_1k: Optional[float] = None,
    deadline_ts: Optional[float] = None,
) -> None:
    """Consume trajectories, keep the learner:actor ratio (§6.2), snapshot the
    league every snapshot_interval_min, export weights for hot reload. Sends
    ("weights", path) messages on control_q for the inference server.

    deadline_ts: absolute time.time() to stop at (the phase's `hours` budget —
    the pilot phase MUST self-terminate so its gate can be read on schedule).
    Also prints rolling win-rates per opponent kind every 50 games: this is the
    §8 pilot-gate signal ("win-rate rising hour-over-hour") — without it a run
    exposes no strength trend at all, since the full gauntlet never runs
    in-loop."""
    learner = Learner(cfg, config, device=device)
    ratio = steps_per_1k if steps_per_1k is not None else \
        float(cfg["learning"]["steps_per_1k_positions"])
    league_path = os.path.join(run_dir, "league.json")
    weights_path = os.path.join(run_dir, "weights_current.pt")
    reload_s = float(cfg["actors"]["weight_reload_s"])
    snap_s = float(cfg["league"]["snapshot_interval_min"]) * 60.0

    last_reload = last_snap = time.time()
    steps_owed = 0.0
    recent = deque(maxlen=200)   # (opponent_kind, current_won) rolling window
    games = 0

    while (max_steps is None or learner.step_count < max_steps) and \
            (deadline_ts is None or time.time() < deadline_ts):
        try:
            meta, positions = trajectory_q.get(timeout=1.0)
        except Exception:
            continue
        learner.ingest(meta, positions)
        steps_owed += ratio * len(positions) / 1000.0

        winner, me = meta.get("winner", -1), meta.get("me")
        if winner >= 0 and me is not None:
            recent.append((meta.get("opponent_kind", "?"), winner == me))
        games += 1
        if games % 50 == 0 and recent:
            by: Dict[str, List[bool]] = {}
            for kind, won in recent:
                by.setdefault(kind, []).append(won)
            print("winrate[last {}] ".format(len(recent)) + "  ".join(
                "{}: {:.0%} ({})".format(k, sum(v) / len(v), len(v))
                for k, v in sorted(by.items())), flush=True)

        while steps_owed >= 1.0:
            metrics = learner.train_step()
            steps_owed -= 1.0
            if metrics is None:
                steps_owed = 0.0
                break
            if metrics["step"] % 50 == 0:
                print("learner", metrics, flush=True)

        now = time.time()
        if now - last_reload >= reload_s and learner.step_count > 0:
            learner.export_weights(weights_path)
            control_q.put(("reload", "current", weights_path))
            last_reload = now
        if now - last_snap >= snap_s and learner.step_count > 0:
            snap_path = os.path.join(
                run_dir, "snap_{:06d}.pt".format(learner.step_count))
            learner.export_weights(snap_path)
            snap_id = learner.league.add_snapshot(snap_path)
            control_q.put(("load_model", snap_id, snap_path))
            learner.league.save(league_path)
            last_snap = now
        learner.league.save(league_path)
