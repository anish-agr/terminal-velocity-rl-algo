"""Decision procedure (ARCHITECTURE §5) — identical in actors and deployment.

Per decision:
  1. Sample K own plans from the policy decoder (+ the greedy plan + the
     all-defense plan), dedup -> K_eff.
  2. Sample M opponent plans from the prediction head (+ the opponent's literal
     previous plan re-legalized + the empty plan), dedup -> M_eff, with
     prediction log-probs.
  3. Fork the sim for every (i, j), play the joint turn, evaluate the value head
     on every resulting state in ONE batch.
  4. score_i = lambda * sum_j w_j v_ij + (1 - lambda) * min_j v_ij,
     w_j proportional to p_j^0.5 (temperature-flattened prediction probs).
  5. a* = argmax score; expert target pi* = softmax(score / tau_tgt) (§5.3).

The net lives behind a NetClient with three calls (mirroring the §5.5 server
request types): sample_plans / score_plans / values. LocalNetClient runs the
same code in-process (deployment and tests); the shared-memory client in
actor.py speaks to infer_server.py. The train/deploy split happens HERE and
only here.

Anytime budget (§9.3): with budget_s set, choose() starts at (k_start, m_start)
and doubles K then M while elapsed time allows, always keeping the best
completed round's answer. The first own candidate scored is always the
all-defense plan, so a submittable answer exists as soon as round one finishes.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import DeployHistory, build_planes
from .tokens import (
    DEPLOY_TYPES, END, END_TOKEN, Token,
    PlanScratch, ScratchSpec, decode_commands, encode_plan,
)

Plan = Tuple[Token, ...]
Command = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Net client protocol + local implementation
# ---------------------------------------------------------------------------

class NetClient:
    """The three §5.5 request types. Implementations: LocalNetClient (below),
    SharedMemClient (actor.py)."""

    def sample_plans(self, board, scalars, scratch_factory, k, tau, head,
                     greedy_extra=False, mask_deploys_extra=False):
        """-> list of (plan, seq_logp). head in {"policy", "predict"}."""
        raise NotImplementedError

    def score_plans(self, board, scalars, scratch_factory, plans, head):
        """-> [len(plans)] sequence log-probs under `head` (teacher-forced)."""
        raise NotImplementedError

    def values(self, boards, scalars):
        """[N,18,28,28], [N,14] -> [N] value-head outputs."""
        raise NotImplementedError


class LocalNetClient(NetClient):
    """Runs the net in-process on CPU/GPU. Used by deployment and tests; the
    GPU inference server wraps the same three routines."""

    def __init__(self, net, device="cpu"):
        import torch  # local import: deploy fallback must not need torch at module load

        self._torch = torch
        self.net = net.to(device).eval()
        self.device = device

    def _torso(self, board, scalars):
        t = self._torch
        b = t.from_numpy(np.ascontiguousarray(board[None])).to(self.device)
        s = t.from_numpy(np.ascontiguousarray(scalars[None])).to(self.device)
        with t.no_grad():
            feat, g = self.net.forward_torso(b, s)
        return feat, g

    def sample_plans(self, board, scalars, scratch_factory, k, tau, head,
                     greedy_extra=False, mask_deploys_extra=False):
        with self._torch.no_grad():
            feat, g = self._torso(board, scalars)
            decoder = self.net.policy if head == "policy" else self.net.predict
            out = []
            for i in range(k):
                out.append(_sample_one(self._torch, decoder, feat, g,
                                        scratch_factory(), tau, greedy=False))
            if greedy_extra:
                out.append(_sample_one(self._torch, decoder, feat, g,
                                        scratch_factory(), tau, greedy=True))
            if mask_deploys_extra:
                out.append(_sample_one(self._torch, decoder, feat, g,
                                        scratch_factory(), tau, greedy=True,
                                        mask_deploys=True))
        return out

    def score_plans(self, board, scalars, scratch_factory, plans, head):
        with self._torch.no_grad():
            feat, g = self._torso(board, scalars)
            decoder = self.net.policy if head == "policy" else self.net.predict
            return [
                _score_one(self._torch, decoder, feat, g, scratch_factory(), plan)
                for plan in plans
            ]

    def values(self, boards, scalars):
        t = self._torch
        b = t.from_numpy(np.ascontiguousarray(boards)).to(self.device)
        s = t.from_numpy(np.ascontiguousarray(scalars)).to(self.device)
        with t.no_grad():
            _, g = self.net.forward_torso(b, s)
            v = self.net.value(g)
        return v.cpu().numpy()


def _masked_categorical(torch, logits, mask, tau, greedy):
    logits = logits.masked_fill(~torch.from_numpy(mask[None]).to(logits.device), -1e9)
    if greedy:
        return int(logits.argmax(dim=1)), logits
    return int(torch.distributions.Categorical(logits=logits / tau).sample()), logits


def _sample_one(torch, decoder, feat, g, scratch, tau, greedy=False,
                mask_deploys=False):
    """Autoregressively sample one plan against live legality masks.
    Returns (plan tuple ending in END_TOKEN, sequence log-prob at tau=1)."""
    import torch.nn.functional as F_t

    c, keys = decoder.init(feat, g)
    plan: List[Token] = []
    logp = 0.0
    for _ in range(23):  # T_MAX - 1 real tokens, then forced END
        tmask = scratch.type_mask()
        if mask_deploys:
            for t in DEPLOY_TYPES:
                tmask[t] = False
        ttype, lt = _masked_categorical(torch, decoder.type_logits(c), tmask, tau, greedy)
        logp += float(F_t.log_softmax(lt, dim=1)[0, ttype])
        if ttype == END:
            break
        lmask = scratch.loc_mask(ttype)
        loc, ll = _masked_categorical(torch, decoder.loc_logits(c, keys), lmask, tau, greedy)
        logp += float(F_t.log_softmax(ll, dim=1)[0, loc])
        cmask = scratch.count_mask(ttype)
        loc_t = torch.tensor([loc], device=feat.device)
        count, lc = _masked_categorical(
            torch, decoder.count_logits(c, feat, loc_t), cmask, tau, greedy
        )
        if ttype in DEPLOY_TYPES:
            logp += float(F_t.log_softmax(lc, dim=1)[0, count])
        else:
            count = 0
        tok = Token(ttype, loc, count)
        scratch.apply(tok)
        plan.append(tok)
        c = decoder.advance(
            c, feat,
            torch.tensor([ttype], device=feat.device),
            loc_t,
            torch.tensor([count], device=feat.device),
        )
    plan.append(END_TOKEN)
    return tuple(plan), logp


def _score_one(torch, decoder, feat, g, scratch, plan):
    """Teacher-forced sequence log-prob of `plan`, masking exactly as sampling
    would. Illegal-under-current-state tokens are skipped (re-legalization)."""
    import torch.nn.functional as F_t

    c, keys = decoder.init(feat, g)
    logp = 0.0
    for tok in plan:
        tmask = scratch.type_mask()
        lt = decoder.type_logits(c).masked_fill(
            ~torch.from_numpy(tmask[None]).to(feat.device), -1e9
        )
        if not tmask[tok.type]:
            continue  # token impossible here; skip rather than poison the score
        logp += float(F_t.log_softmax(lt, dim=1)[0, tok.type])
        if tok.type == END:
            break
        lmask = scratch.loc_mask(tok.type)
        if not lmask[tok.loc]:
            continue
        ll = decoder.loc_logits(c, keys).masked_fill(
            ~torch.from_numpy(lmask[None]).to(feat.device), -1e9
        )
        logp += float(F_t.log_softmax(ll, dim=1)[0, tok.loc])
        if tok.type in DEPLOY_TYPES:
            cmask = scratch.count_mask(tok.type)
            if not cmask[tok.count]:
                continue
            lc = decoder.count_logits(
                c, feat, torch.tensor([tok.loc], device=feat.device)
            ).masked_fill(~torch.from_numpy(cmask[None]).to(feat.device), -1e9)
            logp += float(F_t.log_softmax(lc, dim=1)[0, tok.count])
        scratch.apply(tok)
        c = decoder.advance(
            c, feat,
            torch.tensor([tok.type], device=feat.device),
            torch.tensor([tok.loc], device=feat.device),
            torch.tensor([tok.count], device=feat.device),
        )
    return logp


# ---------------------------------------------------------------------------
# Re-legalization of a remembered opponent plan (§5.1)
# ---------------------------------------------------------------------------

def relegalize(plan: Sequence[Token], scratch: PlanScratch) -> Plan:
    """Drop tokens that are illegal in the current state, keep the rest."""
    out: List[Token] = []
    for tok in plan:
        if tok.type == END:
            break
        try:
            scratch.apply(tok)
        except ValueError:
            continue
        out.append(tok)
    out.append(END_TOKEN)
    return tuple(out)


# ---------------------------------------------------------------------------
# choose() — the §5 decision
# ---------------------------------------------------------------------------

def choose(
    game,
    net_client: NetClient,
    cfg: dict,
    player: int,
    hist_own: DeployHistory,
    hist_opp: DeployHistory,
    config: dict,
    costs,
    prev_opp_plan: Optional[Plan] = None,
    k: Optional[int] = None,
    m: Optional[int] = None,
    tau: Optional[float] = None,
    budget_s: Optional[float] = None,
    pathfind: Optional[Callable] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Plan, Dict[Plan, float], dict]:
    """-> (chosen plan, pi_star over candidate plans, diagnostics).

    Training actors call this with fixed (k, m, tau); the deployment driver
    passes budget_s to activate the anytime widening ladder (§9.3).
    """
    scfg = cfg["search"]
    lam = float(scfg["lambda_security"])
    wtemp = float(scfg["opponent_weight_temp"])
    tau_tgt = float(scfg["tau_target"])
    early_v = float(scfg["early_exit_v"])
    k = int(k if k is not None else scfg["k_train"])
    m = int(m if m is not None else scfg["m_train"])
    tau = float(tau if tau is not None else scfg["tau_act_start"])

    t0 = time.monotonic()
    opp = 1 - player
    flip_own = player == 1
    flip_opp = opp == 1

    board_own, scal_own = build_planes(game, player, hist_own)
    board_opp, scal_opp = build_planes(game, opp, hist_opp)

    structures = game.structures()
    sp_o, mp_o = game.stats(player)[1], game.stats(player)[2]
    sp_e, mp_e = game.stats(opp)[1], game.stats(opp)[2]

    scratch_own = ScratchSpec(costs, structures, sp_o, mp_o, flip_own, player,
                              pathfind=pathfind)
    scratch_opp = ScratchSpec(costs, structures, sp_e, mp_e, flip_opp, opp,
                              pathfind=pathfind)

    best: Optional[Tuple[Plan, Dict[Plan, float], dict]] = None
    k_cur, m_cur = (int(cfg["deployment"]["k_start"]), int(cfg["deployment"]["m_start"])) \
        if budget_s else (k, m)

    while True:
        result = _one_round(
            game, net_client, player, board_own, scal_own, board_opp, scal_opp,
            scratch_own, scratch_opp, hist_own, config, prev_opp_plan,
            k_cur, m_cur, tau, lam, wtemp, tau_tgt,
        )
        best = result
        elapsed = time.monotonic() - t0
        result[2]["elapsed_s"] = elapsed
        if budget_s is None:
            break
        if result[2]["max_min_v"] > early_v:      # §5.4 early exit when winning
            break
        if k_cur < k and elapsed < budget_s / 2:
            k_cur = min(2 * k_cur, k)
            continue
        if m_cur < m and elapsed < budget_s / 2:
            m_cur = min(2 * m_cur, m)
            continue
        break
    return best


def _one_round(
    game, net_client, player, board_own, scal_own, board_opp, scal_opp,
    scratch_own, scratch_opp, hist_own, config, prev_opp_plan,
    k, m, tau, lam, wtemp, tau_tgt,
):
    opp = 1 - player

    # --- own candidates: K sampled + greedy + all-defense, deduped ----------
    sampled = net_client.sample_plans(
        board_own, scal_own, scratch_own, k, tau, "policy",
        greedy_extra=True, mask_deploys_extra=True,
    )
    own_plans: List[Plan] = []
    seen = set()
    all_defense_first: List[Plan] = []
    for idx, (plan, _lp) in enumerate(sampled):
        if plan in seen:
            continue
        seen.add(plan)
        # the all-defense plan (last extra) is scored FIRST (§9.3 anytime rule)
        (all_defense_first if idx == len(sampled) - 1 else own_plans).append(plan)
    own_plans = all_defense_first + own_plans

    # --- opponent candidates: M sampled + literal previous + empty ----------
    osampled = net_client.sample_plans(
        board_opp, scal_opp, scratch_opp, m, 1.0, "predict",
    )
    opp_plans: List[Plan] = []
    opp_logps: List[float] = []
    oseen = set()
    for plan, lp in osampled:
        if plan in oseen:
            continue
        oseen.add(plan)
        opp_plans.append(plan)
        opp_logps.append(lp)
    extras = [tuple([END_TOKEN])]                       # the empty plan
    if prev_opp_plan is not None:
        extras.insert(0, relegalize(prev_opp_plan, scratch_opp()))
    for extra in extras:
        if extra not in oseen:
            oseen.add(extra)
            opp_plans.append(extra)
    if len(opp_logps) < len(opp_plans):                  # score the extras
        extra_lps = net_client.score_plans(
            board_opp, scal_opp, scratch_opp,
            opp_plans[len(opp_logps):], "predict",
        )
        opp_logps.extend(float(x) for x in extra_lps)

    # w_j proportional to p_j^0.5  ==  softmax(0.5 * logp_j)
    lw = wtemp * np.array(opp_logps, dtype=np.float64)
    lw -= lw.max()
    w = np.exp(lw)
    w /= w.sum()

    # --- K_eff x M_eff joint rollouts, one value batch ----------------------
    n_i, n_j = len(own_plans), len(opp_plans)
    boards = np.empty((n_i * n_j, board_own.shape[0], 28, 28), dtype=np.float32)
    scals = np.empty((n_i * n_j, scal_own.shape[0]), dtype=np.float32)
    for i, pi in enumerate(own_plans):
        own_cmds = encode_plan(list(pi), scratch_own())
        for j, pj in enumerate(opp_plans):
            opp_cmds = encode_plan(list(pj), scratch_opp())
            f = game.fork()
            if player == 0:
                res = f.play_turn(own_cmds, opp_cmds)
            else:
                res = f.play_turn(opp_cmds, own_cmds)
            frames, b1, b2, d1, d2 = res
            hist_sim = _advance_history(hist_own, config, opp_cmds, player, res)
            b, s = build_planes(f, player, hist_sim)
            boards[i * n_j + j] = b
            scals[i * n_j + j] = s

    v = np.asarray(net_client.values(boards, scals), dtype=np.float64)
    v = v.reshape(n_i, n_j)

    scores = lam * (v * w[None, :]).sum(axis=1) + (1.0 - lam) * v.min(axis=1)
    a_star = int(np.argmax(scores))

    z = (scores - scores.max()) / tau_tgt
    pi = np.exp(z)
    pi /= pi.sum()
    pi_star = {plan: float(p) for plan, p in zip(own_plans, pi)}

    diag = {
        "k_eff": n_i,
        "m_eff": n_j,
        "values": v,
        "opp_weights": w,
        "scores": scores,
        "max_min_v": float(v.min(axis=1).max()),
        "chosen_idx": a_star,
    }
    return own_plans[a_star], pi_star, diag


def _advance_history(hist: DeployHistory, config, opp_cmds, player, turn_result):
    """A simulated turn's history planes: copy the live history and record the
    opponent deploys + damage flows the simulated turn produced."""
    frames, b1, b2, d1, d2 = turn_result
    h = DeployHistory(config)
    h.ema = hist.ema.copy()
    h.last = hist.last.copy()
    own_breach = b1 if player == 0 else b2
    opp_breach = b2 if player == 0 else b1
    own_dmg = d1 if player == 0 else d2
    opp_dmg = d2 if player == 0 else d1
    h.record_turn(
        [c for c in opp_cmds if c[0] in (3, 4, 5)],
        own_breach, opp_breach, own_dmg, opp_dmg,
    )
    return h
