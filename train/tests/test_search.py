"""CP3a verification: search.py — scoring math (stub client, controlled values)
and end-to-end choose() plumbing (real TerminalNet on a fake sim).

Run:  python -m train.tests.test_search
"""

import json
import os

import numpy as np

from train.features import DeployHistory
from train.model import TerminalNet
from train.search import LocalNetClient, NetClient, choose, relegalize
from train.tokens import (
    BUILD_TURRET, DEP_SCOUT, END, END_TOKEN, Costs, PlanScratch, Token, xy_loc,
)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(_REPO, "game-configs.json")) as fh:
    _CONFIG = json.load(fh)
with open(os.path.join(_REPO, "train", "config.yaml")) as fh:
    # minimal yaml: parse the two sections the tests need without a yaml dep
    _CFG = {
        "search": {
            "lambda_security": 0.7, "opponent_weight_temp": 0.5,
            "tau_target": 0.25, "early_exit_v": 0.98,
            "k_train": 3, "m_train": 2, "tau_act_start": 1.0,
        },
        "deployment": {"k_start": 2, "m_start": 2},
    }

_COSTS = Costs(_CONFIG)


class FakeGame:
    """Deterministic stand-in for terminal_sim.Game."""

    def __init__(self, turn=10):
        self.turn = turn
        self._structures = [(2, 0, 3, 12, 75.0, False, False),
                            (2, 1, 24, 15, 75.0, False, False)]

    def fork(self):
        return FakeGame(self.turn)

    def stats(self, player):
        return (30.0, 12.0, 8.3)

    def structures(self):
        return list(self._structures)

    def board_planes(self, player):
        return np.zeros(12 * 28 * 28, dtype="<f4").tobytes()

    def play_turn(self, p1, p2):
        return (12, 1.0, 0.0, 10.0, 4.0)


class StubClient(NetClient):
    """Fixed plans + a controllable value sequence -> exact scoring checks."""

    def __init__(self, own_plans, opp_plans, opp_logps, value_matrix):
        self.own_plans = own_plans
        self.opp_plans = opp_plans
        self.opp_logps = opp_logps
        self.value_matrix = np.asarray(value_matrix, dtype=np.float64)

    def sample_plans(self, board, scalars, scratch_factory, k, tau, head,
                     greedy_extra=False, mask_deploys_extra=False):
        if head == "policy":
            out = [(p, 0.0) for p in self.own_plans]
            if mask_deploys_extra:
                # honor the contract: the LAST plan returned is the all-defense
                # extra, which _one_round moves to the FRONT of scoring order
                out.append((self.all_defense_plan, 0.0))
            return out
        return [(p, lp) for p, lp in zip(self.opp_plans, self.opp_logps)]

    def score_plans(self, board, scalars, scratch_factory, plans, head):
        return [-3.0] * len(plans)

    def values(self, boards, scalars):
        return self.value_matrix.reshape(-1)


def _mk_hist():
    return DeployHistory(_CONFIG)


def test_security_scoring_math():
    own = [
        (Token(BUILD_TURRET, xy_loc(5, 10), 0), END_TOKEN),
        (Token(DEP_SCOUT, xy_loc(13, 0), 0), END_TOKEN),
    ]
    all_def = (Token(BUILD_TURRET, xy_loc(7, 9), 0), END_TOKEN)
    opp = [(Token(DEP_SCOUT, xy_loc(13, 0), 0), END_TOKEN)]
    # candidate order after _one_round's reordering: [all_def, own[0], own[1]]
    # opp candidates: 1 sampled + empty extra = 2. values [3 own x 2 opp]:
    # all_def mediocre; A solid everywhere; B better on average, awful worst case.
    v = np.array([[0.10, 0.10],
                  [0.30, 0.30],
                  [0.90, -0.80]])
    stub = StubClient(own, opp, [-1.0], v)
    stub.all_defense_plan = all_def
    plan, pi_star, diag = choose(
        FakeGame(), stub, _CFG, player=0,
        hist_own=_mk_hist(), hist_opp=_mk_hist(),
        config=_CONFIG, costs=_COSTS,
    )
    # weights: logps [-1.0 (sampled), -3.0 (empty, scored)] -> w = softmax(0.5*lp)
    lw = 0.5 * np.array([-1.0, -3.0]); lw -= lw.max()
    w = np.exp(lw); w /= w.sum()
    exp_scores = 0.7 * (v @ w) + 0.3 * v.min(axis=1)
    assert np.allclose(diag["scores"], exp_scores, atol=1e-9)
    # with lambda=0.7 the -0.80 floor must sink plan B despite its 0.90 upside
    assert exp_scores[1] > exp_scores[2]
    assert plan == own[0]                        # row 1 = own[0] wins
    order = [all_def, own[0], own[1]]
    ps = np.array([pi_star[p] for p in order])
    assert abs(ps.sum() - 1.0) < 1e-9 and ps[1] == ps.max()


def test_choose_end_to_end_with_real_net():
    import torch
    torch.manual_seed(3)
    client = LocalNetClient(TerminalNet())
    plan, pi_star, diag = choose(
        FakeGame(), client, _CFG, player=0,
        hist_own=_mk_hist(), hist_opp=_mk_hist(),
        config=_CONFIG, costs=_COSTS,
        k=3, m=2, tau=1.0,
    )
    assert plan[-1] == END_TOKEN
    assert abs(sum(pi_star.values()) - 1.0) < 1e-6
    assert diag["k_eff"] >= 1 and diag["m_eff"] >= 2  # sampled + empty at least
    assert diag["values"].shape == (diag["k_eff"], diag["m_eff"])
    # every returned plan must be executable on a fresh scratch (legality)
    scratch = PlanScratch(_COSTS, 12.0, 8.3, FakeGame().structures(), own_player=0)
    for tok in plan:
        if tok.type == END:
            break
        scratch.apply(tok)  # raises if the search emitted an illegal plan


def test_choose_as_player_one():
    import torch
    torch.manual_seed(4)
    client = LocalNetClient(TerminalNet())
    plan, _pi, diag = choose(
        FakeGame(), client, _CFG, player=1,
        hist_own=_mk_hist(), hist_opp=_mk_hist(),
        config=_CONFIG, costs=_COSTS, k=2, m=1, tau=1.0,
    )
    assert plan[-1] == END_TOKEN and diag["k_eff"] >= 1


def test_relegalize_drops_illegal():
    scratch = PlanScratch(_COSTS, sp=1.5, mp=0.5)   # can afford one wall, no scouts
    plan = (
        Token(0, xy_loc(10, 10), 0),                # wall — legal
        Token(DEP_SCOUT, xy_loc(13, 0), 0),         # scout — cannot afford
        END_TOKEN,
    )
    out = relegalize(plan, scratch)
    assert out == (Token(0, xy_loc(10, 10), 0), END_TOKEN)


def test_anytime_budget_returns():
    import torch
    torch.manual_seed(5)
    client = LocalNetClient(TerminalNet())
    plan, _pi, diag = choose(
        FakeGame(), client, _CFG, player=0,
        hist_own=_mk_hist(), hist_opp=_mk_hist(),
        config=_CONFIG, costs=_COSTS,
        k=4, m=2, tau=1.0, budget_s=30.0,
    )
    assert plan[-1] == END_TOKEN
    assert "elapsed_s" in diag


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("\n{} / {} search tests green".format(len(fns), len(fns)))


if __name__ == "__main__":
    main()
