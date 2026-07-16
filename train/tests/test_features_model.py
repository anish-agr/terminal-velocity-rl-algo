"""CP2 verification: features.py (against a fake Game) + model.py (torch CPU).

Run:  python -m train.tests.test_features_model
"""

import json
import os

import numpy as np
import torch

from train.features import (
    BRIDGE_PLANES, DeployHistory, GRID, N_PLANES, N_SCALARS, build_planes,
    mirror_board,
)
from train.model import PolicyDecoder, TerminalNet, count_parameters
from train.tokens import (
    COUNT_BUCKETS, DEP_SCOUT, END, N_BUCKETS, N_LOCS, N_TYPES, Costs,
    PlanScratch, Token, xy_loc,
)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(_REPO, "game-configs.json")) as fh:
    _CONFIG = json.load(fh)


class FakeGame:
    """Duck-typed stand-in for terminal_sim.Game (the .pyd is corrupted in git;
    the real-bridge integration test runs in the pod bootstrap)."""

    def __init__(self, turn=7):
        self.turn = turn
        self._planes = np.zeros((BRIDGE_PLANES, GRID, GRID), dtype="<f4")
        self._planes[2, 3, 12] = 1.0          # an own turret at full health
        self._planes[10] = 1.0                # in-arena mask (coarse; fine for tests)

    def board_planes(self, player):
        return self._planes.tobytes()

    def stats(self, player):
        return (24.0, 12.0, 6.6) if player == 0 else (18.0, 4.0, 9.1)


def test_history_and_planes():
    h = DeployHistory(_CONFIG)
    # enemy sent 12 scouts at (14,27) and 2 demolishers at (24,17)
    h.record_turn([(3, 14, 27)] * 12 + [(4, 24, 17)] * 2, 2.0, 0.0, 30.0, 5.0)
    board, scalars = build_planes(FakeGame(), 0, h)
    assert board.shape == (N_PLANES, GRID, GRID) and board.dtype == np.float32
    assert scalars.shape == (N_SCALARS,)
    assert board[2, 3, 12] == 1.0                      # bridge plane passthrough
    assert board[12, 14, 27] == 1.0                    # 12 scouts -> 1.2 clamped
    assert abs(board[13, 24, 17] - 0.2) < 1e-6         # 2 demolishers / 10
    assert abs(board[15, 14, 27] - 1.0) < 1e-6         # EMA clamped at 1
    # second, quiet turn: last-turn planes clear, EMA decays by 0.7
    h.record_turn([], 0.0, 1.0, 0.0, 12.0)
    board2, scalars2 = build_planes(FakeGame(), 0, h)
    assert board2[12].sum() == 0.0
    assert abs(board2[15, 14, 27] - 0.7) < 1e-6
    assert abs(scalars2[11] - 1.0 / 5.0) < 1e-6        # breach taken norm
    assert abs(scalars2[13] - 12.0 / 50.0) < 1e-6      # struct dmg taken norm


def test_scalars_math():
    h = DeployHistory(_CONFIG)
    board, s = build_planes(FakeGame(turn=25), 0, h)
    assert abs(s[0] - 24.0 / 30.0) < 1e-6
    assert abs(s[6] - 25.0 / 100.0) < 1e-6
    # income at turn 25: 5 + floor(25/10)*1 = 7
    assert abs(s[7] - 7.0 / 10.0) < 1e-6
    # banked own: 6.6*0.75 + income(26)=7 -> 11.95 / 15
    assert abs(s[8] - (6.6 * 0.75 + 7.0) / 15.0) < 1e-5
    # perspective flip: history planes rotate 180 for player 1
    h.record_turn([(3, 14, 27)], 0, 0, 0, 0)
    b1, _ = build_planes(FakeGame(), 1, h)
    assert b1[12, 13, 0] == 0.1 and b1[12, 14, 27] == 0.0


def test_mirror_board():
    h = DeployHistory(_CONFIG)
    h.record_turn([(4, 24, 17)], 0, 0, 0, 0)
    board, _ = build_planes(FakeGame(), 0, h)
    m = mirror_board(board)
    assert m[13, 3, 17] == board[13, 24, 17]           # x -> 27-x
    assert np.array_equal(mirror_board(m), board)      # involution


def test_net_shapes_and_size():
    net = TerminalNet()
    n = count_parameters(net)
    assert 500_000 < n < 4_000_000, n
    board = torch.randn(3, N_PLANES, GRID, GRID)
    scal = torch.randn(3, N_SCALARS)
    feat, g = net.forward_torso(board, scal)
    assert feat.shape == (3, 64, GRID, GRID) and g.shape == (3, 128)
    v = net.value(g)
    assert v.shape == (3,) and (v.abs() <= 1.0).all()
    assert net.aux(g).shape == (3, 3)
    print("  params: {:,}".format(n))


def test_decoder_respects_masks():
    torch.manual_seed(0)
    net = TerminalNet()
    board = torch.randn(1, N_PLANES, GRID, GRID)
    feat, g = net.forward_torso(board, torch.randn(1, N_SCALARS))
    c, keys = net.policy.init(feat, g)

    costs = Costs(_CONFIG)
    scratch = PlanScratch(costs, sp=3.0, mp=4.0)
    for _ in range(6):  # walk a few sampled tokens; every one must be legal
        tmask = torch.from_numpy(scratch.type_mask()[None])
        lt = net.policy.type_logits(c).masked_fill(~tmask, -1e9)
        ttype = int(torch.distributions.Categorical(logits=lt).sample())
        assert scratch.type_mask()[ttype], "sampled illegal type"
        if ttype == END:
            break
        lmask = torch.from_numpy(scratch.loc_mask(ttype)[None])
        ll = net.policy.loc_logits(c, keys).masked_fill(~lmask, -1e9)
        loc = int(torch.distributions.Categorical(logits=ll).sample())
        assert scratch.loc_mask(ttype)[loc], "sampled illegal loc"
        cmask = torch.from_numpy(scratch.count_mask(ttype)[None])
        lc = net.policy.count_logits(c, feat, torch.tensor([loc])).masked_fill(
            ~cmask, -1e9
        )
        count = int(torch.distributions.Categorical(logits=lc).sample())
        scratch.apply(Token(ttype, loc, count))
        c = net.policy.advance(
            c, feat, torch.tensor([ttype]), torch.tensor([loc]), torch.tensor([count])
        )


def test_plan_nll_finite_and_padded():
    torch.manual_seed(1)
    net = TerminalNet()
    bsz, t_max = 4, 6
    board = torch.randn(bsz, N_PLANES, GRID, GRID)
    feat, g = net.forward_torso(board, torch.randn(bsz, N_SCALARS))

    plans = torch.zeros(bsz, t_max, 3, dtype=torch.long)
    plans[:, 0] = torch.tensor([DEP_SCOUT, xy_loc(13, 0), 2])
    plans[:, 1, 0] = END
    lengths = torch.full((bsz,), 2, dtype=torch.long)
    tm = torch.ones(bsz, t_max, N_TYPES, dtype=torch.bool)
    lm = torch.ones(bsz, t_max, N_LOCS, dtype=torch.bool)
    cm = torch.ones(bsz, t_max, N_BUCKETS, dtype=torch.bool)

    nll, ent = net.policy.plan_nll(feat, g, plans, lengths, tm, lm, cm)
    assert nll.shape == (bsz,) and torch.isfinite(nll).all() and (nll > 0).all()
    assert torch.isfinite(ent).all()
    # padded steps contribute nothing: extending T with garbage changes nothing
    plans2 = torch.cat([plans, torch.randint(0, 5, (bsz, 3, 3))], dim=1)
    tm2 = torch.ones(bsz, t_max + 3, N_TYPES, dtype=torch.bool)
    lm2 = torch.ones(bsz, t_max + 3, N_LOCS, dtype=torch.bool)
    cm2 = torch.ones(bsz, t_max + 3, N_BUCKETS, dtype=torch.bool)
    nll2, _ = net.policy.plan_nll(feat, g, plans2, lengths, tm2, lm2, cm2)
    assert torch.allclose(nll, nll2, atol=1e-5)
    # gradient flows
    nll.sum().backward()
    grads = [p.grad for p in net.policy.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_two_decoders_independent():
    net = TerminalNet()
    p_ids = {id(p) for p in net.policy.parameters()}
    q_ids = {id(p) for p in net.predict.parameters()}
    assert not (p_ids & q_ids), "policy and prediction decoders share weights"


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("\n{} / {} feature+model tests green".format(len(fns), len(fns)))


if __name__ == "__main__":
    main()
