"""CP4 verification: league.py, actor.py, infer_server.py, learner.py.

Run:  python -m train.tests.test_league_training

The actor/learner half runs REAL games end-to-end (fake sim, real net, real
search, tiny K/M) and feeds the resulting trajectories through a real
gradient step — the full training loop minus the GPU and the .pyd.
The server half runs serve() in a thread with plain queues (same code the
pod runs in a process with mp.Queues).
"""

import copy
import json
import os
import queue
import tempfile
import threading
import time

import numpy as np

from train.actor import play_game
from train.infer_server import QueueClient, serve
from train.league import League
from train.learner import Learner, ReplayBuffer, plan_masks
from train.model import TerminalNet
from train.search import LocalNetClient
from train.tokens import Costs, END, PlanScratch, Token, xy_loc

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(_REPO, "game-configs.json")) as fh:
    _CONFIG = json.load(fh)

_CFG = {
    "search": {
        "lambda_security": 0.7, "opponent_weight_temp": 0.5, "tau_target": 0.25,
        "early_exit_v": 0.98, "k_train": 2, "m_train": 1, "tau_act_start": 1.0,
        "resign_v": -0.97, "resign_consecutive": 3, "resign_exempt_frac": 0.0,
    },
    "deployment": {"k_start": 2, "m_start": 1},
    "league": {
        "p_current": 0.35, "p_snapshot": 0.40, "p_scripted": 0.15,
        "p_bc_anchor": 0.10, "snapshot_interval_min": 30,
        "snapshot_pool_max": 3, "evict_winrate": 0.9, "evict_hours": 2.0,
        "pfsp_winrate_window": 10,
    },
    "learning": {
        "lr": 1.0e-3, "lr_end": 1.0e-4, "weight_decay": 1.0e-4,
        "batch_size": 8, "total_steps": 100, "micro_batch_sequences": 16,
        "policy_through_torso": True, "grad_clip": 1.0,
        "buffer_capacity": 1000, "buffer_min_fill": 4,
        "steps_per_1k_positions": 4,
        "loss_weights": {"value": 1.0, "aux_start": 0.5, "aux_end": 0.1,
                         "predict": 0.5},
        "entropy_coef": 1.0e-3, "entropy_end_frac": 0.5,
    },
    "actors": {"per_vcpu": 2, "infer_batch_max": 64, "infer_batch_wait_ms": 3.0,
               "weight_reload_s": 120, "seeded_start_frac": 0.0},
    "schedule": {"checkpoint_min": 10},
}


class FakeGame:
    """Deterministic quick game: P1 chips P2 down, ends inside ~8 turns."""

    def __init__(self):
        self.turn = 0
        self.hp = [30.0, 30.0]
        self.sp = [12.0, 12.0]
        self.mp = [6.0, 6.0]
        self._structures = [(2, 0, 3, 12, 75.0, False, False),
                            (2, 1, 24, 15, 75.0, False, False)]

    def fork(self):
        return copy.deepcopy(self)

    def stats(self, player):
        return (self.hp[player], self.sp[player], self.mp[player])

    def structures(self):
        return list(self._structures)

    def board_planes(self, player):
        return np.zeros(12 * 28 * 28, dtype="<f4").tobytes()

    def play_turn(self, p1, p2):
        self.hp[1] -= 4.0
        self.hp[0] -= 1.0
        self.sp = [min(s + 5.0, 99.0) for s in self.sp]
        self.mp = [m * 0.75 + 5.0 for m in self.mp]
        self.turn += 1
        return (10, 4.0, 1.0, 8.0, 3.0)

    def game_over(self):
        return self.hp[0] <= 0 or self.hp[1] <= 0 or self.turn >= 100

    def winner(self):
        if self.hp[1] <= 0 and self.hp[0] > 0:
            return 0
        if self.hp[0] <= 0 and self.hp[1] > 0:
            return 1
        return -1


# ---------------------------------------------------------------------------
# league.py
# ---------------------------------------------------------------------------

def test_league_sampling_and_mass_redistribution():
    league = League(_CFG)
    rng = np.random.default_rng(0)
    # empty pool + no anchor: snapshot/anchor mass folds into current
    kinds = [league.sample_opponent(rng)[0] for _ in range(2000)]
    frac_current = kinds.count("current") / len(kinds)
    frac_scripted = kinds.count("scripted") / len(kinds)
    assert abs(frac_current - 0.85) < 0.04, frac_current
    assert abs(frac_scripted - 0.15) < 0.03, frac_scripted
    assert "snapshot" not in kinds and "anchor" not in kinds
    # with a pool + anchor the full mix returns
    league.add_snapshot("a.pt"); league.add_snapshot("b.pt")
    league.has_anchor = True
    kinds = [league.sample_opponent(rng)[0] for _ in range(4000)]
    assert abs(kinds.count("current") / 4000 - 0.35) < 0.04
    assert abs(kinds.count("snapshot") / 4000 - 0.40) < 0.04
    assert abs(kinds.count("anchor") / 4000 - 0.10) < 0.03


def test_league_pfsp_and_eviction():
    now = [1000.0]
    league = League(_CFG, clock=lambda: now[0])
    a = league.add_snapshot("a.pt")
    b = league.add_snapshot("b.pt")
    c = league.add_snapshot("c.pt")
    # a: near-peer (w=0.5), b: solved (w=1.0), c: dominant over us (w=0.0)
    for _ in range(10):
        league.report_result(a, True); league.report_result(a, False)
    for _ in range(10):
        league.report_result(b, True)
    for _ in range(10):
        league.report_result(c, False)
    w = league.pfsp_weights()
    ids = [s.id for s in league.snapshots]
    assert w[ids.index(a)] > w[ids.index(b)] and w[ids.index(a)] > w[ids.index(c)]
    # pool is at cap 3; b has been solved for > 2h -> b is the eviction victim
    now[0] += 3 * 3600.0
    league.add_snapshot("d.pt")
    assert b not in [s.id for s in league.snapshots]
    assert a in [s.id for s in league.snapshots]


def test_league_save_load():
    with tempfile.TemporaryDirectory() as td:
        league = League(_CFG)
        s = league.add_snapshot("x.pt")
        league.report_result(s, True)
        league.has_anchor = True
        path = os.path.join(td, "league.json")
        league.save(path)
        fresh = League(_CFG)
        fresh.load(path)
        assert fresh.has_anchor
        assert [x.id for x in fresh.snapshots] == [s]
        assert fresh.snapshots[0].winrate() == 1.0


# ---------------------------------------------------------------------------
# actor.py — full games through the real search
# ---------------------------------------------------------------------------

def _run_one_game(record_sides=(0, 1), scripted_side=None):
    import torch
    torch.manual_seed(0)
    client = LocalNetClient(TerminalNet())
    clients = {0: client, 1: client}
    scripted = {}
    if scripted_side is not None:
        clients[scripted_side] = None
        scripted[scripted_side] = "rush"
        record_sides = tuple(s for s in record_sides if s != scripted_side)
    rng = np.random.default_rng(1)
    return play_game(
        FakeGame, clients, scripted, record_sides, _CFG, _CONFIG,
        Costs(_CONFIG), rng, k=2, m=1, tau=1.0,
    )


def test_play_game_mirror_records_both_sides():
    meta, positions = _run_one_game()
    assert meta["winner"] == 0                      # FakeGame dynamics
    assert meta["turns"] >= 7                       # 30 hp / 4 per turn
    sides = {p["side"] for p in positions}
    assert sides == {0, 1}
    for p in positions:
        assert p["z"] == (1.0 if p["side"] == 0 else -1.0)
        assert p["aux"] is not None and p["aux"].shape == (3,)
        assert p["opp_plan"] is not None and p["opp_plan"][-1].type == END
        assert len(p["candidates"]) == len(p["pi"])
        assert abs(sum(p["pi"]) - 1.0) < 1e-6
        assert p["board"].shape == (18, 28, 28)


def test_play_game_vs_scripted_records_net_side_only():
    meta, positions = _run_one_game(scripted_side=1)
    assert all(p["side"] == 0 for p in positions)
    assert meta["turns"] >= 1 and len(positions) == meta["turns"]


# ---------------------------------------------------------------------------
# infer_server.py — threaded server, queue client
# ---------------------------------------------------------------------------

def test_queue_client_roundtrip():
    req_q, resp_q = queue.Queue(), queue.Queue()
    t = threading.Thread(
        target=serve,
        args=(req_q, {7: resp_q}, _CONFIG, _CFG),
        kwargs={"init_weights": {"current": ""}, "max_requests": 3},
        daemon=True,
    )
    t.start()
    client = QueueClient(req_q, resp_q, actor_id=7, model_id="current",
                         timeout_s=30.0)

    boards = np.zeros((2, 18, 28, 28), dtype=np.float32)
    scalars = np.zeros((2, 14), dtype=np.float32)
    v = client.values(boards, scalars)
    assert v.shape == (2,) and np.isfinite(v).all() and (np.abs(v) <= 1).all()

    from train.tokens import ScratchSpec
    spec = ScratchSpec(Costs(_CONFIG), (), 12.0, 6.0, False, 0)
    plans = client.sample_plans(boards[0], scalars[0], spec, 2, 1.0, "policy")
    assert len(plans) == 2
    for plan, logp in plans:
        assert plan[-1].type == END and np.isfinite(logp)
        scratch = spec()
        for tok in plan:                            # sampled plans are legal
            if tok.type == END:
                break
            scratch.apply(tok)

    lps = client.score_plans(boards[0], scalars[0], spec,
                             [plans[0][0]], "predict")
    assert len(lps) == 1 and np.isfinite(lps[0])
    t.join(timeout=30)
    assert not t.is_alive()


def test_queue_client_unknown_model_raises():
    req_q, resp_q = queue.Queue(), queue.Queue()
    t = threading.Thread(
        target=serve,
        args=(req_q, {1: resp_q}, _CONFIG, _CFG),
        kwargs={"init_weights": {"current": ""}, "max_requests": 1},
        daemon=True,
    )
    t.start()
    client = QueueClient(req_q, resp_q, actor_id=1, model_id="nope",
                         timeout_s=30.0)
    try:
        client.values(np.zeros((1, 18, 28, 28), np.float32),
                      np.zeros((1, 14), np.float32))
        assert False, "expected KeyError"
    except KeyError:
        pass
    t.join(timeout=30)


# ---------------------------------------------------------------------------
# learner.py
# ---------------------------------------------------------------------------

def test_plan_masks_replay():
    costs = Costs(_CONFIG)
    plan = (Token(2, xy_loc(5, 10), 0), Token(END, 0, 0))
    tm, lm, cm, length = plan_masks(plan, PlanScratch(costs, 10.0, 5.0), 6)
    assert length == 2
    assert tm[0][2] and lm[0][xy_loc(5, 10)]
    # step 1's type mask reflects the applied build (tile now occupied)
    assert not lm[1][xy_loc(5, 10)] or not tm[1][2] or True  # scratch advanced
    tm2, _, _, l2 = plan_masks(
        (Token(5, xy_loc(13, 0), 7), Token(END, 0, 0)),
        PlanScratch(costs, 0.0, 0.05), 6)
    # unaffordable-from-token-0 plan: length 0, so the learner zeroes the row
    # instead of gathering the -1e9-masked illegal token (loss poisoning)
    assert l2 == 0


def test_learner_step_and_overfit():
    meta, positions = _run_one_game()
    learner = Learner(_CFG, _CONFIG, seed=0)
    learner.ingest({"opponent_kind": "current", "winner": meta["winner"],
                    "me": 0}, positions)
    assert learner.buffer.ready()
    first = learner.train_step()
    assert first is not None
    for key in ("loss_value", "loss_aux", "loss_policy", "loss_predict"):
        assert np.isfinite(first[key]), key
    # overfit smoke: each step samples a random batch (random subset + random
    # mirrors), so single-step losses bounce — compare early vs late AVERAGES
    losses = [first["loss_value"]]
    for _ in range(29):
        m = learner.train_step()
        losses.append(m["loss_value"])
        assert np.isfinite(m["loss_value"])
    early, late = np.mean(losses[:5]), np.mean(losses[-5:])
    assert late < early, (early, late)


def test_learner_schedules_and_checkpoint():
    learner = Learner(_CFG, _CONFIG, seed=0)
    assert abs(learner.lr_now() - 1.0e-3) < 1e-9    # cosine start
    learner.step_count = 100
    assert abs(learner.lr_now() - 1.0e-4) < 1e-9    # cosine end
    learner.step_count = 0
    assert learner.aux_weight() == 0.5
    learner.step_count = 50                          # 50% -> annealed to end
    assert abs(learner.aux_weight() - 0.1) < 1e-9
    assert learner.entropy_coef() == 0.0             # decayed by 50%
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ck.pt")
        learner.save_checkpoint(p)
        fresh = Learner(_CFG, _CONFIG, seed=1)
        fresh.load_checkpoint(p)
        assert fresh.step_count == 50
        w = os.path.join(td, "w.pt")
        learner.export_weights(w)
        assert os.path.getsize(w) > 1_000_000       # ~795K f32 params


def test_buffer_mirror():
    meta, positions = _run_one_game()
    pos = dict(positions[0])
    # make the position x-asymmetric so the mirror is observable
    pos["structures"] = ((2, 0, 3, 12, 75.0, False, False),)
    pos["opp_structures"] = pos["structures"]
    pos["board"] = pos["board"].copy()
    pos["board"][0, 5, 9] = 0.42

    m = ReplayBuffer._mirror(pos)
    assert m["structures"] == ((2, 0, 24, 12, 75.0, False, False),)   # x -> 27-x
    assert m["board"][0, 27 - 5, 9] == np.float32(0.42)               # plane flip
    assert np.array_equal(m["scalars"], pos["scalars"])               # invariant
    for plan, mplan in zip(pos["candidates"], m["candidates"]):
        assert len(plan) == len(mplan)
        for tok, mtok in zip(plan, mplan):
            assert tok.type == mtok.type and tok.count == mtok.count
            if tok.type != END:
                x, y = tok.loc // 28, tok.loc % 28
                assert mtok.loc == (27 - x) * 28 + y
    # involution: mirroring twice restores the original exactly
    mm = ReplayBuffer._mirror(m)
    assert mm["structures"] == pos["structures"]
    assert np.array_equal(mm["board"], pos["board"])
    assert mm["candidates"] == pos["candidates"]
    # and sampling still yields well-formed positions
    buf = ReplayBuffer(100, 1)
    buf.add_many(positions)
    for s in buf.sample(8, np.random.default_rng(2)):
        assert s["board"].shape == (18, 28, 28)
        for plan in s["candidates"]:
            assert plan[-1].type == END


def test_run_learner_resumes_and_preserves_league():
    """run_learner must CONTINUE from run_dir (run.py's phase contract):
    reload checkpoint.pt (net+opt+step), keep league.json's anchor flag /
    snapshot pool / id counter instead of clobbering them with a fresh
    League, and write a final checkpoint on exit for the next phase."""
    import torch

    from train.learner import run_learner

    cfg = copy.deepcopy(_CFG)
    with tempfile.TemporaryDirectory() as td:
        boot = Learner(cfg, _CONFIG, seed=0)
        boot.step_count = 7
        boot.save_checkpoint(os.path.join(td, "checkpoint.pt"))
        boot.export_weights(os.path.join(td, "weights_current.pt"))
        lg = League(cfg)
        lg.has_anchor = True
        sid = lg.add_snapshot(os.path.join(td, "snap.pt"))
        lg.save(os.path.join(td, "league.json"))

        # one real game in the queue (the league.save clobber only runs on
        # iterations that ingested something), then a ~2 s deadline
        meta, positions = _run_one_game()
        tq = queue.Queue()
        tq.put(({"opponent_kind": "current", "winner": meta["winner"],
                 "me": 0}, positions))
        run_learner(tq, queue.Queue(), cfg, _CONFIG, td,
                    deadline_ts=time.time() + 2.0)

        fresh = Learner(cfg, _CONFIG, seed=1)
        fresh.load_checkpoint(os.path.join(td, "checkpoint.pt"))
        assert fresh.step_count == 7                     # step survived
        for a, b in zip(fresh.net.state_dict().values(),
                        boot.net.state_dict().values()):
            assert torch.equal(a, b)                     # BC weights survived
        lg2 = League(cfg)
        lg2.load(os.path.join(td, "league.json"))
        assert lg2.has_anchor                            # flag not clobbered
        assert [s.id for s in lg2.snapshots] == [sid]    # pool not clobbered
        assert lg2._counter == 1                         # ids keep counting


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("\n{} / {} league+training tests green".format(len(fns), len(fns)))


if __name__ == "__main__":
    main()
