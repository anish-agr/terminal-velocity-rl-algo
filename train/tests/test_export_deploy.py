"""CP5 verification: export/parity, numpy inference client, gauntlet +
promotion rule, bootstrap BC-position construction, fallback bot, config load.

Run:  python -m train.tests.test_export_deploy
"""

import json
import os
import sys
import tempfile

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "deploy"))

from train.evaluate import run_gauntlet, save_report, should_promote
from train.run import bc_positions_from_corpus, load_cfg
from train.search import LocalNetClient, choose
from train.tokens import Costs, END
from train.tests.test_league_training import _CFG, FakeGame
from train.tests.test_scripted_replays import _write_synthetic_replay

with open(os.path.join(_REPO, "game-configs.json")) as fh:
    _CONFIG = json.load(fh)


def _exported_net(td):
    import torch
    from train.export import export_checkpoint
    from train.model import TerminalNet
    torch.manual_seed(0)
    net = TerminalNet()
    path = os.path.join(td, "weights.bin")
    export_checkpoint(net, path)
    return net, path


def test_parity_gate():
    from train.export import parity_check
    with tempfile.TemporaryDirectory() as td:
        net, path = _exported_net(td)
        worst = parity_check(net, path, n_states=8)
        assert worst < 1e-4, worst


def test_numpy_client_drives_search():
    from npforward import NumpyNet, NumpyNetClient
    with tempfile.TemporaryDirectory() as td:
        _net, path = _exported_net(td)
        client = NumpyNetClient(NumpyNet(path), seed=1)
        from train.features import DeployHistory
        plan, pi_star, diag = choose(
            FakeGame(), client, _CFG, player=0,
            hist_own=DeployHistory(_CONFIG), hist_opp=DeployHistory(_CONFIG),
            config=_CONFIG, costs=Costs(_CONFIG), k=2, m=1, tau=1.0,
        )
        assert plan[-1].type == END
        assert abs(sum(pi_star.values()) - 1.0) < 1e-6
        assert diag["values"].shape == (diag["k_eff"], diag["m_eff"])


def test_gauntlet_and_promotion():
    import torch
    from train.model import TerminalNet
    torch.manual_seed(0)
    client = LocalNetClient(TerminalNet())
    report = run_gauntlet(
        FakeGame, lambda mid: client, [("scripted", "rush")],
        _CFG, _CONFIG, n_games=2, k=2, m=1,
    )
    r = report["scripted:rush"]
    # FakeGame dynamics: P1 always wins -> seat alternation gives 1 win, 1 loss
    assert r["wins"] == 1 and r["losses"] == 1 and r["crashes"] == 0
    assert np.isfinite(r["mean_margin"])
    with tempfile.TemporaryDirectory() as td:
        path = save_report(report, td)
        assert os.path.exists(path)

    # promotion rule on synthetic reports (min-based, §8)
    def rep(script_wr, best_wr, timeouts=0):
        return {
            "scripted:rush": {"kind": "scripted", "name": "rush",
                              "win_rate": script_wr, "crashes": 0,
                              "timeouts": timeouts},
            "client:prev_best": {"kind": "client", "name": "prev_best",
                                 "win_rate": best_wr, "crashes": 0,
                                 "timeouts": 0},
        }
    cfg = {"evaluation": {"promote_vs_best": 0.55, "promote_vs_scripted": 0.85,
                          "promote_max_crashes": 0}}
    assert should_promote(rep(0.9, 0.6), cfg)[0]
    assert not should_promote(rep(0.8, 0.6), cfg)[0]      # weak vs a script
    assert not should_promote(rep(0.9, 0.5), cfg)[0]      # weak vs prev best
    assert not should_promote(rep(0.9, 0.6, timeouts=1), cfg)[0]


def test_bc_positions_from_corpus():
    cfg = load_cfg()
    with tempfile.TemporaryDirectory() as td:
        _write_synthetic_replay(os.path.join(td, "g0.replay"))
        positions = [p for chunk in bc_positions_from_corpus(cfg, _CONFIG, td)
                     for p in chunk]                      # streamed per replay
        assert len(positions) == 4                        # 2 turns x 2 sides
        winners = [p for p in positions if p["candidates"]]
        losers = [p for p in positions if not p["candidates"]]
        assert winners and losers
        for p in winners:                                  # P1 won -> side 0 BC
            assert p["side"] == 0 and p["pi"] == [1.0] and p["z"] == 1.0
        for p in positions:                                # learner-schema keys
            for key in ("board", "scalars", "structures", "sp", "mp",
                        "opp_board", "opp_plan", "aux"):
                assert key in p, key
        # and the learner actually accepts them
        from train.learner import Learner
        learner = Learner(_CFG, _CONFIG, seed=0)
        learner.buffer.add_many(positions)
        m = learner.train_step()
        assert m is not None and np.isfinite(m["loss_policy"])


class _StubState:
    """Records gamelib calls the fallback bot makes."""

    def __init__(self, mp=10.0):
        self.spawned, self.upgraded = [], []
        self._mp = mp

    def attempt_spawn(self, unit, locs, num=1):
        self.spawned.append((unit, len(locs) if isinstance(locs[0], list) else 1))
        return 1

    def attempt_upgrade(self, locs):
        self.upgraded.append(len(locs))
        return 1

    def get_resource(self, rt):
        return self._mp


def test_fallback_bot():
    from fallback import FallbackBot
    bot = FallbackBot(_CONFIG)
    st = _StubState(mp=10.0)
    bot.apply(st)
    kinds = [k for k, _n in st.spawned]
    assert bot.TURRET in kinds and bot.WALL in kinds and bot.DEMOLISHER in kinds
    assert st.upgraded
    quiet = _StubState(mp=1.0)                     # below wave threshold: banks
    bot.apply(quiet)
    assert bot.DEMOLISHER not in [k for k, _n in quiet.spawned]

    class _Broken:                                 # fallback must never raise
        def __getattr__(self, name):
            raise RuntimeError("broken state")
    bot.apply(_Broken())


def test_config_yaml_loads_completely():
    cfg = load_cfg()
    for section in ("state", "action", "network", "search", "learning",
                    "league", "replays", "evaluation", "actors", "cold_start",
                    "deployment", "schedule"):
        assert section in cfg, section
    assert cfg["search"]["k_deploy"] == 16
    assert cfg["learning"]["policy_through_torso"] is True


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("\n{} / {} export+deploy tests green".format(len(fns), len(fns)))


if __name__ == "__main__":
    main()
