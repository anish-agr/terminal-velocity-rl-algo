"""CP3b verification: scripted.py (league bots) + replays.py (corpus reader).

Run:  python -m train.tests.test_scripted_replays

The replay half builds a synthetic 2-turn .replay from the REAL config with a
known command script on both sides, then asserts the reader reconstructs
exactly those commands (builds from spawns, upgrades from marker-list diffs,
removal marks shifted back one turn), the flows, the winner, the plan tokens,
and the lazily-built feature tensors.
"""

import json
import os
import tempfile

import numpy as np

from train.replays import (
    build_bc_index, check_corpus, config_matches, iter_positions, load_game,
)
from train.scripted import SCRIPTED_BOTS, K_REMOVE, K_UPGRADE
from train.tokens import (
    BUILD_TURRET, DEP_SCOUT, DEP_DEMOLISHER, END, REMOVE, UPGRADE, Token,
    in_arena, to_abs, xy_loc,
)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(_REPO, "game-configs.json")) as fh:
    _CONFIG = json.load(fh)


# ---------------------------------------------------------------------------
# scripted.py
# ---------------------------------------------------------------------------

class FakeGame:
    def __init__(self, turn=0, structures=(), stats=(30.0, 40.0, 8.0)):
        self.turn = turn
        self._structures = list(structures)
        self._stats = stats

    def structures(self):
        return list(self._structures)

    def stats(self, player):
        return self._stats


def test_bots_deterministic_and_legal():
    for name, bot in SCRIPTED_BOTS.items():
        for player in (0, 1):
            for turn in (0, 3, 8, 11):
                g = FakeGame(turn=turn)
                a = bot(g, player, _CONFIG)
                b = bot(g, player, _CONFIG)
                assert a == b, "{} nondeterministic".format(name)
                for (kind, x, y) in a:
                    assert 0 <= kind <= 7
                    assert in_arena(x, y), "{} out of arena: {}".format(name, (kind, x, y))


def test_bots_mirror_between_seats():
    """On an empty board the two seats' command sets must be exact flips."""
    for name, bot in SCRIPTED_BOTS.items():
        g = FakeGame(turn=0)
        c0 = bot(g, 0, _CONFIG)
        c1 = bot(g, 1, _CONFIG)
        flipped = [(k, 27 - x, 27 - y) for (k, x, y) in c0]
        assert flipped == c1, "{} seats diverge".format(name)


def test_rush_attacks_and_banks():
    g_rich = FakeGame(stats=(30.0, 40.0, 8.0))
    scouts = [c for c in SCRIPTED_BOTS["rush"](g_rich, 0, _CONFIG) if c[0] == 3]
    assert len(scouts) == 8                     # all-in
    g_poor = FakeGame(stats=(30.0, 40.0, 3.0))
    scouts = [c for c in SCRIPTED_BOTS["rush"](g_poor, 0, _CONFIG) if c[0] == 3]
    assert scouts == []                          # below wave threshold: bank


def test_torture_phases():
    # phase 3 marks the forward walls built in phases 0-2 for removal
    walls = [(0, 0, x, 13, 40.0, False, False) for x in range(9, 13)]
    g = FakeGame(turn=3, structures=walls)
    cmds = SCRIPTED_BOTS["torture"](g, 0, _CONFIG)
    removes = [c for c in cmds if c[0] == K_REMOVE]
    assert sorted(removes) == [(K_REMOVE, x, 13) for x in range(9, 13)]
    # phase 5 upgrades supports it owns
    sup = [(1, 0, 13, 3, 30.0, False, False)]
    g5 = FakeGame(turn=5, structures=sup)
    ups = [c for c in SCRIPTED_BOTS["torture"](g5, 0, _CONFIG) if c[0] == K_UPGRADE]
    assert (K_UPGRADE, 13, 3) in ups


def test_funnel_upgrades_only_own_unupgraded():
    own_upgraded = [(0, 0, 12, 13, 120.0, True, False)]
    g = FakeGame(turn=2, structures=own_upgraded, stats=(30.0, 40.0, 0.0))
    ups = [c for c in SCRIPTED_BOTS["funnel"](g, 0, _CONFIG) if c[0] == K_UPGRADE]
    assert (K_UPGRADE, 12, 13) not in ups


# ---------------------------------------------------------------------------
# replays.py — synthetic replay with a known script
# ---------------------------------------------------------------------------

def _frame(turn, ftype, aframe, p1u, p2u, p1s, p2s, events=None, end=None):
    ev = {k: [] for k in ("selfDestruct", "breach", "damage", "shield", "move",
                          "spawn", "death", "attack", "melee")}
    if events:
        ev.update(events)
    d = {
        "turnInfo": [ftype, turn, aframe, 0],
        "p1Units": p1u, "p2Units": p2u,
        "p1Stats": p1s, "p2Stats": p2s,
        "events": ev,
    }
    if end:
        d["endStats"] = end
    return d


def _units(*, walls=(), supports=(), turrets=(), upgrades=()):
    def fmt(lst):
        return [[x, y, hp, str(uid)] for (x, y, hp, uid) in lst]
    return [fmt(walls), fmt(supports), fmt(turrets), [], [], [], [], fmt(upgrades)]


def _write_synthetic_replay(path):
    """Two playable turns with a known command script:

    turn 0: P1 builds a turret (3,12) + deploys 2 scouts (13,0); P2 builds a
            wall (24,15). One P1 scout damages the wall (2.0), the other
            breaches (1.0).
    turn 1: P1 upgrades the turret; P2 marks the wall for removal (death with
            was_removed appears in the turn-2 frame) and deploys a demolisher.
    end:    P1 wins 10 vs -2.
    """
    lines = [json.dumps(_CONFIG)]
    empty = _units()
    s0 = [30.0, 40.0, 5.0, 0]
    lines.append(json.dumps(_frame(0, 0, -1, empty, empty, s0, s0)))
    lines.append(json.dumps(_frame(0, 1, 0, empty, empty, s0, s0, events={
        "spawn": [
            [[3, 12], 2, "1", 1],
            [[13, 0], 3, "2", 1], [[13, 0], 3, "3", 1],
            [[24, 15], 0, "4", 2],
        ],
    })))
    lines.append(json.dumps(_frame(0, 1, 5, empty, empty, s0, s0, events={
        "damage": [[[24, 15], 2.0, 0, "4", 2]],
        "breach": [[[24, 17], 1.0, 3, "3", 1]],
    })))

    p1_t1 = _units(turrets=[(3, 12, 75.0, "1")])
    p2_t1 = _units(walls=[(24, 15, 38.0, "4")])
    s1a, s1b = [29.0, 42.0, 6.0, 0], [29.0, 41.0, 6.0, 0]
    lines.append(json.dumps(_frame(1, 0, -1, p1_t1, p2_t1, s1a, s1b)))
    lines.append(json.dumps(_frame(1, 1, 0, p1_t1, p2_t1, s1a, s1b, events={
        "spawn": [[[14, 27], 4, "6", 2]],
    })))

    # turn-2 frame: upgrade marker present for P1; P2 wall's removal executes
    p1_t2 = _units(turrets=[(3, 12, 75.0, "5")], upgrades=[(3, 12, 0.0, "5")])
    p2_t2 = _units()
    s2a, s2b = [28.0, 40.0, 7.0, 0], [28.0, 44.0, 7.0, 0]
    lines.append(json.dumps(_frame(2, 0, -1, p1_t2, p2_t2, s2a, s2b, events={
        "death": [[[24, 15], 0, "4", 2, True]],
    })))

    end = {"winner": 1, "duration": 100}
    lines.append(json.dumps(_frame(2, 2, -1, p1_t2, p2_t2,
                                   [10.0, 0.0, 0.0, 0], [-2.0, 0.0, 0.0, 0],
                                   end={"winner": 1})))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def test_config_matches():
    assert config_matches(_CONFIG, json.loads(json.dumps(_CONFIG)))
    icon_diff = json.loads(json.dumps(_CONFIG))
    icon_diff["unitInformation"][0]["icon"] = "different_icon"
    assert config_matches(_CONFIG, icon_diff)          # icons don't matter
    gameplay_diff = json.loads(json.dumps(_CONFIG))
    gameplay_diff["unitInformation"][4]["cost2"] = 3.0
    assert not config_matches(_CONFIG, gameplay_diff)  # gameplay does


def test_replay_reconstruction():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "synthetic.replay")
        _write_synthetic_replay(path)
        rec = load_game(path, _CONFIG)
        assert rec is not None
        assert rec.winner == 0                          # P1, 0-based
        assert len(rec.turns) == 2

        t0, t1 = rec.turns
        assert t0.commands[0] == ((2, 3, 12), (3, 13, 0), (3, 13, 0))
        assert t0.commands[1] == ((0, 24, 15),)
        assert t0.flows[0] == (1.0, 2.0)                # P1 breached 1, dealt 2
        assert t0.flows[1] == (0.0, 0.0)
        # turn 1: upgrade from marker diff; removal mark shifted back; deploy
        assert t1.commands[0] == ((7, 3, 12),)
        assert t1.commands[1] == ((6, 24, 15), (4, 14, 27))
        # snapshot carries the wall P2 owns at turn 1
        walls = [s for s in t1.structures if s[0] == 0 and s[1] == 1]
        assert walls == [(0, 1, 24, 15, 38.0, False, False)]


def test_positions_and_tokens():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "synthetic.replay")
        _write_synthetic_replay(path)
        rec = load_game(path, _CONFIG)
        pos = list(iter_positions(rec, _CONFIG))
        assert len(pos) == 4                            # 2 turns x 2 sides
        p1t0 = pos[0]
        assert p1t0.z == 1.0 and pos[1].z == -1.0
        assert p1t0.plan == (
            Token(BUILD_TURRET, xy_loc(3, 12), 0),
            Token(DEP_SCOUT, xy_loc(13, 0), 1),         # bucket idx 1 == 2 units
            Token(END, 0, 0),
        )
        # P2's turn-1 plan, perspective-flipped: remove + demolisher deploy
        p2t1 = pos[3]
        assert p2t1.side == 1
        assert p2t1.plan == (
            Token(REMOVE, xy_loc(3, 12), 0),            # (24,15) flipped
            Token(DEP_DEMOLISHER, xy_loc(13, 0), 0),    # (14,27) flipped
            Token(END, 0, 0),
        )
        # P2's turn-1 history saw P1's 2 scouts at (13,0) -> flipped (14,27)
        assert abs(p2t1.board[12, 14, 27] - 0.2) < 1e-6
        # and its scalars carry the breach it took
        assert abs(p2t1.scalars[11] - 1.0 / 5.0) < 1e-6
        # P1's turn-1 board: own upgraded... not yet (upgrade lands turn 1);
        # its enemy wall plane shows P2's wall at flipped-off (no flip, side 0)
        p1t1 = pos[2]
        assert abs(p1t1.board[5, 24, 15] - 38.0 / 120.0) < 1e-6


def test_bc_index_and_gate():
    with tempfile.TemporaryDirectory() as td:
        for i in range(3):
            _write_synthetic_replay(os.path.join(td, "g{}.replay".format(i)))
        idx = build_bc_index(td, _CONFIG, cap_frac=0.5)
        # 3 identical winner fingerprints, cap = floor(3*0.5)=1 -> capped to 1
        assert len(idx) == 1 and idx[0][1] == 0
        assert check_corpus(td, _CONFIG)                # the §0.3 gate passes


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("\n{} / {} scripted+replay tests green".format(len(fns), len(fns)))


if __name__ == "__main__":
    main()
