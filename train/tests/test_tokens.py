"""CP1 verification for train/tokens.py — pure python, no sim required.

Run:  python -m train.tests.test_tokens
Covers geometry, config-driven costs, legality masks, margin behaviour, scratch
accounting, encode/decode round trips, mirror involution, and perspective flip.
The sim-dependent gate (§0.2: mask vs apply_commands on 10K random commands)
lives in the pod bootstrap — this file is everything provable without the .so.
"""

import json
import os

import numpy as np

from train.tokens import (
    ALL_BUCKET, BUILD_TURRET, BUILD_WALL, COUNT_BUCKETS, DEP_DEMOLISHER,
    DEP_SCOUT, END, END_TOKEN, GRID, N_LOCS, OWN_EDGES, OWN_HALF, REMOVE,
    Token, UPGRADE, Costs, PlanScratch, decode_commands, encode_plan,
    from_abs, in_arena, loc_xy, mirror_plan, to_abs, xy_loc,
)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _costs():
    with open(os.path.join(_REPO, "game-configs.json")) as fh:
        return Costs(json.load(fh))


def _scratch(sp=40.0, mp=15.0, structures=(), **kw):
    return PlanScratch(_costs(), sp, mp, structures, **kw)


def test_geometry():
    assert int(OWN_HALF.sum()) == 210, OWN_HALF.sum()   # bottom diamond half
    assert int(OWN_EDGES.sum()) == 28, OWN_EDGES.sum()  # two 14-tile deploy edges
    assert not (OWN_EDGES & ~OWN_HALF).any()            # edges lie in own half
    assert in_arena(13, 0) and in_arena(14, 0) and in_arena(0, 13)
    assert not in_arena(0, 0) and not in_arena(27, 27)
    for x in range(GRID):
        for y in range(GRID):
            assert from_abs(*to_abs(x, y, True), True) == (x, y)  # involution


def test_costs_from_real_config():
    c = _costs()
    assert c.build_sp == [1.0, 4.0, 2.0]
    # upgrade cost = upgrade-block cost1 if present else base cost (wall -> 1)
    assert c.upgrade_sp == [1.0, 4.0, 4.0]
    assert c.deploy_mp == [1.0, 2.0, 1.0]


def test_build_legality_and_margin():
    s = _scratch(sp=40.0, mp=0.0)
    m = s.loc_mask(BUILD_WALL)
    assert m.sum() == 210                       # empty board: whole own half
    tgt = xy_loc(13, 11)
    s.apply(Token(BUILD_WALL, tgt, 0))
    assert s.sp == 39.0
    assert not s.loc_mask(BUILD_WALL)[tgt]      # occupied now
    assert s.loc_mask(BUILD_TURRET)[xy_loc(14, 11)]
    # margin: sp=1.05 -> avail 0.95 < wall cost 1.0 -> masked
    tight = _scratch(sp=1.05, mp=0.0)
    assert not tight.type_mask()[BUILD_WALL]
    ok = _scratch(sp=1.15, mp=0.0)
    assert ok.type_mask()[BUILD_WALL]
    assert ok.type_mask()[END]                  # END always legal


def test_upgrade_remove():
    tgt = xy_loc(5, 10)
    s = _scratch(structures=[(2, 0, 5, 10, 75.0, False, False)])  # own turret
    assert s.loc_mask(UPGRADE)[tgt] and s.loc_mask(REMOVE)[tgt]
    s.apply(Token(UPGRADE, tgt, 0))
    assert s.sp == 36.0                          # turret upgrade costs 4
    assert not s.loc_mask(UPGRADE)[tgt]          # already upgraded
    s.apply(Token(REMOVE, tgt, 0))
    assert not s.loc_mask(REMOVE)[tgt]           # pending: repeat is masked
    # enemy structure is neither upgradable nor removable nor buildable-over
    e = _scratch(structures=[(0, 1, 20, 12, 40.0, False, False)])
    loc = xy_loc(20, 12)
    assert not e.loc_mask(UPGRADE)[loc] and not e.loc_mask(REMOVE)[loc]
    assert not e.loc_mask(BUILD_WALL)[loc]


def test_deploy_masks_and_all_bucket():
    s = _scratch(sp=0.0, mp=7.3)
    m = s.loc_mask(DEP_SCOUT)
    assert m.sum() == 28                         # all edges open
    cm = s.count_mask(DEP_DEMOLISHER)            # cost 2, avail 7.2 -> 1,2,3 & ALL
    legal = {COUNT_BUCKETS[i] for i in np.nonzero(cm)[0]}
    assert legal == {1, 2, 3, -1}, legal
    lane = xy_loc(13, 0)
    n = s.apply(Token(DEP_DEMOLISHER, lane, ALL_BUCKET))
    assert n == 3 and abs(s.mp - 1.3) < 1e-6     # floor(7.3/2)=3, mp 7.3-6=1.3
    n2 = s.apply(Token(DEP_SCOUT, lane, 0))      # stacking on same tile is legal
    assert n2 == 1 and abs(s.mp - 0.3) < 1e-6
    assert not s.type_mask()[DEP_SCOUT]          # 0.3-margin cannot afford more
    assert not s.loc_mask(BUILD_WALL)[lane]      # build masked off deployed tile


def test_encode_decode_roundtrip():
    s = _scratch(sp=10.0, mp=8.3)  # 8.3: margin must not block the final 2-scout deploy
    plan = [
        Token(BUILD_TURRET, xy_loc(3, 12), 0),
        Token(UPGRADE, xy_loc(3, 12), 0),
        Token(DEP_DEMOLISHER, xy_loc(14, 0), COUNT_BUCKETS.index(3)),
        Token(DEP_SCOUT, xy_loc(13, 0), COUNT_BUCKETS.index(2)),
        END_TOKEN,
    ]
    cmds = encode_plan(plan, s)
    assert cmds == [
        (2, 3, 12), (7, 3, 12),
        (4, 14, 0), (4, 14, 0), (4, 14, 0),
        (3, 13, 0), (3, 13, 0),
    ]
    back = decode_commands(cmds, flip=False)
    assert back == plan
    # flip: same perspective plan for the top player lands mirrored-absolute
    cmds_f = encode_plan(plan, _scratch(sp=10.0, mp=8.3, flip=True))
    assert cmds_f[0] == (2, 24, 15) and cmds_f[2] == (4, 13, 27)
    assert decode_commands(cmds_f, flip=True) == plan


def test_decode_bucketing_canonical():
    cmds = [(3, 13, 0)] * 7                      # 7 scouts -> 5 + 2
    toks = decode_commands(cmds, flip=False)
    vals = [COUNT_BUCKETS[t.count] for t in toks if t.type == DEP_SCOUT]
    assert vals == [5, 2], vals
    assert toks[-1] == END_TOKEN


def test_mirror():
    plan = [
        Token(BUILD_WALL, xy_loc(0, 13), 0),
        Token(DEP_SCOUT, xy_loc(13, 0), 3),
        END_TOKEN,
    ]
    mirrored = mirror_plan(plan)
    assert loc_xy(mirrored[0].loc) == (27, 13)
    assert loc_xy(mirrored[1].loc) == (14, 0)
    assert mirror_plan(mirrored) == plan          # involution
    # legality is mirror-invariant on an empty board
    s = _scratch()
    for tok in mirrored[:-1]:
        s.apply(tok)                              # raises if illegal


def test_null_deploy_mask():
    lane = xy_loc(13, 0)
    ax, ay = 13, 0
    # fake pathfind: everything self-destructs after 3 steps inside own half
    def stuck(x, y):
        return [(x, y), (x, y + 1), (x, y + 2)] if (x, y) == (ax, ay) else \
               [(x, y)] * 6 + [(x, 20)]
    s = _scratch(pathfind=stuck)
    m = s.loc_mask(DEP_SCOUT)
    assert not m[lane]                            # provably null -> masked
    assert m.sum() == 27                          # every other edge tile stays legal
    off = _scratch(pathfind=stuck, null_deploy_mask=False)
    assert off.loc_mask(DEP_SCOUT)[lane]          # ablation switch works


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("\n{} / {} token tests green".format(len(fns), len(fns)))


if __name__ == "__main__":
    main()
