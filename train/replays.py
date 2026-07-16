"""Scraped-replay ingestion (ARCHITECTURE §7, §13 Stage A).

Reconstructs, for every turn of every same-config replay, both sides'
(decision state, executed plan, outcome) — the raw material for BC warm start
(winners only, fingerprint-capped), opponent-prediction training (ALL sides),
and value/aux targets (both sides).

Command reconstruction from a replay (nothing in the file says "commands"):
  builds/deploys  structure/mobile spawn events in the turn's action frames
  upgrades        diff of the upgrade-marker list (pXUnits[7]) between this
                  turn's snapshot and the next (an upgrade whose structure dies
                  the same turn is missed — rare, accepted)
  removals        was_removed deaths appear in the NEXT turn frame (they
                  execute at the following restore), so each is attributed
                  back to the turn the mark was issued

Memory design: 3K replays x ~100 turns x 2 sides x ~56 KB of tensors is ~30 GB
— far too big to materialize. load_game() keeps compact per-turn records;
iter_positions() featurizes lazily IN TURN ORDER (deploy-history planes need
the running past); the learner samples games, not positions, and walks them.

Gate (§0.3): `python -m train.replays --check replays/scraped` parses every
replay, featurizes both sides of every turn, and cross-checks winner labels
against endStats. Run it on the pod after rsyncing the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

# scripts/ is not a package; it is the format authority for .replay files.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))
from replay_utils import load_replay  # noqa: E402

from .features import DeployHistory, GRID, N_PLANES, N_SCALARS  # noqa: E402
from .tokens import Costs, Token, decode_commands, in_arena  # noqa: E402

_STRUCT_KINDS = (0, 1, 2)
_MOBILE_KINDS = (3, 4, 5)
_UPGRADE_LIST_IDX = 7

# gameplay-relevant config fields (icon/display fields differ harmlessly)
_UNIT_FIELDS = (
    "cost1", "cost2", "startHealth", "attackDamageWalker", "attackDamageTower",
    "attackRange", "shieldPerUnit", "shieldRange", "shieldBonusPerY", "speed",
    "selfDestructDamageWalker", "selfDestructDamageTower", "selfDestructRange",
    "selfDestructStepsRequired", "playerBreachDamage", "refundPercentage",
)


class TurnRecord(NamedTuple):
    turn: int
    # per side (index 0 = player1, 1 = player2):
    structures: Tuple[tuple, ...]      # (kind, owner0based, x, y, hp, upgraded, False)
    stats: Tuple[Tuple[float, float, float], ...]   # (hp, sp, mp) per side
    commands: Tuple[Tuple[tuple, ...], ...]         # engine (kind,x,y) per side
    flows: Tuple[Tuple[float, float], ...]          # (breach_dealt, struct_dmg_dealt)


class GameRecord(NamedTuple):
    path: str
    winner: int                        # 0-based side, -1 tie
    turns: List[TurnRecord]
    fingerprints: Tuple[str, str]      # per side: turn 0-3 build-sequence hash


class Position(NamedTuple):
    board: np.ndarray                  # [18,28,28] f32
    scalars: np.ndarray                # [14] f32
    plan: Tuple[Token, ...]
    z: float                           # +1 win / -1 loss / 0 tie, this side
    aux: np.ndarray                    # [3] f32 — Δ_3 targets (§4.3)
    side: int
    turn: int
    structures: Tuple[tuple, ...]      # scratch ingredients for mask rebuild
    sp: float
    mp: float


# ---------------------------------------------------------------------------
# Config equivalence
# ---------------------------------------------------------------------------

def config_matches(ours: dict, theirs: dict) -> bool:
    """Gameplay-field equality on unitInformation + resources (§7 scraping)."""
    try:
        a, b = ours["unitInformation"], theirs["unitInformation"]
        if len(a) != len(b):
            return False
        for ua, ub in zip(a, b):
            for f in _UNIT_FIELDS:
                if ua.get(f) != ub.get(f):
                    return False
            upa, upb = ua.get("upgrade", {}), ub.get("upgrade", {})
            for f in _UNIT_FIELDS:
                if upa.get(f) != upb.get(f):
                    return False
        return ours["resources"] == theirs["resources"]
    except (KeyError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Replay -> GameRecord
# ---------------------------------------------------------------------------

def load_game(path: str, config: dict) -> Optional[GameRecord]:
    """Parse one replay into a compact GameRecord; None if the embedded config
    differs on gameplay fields or the file is structurally unusable."""
    replay = load_replay(path)
    if replay.config is None or not config_matches(config, replay.config):
        return None
    deploy_frames = replay.deploy_frames()
    if len(deploy_frames) < 2:
        return None
    by_turn = replay.turns()

    # upgrade-marker sets per snapshot, per side: {(x, y), ...}
    upg_sets: List[Tuple[set, set]] = [
        (
            {(u.x, u.y) for u in f.units(1, _UPGRADE_LIST_IDX)},
            {(u.x, u.y) for u in f.units(2, _UPGRADE_LIST_IDX)},
        )
        for f in deploy_frames
    ]

    turns: List[TurnRecord] = []
    build_seqs: Tuple[List[tuple], List[tuple]] = ([], [])

    for t in range(len(deploy_frames) - 1):
        f_now, f_next = deploy_frames[t], deploy_frames[t + 1]
        frames = by_turn.get(f_now.turn, [])

        structures = []
        for pid in (1, 2):
            groups = f_now.units(pid)
            upgraded = upg_sets[t][pid - 1]
            for kind in _STRUCT_KINDS:
                for u in groups[kind]:
                    structures.append(
                        (kind, pid - 1, u.x, u.y, u.health,
                         (u.x, u.y) in upgraded, False)
                    )

        stats = tuple(
            (f_now.stats(pid).health, f_now.stats(pid).structure_points,
             f_now.stats(pid).mobile_points)
            for pid in (1, 2)
        )

        # spawns during this turn's action phase -> builds + deploys, in order
        builds: Tuple[List[tuple], List[tuple]] = ([], [])
        deploys: Tuple[List[tuple], List[tuple]] = ([], [])
        for fr in frames:
            for s in fr.spawns():
                if s.unit_type in _STRUCT_KINDS:
                    builds[s.player - 1].append((s.unit_type, s.x, s.y))
                elif s.unit_type in _MOBILE_KINDS:
                    deploys[s.player - 1].append((s.unit_type, s.x, s.y))
        # upgrades issued this turn: marker-set growth into the next snapshot
        upgrades = tuple(
            [(7, x, y) for (x, y) in sorted(upg_sets[t + 1][i] - upg_sets[t][i])]
            for i in (0, 1)
        )
        # removal marks issued this turn: was_removed deaths in the NEXT frame
        removes: Tuple[List[tuple], List[tuple]] = ([], [])
        for d in f_next.deaths():
            if d.was_removed and d.unit_type in _STRUCT_KINDS:
                removes[d.player - 1].append((6, d.x, d.y))

        commands = tuple(
            tuple(builds[i] + upgrades[i] + removes[i] + deploys[i]) for i in (0, 1)
        )
        if t < 4:
            for i in (0, 1):
                build_seqs[i].extend(builds[i])

        # damage flows: breach events credit the BREACHER's side; structure
        # damage events carry the DAMAGED owner -> dealt = opponent's taken
        breach_dealt = [0.0, 0.0]
        struct_taken = [0.0, 0.0]
        for fr in frames:
            for b in fr.breaches():
                breach_dealt[b.player - 1] += b.damage
            for dmg in fr.damages():
                if dmg.unit_type in _STRUCT_KINDS:
                    struct_taken[dmg.player - 1] += dmg.damage
        flows = tuple(
            (breach_dealt[i], struct_taken[1 - i]) for i in (0, 1)
        )

        turns.append(TurnRecord(f_now.turn, tuple(structures), stats, commands, flows))

    res = replay.final_result()
    engine_w = res.get("engine_winner")
    if res["winner"] != 0:
        winner = res["winner"] - 1
    elif engine_w in (1, 2):
        winner = int(engine_w) - 1      # health tie: engine's compute tiebreak
    else:
        winner = -1

    fps = tuple(
        hashlib.sha1(repr(seq).encode()).hexdigest()[:16] for seq in build_seqs
    )
    return GameRecord(path, winner, turns, fps)


# ---------------------------------------------------------------------------
# GameRecord -> positions (lazy, turn-ordered)
# ---------------------------------------------------------------------------

_ARENA = np.zeros((GRID, GRID), dtype=np.float32)
_OWN_HALF_PLANE = np.zeros((GRID, GRID), dtype=np.float32)
for _x in range(GRID):
    for _y in range(GRID):
        if in_arena(_x, _y):
            _ARENA[_x, _y] = 1.0
            if _y < 14:
                _OWN_HALF_PLANE[_x, _y] = 1.0


def _board_from_units(config, structures, side: int) -> np.ndarray:
    """Planes 0-11 rebuilt from a unit list (mirrors the sim bridge layout)."""
    info = config["unitInformation"]
    norm = [
        float(info[k].get("upgrade", {}).get("startHealth",
                                             info[k].get("startHealth", 1.0)))
        for k in _STRUCT_KINDS
    ]
    board = np.zeros((12, GRID, GRID), dtype=np.float32)
    flip = side == 1
    for (kind, owner, ax, ay, hp, upgraded, _pending) in structures:
        x, y = (GRID - 1 - ax, GRID - 1 - ay) if flip else (ax, ay)
        base = 0 if owner == side else 5
        board[base + kind, x, y] = hp / norm[kind]
        if upgraded:
            board[base + 3, x, y] = 1.0
    board[10] = _ARENA
    board[11] = _OWN_HALF_PLANE
    return board


def _networth(config, structures, owner: int) -> float:
    """Invested SP x health%, summed over a side's structures (§4.3)."""
    costs = Costs(config)
    info = config["unitInformation"]
    total = 0.0
    for (kind, own, _x, _y, hp, upgraded, _p) in structures:
        if own != owner:
            continue
        invested = costs.build_sp[kind] + (costs.upgrade_sp[kind] if upgraded else 0.0)
        max_hp = float(info[kind].get("upgrade", {}).get("startHealth",
                                                         info[kind]["startHealth"])
                       if upgraded else info[kind]["startHealth"])
        total += invested * (hp / max_hp if max_hp > 0 else 0.0)
    return total


def _aux_targets(config, turns: List[TurnRecord], t: int, side: int) -> np.ndarray:
    """Δ over 3 turns (clipped at game end) of hp diff, net-worth diff,
    resource-total diff — each normalized per §4.3."""
    t3 = min(t + 3, len(turns) - 1)
    now, fut = turns[t], turns[t3]

    def snapshot(tr: TurnRecord):
        hp = tr.stats[side][0] - tr.stats[1 - side][0]
        nw = _networth(config, tr.structures, side) - _networth(
            config, tr.structures, 1 - side)
        res = sum(tr.stats[side][1:]) - sum(tr.stats[1 - side][1:])
        return hp, nw, res

    h0, n0, r0 = snapshot(now)
    h1, n1, r1 = snapshot(fut)
    return np.array(
        [(h1 - h0) / 10.0, (n1 - n0) / 50.0, (r1 - r0) / 20.0], dtype=np.float32
    )


def iter_positions(
    record: GameRecord, config: dict, sides: Sequence[int] = (0, 1)
) -> Iterator[Position]:
    """Featurize a game lazily, in turn order (history planes need the past)."""
    hists = {s: DeployHistory(config) for s in sides}
    for t, tr in enumerate(record.turns):
        for s in sides:
            board = np.empty((N_PLANES, GRID, GRID), dtype=np.float32)
            board[:12] = _board_from_units(config, tr.structures, s)
            hist = hists[s]
            hplanes = np.concatenate(
                [np.clip(hist.last / 10.0, 0.0, 1.0), hist.ema], axis=0
            )
            if s == 1:
                hplanes = hplanes[:, ::-1, ::-1]
            board[12:] = hplanes

            hp, sp, mp = tr.stats[s]
            ehp, esp, emp = tr.stats[1 - s]
            scalars = np.array(
                [
                    hp / 30.0, sp / 40.0, mp / 15.0,
                    ehp / 30.0, esp / 40.0, emp / 15.0,
                    tr.turn / 100.0,
                    hist.income(tr.turn) / 10.0,
                    hist.banked_mp(mp, tr.turn) / 15.0,
                    hist.banked_mp(emp, tr.turn) / 15.0,
                    hist.breach_dealt / 5.0, hist.breach_taken / 5.0,
                    hist.struct_dmg_dealt / 50.0, hist.struct_dmg_taken / 50.0,
                ],
                dtype=np.float32,
            )

            plan = tuple(decode_commands(tr.commands[s], flip=(s == 1)))
            z = 0.0 if record.winner < 0 else (1.0 if record.winner == s else -1.0)
            yield Position(
                board, scalars, plan, z,
                _aux_targets(config, record.turns, t, s),
                s, tr.turn, tr.structures, sp, mp,
            )
        # advance both histories with this turn's observed deploys + flows
        for s in sides:
            enemy = 1 - s
            hists[s].record_turn(
                [c for c in tr.commands[enemy] if c[0] in _MOBILE_KINDS],
                tr.flows[s][0], tr.flows[enemy][0],
                tr.flows[s][1], tr.flows[enemy][1],
            )


# ---------------------------------------------------------------------------
# Corpus indexing: winner filter + fingerprint caps (§7.3)
# ---------------------------------------------------------------------------

def build_bc_index(
    replay_dir: str, config: dict, cap_frac: float = 0.25
) -> List[Tuple[str, int]]:
    """(path, winning_side) pairs for BC, with no opening fingerprint owning
    more than cap_frac of the dataset. Ties are excluded (no winner to imitate)."""
    entries: List[Tuple[str, int, str]] = []
    for name in sorted(os.listdir(replay_dir)):
        if not name.endswith(".replay"):
            continue
        rec = load_game(os.path.join(replay_dir, name), config)
        if rec is None or rec.winner < 0:
            continue
        entries.append((rec.path, rec.winner, rec.fingerprints[rec.winner]))

    cap = max(1, int(len(entries) * cap_frac))
    by_fp: Dict[str, int] = defaultdict(int)
    out: List[Tuple[str, int]] = []
    for path, side, fp in entries:
        if by_fp[fp] >= cap:
            continue
        by_fp[fp] += 1
        out.append((path, side))
    return out


# ---------------------------------------------------------------------------
# §0.3 gate
# ---------------------------------------------------------------------------

def check_corpus(replay_dir: str, config: dict, limit: Optional[int] = None) -> bool:
    """Parse + featurize every replay; verify winner labels match endStats."""
    names = [n for n in sorted(os.listdir(replay_dir)) if n.endswith(".replay")]
    if limit:
        names = names[:limit]
    ok = skipped = failed = 0
    for name in names:
        path = os.path.join(replay_dir, name)
        try:
            rec = load_game(path, config)
            if rec is None:
                skipped += 1
                continue
            n_pos = 0
            for pos in iter_positions(rec, config):
                assert pos.board.shape == (N_PLANES, GRID, GRID)
                assert pos.scalars.shape == (N_SCALARS,)
                assert np.isfinite(pos.board).all() and np.isfinite(pos.scalars).all()
                assert pos.plan[-1].type == 8  # END-terminated
                n_pos += 1
            assert n_pos == 2 * len(rec.turns)
            ok += 1
        except Exception as exc:  # gate tool: report, never crash the sweep
            failed += 1
            print("FAIL {}: {!r}".format(name, exc))
    print("corpus check: {} ok, {} config-skipped, {} failed / {} files".format(
        ok, skipped, failed, len(names)))
    return failed == 0


def main():
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", metavar="DIR", help="run the §0.3 corpus gate")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=os.path.join(_REPO, "game-configs.json"))
    args = ap.parse_args()
    with open(args.config) as fh:
        config = json.load(fh)
    if args.check:
        raise SystemExit(0 if check_corpus(args.check, config, args.limit) else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
