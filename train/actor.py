"""Self-play actor (ARCHITECTURE §5.4-§5.5): plays whole games, emits trajectories.

One actor process = a loop of complete games. Each game:
  1. Sample the opponent controller from the league state file (§6.4) —
     current theta / PFSP snapshot / scripted bot / BC anchor.
  2. Every turn, each net-controlled side runs the §5 search (search.choose)
     through its NetClient; scripted sides call their pure function.
  3. Both sides' deploy histories advance from the ACTUAL executed commands.
  4. Positions are recorded for current-theta sides only (both seats in mirror
     games); z and the aux Delta_3 targets are filled in when the game ends.
  5. Resignation (§5.4): value(current state) < resign_v for 3 consecutive own
     turns -> resign, unless this game is in the 10% exemption quota
     (value-blind-spot insurance).

Everything is injected (game_factory, clients, rng), so the full loop runs
locally against a fake sim; the pod runs it with terminal_sim.Game and
QueueClients against the GPU server.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .features import DeployHistory, build_planes
from .league import League
from .scripted import SCRIPTED_BOTS
from .search import NetClient, choose
from .tokens import Costs, ScratchSpec, Token, decode_commands, encode_plan

_MOBILE_KINDS = (3, 4, 5)


# ---------------------------------------------------------------------------
# Per-turn bookkeeping helpers
# ---------------------------------------------------------------------------

def _networth(config: dict, structures, owner: int) -> float:
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


def _aux_snapshot(config: dict, game, side: int) -> Tuple[float, float, float]:
    """(hp diff, net-worth diff, resource-total diff) from `side`'s view."""
    s, e = game.stats(side), game.stats(1 - side)
    structures = game.structures()
    return (
        s[0] - e[0],
        _networth(config, structures, side) - _networth(config, structures, 1 - side),
        (s[1] + s[2]) - (e[1] + e[2]),
    )


def _aux_targets(snapshots: List[Tuple[float, float, float]], t: int) -> np.ndarray:
    t3 = min(t + 3, len(snapshots) - 1)
    h0, n0, r0 = snapshots[t]
    h1, n1, r1 = snapshots[t3]
    return np.array(
        [(h1 - h0) / 10.0, (n1 - n0) / 50.0, (r1 - r0) / 20.0], dtype=np.float32
    )


# ---------------------------------------------------------------------------
# One full game
# ---------------------------------------------------------------------------

def play_game(
    game_factory: Callable,
    clients: Dict[int, Optional[NetClient]],
    scripted: Dict[int, str],
    record_sides: Tuple[int, ...],
    cfg: dict,
    config: dict,
    costs: Costs,
    rng: np.random.Generator,
    k: Optional[int] = None,
    m: Optional[int] = None,
    tau: float = 1.0,
    max_turns: int = 100,
    record_scripted: bool = False,
) -> Tuple[dict, List[dict]]:
    """Play one game. clients[side] is a NetClient or None; scripted[side]
    names a SCRIPTED_BOTS entry for None-client sides. Returns (meta, positions).

    record_scripted: also record positions for scripted sides, with EMPTY
    candidate lists (no policy target — §6.3 cold-start value/aux/prediction
    data only; the learner skips policy loss for candidate-less positions).
    """
    scfg = cfg["search"]
    resign_v = float(scfg["resign_v"])
    resign_n = int(scfg["resign_consecutive"])
    exempt = rng.random() < float(scfg["resign_exempt_frac"])

    game = game_factory()
    hists = {0: DeployHistory(config), 1: DeployHistory(config)}
    prev_plans: Dict[int, Optional[tuple]] = {0: None, 1: None}
    resign_streak = {0: 0, 1: 0}
    aux_snaps: Dict[int, List[Tuple[float, float, float]]] = {s: [] for s in record_sides}
    positions: List[dict] = []
    resigned: Optional[int] = None

    max_turn_s = 0.0

    while not game.game_over() and game.turn < max_turns and resigned is None:
        t_turn = time.monotonic()
        turn_cmds: Dict[int, list] = {}
        turn_meta: Dict[int, dict] = {}

        for side in (0, 1):
            client = clients.get(side)
            if client is None:
                turn_cmds[side] = SCRIPTED_BOTS[scripted[side]](game, side, config)
                continue

            plan, pi_star, diag = choose(
                game, client, cfg, side,
                hists[side], hists[1 - side], config, costs,
                prev_opp_plan=prev_plans[1 - side],
                k=k, m=m, tau=tau,
            )
            spec = ScratchSpec(
                costs, game.structures(),
                game.stats(side)[1], game.stats(side)[2],
                side == 1, side,
            )
            turn_cmds[side] = encode_plan(list(plan), spec())
            turn_meta[side] = {"plan": plan, "pi_star": pi_star, "diag": diag}

        # record decision states BEFORE the turn resolves
        for side in record_sides:
            if side not in turn_meta and not record_scripted:
                continue
            board, scalars = build_planes(game, side, hists[side])
            board_opp, scalars_opp = build_planes(game, 1 - side, hists[1 - side])
            aux_snaps[side].append(_aux_snapshot(config, game, side))
            pi = turn_meta[side]["pi_star"] if side in turn_meta else {}
            positions.append({
                "board": board, "scalars": scalars,
                "structures": tuple(game.structures()),
                "sp": game.stats(side)[1], "mp": game.stats(side)[2],
                "side": side, "turn": game.turn,
                "candidates": list(pi.keys()), "pi": list(pi.values()),
                "opp_board": board_opp, "opp_scalars": scalars_opp,
                "opp_structures": tuple(game.structures()),
                "opp_sp": game.stats(1 - side)[1], "opp_mp": game.stats(1 - side)[2],
                "opp_plan": None,   # filled below once the opponent commits
                "z": 0.0, "aux": None,
            })

        result = game.play_turn(turn_cmds[0], turn_cmds[1])
        frames, b1, b2, d1, d2 = result
        max_turn_s = max(max_turn_s, time.monotonic() - t_turn)

        # histories + literal-previous plans from what actually executed
        for side in (0, 1):
            enemy = 1 - side
            enemy_deploys = [c for c in turn_cmds[enemy] if c[0] in _MOBILE_KINDS]
            own_b, opp_b = (b1, b2) if side == 0 else (b2, b1)
            own_d, opp_d = (d1, d2) if side == 0 else (d2, d1)
            hists[side].record_turn(enemy_deploys, own_b, opp_b, own_d, opp_d)
            prev_plans[side] = tuple(
                decode_commands(list(turn_cmds[side]), flip=(side == 1))
            )

        # attach the opponent's executed plan to this turn's recorded positions
        for pos in positions:
            if pos["opp_plan"] is None and pos["turn"] == game.turn - 1:
                pos["opp_plan"] = prev_plans[1 - pos["side"]]

        # resignation check on the post-turn state (§5.4)
        if not exempt and not game.game_over():
            for side in record_sides:
                client = clients.get(side)
                if client is None:
                    continue
                board, scalars = build_planes(game, side, hists[side])
                v = float(client.values(board[None], scalars[None])[0])
                resign_streak[side] = resign_streak[side] + 1 if v < resign_v else 0
                if resign_streak[side] >= resign_n:
                    resigned = side
                    break

    # -- outcomes --------------------------------------------------------------
    if resigned is not None:
        winner = 1 - resigned
    else:
        winner = game.winner()          # 0 | 1 | 2 (tie) | -1 (turn cap, use hp)
        if winner in (-1, 2):
            hp0, hp1 = game.stats(0)[0], game.stats(1)[0]
            winner = 0 if hp0 > hp1 else (1 if hp1 > hp0 else -1)

    turn_index = {s: 0 for s in record_sides}
    for pos in positions:
        s = pos["side"]
        pos["z"] = 0.0 if winner < 0 else (1.0 if winner == s else -1.0)
        pos["aux"] = _aux_targets(aux_snaps[s], turn_index[s])
        turn_index[s] += 1
        if pos["opp_plan"] is None:     # final turn's opponent plan
            pos["opp_plan"] = prev_plans[1 - s] or (Token(8, 0, 0),)

    meta = {
        "winner": winner,
        "turns": game.turn,
        "resigned": resigned,
        "exempt": exempt,
        "final_hp": (game.stats(0)[0], game.stats(1)[0]),
        "max_turn_s": max_turn_s,
    }
    return meta, positions


# ---------------------------------------------------------------------------
# Actor process
# ---------------------------------------------------------------------------

def run_actor(
    actor_id: int,
    game_factory: Callable,
    make_client: Callable[[str], NetClient],
    trajectory_q,
    cfg: dict,
    config: dict,
    league_path: str,
    seed: int,
    n_games: Optional[int] = None,
    tau_fn: Optional[Callable[[int], float]] = None,
) -> None:
    """Process target. make_client(model_id) builds a NetClient bound to that
    model ("current", "snap00xx", "bc"). league_path is the JSON the learner
    maintains; it is re-read every game so the pool stays fresh.

    n_games bounds the loop for tests; None runs until the process is killed.
    """
    rng = np.random.default_rng(seed)
    costs = Costs(config)
    league = League(cfg)
    played = 0

    while n_games is None or played < n_games:
        if os.path.exists(league_path):
            try:
                league.load(league_path)
            except Exception:
                pass  # racing the learner's atomic rewrite; last state is fine

        kind, detail = league.sample_opponent(rng)
        me = int(rng.integers(2))       # play both seats over time
        opp = 1 - me

        clients: Dict[int, Optional[NetClient]] = {me: make_client("current")}
        scripted: Dict[int, str] = {}
        if kind == "current":
            clients[opp] = make_client("current")
            record = (0, 1)             # mirror: both seats are current theta
        elif kind == "snapshot":
            clients[opp] = make_client(detail)
            record = (me,)
        elif kind == "anchor":
            clients[opp] = make_client("bc")
            record = (me,)
        else:                           # scripted
            clients[opp] = None
            scripted[opp] = detail
            record = (me,)

        tau = tau_fn(played) if tau_fn else float(cfg["search"]["tau_act_start"])
        try:
            meta, positions = play_game(
                game_factory, clients, scripted, record, cfg, config, costs,
                rng, k=int(cfg["search"]["k_train"]),
                m=int(cfg["search"]["m_train"]), tau=tau,
            )
        except Exception as exc:
            # a transient failure (server hiccup, unknown snapshot model, one
            # bad state) must cost ONE game, not this actor for the whole run
            print("actor {} game vs {}:{} failed: {!r}".format(
                actor_id, kind, detail, exc), flush=True)
            time.sleep(5.0)
            played += 1
            continue
        meta.update({
            "actor": actor_id, "opponent_kind": kind, "opponent": detail,
            "me": me, "t_done": time.time(),
        })
        trajectory_q.put((meta, positions))
        played += 1
