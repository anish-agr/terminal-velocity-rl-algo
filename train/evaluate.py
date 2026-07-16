"""Gauntlet + promotion rule (ARCHITECTURE §8).

A gauntlet is n_games per opponent, seats alternating (half as P1, half as P2),
against {every scripted bot, the BC anchor, the previous best checkpoint}.
Per-opponent metrics: win/loss rate, mean health margin, crash + timeout counts.

Promotion is MIN-based, not mean-based (single-elimination logic): promote iff
    win-rate vs previous best >= 55%
AND win-rate vs EVERY scripted bot >= 85%
AND zero crashes AND zero timeouts.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .actor import play_game
from .tokens import Costs

TIMEOUT_S = 5.0


def run_gauntlet(
    game_factory: Callable,
    make_client: Callable[[str], object],
    opponents: Sequence[Tuple[str, str]],
    cfg: dict,
    config: dict,
    n_games: int,
    seed: int = 0,
    k: Optional[int] = None,
    m: Optional[int] = None,
) -> dict:
    """opponents: list of ("scripted", bot_name) or ("client", model_id).
    The challenger is always model_id "current". Returns the report dict."""
    costs = Costs(config)
    rng = np.random.default_rng(seed)
    tau = float(cfg["search"].get("tau_deploy", 0.5))
    report: Dict[str, dict] = {}

    for kind, name in opponents:
        wins = losses = ties = crashes = timeouts = 0
        margins: List[float] = []
        for i in range(n_games):
            me = i % 2                                # alternate seats
            opp = 1 - me
            clients = {me: make_client("current")}
            scripted: Dict[int, str] = {}
            if kind == "scripted":
                clients[opp] = None
                scripted[opp] = name
            else:
                clients[opp] = make_client(name)
            try:
                meta, _ = play_game(
                    game_factory, clients, scripted, (), cfg, config, costs,
                    rng, k=k, m=m, tau=tau,
                )
            except Exception:
                crashes += 1
                continue
            if meta["max_turn_s"] > TIMEOUT_S:
                timeouts += 1
            margin = meta["final_hp"][me] - meta["final_hp"][opp]
            margins.append(float(margin))
            if meta["winner"] == me:
                wins += 1
            elif meta["winner"] == opp:
                losses += 1
            else:
                ties += 1
        played = max(1, wins + losses + ties)
        report["{}:{}".format(kind, name)] = {
            "kind": kind, "name": name, "games": n_games,
            "wins": wins, "losses": losses, "ties": ties,
            "win_rate": wins / played, "loss_rate": losses / played,
            "mean_margin": float(np.mean(margins)) if margins else 0.0,
            "crashes": crashes, "timeouts": timeouts,
        }
    report["_meta"] = {"n_games": n_games, "t": time.time()}
    return report


def should_promote(report: dict, cfg: dict,
                   prev_best_key: str = "client:prev_best") -> Tuple[bool, str]:
    """Apply the §8 rule to a gauntlet report. Returns (promote, reason)."""
    ec = cfg["evaluation"]
    need_best = float(ec["promote_vs_best"])
    need_scripted = float(ec["promote_vs_scripted"])
    max_crashes = int(ec["promote_max_crashes"])

    for key, r in report.items():
        if key.startswith("_"):
            continue
        if r["crashes"] > max_crashes or r["timeouts"] > 0:
            return False, "{}: {} crashes / {} timeouts".format(
                key, r["crashes"], r["timeouts"])
        if r["kind"] == "scripted" and r["win_rate"] < need_scripted:
            return False, "{}: win-rate {:.2f} < {:.2f}".format(
                key, r["win_rate"], need_scripted)
    prev = report.get(prev_best_key)
    if prev is not None and prev["win_rate"] < need_best:
        return False, "vs previous best: {:.2f} < {:.2f}".format(
            prev["win_rate"], need_best)
    return True, "all gates passed"


def save_report(report: dict, run_dir: str) -> str:
    path = os.path.join(run_dir, "eval", "report.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=2)
    os.replace(tmp, path)
    return path
