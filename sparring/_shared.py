"""Shared bootstrap for the sparring panel.

The sparring bots are fixed yardsticks: deterministic, non-adaptive, each a
pure archetype. Their job is to answer "did this change make our real bot
worse against anything?" -- which only works if they never change behaviour
between runs. Hence two hard rules for every bot in this directory:

  1. NO randomness. Every decision is a pure function of the visible board.
     (The frozen starter bot is grandfathered in with a pinned seed.)
  2. NO adaptation across games. A loss to a panel bot must always mean the
     same weakness.

Bots reuse the real python-algo/gamelib via bootstrap_gamelib() so there is
exactly one engine interface in the repo and it cannot drift.
"""

from __future__ import annotations

import os
import sys

# Resource slots, matching gamelib's convention.
SP = 0
MP = 1


def bootstrap_gamelib():
    """Make `import gamelib` resolve to python-algo/gamelib."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    algo_dir = os.path.join(repo, "python-algo")
    if algo_dir not in sys.path:
        sys.path.insert(0, algo_dir)


def unit_shorthands(config):
    """Unit shorthand strings, read from config -- never hardcoded."""
    info = config["unitInformation"]
    return {
        "WALL": info[0]["shorthand"],
        "SUPPORT": info[1]["shorthand"],
        "TURRET": info[2]["shorthand"],
        "SCOUT": info[3]["shorthand"],
        "DEMOLISHER": info[4]["shorthand"],
        "INTERCEPTOR": info[5]["shorthand"],
    }


def mobile_cost(config, unit_idx):
    return config["unitInformation"][unit_idx].get("cost2", 0.0)


def least_damage_lane(game_state, options, turret_damage):
    """Deterministically pick the spawn among `options` whose path eats the
    least turret fire. Ties break toward the earlier option, so the result is
    a pure function of the board."""
    best, best_damage = None, None
    for location in options:
        if game_state.contains_stationary_unit(location):
            continue
        path = game_state.find_path_to_edge(location)
        if not path:
            continue
        damage = sum(
            len(game_state.get_attackers(step, 0)) * turret_damage for step in path
        )
        if best_damage is None or damage < best_damage:
            best, best_damage = location, damage
    return best
