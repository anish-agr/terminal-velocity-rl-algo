"""The five scripted league bots (ARCHITECTURE §6.4, §14).

Each bot is a PURE function (game, player, config) -> [(kind, x, y), ...]
absolute engine commands:

    rush             mass scouts early — the classic turtle-killer
    funnel           wall row with a center gap, turrets ringing it, demolisher
                     chips through its own funnel
    demolisher_line  modest defense, banks to a demolisher wave down one lane
    turtle           interceptor-heavy active defense, scout counter on a bank
    torture          turn-scripted mechanics gauntlet (removals, upgrades,
                     trapped deploys, banking) — a deliberately weird opponent

Rules of this module: stateless across turns (anything time-varying derives
from game.turn), deterministic (no RNG anywhere), every cost read from config,
written in perspective space so each bot plays both seats identically. The
engine silently skips commands it cannot afford or place — bots emit their
wishlist in priority order and let the engine truncate.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from .tokens import Costs, to_abs, xy_loc

Command = Tuple[int, int, int]

# engine command kinds
K_WALL, K_SUPPORT, K_TURRET, K_SCOUT, K_DEMOLISHER, K_INTERCEPTOR = 0, 1, 2, 3, 4, 5
K_REMOVE, K_UPGRADE = 6, 7


def _occupied(game) -> set:
    return {(s[2], s[3]) for s in game.structures() if True}


def _own_structs(game, player) -> Dict[Tuple[int, int], Tuple[int, bool]]:
    """(x_abs, y_abs) -> (kind, upgraded) for the player's alive structures."""
    return {(s[2], s[3]): (s[0], s[5]) for s in game.structures() if s[1] == player}


def _emit(cmds: List[Command], kind: int, spots, flip: bool, occupied=None):
    """Append build commands for free perspective-space spots."""
    for (px, py) in spots:
        ax, ay = to_abs(px, py, flip)
        if occupied is not None and (ax, ay) in occupied:
            continue
        cmds.append((kind, ax, ay))


def _upgrades(cmds: List[Command], game, player, flip, spots):
    own = _own_structs(game, player)
    for (px, py) in spots:
        a = to_abs(px, py, flip)
        if a in own and not own[a][1]:
            cmds.append((K_UPGRADE, a[0], a[1]))


def _deploy(cmds: List[Command], kind, spot, flip, n):
    ax, ay = to_abs(spot[0], spot[1], flip)
    cmds.extend([(kind, ax, ay)] * int(n))


# ---------------------------------------------------------------------------

def rush(game, player: int, config: dict) -> List[Command]:
    """Thin defense; every turn it can field 5+ scouts, it sends everything."""
    costs = Costs(config)
    flip = player == 1
    occ = _occupied(game)
    cmds: List[Command] = []
    _emit(cmds, K_TURRET, [(3, 12), (24, 12), (13, 11), (14, 11)], flip, occ)
    mp = game.stats(player)[2]
    scouts = int(mp // costs.deploy_mp[0])
    if scouts >= 5:
        lane = (13, 0) if game.turn % 2 == 0 else (14, 0)
        _deploy(cmds, K_SCOUT, lane, flip, scouts)
    return cmds


def funnel(game, player: int, config: dict) -> List[Command]:
    """Wall row at y=13 with a two-tile gap; turrets ring the gap; demolishers
    chip through the funnel when a 3-wave is affordable."""
    costs = Costs(config)
    flip = player == 1
    occ = _occupied(game)
    cmds: List[Command] = []
    turrets = [(12, 12), (15, 12), (13, 11), (14, 11), (11, 11), (16, 11),
               (3, 12), (24, 12)]
    walls = [(x, 13) for x in range(28) if x not in (13, 14)]
    _emit(cmds, K_TURRET, turrets, flip, occ)
    _emit(cmds, K_WALL, walls, flip, occ)
    # harden the gap shoulders first (they take the most fire), then outward
    _upgrades(cmds, game, player, flip,
              [(12, 13), (15, 13), (11, 13), (16, 13)] + walls)
    mp = game.stats(player)[2]
    if mp >= 3 * costs.deploy_mp[1]:
        _deploy(cmds, K_DEMOLISHER, (14, 0), flip, int(mp // costs.deploy_mp[1]))
    return cmds


def demolisher_line(game, player: int, config: dict) -> List[Command]:
    """Corner turrets + upgraded corner walls; banks to a 4-demolisher wave."""
    costs = Costs(config)
    flip = player == 1
    occ = _occupied(game)
    cmds: List[Command] = []
    turrets = [(3, 12), (24, 12), (13, 11), (14, 11), (7, 11), (20, 11)]
    walls = [(0, 13), (1, 13), (26, 13), (27, 13), (3, 13), (24, 13)]
    _emit(cmds, K_TURRET, turrets, flip, occ)
    _emit(cmds, K_WALL, walls, flip, occ)
    _upgrades(cmds, game, player, flip, walls)
    mp = game.stats(player)[2]
    if mp >= 4 * costs.deploy_mp[1]:
        _deploy(cmds, K_DEMOLISHER, (13, 0), flip, int(mp // costs.deploy_mp[1]))
    return cmds


def turtle(game, player: int, config: dict) -> List[Command]:
    """Active defense: a standing interceptor screen every turn, upgraded
    turrets behind walls, scout counterattack only from a large bank."""
    costs = Costs(config)
    flip = player == 1
    occ = _occupied(game)
    cmds: List[Command] = []
    turrets = [(3, 12), (24, 12), (8, 11), (19, 11), (13, 11), (14, 11)]
    walls = [(0, 13), (1, 13), (26, 13), (27, 13), (8, 12), (19, 12),
             (13, 12), (14, 12)]
    _emit(cmds, K_TURRET, turrets, flip, occ)
    _emit(cmds, K_WALL, walls, flip, occ)
    _upgrades(cmds, game, player, flip, turrets)
    _upgrades(cmds, game, player, flip, walls)
    mp = game.stats(player)[2]
    icost = costs.deploy_mp[2]
    screens = [(6, 7), (21, 7), (13, 0)]
    if mp >= 3 * icost:
        for spot in screens:
            _deploy(cmds, K_INTERCEPTOR, spot, flip, 1)
        mp -= 3 * icost
    if mp >= 12.0:
        _deploy(cmds, K_SCOUT, (14, 0), flip, int(mp // costs.deploy_mp[0]))
    return cmds


def torture(game, player: int, config: dict) -> List[Command]:
    """Deterministic mechanics gauntlet on a 12-turn cycle: builds throwaway
    wall lines, marks them for removal, upgrades mid-game, deploys into its own
    pocket (trapped self-destructs), and banks MP across phases. Exists to keep
    the league honest about weird-but-legal play (§6.4), mirroring the fidelity
    corpus bot in bots/torture/."""
    costs = Costs(config)
    flip = player == 1
    occ = _occupied(game)
    own = _own_structs(game, player)
    cmds: List[Command] = []
    phase = game.turn % 12

    _emit(cmds, K_TURRET, [(3, 12), (24, 12), (13, 10), (14, 10)], flip, occ)

    if phase in (0, 1, 2):
        # throwaway forward wall segment + a pocket that traps our own deploys
        _emit(cmds, K_WALL, [(x, 13) for x in range(9, 13)], flip, occ)
        _emit(cmds, K_WALL, [(5, 9), (7, 9), (6, 10)], flip, occ)   # pocket walls
    elif phase == 3:
        # mark the forward segment for removal (executes next restore)
        for x in range(9, 13):
            a = to_abs(x, 13, flip)
            if a in own:
                cmds.append((K_REMOVE, a[0], a[1]))
    elif phase in (4, 5):
        _emit(cmds, K_SUPPORT, [(13, 3), (14, 3)], flip, occ)
        _upgrades(cmds, game, player, flip, [(13, 3), (14, 3), (3, 12), (24, 12)])
    # phases 6-7: bank MP (deploy nothing)
    mp = game.stats(player)[2]
    if phase == 8 and mp >= 1.0:
        _deploy(cmds, K_SCOUT, (6, 7), flip, 1)     # into the pocket: trapped SD
    elif phase in (9, 10, 11) and mp >= 6.0:
        _deploy(cmds, K_DEMOLISHER, (14, 0), flip, int(mp // costs.deploy_mp[1]))
        _deploy(cmds, K_INTERCEPTOR, (13, 0), flip, 1)
    return cmds


SCRIPTED_BOTS: Dict[str, Callable] = {
    "rush": rush,
    "funnel": funnel,
    "demolisher_line": demolisher_line,
    "turtle": turtle,
    "torture": torture,
}
