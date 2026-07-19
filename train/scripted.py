"""The scripted league bots (ARCHITECTURE §6.4, §14).

Each bot is a PURE function (game, player, config) -> [(kind, x, y), ...]
absolute engine commands:

    rush             mass scouts early — the classic turtle-killer
    funnel           wall row with a center gap, turrets ringing it, demolisher
                     chips through its own funnel
    demolisher_line  modest defense, banks to a demolisher wave down one lane
    turtle           interceptor-heavy active defense, scout counter on a bank
    torture          turn-scripted mechanics gauntlet (removals, upgrades,
                     trapped deploys, banking) — a deliberately weird opponent
    corner_hammer    the ranked-meta distillation (2026-07-18 replay study of
                     the visible #1 vs the hidden >2000 pool): winners' full
                     turret line + banked scout waves, plus the corner/flank
                     hardening and funnel the whole field under-builds
    line_grinder     the mid-ladder counter-meta (2026-07-18 study of our own
                     ranked losses): SOLID turret line + gated demolisher
                     grind waves — the archetype our net lost 4 of 5 ranked
                     losses to and has no learned answer for

Rules of this module: stateless across turns (anything time-varying derives
from game.turn), deterministic (no RNG anywhere), every cost read from config,
written in perspective space so each bot plays both seats identically. The
engine silently skips commands it cannot afford or place — bots emit their
wishlist in priority order and let the engine truncate.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from .tokens import Costs, from_abs, to_abs, xy_loc

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


# ---------------------------------------------------------------------------
# corner_hammer — the ranked-meta distillation (2026-07-18 replay study)
# ---------------------------------------------------------------------------
#
# All layout constants are in perspective space and were extracted from the 19
# distinct games of the visible #1 seed vs the (hidden) >2000 pool:
#   * winners run a full turret line across y=13 with corner WALLS at (0,13)/
#     (27,13), thin y=12 flank backing, and supports shielding the spawn cells;
#   * winners bank MP to ~0.875 x income/decay (18 MP at 5 income — exactly
#     full banking from turn 0, reached at turn 7) then commit the whole bank
#     as one scout wave every 3-8 turns from (13,0)/(14,0) (+ a split stack);
#   * EVERY defense in the sample — the #1's included — leaks at the corners
#     (aggregate breach lanes L:231 C:210 R:81, hot cells (0,13),(3,10),(4,9)),
#     so the corners get the extra turrets + the first upgrades here.
#
# Geometry invariant: columns x=13,14 stay COMPLETELY empty on our half so our
# own waves path straight up and out through the (13,13)/(14,13) gap, while
# enemy mobiles crossing anywhere must converge into that same gap and run the
# ring-turret gauntlet down the open column. (The funnel bot above violates
# this — its ring seals the gap from below — which is left as-is: its value to
# the league is its defensive shape, not its offense.)

_CH_CORNER_WALLS = ((0, 13), (27, 13))
_CH_FLANK_TURRETS = ((2, 13), (3, 13), (24, 13), (25, 13), (1, 12), (26, 12))
_CH_FIRST_UPGRADES = ((2, 13), (25, 13))          # one-shot scouts at each corner
_CH_RING_TURRETS = ((12, 11), (15, 11), (11, 12), (16, 12))
# y=13 line fill, corners inward, skipping flankers (built above) and the gap
_CH_ROW_TURRETS = tuple(
    (x, 13) for x in (1, 26, 4, 23, 5, 22, 6, 21, 7, 20,
                      8, 19, 9, 18, 10, 17, 11, 16, 12, 15)
)
_CH_BACK_TURRETS = ((2, 12), (25, 12), (5, 12), (22, 12), (8, 12), (19, 12))
_CH_SUPPORTS = ((12, 2), (15, 2), (11, 3), (16, 3))
# late-game SP sink: upgrade every other line turret; odd-x turrets stay at
# base range (4.5) so demolishers (range 4.5) can never outrange the whole
# line once upgrades (range 3.5) land
_CH_LATE_UPGRADES = tuple((x, 13) for x in (4, 6, 8, 10, 12, 15, 17, 19, 21, 23))
# corner-zone bounds for the lane pick (enemy half, perspective space)
_CH_ZONE_Y = (14, 20)
_CH_ZONE_A_XMIN = 21      # their LEFT corner shows up top-right for us
_CH_ZONE_B_XMAX = 6       # their RIGHT corner, top-left for us


def _ch_weak_side_is_their_left(game, player: int, config: dict,
                                flip: bool) -> bool:
    """Score the two enemy corner zones by defensive strength (turret damage,
    upgrade-aware, plus wall presence) and aim at the weaker one. Tie -> their
    left, the hottest leak lane in the ranked sample. Stateless: re-reads the
    live board every turn, so it re-aims as their defense shifts."""
    info = config["unitInformation"]
    t_dmg = float(info[K_TURRET].get("attackDamageWalker", 0.0))
    t_dmg_up = float(info[K_TURRET].get("upgrade", {})
                     .get("attackDamageWalker", t_dmg))
    wall_hp = float(info[K_WALL].get("startHealth", 1.0)) or 1.0
    a = b = 0.0
    for s in game.structures():
        if s[1] == player:
            continue
        px, py = from_abs(s[2], s[3], flip)
        if not (_CH_ZONE_Y[0] <= py <= _CH_ZONE_Y[1]):
            continue
        kind = s[0]
        if kind == K_TURRET:
            val = t_dmg_up if s[5] else t_dmg
        elif kind == K_WALL:
            val = s[4] / wall_hp
        else:
            continue
        if px >= _CH_ZONE_A_XMIN:
            a += val
        elif px <= _CH_ZONE_B_XMAX:
            b += val
    return a <= b


def _ch_fill(cmds: List[Command], game, player: int, config: dict) -> None:
    costs = Costs(config)
    flip = player == 1
    occ = _occupied(game)

    # -- defense wishlist, priority order (engine truncates at the SP line) --
    _emit(cmds, K_WALL, _CH_CORNER_WALLS, flip, occ)
    _upgrades(cmds, game, player, flip, _CH_CORNER_WALLS)   # 120 HP for 1 SP
    _emit(cmds, K_TURRET, _CH_FLANK_TURRETS, flip, occ)
    _upgrades(cmds, game, player, flip, _CH_FIRST_UPGRADES)
    _emit(cmds, K_TURRET, _CH_RING_TURRETS, flip, occ)
    _emit(cmds, K_TURRET, _CH_ROW_TURRETS, flip, occ)
    _upgrades(cmds, game, player, flip, _CH_FLANK_TURRETS)
    _emit(cmds, K_TURRET, _CH_BACK_TURRETS, flip, occ)
    _emit(cmds, K_SUPPORT, _CH_SUPPORTS, flip, occ)
    _upgrades(cmds, game, player, flip, _CH_SUPPORTS)
    _upgrades(cmds, game, player, flip, _CH_LATE_UPGRADES)

    # -- offense: bank everything, commit the whole bank as one wave ---------
    res = config.get("resources", {})
    income = float(res.get("bitsPerRound", 5.0))
    interval = int(res.get("turnIntervalForBitSchedule", 10)) or 10
    income += float(res.get("bitGrowthRate", 1.0)) * (game.turn // interval)
    decay = float(res.get("bitDecayPerRound", 0.25)) or 0.25
    threshold = 0.875 * income / decay      # winners' observed commit point

    mp = float(game.stats(player)[2])
    scout_cost = costs.deploy_mp[0]
    if mp >= threshold and scout_cost > 0:
        n = int(mp // scout_cost)
        left = _ch_weak_side_is_their_left(game, player, config, flip)
        deep, split = ((13, 0), (11, 2)) if left else ((14, 0), (16, 2))
        n_split = n // 4 if n >= 12 else 0  # trailing second stack, winners' ratio
        _deploy(cmds, K_SCOUT, deep, flip, n - n_split)
        if n_split:
            _deploy(cmds, K_SCOUT, split, flip, n_split)


def corner_hammer(game, player: int, config: dict) -> List[Command]:
    """Winners' line + banked scout waves + the corner hardening and funnel
    the entire ranked field under-builds. Never raises: whatever was staged
    before a failure is still a legal wishlist."""
    cmds: List[Command] = []
    try:
        _ch_fill(cmds, game, player, config)
    except Exception:
        pass
    return cmds


# ---------------------------------------------------------------------------
# line_grinder — the mid-ladder counter-meta (2026-07-18 ranked-loss study)
# ---------------------------------------------------------------------------
#
# Distilled from the three opponents that beat our deployed net with the same
# plan (replays 23-0-3 / 23-0-9 / 23-1-43): a COMPLETELY solid turret line
# across y=13 (27-35 turrets, no walls, no gap — scout waves of any size
# scored zero breaches against it all game), attacking ONLY with banked
# demolisher waves (9-16 at a time, every ~6 turns) that outrange upgraded
# turrets (4.5 vs 3.5) and ground our corner cells open — their breaches
# landed ON (0,13)/(27,13) themselves. Our net played 0 demolishers in 9 of
# those 10 games: this archetype exists in the league so the net is forced to
# learn both sides of it.
#
# Gating matches the real ladder grinders (probe audit of the loss replays):
# they open their own CORNER — removals of (26,13),(27,13) / (0,13),(1,13)
# right before each wave — so the wave runs the edge diagonal and pops out
# directly beside the enemy's corner cell (their breaches landed ON our
# (0,13)/(27,13)). One removed corner also alternates demo waves with scout
# waves, and the ramp variant waved every 3 turns; period is 5 here with a
# strict demo/scout alternation. Everything derives from game.turn / live
# MP -> stateless. (Removal verified: cell empty by end of the marking turn,
# 75% refund; wave next turn; rebuilt over the following turns.)

_LG_GATE_R = ((26, 13), (27, 13))   # exit toward the enemy's LEFT corner
_LG_GATE_L = ((0, 13), (1, 13))     # exit toward the enemy's RIGHT corner
# corner blocks FIRST — but ONLY on line/interior cells: the edge-diagonal
# cells below the line ((1,12),(2,11)/(26,12),(25,11)) stay EMPTY so our own
# gated waves can walk the diagonal to the opened corner (the real grinders'
# boards have exactly this shape: depth behind the corner, never on the edge)
_LG_CORE = ((0, 13), (27, 13), (1, 13), (26, 13), (2, 13), (25, 13),
            (2, 12), (25, 12), (3, 12), (24, 12))
_LG_LINE = tuple(
    (x, 13) for x in (3, 24, 4, 23, 5, 22, 6, 21, 7, 20,
                      8, 19, 9, 18, 10, 17, 11, 16, 12, 15, 13, 14)
)
_LG_DEEP = ((3, 11), (24, 11), (4, 12), (23, 12))
# corner turrets upgraded first (16 dmg one-shots scouts at the hot cells);
# the rest of the line stays base range 4.5 so demolishers can't outrange it
_LG_UPGRADES = ((0, 13), (1, 13), (26, 13), (27, 13), (2, 12), (25, 12))
_LG_SUPPORTS = ((12, 2), (15, 2))
_LG_PERIOD = 5           # real cadence: waves every 3-6 turns (was 6)
_LG_MIN_DEMOS = 5        # observed first waves: 5-9 demolishers
_LG_MIN_SCOUTS = 10      # scout-wave floor on alternation cycles


def _lg_fill(cmds: List[Command], game, player: int, config: dict) -> None:
    costs = Costs(config)
    flip = player == 1
    occ = _occupied(game)
    own = _own_structs(game, player)
    phase = game.turn % _LG_PERIOD

    res = config.get("resources", {})
    income = float(res.get("bitsPerRound", 5.0))
    interval = int(res.get("turnIntervalForBitSchedule", 10)) or 10
    income += float(res.get("bitGrowthRate", 1.0)) * (game.turn // interval)
    decay = float(res.get("bitDecayPerRound", 0.25))
    demo_cost = costs.deploy_mp[1] or 1.0
    scout_cost = costs.deploy_mp[0] or 1.0
    mp = float(game.stats(player)[2])
    gate_r = tuple(to_abs(x, y, flip) for (x, y) in _LG_GATE_R)
    gate_l = tuple(to_abs(x, y, flip) for (x, y) in _LG_GATE_L)

    # -- the corner-gate cycle -----------------------------------------------
    # NOTE: no "only open when the enemy bank is low" guard — tried and
    # reverted. Against any banking opponent the gate then never opens and
    # the bot goes fully passive (ties/losses from pure inaction).
    if phase == _LG_PERIOD - 2:
        # open the weaker enemy corner's gate if next turn funds a wave
        if mp * (1.0 - decay) + income >= _LG_MIN_DEMOS * demo_cost:
            left_weak = _ch_weak_side_is_their_left(game, player, config, flip)
            for (ax, ay) in (gate_r if left_weak else gate_l):
                if (ax, ay) in own:
                    cmds.append((K_REMOVE, ax, ay))
    elif phase == _LG_PERIOD - 1:
        # spawn choice keys on which gate is ACTUALLY open (not a re-score,
        # which could flip sides between the remove turn and the wave turn):
        # our right corner exit attacks their left corner and vice versa
        r_open = any(a not in own for a in gate_r)
        l_open = any(a not in own for a in gate_l)
        if r_open or l_open:
            spot = (13, 0) if r_open else (14, 0)
            if (game.turn // _LG_PERIOD) % 2 == 0:     # demo cycle
                if mp >= _LG_MIN_DEMOS * demo_cost:
                    _deploy(cmds, K_DEMOLISHER, spot, flip,
                            int(mp // demo_cost))
            else:                                       # scout cycle
                if mp >= _LG_MIN_SCOUTS * scout_cost:
                    _deploy(cmds, K_SCOUT, spot, flip,
                            int(mp // scout_cost))

    # -- defense wishlist (engine truncates at the SP line) ------------------
    # while a gate is open (remove + wave turns), keep the gate cells OUT of
    # the rebuild list or the wave turn would re-block its own exit; they
    # rebuild over the following three turns of the cycle
    core = _LG_CORE
    if phase >= _LG_PERIOD - 2:
        gates = set(_LG_GATE_R) | set(_LG_GATE_L)
        core = tuple(c for c in _LG_CORE if c not in gates)
    _emit(cmds, K_TURRET, core, flip, occ)
    _emit(cmds, K_TURRET, _LG_LINE, flip, occ)
    _upgrades(cmds, game, player, flip, _LG_UPGRADES)
    _emit(cmds, K_TURRET, _LG_DEEP, flip, occ)
    _emit(cmds, K_SUPPORT, _LG_SUPPORTS, flip, occ)
    _upgrades(cmds, game, player, flip, _LG_SUPPORTS)


def line_grinder(game, player: int, config: dict) -> List[Command]:
    """Solid turret line + gated demolisher grind — the archetype behind 4 of
    our 5 newest ranked losses. Never raises: whatever was staged before a
    failure is still a legal wishlist."""
    cmds: List[Command] = []
    try:
        _lg_fill(cmds, game, player, config)
    except Exception:
        pass
    return cmds


SCRIPTED_BOTS: Dict[str, Callable] = {
    "rush": rush,
    "funnel": funnel,
    "demolisher_line": demolisher_line,
    "turtle": turtle,
    "torture": torture,
    "corner_hammer": corner_hammer,
    "line_grinder": line_grinder,
}
