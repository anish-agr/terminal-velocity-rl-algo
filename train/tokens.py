"""Action space: a turn as a token sequence (ARCHITECTURE §3).

A plan is a sequence of <= T_MAX tokens, each (type, loc, count_bucket):

    type  in 0..8   BUILD_WALL, BUILD_SUPPORT, BUILD_TURRET, UPGRADE, REMOVE,
                    DEP_SCOUT, DEP_DEMOLISHER, DEP_INTERCEPTOR, END
    loc   in 0..783 flattened 28x28 cell, PLAYER-PERSPECTIVE coordinates
    count in 0..7   index into COUNT_BUCKETS, meaningful for DEP_* only

Coordinate convention: every token, mask, and scratch in this module lives in
player-perspective space — the acting player's half is ALWAYS y < 14. For the
absolute-top player (owner index 1 in the sim) `flip=True` maps perspective
(x, y) -> absolute (27-x, 27-y) on the way to engine commands and back. This is
what makes x-mirror augmentation a pure remap (§2.3) and lets one network play
both seats.

Engine-order semantics (§3.1): the sim bridge splits a command list into builds
(applied first, in token order) and mobile deploys (applied after ALL builds).
The PlanScratch therefore marks a build's tile as occupied immediately, and
masks builds off tiles the plan has already deployed mobiles onto — every plan
this module accepts is engine-consistent by construction.

Affordability margin (§3.2 / MECHANICS Open-fix 3): the engine's MP chain can
drift from any reconstruction by < 0.1, so legality thresholds use
(resource - MARGIN). The ALL bucket expands optimistically against raw scratch
MP — the engine silently skips commands it cannot afford, so attempting the
marginal unit is free; the search (§5) is what prices in that it may not spawn.

No dependency on the compiled sim: geometry is computed here, and the
provably-null deploy mask (§3.3) takes a `pathfind(x_abs, y_abs) -> [(x, y)]`
callable (e.g. Game.pathfind) that may be None to disable.
"""

from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

BUILD_WALL, BUILD_SUPPORT, BUILD_TURRET = 0, 1, 2
UPGRADE, REMOVE = 3, 4
DEP_SCOUT, DEP_DEMOLISHER, DEP_INTERCEPTOR = 5, 6, 7
END = 8
N_TYPES = 9

BUILD_TYPES = (BUILD_WALL, BUILD_SUPPORT, BUILD_TURRET)
DEPLOY_TYPES = (DEP_SCOUT, DEP_DEMOLISHER, DEP_INTERCEPTOR)
LOC_TYPES = BUILD_TYPES + (UPGRADE, REMOVE) + DEPLOY_TYPES  # all but END take a loc

T_MAX = 24
GRID = 28
N_LOCS = GRID * GRID

# count buckets (§3.1); -1 == ALL (spend-remaining at this tile)
COUNT_BUCKETS: Tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21, -1)
ALL_BUCKET = len(COUNT_BUCKETS) - 1
N_BUCKETS = len(COUNT_BUCKETS)
_BUCKETS_DESC = (21, 13, 8, 5, 3, 2, 1)  # greedy decomposition for decode

MARGIN = 0.1

# sim bridge command kinds (sim/src/py.rs): 0..2 build, 3..5 deploy, 6 remove, 7 upgrade
_TYPE_TO_KIND = {
    BUILD_WALL: 0, BUILD_SUPPORT: 1, BUILD_TURRET: 2,
    UPGRADE: 7, REMOVE: 6,
    DEP_SCOUT: 3, DEP_DEMOLISHER: 4, DEP_INTERCEPTOR: 5,
}
_KIND_TO_TYPE = {v: k for k, v in _TYPE_TO_KIND.items()}
_DEPLOY_KIND_UNIT = {3: 0, 4: 1, 5: 2}  # kind -> index into mobile cost table


class Token(NamedTuple):
    type: int
    loc: int    # flattened perspective cell x*28+y; 0 for END
    count: int  # bucket index; 0 for non-deploy tokens


END_TOKEN = Token(END, 0, 0)


def loc_xy(loc: int) -> Tuple[int, int]:
    return loc // GRID, loc % GRID


def xy_loc(x: int, y: int) -> int:
    return x * GRID + y


# ---------------------------------------------------------------------------
# Geometry (perspective space; own half is y < 14)
# ---------------------------------------------------------------------------

def in_arena(x: int, y: int) -> bool:
    """The 28x28 diamond: row y (bottom half) spans x in [13-y, 14+y], mirrored on top."""
    if not (0 <= x < GRID and 0 <= y < GRID):
        return False
    if y < 14:
        return 13 - y <= x <= 14 + y
    return y - 14 <= x <= 41 - y


def _own_half_mask() -> np.ndarray:
    m = np.zeros(N_LOCS, dtype=bool)
    for x in range(GRID):
        for y in range(14):
            if in_arena(x, y):
                m[xy_loc(x, y)] = True
    return m


def _edge_mask() -> np.ndarray:
    """Own deploy edges: bottom-left y = 13-x (x 0..13), bottom-right y = x-14 (x 14..27)."""
    m = np.zeros(N_LOCS, dtype=bool)
    for x in range(14):
        m[xy_loc(x, 13 - x)] = True
    for x in range(14, GRID):
        m[xy_loc(x, x - 14)] = True
    return m


OWN_HALF = _own_half_mask()          # 210 cells
OWN_EDGES = _edge_mask()             # 28 cells

_MIRROR_LOC = np.array(
    [xy_loc(GRID - 1 - x, y) for x in range(GRID) for y in range(GRID)],
    dtype=np.int32,
)


def to_abs(x: int, y: int, flip: bool) -> Tuple[int, int]:
    return (GRID - 1 - x, GRID - 1 - y) if flip else (x, y)


def from_abs(x: int, y: int, flip: bool) -> Tuple[int, int]:
    return to_abs(x, y, flip)  # the flip is an involution


# ---------------------------------------------------------------------------
# Costs (read from game config at runtime — NEVER hardcoded)
# ---------------------------------------------------------------------------

class Costs:
    """Extracted once from game-configs.json (the dict the engine hands us)."""

    def __init__(self, config: dict):
        info = config["unitInformation"]
        # structures: index 0 wall, 1 support, 2 turret
        self.build_sp = [float(info[i].get("cost1", 0.0)) for i in range(3)]
        # engine rule (MECHANICS): upgrade cost = upgrade-block cost1 if present,
        # else the base cost (wall upgrade therefore costs 1).
        self.upgrade_sp = [
            float(info[i].get("upgrade", {}).get("cost1", info[i].get("cost1", 0.0)))
            for i in range(3)
        ]
        # mobiles: index 0 scout, 1 demolisher, 2 interceptor (unit info 3,4,5)
        self.deploy_mp = [float(info[i + 3].get("cost2", 0.0)) for i in range(3)]

    def deploy_cost(self, token_type: int) -> float:
        return self.deploy_mp[token_type - DEP_SCOUT]

    def build_cost(self, token_type: int) -> float:
        return self.build_sp[token_type]


# ---------------------------------------------------------------------------
# Plan scratch: incremental legality + resource accounting (§3.2, §3.4)
# ---------------------------------------------------------------------------

class PlanScratch:
    """Mutable within-turn state for masking and encoding one plan.

    Built from the sim's `structures()` + raw `stats()` (absolute coords), or
    given directly in perspective coords for tests. All queries/updates are in
    perspective space.
    """

    def __init__(
        self,
        costs: Costs,
        sp: float,
        mp: float,
        structures: Sequence[Tuple[int, int, int, int, float, bool, bool]] = (),
        flip: bool = False,
        pathfind: Optional[Callable[[int, int], list]] = None,
        own_player: int = 0,
        null_deploy_mask: bool = True,
        null_deploy_min_steps: int = 5,
    ):
        self.costs = costs
        self.sp = float(sp)
        self.mp = float(mp)
        self.flip = flip
        self._pathfind = pathfind
        self._null_mask_on = null_deploy_mask
        self._null_min_steps = null_deploy_min_steps

        # occupancy over ALL cells (any structure blocks builds and deploys there)
        self.occupied = np.zeros(N_LOCS, dtype=bool)
        # own structures: loc -> [kind, upgraded, pending_removal]
        self.own: Dict[int, list] = {}
        # tiles this plan has deployed mobiles onto (builds masked off them)
        self.deployed_tiles: set = set()

        for (kind, owner, ax, ay, _hp, upgraded, pending) in structures:
            px, py = from_abs(ax, ay, flip)
            loc = xy_loc(px, py)
            self.occupied[loc] = True
            if owner == own_player:
                self.own[loc] = [int(kind), bool(upgraded), bool(pending)]

        self._null_deploy_cache: Optional[np.ndarray] = None

    # -- resources, with the §3.2 margin ------------------------------------

    @property
    def sp_avail(self) -> float:
        return self.sp - MARGIN

    @property
    def mp_avail(self) -> float:
        return self.mp - MARGIN

    # -- provably-null deploy tiles (§3.3) -----------------------------------

    def _null_deploy(self) -> np.ndarray:
        """True where deploying is provably zero-effect: the pocket cannot reach
        the enemy half AND the walk is < 5 steps (self-destruct with no damage)."""
        if self._null_deploy_cache is not None:
            return self._null_deploy_cache
        null = np.zeros(N_LOCS, dtype=bool)
        if self._null_mask_on and self._pathfind is not None:
            for loc in np.nonzero(OWN_EDGES & ~self.occupied)[0]:
                px, py = loc_xy(int(loc))
                ax, ay = to_abs(px, py, self.flip)
                path = self._pathfind(ax, ay)
                if not path:
                    continue  # blocked spawn handled by occupancy; be permissive
                # perspective y of the walk's last tile
                _, ly = from_abs(int(path[-1][0]), int(path[-1][1]), self.flip)
                if ly < 14 and len(path) < self._null_min_steps:
                    null[loc] = True
        self._null_deploy_cache = null
        return null

    # -- masks ----------------------------------------------------------------

    def type_mask(self) -> np.ndarray:
        """[N_TYPES] bool: token types with at least one legal (loc, count)."""
        m = np.zeros(N_TYPES, dtype=bool)
        m[END] = True
        for t in BUILD_TYPES:
            if self.sp_avail >= self.costs.build_cost(t) and self.loc_mask(t).any():
                m[t] = True
        if any(
            not up and not pend and self.sp_avail >= self.costs.upgrade_sp[kind]
            for kind, up, pend in self.own.values()
        ):
            m[UPGRADE] = True
        if any(not pend for _k, _u, pend in self.own.values()):
            m[REMOVE] = True
        for t in DEPLOY_TYPES:
            if self.mp_avail >= self.costs.deploy_cost(t) and self.loc_mask(t).any():
                m[t] = True
        return m

    def loc_mask(self, token_type: int) -> np.ndarray:
        """[N_LOCS] bool of legal cells for this token type."""
        m = np.zeros(N_LOCS, dtype=bool)
        if token_type in BUILD_TYPES:
            if self.sp_avail >= self.costs.build_cost(token_type):
                m = OWN_HALF & ~self.occupied
                if self.deployed_tiles:
                    m = m.copy()
                    m[list(self.deployed_tiles)] = False
        elif token_type == UPGRADE:
            for loc, (kind, up, pend) in self.own.items():
                if not up and not pend and self.sp_avail >= self.costs.upgrade_sp[kind]:
                    m[loc] = True
        elif token_type == REMOVE:
            for loc, (_kind, _up, pend) in self.own.items():
                if not pend:  # marking twice is legal-but-null -> masked (§3.2)
                    m[loc] = True
        elif token_type in DEPLOY_TYPES:
            if self.mp_avail >= self.costs.deploy_cost(token_type):
                m = OWN_EDGES & ~self.occupied & ~self._null_deploy()
        return m

    def count_mask(self, token_type: int) -> np.ndarray:
        """[N_BUCKETS] bool. Non-deploy tokens fix bucket 0 by convention."""
        m = np.zeros(N_BUCKETS, dtype=bool)
        if token_type in DEPLOY_TYPES:
            cost = self.costs.deploy_cost(token_type)
            for i, b in enumerate(COUNT_BUCKETS):
                units = 1 if b == -1 else b  # ALL is legal if one unit is
                if self.mp_avail >= cost * units:
                    m[i] = True
        else:
            m[0] = True
        return m

    # -- application ----------------------------------------------------------

    def apply(self, tok: Token) -> int:
        """Update the scratch as if `tok` executes. Returns units deployed
        (deploy tokens), else 0. Raises ValueError on an illegal token — the
        decoder must never produce one, so this is an invariant check."""
        t = tok.type
        if t == END:
            return 0
        if not self.loc_mask(t)[tok.loc]:
            raise ValueError("illegal token loc: {}".format(tok))
        if t in BUILD_TYPES:
            self.sp -= self.costs.build_cost(t)
            self.occupied[tok.loc] = True
            self.own[tok.loc] = [t, False, False]
            self._null_deploy_cache = None  # layout changed
            return 0
        if t == UPGRADE:
            kind = self.own[tok.loc][0]
            self.sp -= self.costs.upgrade_sp[kind]
            self.own[tok.loc][1] = True
            return 0
        if t == REMOVE:
            self.own[tok.loc][2] = True
            return 0
        # deploy
        if not self.count_mask(t)[tok.count]:
            raise ValueError("illegal token count: {}".format(tok))
        cost = self.costs.deploy_cost(t)
        b = COUNT_BUCKETS[tok.count]
        units = int(self.mp // cost) if b == -1 else b
        units = max(units, 0)
        self.mp -= cost * units
        self.deployed_tiles.add(tok.loc)
        return units


class ScratchSpec:
    """Picklable PlanScratch factory.

    search.choose() hands these to NetClients: LocalNetClient simply CALLS it
    (in-process), while QueueClient ships `.blob` over the wire and the
    inference server reconstructs the factory there. The pathfind callable is
    process-local and never crosses the wire, so server-side scratches run
    without null-deploy masking (§3.3 is an optimization, legal to skip).
    """

    def __init__(self, costs: Costs, structures, sp: float, mp: float,
                 flip: bool, player: int, pathfind=None):
        self.costs = costs
        self.blob = (tuple(structures), float(sp), float(mp), bool(flip), int(player))
        self._pathfind = pathfind

    def __call__(self) -> "PlanScratch":
        structures, sp, mp, flip, player = self.blob
        return PlanScratch(self.costs, sp, mp, structures, flip=flip,
                           pathfind=self._pathfind, own_player=player)


# ---------------------------------------------------------------------------
# Plan <-> engine commands
# ---------------------------------------------------------------------------

def encode_plan(
    plan: Sequence[Token], scratch: PlanScratch
) -> List[Tuple[int, int, int]]:
    """Expand a token plan into sim command tuples (kind, x_abs, y_abs).

    Consumes the scratch (call on a fresh one). Deploy tokens repeat their
    command `units` times — the bridge preserves order within builds and within
    deploys, and the engine applies all builds before all deploys.
    """
    cmds: List[Tuple[int, int, int]] = []
    for tok in plan:
        if tok.type == END:
            break
        units = scratch.apply(tok)
        px, py = loc_xy(tok.loc)
        ax, ay = to_abs(px, py, scratch.flip)
        kind = _TYPE_TO_KIND[tok.type]
        if tok.type in DEPLOY_TYPES:
            cmds.extend([(kind, ax, ay)] * units)
        else:
            cmds.append((kind, ax, ay))
    return cmds


def decode_commands(
    cmds: Sequence[Tuple[int, int, int]], flip: bool
) -> List[Token]:
    """Inverse-ish of encode_plan for replay ingestion (§7): engine command
    tuples -> canonical token plan. Deploy runs at the same (kind, tile) are
    re-bucketed greedily largest-first (7 -> 5+2); ALL is never produced.
    Truncated to T_MAX-1 tokens + END (BC is a prior, not a target)."""
    toks: List[Token] = []
    i = 0
    n = len(cmds)
    while i < n:
        kind, ax, ay = cmds[i]
        px, py = from_abs(ax, ay, flip)
        loc = xy_loc(px, py)
        if kind in _DEPLOY_KIND_UNIT:
            j = i
            while j < n and cmds[j] == cmds[i]:
                j += 1
            run = j - i
            for b in _BUCKETS_DESC:
                while run >= b:
                    toks.append(Token(_KIND_TO_TYPE[kind], loc, COUNT_BUCKETS.index(b)))
                    run -= b
            i = j
        else:
            toks.append(Token(_KIND_TO_TYPE[kind], loc, 0))
            i += 1
    toks = toks[: T_MAX - 1]
    toks.append(END_TOKEN)
    return toks


# ---------------------------------------------------------------------------
# Mirror augmentation (§2.3)
# ---------------------------------------------------------------------------

def mirror_token(tok: Token) -> Token:
    if tok.type == END:
        return tok
    return Token(tok.type, int(_MIRROR_LOC[tok.loc]), tok.count)


def mirror_plan(plan: Sequence[Token]) -> List[Token]:
    return [mirror_token(t) for t in plan]
