"""State featurization (ARCHITECTURE §2): 18 board planes + 14 scalars.

Planes 0-11 come from the sim bridge (`Game.board_planes`, already
player-perspective). Planes 12-17 are deploy-history planes maintained here,
because the sim has no memory of previous turns' commands:

    12-14  enemy deploys LAST turn (scout/demolisher/interceptor) count/10, clamp 1
    15-17  per-kind EMA over the match: ema = decay*ema + count/10, clamp 1

`DeployHistory` records in ABSOLUTE board coordinates; `build_planes` applies
the player-perspective flip at read time, so one history object serves both
self-play seats and the deployment driver identically.

Array convention throughout: [C, x, y] — matching the bridge layout
planes[p*784 + x*28 + y]. The x-mirror for augmentation is therefore
np.flip(axis=1).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

BRIDGE_PLANES = 12
HISTORY_PLANES = 6
N_PLANES = BRIDGE_PLANES + HISTORY_PLANES  # 18
GRID = 28
N_SCALARS = 14

# sim deploy command kinds -> history plane slot
_KIND_SLOT = {3: 0, 4: 1, 5: 2}

_COUNT_NORM = 10.0
_EMA_DECAY = 0.7


class DeployHistory:
    """Per-match memory of the ENEMY's mobile deploys + last turn's damage flows.

    One instance per (game, perspective-player). The actor records after every
    played turn; the deployment driver records from observed spawn events.
    All coordinates recorded are ABSOLUTE.
    """

    def __init__(self, config: dict):
        res = config["resources"]
        self.mp_per_round = float(res["bitsPerRound"])
        self.mp_growth = float(res["bitGrowthRate"])
        self.mp_interval = int(res["turnIntervalForBitSchedule"])
        self.mp_decay = float(res["bitDecayPerRound"])
        self.reset()

    def reset(self) -> None:
        self.last = np.zeros((3, GRID, GRID), dtype=np.float32)
        self.ema = np.zeros((3, GRID, GRID), dtype=np.float32)
        # last-turn flows, own perspective: dealt/taken breach, dealt/taken struct dmg
        self.breach_dealt = 0.0
        self.breach_taken = 0.0
        self.struct_dmg_dealt = 0.0
        self.struct_dmg_taken = 0.0

    def record_turn(
        self,
        enemy_deploys: Sequence[Tuple[int, int, int]],
        breach_dealt: float,
        breach_taken: float,
        struct_dmg_dealt: float,
        struct_dmg_taken: float,
    ) -> None:
        """Call once per completed turn.

        enemy_deploys: (kind, x_abs, y_abs) mobile commands the enemy executed
        this turn (kinds 3..5; other kinds are ignored defensively).
        """
        self.ema *= _EMA_DECAY
        self.last[:] = 0.0
        for kind, x, y in enemy_deploys:
            slot = _KIND_SLOT.get(kind)
            if slot is None or not (0 <= x < GRID and 0 <= y < GRID):
                continue
            self.last[slot, x, y] += 1.0
        self.ema += self.last / _COUNT_NORM
        np.clip(self.ema, 0.0, 1.0, out=self.ema)

        self.breach_dealt = float(breach_dealt)
        self.breach_taken = float(breach_taken)
        self.struct_dmg_dealt = float(struct_dmg_dealt)
        self.struct_dmg_taken = float(struct_dmg_taken)

    def income(self, turn: int) -> float:
        return self.mp_per_round + (turn // self.mp_interval) * self.mp_growth

    def banked_mp(self, mp_now: float, turn: int) -> float:
        """MP next turn if we deploy nothing: decay then next turn's income."""
        return mp_now * (1.0 - self.mp_decay) + self.income(turn + 1)


def build_planes(
    game, player: int, history: DeployHistory
) -> Tuple[np.ndarray, np.ndarray]:
    """(board [18,28,28] f32, scalars [14] f32) from `player`'s perspective.

    `game` is a terminal_sim.Game (or any object with the same board_planes /
    stats / turn surface — tests use a fake).
    """
    raw = np.frombuffer(game.board_planes(player), dtype="<f4")
    board = np.empty((N_PLANES, GRID, GRID), dtype=np.float32)
    board[:BRIDGE_PLANES] = raw.reshape(BRIDGE_PLANES, GRID, GRID)

    hist_last = history.last / _COUNT_NORM
    np.clip(hist_last, 0.0, 1.0, out=hist_last)
    hist = np.concatenate([hist_last, history.ema], axis=0)
    if player == 1:
        # absolute -> perspective for the top player: rotate 180° (x and y flip)
        hist = hist[:, ::-1, ::-1]
    board[BRIDGE_PLANES:] = hist

    own_hp, own_sp, own_mp = game.stats(player)
    en_hp, en_sp, en_mp = game.stats(1 - player)
    turn = float(game.turn)

    scalars = np.array(
        [
            own_hp / 30.0,
            own_sp / 40.0,
            own_mp / 15.0,
            en_hp / 30.0,
            en_sp / 40.0,
            en_mp / 15.0,
            turn / 100.0,
            history.income(int(turn)) / 10.0,
            history.banked_mp(own_mp, int(turn)) / 15.0,
            history.banked_mp(en_mp, int(turn)) / 15.0,
            history.breach_dealt / 5.0,
            history.breach_taken / 5.0,
            history.struct_dmg_dealt / 50.0,
            history.struct_dmg_taken / 50.0,
        ],
        dtype=np.float32,
    )
    return board, scalars


def mirror_board(board: np.ndarray) -> np.ndarray:
    """x-mirror augmentation (§2.3); scalars are mirror-invariant."""
    return board[:, ::-1, :].copy()
