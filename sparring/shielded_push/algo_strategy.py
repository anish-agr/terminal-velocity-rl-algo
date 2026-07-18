"""Sparring bot: SHIELDED PUSH -- buffed-HP wave attacks.

Tests a failure mode neither the rush nor the demolisher line covers: attackers
whose effective HP has been raised past what your turret math expects. In this
config the Support UPGRADE is enormous -- shield range 2.5 -> 7, shield 2 -> 4,
plus 0.3 per tile of depth -- so a few upgraded supports blanket the whole exit
corridor. A 15 HP Scout leaving through two upgraded supports walks out with
~23+ HP, which more than doubles the shots a base turret needs.

Cycle: build & upgrade supports on the corridor, bank to a 12+ MP wave, send
every Scout down one fixed lane, repeat.

Deterministic: fixed layout, fixed lane, threshold on MP only.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from _shared import bootstrap_gamelib, unit_shorthands, mobile_cost, SP, MP  # noqa: E402

bootstrap_gamelib()
import gamelib  # noqa: E402


class AlgoStrategy(gamelib.AlgoCore):
    def on_game_start(self, config):
        self.config = config
        self.t = unit_shorthands(config)
        self.scout_cost = mobile_cost(config, 3)

        # Light defense -- enough to not fold instantly.
        self.turrets = [[3, 12], [24, 12], [13, 11], [14, 11]]

        # Supports stacked on the exit corridor. Build order matters: first
        # two, upgraded (range 7 covers the whole lane), then more as SP allows.
        self.supports = [[13, 2], [14, 2], [13, 3], [14, 3], [13, 4], [14, 4]]

        self.lane = [14, 0]
        self.wave_mp = 12.0

    def on_turn(self, turn_state):
        game_state = gamelib.GameState(self.config, turn_state)
        game_state.suppress_warnings(True)

        game_state.attempt_spawn(self.t["TURRET"], self.turrets)

        # Supports before walls, upgrades before expansion: an upgraded
        # support is worth more than two base ones here.
        game_state.attempt_spawn(self.t["SUPPORT"], self.supports[:2])
        game_state.attempt_upgrade(self.supports[:2])
        game_state.attempt_spawn(self.t["SUPPORT"], self.supports[2:])
        game_state.attempt_upgrade(self.supports[2:])

        if game_state.get_resource(MP) >= self.wave_mp:
            game_state.attempt_spawn(self.t["SCOUT"], self.lane, 1000)

        game_state.submit_turn()


if __name__ == "__main__":
    AlgoStrategy().start()
