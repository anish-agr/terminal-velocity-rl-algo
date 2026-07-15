"""Sparring bot: DEMOLISHER LINE -- steady structure-grinding pressure.

The defining threat of this ruleset: Demolishers cost 2 MP for 6 structure
damage (3.0 dmg/MP vs the Scout's 2.0) and match the base Turret's 4.5 range,
so massed demolishers trade into almost any defense. This bot plays the
archetype straight: a modest base-turret defense (kept UN-upgraded on purpose
-- the turret upgrade trades range 4.5 -> 3.5, and this bot wants reach), and
every time it can afford a 4-demolisher wave it dumps everything down one
fixed lane.

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
        self.demolisher_cost = mobile_cost(config, 4)

        # Modest defense: corner + center turrets, corner walls in front.
        self.turrets = [[3, 12], [24, 12], [13, 11], [14, 11], [7, 11], [20, 11]]
        self.walls = [[0, 13], [1, 13], [26, 13], [27, 13], [3, 13], [24, 13]]

        # One fixed lane. Static bots make no unforced errors.
        self.lane = [13, 0]
        # A real wave, not a trickle: ones and twos just feed the turrets.
        self.wave_size = 4

    def on_turn(self, turn_state):
        game_state = gamelib.GameState(self.config, turn_state)
        game_state.suppress_warnings(True)

        game_state.attempt_spawn(self.t["TURRET"], self.turrets)
        game_state.attempt_spawn(self.t["WALL"], self.walls)
        # Upgrade WALLS only (40 -> 120 HP for 1 SP is the best armor in this
        # config). Turrets stay base for the 4.5 range.
        game_state.attempt_upgrade(self.walls)

        if game_state.get_resource(MP) >= self.demolisher_cost * self.wave_size:
            game_state.attempt_spawn(self.t["DEMOLISHER"], self.lane, 1000)

        game_state.submit_turn()


if __name__ == "__main__":
    AlgoStrategy().start()
