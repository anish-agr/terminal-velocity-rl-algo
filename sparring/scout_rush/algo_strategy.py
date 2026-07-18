"""Sparring bot: SCOUT RUSH -- early mass-scout aggression.

The classic turtle-killer. Minimal defense; from the very first turn, any
time it can field 5+ Scouts it sends the whole wave down whichever of two
fixed lanes currently takes less turret fire. Turrets hit one target at a
time, so a big-enough wave saturates the defense and survivors walk through.
With starting health at 30 in this config, an unanswered early rush closes
games fast -- this bot exists to catch defenses that set up too slowly.

Deterministic: fixed layout, two fixed lane options with deterministic
tie-breaking, threshold on MP only.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from _shared import (  # noqa: E402
    bootstrap_gamelib,
    unit_shorthands,
    mobile_cost,
    least_damage_lane,
    SP,
    MP,
)

bootstrap_gamelib()
import gamelib  # noqa: E402


class AlgoStrategy(gamelib.AlgoCore):
    def on_game_start(self, config):
        self.config = config
        self.t = unit_shorthands(config)
        self.scout_cost = mobile_cost(config, 3)
        self.turret_damage = config["unitInformation"][2].get("attackDamageWalker", 0.0)

        # Deliberately thin -- this bot buys attacks, not safety.
        self.turrets = [[3, 12], [24, 12], [13, 11], [14, 11]]

        self.lanes = [[13, 0], [14, 0]]
        self.wave_size = 5

    def on_turn(self, turn_state):
        game_state = gamelib.GameState(self.config, turn_state)
        game_state.suppress_warnings(True)

        game_state.attempt_spawn(self.t["TURRET"], self.turrets)

        if game_state.get_resource(MP) >= self.scout_cost * self.wave_size:
            lane = least_damage_lane(game_state, self.lanes, self.turret_damage)
            if lane is not None:
                game_state.attempt_spawn(self.t["SCOUT"], lane, 1000)

        game_state.submit_turn()


if __name__ == "__main__":
    AlgoStrategy().start()
