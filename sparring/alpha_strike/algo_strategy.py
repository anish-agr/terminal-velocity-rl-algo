"""Sparring bot: ALPHA STRIKE -- burst damage on a cycle.

Tests SPIKE survival, which is a different question from steady pressure.
A defense tuned against constant chip damage (the demolisher-line profile)
can still collapse when everything arrives at once: turrets kill one unit at
a time, so twelve bodies in one wave get through where four-a-turn die. With
30 starting health in this config, one unhandled spike can decide a game.

Rhythm: build defense, bank MP to ~15 (near the decay ceiling -- income 5,
decay 25%, so banking runs 5 -> 8.75 -> 11.6 -> 13.7 -> 15.3), then dump the
entire bank as one mixed wave: a demolisher spearhead to open structures,
scouts flooding through behind. Roughly every 5th turn, all game.

Deterministic: fixed lane, fixed mix, MP threshold only.
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

        self.turrets = [[3, 12], [24, 12], [13, 11], [14, 11], [7, 11], [20, 11]]
        self.walls = [[0, 13], [1, 13], [26, 13], [27, 13]]

        self.lane = [13, 0]
        self.strike_mp = 15.0
        self.demolishers_per_wave = 4

    def on_turn(self, turn_state):
        game_state = gamelib.GameState(self.config, turn_state)
        game_state.suppress_warnings(True)

        game_state.attempt_spawn(self.t["TURRET"], self.turrets)
        game_state.attempt_spawn(self.t["WALL"], self.walls)
        game_state.attempt_upgrade(self.walls)

        # Bank until the spike threshold, then commit EVERYTHING at once.
        if game_state.get_resource(MP) >= self.strike_mp:
            game_state.attempt_spawn(
                self.t["DEMOLISHER"], self.lane, self.demolishers_per_wave
            )
            game_state.attempt_spawn(self.t["SCOUT"], self.lane, 1000)

        game_state.submit_turn()


if __name__ == "__main__":
    AlgoStrategy().start()
