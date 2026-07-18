"""Sparring bot: CORNER GUN -- edge-biased attacker.

Tests whether the challenger's defense is SYMMETRIC. Asymmetric defenses are
one of the most common hidden weaknesses: a bot reinforces wherever it happens
to get hit early, ends up strong on one side, and never notices the other side
leaks. This bot punishes that by committing every wave to the same flank, all
game, every game.

All waves spawn from the extreme right-edge tile, so the attack path always
hugs the right flank into the opponent's left corner region. Wave is a mixed
demolisher-front + scout-follow: demolishers (speed 0.5) open structures,
scouts (speed 1) run the hole. To test the OTHER flank, run the mirror of
this bot -- don't make this one adaptive.

Deterministic: one lane, fixed mix, MP threshold only.
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
        self.scout_cost = mobile_cost(config, 3)

        self.turrets = [[3, 12], [24, 12], [13, 11], [14, 11]]
        self.walls = [[0, 13], [1, 13], [26, 13], [27, 13]]

        # Extreme right-edge spawn (y = x - 14 -> [24, 10] is on the edge).
        # Spawning this deep right keeps the whole path hugging the flank.
        self.lane = [24, 10]

        self.wave_mp = 9.0
        self.demolishers_per_wave = 3

    def on_turn(self, turn_state):
        game_state = gamelib.GameState(self.config, turn_state)
        game_state.suppress_warnings(True)

        game_state.attempt_spawn(self.t["TURRET"], self.turrets)
        game_state.attempt_spawn(self.t["WALL"], self.walls)
        game_state.attempt_upgrade(self.walls)

        if game_state.get_resource(MP) >= self.wave_mp:
            # Demolishers open, scouts pour through with whatever MP remains.
            game_state.attempt_spawn(
                self.t["DEMOLISHER"], self.lane, self.demolishers_per_wave
            )
            game_state.attempt_spawn(self.t["SCOUT"], self.lane, 1000)

        game_state.submit_turn()


if __name__ == "__main__":
    AlgoStrategy().start()
