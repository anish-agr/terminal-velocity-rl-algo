"""Sparring bot: TURRET WALL FLOOD -- the ladder archetype that beat us.

Mimics the opponents from ladder 15344900/15344930/15344940 (all losses):
a dense line of base turrets across row 14 built immediately, corner walls,
steady turret reinforcement, and periodic mass-scout floods once the bank
fills. Scout waves into that front line die dealing ~0 unless the attacker
leads with demolishers or finds an unwatched corner.

Deterministic: fixed build order, fixed flood threshold, lane choice is a
pure function of the board (least_damage_lane).
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

        # dense base-turret front line, center-out (mirror of 15344930's t0:
        # turrets at 26,25,24,22,21,20,18,16,14,13... along its front row)
        self.front = [[x, 13] for x in
                      (13, 14, 16, 18, 20, 21, 22, 24, 25, 26,
                       11, 9, 7, 6, 5, 3, 2, 1)]
        self.corner_walls = [[0, 13], [27, 13], [1, 12], [26, 12]]
        self.back_turrets = [[13, 12], [14, 12], [11, 12], [16, 12],
                             [9, 12], [18, 12]]

        self.lanes = [[13, 0], [14, 0], [7, 6], [20, 6]]
        self.flood_mp = 12.0     # bank then flood, like the ladder games

    def on_turn(self, turn_state):
        game_state = gamelib.GameState(self.config, turn_state)
        game_state.suppress_warnings(True)

        game_state.attempt_spawn(self.t["TURRET"], self.front)
        game_state.attempt_spawn(self.t["WALL"], self.corner_walls)
        game_state.attempt_spawn(self.t["TURRET"], self.back_turrets)
        game_state.attempt_upgrade(self.front[:6])

        if game_state.get_resource(MP) >= self.flood_mp:
            lane = least_damage_lane(game_state, self.lanes, self.turret_damage)
            if lane is not None:
                game_state.attempt_spawn(self.t["SCOUT"], lane, 1000)

        game_state.submit_turn()


if __name__ == "__main__":
    AlgoStrategy().start()
