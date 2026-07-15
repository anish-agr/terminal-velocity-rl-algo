"""Sparring bot: INTERCEPTOR WALL -- active anti-mobile defense.

Tests whether the challenger's attacks survive ACTIVE defense -- a different
failure mode than static structures. A defense of turrets is geometry; a
defense of Interceptors is bodies in the lanes. In this config Interceptors
are excellent defenders: 40 HP, 15 damage, range 4.5, and at speed 0.25 they
crawl, which keeps them on the board (and in the way) for a long time.

Every single turn this bot converts a fixed slice of MP into Interceptors
spread across both flanks and the center, keeping a standing screen alive.
Whatever MP accumulates beyond the screen goes out as an occasional scout
counter so the bot can actually win games it dominates.

Deterministic: fixed spawn rotation, fixed thresholds, no adaptation.
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
        self.interceptor_cost = mobile_cost(config, 5)
        self.scout_cost = mobile_cost(config, 3)

        # Light structures -- the interceptors ARE the defense.
        self.turrets = [[3, 12], [24, 12], [13, 11], [14, 11]]
        self.walls = [[0, 13], [27, 13]]

        # Screen spawns across both flanks + center. All verified edge tiles
        # (left edge y = 13 - x, right edge y = x - 14).
        self.screen_spawns = [[6, 7], [21, 7], [11, 2], [16, 2]]

        # Fixed slice of income converted to screen every turn.
        self.screen_count = 3
        # Counterattack only from real mass.
        self.counter_mp = 10.0
        self.counter_lane = [14, 0]

    def on_turn(self, turn_state):
        game_state = gamelib.GameState(self.config, turn_state)
        game_state.suppress_warnings(True)

        game_state.attempt_spawn(self.t["TURRET"], self.turrets)
        game_state.attempt_spawn(self.t["WALL"], self.walls)
        game_state.attempt_upgrade(self.turrets)

        # Standing screen: same rotation every turn, cost-capped.
        spawned = 0
        for loc in self.screen_spawns:
            if spawned >= self.screen_count:
                break
            if game_state.get_resource(MP) < self.interceptor_cost:
                break
            spawned += game_state.attempt_spawn(self.t["INTERCEPTOR"], loc)

        # Occasional counter once MP piles up past the screen budget.
        if game_state.get_resource(MP) >= self.counter_mp:
            game_state.attempt_spawn(self.t["SCOUT"], self.counter_lane, 1000)

        game_state.submit_turn()


if __name__ == "__main__":
    AlgoStrategy().start()
