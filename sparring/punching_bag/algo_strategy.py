"""Sparring bot: PUNCHING BAG -- defends well, never attacks.

Isolates exactly one question: can the challenger reliably CLOSE OUT a
passive opponent? A bot that cannot convert against a target that never
punches back has a broken offense -- and that brokenness hides in normal
matches, where wins can come from the opponent's mistakes rather than our
own finishing. Games against this bot that reach round 100 are the tell.

The guard is deliberately solid (the test is meaningless against a pushover):
a spread two-ring turret defense with upgraded walls in front and turret
upgrades layered behind the front line, spending every Structure point --
but it never deploys a single mobile unit. Its MP just decays, by design.

Deterministic: fixed layout, no attacks, no adaptation, no randomness.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from _shared import bootstrap_gamelib, unit_shorthands, SP, MP  # noqa: E402

bootstrap_gamelib()
import gamelib  # noqa: E402


class AlgoStrategy(gamelib.AlgoCore):
    def on_game_start(self, config):
        self.config = config
        self.t = unit_shorthands(config)

        # Two rings: front-line turrets at base range (4.5 reach), a deeper
        # ring that gets the damage upgrade (16 dmg at 3.5 range is fine one
        # row back), walls shielding the front.
        self.front_turrets = [[3, 12], [24, 12], [8, 12], [19, 12], [13, 11], [14, 11]]
        self.deep_turrets = [[5, 10], [22, 10], [11, 9], [16, 9], [13, 9], [14, 9]]
        self.walls = [
            [0, 13], [1, 13], [2, 13], [3, 13],
            [24, 13], [25, 13], [26, 13], [27, 13],
            [8, 13], [13, 12], [14, 12], [19, 13],
        ]

    def on_turn(self, turn_state):
        game_state = gamelib.GameState(self.config, turn_state)
        game_state.suppress_warnings(True)

        game_state.attempt_spawn(self.t["TURRET"], self.front_turrets)
        game_state.attempt_spawn(self.t["WALL"], self.walls)
        game_state.attempt_spawn(self.t["TURRET"], self.deep_turrets)
        game_state.attempt_upgrade(self.walls)
        game_state.attempt_upgrade(self.deep_turrets)

        # Never attacks. That is the entire point.
        game_state.submit_turn()


if __name__ == "__main__":
    AlgoStrategy().start()
