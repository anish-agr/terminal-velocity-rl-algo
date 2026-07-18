"""Sparring bot: STATIC MAZE -- pure turtle behind a funnel wall.

Tests our OFFENSE: can the challenger break a fortified, wall-heavy defense
at all? (The other panel bots mostly test defense; a bot that only ever beats
aggressive opponents would look great against them and then stall out against
every turtle on the ladder.)

Design, tuned to this config: a full wall row across y=13 with one two-tile
funnel gap in the center, walls UPGRADED (40 -> 120 HP for 1 SP -- upgraded
walls are the cheapest armor in this ruleset), and base-range turrets ringing
the gap (kept un-upgraded: the upgrade trades range 4.5 -> 3.5, and funnel
coverage wants reach). Chips with a small demolisher wave through its own
funnel whenever affordable -- attrition, not knockout.

Deterministic: fixed layout, fixed lane, MP threshold only. Never adapts.
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

        # Wall row across y=13 with a funnel gap at x=13,14. Our own attackers
        # exit through the same gap, so the maze never blocks itself.
        self.walls = [[x, 13] for x in range(28) if x not in (13, 14)]

        # Turrets ring the gap; corners get their own guns.
        self.turrets = [
            [12, 12], [15, 12], [13, 11], [14, 11],
            [11, 11], [16, 11], [3, 12], [24, 12],
        ]

        # Upgrade priority: the walls nearest the gap take the most fire.
        self.upgrade_order = (
            [[12, 13], [15, 13], [11, 13], [16, 13], [10, 13], [17, 13]]
            + [[x, 13] for x in list(range(0, 10)) + list(range(18, 28)) if x not in (13, 14)]
        )

        self.lane = [14, 0]
        self.wave_size = 3

    def on_turn(self, turn_state):
        game_state = gamelib.GameState(self.config, turn_state)
        game_state.suppress_warnings(True)

        # Guns first (they do the killing), then the shell, then hardening.
        game_state.attempt_spawn(self.t["TURRET"], self.turrets)
        game_state.attempt_spawn(self.t["WALL"], self.walls)
        game_state.attempt_upgrade(self.upgrade_order)

        if game_state.get_resource(MP) >= self.demolisher_cost * self.wave_size:
            game_state.attempt_spawn(self.t["DEMOLISHER"], self.lane, 1000)

        game_state.submit_turn()


if __name__ == "__main__":
    AlgoStrategy().start()
