"""Torture bot: deterministic per-turn script that exercises every engine mechanic the
starter replay misses, for the sim fidelity corpus (sim/MECHANICS.md).

Covered: trapped-spawn self-destructs (0/1 steps, no damage), >=5-step self-destructs into
enemy structures (wall-breaking -> mid-phase repath/double-back), removals of damaged +
upgraded structures (refund), mid-game upgrades (turret range drop, support y-bonus),
upgraded-support shields at different y, exact-3.0-MP banking (decay/display check),
mutual scout kills (attack ordering), stationary interceptors inside support range,
demolisher lines, corner attacks, edge-standing kills.
"""

import gamelib


class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()

    def on_game_start(self, config):
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]

    def on_turn(self, turn_state):
        gs = gamelib.GameState(self.config, turn_state)
        gs.suppress_warnings(True)
        t = gs.turn_number

        if t == 0:
            # trap both bottom spawn tiles -> 0/1-step SD, no damage
            gs.attempt_spawn(WALL, [[13, 1], [14, 1]])
            gs.attempt_spawn(TURRET, [[10, 3], [17, 3]])
            gs.attempt_spawn(SUPPORT, [[12, 3], [12, 9]])
            gs.attempt_spawn(SCOUT, [13, 0])
            gs.attempt_spawn(SCOUT, [14, 0])
            gs.attempt_spawn(DEMOLISHER, [20, 6])
            gs.attempt_spawn(INTERCEPTOR, [24, 10])
        elif t == 1:
            # upgrades: supports at two depths (y-bonus formula), turret (range DROP)
            gs.attempt_upgrade([[12, 3], [12, 9], [10, 3]])
            gs.attempt_spawn(WALL, [0, 13])
            gs.attempt_spawn(SCOUT, [7, 6])
            gs.attempt_spawn(SCOUT, [7, 6])
            gs.attempt_spawn(DEMOLISHER, [21, 7])
            gs.attempt_spawn(INTERCEPTOR, [6, 7])
        elif t == 2:
            # removals: fresh wall (0.8 refund) + turret (0.75); build a long wall line
            gs.attempt_remove([[13, 1], [17, 3]])
            line = [[x, 13] for x in range(1, 28)]
            gs.attempt_spawn(WALL, line)
            gs.attempt_spawn(DEMOLISHER, [20, 6])
            gs.attempt_spawn(DEMOLISHER, [20, 6])
            gs.attempt_spawn(SCOUT, [13, 0])
        elif t == 3:
            # bank exactly 3.0 MP (decay display distinguisher)
            gs.attempt_spawn(SCOUT, [7, 6])
            gs.attempt_spawn(SCOUT, [7, 6])
            gs.attempt_upgrade([[1, 13]])
            gs.attempt_remove([[14, 1]])
        elif t == 4:
            gs.attempt_spawn(TURRET, [13, 11])
            gs.attempt_upgrade([[13, 11]])
            gs.attempt_spawn(DEMOLISHER, [20, 6])
            gs.attempt_spawn(DEMOLISHER, [20, 6])
            gs.attempt_spawn(DEMOLISHER, [20, 6])
            gs.attempt_spawn(SCOUT, [7, 6])
        elif t == 5:
            # remove damaged/upgraded walls from the line (partial-health refunds)
            gs.attempt_remove([[x, 13] for x in range(1, 8)])
            gs.attempt_spawn(SCOUT, [7, 6], 1000)
        elif t == 6:
            # mutual scout waves meeting mid-board + interceptor parked in support range
            gs.attempt_spawn(SCOUT, [13, 0], 3)
            gs.attempt_spawn(INTERCEPTOR, [11, 2])
            gs.attempt_spawn(INTERCEPTOR, [16, 2])
        else:
            # sustained mixed pressure; alternate corners; rebuild/remove a wall each turn
            if gs.contains_stationary_unit([20, 13]):
                gs.attempt_remove([[20, 13]])
            else:
                gs.attempt_spawn(WALL, [20, 13])
            if t % 2 == 0:
                gs.attempt_spawn(DEMOLISHER, [20, 6], 2)
                gs.attempt_spawn(SCOUT, [7, 6], 1000)
            else:
                gs.attempt_spawn(SCOUT, [14, 0], 1000)
            if t % 5 == 0:
                gs.attempt_spawn(SUPPORT, [11, 5])
                gs.attempt_upgrade([[11, 5]])

        gs.submit_turn()


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
