"""Zero-dependency scripted funnel bot (ARCHITECTURE §9.2) — the last rung.

Pure python + gamelib. Runs the whole match if the .so or weights are missing
or broken, and supplies the guaranteed per-turn submission the watchdog falls
back to when the search misses its deadline. Deterministic, config-driven,
no imports beyond gamelib (which the driver passes in — this module imports
NOTHING at module level so it can never be the thing that crashes).

Layout: wall row at y=13 with a two-tile center gap, turrets ringing the gap,
corner guns, gap-shoulder wall upgrades first, then a demolisher wave through
the funnel whenever a 3-wave is affordable.
"""


class FallbackBot:
    def __init__(self, config):
        info = config["unitInformation"]
        self.WALL = info[0]["shorthand"]
        self.SUPPORT = info[1]["shorthand"]
        self.TURRET = info[2]["shorthand"]
        self.SCOUT = info[3]["shorthand"]
        self.DEMOLISHER = info[4]["shorthand"]
        self.INTERCEPTOR = info[5]["shorthand"]
        self.demolisher_cost = float(info[4].get("cost2", 2.0))
        self.MP = 1

        self.turrets = [[12, 12], [15, 12], [13, 11], [14, 11],
                        [11, 11], [16, 11], [3, 12], [24, 12]]
        self.walls = [[x, 13] for x in range(28) if x not in (13, 14)]
        self.upgrade_first = [[12, 13], [15, 13], [11, 13], [16, 13]]
        self.lane = [14, 0]

    def apply(self, game_state):
        """Stage this turn's commands onto a gamelib GameState. Never raises."""
        try:
            game_state.attempt_spawn(self.TURRET, self.turrets)
            game_state.attempt_spawn(self.WALL, self.walls)
            game_state.attempt_upgrade(self.upgrade_first + self.turrets +
                                       self.walls)
            if game_state.get_resource(self.MP) >= 3 * self.demolisher_cost:
                game_state.attempt_spawn(self.DEMOLISHER, self.lane, 1000)
        except Exception:
            pass  # a broken fallback must still submit an (empty) turn
