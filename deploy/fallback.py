"""Zero-dependency scripted bots (ARCHITECTURE §9.2) — the last rung.

Pure python + gamelib. FallbackBot runs the whole match if the .so or weights
are missing or broken, and supplies the guaranteed per-turn submission the
watchdog falls back to when the search misses its deadline. AntiRushBot is the
driver's opponent-adaptive override: a scout-rush detector plus a scripted
counter the driver plays INSTEAD of the net while the detector is engaged.
Both are deterministic, config-driven, no imports beyond gamelib (which the
driver passes in — this module imports NOTHING at module level so it can
never be the thing that crashes).

FallbackBot layout: wall row at y=13 with a two-tile center gap, turrets
ringing the gap, corner guns, gap-shoulder wall upgrades first, then a
demolisher wave through the funnel whenever a 3-wave is affordable.
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


class AntiRushBot:
    """Mass-scout-rush detector + scripted counter for the deployment driver.

    The driver feeds observe() one completed enemy turn at a time (scouts /
    demolishers / interceptors deployed, hp lost to breaches, enemy MP bank)
    and, while `engaged`, plays apply() instead of running the net search.
    The counter combines the FallbackBot funnel (a sealed wall row whose only
    opening feeds a turret ring — scouts cannot leak around it) with
    train/scripted.py::turtle's active pieces rewritten against the gamelib
    GameState API: an interceptor screen that holds the lanes while the wall
    is still going up or the opponent shows a banked flood, and a banked
    scout counterattack cycled through a ring sally gate (the trap pocket
    that stops THEIR units also blocks ours, so one ring turret steps aside
    for exactly one combat turn per wave).

    Hysteresis is Schmitt-trigger style so one noisy turn never flips the
    mode in either direction. ENTRY: >= ENGAGE_OF of the last WINDOW turns
    flagged (a big scout wave, or heavy breaches from a scout-dominant mix).
    EXIT: EXIT_CLEAN consecutive clean turns — and while engaged a turn stays
    dirty on a smaller sustained scout wave or a large enemy MP bank, because
    a rusher reloading between floods looks quiet exactly when dropping the
    screen would be fatal.
    """

    ENTRY_SCOUTS = 8        # scouts in one enemy turn -> flagged
    BREACH_SPIKE = 4.0      # hp lost in one turn with a scout-dominant mix...
    BREACH_MIN_SCOUTS = 5   # ...but only alongside a real scout wave
    BANK_ENTRY_INCOMES = 3.5   # enemy bank >= 3.5 turns of income -> flagged
    #   (a mass-flood rusher banking toward one giant wave is visible for many
    #   turns before the flood lands — decay caps a bank at 4x income, so 3.5x
    #   only ever flags deliberate max-banking, never save-and-spend cycles)
    SUSTAIN_SCOUTS = 4      # while engaged, this many scouts is still dirty
    BANK_HOLD_INCOMES = 2.5    # while engaged, a bank this big is dirty
    WINDOW = 3
    ENGAGE_OF = 2           # engage when >= 2 of the last WINDOW are flagged
    EXIT_CLEAN = 3          # disengage after this many consecutive clean turns

    def __init__(self, config):
        info = config["unitInformation"]
        self.WALL = info[0]["shorthand"]
        self.SUPPORT = info[1]["shorthand"]
        self.TURRET = info[2]["shorthand"]
        self.SCOUT = info[3]["shorthand"]
        self.INTERCEPTOR = info[5]["shorthand"]
        self.scout_cost = float(info[3].get("cost2", 1.0))
        self.interceptor_cost = float(info[5].get("cost2", 1.0))
        self.MP = 1
        res = config.get("resources", {})
        self.mp_per_round = float(res.get("bitsPerRound", 5.0))
        self.mp_growth = float(res.get("bitGrowthRate", 1.0))
        self.mp_interval = int(res.get("turnIntervalForBitSchedule", 10)) or 10

        # FallbackBot funnel geometry in absolute coordinates — the driver
        # always plays the bottom seat, so no flip is needed
        self.turrets = [[12, 12], [15, 12], [13, 11], [14, 11],
                        [11, 11], [16, 11], [3, 12], [24, 12]]
        self.walls = [[x, 13] for x in range(28) if x not in (13, 14)]
        # the gap stays open as a one-way trap: the turret ring below it forms
        # a closed pocket, so enemy units that walk in dead-end and
        # self-destruct under six turrets. (A net-built structure left in the
        # gap would merely turn the trap into a plain sealed wall.)
        self.gap = [[13, 13], [14, 13]]
        # sally gate: the ring pocket that traps THEIR units also blocks OUR
        # counterattack, so offense cycles one ring turret — marked for
        # removal on the prep turn, the wave exits through the hole the next
        # turn, and the turret is rebuilt the turn after. An enemy leak
        # through the open ring must run the center corridor covered by the
        # four remaining ring turrets; the wall row itself never opens.
        self.gate = [[14, 11]]
        # the wave's corridor (center columns up to the ring) must stay free
        # of leftover net-built structures or the wave seals itself in
        self.lane_clear = [[13, y] for y in range(11)] + \
                          [[14, y] for y in range(11)]
        self.upgrade_first = [[12, 13], [15, 13], [11, 13], [16, 13]]
        # shield pylons flanking the counterattack lane (x 13/14): upgraded
        # supports roughly double a scout's effective hp, which is what lets
        # a counter wave survive the turrets guarding the funnel exit
        self.supports = [[12, 10], [15, 10], [12, 8], [15, 8]]
        self.screens = [[13, 0], [6, 7], [21, 7]]  # gap mouth, then flanks
        # counterattack spawn candidates — all on the bottom-LEFT edge so
        # every wave shares one target edge (top-right) beyond the gap; any
        # cell may be blocked by a structure a previous (net) turn built, so
        # apply() takes the first free one
        self.counter_lanes = [[13, 0], [12, 1], [11, 2], [10, 3]]

        self.flags = []        # rush flags for the last WINDOW observed turns
        self.clean = 0         # consecutive clean turns
        self.engaged = False
        self.gate_open = False  # gate turret marked last turn: wave turn

    def observe(self, scouts, demolishers, interceptors, breaches_taken,
                enemy_mp=0.0, turn=0):
        """Ingest one completed enemy turn; returns `engaged`. Never raises.

        Bank thresholds scale with the per-turn MP income (which grows over
        the match) — a static cutoff would eventually flag every opponent's
        ordinary working balance and lock the counter in permanently.
        """
        try:
            scouts = int(scouts)
            income = self.mp_per_round + \
                self.mp_growth * (int(turn) // self.mp_interval)
            scout_dominant = scouts > int(demolishers) + int(interceptors)
            entry = (scouts >= self.ENTRY_SCOUTS
                     or (float(breaches_taken) >= self.BREACH_SPIKE
                         and scout_dominant
                         and scouts >= self.BREACH_MIN_SCOUTS)
                     or float(enemy_mp) >= self.BANK_ENTRY_INCOMES * income)
            dirty = entry or (self.engaged and (
                scouts >= self.SUSTAIN_SCOUTS or
                float(enemy_mp) >= self.BANK_HOLD_INCOMES * income))
            self.flags.append(bool(entry))
            del self.flags[:-self.WINDOW]
            self.clean = 0 if dirty else self.clean + 1
            if self.engaged:
                self.engaged = self.clean < self.EXIT_CLEAN
            else:
                self.engaged = sum(self.flags) >= self.ENGAGE_OF
        except Exception:
            pass
        return self.engaged

    def apply(self, game_state):
        """Stage one anti-rush turn onto a gamelib GameState. Never raises."""
        try:
            turrets = [t for t in self.turrets if t not in self.gate] \
                if self.gate_open else self.turrets
            game_state.attempt_spawn(self.TURRET, turrets)
            game_state.attempt_spawn(self.WALL, self.walls)
            # keep the trap gap and the wave corridor clear of leftover
            # net-built structures
            for loc in self.gap + self.lane_clear:
                if game_state.contains_stationary_unit(loc):
                    game_state.attempt_remove([loc])
            # posture from the wall line as it will stand THIS turn (gamelib
            # adds staged builds to the map, and builds land before combat);
            # an intentionally open sally gate still counts as sealed
            try:
                missing = sum(1 for loc in self.walls
                              if not game_state.contains_stationary_unit(loc))
            except Exception:
                missing = 0
            sealed = missing <= 2
            if sealed:   # shield pylons only once the seal is paid for
                game_state.attempt_spawn(self.SUPPORT, self.supports)
            game_state.attempt_upgrade(self.upgrade_first + self.supports +
                                       self.turrets + self.walls)
            mp = game_state.get_resource(self.MP)
            try:
                enemy_mp = float(game_state.get_resource(self.MP, 1))
            except Exception:
                enemy_mp = 0.0
            # interceptor screen: while the funnel is open, keep bodies in the
            # lanes every turn; once sealed, spend only to pre-empt a visible
            # banked flood (one interceptor per ~6 MP banked), capped, never
            # overdrawn — skipping screen spots blocked by earlier builds
            spots = [s for s in self.screens
                     if not game_state.contains_stationary_unit(s)]
            want = min((0 if sealed else 3) + int(enemy_mp // 6.0), 9)
            n = min(want, int(mp // self.interceptor_cost)) \
                if (spots and self.interceptor_cost > 0) else 0
            for i in range(n):
                game_state.attempt_spawn(self.INTERCEPTOR,
                                         [spots[i % len(spots)]])
            # counterattack: offense runs on a two-turn gate cycle. Prep
            # turn: once the bank is a turn short of a wave, mark the gate
            # turret for removal. Wave turn: dump the banked scouts from a
            # bottom-left edge cell — they climb the shield-pylon lane and
            # exit through the ring hole and the gap (the turret build above
            # skipped the gate cell this turn); the turn after, the turret
            # build restores the ring and the trap is whole again.
            mp = game_state.get_resource(self.MP)
            if self.gate_open:
                if mp >= 10.0 and self.scout_cost > 0:
                    count = int(mp // self.scout_cost)
                    for lane in self.counter_lanes:
                        if (game_state.attempt_spawn(self.SCOUT, [lane],
                                                     count) or 0) > 0:
                            break
                self.gate_open = False
            elif sealed and mp >= 7.0:
                for loc in self.gate:
                    if game_state.contains_stationary_unit(loc):
                        game_state.attempt_remove([loc])
                self.gate_open = True
        except Exception:
            pass  # a broken counter must still submit an (empty) turn
