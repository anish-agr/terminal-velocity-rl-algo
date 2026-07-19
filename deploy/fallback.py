"""Zero-dependency scripted bots -- the deployment stack's bottom rungs.

Pure python + gamelib. FallbackBot runs the whole match if the sim or weights
are missing or broken, and supplies the guaranteed per-turn submission the
watchdog falls back to when the search misses its deadline. AntiRushBot is
the driver's opponent-adaptive override: a mobile-rush detector plus a
scripted counter the driver plays instead of the net while the detector is
engaged. Both are deterministic, config-driven, and import nothing at module
level, so this module can never be the thing that crashes.

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
    """Mobile-rush detector + scripted counter for the deployment driver.

    Covers both rush archetypes: mass scouts and mass demolishers
    (thresholds are cost-scaled, so a demolisher wave counts with double
    weight per unit). The driver feeds observe() one completed enemy turn at
    a time (units deployed, hp lost to breaches and where, hp we breached,
    enemy MP bank) and, while `engaged`, plays apply() instead of running
    the net search.

    The counter combines a sealed wall funnel (its only opening feeds a
    turret ring, so scouts cannot leak around it) with an interceptor screen
    that holds the lanes while the wall is still going up or the opponent
    shows a banked flood, and a banked counterattack cycled through a ring
    sally gate (the trap pocket that stops their units also blocks ours, so
    one ring turret steps aside for exactly one combat turn per wave).

    Hysteresis is Schmitt-trigger style so one noisy turn never flips the
    mode in either direction. Entry: enough flagged turns in the recent
    window (a genuinely large wave, heavy breaches taken alongside a real
    wave, or an enemy bank several turns of income deep) -- count and bank
    flags are suppressed while we hold a large breach lead, so an opponent
    the net is out-racing never benches the net. Exit: several consecutive
    clean turns -- and while engaged a turn stays dirty on a sustained large
    wave or a deep enemy bank, because a rusher reloading between floods
    looks quiet exactly when dropping the screen would be fatal.

    All thresholds are income-scaled: MP income grows over the match, so a
    static cutoff would eventually flag every opponent's ordinary working
    balance. Bank thresholds sit above the decay cap ordinary banking rides,
    so a bank alone neither engages nor holds the counter -- banked floods
    are caught on launch, and breach evidence always engages immediately.
    """

    ENTRY_WAVE_INCOMES = 4.5   # one wave this many incomes deep = real flood
    BREACH_SPIKE = 4.0         # hp lost in one enemy turn...
    BREACH_MIN_SCOUTS = 5      # ...alongside a real scout wave, or
    BREACH_MIN_DEMOS = 3       # ...a real demolisher wave -> flagged
    BANK_ENTRY_INCOMES = 5.0   # bank flag threshold; above the decay cap
    WIN_MARGIN = 8.0           # breach lead at which count/bank flags are
    #   suppressed (breach-driven flags stay live)
    ALERT_WAVE_INCOMES = 3.0   # follow-up wave size that confirms an alert
    SUSTAIN_INCOME_FRAC = 2.5  # engaged: only a wave this large stays dirty
    WAVE_MEMORY_FRAC = 0.8     # waves this size update screen-aim memory
    BANK_HOLD_INCOMES = 5.0    # engaged: a bank this deep stays dirty
    FLOODER_BANK_INCOMES = 3.0  # a proven flooder reloading this deep is
    #   about to flood again -> re-engage before the launch, not after
    WINDOW = 4                 # flag window length
    ENGAGE_OF = 2              # engage when this many of WINDOW are flagged
    EXIT_CLEAN = 3             # disengage after this many clean turns
    THREAT_DECAY = 0.75        # per-turn fade of the remembered wave size
    THREAT_AFTERGLOW = 0.5     # weight of that memory in screen sizing
    MP_PER_INTERCEPTOR = 5.0   # screen sizing: one interceptor per ~5 MP of
    #   measured threat
    SCREEN_MIN_OPEN = 3        # screen floor while the wall line is unsealed
    SCREEN_CAP = 12            # sanity cap on one turn's screen
    PRESSURE_INCOMES = 2.0     # live bank at which defense preempts offense;
    #   keyed on the current bank because the post-flood turns (bank spent)
    #   are exactly when a counterattack is safe
    SHOWN_WAVE_INCOMES = 1.0   # a wave this size proves the opponent floods;
    #   until then a big idle bank only ever draws a token screen
    IDLE_BANK_INCOMES = 1.5    # a proven flooder below this bank is between
    #   floods: throttle the screen and bank for the counterattack
    SCREEN_IDLE_MAX = 1        # token screen while a flooder is refilling
    FLOODER_RESERVE_MP = 4.0   # extra bank the counterattack must clear
    #   against a proven flooder -- offense only from true surplus
    WAVE_MP = 10.0             # bank needed to launch a counterattack wave
    GATE_PREP_MP = 7.0         # bank at which the sally gate is marked
    HOLD_TURNS = 3             # breach-free turns = rush repelled; with a
    #   spent enemy bank this is a safe-siege window: drop the reserve and
    #   push instead of banking to a tiebreak loss
    HOLD_DEMOS_MP = 12.0       # safe siege: switch the sally to demolishers
    MEGA_BANK_INCOMES = 3.0    # under pressure, a bank this deep will blow
    #   through the trap pocket: seal the gap for the turn, restore the trap
    #   on the next quiet turn (never sealed while our own wave is out)
    TURRET_DMG_EST = 6.0       # per-attacker-per-step damage estimate for
    #   lane scoring, biased low so counterattacks are not over-held
    SUICIDE_RATIO = 1.5        # skip a wave when projected path damage
    #   exceeds this multiple of its hp; bank instead and fire later
    STALL_TURNS = 4            # breach-free turns against a bank-only
    #   opponent = a stalemate our defense is winning: push instead of
    #   pouring MP into interceptors until the tiebreak. Self-correcting --
    #   any breach resets the counter and defense resumes

    def __init__(self, config):
        info = config["unitInformation"]
        self.WALL = info[0]["shorthand"]
        self.SUPPORT = info[1]["shorthand"]
        self.TURRET = info[2]["shorthand"]
        self.SCOUT = info[3]["shorthand"]
        self.DEMOLISHER = info[4]["shorthand"]
        self.INTERCEPTOR = info[5]["shorthand"]
        self.scout_cost = float(info[3].get("cost2", 1.0))
        self.demolisher_cost = float(info[4].get("cost2", 2.0))
        self.interceptor_cost = float(info[5].get("cost2", 1.0))
        self.scout_hp = float(info[3].get("startHealth", 15.0))
        self.demolisher_hp = float(info[4].get("startHealth", 5.0))
        self.MP = 1
        res = config.get("resources", {})
        self.mp_per_round = float(res.get("bitsPerRound", 5.0))
        self.mp_growth = float(res.get("bitGrowthRate", 1.0))
        self.mp_interval = int(res.get("turnIntervalForBitSchedule", 10)) or 10

        # Funnel geometry in absolute coordinates -- the driver always plays
        # the bottom seat, so no flip is needed. List order is build
        # priority, and every cell is re-attempted every turn, so anything a
        # flood destroys is rebuilt the next turn. The second row densifies
        # the flank approaches the 6-turret ring does not cover.
        self.turrets = [[12, 12], [15, 12], [13, 11], [14, 11],
                        [11, 11], [16, 11], [3, 12], [24, 12],
                        [7, 12], [20, 12], [10, 12], [17, 12]]
        self.walls = [[x, 13] for x in range(28) if x not in (13, 14)]
        # The gap stays open as a one-way trap: the turret ring below it
        # forms a closed pocket, so enemy units that walk in dead-end and
        # self-destruct under six turrets.
        self.gap = [[13, 13], [14, 13]]
        # Sally gate: the ring pocket that traps their units also blocks our
        # counterattack, so offense cycles one ring turret -- marked for
        # removal on the prep turn, the wave exits through the hole the next
        # turn, and the turret is rebuilt the turn after.
        self.gate = [[14, 11]]
        # The wave's corridor must stay free of leftover structures or the
        # wave seals itself in.
        self.lane_clear = [[13, y] for y in range(11)] + \
                          [[14, y] for y in range(11)]
        self.upgrade_first = [[12, 13], [15, 13], [11, 13], [16, 13]]
        # Upgrade groups per lane so apply() can harden whichever lane the
        # opponent is actually breaching first.
        self.lane_upgrades = {
            "center": self.upgrade_first + [[13, 11], [14, 11], [12, 12],
                                            [15, 12], [11, 11], [16, 11],
                                            [10, 12], [17, 12]] +
                      [[x, 13] for x in range(10, 18) if x not in (13, 14)],
            "left": [[3, 12], [7, 12]] + [[x, 13] for x in range(0, 10)],
            "right": [[24, 12], [20, 12]] + [[x, 13] for x in range(18, 28)],
        }
        # Shield pylons flanking the counterattack lane: upgraded supports
        # roughly double a scout's effective hp, which is what lets a
        # counter wave survive the turrets guarding the funnel exit.
        self.supports = [[12, 10], [15, 10], [12, 8], [15, 8]]
        self.screens = [[13, 0], [6, 7], [21, 7]]  # gap mouth, then flanks
        # Counterattack spawn candidates on both bottom edges -- left-edge
        # cells target the top-right edge and vice versa, so the two groups
        # produce genuinely different paths. apply() scores every free
        # candidate with real gamelib pathing and fires the least-defended
        # one instead of always ramming the same corridor.
        self.counter_lanes = [[13, 0], [12, 1], [11, 2], [10, 3],
                              [14, 0], [15, 1], [16, 2], [17, 3]]

        self.flags = []        # rush flags for the last WINDOW observed turns
        self.clean = 0         # consecutive clean turns
        self.engaged = False
        self.gate_open = False  # gate turret marked last turn: wave turn
        self.income = self.mp_per_round  # refreshed by observe() each turn
        self.threat_mp = 0.0   # decayed memory of the biggest recent wave MP
        self.last_taken = 0.0  # breach hp taken on the last observed turn
        self.breach_free = 0   # consecutive observed turns taking no breaches
        self.total_dealt = 0.0   # cumulative breach ledger, both directions
        self.total_taken = 0.0
        self.hot = {"center": 0.0, "left": 0.0, "right": 0.0}  # breach heat
        self.alert = False     # any flag in the window: pre-harden signal
        self.is_flooder = False  # sticky: the opponent has breached us with
        #   a real wave at least once; their reloaded bank re-engages the
        #   screen for the rest of the match
        self.attack_cols = []  # mirrored columns of the last observed wave
        #   (an attacker crossing the diamond exits on the opposite flank)

    def observe(self, scouts, demolishers, interceptors, breaches_taken,
                enemy_mp=0.0, turn=0, breaches_dealt=0.0, breach_xs=None,
                spawn_xs=None):
        """Ingest one completed enemy turn; returns `engaged`. Never raises."""
        try:
            scouts = int(scouts)
            demos = int(demolishers)
            income = self.mp_per_round + \
                self.mp_growth * (int(turn) // self.mp_interval)
            self.income = income
            wave_mp = scouts * self.scout_cost + demos * self.demolisher_cost
            self.threat_mp = max(wave_mp, self.threat_mp * self.THREAT_DECAY)
            self.total_dealt += float(breaches_dealt)
            self.total_taken += float(breaches_taken)
            self.last_taken = float(breaches_taken)
            self.breach_free = 0 if float(breaches_taken) > 0 \
                else self.breach_free + 1
            for lane in self.hot:
                self.hot[lane] *= 0.5
            for x in (breach_xs or ()):
                x = int(x)
                self.hot["left" if x < 10 else
                         ("right" if x > 17 else "center")] += 1.0
            if spawn_xs:   # a real wave: remember its arrival columns
                if wave_mp >= self.WAVE_MEMORY_FRAC * income:
                    self.attack_cols = sorted(
                        {27 - int(x) for x in spawn_xs})[:3]
            winning = self.total_dealt - self.total_taken >= self.WIN_MARGIN
            hurt = float(breaches_taken) >= self.BREACH_SPIKE and (
                scouts >= self.BREACH_MIN_SCOUTS or
                demos >= self.BREACH_MIN_DEMOS)
            # Any breach that comes with a real wave marks a proven flooder,
            # not only a heavy one: even a small poke proves the lane leaks,
            # so the next big bank should re-engage before the flood.
            if float(breaches_taken) > 0 and (
                    scouts >= self.BREACH_MIN_SCOUTS or
                    demos >= self.BREACH_MIN_DEMOS):
                self.is_flooder = True   # remembered for the rest of the match
            spike = (wave_mp >= self.ENTRY_WAVE_INCOMES * income
                     or float(enemy_mp) >= self.BANK_ENTRY_INCOMES * income
                     or (self.is_flooder and
                         float(enemy_mp) >= self.FLOODER_BANK_INCOMES * income)
                     or (self.alert and
                         wave_mp >= self.ALERT_WAVE_INCOMES * income))
            entry = hurt or (spike and not winning)
            # Once we hold a real breach lead the net is out-racing them:
            # sustained waves must not keep us engaged. A genuine reversal
            # still re-engages instantly through the breach-driven path.
            dirty = entry or (self.engaged and not winning and (
                wave_mp >= self.SUSTAIN_INCOME_FRAC * income or
                float(enemy_mp) >= self.BANK_HOLD_INCOMES * income or
                # A proven flooder refilling past a working bank is loading
                # the next wave; bank decay keeps a multi-turn reload just
                # under the entry thresholds, so without this the exit fires
                # right before the kill wave.
                (self.is_flooder and
                 float(enemy_mp) >= self.PRESSURE_INCOMES * income)))
            self.flags.append(bool(entry))
            del self.flags[:-self.WINDOW]
            self.clean = 0 if dirty else self.clean + 1
            if self.engaged:
                self.engaged = self.clean < self.EXIT_CLEAN
                if not self.engaged:
                    self.gate_open = False  # never fire a stale sally wave
            else:
                # A single hard breach engages NOW: waiting for a second
                # window flag hands a corner banker several free floods,
                # since its sub-threshold waves in between never flag.
                # hurt is damage-gated, so opponents that never breach us
                # leave the net in control.
                self.engaged = sum(self.flags) >= self.ENGAGE_OF or hurt
            # Pre-harden only while a threat is live and we are not already
            # ahead -- a winning game must not bleed SP into walls it does
            # not need.
            self.alert = (self.engaged or sum(self.flags) >= 1) and not winning
        except Exception:
            pass
        return self.engaged

    def _upgrade_order(self):
        """Lane upgrade lists, hottest (most-breached) lane first; ties keep
        the center-first default. gamelib skips already-upgraded cells, so
        repeated entries cost nothing."""
        lanes = ("center", "left", "right")
        out = []
        for lane in sorted(lanes, key=lambda l: (-self.hot.get(l, 0.0),
                                                 lanes.index(l))):
            out += self.lane_upgrades.get(lane, [])
        return out

    def _wall_order(self):
        """Wall row in build order, hottest (most-breached) lane first, so
        SP goes where the damage is instead of patching left-to-right while
        the breached corner stays open."""
        left = [[x, 13] for x in range(0, 10)]
        center = [[x, 13] for x in range(10, 18) if x not in (13, 14)]
        right = [[x, 13] for x in range(18, 28)]
        seg = {"left": left, "center": center, "right": right}
        lanes = ("center", "left", "right")
        out = []
        for lane in sorted(lanes, key=lambda l: (-self.hot.get(l, 0.0),
                                                 lanes.index(l))):
            out += seg[lane]
        return out

    def _best_lane(self, game_state, wave_hp, desperate=False):
        """Least-defended counterattack spawn cell by real gamelib pathing.

        Scores each free candidate by the number of enemy attackers covering
        every step of the predicted path. Lanes that dead-end on our half
        would feed the wave to our own funnel, so they are always skipped.
        When not `desperate`, a lane whose projected damage exceeds
        SUICIDE_RATIO x the wave's hp returns None so the caller banks the
        MP for a bigger wave. When `desperate` (a stalemate we are losing --
        holding just donates the health tiebreak) the least-defended
        reaching lane is returned regardless. Never raises."""
        best, best_danger = None, None
        for lane in self.counter_lanes:
            try:
                if game_state.contains_stationary_unit(lane):
                    continue
                path = game_state.find_path_to_edge(lane)
                if not path or len(path) < 2:
                    continue
                if path[-1][1] < 14:
                    continue   # dead-ends on our half: self-destruct feed
                danger = 0
                for cell in path:
                    danger += len(game_state.get_attackers(cell, 0) or ())
            except Exception:
                continue
            if best_danger is None or danger < best_danger:
                best, best_danger = lane, danger
        if best is None:
            return None
        if not desperate and \
                best_danger * self.TURRET_DMG_EST > wave_hp * self.SUICIDE_RATIO:
            return None
        return best

    def preharden(self, game_state):
        """Defense-only fortification staged on a rush alert while the net
        still plays the turn: walls, turrets, and upgrades only -- no
        removals, supports, or mobiles. The net's plan is staged after this,
        so gamelib simply drops whatever it can no longer afford; by the
        time the detector fully engages, the fortress is already standing.
        Never raises."""
        try:
            game_state.attempt_spawn(self.TURRET, self.turrets)
            game_state.attempt_spawn(self.WALL, self._wall_order())
            game_state.attempt_upgrade(self._upgrade_order() +
                                       self.turrets + self.walls)
        except Exception:
            pass

    def apply(self, game_state):
        """Stage one anti-rush turn onto a gamelib GameState. Never raises."""
        try:
            turrets = [t for t in self.turrets if t not in self.gate] \
                if self.gate_open else self.turrets
            game_state.attempt_spawn(self.TURRET, turrets)
            game_state.attempt_spawn(self.WALL, self._wall_order())
            try:
                enemy_mp = float(game_state.get_resource(self.MP, 1))
            except Exception:
                enemy_mp = 0.0
            # Standing defense has held for STALL_TURNS: whatever they are
            # banking is not breaking through -- a stalemate to be broken by
            # offense, not defended to a tiebreak.
            holding = self.breach_free >= self.STALL_TURNS
            # Mega-flood seal: a bank the trap pocket cannot eat is about to
            # launch -- plug the gap for this turn; the next quieter turn
            # restores the trap.
            mega = (not self.gate_open and not holding and
                    enemy_mp >= self.MEGA_BANK_INCOMES * self.income)
            if mega:
                game_state.attempt_spawn(self.WALL, self.gap)
            # Keep the trap gap and the wave corridor clear of leftover
            # structures.
            clear = self.lane_clear if mega else self.gap + self.lane_clear
            for loc in clear:
                if game_state.contains_stationary_unit(loc):
                    game_state.attempt_remove([loc])
            # Posture from the wall line as it will stand THIS turn (gamelib
            # adds staged builds to the map, and builds land before combat);
            # an intentionally open sally gate still counts as sealed.
            try:
                missing = sum(1 for loc in self.walls
                              if not game_state.contains_stationary_unit(loc))
            except Exception:
                missing = 0
            sealed = missing <= 2
            if sealed:   # shield pylons only once the seal is paid for
                game_state.attempt_spawn(self.SUPPORT, self.supports)
            game_state.attempt_upgrade(self._upgrade_order() + self.supports +
                                       self.turrets + self.walls)
            mp = game_state.get_resource(self.MP)
            # Measured threat: the bank they could throw right now, floored
            # by an afterglow of the biggest recent wave. An opponent who
            # has never shown a real wave only ever draws a token screen,
            # however big their idle bank; a proven flooder's bank is always
            # a real threat, even after a long quiet reload.
            shown = self.is_flooder or \
                self.threat_mp >= self.SHOWN_WAVE_INCOMES * self.income
            threat = max(enemy_mp if shown else min(enemy_mp, self.income),
                         self.threat_mp * self.THREAT_AFTERGLOW)
            pressure = (self.last_taken > 0 or
                        (shown and
                         enemy_mp >= self.PRESSURE_INCOMES * self.income))
            # Interceptor screen sized to the threat and concentrated on the
            # turns a flood is imminent; throttled to a token while a proven
            # flooder is refilling, so spend follows the flood cycle.
            spots = []
            for c in self.attack_cols:   # observed arrival columns first
                c = min(25, max(2, int(c)))
                cell = [c, 13 - c] if c <= 13 else [c, c - 14]
                if not game_state.contains_stationary_unit(cell) \
                        and cell not in spots:
                    spots.append(cell)
            for s in self.screens:       # then the default gap/flank spots
                if not game_state.contains_stationary_unit(s) \
                        and s not in spots:
                    spots.append(s)
            want = int(threat // self.MP_PER_INTERCEPTOR)
            if holding or (shown and
                           enemy_mp < self.IDLE_BANK_INCOMES * self.income):
                want = min(want, self.SCREEN_IDLE_MAX)
            if not sealed:
                want = max(want, self.SCREEN_MIN_OPEN)
            want = min(want, self.SCREEN_CAP)
            n = min(want, int(mp // self.interceptor_cost)) \
                if (spots and self.interceptor_cost > 0) else 0
            for i in range(n):
                game_state.attempt_spawn(self.INTERCEPTOR,
                                         [spots[i % len(spots)]])
            # Counterattack: offense runs on a two-turn gate cycle. Prep
            # turn: once the bank is a turn short of a wave, mark the gate
            # turret for removal. Wave turn: dump the banked units from the
            # scored lane; the turn after, the turret build restores the
            # ring and the trap is whole again.
            mp = game_state.get_resource(self.MP)
            # Against a proven flooder the counterattack must clear an extra
            # reserve on top of the wave cost. Once the rush is repelled AND
            # their bank is spent (safe siege), drop the reserve and push
            # hard so a held stalemate is not surrendered to a tiebreak.
            safe_siege = (self.breach_free >= self.HOLD_TURNS and
                          enemy_mp < self.IDLE_BANK_INCOMES * self.income)
            push_now = safe_siege or holding
            reserve = 0.0 if push_now else \
                (self.FLOODER_RESERVE_MP if shown else 0.0)
            if self.gate_open:
                # The gate was already committed on the prep turn (a ring
                # turret is down), so fire the banked wave whenever we are
                # not actively being breached -- even if the flooder's bank
                # has since refilled. The defensive screen is already funded
                # above and `reserve` stays banked, so this only ever spends
                # true surplus; a live breach still holds the wave.
                if self.last_taken == 0 and mp >= self.WAVE_MP + reserve and \
                        self.scout_cost > 0:
                    if push_now and mp >= self.HOLD_DEMOS_MP and \
                            self.demolisher_cost > 0:
                        kind, unit_cost, unit_hp = (
                            self.DEMOLISHER, self.demolisher_cost,
                            self.demolisher_hp)
                    else:
                        kind, unit_cost, unit_hp = (
                            self.SCOUT, self.scout_cost, self.scout_hp)
                    count = int(mp // unit_cost)
                    # Fire down the least-defended real path; normally hold
                    # the bank instead of feeding a hopeless lane, but in a
                    # stalemate we are losing on the breach ledger, grind
                    # anyway -- banking to turn 100 donates the tiebreak.
                    desperate = holding and self.total_taken > self.total_dealt
                    lane = self._best_lane(game_state, count * unit_hp,
                                           desperate=desperate)
                    if lane is not None:
                        game_state.attempt_spawn(kind, [lane], count)
                self.gate_open = False
            elif sealed and (holding or not pressure) and \
                    mp >= self.GATE_PREP_MP + reserve:
                for loc in self.gate:
                    if game_state.contains_stationary_unit(loc):
                        game_state.attempt_remove([loc])
                self.gate_open = True
        except Exception:
            pass  # a broken counter must still submit an (empty) turn
