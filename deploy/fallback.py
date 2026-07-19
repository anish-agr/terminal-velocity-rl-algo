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
    """Mobile-rush detector + scripted counter for the deployment driver.

    Covers both rush archetypes seen on ladder: mass scouts and mass
    demolishers (thresholds are cost-scaled, so a demolisher wave counts
    with double weight per unit). The driver feeds observe() one completed
    enemy turn at a time (scouts / demolishers / interceptors deployed, hp
    lost to breaches and where, hp we breached, enemy MP bank) and, while
    `engaged`, plays apply() instead of running the net search.
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
    flagged (a big scout or demolisher wave, heavy breaches taken alongside
    a real wave, or an enemy bank several turns of income deep) — count and
    bank flags are suppressed while we hold a large breach lead, so an
    opponent the net is out-racing never benches the net. EXIT: EXIT_CLEAN
    consecutive clean turns — and while engaged a turn stays dirty on a
    sustained mobile wave (cost-scaled vs income) or a large enemy MP bank,
    because a rusher reloading between floods looks quiet exactly when
    dropping the screen would be fatal.
    """

    # DETECTOR RETUNE (2026-07-18, ranked-replay ms-taken audit): with the
    # old raw-count entry (8 scouts) and bank triggers (3.0x/2.5x income),
    # the override engaged by ~turn 11 in 9 of our 10 newest ranked games and
    # NEVER disengaged — the ladder median wave is ~19 scouts and normal
    # banking rides the 4x-income decay cap, so ordinary meta play flagged
    # every window and every held bank stayed "dirty". Measured result: the
    # net played only 5-9 turns/game; the one game it played fully was our
    # cleanest win, while the override lost 4 grinder games. Entry is now
    # income-scaled to true flood size, bank thresholds sit ABOVE the 4x
    # decay cap (a bank alone neither engages nor holds), and sustain only
    # marks genuinely large waves so EXIT_CLEAN can actually fire between an
    # ordinary opponent's waves. The breach-evidence path is unchanged: a
    # rush that is actually hurting us still engages immediately.
    ENTRY_WAVE_INCOMES = 4.5   # one enemy wave >= 4.5 turns of income ->
    #   flagged (income 5 -> 22+ MP: real floods only; the meta's banked
    #   2.5-4x commit waves stay under this at every income level)
    BREACH_SPIKE = 4.0      # hp lost to breaches in one enemy turn...
    BREACH_MIN_SCOUTS = 5   # ...alongside a real scout wave, or
    BREACH_MIN_DEMOS = 3    # ...a real demolisher wave -> flagged
    BANK_ENTRY_INCOMES = 5.0   # enemy bank -> flagged. Above the 4x-income
    #   decay cap: unreachable by ordinary banking, only touched by configs
    #   with weaker decay — banked floods are caught on launch instead
    WIN_MARGIN = 8.0        # cumulative net breach lead at which count/bank
    #   flags are suppressed: an opponent we are out-racing is not a rush,
    #   however many units they field (breach-driven flags stay live)
    ALERT_WAVE_INCOMES = 3.0   # while ALERTED, a repeat wave must still be
    #   flood-sized (3x income) to count as the follow-up flag — at the old
    #   1.0x every ordinary meta wave confirmed the alert
    SUSTAIN_INCOME_FRAC = 2.5  # while engaged, only a wave this large keeps
    #   the turn dirty — the wave turn itself, not the quiet turns between
    #   an ordinary opponent's commits, so the exit can actually fire
    WAVE_MEMORY_FRAC = 0.8  # separate, unchanged: waves this size still
    #   update the screen-aim column memory (attack_cols)
    BANK_HOLD_INCOMES = 5.0    # while engaged, a bank this big is dirty —
    #   above the decay cap for the same reason as BANK_ENTRY_INCOMES (the
    #   old 2.5x kept every normal banker's reload "dirty" forever)
    WINDOW = 4              # flag window: floods landing every 3rd turn
    #   (the ladder pattern that engaged 20+ turns late) still meet ENGAGE_OF
    ENGAGE_OF = 2           # engage when >= 2 of the last WINDOW are flagged
    EXIT_CLEAN = 3          # disengage after this many consecutive clean turns
    THREAT_DECAY = 0.75     # per-turn fade of the remembered attack size
    THREAT_AFTERGLOW = 0.5  # weight of that memory in screen sizing — small
    #   insurance right after a flood without starving the counterattack
    MP_PER_INTERCEPTOR = 5.0   # screen sizing: one interceptor per ~5 MP of
    #   measured threat (a 40-hp interceptor one-shots 5-hp demolishers and
    #   trades into several scouts)
    SCREEN_MIN_OPEN = 3     # screen floor while the wall line is unsealed
    SCREEN_CAP = 12         # sanity cap on one turn's screen
    PRESSURE_INCOMES = 2.0  # enemy BANK >= 2 turns of income, or breaches
    #   taken last turn -> defense first: no sally prep, no counterattack.
    #   Deliberately keyed on the current bank, not the remembered wave: the
    #   turns right after a flood (their bank is empty) are exactly when a
    #   counterattack is safe — and the only way to win the hp race
    SHOWN_WAVE_INCOMES = 1.0   # a wave worth one turn's income proves the
    #   opponent actually floods. Until then a big bank is hypothetical: it
    #   gets a token screen and never suppresses our counterattack — an idle
    #   bank must not bleed us dry on interceptors (ladder 15340295)
    IDLE_BANK_INCOMES = 1.5    # a proven flooder below this bank is between
    #   floods: throttle the screen to SCREEN_IDLE_MAX and bank the rest, so
    #   interceptor spend concentrates on the turns a flood is imminent
    SCREEN_IDLE_MAX = 1     # token screen while the flooder is refilling
    FLOODER_RESERVE_MP = 4.0   # extra bank the counterattack must clear on
    #   top of WAVE_MP against a proven flooder — offense only from true
    #   surplus after the defensive screen is funded
    WAVE_MP = 10.0          # bank needed to launch a counterattack wave
    GATE_PREP_MP = 7.0      # bank at which the sally gate is marked
    HOLD_TURNS = 3          # engaged this many breach-free turns = the rush is
    #   repelled. Combined with a spent enemy bank (they cannot punish an
    #   aggressive push) this is a SAFE SIEGE window — it can never overlap the
    #   rush defense because it needs both zero breaches taken AND an empty
    #   enemy bank. There we drop the counterattack reserve and push
    #   demolishers instead of banking to a tiebreak loss (ladder 15341198 sat
    #   at 0-0 breaches to turn 99 and lost the coin flip)
    HOLD_DEMOS_MP = 12.0    # in the safe-siege window, bank at which the sally
    #   switches to demolishers — they break the structures scouts bounce off

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
        self.MP = 1
        res = config.get("resources", {})
        self.mp_per_round = float(res.get("bitsPerRound", 5.0))
        self.mp_growth = float(res.get("bitGrowthRate", 1.0))
        self.mp_interval = int(res.get("turnIntervalForBitSchedule", 10)) or 10

        # FallbackBot funnel geometry in absolute coordinates — the driver
        # always plays the bottom seat, so no flip is needed. List order is
        # build priority, and every cell is re-attempted every turn, so
        # anything a flood destroys is rebuilt the next turn. The second
        # row ([7]/[10]/[17]/[20], 12) densifies the flank approaches the
        # 6-turret ring does not cover against a 20-40 MP mixed wave.
        self.turrets = [[12, 12], [15, 12], [13, 11], [14, 11],
                        [11, 11], [16, 11], [3, 12], [24, 12],
                        [7, 12], [20, 12], [10, 12], [17, 12]]
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
        # upgrade groups per lane so apply() can harden whichever lane the
        # opponent is actually breaching first (hot-lane priority): each
        # lane's turrets plus its full third of the front wall — upgraded
        # walls have 3x the HP and blunt demolisher fire on that lane
        self.lane_upgrades = {
            "center": self.upgrade_first + [[13, 11], [14, 11], [12, 12],
                                            [15, 12], [11, 11], [16, 11],
                                            [10, 12], [17, 12]] +
                      [[x, 13] for x in range(10, 18) if x not in (13, 14)],
            "left": [[3, 12], [7, 12]] + [[x, 13] for x in range(0, 10)],
            "right": [[24, 12], [20, 12]] + [[x, 13] for x in range(18, 28)],
        }
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
        self.income = self.mp_per_round  # refreshed by observe() each turn
        self.threat_mp = 0.0   # decayed memory of the biggest recent wave MP
        self.last_taken = 0.0  # breach hp taken on the last observed turn
        self.breach_free = 0   # consecutive observed turns taking no breaches
        self.total_dealt = 0.0   # cumulative breach ledger, both directions —
        self.total_taken = 0.0   # a big lead means we are winning, not rushed
        self.hot = {"center": 0.0, "left": 0.0, "right": 0.0}  # breach heat
        self.alert = False     # any flag in the window: pre-harden signal
        self.attack_cols = []  # mirrored columns of the last observed wave —
        #   an attacker crossing the diamond exits on the opposite flank, so
        #   the screen spawns under 27-x for each enemy spawn column x

    def observe(self, scouts, demolishers, interceptors, breaches_taken,
                enemy_mp=0.0, turn=0, breaches_dealt=0.0, breach_xs=None,
                spawn_xs=None):
        """Ingest one completed enemy turn; returns `engaged`. Never raises.

        Bank thresholds scale with the per-turn MP income (which grows over
        the match) — a static cutoff would eventually flag every opponent's
        ordinary working balance and lock the counter in permanently. Count
        and bank flags are suppressed while we hold a big cumulative breach
        lead (we are out-racing them — not a rush we need to turtle against),
        but flags driven by breaches WE take always stay live.
        """
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
            spike = (wave_mp >= self.ENTRY_WAVE_INCOMES * income
                     or float(enemy_mp) >= self.BANK_ENTRY_INCOMES * income
                     or (self.alert and
                         wave_mp >= self.ALERT_WAVE_INCOMES * income))
            entry = hurt or (spike and not winning)
            dirty = entry or (self.engaged and (
                wave_mp >= self.SUSTAIN_INCOME_FRAC * income or
                float(enemy_mp) >= self.BANK_HOLD_INCOMES * income))
            self.flags.append(bool(entry))
            del self.flags[:-self.WINDOW]
            self.clean = 0 if dirty else self.clean + 1
            if self.engaged:
                self.engaged = self.clean < self.EXIT_CLEAN
                if not self.engaged:
                    self.gate_open = False  # never fire a stale sally wave
                    #   into a ring the net may have rebuilt meanwhile
            else:
                self.engaged = sum(self.flags) >= self.ENGAGE_OF
            self.alert = self.engaged or sum(self.flags) >= 1
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

    def preharden(self, game_state):
        """Defense-only fortification staged on a rush ALERT (a single flag)
        while the net still plays the turn: walls, turrets, and upgrades
        only — no removals, supports, or mobiles. The net's plan is staged
        after this, so gamelib simply drops whatever it can no longer
        afford; by the time the detector fully engages, the fortress is
        already standing instead of still going up. Never raises."""
        try:
            game_state.attempt_spawn(self.TURRET, self.turrets)
            game_state.attempt_spawn(self.WALL, self.walls)
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
            game_state.attempt_upgrade(self._upgrade_order() + self.supports +
                                       self.turrets + self.walls)
            mp = game_state.get_resource(self.MP)
            try:
                enemy_mp = float(game_state.get_resource(self.MP, 1))
            except Exception:
                enemy_mp = 0.0
            # measured threat: the bank they could throw right now, floored
            # by an afterglow of the biggest recent wave — but an opponent
            # who has never shown a real wave only ever gets a token screen,
            # however big their idle bank. Pressure — a proven flooder's
            # bank worth a real flood, or breaches taken last turn — sends
            # every MP to defense; the post-flood turns (bank spent) are the
            # sally window.
            shown = self.threat_mp >= self.SHOWN_WAVE_INCOMES * self.income
            threat = max(enemy_mp if shown else min(enemy_mp, self.income),
                         self.threat_mp * self.THREAT_AFTERGLOW)
            pressure = (self.last_taken > 0 or
                        (shown and
                         enemy_mp >= self.PRESSURE_INCOMES * self.income))
            # interceptor screen sized to that threat, concentrated on the
            # turns a flood is imminent (bank full) and throttled to a token
            # while a proven flooder is refilling — spend follows the flood
            # cycle instead of dripping a fixed screen every turn
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
            if shown and enemy_mp < self.IDLE_BANK_INCOMES * self.income:
                want = min(want, self.SCREEN_IDLE_MAX)
            if not sealed:
                want = max(want, self.SCREEN_MIN_OPEN)
            want = min(want, self.SCREEN_CAP)
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
            # vs a proven flooder the counterattack must clear an extra
            # reserve on top of the wave cost: offense only from true
            # surplus once the defensive screen is already funded. But once
            # the rush is repelled (breach-free) AND their bank is spent, we
            # are safe to push hard: drop the reserve and, on a real bank,
            # siege with demolishers so a held stalemate is not surrendered to
            # a tiebreak. The empty-bank guard keeps this out of the flooder
            # defense entirely — a refilling flooder still gets the reserve.
            safe_siege = (self.breach_free >= self.HOLD_TURNS and
                          enemy_mp < self.IDLE_BANK_INCOMES * self.income)
            reserve = 0.0 if safe_siege else \
                (self.FLOODER_RESERVE_MP if shown else 0.0)
            if self.gate_open:
                if not pressure and mp >= self.WAVE_MP + reserve and \
                        self.scout_cost > 0:
                    if safe_siege and mp >= self.HOLD_DEMOS_MP and \
                            self.demolisher_cost > 0:
                        kind, unit_cost = self.DEMOLISHER, self.demolisher_cost
                    else:
                        kind, unit_cost = self.SCOUT, self.scout_cost
                    count = int(mp // unit_cost)
                    for lane in self.counter_lanes:
                        if (game_state.attempt_spawn(kind, [lane],
                                                     count) or 0) > 0:
                            break
                self.gate_open = False
            elif sealed and not pressure and \
                    mp >= self.GATE_PREP_MP + reserve:
                for loc in self.gate:
                    if game_state.contains_stationary_unit(loc):
                        game_state.attempt_remove([loc])
                self.gate_open = True
        except Exception:
            pass  # a broken counter must still submit an (empty) turn
