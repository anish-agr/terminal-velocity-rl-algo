"""CornerHammerBot -- the standalone corner_hammer algo adapted to the
deploy stack's scripted-layer interface (FallbackBot/AntiRushBot pattern).

This is bots/corner_hammer/algo_strategy.py (branch shane/corner-hammer-algo,
v9.1) with the AlgoCore shell removed: the driver owns the turn loop and
submit; this class only stages commands via apply(game_state). Keep the two
files in sync when the standalone bot iterates.

Why it exists: the ladder-proven scripted layer. The standalone bot holds
~1400-1500 on the ranked ladder across five audit-fix iterations, while
AntiRushBot (the previous degradation target) is a rush-counter, not a full
game plan. Every net-degradation path (watchdog miss, double mirror failure,
turn error, import-failure fallback mode) now lands here instead.

Integration contract:
  * __init__(config)            -- build once at game start
  * on_action_frame(turn_str)   -- feed EVERY action frame (any mode): breach
                                   heat + enemy wave composition accumulate
  * observe(game_state)         -- call once per turn (idempotent) even when
                                   the net plays: launch-level learning and
                                   screen arming must track the whole game
  * apply(game_state)           -- stage this turn's commands (never raises,
                                   never submits)

State notes for mid-game takeover: the defense wishlist rebuilds toward the
corner_hammer layout from whatever board the net left (attempt_* calls are
SP-throttled and skip occupied cells); the gate state machine self-corrects
in one turn. Pure python + gamelib game_state calls only -- works in the
no-numpy fallback environment too.
"""

import json


# ---------------------------------------------------------------------------
# layout (engine always presents our side as the bottom half -- no flip)
# ---------------------------------------------------------------------------
CORNER_WALLS = [[0, 13], [27, 13]]
FLANK_TURRETS = [[2, 13], [3, 13], [24, 13], [25, 13], [1, 12], [26, 12]]
FIRST_UPGRADES = [[2, 13], [25, 13], [1, 12], [26, 12]]
GATE_WALLS = [[13, 13], [14, 13]]     # center funnel gap -- NORMALLY CLOSED
RING_TURRETS = [[12, 11], [15, 11], [11, 12], [16, 12]]
ROW_TURRETS = [[x, 13] for x in (1, 26, 4, 23, 5, 22, 6, 21, 7, 20,
                                 8, 19, 9, 18, 10, 17, 11, 16, 12, 15)]
BACK_TURRETS = [[2, 12], [25, 12], [5, 12], [22, 12], [8, 12], [19, 12]]
SUPPORTS = [[12, 2], [15, 2], [11, 3], [16, 3],
            [12, 4], [15, 4], [11, 5], [16, 5]]
LATE_UPGRADES = [[x, 13] for x in (4, 6, 8, 10, 12, 15, 17, 19, 21, 23)]
DIAG_TURRETS = [[4, 11], [23, 11]]
DEEP_TURRETS = [[6, 9], [21, 9]]
HOT_LEFT = [[4, 12], [3, 12], [1, 12], [6, 12], [4, 11]]
HOT_RIGHT = [[23, 12], [24, 12], [26, 12], [21, 12], [23, 11]]
DIAG_WALLS_L = [[2, 11], [3, 10]]
DIAG_WALLS_R = [[25, 11], [24, 10]]
SCREEN_L = [[2, 11], [3, 10], [4, 9], [5, 8], [6, 7], [7, 6]]
SCREEN_R = [[25, 11], [24, 10], [23, 9], [22, 8], [21, 7], [20, 6]]

WAVE_FRACTION = 0.875
STALL_ESCALATE = 1.12
BANK_CAP_FRAC = 0.95
WAVE_SEEN_INCOMES = 1.5
PREDICT_FRAC = 0.97
THREAT_BANK_FRAC = 0.70
REARM_TURNS = 3
EARLY_TURNS = 12
EARLY_BANK_FRAC = 0.45
MEGA_INCOMES = 3.0
STALL_WAVES = 2
CLOSEOUT_TURN = 80
CLOSEOUT_LEAD = 8
# A dense BASE-turret front line (the ladder archetype that farmed the net:
# 15344900/30/40 all built 10+ turrets across their front row by mid-game)
# shreds pure scout waves just like an upgraded one. Lead with demolishers
# (outrange turrets, clear the line) and send scouts BEHIND them to breach.
DEMO_FRONT_TURRETS = 7
DEMO_LEAD_FRAC = 0.6


class CornerHammerBot:
    def __init__(self, config):
        self.config = config
        info = config["unitInformation"]
        self.WALL = info[0]["shorthand"]
        self.SUPPORT = info[1]["shorthand"]
        self.TURRET = info[2]["shorthand"]
        self.SCOUT = info[3]["shorthand"]
        self.DEMOLISHER = info[4]["shorthand"]
        self.INTERCEPTOR = info[5]["shorthand"]
        self.scout_cost = float(info[3].get("cost2", 1.0))
        self.demo_cost = float(info[4].get("cost2", 2.0))
        self.intc_cost = float(info[5].get("cost2", 1.0))
        self.t_dmg = float(info[2].get("attackDamageWalker", 5.0))
        self.t_dmg_up = float(info[2].get("upgrade", {})
                              .get("attackDamageWalker", self.t_dmg))
        self.wall_hp = float(info[0].get("startHealth", 40.0)) or 40.0
        res = config.get("resources", {})
        self.mp_base = float(res.get("bitsPerRound", 5.0))
        self.mp_growth = float(res.get("bitGrowthRate", 1.0))
        self.mp_interval = int(res.get("turnIntervalForBitSchedule", 10)) or 10
        self.mp_decay = float(res.get("bitDecayPerRound", 0.25)) or 0.25
        # cross-turn adaptive state
        self.heat = {"L": 0.0, "R": 0.0}
        self.prev_enemy_mp = 0.0
        self.enemy_scout_mp = 0.0
        self.enemy_demo_mp = 0.0
        self.launch_mps = []
        self.screen_armed = True
        self.screen_turn = None
        self.rearm_used = False
        self.demo_threat = False
        self.mega_threat = False
        self.enemy_spawn_xs = []
        self.side_hint_left = None
        self.stalled_waves = 0
        self.demo_mode = False
        self.wave_idx = 0
        self.last_wave_turn = None
        self.enemy_hp_at_wave = None
        self.gate_state = "closed"
        self.postponed = 0
        self.screen_ema = 0.0
        self._observed_turn = None

    # ------------------------------------------------------------------
    def on_action_frame(self, turn_string):
        """Breach heat (where WE are being hit) + enemy wave composition."""
        try:
            state = json.loads(turn_string)
            events = state.get("events", {})
            for b in events.get("breach", []):
                if len(b) >= 5 and b[4] == 2:          # they breached us
                    x = int(b[0][0])
                    if x < 9:
                        self.heat["L"] += 1.0
                    elif x > 18:
                        self.heat["R"] += 1.0
            for s in events.get("spawn", []):
                if len(s) >= 4 and s[3] == 2:
                    if s[1] == 3:
                        self.enemy_scout_mp += self.scout_cost
                        self.enemy_spawn_xs.append(int(s[0][0]))
                    elif s[1] == 4:
                        self.enemy_demo_mp += self.demo_cost
                        self.enemy_spawn_xs.append(int(s[0][0]))
        except Exception:
            pass

    # ------------------------------------------------------------------
    def observe(self, gs):
        """Settle last turn's enemy observations. Idempotent per turn; the
        driver calls this on NET turns too so launch levels, screen arming
        and heat stay warm for a mid-game takeover."""
        try:
            self._observe(gs)
        except Exception:
            pass

    def _observe(self, gs):
        turn = gs.turn_number
        if self._observed_turn == turn:
            return
        self._observed_turn = turn
        income = self.mp_base + self.mp_growth * (turn // self.mp_interval)
        wave_mp = self.enemy_scout_mp + self.enemy_demo_mp
        if wave_mp >= WAVE_SEEN_INCOMES * income:
            # a real launch: their bank last turn IS their commit level
            self.launch_mps.append(self.prev_enemy_mp)
            self.demo_threat = self.enemy_demo_mp > self.enemy_scout_mp
            self.screen_armed = True
            self.rearm_used = False
            if wave_mp >= MEGA_INCOMES * income:
                self.mega_threat = True
            if self.enemy_spawn_xs:
                mean_x = sum(self.enemy_spawn_xs) / len(self.enemy_spawn_xs)
                self.side_hint_left = mean_x > 13.5
        self.enemy_scout_mp = self.enemy_demo_mp = 0.0
        self.enemy_spawn_xs = []
        for k in self.heat:
            self.heat[k] *= 0.5
        self.prev_enemy_mp = gs.get_resource(gs.MP, 1)

    # ------------------------------------------------------------------
    def apply(self, gs):
        """Stage this turn's commands. Never raises, never submits."""
        try:
            self._observe(gs)
        except Exception:
            pass
        try:
            self._act(gs)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _act(self, gs):
        turn = gs.turn_number
        income = self.mp_base + self.mp_growth * (turn // self.mp_interval)
        fixed_point = income / self.mp_decay

        # ---- did our last wave actually score? ---------------------------
        if self.last_wave_turn is not None and turn > self.last_wave_turn:
            if gs.enemy_health >= self.enemy_hp_at_wave:
                self.stalled_waves += 1
            else:
                self.stalled_waves = 0
                self.demo_mode = False
            self.last_wave_turn = None
        if self.stalled_waves >= STALL_WAVES and not self.demo_mode:
            self.demo_mode = self._their_front_upgraded_heavy(gs)

        closeout = (turn >= CLOSEOUT_TURN and
                    gs.my_health - gs.enemy_health >= CLOSEOUT_LEAD)
        gate_open_turn = self.gate_state == "opening"   # marked last turn

        if turn == 0:
            gs.attempt_spawn(self.INTERCEPTOR, [[4, 9]], 1)
            gs.attempt_spawn(self.INTERCEPTOR, [[23, 9]], 1)

        # ---- defense wishlist, priority order (gamelib truncates at SP) --
        gs.attempt_spawn(self.WALL, CORNER_WALLS)
        gs.attempt_upgrade(CORNER_WALLS)
        gs.attempt_spawn(self.TURRET, FLANK_TURRETS)
        if not gate_open_turn:
            gs.attempt_spawn(self.WALL, GATE_WALLS)
        # seal the deep edge diagonals from the start, not just on mega
        # threat: every close game bled there (arena: shielded_push breached
        # (25,11)/(24,10)/(23,9) t3-13, flood breached (8,5)/(7,6)) -- rushes
        # hug the diagonal and slip under the y13/12 line
        gs.attempt_spawn(self.WALL, DIAG_WALLS_L + DIAG_WALLS_R)
        gs.attempt_spawn(self.TURRET, DIAG_TURRETS)
        gs.attempt_spawn(self.TURRET, DEEP_TURRETS)
        gs.attempt_upgrade(FIRST_UPGRADES)
        gs.attempt_spawn(self.SUPPORT, SUPPORTS[:2])
        gs.attempt_upgrade(SUPPORTS[:2])
        if self.heat["L"] > self.heat["R"] and self.heat["L"] >= 2.0:
            gs.attempt_spawn(self.TURRET, HOT_LEFT)
            gs.attempt_upgrade(HOT_LEFT + FLANK_TURRETS[:2])
        elif self.heat["R"] > self.heat["L"] and self.heat["R"] >= 2.0:
            gs.attempt_spawn(self.TURRET, HOT_RIGHT)
            gs.attempt_upgrade(HOT_RIGHT + FLANK_TURRETS[2:4])
        if self.mega_threat:
            gs.attempt_spawn(self.WALL, DIAG_WALLS_L + DIAG_WALLS_R)
            gs.attempt_upgrade(DIAG_WALLS_L + DIAG_WALLS_R)
        gs.attempt_spawn(self.SUPPORT, SUPPORTS[:4])
        gs.attempt_upgrade(SUPPORTS[:4])
        gs.attempt_spawn(self.TURRET, RING_TURRETS)
        gs.attempt_spawn(self.TURRET, ROW_TURRETS)
        gs.attempt_upgrade(FLANK_TURRETS)
        gs.attempt_spawn(self.SUPPORT, SUPPORTS)
        gs.attempt_upgrade(SUPPORTS)
        gs.attempt_upgrade(LATE_UPGRADES)
        gs.attempt_spawn(self.TURRET, BACK_TURRETS)

        # ---- offense: closed-gate wave cycle -----------------------------
        mp = gs.get_resource(gs.MP)
        eff_income = max(0.6 * income, income - self.screen_ema)
        esc = STALL_ESCALATE ** min(self.stalled_waves, 2)
        threshold = (min(WAVE_FRACTION * esc, BANK_CAP_FRAC) *
                     eff_income / self.mp_decay)
        enemy_mp = gs.get_resource(gs.MP, 1)
        launch_level = self._enemy_launch_level(fixed_point)
        enemy_imminent = enemy_mp >= launch_level
        fired = False
        if not closeout:
            if gate_open_turn:
                self.gate_state = "closed"
                if mp >= 0.95 * threshold:
                    self._fire_wave(gs, gs.get_resource(gs.MP), turn)
                    fired = True
                    self.postponed = 0
            elif (1.0 - self.mp_decay) * mp + income >= threshold:
                first_wave_window = (not self.launch_mps and turn <= 14 and
                                     enemy_mp >= 0.55 * fixed_point)
                if first_wave_window:
                    pass                              # stay sealed, keep banking
                elif enemy_imminent and self.postponed < 2:
                    self.postponed += 1               # don't open into their wave
                elif all(gs.contains_stationary_unit(c) for c in GATE_WALLS):
                    gs.attempt_remove(GATE_WALLS)     # open for NEXT turn only
                    self.gate_state = "opening"
                elif mp >= threshold:
                    self._fire_wave(gs, mp, turn)     # gate already open/dead
                    fired = True
        elif gate_open_turn:
            self.gate_state = "closed"                # close-out: stay sealed

        spent = 0
        if not fired and self.gate_state != "opening":
            spent = self._maybe_screen(gs, turn, income, enemy_mp,
                                       launch_level, threshold, closeout)
        self.screen_ema = 0.9 * self.screen_ema + 0.1 * spent * self.intc_cost

    # ------------------------------------------------------------------
    def _enemy_launch_level(self, fixed_point):
        if self.launch_mps:
            return PREDICT_FRAC * max(self.launch_mps[-3:])
        return THREAT_BANK_FRAC * fixed_point

    def _maybe_screen(self, gs, turn, income, enemy_mp, launch_level,
                      threshold, closeout):
        if not self.launch_mps and turn <= EARLY_TURNS:
            fixed_point = income / self.mp_decay
            rusher = (enemy_mp >= EARLY_BANK_FRAC * fixed_point and
                      self._enemy_struct_count(gs) < 8)
            first_wave = enemy_mp >= 0.60 * fixed_point
            if (rusher or first_wave) and \
                    (self.screen_turn is None or
                     turn - self.screen_turn >= 2):
                spent = (self._screen_side(gs, SCREEN_R, 2) +
                         self._screen_side(gs, SCREEN_L, 2))
                if spent:
                    self.screen_turn = turn
                return spent
            return 0
        if (not self.screen_armed and not self.rearm_used and
                self.screen_turn is not None and
                turn - self.screen_turn >= REARM_TURNS and
                enemy_mp >= launch_level):
            self.screen_armed = True               # they hovered; re-arm once
            self.rearm_used = True
        if not (self.screen_armed and enemy_mp >= launch_level):
            return 0
        mega = self.mega_threat and enemy_mp >= MEGA_INCOMES * income
        if not closeout and not mega and \
                gs.get_resource(gs.MP) >= 0.7 * threshold:
            return 0
        if abs(self.heat["R"] - self.heat["L"]) >= 1.0:
            hot_r = self.heat["R"] > self.heat["L"]
        elif self.side_hint_left is not None:
            hot_r = not self.side_hint_left
        else:
            hot_r = True
        hot, cold = (SCREEN_R, SCREEN_L) if hot_r else (SCREEN_L, SCREEN_R)
        if closeout:
            n_hot, n_cold = 4, 2
        elif self.demo_threat and enemy_mp >= 2.5 * income:
            n_hot, n_cold = 4, 2
        elif mega:
            n_hot, n_cold = 4, 1
        elif self.demo_threat:
            n_hot, n_cold = 3, 2
        else:
            n_hot, n_cold = 2, 1
        spent = 0
        spent += self._screen_side(gs, hot, n_hot)
        spent += self._screen_side(gs, cold, n_cold)
        if spent:
            self.screen_armed = False
            self.screen_turn = turn
        return spent

    def _enemy_struct_count(self, gs):
        n = 0
        try:
            for y in range(14, 28):
                for x in range(28):
                    if not gs.game_map.in_arena_bounds([x, y]):
                        continue
                    for u in gs.game_map[x, y] or []:
                        if u.stationary and u.player_index == 1:
                            n += 1
        except Exception:
            return 99                  # unknown -> assume builder, no screen
        return n

    def _screen_side(self, gs, cells, n):
        spent = 0
        for spot in cells:
            if spent >= n or gs.get_resource(gs.MP) < self.intc_cost:
                break
            if gs.contains_stationary_unit(spot):
                continue
            spent += gs.attempt_spawn(self.INTERCEPTOR, [spot], 1)
        return spent

    # ------------------------------------------------------------------
    def _fire_wave(self, gs, mp, turn):
        self.wave_idx += 1
        heavy, light = self._pick_lanes(gs)
        dense_front = self._front_turrets(gs) >= DEMO_FRONT_TURRETS
        demo_wave = (self.demo_mode or self.wave_idx % 3 == 0 or dense_front or
                     self._their_front_upgraded_heavy(gs))
        if demo_wave:
            if dense_front:
                # demo-led punch: demolishers outrange the line and clear it,
                # scouts follow the SAME lane and pour through the hole. A
                # pure demo wave kills structures but rarely breaches; a pure
                # scout wave dies to the line. The mix does both.
                demo_mp = DEMO_LEAD_FRAC * mp
                n_d = int(demo_mp // self.demo_cost)
                if n_d >= 2:
                    gs.attempt_spawn(self.DEMOLISHER, [heavy], n_d)
                    n_s = int((mp - n_d * self.demo_cost) // self.scout_cost)
                    if n_s > 0:
                        gs.attempt_spawn(self.SCOUT, [heavy], n_s)
                else:
                    demo_wave = False
            else:
                n = int(mp // self.demo_cost)
                if n >= 3:
                    n_h = max(1, (2 * n) // 3)
                    gs.attempt_spawn(self.DEMOLISHER, [heavy], n_h)
                    gs.attempt_spawn(self.DEMOLISHER, [light], n - n_h)
                else:
                    demo_wave = False
        if not demo_wave:
            n = int(mp // self.scout_cost)
            n_h = max(1, (7 * n) // 10)
            gs.attempt_spawn(self.SCOUT, [heavy], n_h)
            if n - n_h > 0:
                gs.attempt_spawn(self.SCOUT, [light], n - n_h)
        self.last_wave_turn = turn
        self.enemy_hp_at_wave = gs.enemy_health

    # ------------------------------------------------------------------
    def _pick_lanes(self, gs):
        cands = [[13, 0], [14, 0], [5, 8], [22, 8]]
        scored = []
        try:
            for c in cands:
                d = self._path_damage(gs, c)
                if d is not None:
                    scored.append((d, c))
        except Exception:
            pass
        if len(scored) >= 2:
            scored.sort(key=lambda t: t[0])
            return scored[0][1], scored[1][1]
        left = self._their_left_is_weaker(gs)
        return ([13, 0], [14, 0]) if left else ([14, 0], [13, 0])

    def _path_damage(self, gs, spawn):
        path = gs.find_path_to_edge(spawn)
        if not path:
            return None
        dmg = 0.0
        for cell in path:
            if cell[1] < 14:
                continue
            for u in gs.get_attackers(cell, 0):
                dmg += self.t_dmg_up if u.upgraded else self.t_dmg
        edge = gs.game_map.get_edge_locations(gs.get_target_edge(spawn))
        if list(path[-1]) not in [list(c) for c in edge]:
            dmg += 60.0            # self-destruct path: heavy penalty
        return dmg

    # ------------------------------------------------------------------
    def _front_turrets(self, gs):
        """How many turrets (any level) the enemy has on their front rows."""
        n = 0
        try:
            for y in (14, 15, 16):
                for x in range(28):
                    if not gs.game_map.in_arena_bounds([x, y]):
                        continue
                    for u in gs.game_map[x, y] or []:
                        if (u.stationary and u.player_index == 1 and
                                u.unit_type == self.TURRET):
                            n += 1
        except Exception:
            return 0
        return n

    def _their_front_upgraded_heavy(self, gs):
        up = base = 0
        try:
            for y in (14, 15, 16):
                for x in range(28):
                    if not gs.game_map.in_arena_bounds([x, y]):
                        continue
                    for u in gs.game_map[x, y] or []:
                        if (u.stationary and u.player_index == 1 and
                                u.unit_type == self.TURRET):
                            if u.upgraded:
                                up += 1
                            else:
                                base += 1
        except Exception:
            return False
        return up > base

    # ------------------------------------------------------------------
    def _their_left_is_weaker(self, gs):
        a = b = 0.0
        try:
            for x, y, right_zone in self._zone_cells():
                if not gs.game_map.in_arena_bounds([x, y]):
                    continue
                for u in gs.game_map[x, y] or []:
                    if not u.stationary or u.player_index != 1:
                        continue
                    if u.unit_type == self.TURRET:
                        val = self.t_dmg_up if u.upgraded else self.t_dmg
                    elif u.unit_type == self.WALL:
                        val = u.health / self.wall_hp
                    else:
                        continue
                    if right_zone:
                        a += val         # our top-right == their left corner
                    else:
                        b += val
        except Exception:
            pass
        return a <= b

    @staticmethod
    def _zone_cells():
        for y in range(14, 21):
            for x in range(21, 28):
                yield x, y, True
            for x in range(0, 7):
                yield x, y, False
