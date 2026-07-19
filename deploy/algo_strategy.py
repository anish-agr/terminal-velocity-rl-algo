"""Deployment driver (ARCHITECTURE §9.3): thin gamelib shim around the search.

Startup ladder (each rung degrades gracefully to the next):
  1. terminal_sim .so + weights.bin + numpy  -> full K x M search (anytime budget)
  2. weights.bin + numpy, no .so             -> currently also fallback (the
     search needs sim forks; a net-only greedy mode is a possible upgrade)
  3. anything missing or crashed             -> FallbackBot plays the match

Per turn under rung 1:
  - reconstruct the enemy's last-turn commands from observed action frames
    (mobile spawns) + turn-frame diffs (builds, upgrades, removal deaths), and
    replay BOTH sides' command logs into a fresh sim -> the mirror state
  - cross-check the mirror's structures against the server's turn frame; on
    mismatch, rebuild the mirror FROM the frame itself (_frame_ground_mirror)
    so the net keeps playing; only if that also fails, scripted plan
  - if the anti-rush detector is engaged (opponent is mass-scouting), stage
    the scripted AntiRushBot counter instead of running the search at all
  - run search.choose with the anytime budget in a worker thread; a watchdog
    submits the FallbackBot plan if the worker misses its deadline
  - stage the chosen commands through gamelib and append them to our log

The engine always presents OUR side as the bottom half, so this driver is
always player 0 in the mirror sim; the opponent is player 1.
"""

import json
import os
import sys
import threading

# Cap BLAS/OpenMP threads BEFORE numpy is ever imported (here or in npforward).
# The competition container is process/thread-restricted; uncapped OpenBLAS tries
# to spawn one thread per host core, hits RLIMIT_NPROC, and the import/first matmul
# raises -> the search worker dies every turn -> FallbackBot plays the whole match
# (looks fine in playground, which has more headroom). setdefault so an explicit
# env override still wins.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gamelib  # noqa: E402

from fallback import AntiRushBot, FallbackBot  # noqa: E402
try:
    from corner_hammer_bot import CornerHammerBot  # noqa: E402
except Exception:                     # never let the scripted layer's import
    CornerHammerBot = None            # take down the whole algo

# The shared ranked box runs ~2-4x slower per thread than a dev machine, so the
# search must self-limit early (budget) and the watchdog needs headroom under
# the engine's 5s per-turn cap. At 2.5/3.8 the k16/m8 floor round overran every
# turn and the driver submitted FallbackBot for the whole match.
_SEARCH_BUDGET_S = 1.6
_WATCHDOG_S = 4.5
_STRUCT_KINDS = (0, 1, 2)
_MOBILE_KINDS = (3, 4, 5)
# The net is far stronger than the scripted fallback (pod arena: 9-0 vs the
# whole panel, offense connecting, scout_rush +34 / static_maze +36), so the
# driver's job is to keep the net planning on a CORRECT board every turn.
# The log-replay mirror drifts when the opponent's submission order is
# ambiguous (combat tie-breaks shift kills by a frame); when that happens the
# driver no longer benches the net -- it rebuilds the mirror FROM THE SERVER
# FRAME itself (_frame_ground_mirror), which cannot drift because it is
# re-derived from observed truth each turn. Structure positions are exact by
# construction; only structure damage is approximated (full health), which is
# a far smaller error than handing the match to the scripted layer.


class _GameView:
    """Sim mirror that reports the REAL banks from the server frame.

    The frame-grounded mirror cannot reproduce SP/MP exactly (the sim API has
    no state injection), and a cap-level mirror bank made the search plan big
    waves that encode-clamping then shrank to 3-4 units -- the net dribbled
    undersized attacks into massed turrets all game (ladder 15343220: 110
    spawns at one cell, 5 dmg dealt). Plan generation and the net's scalar
    inputs read banks through stats()/scalar_features(), so overriding those
    two with frame truth makes every generated plan affordable FOR REAL.
    Forks are handed back unwrapped: they exist to evolve the sim forward,
    and the candidate plans are already bank-clamped by then."""

    def __init__(self, game, sp0, mp0, sp1, mp1):
        self._g = game
        self._b = (float(sp0), float(mp0), float(sp1), float(mp1))

    def stats(self, player):
        hp, sp, mp = self._g.stats(player)
        sp0, mp0, sp1, mp1 = self._b
        return (hp, sp0, mp0) if player == 0 else (hp, sp1, mp1)

    def scalar_features(self, player):
        sf = list(self._g.scalar_features(player))
        sp0, mp0, sp1, mp1 = self._b
        try:
            if player == 0:
                sf[1], sf[2], sf[4], sf[5] = sp0, mp0, sp1, mp1
            else:
                sf[1], sf[2], sf[4], sf[5] = sp1, mp1, sp0, mp0
        except Exception:
            pass
        return sf

    def __getattr__(self, name):
        return getattr(self._g, name)


class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        self.mode = "fallback"

    # ------------------------------------------------------------------
    def on_game_start(self, config):
        self.config = config
        self.fallback = FallbackBot(config)
        try:
            self.antirush = AntiRushBot(config)
        except Exception:
            self.antirush = None
        try:
            # ladder-proven full game plan (~1400-1500 standalone) -- every
            # net-degradation path lands here instead of the rush counter
            self.ch = CornerHammerBot(config) if CornerHammerBot else None
        except Exception:
            self.ch = None
        self.our_log = []          # [turn] -> [(kind, x, y), ...] we attempted
        self.enemy_log = []        # [turn] -> reconstructed enemy commands
        self.turn_frames = []      # raw parsed turn-frame dicts
        self.enemy_spawns = []     # mobile spawn events (enemy), current turn
        self.flow = [0.0, 0.0, 0.0, 0.0]  # breach dealt/taken, dmg dealt/taken
        self.breach_xs = []        # x of each breach WE took, current turn

        try:
            import numpy as np  # noqa: F401

            import terminal_sim
            from npforward import NumpyNet, NumpyNetClient
            from train.features import DeployHistory
            from train.tokens import Costs

            weights = os.path.join(_HERE, "weights.bin")
            self.sim_config_str = json.dumps(config)
            self.terminal_sim = terminal_sim
            self.client = NumpyNetClient(NumpyNet(weights))
            self.costs = Costs(config)
            self.hist_own = DeployHistory(config)
            self.hist_opp = DeployHistory(config)
            with open(os.path.join(_HERE, "deploy_config.json")) as fh:
                self.cfg = json.load(fh)
            self.prev_opp_plan = None
            self.mode = "search"
            gamelib.debug_write("TV: full search mode")
        except Exception as exc:
            gamelib.debug_write("TV: fallback mode ({!r})".format(exc))

    # ------------------------------------------------------------------
    def on_action_frame(self, turn_string):
        # corner_hammer's breach-heat / wave-composition tracking runs in
        # EVERY mode: its state must be warm if a degradation hands it the
        # game mid-match (and it is the whole game plan in fallback mode)
        if self.ch is not None:
            self.ch.on_action_frame(turn_string)
        if self.mode != "search":
            return
        try:
            state = json.loads(turn_string)
            for s in state.get("events", {}).get("spawn", []):
                if len(s) >= 4 and s[3] == 2 and s[1] in _MOBILE_KINDS:
                    self.enemy_spawns.append((int(s[1]), int(s[0][0]), int(s[0][1])))
            for b in state.get("events", {}).get("breach", []):
                if len(b) >= 5:
                    self.flow[0 if b[4] == 1 else 1] += float(b[1])
                    if b[4] != 1:
                        self.breach_xs.append(int(b[0][0]))
            for d in state.get("events", {}).get("damage", []):
                if len(d) >= 5 and d[2] in _STRUCT_KINDS:
                    self.flow[3 if d[4] == 1 else 2] += float(d[1])
        except Exception:
            pass

    # ------------------------------------------------------------------
    def on_turn(self, turn_state):
        game_state = gamelib.GameState(self.config, turn_state)
        game_state.suppress_warnings(True)
        try:
            if self.mode == "search":
                self._search_turn(game_state, turn_state)
            else:
                self._scripted(game_state)
        except Exception as exc:
            gamelib.debug_write("TV: turn error {!r} -> scripted".format(exc))
            try:
                self._scripted(game_state)
            except Exception:
                pass
        # exactly one log entry per turn frame, recording what gamelib ACTUALLY
        # staged (fallback turns included, and net of gamelib's own filtering).
        # Logging [] here instead would leave every future mirror rebuild
        # missing this turn's builds -> _mirror_in_sync fails -> permanent
        # fallback lock-in after a single watchdog miss.
        if self.mode == "search" and len(self.our_log) < len(self.turn_frames):
            self.our_log.append(self._staged_cmds(game_state))
        game_state.submit_turn()

    def _scripted(self, game_state):
        """Strongest scripted layer available. corner_hammer is a full
        ladder-proven game plan (~1400-1500 standalone); AntiRushBot is a
        rush counter; FallbackBot is the floor."""
        if self.ch is not None:
            self.ch.apply(game_state)
        elif self.antirush is not None:
            self.antirush.apply(game_state)
        else:
            self.fallback.apply(game_state)

    def _staged_cmds(self, game_state):
        """gamelib's per-turn stacks -> engine (kind, x, y) command tuples.
        unitInformation index IS the engine kind (0-5 units, 6 remove, 7
        upgrade), so the shorthand map covers every stack entry."""
        try:
            short2kind = {
                info["shorthand"]: k
                for k, info in enumerate(self.config["unitInformation"])
                if "shorthand" in info
            }
            out = []
            for (sh, x, y) in list(game_state._build_stack) + \
                    list(game_state._deploy_stack):
                kind = short2kind.get(sh)
                if kind is not None:
                    out.append((int(kind), int(x), int(y)))
            return out
        except Exception:
            return []

    # ------------------------------------------------------------------
    def _search_turn(self, game_state, turn_state):
        from train.search import choose
        from train.tokens import decode_commands, encode_plan, ScratchSpec

        frame = json.loads(turn_state)
        self.turn_frames.append(frame)
        turn = int(frame["turnInfo"][1])

        # keep corner_hammer's per-turn state warm even on net turns (launch
        # levels, screen arming, gate machine) -- idempotent, ~0 ms
        if self.ch is not None:
            self.ch.observe(game_state)

        # finalize LAST turn's enemy reconstruction + histories
        if turn > 0:
            enemy_cmds = self._reconstruct_enemy(len(self.turn_frames) - 2)
            self.enemy_log.append(enemy_cmds)
            self.hist_own.record_turn(
                [c for c in enemy_cmds if c[0] in _MOBILE_KINDS],
                self.flow[0], self.flow[1], self.flow[2], self.flow[3])
            own_deploys = [c for c in self.our_log[-1] if c[0] in _MOBILE_KINDS] \
                if self.our_log else []
            self.hist_opp.record_turn(
                own_deploys, self.flow[1], self.flow[0], self.flow[3], self.flow[2])
            self.prev_opp_plan = tuple(decode_commands(enemy_cmds, flip=True))
            if self.antirush is not None:
                try:
                    enemy_mp = float(game_state.get_resource(game_state.MP, 1))
                except Exception:
                    enemy_mp = 0.0
                self.antirush.observe(
                    sum(1 for s in self.enemy_spawns if s[0] == 3),
                    sum(1 for s in self.enemy_spawns if s[0] == 4),
                    sum(1 for s in self.enemy_spawns if s[0] == 5),
                    self.flow[1], enemy_mp, turn,
                    self.flow[0], list(self.breach_xs),
                    [s[1] for s in self.enemy_spawns])
        self.enemy_spawns = []
        self.flow = [0.0, 0.0, 0.0, 0.0]
        self.breach_xs = []

        # anti-rush override (§9.2): while engaged, the scripted counter plays
        # the turn — no mirror rebuild or search needed, so this path also
        # rescues games where the mirror has desynced into permanent fallback
        if self.antirush is not None and self.antirush.engaged:
            gamelib.debug_write("TV: anti-rush override turn {}".format(turn))
            self.antirush.apply(game_state)
            return   # on_turn logs the staged commands for this turn

        # pre-harden on a rush ALERT (a single flag, before full engagement):
        # stage defense-only builds now, then let the net play the rest of
        # the turn — gamelib drops whatever the net can no longer afford
        if self.antirush is not None and \
                getattr(self.antirush, "alert", False):
            try:
                self.antirush.preharden(game_state)
            except Exception:
                pass

        # mirror ladder: log-replay (exact history, real structure damage)
        # -> frame-grounded (exact positions/hp rebuilt from the server frame
        # itself, so a desync can no longer bench the net) -> scripted.
        mirror = self._rebuild_mirror()
        if mirror is not None and not self._mirror_in_sync(mirror, frame):
            mirror = None
        if mirror is None:
            mirror = self._frame_ground_mirror(frame)
            if mirror is not None:
                gamelib.debug_write(
                    "TV: frame-grounded mirror turn {}".format(turn))
        if mirror is None:
            gamelib.debug_write("TV: mirror out of sync turn {}".format(turn))
            # Both mirrors failed (should be rare now): degrade to the
            # STRONGEST scripted layer -- corner_hammer, the full ladder-
            # proven game plan (~1400-1500 standalone), with its adaptive
            # state kept warm by observe() every turn since game start.
            self._scripted(game_state)
            return   # on_turn logs the staged commands for this turn

        # the search must plan against the REAL banks (frame truth), not the
        # mirror's approximation -- see _GameView
        try:
            view = _GameView(
                mirror,
                float(game_state.get_resource(game_state.SP)),
                float(game_state.get_resource(game_state.MP)),
                float(game_state.get_resource(game_state.SP, 1)),
                float(game_state.get_resource(game_state.MP, 1)))
        except Exception:
            view = mirror

        # search in a worker; watchdog submits the fallback plan on a miss
        result = {}

        def work():
            try:
                result["plan"] = choose(
                    view, self.client, self.cfg, 0,
                    self.hist_own, self.hist_opp, self.config, self.costs,
                    prev_opp_plan=self.prev_opp_plan,
                    k=int(self.cfg["search"]["k_deploy"]),
                    m=int(self.cfg["search"]["m_deploy"]),
                    tau=float(self.cfg["search"]["tau_deploy"]),
                    budget_s=_SEARCH_BUDGET_S,
                )[0]
            except Exception as exc:  # noqa: BLE001
                result["error"] = repr(exc)

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        worker.join(timeout=_WATCHDOG_S)

        if "plan" not in result:
            gamelib.debug_write("TV: watchdog ({})".format(
                result.get("error", "deadline")))
            # a missed deadline is a timing problem, not a scripted-layer
            # bug: degrade to the strongest scripted turn available
            self._scripted(game_state)
            return   # on_turn logs the staged commands for this turn

        # encode against the REAL banks from the server frame, not the
        # mirror's (a frame-grounded mirror approximates SP/MP; the plan must
        # be clamped to what we can actually afford this turn)
        try:
            sp_real = float(game_state.get_resource(game_state.SP))
            mp_real = float(game_state.get_resource(game_state.MP))
        except Exception:
            sp_real = mirror.stats(0)[1]
            mp_real = mirror.stats(0)[2]
        spec = ScratchSpec(self.costs, mirror.structures(),
                           sp_real, mp_real, False, 0)
        cmds = encode_plan(list(result["plan"]), spec())
        self._stage(game_state, cmds)
        # on_turn records the staged stacks (what the server will actually
        # get) rather than `cmds` — gamelib may have filtered some attempts

    # ------------------------------------------------------------------
    def _stage(self, game_state, cmds):
        info = self.config["unitInformation"]
        attack_cell = "unset"   # lazily computed best working edge, once/turn
        for (kind, x, y) in cmds:
            if kind in _STRUCT_KINDS:
                game_state.attempt_spawn(info[kind]["shorthand"], [[x, y]])
            elif kind in _MOBILE_KINDS:
                loc = [x, y]
                # OFFENSE (scout/demolisher) that self-destructs in our own
                # half does nothing (ladder 15343256: 123 of the net's scouts
                # died at y=12, 0 dmg). Reroute it to a lane that actually
                # reaches the enemy. Interceptors are left alone -- dying in
                # our half IS their job (screening).
                if kind in (3, 4) and self._self_traps(game_state, loc):
                    if attack_cell == "unset":
                        attack_cell = self._best_attack_cell(game_state)
                    if attack_cell is not None:
                        loc = list(attack_cell)
                game_state.attempt_spawn(info[kind]["shorthand"], [loc])
            elif kind == 6:
                game_state.attempt_remove([[x, y]])
            elif kind == 7:
                game_state.attempt_upgrade([[x, y]])

    @staticmethod
    def _reaches_enemy(path):
        return bool(path) and len(path) >= 1 and path[-1][1] >= 14

    def _self_traps(self, game_state, loc):
        """True if a mobile unit spawned at loc self-destructs in our half
        (its path never reaches the enemy side)."""
        try:
            if game_state.contains_stationary_unit(loc):
                return False
            return not self._reaches_enemy(game_state.find_path_to_edge(loc))
        except Exception:
            return False

    def _best_attack_cell(self, game_state):
        """Least-defended bottom-edge deploy cell whose path reaches the enemy
        half; None if we are fully walled in (leave the net's choice as-is)."""
        try:
            gm = game_state.game_map
            cells = gm.get_edge_locations(gm.BOTTOM_LEFT) + \
                gm.get_edge_locations(gm.BOTTOM_RIGHT)
        except Exception:
            return None
        best, best_danger = None, None
        for c in cells:
            try:
                if game_state.contains_stationary_unit(c):
                    continue
                path = game_state.find_path_to_edge(c)
                if not self._reaches_enemy(path):
                    continue
                danger = sum(len(game_state.get_attackers(cell, 0) or ())
                             for cell in path)
            except Exception:
                continue
            if best_danger is None or danger < best_danger:
                best, best_danger = list(c), danger
        return best

    def _reconstruct_enemy(self, prev_idx):
        """Enemy commands for the turn between frames prev_idx and prev_idx+1:
        builds from tracked structure diffs, upgrades from p2Units[7] diffs,
        removal deaths from the newer turn frame, mobiles from spawn events."""
        cmds = list(self.enemy_spawns)  # mobiles observed live
        try:
            prev, cur = self.turn_frames[prev_idx], self.turn_frames[prev_idx + 1]

            def units(frame, idx):
                lists = frame.get("p2Units", [])
                return {(int(u[0]), int(u[1])) for u in
                        (lists[idx] if idx < len(lists) else [])}

            for kind in _STRUCT_KINDS:
                for (x, y) in sorted(units(cur, kind) - units(prev, kind)):
                    cmds.insert(0, (kind, x, y))
            for (x, y) in sorted(units(cur, 7) - units(prev, 7)):
                cmds.insert(0, (7, x, y))
            for d in cur.get("events", {}).get("death", []):
                if len(d) >= 5 and d[4] and d[3] == 2 and d[1] in _STRUCT_KINDS:
                    cmds.insert(0, (6, int(d[0][0]), int(d[0][1])))
        except Exception:
            pass
        return cmds

    def _rebuild_mirror(self):
        try:
            g = self.terminal_sim.Game(self.sim_config_str)
            for t in range(len(self.enemy_log)):
                ours = self.our_log[t] if t < len(self.our_log) else []
                g.play_turn(list(ours), list(self.enemy_log[t]))
            return g
        except Exception:
            return None

    def _frame_ground_mirror(self, frame):
        """Fresh sim snapped to the server's CURRENT frame (the desync cure).

        The log-replay mirror drifts because reconstructed opponent commands
        cannot recover submission order, so combat tie-breaks diverge. This
        builder never replays history: it re-derives the state from the frame
        itself, so it cannot drift. The sim API has no state injection, so the
        state is reproduced with catch-up turns:
          1. HP: every mobile unit breaches for 1 on this config, so scouts
             crossing the EMPTY board set both players' hp exactly (sides
             alternate turns so the waves never meet mid-board).
          2. Board: issue every structure the frame shows for both sides each
             turn until placed (the engine skips unaffordable commands, and
             cumulative income guarantees eventual affordability); upgrades
             the same way. No mobiles are in play, so nothing fights.
          3. Clock: pad empty turns so the sim turn matches the real turn and
             the search's lookahead sees the right income schedule.
        Structure positions and player hp are exact (verified before return);
        structure DAMAGE resets to full and MP/SP banks approximate -- far
        smaller errors than benching the net, and the encode-time ScratchSpec
        uses the REAL banks from gamelib anyway. Returns None on any failure.
        """
        try:
            turn = int(frame["turnInfo"][1])
            res = self.config.get("resources", {})
            start_hp = float(res.get("startingHP", 30.0))
            hp_tgt = [float(frame["p1Stats"][0]), float(frame["p2Stats"][0])]
            want = {0: set(), 1: set()}
            upg = {0: set(), 1: set()}
            for pid, key in ((0, "p1Units"), (1, "p2Units")):
                lists = frame.get(key, [])
                for kind in _STRUCT_KINDS:
                    for u in (lists[kind] if kind < len(lists) else []):
                        want[pid].add((kind, int(u[0]), int(u[1])))
                if len(lists) > 7:
                    upg[pid] = {(int(u[0]), int(u[1])) for u in lists[7]}
            info = self.config["unitInformation"]
            scout_cost = float(info[3].get("cost2", 1.0)) or 1.0

            g = self.terminal_sim.Game(self.sim_config_str)
            budget = turn + 24   # hard stop: never loop unbounded

            # -- 1. hp via breaches on the empty board ----------------------
            spawn = {0: (3, 13, 0), 1: (3, 14, 27)}
            used = 0
            while used < budget:
                # damage each side still has to deal (their scouts hit the
                # OTHER player's hp)
                d0 = int(round(g.stats(1)[0] - hp_tgt[1]))
                d1 = int(round(g.stats(0)[0] - hp_tgt[0]))
                if d0 <= 0 and d1 <= 0:
                    break
                side = 0 if d0 >= d1 else 1
                need = d0 if side == 0 else d1
                n = min(need, int(g.stats(side)[2] // scout_cost))
                cmds = [[], []]
                cmds[side] = [spawn[side]] * max(n, 0)
                g.play_turn(cmds[0], cmds[1])
                used += 1

            # -- 2. structures + upgrades, both sides at once ---------------
            while used < budget:
                placed = {0: set(), 1: set()}
                upped = {0: set(), 1: set()}
                for s in g.structures():
                    placed[s[1]].add((s[0], s[2], s[3]))
                    if s[5]:
                        upped[s[1]].add((s[2], s[3]))
                cmds = []
                done = True
                for p in (0, 1):
                    cs = [(k, x, y) for (k, x, y) in sorted(want[p] - placed[p])]
                    cells = {(x, y) for (_, x, y) in want[p]}
                    cs += [(7, x, y) for (x, y) in
                           sorted((upg[p] & cells) - upped[p])]
                    if cs:
                        done = False
                    cmds.append(cs)
                if done:
                    break
                g.play_turn(cmds[0], cmds[1])
                used += 1

            # -- 3. clock alignment ----------------------------------------
            while g.turn < turn and used < budget:
                g.play_turn([], [])
                used += 1

            # -- acceptance: exact positions, exact hp ---------------------
            ours = {(s[0], s[1], s[2], s[3]) for s in g.structures()}
            if ours != self._server_structs(frame):
                return None
            if abs(g.stats(0)[0] - hp_tgt[0]) > 0.5 or \
                    abs(g.stats(1)[0] - hp_tgt[1]) > 0.5:
                return None
            return g
        except Exception:
            return None

    def _server_structs(self, frame):
        """(kind, player, x, y) set from a server turn frame."""
        server = set()
        for pid, key in ((0, "p1Units"), (1, "p2Units")):
            lists = frame.get(key, [])
            for kind in _STRUCT_KINDS:
                for u in (lists[kind] if kind < len(lists) else []):
                    server.add((kind, pid, int(u[0]), int(u[1])))
        return server

    def _mirror_in_sync(self, mirror, frame):
        """Structure position multisets must match the server exactly. (A
        drift tolerance was tried on ladder and lost anyway -- the net paths
        against walls that are not there. The frame-grounded rebuild replaces
        a desynced mirror with an exact one instead of tolerating drift.)"""
        try:
            ours = {(s[0], s[1], s[2], s[3]) for s in mirror.structures()}
            return ours == self._server_structs(frame)
        except Exception:
            return False


if __name__ == "__main__":
    AlgoStrategy().start()
