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
    mismatch, log and fall back to the scripted plan for this turn
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

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gamelib  # noqa: E402

from fallback import FallbackBot  # noqa: E402

_SEARCH_BUDGET_S = 2.5
_WATCHDOG_S = 3.8
_STRUCT_KINDS = (0, 1, 2)
_MOBILE_KINDS = (3, 4, 5)


class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        self.mode = "fallback"

    # ------------------------------------------------------------------
    def on_game_start(self, config):
        self.config = config
        self.fallback = FallbackBot(config)
        self.our_log = []          # [turn] -> [(kind, x, y), ...] we attempted
        self.enemy_log = []        # [turn] -> reconstructed enemy commands
        self.turn_frames = []      # raw parsed turn-frame dicts
        self.enemy_spawns = []     # mobile spawn events (enemy), current turn
        self.flow = [0.0, 0.0, 0.0, 0.0]  # breach dealt/taken, dmg dealt/taken

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
                self.fallback.apply(game_state)
        except Exception as exc:
            gamelib.debug_write("TV: turn error {!r} -> fallback".format(exc))
            try:
                self.fallback.apply(game_state)
            except Exception:
                pass
        game_state.submit_turn()

    # ------------------------------------------------------------------
    def _search_turn(self, game_state, turn_state):
        from train.search import choose
        from train.tokens import decode_commands, encode_plan, ScratchSpec

        frame = json.loads(turn_state)
        self.turn_frames.append(frame)
        turn = int(frame["turnInfo"][1])

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
        self.enemy_spawns = []
        self.flow = [0.0, 0.0, 0.0, 0.0]

        mirror = self._rebuild_mirror()
        if mirror is None or not self._mirror_in_sync(mirror, frame):
            gamelib.debug_write("TV: mirror out of sync turn {}".format(turn))
            self.fallback.apply(game_state)
            self.our_log.append([])   # we know exactly what fallback staged is
            return                    # approximate; resync from server next turn

        # search in a worker; watchdog submits the fallback plan on a miss
        result = {}

        def work():
            try:
                result["plan"] = choose(
                    mirror, self.client, self.cfg, 0,
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
            self.fallback.apply(game_state)
            self.our_log.append([])
            return

        spec = ScratchSpec(self.costs, mirror.structures(),
                           mirror.stats(0)[1], mirror.stats(0)[2], False, 0)
        cmds = encode_plan(list(result["plan"]), spec())
        self._stage(game_state, cmds)
        self.our_log.append(cmds)

    # ------------------------------------------------------------------
    def _stage(self, game_state, cmds):
        info = self.config["unitInformation"]
        for (kind, x, y) in cmds:
            if kind in _STRUCT_KINDS or kind in _MOBILE_KINDS:
                game_state.attempt_spawn(info[kind]["shorthand"], [[x, y]])
            elif kind == 6:
                game_state.attempt_remove([[x, y]])
            elif kind == 7:
                game_state.attempt_upgrade([[x, y]])

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

    def _mirror_in_sync(self, mirror, frame):
        """Structure position multisets must match the server exactly."""
        try:
            server = set()
            for pid, key in ((0, "p1Units"), (1, "p2Units")):
                lists = frame.get(key, [])
                for kind in _STRUCT_KINDS:
                    for u in (lists[kind] if kind < len(lists) else []):
                        server.add((kind, pid, int(u[0]), int(u[1])))
            ours = {(s[0], s[1], s[2], s[3]) for s in mirror.structures()}
            return ours == server
        except Exception:
            return False


if __name__ == "__main__":
    AlgoStrategy().start()
