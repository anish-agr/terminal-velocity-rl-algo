"""Shared parser for Terminal engine .replay files.

This is the one place in the repo that knows the replay format. The arena
harness (result scoring), the simulator validator (frame diffing), and any
opponent-behaviour analysis should all import from here rather than parsing
JSON themselves, so a format discovery only ever has to be fixed once.

File format
-----------
A .replay is newline-delimited JSON:

  line 1        : the full game config (has key "debug")
  other lines   : frames, each with "turnInfo": [state_type, turn, action_frame, overall_frame]
                    state_type 0 = start-of-turn (deploy phase) snapshot
                    state_type 1 = one action-phase frame
                    state_type 2 = end-of-game frame (carries "endStats")

Frame fields
------------
  pXStats  : [health, structure_points, mobile_points, ms_taken]
  pXUnits  : 8 lists indexed by unit-type index (0 wall, 1 support, 2 turret,
             3 scout, 4 demolisher, 5 interceptor, 6 remove, 7 upgrade),
             each entry [x, y, health, unit_id_string]
  events   : dict of event lists. Field layouts VERIFIED against real engine
             output -- do not guess these; spawn is the odd one out:

    spawn        [[x,y], unit_type_idx, unit_id, player]                (4)
    breach       [[x,y], damage, unit_type_idx, unit_id, player]        (5)
    damage       [[x,y], damage, unit_type_idx, unit_id, player]        (5)
    death        [[x,y], unit_type_idx, unit_id, player, was_removed]   (5)
    attack       [[sx,sy], [tx,ty], damage, unit_type_idx,
                  source_id, target_id, player]                         (7)
    move         [[fx,fy], [tx,ty], desired_next_delta?, unit_type_idx,
                  unit_id, player]                                      (6)
    shield / selfDestruct / melee : kept raw (formats not yet verified)

  player ids in events: 1 = player 1, 2 = player 2. This is NOT gamelib's
  0/1 player_index convention.

Everything here is pure stdlib.
"""

from __future__ import annotations

import hashlib
import json
from collections import namedtuple


# Unit-type indices in pXUnits and in event unit_type_idx fields.
WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, REMOVE, UPGRADE = range(8)

STRUCTURE_IDXS = (WALL, SUPPORT, TURRET)
MOBILE_IDXS = (SCOUT, DEMOLISHER, INTERCEPTOR)


Spawn = namedtuple("Spawn", "x y unit_type unit_id player")
Breach = namedtuple("Breach", "x y damage unit_type unit_id player")
Damage = namedtuple("Damage", "x y damage unit_type unit_id player")
Death = namedtuple("Death", "x y unit_type unit_id player was_removed")
Attack = namedtuple(
    "Attack", "sx sy tx ty damage unit_type source_id target_id player"
)

Unit = namedtuple("Unit", "x y health unit_id")

PlayerStats = namedtuple("PlayerStats", "health structure_points mobile_points ms")


class Frame:
    """One parsed frame (one JSON line) of a replay."""

    def __init__(self, raw):
        self.raw = raw
        info = raw.get("turnInfo", [-1, -1, -1, -1])
        self.state_type = int(info[0])   # 0 deploy, 1 action, 2 end
        self.turn = int(info[1])
        self.action_frame = int(info[2]) if len(info) > 2 else -1
        self.overall_frame = int(info[3]) if len(info) > 3 else -1

    # -- stats ---------------------------------------------------------

    def stats(self, player):
        """PlayerStats for player 1 or 2."""
        key = "p{}Stats".format(player)
        s = self.raw[key]
        return PlayerStats(float(s[0]), float(s[1]), float(s[2]), s[3] if len(s) > 3 else None)

    # -- units ---------------------------------------------------------

    def units(self, player, unit_type=None):
        """Units on the board for player 1 or 2.

        Returns {unit_type_idx: [Unit, ...]} or, if unit_type is given,
        just that list. Skips the remove/upgrade pseudo-unit slots unless
        asked for explicitly.
        """
        key = "p{}Units".format(player)
        groups = self.raw.get(key, [])

        def parse(idx):
            out = []
            if idx < len(groups):
                for entry in groups[idx]:
                    out.append(Unit(int(entry[0]), int(entry[1]), float(entry[2]), str(entry[3])))
            return out

        if unit_type is not None:
            return parse(unit_type)
        return {idx: parse(idx) for idx in STRUCTURE_IDXS + MOBILE_IDXS}

    # -- events --------------------------------------------------------

    def _events(self, name):
        return self.raw.get("events", {}).get(name, [])

    def spawns(self):
        out = []
        for e in self._events("spawn"):
            if len(e) >= 4:
                out.append(Spawn(int(e[0][0]), int(e[0][1]), int(e[1]), str(e[2]), int(e[3])))
        return out

    def breaches(self):
        out = []
        for e in self._events("breach"):
            if len(e) >= 5:
                out.append(Breach(int(e[0][0]), int(e[0][1]), float(e[1]), int(e[2]), str(e[3]), int(e[4])))
        return out

    def damages(self):
        out = []
        for e in self._events("damage"):
            if len(e) >= 5:
                out.append(Damage(int(e[0][0]), int(e[0][1]), float(e[1]), int(e[2]), str(e[3]), int(e[4])))
        return out

    def deaths(self):
        out = []
        for e in self._events("death"):
            if len(e) >= 5:
                out.append(Death(int(e[0][0]), int(e[0][1]), int(e[1]), str(e[2]), int(e[3]), bool(e[4])))
        return out

    def attacks(self):
        out = []
        for e in self._events("attack"):
            if len(e) >= 7:
                out.append(
                    Attack(
                        int(e[0][0]), int(e[0][1]), int(e[1][0]), int(e[1][1]),
                        float(e[2]), int(e[3]), str(e[4]), str(e[5]), int(e[6]),
                    )
                )
        return out


class Replay:
    def __init__(self, config, frames):
        self.config = config
        self.frames = frames

    # -- structure -------------------------------------------------------

    def deploy_frames(self):
        """The state_type-0 snapshot at the start of each turn."""
        return [f for f in self.frames if f.state_type == 0]

    def turns(self):
        """{turn_number: [frames of that turn, in order]}."""
        by_turn = {}
        for f in self.frames:
            by_turn.setdefault(f.turn, []).append(f)
        return by_turn

    def end_frame(self):
        for f in reversed(self.frames):
            if f.state_type == 2:
                return f
        return None

    # -- results ---------------------------------------------------------

    def end_stats(self):
        end = self.end_frame()
        return end.raw.get("endStats", {}) if end else {}

    def final_result(self):
        """{winner (1|2|0 for draw), p1_health, p2_health, turns}.

        Health is read from the last frame's stats. endStats' "winner" is
        also reported when present (the engine names one even on ties).
        """
        end = self.end_frame() or self.frames[-1]
        p1 = end.stats(1).health
        p2 = end.stats(2).health
        if p1 > p2:
            winner = 1
        elif p2 > p1:
            winner = 2
        else:
            winner = 0
        return {
            "winner": winner,
            "engine_winner": self.end_stats().get("winner"),
            "p1_health": p1,
            "p2_health": p2,
            "turns": end.turn,
        }

    # -- determinism -----------------------------------------------------

    def canonical_digest(self):
        """Hash of the game-relevant content only.

        Two runs of the same deterministic pairing produce byte-DIFFERENT
        replay files (bot wall-clock times are embedded), so byte comparison
        cannot verify determinism. This digest covers unit positions/health,
        player health/resources, and event streams -- and deliberately
        excludes every timing field.
        """
        h = hashlib.sha256()
        for f in self.frames:
            core = {
                "t": [f.state_type, f.turn, f.action_frame],
                "s1": list(f.stats(1))[:3],  # health, SP, MP -- not ms
                "s2": list(f.stats(2))[:3],
                "u1": f.raw.get("p1Units"),
                "u2": f.raw.get("p2Units"),
                "e": f.raw.get("events"),
            }
            h.update(json.dumps(core, sort_keys=True).encode())
        return h.hexdigest()


def load_replay(path):
    """Parse a .replay file into a Replay."""
    config = None
    frames = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except ValueError:
                continue
            if "turnInfo" in data:
                frames.append(Frame(data))
            elif "debug" in data and config is None:
                config = data
    return Replay(config, frames)
