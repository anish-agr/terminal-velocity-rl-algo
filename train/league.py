"""League opponent sampling (ARCHITECTURE §6.4): PFSP snapshot pool + mix.

Every self-play game draws its opponent controller:
    35%  current theta (mirror self-play)
    40%  snapshot pool, PFSP-weighted f(w) = w * (1 - w) over each snapshot's
         rolling win-rate w against current (prioritizes near-peers)
    15%  scripted archetypes (rush / funnel / demolisher_line / turtle / torture)
    10%  frozen BC-anchor policy

Pool capped at 20 snapshots; a snapshot whose rolling win-rate has sat above
the eviction threshold for 2+ hours is "solved" and evicted first (it carries
no learning signal). Mass for empty categories (no snapshots yet, no anchor)
redistributes to mirror self-play, so the sampler is total from minute zero.

Pure logic — no torch, no processes. The clock is injectable for tests, and
the whole state serializes to JSON for resumability.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .scripted import SCRIPTED_BOTS


class Snapshot:
    def __init__(self, snap_id: str, path: str, created: float, window: int):
        self.id = snap_id
        self.path = path
        self.created = created
        self.results: deque = deque(maxlen=window)
        # wall-clock moment the rolling win-rate first exceeded the eviction
        # threshold, None while it is still contested
        self.solved_since: Optional[float] = None

    def winrate(self) -> float:
        """Current-theta's win-rate vs this snapshot. Optimistic 0.5 prior
        while unplayed so fresh snapshots get sampled."""
        if not self.results:
            return 0.5
        return float(sum(self.results)) / len(self.results)

    def to_json(self) -> dict:
        return {
            "id": self.id, "path": self.path, "created": self.created,
            "results": list(self.results), "solved_since": self.solved_since,
        }

    @classmethod
    def from_json(cls, d: dict, window: int) -> "Snapshot":
        s = cls(d["id"], d["path"], d["created"], window)
        s.results.extend(d["results"])
        s.solved_since = d["solved_since"]
        return s


class League:
    def __init__(self, cfg: dict, clock: Callable[[], float] = time.time):
        lc = cfg["league"]
        self.p_current = float(lc["p_current"])
        self.p_snapshot = float(lc["p_snapshot"])
        self.p_scripted = float(lc["p_scripted"])
        self.p_anchor = float(lc["p_bc_anchor"])
        self.pool_max = int(lc["snapshot_pool_max"])
        self.evict_winrate = float(lc["evict_winrate"])
        self.evict_seconds = float(lc["evict_hours"]) * 3600.0
        self.window = int(lc["pfsp_winrate_window"])
        self.clock = clock

        self.snapshots: List[Snapshot] = []
        self.has_anchor = False
        self._counter = 0
        self.scripted_names = sorted(SCRIPTED_BOTS.keys())

    # -- pool management ------------------------------------------------------

    def add_snapshot(self, path: str) -> str:
        self._counter += 1
        snap_id = "snap{:04d}".format(self._counter)
        if len(self.snapshots) >= self.pool_max:
            self._evict()
        self.snapshots.append(Snapshot(snap_id, path, self.clock(), self.window))
        return snap_id

    def _evict(self) -> None:
        """Drop the lowest-information member: solved-for-2h+ first (oldest
        solved wins the exit), else the highest-winrate snapshot."""
        now = self.clock()
        solved = [
            s for s in self.snapshots
            if s.solved_since is not None and now - s.solved_since >= self.evict_seconds
        ]
        victim = min(solved, key=lambda s: s.solved_since) if solved else \
            max(self.snapshots, key=lambda s: s.winrate())
        self.snapshots.remove(victim)

    def report_result(self, snap_id: str, current_won: bool) -> None:
        for s in self.snapshots:
            if s.id == snap_id:
                s.results.append(1.0 if current_won else 0.0)
                if s.winrate() > self.evict_winrate:
                    if s.solved_since is None:
                        s.solved_since = self.clock()
                else:
                    s.solved_since = None
                return

    # -- sampling ---------------------------------------------------------------

    def pfsp_weights(self) -> np.ndarray:
        """f(w) = w(1-w) + eps over the pool (eps keeps solved/unsolved alive)."""
        w = np.array([s.winrate() for s in self.snapshots], dtype=np.float64)
        f = w * (1.0 - w) + 1e-3
        return f / f.sum()

    def sample_opponent(self, rng: np.random.Generator) -> Tuple[str, str]:
        """-> (kind, detail): ("current", ""), ("snapshot", snap_id),
        ("scripted", bot_name), or ("anchor", "")."""
        p_snap = self.p_snapshot if self.snapshots else 0.0
        p_anchor = self.p_anchor if self.has_anchor else 0.0
        p_scripted = self.p_scripted
        p_current = 1.0 - p_snap - p_anchor - p_scripted  # absorbs missing mass

        r = rng.random()
        if r < p_current:
            return "current", ""
        r -= p_current
        if r < p_snap:
            idx = int(rng.choice(len(self.snapshots), p=self.pfsp_weights()))
            return "snapshot", self.snapshots[idx].id
        r -= p_snap
        if r < p_scripted:
            return "scripted", self.scripted_names[
                int(rng.integers(len(self.scripted_names)))
            ]
        return "anchor", ""

    def snapshot_path(self, snap_id: str) -> Optional[str]:
        for s in self.snapshots:
            if s.id == snap_id:
                return s.path
        return None

    # -- persistence -------------------------------------------------------------

    def save(self, path: str) -> None:
        state = {
            "counter": self._counter,
            "has_anchor": self.has_anchor,
            "snapshots": [s.to_json() for s in self.snapshots],
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(state, fh)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        with open(path) as fh:
            state = json.load(fh)
        self._counter = state["counter"]
        self.has_anchor = state["has_anchor"]
        self.snapshots = [
            Snapshot.from_json(d, self.window) for d in state["snapshots"]
        ]
