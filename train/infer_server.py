"""GPU inference server + queue client (ARCHITECTURE §5.5).

One server process owns every net; N actor processes talk to it through
multiprocessing queues. Requests carry a `model_id`, which is how league
snapshots and the BC anchor play through the same server: the server holds
{model_id: net} and loads snapshot weights on demand via control messages.

The server executes requests by wrapping each net in search.LocalNetClient —
the sampling/scoring code the GPU runs is byte-for-byte the code the
deployment driver runs locally. The train/deploy skew guarantee (§5) is
enforced structurally, not by discipline.

Batching: the loop drains up to `infer_batch_max` requests or waits
`infer_batch_wait_ms`; VALUE requests across actors are coalesced into one
forward pass (they dominate the load at K x M states per decision); sample /
score requests run per-request (their sequential decoder steps do not batch
across requests, and their torso cost is one state).

Wire formats (pickled tuples on mp.Queue):
    ("sample", actor, req, model, board, scalars, scratch_blob, k, tau,
               greedy_extra, mask_deploys_extra, head)
    ("score",  actor, req, model, board, scalars, scratch_blob, plans, head)
    ("value",  actor, req, model, boards, scalars)
    ("load_model",  model_id, weights_path)       control: add/replace a net
    ("reload",      model_id, weights_path)       control: hot-reload weights
    ("stop",)
scratch_blob = (structures, sp, mp, flip, player) — the server rebuilds
PlanScratch factories locally from these plus the game config.
Responses: (req, payload) on the actor's own response queue.
"""

from __future__ import annotations

import queue as queue_mod
import time
from typing import Dict, Optional

from .search import LocalNetClient, NetClient
from .tokens import Costs, ScratchSpec


# ---------------------------------------------------------------------------
# Client side (runs inside each actor process)
# ---------------------------------------------------------------------------

class QueueClient(NetClient):
    """NetClient over multiprocessing queues. One instance per (actor,
    model_id) pair; blocking round-trips (an actor decision is sequential)."""

    def __init__(self, request_q, response_q, actor_id: int, model_id: str,
                 timeout_s: float = 60.0):
        self.request_q = request_q
        self.response_q = response_q
        self.actor_id = actor_id
        self.model_id = model_id
        self.timeout_s = timeout_s
        self._req = 0

    def _next(self):
        self._req += 1
        return self._req

    def sample_plans(self, board, scalars, scratch_factory, k, tau, head,
                     greedy_extra=False, mask_deploys_extra=False):
        blob = scratch_factory.blob  # see ScratchSpec below
        req = self._next()
        self.request_q.put((
            "sample", self.actor_id, req, self.model_id, board, scalars, blob,
            k, tau, greedy_extra, mask_deploys_extra, head,
        ))
        return self._await(req)

    def score_plans(self, board, scalars, scratch_factory, plans, head):
        blob = scratch_factory.blob
        req = self._next()
        self.request_q.put((
            "score", self.actor_id, req, self.model_id, board, scalars, blob,
            list(plans), head,
        ))
        return self._await(req)

    def values(self, boards, scalars):
        req = self._next()
        self.request_q.put(
            ("value", self.actor_id, req, self.model_id, boards, scalars)
        )
        return self._await(req)

    def _await(self, want):
        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("inference server unresponsive")
            try:
                req, payload = self.response_q.get(timeout=remaining)
            except queue_mod.Empty:
                continue
            if req == want:
                if isinstance(payload, Exception):
                    raise payload
                return payload


# ---------------------------------------------------------------------------
# Server side
# ---------------------------------------------------------------------------

def serve(request_q, response_qs, game_config: dict, cfg: dict,
          init_weights: Optional[Dict[str, str]] = None, device: str = "cpu",
          max_requests: Optional[int] = None) -> None:
    """Server process main loop. response_qs: {actor_id: Queue}.

    init_weights: {model_id: checkpoint_path or ""} — "" means fresh random
    weights (used by tests and the very first bootstrap minutes).
    max_requests: exit after N data requests (tests only; None = run forever).
    """
    import torch

    from .model import TerminalNet

    costs = Costs(game_config)
    batch_max = int(cfg["actors"]["infer_batch_max"])
    wait_s = float(cfg["actors"]["infer_batch_wait_ms"]) / 1000.0

    clients: Dict[str, LocalNetClient] = {}

    def load_model(model_id: str, path: str) -> None:
        net = TerminalNet()
        if path:
            net.load_state_dict(torch.load(path, map_location="cpu"))
        clients[model_id] = LocalNetClient(net, device=device)

    for mid, path in (init_weights or {"current": ""}).items():
        load_model(mid, path)

    served = 0
    while True:
        # -- drain a batch window ------------------------------------------
        batch = []
        try:
            batch.append(request_q.get(timeout=1.0))
        except queue_mod.Empty:
            continue
        t0 = time.monotonic()
        while len(batch) < batch_max:
            remaining = wait_s - (time.monotonic() - t0)
            if remaining <= 0:
                break
            try:
                batch.append(request_q.get(timeout=remaining))
            except queue_mod.Empty:
                break

        # -- control messages first ------------------------------------------
        data_reqs = []
        stop = False
        for msg in batch:
            kind = msg[0]
            if kind == "stop":
                stop = True
            elif kind in ("load_model", "reload"):
                try:
                    load_model(msg[1], msg[2])
                except Exception:
                    pass  # keep serving with the previous weights
            else:
                data_reqs.append(msg)

        # -- coalesce VALUE requests per model into one forward each ---------
        import numpy as np

        by_model: Dict[str, list] = {}
        for msg in data_reqs:
            if msg[0] == "value":
                by_model.setdefault(msg[3], []).append(msg)
        for model_id, msgs in by_model.items():
            client = clients.get(model_id)
            if client is None:
                for m in msgs:
                    response_qs[m[1]].put((m[2], KeyError(model_id)))
                continue
            boards = np.concatenate([m[4] for m in msgs], axis=0)
            scalars = np.concatenate([m[5] for m in msgs], axis=0)
            try:
                v = client.values(boards, scalars)
                off = 0
                for m in msgs:
                    n = m[4].shape[0]
                    response_qs[m[1]].put((m[2], v[off:off + n]))
                    off += n
            except Exception as exc:
                for m in msgs:
                    response_qs[m[1]].put((m[2], exc))

        # -- sample / score, per request ---------------------------------------
        for msg in data_reqs:
            kind = msg[0]
            if kind == "value":
                continue
            actor, req, model_id = msg[1], msg[2], msg[3]
            client = clients.get(model_id)
            if client is None:
                response_qs[actor].put((req, KeyError(model_id)))
                continue
            try:
                if kind == "sample":
                    (_, _, _, _, board, scalars, blob, k, tau,
                     greedy_extra, mask_deploys_extra, head) = msg
                    factory = ScratchSpec(costs, *blob)
                    out = client.sample_plans(
                        board, scalars, factory, k, tau, head,
                        greedy_extra=greedy_extra,
                        mask_deploys_extra=mask_deploys_extra,
                    )
                elif kind == "score":
                    (_, _, _, _, board, scalars, blob, plans, head) = msg
                    factory = ScratchSpec(costs, *blob)
                    out = client.score_plans(board, scalars, factory, plans, head)
                else:
                    out = ValueError("unknown request {!r}".format(kind))
                response_qs[actor].put((req, out))
            except Exception as exc:
                response_qs[actor].put((req, exc))

        served += len(data_reqs)
        if stop or (max_requests is not None and served >= max_requests):
            return
