"""Orchestrator (ARCHITECTURE §14): bootstrap / pilot / main / package.

    python -m train.run --phase bootstrap   Stage A: BC warm start from the
                                            scraped corpus + scripted seed
                                            games; saves bc_anchor + current
    python -m train.run --phase pilot       2 h small-scale full loop (K=8,M=4)
    python -m train.run --phase main        full-scale run until stopped
    python -m train.run --phase package     weights.bin + parity gate + dist/

bootstrap/pilot/main need the compiled sim (pod); package runs anywhere with
torch. Everything writes under --run-dir (default runs/) and resumes from the
latest checkpoint if one exists.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cfg(path: str = None) -> dict:
    with open(path or os.path.join(_REPO, "train", "config.yaml"),
              encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_game_config(path: str = None) -> dict:
    with open(path or os.path.join(_REPO, "game-configs.json"),
              encoding="utf-8") as fh:
        return json.load(fh)


def _require_sim():
    try:
        import terminal_sim
        return terminal_sim
    except ImportError as exc:
        raise SystemExit(
            "terminal_sim not importable ({}). Run train/setup_runpod.sh first "
            "— this phase needs the compiled simulator.".format(exc))


# picklable process helpers (mp spawn) -------------------------------------

def _make_game(config_str: str):
    import terminal_sim
    return terminal_sim.Game(config_str)


class GameFactory:
    def __init__(self, config_str: str):
        self.config_str = config_str

    def __call__(self):
        return _make_game(self.config_str)


class ClientFactory:
    def __init__(self, request_q, response_q, actor_id: int):
        self.request_q = request_q
        self.response_q = response_q
        self.actor_id = actor_id

    def __call__(self, model_id: str):
        from .infer_server import QueueClient
        return QueueClient(self.request_q, self.response_q, self.actor_id,
                           model_id)


# ---------------------------------------------------------------------------
# Stage A — bootstrap (§13)
# ---------------------------------------------------------------------------

def bc_positions_from_corpus(cfg: dict, config: dict, replay_dir: str,
                             limit: int = None):
    """Replay corpus -> learner-schema positions. Winner sides get their
    executed plan as the single candidate (pi=1) — the BC prior; loser sides
    get EMPTY candidates (value/aux/prediction signal only, §13)."""
    from .replays import build_bc_index, iter_positions, load_game

    bc_index = dict(build_bc_index(replay_dir, config,
                                   float(cfg["replays"]["bc_fingerprint_cap"])))
    names = [n for n in sorted(os.listdir(replay_dir)) if n.endswith(".replay")]
    if limit:
        names = names[:limit]
    out = []
    for name in names:
        path = os.path.join(replay_dir, name)
        rec = load_game(path, config)
        if rec is None:
            continue
        bc_side = bc_index.get(path)          # None -> no policy target
        game_positions = list(iter_positions(rec, config))
        by_key = {(p.side, p.turn): p for p in game_positions}
        for p in game_positions:
            opp = by_key.get((1 - p.side, p.turn))
            if opp is None:
                continue
            is_bc = (p.side == bc_side and p.z > 0)
            out.append({
                "board": p.board, "scalars": p.scalars,
                "structures": p.structures, "sp": p.sp, "mp": p.mp,
                "side": p.side, "turn": p.turn,
                "candidates": [p.plan] if is_bc else [],
                "pi": [1.0] if is_bc else [],
                "opp_board": opp.board, "opp_scalars": opp.scalars,
                "opp_structures": opp.structures, "opp_sp": opp.sp,
                "opp_mp": opp.mp, "opp_plan": opp.plan,
                "z": p.z, "aux": p.aux,
            })
    return out


def phase_bootstrap(cfg: dict, config: dict, run_dir: str,
                    replay_dir: str = None, bc_steps: int = 2000,
                    seed_games: int = None) -> None:
    from .actor import play_game
    from .learner import Learner
    from .tokens import Costs

    sim = _require_sim()
    os.makedirs(run_dir, exist_ok=True)
    replay_dir = replay_dir or os.path.join(_REPO,
                                            cfg["replays"]["scraped_dir"])
    learner = Learner(cfg, config,
                      device="cuda" if _cuda() else "cpu")

    print("== Stage A: corpus ingestion ==", flush=True)
    positions = bc_positions_from_corpus(cfg, config, replay_dir)
    learner.buffer.add_many(positions)
    print("corpus positions: {}".format(len(positions)), flush=True)

    print("== Stage A: scripted seed games (§6.3) ==", flush=True)
    from .scripted import SCRIPTED_BOTS
    names = sorted(SCRIPTED_BOTS.keys())
    rng = np.random.default_rng(0)
    costs = Costs(config)
    factory = GameFactory(json.dumps(config))
    n_seed = seed_games if seed_games is not None else \
        int(cfg["cold_start"]["scripted_seed_games"])
    for i in range(n_seed):
        a, b = rng.choice(len(names), size=2, replace=True)
        meta, pos = play_game(
            factory, {0: None, 1: None},
            {0: names[int(a)], 1: names[int(b)]}, (0, 1),
            cfg, config, costs, rng, record_scripted=True)
        learner.ingest(meta, pos)
        if (i + 1) % 500 == 0:
            print("seed games: {}/{}".format(i + 1, n_seed), flush=True)

    print("== Stage A: training {} steps ==".format(bc_steps), flush=True)
    for i in range(bc_steps):
        metrics = learner.train_step()
        if metrics and metrics["step"] % 100 == 0:
            print("bc", metrics, flush=True)

    anchor = os.path.join(run_dir, "bc_anchor.pt")
    current = os.path.join(run_dir, "weights_current.pt")
    learner.export_weights(anchor)
    learner.export_weights(current)
    learner.save_checkpoint(os.path.join(run_dir, "checkpoint.pt"))
    learner.league.has_anchor = True
    learner.league.save(os.path.join(run_dir, "league.json"))
    print("Stage A done -> {} + {}".format(anchor, current), flush=True)


# ---------------------------------------------------------------------------
# Pilot / main — the full loop (§5.5 topology)
# ---------------------------------------------------------------------------

def phase_loop(cfg: dict, config: dict, run_dir: str, hours: float,
               k: int, m: int) -> None:
    import multiprocessing as mp

    from .infer_server import serve
    from .learner import run_learner

    _require_sim()
    mp.set_start_method("spawn", force=True)
    n_actors = max(1, (os.cpu_count() or 2) - 2) * \
        int(cfg["actors"]["per_vcpu"]) // 2

    cfg = json.loads(json.dumps(cfg))          # deep copy; apply phase K/M
    cfg["search"]["k_train"], cfg["search"]["m_train"] = k, m

    request_q = mp.Queue()
    response_qs = {i: mp.Queue() for i in range(n_actors)}
    trajectory_q = mp.Queue(maxsize=256)

    current = os.path.join(run_dir, "weights_current.pt")
    anchor = os.path.join(run_dir, "bc_anchor.pt")
    init = {"current": current if os.path.exists(current) else ""}
    if os.path.exists(anchor):
        init["bc"] = anchor

    server = mp.Process(target=serve, args=(request_q, response_qs, config, cfg),
                        kwargs={"init_weights": init,
                                "device": "cuda" if _cuda() else "cpu"},
                        daemon=True)
    server.start()

    from .actor import run_actor
    factory = GameFactory(json.dumps(config))
    actors = []
    for i in range(n_actors):
        p = mp.Process(target=run_actor, args=(
            i, factory, ClientFactory(request_q, response_qs[i], i),
            trajectory_q, cfg, config,
            os.path.join(run_dir, "league.json"), 1000 + i), daemon=True)
        p.start()
        actors.append(p)
    print("loop: {} actors + server up; running {:.1f} h".format(
        n_actors, hours), flush=True)

    deadline = time.time() + hours * 3600.0
    try:
        run_learner(trajectory_q, request_q, cfg, config, run_dir,
                    device="cuda" if _cuda() else "cpu",
                    max_steps=None if hours >= 1e6 else
                    int(cfg["learning"]["total_steps"]))
    finally:
        request_q.put(("stop",))
        for p in actors:
            p.terminate()
        server.terminate()
    del deadline  # learner max_steps bounds the run; hours is advisory


# ---------------------------------------------------------------------------
# Package (§9.2)
# ---------------------------------------------------------------------------

def phase_package(cfg: dict, config: dict, run_dir: str,
                  out_dir: str = None) -> None:
    import torch

    from .export import export_checkpoint, parity_check
    from .model import TerminalNet

    out_dir = out_dir or os.path.join(_REPO, "dist", "python-algo")
    current = os.path.join(run_dir, "weights_current.pt")
    if not os.path.exists(current):
        raise SystemExit("no {} — train first".format(current))

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    weights_bin = os.path.join(out_dir, "weights.bin")
    export_checkpoint(current, weights_bin)
    net = TerminalNet()
    net.load_state_dict(torch.load(current, map_location="cpu"))
    worst = parity_check(net, weights_bin)
    print("parity gate: {:.2e} < 1e-4 OK".format(worst), flush=True)

    # driver + inference + fallback, flat in the algo dir
    for name in ("algo_strategy.py", "fallback.py", "npforward.py"):
        shutil.copy(os.path.join(_REPO, "deploy", name),
                    os.path.join(out_dir, name))
    # the train modules the driver imports, as a real subpackage
    os.makedirs(os.path.join(out_dir, "train"))
    for name in ("__init__.py", "tokens.py", "features.py", "search.py",
                 "export.py"):
        shutil.copy(os.path.join(_REPO, "train", name),
                    os.path.join(out_dir, "train", name))
    shutil.copytree(os.path.join(_REPO, "python-algo", "gamelib"),
                    os.path.join(out_dir, "gamelib"))
    for name in ("run.sh", "run.ps1", "algo.json"):
        src = os.path.join(_REPO, "python-algo", name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, name))
    with open(os.path.join(out_dir, "deploy_config.json"), "w") as fh:
        json.dump(cfg, fh)
    for so in glob.glob(os.path.join(_REPO, "sim", "target", "wheels", "*.so")) + \
            glob.glob(os.path.join(_REPO, "sim", "*.so")):
        shutil.copy(so, out_dir)

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _dn, fn in os.walk(out_dir) for f in fn)
    print("dist/python-algo: {:.1f} MB unpacked".format(total / 1e6), flush=True)
    if total > int(cfg["deployment"]["max_folder_mb"]) * 1e6:
        raise SystemExit("folder exceeds the 50 MB limit")
    print("package OK -> {}".format(out_dir), flush=True)


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["bootstrap", "pilot", "main", "package"])
    ap.add_argument("--run-dir", default=os.path.join(_REPO, "runs", "r1"))
    ap.add_argument("--replay-dir", default=None)
    ap.add_argument("--bc-steps", type=int, default=2000)
    ap.add_argument("--seed-games", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg()
    config = load_game_config()
    if args.phase == "bootstrap":
        phase_bootstrap(cfg, config, args.run_dir, args.replay_dir,
                        args.bc_steps, args.seed_games)
    elif args.phase == "pilot":
        phase_loop(cfg, config, args.run_dir, hours=2.0, k=8, m=4)
    elif args.phase == "main":
        phase_loop(cfg, config, args.run_dir, hours=1e9,
                   k=int(cfg["search"]["k_train"]),
                   m=int(cfg["search"]["m_train"]))
    else:
        phase_package(cfg, config, args.run_dir)


if __name__ == "__main__":
    main()
