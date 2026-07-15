"""Platform probe: plays a safe minimal game while logging the competition container's
environment to stderr (visible in the match log / replay debug output). Upload this, play
one match against your own algo on the platform, then read the logs.

Everything is wrapped so a probe failure can never crash the algo.
"""

import json
import os
import sys
import time

import gamelib


def _log(tag, value):
    gamelib.debug_write("PROBE|{}|{}".format(tag, value))


def run_probe():
    _log("python", sys.version.replace("\n", " "))
    _log("executable", sys.executable)
    _log("platform", sys.platform)
    try:
        import platform as _p

        _log("machine", _p.machine())
        _log("libc", str(_p.libc_ver()))
    except Exception as e:
        _log("platform_err", repr(e))
    _log("cpu_count", os.cpu_count())
    try:
        _log("sched_affinity", len(os.sched_getaffinity(0)))
    except Exception as e:
        _log("sched_affinity_err", repr(e))
    # cgroup limits (docker)
    for path in (
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            with open(path) as fh:
                _log("cgroup:" + path, fh.read().strip()[:80])
        except Exception:
            pass
    # numpy / scipy / onnx / torch availability + BLAS speed
    try:
        t0 = time.time()
        import numpy as np

        _log("numpy", np.__version__ + " import_ms=%d" % ((time.time() - t0) * 1000))
        a = np.random.rand(256, 256).astype(np.float32)
        t0 = time.time()
        for _ in range(20):
            a = a @ a
            a /= np.abs(a).max() + 1.0
        _log("numpy_matmul20_256_ms", int((time.time() - t0) * 1000))
    except Exception as e:
        _log("numpy_err", repr(e))
    for mod in ("scipy", "onnxruntime", "torch", "numba"):
        try:
            t0 = time.time()
            m = __import__(mod)
            _log(mod, getattr(m, "__version__", "?") + " import_ms=%d" % ((time.time() - t0) * 1000))
        except Exception as e:
            _log(mod + "_err", repr(e)[:120])
    # can we load a bundled shared library / execute a bundled binary?
    here = os.path.dirname(os.path.abspath(__file__))
    _log("algo_dir", here)
    try:
        _log("dir_listing", ",".join(sorted(os.listdir(here))[:30]))
    except Exception as e:
        _log("dir_err", repr(e))
    try:
        import ctypes

        _log("ctypes", "ok")
        try:
            ctypes.CDLL(os.path.join(here, "nonexistent_test.so"))
        except OSError as e:
            _log("cdll_expected_err", repr(e)[:120])
    except Exception as e:
        _log("ctypes_err", repr(e))
    try:
        import subprocess

        r = subprocess.run(["uname", "-a"], capture_output=True, text=True, timeout=2)
        _log("uname", r.stdout.strip()[:120])
    except Exception as e:
        _log("subprocess_err", repr(e)[:120])
    try:
        import shutil

        for tool in ("gcc", "cc", "python3"):
            _log("which_" + tool, shutil.which(tool))
    except Exception as e:
        _log("which_err", repr(e))


class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        self.probed = False

    def on_game_start(self, config):
        self.config = config
        global WALL, TURRET, SCOUT, INTERCEPTOR
        WALL = config["unitInformation"][0]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        try:
            t0 = time.time()
            run_probe()
            _log("probe_total_ms", int((time.time() - t0) * 1000))
        except Exception as e:
            _log("probe_fatal", repr(e))

    def on_turn(self, turn_state):
        gs = gamelib.GameState(self.config, turn_state)
        gs.suppress_warnings(True)
        # minimal sane play: basic funnel + interceptors, timing measurement each turn
        t0 = time.time()
        turrets = [[3, 12], [24, 12], [9, 10], [18, 10], [13, 10], [14, 10]]
        gs.attempt_spawn(TURRET, turrets)
        walls = [[x, 13] for x in (0, 1, 2, 3, 24, 25, 26, 27)]
        gs.attempt_spawn(WALL, walls)
        if gs.turn_number % 3 == 0:
            gs.attempt_spawn(INTERCEPTOR, [[6, 7], [21, 7]])
        else:
            gs.attempt_spawn(SCOUT, [13, 0], 1000)
        _log("turn_%d_ms" % gs.turn_number, int((time.time() - t0) * 1000))
        gs.submit_turn()


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
