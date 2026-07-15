#!/usr/bin/env python
"""Arena: run our algo against the sparring panel and report results.

The panel answers one question after every change: "did this make us worse
against anything?" Ladder feedback is slow and luck-dependent; the panel is
fast and fixed. Because the panel bots are deterministic and the game is
deterministic given both algos, ONE match per pairing is meaningful -- and
margins matter as much as results: a win that shrinks from 30-0 to 30-27 is
a regression the W/L column won't show.

Usage:
    python scripts/arena.py                          # challenger vs whole panel
    python scripts/arena.py --algo python-algo       # explicit challenger
    python scripts/arena.py --only scout_rush        # subset of the panel
    python scripts/arena.py --mirror algos/v3        # regression vs a frozen version
    python scripts/arena.py --check-determinism      # verify the yardstick holds still

Challenger is always player 1 in every match and every report.
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_utils import load_replay  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
REPLAY_DIR = os.path.join(REPO, "replays")
IS_WINDOWS = sys.platform.startswith("win")
RUN_FILE = "run.ps1" if IS_WINDOWS else "run.sh"


def run_file(algo_dir):
    path = os.path.join(REPO, algo_dir, RUN_FILE)
    if not os.path.exists(path):
        raise SystemExit("no {} in {}".format(RUN_FILE, algo_dir))
    return path


def play(challenger, opponent):
    """Run one engine match; return (replay_path, stderr, elapsed).

    Replay attribution: snapshot the replays dir before the match and take
    the file that appears after. Matches therefore run sequentially -- do not
    parallelize this without changing attribution.
    """
    before = set(os.listdir(REPLAY_DIR)) if os.path.isdir(REPLAY_DIR) else set()

    started = time.time()
    proc = subprocess.Popen(
        'java -jar engine.jar work "{}" "{}"'.format(run_file(challenger), run_file(opponent)),
        shell=True,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _out, err = proc.communicate()
    elapsed = time.time() - started

    after = set(os.listdir(REPLAY_DIR)) if os.path.isdir(REPLAY_DIR) else set()
    new = [f for f in after - before if f.endswith(".replay")]
    replay = os.path.join(REPLAY_DIR, new[0]) if new else None
    return replay, err.decode(errors="replace"), elapsed


def score(replay_path, stderr, opponent_name, elapsed):
    if replay_path is None:
        return {"opponent": opponent_name, "error": "no replay produced"}
    result = load_replay(replay_path).final_result()
    return {
        "opponent": opponent_name,
        "p1": result["p1_health"],
        "p2": result["p2_health"],
        "margin": result["p1_health"] - result["p2_health"],
        "turns": result["turns"],
        "elapsed": elapsed,
        "crashed": ("Traceback" in stderr),
        "replay": replay_path,
    }


def print_table(rows):
    print(
        "{:<18} {:>6} {:>9} {:>8} {:>6} {:>6}".format(
            "opponent", "result", "health", "margin", "turns", "secs"
        )
    )
    print("-" * 60)
    wins = losses = 0
    for r in rows:
        if "error" in r:
            print("{:<18} {:>6}   {}".format(r["opponent"], "ERR", r["error"]))
            continue
        verdict = "WIN" if r["margin"] > 0 else ("LOSS" if r["margin"] < 0 else "DRAW")
        wins += verdict == "WIN"
        losses += verdict == "LOSS"
        flag = "   <-- CRASH DETECTED" if r["crashed"] else ""
        print(
            "{:<18} {:>6} {:>9} {:>+8.0f} {:>6} {:>6.0f}{}".format(
                r["opponent"],
                verdict,
                "{:.0f}-{:.0f}".format(r["p1"], r["p2"]),
                r["margin"],
                r["turns"],
                r["elapsed"],
                flag,
            )
        )
    print("\n{} wins, {} losses across {} matches".format(wins, losses, len(rows)))


def panel_bots():
    root = os.path.join(REPO, "sparring")
    return sorted(
        name
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name)) and not name.startswith("_")
    )


def check_determinism():
    """Play the same panel pairing twice; the canonical digests must match.

    Uses two panel bots (not the challenger) so the check validates the
    yardstick itself. Digests ignore timing fields -- raw replay bytes always
    differ between runs.
    """
    a, b = "sparring/demolisher_line", "sparring/scout_rush"
    print("determinism check: {} vs {} twice ...".format(a, b))
    digests = []
    for i in (1, 2):
        replay, err, elapsed = play(a, b)
        if replay is None:
            raise SystemExit("run {} produced no replay:\n{}".format(i, err))
        digests.append(load_replay(replay).canonical_digest())
        print("  run {}: {:.0f}s  digest {}...".format(i, elapsed, digests[-1][:16]))
    if digests[0] == digests[1]:
        print("PASS: identical game content. The yardstick holds still.")
    else:
        print("FAIL: same pairing produced different games -- a panel bot is nondeterministic.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="python-algo", help="challenger dir (default python-algo)")
    parser.add_argument("--only", nargs="*", default=None, help="run only these panel bots")
    parser.add_argument("--mirror", default=None, help="also play challenger vs this frozen version dir")
    parser.add_argument("--check-determinism", action="store_true")
    args = parser.parse_args()

    if args.check_determinism:
        check_determinism()
        return

    bots = args.only if args.only else panel_bots()
    opponents = [("sparring/" + b if not b.startswith("sparring") else b) for b in bots]
    if args.mirror:
        opponents.append(args.mirror)

    print("challenger: {}  (always player 1)\n".format(args.algo))
    rows = []
    for opp in opponents:
        replay, err, elapsed = play(args.algo, opp)
        rows.append(score(replay, err, os.path.basename(opp), elapsed))
    print_table(rows)


if __name__ == "__main__":
    main()
