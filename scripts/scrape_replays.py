"""Harvest competition replays from the public API.

Usage:
  python scripts/scrape_replays.py sample <id> [<id> ...]     # inspect specific IDs
  python scripts/scrape_replays.py scan <lo> <hi> [workers]   # scan range, keep matching

Keeps only replays whose embedded config matches OUR game-configs.json on gameplay
fields (unit stats + resources; icon/display fields ignored). Output: replays/scraped/
<id>.replay plus a manifest line per ID in replays/scraped/manifest.tsv
(id, status, turns, winner, frames). Resumable: already-manifested IDs are skipped.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "replays", "scraped")
API = "https://terminal.c1games.com/api/game/replayexpanded/{}"

GAMEPLAY_UNIT_KEYS = [
    "cost1", "cost2", "startHealth", "attackDamageWalker", "attackDamageTower",
    "attackRange", "shieldPerUnit", "shieldRange", "shieldBonusPerY", "speed",
    "selfDestructDamageWalker", "selfDestructDamageTower", "selfDestructRange",
    "selfDestructStepsRequired", "playerBreachDamage", "metalForBreach",
    "refundPercentage", "turnsRequiredToRemove", "unitCategory",
]

# NOTE: coresForPlayerDamage intentionally excluded — deprecated (superseded by per-unit
# metalForBreach) and the server config omits it while local files carry it.
RESOURCE_KEYS = [
    "startingHP", "startingCores", "startingBits", "coresPerRound", "bitsPerRound",
    "bitGrowthRate", "turnIntervalForBitSchedule", "bitDecayPerRound", "maxBits",
]

def _num(v):
    # engines serialize absent numeric fields as missing OR explicit 0.0 — normalize
    return float(v) if isinstance(v, (int, float)) else 0.0

def gameplay_fingerprint(cfg):
    out = []
    for u in cfg.get("unitInformation", [])[:6]:
        base = tuple(_num(u.get(k)) for k in GAMEPLAY_UNIT_KEYS)
        upstats = u.get("upgrade") or {}
        # upgrade semantics: missing values inherit base, so normalize to EFFECTIVE stats
        upg = tuple(
            _num(upstats.get(k)) if upstats.get(k) is not None else _num(u.get(k))
            for k in GAMEPLAY_UNIT_KEYS
        )
        out.append((base, upg, bool(u.get("upgrade"))))
    res = cfg.get("resources", {})
    out.append(tuple(_num(res.get(k)) for k in RESOURCE_KEYS))
    return tuple(out)

def load_ours():
    with open(os.path.join(ROOT, "game-configs.json"), encoding="utf-8") as fh:
        return gameplay_fingerprint(json.load(fh))

OURS = load_ours()

def fetch(mid, timeout=30):
    req = urllib.request.Request(API.format(mid), headers={"User-Agent": "terminal-research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")

def classify(text):
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return "empty", None, None
    try:
        cfg = json.loads(lines[0])
    except json.JSONDecodeError:
        return "badjson", None, None
    if "unitInformation" not in cfg:
        return "noconfig", None, None
    same = gameplay_fingerprint(cfg) == OURS
    turns = winner = None
    try:
        last = json.loads(lines[-1])
        es = last.get("endStats", {})
        turns, winner = es.get("turns"), es.get("winner")
    except json.JSONDecodeError:
        pass
    return ("MATCH" if same else "otherconfig"), turns, winner

def handle(mid):
    try:
        text = fetch(mid)
    except urllib.error.HTTPError as e:
        return mid, f"http{e.code}", None, None, 0
    except Exception as e:
        return mid, "err:" + type(e).__name__, None, None, 0
    status, turns, winner = classify(text)
    nframes = text.count('"turnInfo"')
    if status == "MATCH":
        with open(os.path.join(OUT, f"{mid}.replay"), "w", encoding="utf-8") as fh:
            fh.write(text)
    return mid, status, turns, winner, nframes

def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = os.path.join(OUT, "manifest.tsv")
    done = set()
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as fh:
            done = {int(l.split("\t")[0]) for l in fh if l.strip()}
    mode = sys.argv[1]
    if mode == "sample":
        ids = [int(x) for x in sys.argv[2:]]
    else:
        lo, hi = int(sys.argv[2]), int(sys.argv[3])
        ids = [i for i in range(lo, hi + 1) if i not in done]
    workers = int(sys.argv[4]) if mode == "scan" and len(sys.argv) > 4 else 6
    t0 = time.time()
    kept = 0
    with open(manifest, "a", encoding="utf-8") as mf:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for mid, status, turns, winner, nframes in ex.map(handle, ids):
                mf.write(f"{mid}\t{status}\t{turns}\t{winner}\t{nframes}\n")
                mf.flush()
                if status == "MATCH":
                    kept += 1
                if mode == "sample":
                    print(f"{mid}: {status} turns={turns} winner={winner} frames={nframes}")
    print(f"done: {len(ids)} ids, kept {kept}, {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
