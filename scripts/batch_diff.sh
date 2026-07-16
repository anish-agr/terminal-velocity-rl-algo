#!/bin/bash
# Batch fidelity validation: run tsim diff over every scraped replay, aggregate results.
# Output: replays/scraped/diff_results.tsv (file, turns_ok, turns, frames_ok, frames,
# restore_ok, restore, verdict) + a summary to stdout.
cd "$(dirname "$0")/.."
OUT=replays/scraped/diff_results.tsv
: > "$OUT"
total=0; pass=0; imperfect=0
for r in replays/scraped/*.replay; do
  line=$(sim/target/release/tsim.exe diff "$r" 0 2>/dev/null | tail -2 | head -1)
  # line like: == path: turns A/B ok, frames C/D ok, restore E/F ok
  nums=$(echo "$line" | grep -oE '[0-9]+/[0-9]+' | tr '/' ' ')
  set -- $nums
  ta=${1:-0}; tb=${2:-0}; fa=${3:-0}; fb=${4:-0}; ra=${5:-0}; rb=${6:-0}
  if [ "$ta" = "$tb" ] && [ "$ra" = "$rb" ] && [ "$tb" != "0" ]; then v=PASS; pass=$((pass+1));
  else v=DIVERGE; imperfect=$((imperfect+1)); fi
  echo -e "$(basename $r)\t$ta\t$tb\t$fa\t$fb\t$ra\t$rb\t$v" >> "$OUT"
  total=$((total+1))
done
echo "batch diff: $total replays, $pass PASS, $imperfect with divergences"
awk -F'\t' '{fo+=$4; ft+=$5} END {printf "frames exact: %d / %d (%.4f%%)\n", fo, ft, 100*fo/ft}' "$OUT"
echo "worst 10 by frame mismatches:"
awk -F'\t' '{print $5-$4, $0}' "$OUT" | sort -rn | head -10
