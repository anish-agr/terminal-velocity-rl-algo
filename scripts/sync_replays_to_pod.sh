#!/bin/bash
# Push the scraped replay corpus (not in git — too large / not source) to a training pod.
# Usage: scripts/sync_replays_to_pod.sh <pod-ip> <port> [remote-user]
# Run this AFTER train/setup_runpod.sh has cloned the repo on the pod, so
# ~/terminal-velocity/replays/scraped/ exists as the destination parent.
set -euo pipefail
POD_IP="${1:?usage: sync_replays_to_pod.sh <pod-ip> <port> [user]}"
PORT="${2:?usage: sync_replays_to_pod.sh <pod-ip> <port> [user]}"
USER="${3:-root}"
cd "$(dirname "$0")/.."

N=$(ls replays/scraped/*.replay 2>/dev/null | wc -l)
SIZE=$(du -sh replays/scraped 2>/dev/null | cut -f1)
echo "Syncing $N replays ($SIZE) to $USER@$POD_IP:~/terminal-velocity/replays/scraped/"

rsync -avz --progress -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new" \
  replays/scraped/ "$USER@$POD_IP:~/terminal-velocity/replays/scraped/"

echo "== verifying remote count =="
ssh -p "$PORT" "$USER@$POD_IP" "ls ~/terminal-velocity/replays/scraped/*.replay | wc -l"

echo "== running corpus gate on the pod =="
ssh -p "$PORT" "$USER@$POD_IP" "cd ~/terminal-velocity && python -m train.replays --check replays/scraped"
