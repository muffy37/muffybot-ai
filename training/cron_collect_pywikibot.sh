#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/ubuntu/vandalismai"
LOG_DIR="$REPO_DIR/data/ml_runs/logs"
LOCK_FILE="/tmp/vandalismai_collect.lock"
LOG_FILE="$LOG_DIR/collect_pywikibot_cron.log"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

# Keep process singleton to avoid overlap.
exec 9>"$LOCK_FILE"
flock -n 9 || exit 0

{
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] START collect"
  PYTHONPATH="src:/home/ubuntu/pywikibot-scripts" \
    python3 training/generate_dataset_pywikibot.py \
      --family vikidia \
      --langs fr,en \
      --max-changes 2500 \
      --max-diffs 700 \
      --output-dir data/ml_runs/shards \
      --easy-negative-rate 0.20 \
      --min-context-score 4 \
      --source-prefix harvest
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] DONE collect"
} >>"$LOG_FILE" 2>&1
