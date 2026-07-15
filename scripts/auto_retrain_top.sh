#!/usr/bin/env bash
# Daily cron wrapper: retrain the rank-first top-miner model on the fresh
# benchmark release and deploy only if it does not regress. Single-instance
# via flock; all output appended to models/auto_retrain_top.log.
export PATH="/usr/local/bin:/usr/bin:/bin"

ROOT="/root/bittensor-mining/Poker44-subnet"
LOG="${ROOT}/models/auto_retrain_top.log"
LOCK="/tmp/poker44_auto_retrain_top.lock"

cd "${ROOT}" || exit 1

# Only act in the post-release window (00:00-00:59 UTC) unless forced. This makes
# pm2's immediate start-on-register a no-op; only the scheduled cron-restart runs.
if [[ "${FORCE:-0}" != "1" && "$(date -u +%H)" != "00" ]]; then
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') skip: outside 00:xx UTC release window (set FORCE=1 to override)" >>"${LOG}"
  exit 0
fi

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') skip: previous run still active" >>"${LOG}"
  exit 0
fi

{
  echo "====================================================================="
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') auto_retrain_top START"
  "${ROOT}/miner_env/bin/python" -W ignore "${ROOT}/scripts/train_live_native.py" --deploy
  rc=$?
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') auto_retrain_top END (exit ${rc})"
} >>"${LOG}" 2>&1
