#!/usr/bin/env bash
# Retrain SN126 miner when benchmark releases change or live-chunk drift is detected.
set -euo pipefail

ROOT="/root/bittensor-mining/Poker44-subnet"
PYTHON="${ROOT}/miner_env/bin/python"
STATE_FILE="${ROOT}/models/benchmark_state.json"
LOG_FILE="${ROOT}/models/auto_retrain.log"
RETRAIN_INTERVAL_SECONDS="${RETRAIN_INTERVAL_SECONDS:-21600}"
MAX_FPR="${MAX_FPR:-0.02}"
MAX_POSITIVE_RATE="${MAX_POSITIVE_RATE:-0.05}"
MIN_SOURCE_DATE="${MIN_SOURCE_DATE:-2026-06-13}"
LIVE_AUGMENT="${LIVE_AUGMENT:-1}"
PSEUDO_MAX_BATCHES="${PSEUDO_MAX_BATCHES:-120}"

log() {
  printf '[%s] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*" | tee -a "${LOG_FILE}"
}

fetch_state_json() {
  "${PYTHON}" - <<'PY'
import json
import urllib.request

url = "https://api.poker44.net/api/v1/benchmark/releases"
with urllib.request.urlopen(url, timeout=120) as response:
    payload = json.load(response)["data"]
releases = payload.get("releases") or []
print(json.dumps({
    "release_version": payload.get("releaseVersion"),
    "schema_version": payload.get("schemaVersion"),
    "source_dates": sorted({release["sourceDate"] for release in releases}),
    "release_ids": sorted({release["releaseId"] for release in releases}),
    "sample_count": sum(int(release.get("chunkCount", 0)) for release in releases),
}, indent=2))
PY
}

should_retrain() {
  local new_state old_state
  new_state="$(mktemp)"
  old_state="$(mktemp)"
  fetch_state_json > "${new_state}"
  if [[ ! -f "${STATE_FILE}" ]]; then
    cp "${new_state}" "${STATE_FILE}"
    rm -f "${new_state}" "${old_state}"
    log "No prior benchmark state; seeding ${STATE_FILE}"
    return 1
  fi
  cp "${STATE_FILE}" "${old_state}"
  skip_retrain="$("${PYTHON}" "${ROOT}/scripts/deploy_lock.py" should-skip \
      --new-state "${new_state}" \
      --old-state "${old_state}" | "${PYTHON}" -c "import json,sys; print(json.load(sys.stdin).get('skip', False))")"
  if [[ "${skip_retrain}" == "True" ]]; then
    rm -f "${new_state}" "${old_state}"
    log "Deploy lock active; skipping retrain"
    return 1
  fi
  if cmp -s "${new_state}" "${old_state}"; then
    rm -f "${new_state}" "${old_state}"
    return 1
  fi
  log "Benchmark state changed:"
  diff -u "${old_state}" "${new_state}" | tee -a "${LOG_FILE}" || true
  rm -f "${old_state}"
  return 0
}

run_retrain() {
  log "Starting ${TRAIN_PROFILE:-v6} retrain (max_fpr=${MAX_FPR}, max_positive_rate=${MAX_POSITIVE_RATE}, min_source_date=${MIN_SOURCE_DATE})"
  cd "${ROOT}"
  "${PYTHON}" scripts/train_innovative_model.py \
    --profile "${TRAIN_PROFILE:-v6}" \
    --deploy \
    --max-fpr "${MAX_FPR}" \
    --max-positive-rate "${MAX_POSITIVE_RATE}" \
    --min-source-date "${MIN_SOURCE_DATE}" \
    --folds 3 \
    --n-jobs 1 \
    --pseudo-max-examples 400 \
    --pseudo-max-batches "${PSEUDO_MAX_BATCHES}" \
    ${LIVE_AUGMENT:+--live-augment} \
    2>&1 | tee -a "${LOG_FILE}"

  "${PYTHON}" scripts/generate_release_manifest.py | tee -a "${LOG_FILE}"
  if command -v pm2 >/dev/null 2>&1; then
    pm2 restart sn126-miner --update-env | tee -a "${LOG_FILE}" || true
  fi
  fetch_state_json > "${STATE_FILE}"
  log "Retrain complete; benchmark state refreshed"
}

if [[ "${1:-}" == "--once" ]]; then
  if should_retrain; then
    run_retrain
  else
    log "No retrain needed (--once)"
  fi
  exit 0
fi

log "Auto-retrain watcher started (interval=${RETRAIN_INTERVAL_SECONDS}s)"
while true; do
  if should_retrain; then
    run_retrain
  else
    log "No benchmark change detected"
  fi
  sleep "${RETRAIN_INTERVAL_SECONDS}"
done
