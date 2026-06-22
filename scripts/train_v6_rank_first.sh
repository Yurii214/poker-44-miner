#!/usr/bin/env bash
# Train and deploy reference-dualbranch-v6-rank-first for UID 164.
set -euo pipefail

ROOT="/root/bittensor-mining/Poker44-subnet"
PYTHON="${ROOT}/miner_env/bin/python"
cd "${ROOT}"

MAX_FPR="${MAX_FPR:-0.02}"
MAX_POSITIVE_RATE="${MAX_POSITIVE_RATE:-0.05}"
MIN_SOURCE_DATE="${MIN_SOURCE_DATE:-2026-06-13}"
DEPLOY="${DEPLOY:-1}"

ARGS=(
  scripts/train_innovative_model.py
  --max-fpr "${MAX_FPR}"
  --max-positive-rate "${MAX_POSITIVE_RATE}"
  --min-source-date "${MIN_SOURCE_DATE}"
  --live-augment
  --pseudo-max-examples 400
)

if [[ "${DEPLOY}" == "1" ]]; then
  ARGS+=(--deploy)
fi

echo "Training v6 rank-first model..."
"${PYTHON}" "${ARGS[@]}"

echo "Regenerating model manifest..."
"${PYTHON}" scripts/generate_release_manifest.py

echo "Done. Next: bash scripts/publish_miner_repo.sh && pm2 restart sn126-miner --update-env"
