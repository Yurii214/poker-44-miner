#!/usr/bin/env bash
# Train v12: absolute regime + live replay cap + tighter dual-FPR for R1 recall.
set -euo pipefail

ROOT="/root/bittensor-mining/Poker44-subnet"
PYTHON="${ROOT}/miner_env/bin/python"
cd "${ROOT}"

DEPLOY="${DEPLOY:-1}"
PUBLISH="${PUBLISH:-1}"
MIN_SOURCE_DATE="${MIN_SOURCE_DATE:-2026-05-26}"

echo "Training v12 R1-recall profile (absolute regime grid)..."
ARGS=(
  scripts/train_innovative_model.py
  --profile v12
  --no-live-augment
  --min-source-date "${MIN_SOURCE_DATE}"
  --folds 3
  --n-jobs 1
)

if [[ "${DEPLOY}" == "1" ]]; then
  ARGS+=(--deploy --force-deploy)
fi

"${PYTHON}" "${ARGS[@]}"

echo "Regenerating model manifest..."
"${PYTHON}" scripts/generate_release_manifest.py

if [[ "${PUBLISH}" == "1" && "${DEPLOY}" == "1" ]]; then
  echo "Publishing to GitHub..."
  bash scripts/publish_miner_repo.sh
  pm2 restart sn126-miner --update-env || bash /root/bittensor-mining/scripts/pm2_start_sn126_miner.sh
fi

echo "v12 R1-recall pipeline complete."
