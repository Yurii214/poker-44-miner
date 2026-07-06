#!/usr/bin/env bash
# Train v13: recent-holdout stress + higher bot recall for R2/R3 rounds.
set -euo pipefail

ROOT="/root/bittensor-mining/Poker44-subnet"
PYTHON="${ROOT}/miner_env/bin/python"
cd "${ROOT}"

DEPLOY="${DEPLOY:-1}"
PUBLISH="${PUBLISH:-1}"
MIN_SOURCE_DATE="${MIN_SOURCE_DATE:-2026-05-26}"

echo "Training v13 R2-recall profile (recent holdout stress, absolute regime)..."
ARGS=(
  scripts/train_innovative_model.py
  --profile v13
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

echo "v13 R2-recall pipeline complete."
