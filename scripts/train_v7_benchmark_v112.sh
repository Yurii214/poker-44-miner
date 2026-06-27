#!/usr/bin/env bash
# Train v7 on benchmark v1.12 (2026-05-26+), retune regime, deploy, publish manifest.
set -euo pipefail

ROOT="/root/bittensor-mining/Poker44-subnet"
PYTHON="${ROOT}/miner_env/bin/python"
cd "${ROOT}"

DEPLOY="${DEPLOY:-1}"
PUBLISH="${PUBLISH:-1}"
MAX_LIVE_BATCHES="${MAX_LIVE_BATCHES:-300}"
MIN_SOURCE_DATE="${MIN_SOURCE_DATE:-2026-05-26}"

echo "Step 1: Tune regime threshold from live validator batches..."
"${PYTHON}" scripts/tune_regime_from_live_chunks.py \
  --model "${ROOT}/models/bot_detector_v1.joblib" \
  --max-live-batches "${MAX_LIVE_BATCHES}"

echo "Step 2: Train v7 on benchmark v1.12 (min sourceDate ${MIN_SOURCE_DATE})..."
ARGS=(
  scripts/train_innovative_model.py
  --profile v7
  --no-live-augment
  --min-source-date "${MIN_SOURCE_DATE}"
  --folds 3
  --n-jobs 1
)

if [[ "${DEPLOY}" == "1" ]]; then
  ARGS+=(--deploy --force-deploy)
fi

"${PYTHON}" "${ARGS[@]}"

echo "Step 3: Regenerating model manifest..."
"${PYTHON}" scripts/generate_release_manifest.py

if [[ "${PUBLISH}" == "1" && "${DEPLOY}" == "1" ]]; then
  echo "Step 4: Publishing to GitHub..."
  bash scripts/publish_miner_repo.sh
  pm2 restart sn126-miner --update-env || bash /root/bittensor-mining/scripts/pm2_start_sn126_miner.sh
fi

echo "v7 benchmark-v1.12 pipeline complete."
