#!/usr/bin/env bash
# Train v14 on benchmark v1.13 (30 dates), retune calibration, publish for Top-10 target.
set -euo pipefail

ROOT="/root/bittensor-mining/Poker44-subnet"
PYTHON="${ROOT}/miner_env/bin/python"
cd "${ROOT}"

DEPLOY="${DEPLOY:-1}"
PUBLISH="${PUBLISH:-1}"
MIN_SOURCE_DATE="${MIN_SOURCE_DATE:-2026-06-07}"
RETUNE="${RETUNE:-1}"

echo "Backing up current deployed model..."
if [[ -f models/bot_detector_v1.joblib ]]; then
  cp -f models/bot_detector_v1.joblib \
    "models/bot_detector_v1.pre-v14-$(date -u +%Y%m%dT%H%M%SZ).joblib"
fi

echo "Training v14 profile on benchmark v1.13 (all ${MIN_SOURCE_DATE}+ releases)..."
ARGS=(
  scripts/train_innovative_model.py
  --profile v14
  --no-live-augment
  --min-source-date "${MIN_SOURCE_DATE}"
  --folds 3
  --n-jobs 1
)

if [[ "${DEPLOY}" == "1" ]]; then
  ARGS+=(--deploy --force-deploy)
fi

"${PYTHON}" "${ARGS[@]}"

if [[ "${RETUNE}" == "1" && "${DEPLOY}" == "1" ]]; then
  echo "Retuning v14 calibration on recent holdout (>= 2026-06-28)..."
  cp -f models/bot_detector_v1.joblib \
    models/bot_detector_v1.reference-dualbranch-v14-v113-top10_backup.joblib
  "${PYTHON}" scripts/retune_v14_calibration.py
fi

echo "Regenerating model manifest..."
"${PYTHON}" scripts/generate_release_manifest.py

if [[ "${PUBLISH}" == "1" && "${DEPLOY}" == "1" ]]; then
  echo "Publishing to GitHub..."
  bash scripts/publish_miner_repo.sh
  pm2 restart sn126-miner --update-env || bash /root/bittensor-mining/scripts/pm2_start_sn126_miner.sh
fi

echo "v14 benchmark-v1.13 Top-10 pipeline complete."
