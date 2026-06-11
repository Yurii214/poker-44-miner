#!/usr/bin/env bash
# Publish miner implementation to your public GitHub repo for manifest compliance.
set -euo pipefail

REPO_URL="${POKER44_MODEL_REPO_URL:-https://github.com/Yurii214/poker-44-miner}"
REMOTE_NAME="${REMOTE_NAME:-yurii-miner}"
BRANCH="${BRANCH:-main}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Set GITHUB_TOKEN to a classic PAT with 'repo' scope, then re-run."
  echo "Create one: GitHub → Settings → Developer settings → Personal access tokens"
  exit 1
fi

HOST_PATH="${REPO_URL#https://}"
PUSH_URL="https://${GITHUB_TOKEN}@${HOST_PATH}.git"

if ! git remote get-url "${REMOTE_NAME}" &>/dev/null; then
  git remote add "${REMOTE_NAME}" "${PUSH_URL}"
else
  git remote set-url "${REMOTE_NAME}" "${PUSH_URL}"
fi

git add \
  neurons/miner.py \
  poker44_ml/ \
  poker44/utils/model_manifest.py \
  scripts/patch_live_calibration.py \
  scripts/train_innovative_model.py \
  scripts/train_reference_stack.py \
  scripts/monitor_leaderboard_retune.py

if git diff --cached --quiet; then
  echo "No staged changes; pushing current HEAD."
else
  git commit -m "$(cat <<'EOF'
Publish Poker44 UID 164 miner implementation for manifest compliance.

Includes dual-branch inference stack, calibration, and training scripts.
EOF
)"
fi

git push -u "${REMOTE_NAME}" HEAD:"${BRANCH}"

COMMIT="$(git rev-parse HEAD)"
git remote set-url "${REMOTE_NAME}" "${REPO_URL}.git"

echo ""
echo "Published to ${REPO_URL}"
echo "Commit: ${COMMIT}"
echo ""
echo "Add to /root/bittensor-mining/scripts/start_sn126_miner.sh (or export before restart):"
echo "  export POKER44_MODEL_REPO_COMMIT=${COMMIT}"
echo "  pm2 restart sn126-miner --update-env"
