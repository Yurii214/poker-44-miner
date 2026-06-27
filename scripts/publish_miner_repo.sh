#!/usr/bin/env bash
# Publish miner implementation to your public GitHub repo for manifest compliance.
set -euo pipefail

REPO_URL="${POKER44_MODEL_REPO_URL:-https://github.com/Yurii214/poker-44-miner}"
REMOTE_NAME="${REMOTE_NAME:-yurii-miner}"
BRANCH="${BRANCH:-main}"
DEPLOY_KEY="${POKER44_GITHUB_DEPLOY_KEY:-/root/.ssh/poker44_deploy}"
START_SCRIPT="${POKER44_START_SCRIPT:-/root/bittensor-mining/scripts/start_sn126_miner.sh}"
COMMIT_PIN_FILE="${POKER44_REPO_COMMIT_FILE:-/root/bittensor-mining/.poker44_repo_commit}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
PYTHON="${ROOT}/miner_env/bin/python"

TOKEN_FILE="${POKER44_GITHUB_TOKEN_FILE:-/root/bittensor-mining/.poker44_github_token}"
if [[ -z "${GITHUB_TOKEN:-}" && -f "${TOKEN_FILE}" ]]; then
  export GITHUB_TOKEN="$(tr -d '[:space:]' < "${TOKEN_FILE}")"
fi

register_deploy_key() {
  if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    return 1
  fi
  local pub_key
  pub_key="$(cat "${DEPLOY_KEY}.pub")"
  curl -fsS -X POST \
    -H "Authorization: token ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/Yurii214/poker-44-miner/keys" \
    -d "$(python3 - <<PY
import json, sys
print(json.dumps({
    "title": "poker44-uid164-deploy",
    "key": sys.stdin.read().strip(),
    "read_only": False,
}))
PY
<<<"${pub_key}")" >/dev/null
}

configure_remote() {
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    local host_path="${REPO_URL#https://}"
    local push_url="https://${GITHUB_TOKEN}@${host_path}.git"
    if ! git remote get-url "${REMOTE_NAME}" &>/dev/null; then
      git remote add "${REMOTE_NAME}" "${push_url}"
    else
      git remote set-url "${REMOTE_NAME}" "${push_url}"
    fi
    return 0
  fi

  if [[ -f "${DEPLOY_KEY}" ]]; then
    if ! git remote get-url "${REMOTE_NAME}" &>/dev/null; then
      git remote add "${REMOTE_NAME}" "git@github.com:Yurii214/poker-44-miner.git"
    else
      git remote set-url "${REMOTE_NAME}" "git@github.com:Yurii214/poker-44-miner.git"
    fi
    export GIT_SSH_COMMAND="ssh -i ${DEPLOY_KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
    return 0
  fi

  echo "No GitHub credentials available."
  echo "Option A: export GITHUB_TOKEN=<classic PAT with repo scope> && $0"
  echo "Option B: add deploy key to https://github.com/Yurii214/poker-44-miner/settings/keys"
  echo "          $(cat "${DEPLOY_KEY}.pub" 2>/dev/null || echo '(deploy key missing)')"
  return 1
}

if [[ -n "${GITHUB_TOKEN:-}" ]] && [[ -f "${DEPLOY_KEY}.pub" ]]; then
  register_deploy_key || true
fi

configure_remote

"${PYTHON}" scripts/generate_release_manifest.py
chmod +x verify.py recompute_hash.py scripts/generate_release_manifest.py

git add \
  README.md \
  verify.py \
  recompute_hash.py \
  neurons/miner.py \
  poker44_ml/ \
  poker44/utils/model_manifest.py \
  models/bot_detector_v1.joblib \
  models/model_manifest.json \
  scripts/generate_release_manifest.py \
  scripts/patch_live_calibration.py \
  scripts/train_innovative_model.py \
  scripts/train_reference_stack.py \
  scripts/train_v7_benchmark_v112.sh \
  scripts/train_v6_rank_first.sh \
  scripts/auto_retrain_sn126.sh \
  scripts/monitor_uid164_dashboard.py \
  scripts/monitor_leaderboard_retune.py \
  scripts/publish_miner_repo.sh \
  .gitignore

if ! git diff --cached --quiet; then
  git commit -m "$(cat <<'EOF'
Release Poker44 UID 164 miner with manifest attestation and model artifact.

Adds data_attestation, model card README, verify scripts, and published joblib.
EOF
)"
fi

"${PYTHON}" recompute_hash.py
git add models/model_manifest.json
if ! git diff --cached --quiet; then
  git commit -m "$(cat <<'EOF'
Pin release model_manifest.json to repository commit hash.
EOF
)"
fi

"${PYTHON}" recompute_hash.py
git add models/model_manifest.json
if ! git diff --cached --quiet; then
  git commit -m "$(cat <<'EOF'
Align model_manifest.json repo_commit with release HEAD.
EOF
)"
fi

git push -u "${REMOTE_NAME}" HEAD:"${BRANCH}"

COMMIT="$(git rev-parse HEAD)"
git remote set-url "${REMOTE_NAME}" "${REPO_URL}.git"

printf '%s\n' "${COMMIT}" > "${COMMIT_PIN_FILE}"

if [[ -f "${START_SCRIPT}" ]]; then
  "${PYTHON}" - <<PY
from pathlib import Path
import re

path = Path("${START_SCRIPT}")
text = path.read_text()
line = f'export POKER44_MODEL_REPO_COMMIT="${COMMIT}"'
if "POKER44_MODEL_REPO_COMMIT" in text:
    text = re.sub(
        r'^export POKER44_MODEL_REPO_COMMIT=.*$',
        line,
        text,
        flags=re.M,
    )
else:
    text = text.replace(
        '# Set automatically by scripts/publish_miner_repo.sh after the first successful push.',
        line,
    )
path.write_text(text)
PY
fi

if command -v pm2 >/dev/null 2>&1; then
  pm2 restart sn126-miner --update-env || true
fi

"${PYTHON}" verify.py

MODEL_VERSION="$("${PYTHON}" - <<'PY'
import joblib
from pathlib import Path

path = Path("models/bot_detector_v1.joblib")
if path.exists():
    artifact = joblib.load(path)
    print(artifact.get("model_version") or artifact.get("metadata", {}).get("model_version", ""))
PY
)"
if [[ -n "${MODEL_VERSION}" ]]; then
  "${PYTHON}" scripts/deploy_lock.py write \
    --model-version "${MODEL_VERSION}" \
    --repo-commit "${COMMIT}"
fi

echo ""
echo "Published to ${REPO_URL}"
echo "Commit: ${COMMIT}"
echo "Pinned commit in ${START_SCRIPT}"
echo "Restarted sn126-miner (if PM2 is available)."
