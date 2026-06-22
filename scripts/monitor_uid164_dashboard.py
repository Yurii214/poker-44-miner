#!/usr/bin/env python3
"""Monitor UID 164 competition metrics and manifest status on poker44.net."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LEADERBOARD_URL = "https://api.poker44.net/api/v1/competition/leaderboard"
MINER_URL = "https://api.poker44.net/api/v1/miners/{uid}"
DEFAULT_LOG = REPO_ROOT / "models" / "uid164_monitor.log"

METRIC_KEYS = (
    "rank",
    "compositeScore",
    "rankingQuality",
    "classificationQuality",
    "humanSafetyPenalty",
    "manifestReviewFailed",
    "modelVersion",
    "updatedAt",
)


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def fetch_leaderboard_row(uid: int) -> dict[str, Any]:
    payload = _get_json(LEADERBOARD_URL)["data"]
    rows = payload.get("rows") or []
    row = next((item for item in rows if int(item.get("uid", -1)) == int(uid)), None)
    if not row:
        raise RuntimeError(f"UID {uid} not found in leaderboard rows.")
    return row


def fetch_miner_detail(uid: int) -> dict[str, Any]:
    payload = _get_json(MINER_URL.format(uid=uid))
    return payload.get("data") or payload


def read_local_model(model_path: Path) -> dict[str, Any]:
    if not model_path.is_file():
        return {}
    artifact = joblib.load(model_path)
    metadata = dict(artifact.get("metadata") or {})
    return {
        "model_version": artifact.get("model_version") or metadata.get("model_version"),
        "live_logit_target_median": metadata.get("live_logit_target_median"),
        "live_max_positive_rate": metadata.get("live_max_positive_rate"),
        "live_batch_spread_blend": metadata.get("live_batch_spread_blend"),
        "batch_spearman": (metadata.get("metrics") or {}).get("batch_spearman"),
    }


def format_snapshot(
    uid: int,
    leaderboard: dict[str, Any],
    miner: dict[str, Any],
    local: dict[str, Any],
) -> str:
    manifest = miner.get("modelManifest") or miner.get("model_manifest") or {}
    nested_commit = manifest.get("repo_commit") or manifest.get("repoCommit")
    leaderboard_commit = miner.get("repoCommit") or miner.get("repo_commit")
    lines = [
        f"[{datetime.now(timezone.utc).isoformat()}] uid={uid}",
        "  leaderboard: "
        + " ".join(
            f"{key}={leaderboard.get(key)!r}"
            for key in METRIC_KEYS
            if key in leaderboard
        ),
        f"  miner_repo_commit={leaderboard_commit!r} nested_manifest_commit={nested_commit!r}",
        f"  manifestReviewFailed={miner.get('manifestReviewFailed', leaderboard.get('manifestReviewFailed'))!r}",
        f"  local_model={local!r}",
    ]
    rounds = leaderboard.get("roundScores") or leaderboard.get("rounds") or []
    if rounds:
        recent = rounds[-3:]
        lines.append(f"  recent_rounds={json.dumps(recent, separators=(',', ':'))}")
    return "\n".join(lines)


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uid", type=int, default=164)
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "bot_detector_v1.joblib")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-updates", type=int, default=0, help="0 = run once")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    local = read_local_model(args.model)
    seen_update: str | None = None
    updates = 0

    while True:
        try:
            leaderboard = fetch_leaderboard_row(args.uid)
            miner = fetch_miner_detail(args.uid)
            snapshot = format_snapshot(args.uid, leaderboard, miner, local)
            updated_at = str(leaderboard.get("updatedAt") or "")
            if updated_at != seen_update:
                seen_update = updated_at
                updates += 1
                print(snapshot)
                append_log(args.log_file, snapshot)
            elif args.max_updates == 0 and updates == 0:
                print(snapshot)
                append_log(args.log_file, snapshot)
                updates = 1
        except Exception as exc:  # noqa: BLE001
            message = f"[{datetime.now(timezone.utc).isoformat()}] error={exc}"
            print(message)
            append_log(args.log_file, message)

        if args.max_updates > 0 and updates >= args.max_updates:
            break
        if args.max_updates == 0:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
