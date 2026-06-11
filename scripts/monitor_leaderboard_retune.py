#!/usr/bin/env python3
"""Monitor leaderboard updates and retune live_logit_target_median on flat movement."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LEADERBOARD_URL = "https://api.poker44.net/api/v1/competition/leaderboard"


def fetch_uid_row(uid: int) -> dict[str, Any]:
    with urllib.request.urlopen(LEADERBOARD_URL, timeout=30) as response:
        payload = json.load(response)["data"]
    rows = payload.get("rows") or []
    row = next((item for item in rows if int(item.get("uid", -1)) == int(uid)), None)
    if not row:
        raise RuntimeError(f"UID {uid} not found in leaderboard rows.")
    return row


def read_target_median(model_path: Path) -> float:
    artifact = joblib.load(model_path)
    metadata = artifact.get("metadata") or {}
    return float(metadata.get("live_logit_target_median", 0.28) or 0.28)


def is_flat(recent: list[dict[str, Any]]) -> bool:
    if len(recent) < 3:
        return False
    last = recent[-3:]
    ranking = [float(item.get("rankingQuality") or 0.0) for item in last]
    composite = [float(item.get("compositeScore") or 0.0) for item in last]
    rank_values = [int(item.get("rank") or 9999) for item in last]
    ranking_span = max(ranking) - min(ranking)
    composite_span = max(composite) - min(composite)
    rank_best = min(rank_values)
    rank_latest = rank_values[-1]
    return ranking_span < 0.01 and composite_span < 0.01 and rank_latest >= rank_best


def retune(model_path: Path, target: float) -> None:
    subprocess.run(
        [
            "python3",
            str(REPO_ROOT / "scripts" / "patch_live_calibration.py"),
            "--model",
            str(model_path),
            "--target-median",
            f"{target:.2f}",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    subprocess.run(["pm2", "restart", "sn126-miner"], check=True, cwd=str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uid", type=int, default=164)
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "bot_detector_v1.joblib")
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--max-updates", type=int, default=6)
    args = parser.parse_args()

    history: list[dict[str, Any]] = []
    seen_update: str | None = None
    retuned = False
    target_median = read_target_median(args.model)
    print(f"MONITOR start uid={args.uid} target={target_median:.2f}")

    while len(history) < args.max_updates:
        try:
            row = fetch_uid_row(args.uid)
        except Exception as exc:  # noqa: BLE001
            print(f"MONITOR fetch_error={exc}")
            time.sleep(args.poll_seconds)
            continue

        updated_at = str(row.get("updatedAt") or "")
        if updated_at and updated_at != seen_update:
            seen_update = updated_at
            history.append(row)
            print(
                "UPDATE "
                f"n={len(history)} updatedAt={updated_at} rank={row.get('rank')} "
                f"composite={row.get('compositeScore')} rankingQuality={row.get('rankingQuality')} "
                f"modelVersion={row.get('modelVersion')}"
            )
            if not retuned and is_flat(history):
                new_target = max(0.22, min(0.34, target_median - 0.02))
                if abs(new_target - target_median) >= 1e-6:
                    retune(args.model, new_target)
                    print(f"RETUNED live_logit_target_median {target_median:.2f} -> {new_target:.2f}")
                    target_median = new_target
                    retuned = True
                else:
                    print("RETUNE skipped (target already at boundary)")
        time.sleep(args.poll_seconds)

    print("MONITOR completed")


if __name__ == "__main__":
    main()
