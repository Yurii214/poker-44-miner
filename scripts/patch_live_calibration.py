#!/usr/bin/env python3
"""Patch deployed artifact with auto-adaptive live logit calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_MODEL = REPO_ROOT / "models" / "bot_detector_v1.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--target-median", type=float, default=0.28)
    parser.add_argument("--collapse-median", type=float, default=0.52)
    parser.add_argument("--collapse-std", type=float, default=0.04)
    parser.add_argument("--rank-boost", type=float, default=0.0)
    parser.add_argument("--spread-blend", type=float, default=0.70)
    parser.add_argument("--spread-low", type=float, default=0.12)
    parser.add_argument("--spread-high", type=float, default=0.40)
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    if args.backup:
        backup_path = args.model.with_name(
            f"{args.model.stem}.v41_pre_adaptive_backup.joblib"
        )
        joblib.dump(artifact, backup_path)
        print(f"Backup saved to {backup_path}")

    metadata = dict(artifact.get("metadata") or {})
    current_version = str(
        artifact.get("model_version")
        or metadata.get("model_version")
        or "reference-stack"
    )
    if current_version.endswith("-live-adaptive"):
        patch_version = current_version
    else:
        patch_version = f"{current_version}-live-adaptive"
    metadata.update(
        {
            "model_version": patch_version,
            "live_logit_mode": "adaptive",
            "live_logit_target_median": float(args.target_median),
            "live_logit_collapse_median": float(args.collapse_median),
            "live_logit_collapse_std": float(args.collapse_std),
            "live_batch_rank_boost": float(args.rank_boost),
            "live_batch_spread_blend": float(args.spread_blend),
            "live_batch_spread_low": float(args.spread_low),
            "live_batch_spread_high": float(args.spread_high),
        }
    )
    artifact["metadata"] = metadata
    artifact["model_version"] = patch_version
    joblib.dump(artifact, args.model)
    print(json.dumps({"patched": str(args.model), "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
