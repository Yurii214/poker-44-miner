#!/usr/bin/env python3
"""Write models/model_manifest.json for the public release repository."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker44.utils.model_manifest import build_local_model_manifest, sha256_file


def main() -> int:
    repo_url = "https://github.com/Yurii214/poker-44-miner"
    implementation_files = [
        ROOT / "neurons/miner.py",
        ROOT / "poker44_ml/features.py",
        ROOT / "poker44_ml/inference.py",
        ROOT / "poker44_ml/innovative_model.py",
        ROOT / "poker44_ml/stacked.py",
        ROOT / "poker44_ml/rank_stack.py",
        ROOT / "poker44_ml/calibration.py",
    ]
    artifact_rel = "models/bot_detector_v1.joblib"
    artifact_path = ROOT / artifact_rel
    model_name = "poker44-innovative-dual-branch"
    model_version = "reference-dualbranch-v4-live"
    if artifact_path.is_file():
        artifact = joblib.load(artifact_path)
        metadata = dict(artifact.get("metadata") or {})
        model_name = str(
            metadata.get("model_name")
            or artifact.get("model_name")
            or model_name
        )
        model_version = str(
            artifact.get("model_version")
            or metadata.get("model_version")
            or model_version
        )
    data_statement = (
        "Trained on public Poker44 benchmark releases fetched from "
        "https://api.poker44.net/api/v1/benchmark using miner-visible hand payloads."
    )
    manifest = build_local_model_manifest(
        repo_root=ROOT,
        implementation_files=implementation_files,
        defaults={
            "model_name": model_name,
            "model_version": model_version,
            "framework": "lightgbm-xgboost-batch-rank-stack-quantile-remap",
            "license": "MIT",
            "repo_url": repo_url,
            "open_source": True,
            "inference_mode": "remote",
            "training_data_statement": data_statement,
            "training_data_sources": [
                "https://api.poker44.net/api/v1/benchmark/releases",
            ],
            "private_data_attestation": (
                "This miner does not train on validator-only live evaluation labels."
            ),
            "data_attestation": (
                "Training data is limited to public Poker44 benchmark releases "
                "and miner-visible hand payloads from api.poker44.net. "
                "No validator-private live labels, human PII, or proprietary "
                "table data were used."
            ),
            "artifact_url": artifact_rel,
            "artifact_sha256": sha256_file(artifact_path) if artifact_path.is_file() else "",
            "model_card_url": f"{repo_url}/blob/main/README.md",
            "notes": "Dual-branch benchmark stack with bounded live calibration.",
        },
    )
    out = ROOT / "models" / "model_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    print(f"implementation_sha256={manifest.get('implementation_sha256')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
