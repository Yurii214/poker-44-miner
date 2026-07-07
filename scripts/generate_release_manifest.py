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
    # Must match the runtime rank-head manifest in neurons/miner.py so the
    # attested implementation_sha256 is identical to what the miner attaches.
    implementation_files = [
        ROOT / "neurons/miner.py",
        ROOT / "poker44_ml/top_inference.py",
        ROOT / "poker44_ml/top_model.py",
        ROOT / "poker44_ml/rank_features.py",
        ROOT / "poker44_ml/rank_reward.py",
        ROOT / "poker44_ml/features.py",
        ROOT / "poker44_ml/consistency_features.py",
        ROOT / "poker44_ml/robust_features.py",
    ]
    artifact_rel = "models/bot_detector_top.joblib"
    artifact_path = ROOT / artifact_rel
    model_name = "poker44-rankfirst-ensemble"
    model_version = "top-v1"
    if artifact_path.is_file():
        artifact = joblib.load(artifact_path)
        if isinstance(artifact, dict):
            metadata = dict(artifact.get("metadata") or {})
            raw_version = artifact.get("model_version")
        else:  # dataclass artifact (TopEnsemble) exposes .metadata
            metadata = dict(getattr(artifact, "metadata", {}) or {})
            raw_version = None
        model_name = str(metadata.get("model_name") or model_name)
        model_version = str(raw_version or metadata.get("model_version") or model_version)
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
            "framework": "lightgbm-xgboost-oof-stacked-ensemble-within-batch-rank",
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
            "notes": (
                "Diverse OOF-stacked GBDT ensemble (3x LightGBM + XGBoost + "
                "ExtraTrees + RandomForest) on scale-invariant behavioural, "
                "sequence-collision, temporal-consistency and pot-fraction "
                "features; served via within-batch rank normalisation. Trained on "
                "public benchmark releases (30 dates) from api.poker44.net."
            ),
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
