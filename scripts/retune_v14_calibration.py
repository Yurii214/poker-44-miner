#!/usr/bin/env python3
"""Retune v14 live calibration on recent benchmark dates for new eval format."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from poker44.score.scoring import reward
from poker44_ml.anomaly_branch import HumanIsolationScorer, fuse_hybrid_scores
from poker44_ml.innovative_model import DualBranchBatchAwareModel
from poker44_ml.rank_stack import batch_rank_boost
from train_innovative_model import batch_groups_from_metadata
from train_reference_stack import fetch_training_examples, select_regime_calibration


def hybrid_scores_from_artifact(
    artifact: dict,
    feature_dicts: list[dict[str, float]],
    groups: np.ndarray,
) -> np.ndarray:
    model = artifact["models"][0]
    feature_names = list(artifact["feature_names"])
    x = np.asarray(
        [[float(row.get(name, 0.0)) for name in feature_names] for row in feature_dicts],
        dtype=float,
    )
    if not isinstance(model, DualBranchBatchAwareModel):
        return np.asarray(model.predict_proba(x))[:, 1]

    sup = np.zeros(len(x), dtype=float)
    for group_id in np.unique(groups):
        mask = groups == group_id
        sup[mask] = np.asarray(model.predict_proba(x[mask]))[:, 1]

    iso_payload = artifact.get("anomaly_scorer")
    boost = float((artifact.get("metadata") or {}).get("live_batch_rank_boost", 0.0) or 0.0)
    if iso_payload is not None:
        iso = HumanIsolationScorer.from_dict(iso_payload)
        iso_scores = iso.predict_bot_scores(x)
        if boost > 0.0:
            sup = np.asarray(
                batch_rank_boost(feature_dicts, sup.tolist(), blend=boost),
                dtype=float,
            )
        return fuse_hybrid_scores(sup, iso_scores, mode="max")
    return sup


def recent_holdout_mask(metadata: list[dict], *, min_date: str) -> np.ndarray:
    return np.array(
        [str(row.get("source_date", "")) >= min_date for row in metadata],
        dtype=bool,
    )


def eval_recent(
    model_path: Path,
    feature_dicts,
    y,
    groups,
    metadata,
    min_date: str,
) -> dict:
    artifact = joblib.load(model_path)
    hybrid = hybrid_scores_from_artifact(artifact, feature_dicts, groups)
    from poker44_ml.calibration import simulate_regime_live_miner_scores

    meta = artifact.get("metadata") or {}
    mask = recent_holdout_mask(metadata, min_date=min_date)
    live = simulate_regime_live_miner_scores(
        hybrid,
        regime_threshold=float(meta.get("live_regime_threshold", 0.28)),
        human_spread=tuple(meta.get("live_human_spread") or (0.03, 0.14)),
        bot_spread=tuple(meta.get("live_bot_spread") or (0.28, 0.6)),
        spread_blend=float(meta.get("live_batch_spread_blend", 0.92)),
        batch_size=int(meta.get("live_batch_size", 80)),
        max_positive_rate=float(meta.get("live_max_positive_rate", 0.05)),
        human_max_positive_rate=float(meta.get("live_human_max_positive_rate", 0.0)),
        bot_max_positive_rate=float(meta.get("live_bot_max_positive_rate", 0.34)),
        hard_ceiling=meta.get("live_human_score_ceiling"),
        human_hard_ceiling=meta.get("live_human_score_ceiling"),
        bot_hard_ceiling=meta.get("live_bot_score_ceiling"),
        regime_scores=hybrid,
        chunk_regime=True,
        regime_mode=str(meta.get("live_regime_mode", "absolute")),
        human_fraction=float(meta.get("live_human_fraction", 0.35)),
        apply_positive_cap=False,
        apply_miner_cap_replay=True,
    )
    sel = live[mask]
    lab = y[mask]
    rew, rmeta = reward(sel, lab)
    return {
        "n": int(mask.sum()),
        "reward": float(rew),
        "bot_recall": float(rmeta.get("bot_recall", 0.0)),
        "fpr": float(rmeta.get("fpr", 0.0)),
        "ap": float(rmeta.get("ap_score", 0.0)),
    }


def main() -> None:
    model_path = REPO_ROOT / "models" / "bot_detector_v1.joblib"
    backup_path = REPO_ROOT / "models" / "bot_detector_v1.reference-dualbranch-v14-v113-top10_backup.joblib"
    source_path = backup_path if backup_path.exists() else model_path
    artifact = joblib.load(source_path)
    print(f"Loaded {source_path}")

    feature_dicts, y, metadata, _ = fetch_training_examples(min_source_date="2026-06-07")
    groups = batch_groups_from_metadata(metadata)
    hybrid = hybrid_scores_from_artifact(artifact, feature_dicts, groups)
    holdout = recent_holdout_mask(metadata, min_date="2026-06-28")
    print(f"Rows={len(y)} holdout_recent={int(holdout.sum())} dates>=2026-06-28")

    live_settings, metrics = select_regime_calibration(
        hybrid,
        y,
        max_fpr=0.005,
        batch_size=80,
        max_positive_rate=0.05,
        human_max_positive_rate=0.0,
        bot_max_positive_rate=0.36,
        groups=groups,
        spread_blend=0.92,
        spearman_mask=np.ones(len(y), dtype=bool),
        hard_ceiling=0.49,
        human_hard_ceiling=0.49,
        bot_hard_ceiling=0.70,
        reward_first=True,
        min_bot_recall=0.46,
        regime_scores=hybrid,
        chunk_regime=True,
        holdout_mask=holdout,
        extended_grid=True,
        regime_mode="absolute",
        human_fraction=0.35,
        dual_fpr=True,
        max_full_fpr=0.012,
    )
    cal = metrics.get("calibration_choice") or {}
    live_settings = dict(live_settings)

    version = "reference-dualbranch-v14-v113-calibrated"
    artifact["model_version"] = version
    meta = dict(artifact.get("metadata") or {})
    meta.update(live_settings)
    meta["model_version"] = version
    meta["calibration_choice"] = cal
    meta["metrics"] = {
        **(meta.get("metrics") or {}),
        **metrics,
        "calibration_choice": cal,
    }
    artifact["metadata"] = meta
    artifact["metrics"] = meta["metrics"]

    out = REPO_ROOT / "models" / "bot_detector_v1.joblib"
    joblib.dump(artifact, out)
    print(
        json.dumps(
            {
                "calibration": cal,
                "recent_eval_before": eval_recent(
                    source_path, feature_dicts, y, groups, metadata, "2026-06-28"
                ),
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "recent_eval_after": eval_recent(
                    out, feature_dicts, y, groups, metadata, "2026-06-28"
                ),
            },
            indent=2,
        )
    )
    print(f"Patched calibration -> {out}")


if __name__ == "__main__":
    main()
