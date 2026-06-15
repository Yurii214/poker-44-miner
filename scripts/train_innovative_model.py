#!/usr/bin/env python3
"""Train an innovative dual-branch batch-aware Poker44 detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from poker44.score.scoring import reward
from poker44_ml.calibration import simulate_live_miner_scores
from poker44_ml.innovative_model import (
    DualBranchBatchAwareModel,
    transform_batch_relative_rows,
)
from poker44_ml.rank_stack import batch_rank_boost
from train_reference_stack import (  # noqa: E402
    DEFAULT_STATE_PATH,
    fetch_training_examples,
    fit_model,
    make_base_models,
    sample_weights,
    select_live_calibration,
    vectorize,
)

DEFAULT_OUTPUT = REPO_ROOT / "models" / "bot_detector_innovative.joblib"
DEFAULT_DEPLOY_PATH = REPO_ROOT / "models" / "bot_detector_v1.joblib"
MODEL_VERSION = "reference-dualbranch-v4-live"
HOLDOUT_SOURCE_DATES = 5


def holdout_mask_from_metadata(
    metadata: list[dict[str, Any]],
    *,
    holdout_dates: int = HOLDOUT_SOURCE_DATES,
) -> np.ndarray:
    dates = sorted({str(row["source_date"]) for row in metadata})
    if len(dates) <= 1:
        return np.zeros(len(metadata), dtype=bool)
    keep = max(1, min(int(holdout_dates), len(dates) // 3 or 1))
    holdout = set(dates[-keep:])
    return np.asarray([str(row["source_date"]) in holdout for row in metadata], dtype=bool)


def _simulate_live_scores(
    scores: np.ndarray,
    settings: dict[str, Any],
    *,
    max_positive_rate: float,
) -> np.ndarray:
    return simulate_live_miner_scores(
        scores,
        bias=float(settings.get("score_logit_bias", 2.2) or 2.2),
        temperature=float(settings.get("score_logit_temperature", 0.65) or 0.65),
        spread_blend=float(settings.get("live_batch_spread_blend", 0.70) or 0.70),
        center_blend=float(settings.get("live_batch_center_blend", 0.0) or 0.0),
        spread_low=settings.get("live_batch_spread_low"),
        spread_high=settings.get("live_batch_spread_high"),
        max_positive_rate=max_positive_rate,
        logit_mode=str(settings.get("live_logit_mode", "adaptive") or "adaptive"),
        target_median=float(settings.get("live_logit_target_median", 0.24) or 0.24),
    )


def _live_selection_tuple(
    live_scores: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray,
    holdout_mask: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    from train_reference_stack import _within_group_spearman

    rew, meta = reward(live_scores, y)
    holdout_rew = float(rew)
    if holdout_mask is not None and bool(holdout_mask.any()):
        holdout_rew, _ = reward(live_scores[holdout_mask], y[holdout_mask])
    spearman = float(_within_group_spearman(live_scores, y, groups).get("mean", 0.0))
    bot_recall = float(meta.get("bot_recall", 0.0) or 0.0)
    return (
        float(holdout_rew),
        float(rew),
        bot_recall,
        spearman,
    )


def sweep_blend(
    abs_oof: np.ndarray,
    rel_oof: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray,
    max_fpr: float,
    max_positive_rate: float,
    holdout_mask: np.ndarray | None = None,
) -> tuple[float, dict[str, Any], dict[str, Any], np.ndarray]:
    best_alpha = 0.75
    best_metrics: dict[str, Any] = {}
    best_settings: dict[str, Any] = {}
    best_scores = abs_oof
    best_selection = (-1.0, -1.0, -1.0, -1.0)

    for alpha in np.linspace(0.15, 0.55, 9):
        scores = alpha * abs_oof + (1.0 - alpha) * rel_oof
        settings, metrics = select_live_calibration(
            scores,
            y,
            max_fpr=max_fpr,
            max_positive_rate=max_positive_rate,
            groups=groups,
        )
        live = _simulate_live_scores(scores, settings, max_positive_rate=max_positive_rate)
        selection = _live_selection_tuple(
            live,
            y,
            groups=groups,
            holdout_mask=holdout_mask,
        )
        batch_spearman = selection[3]
        print(
            f"alpha={alpha:.2f} holdout={selection[0]:.4f} reward={selection[1]:.4f} "
            f"bot_recall={selection[2]:.4f} fpr={metrics.get('reward_meta', {}).get('fpr', 0.0):.4f} "
            f"batch_spearman={batch_spearman:.4f}"
        )
        if selection > best_selection:
            best_selection = selection
            best_alpha = float(alpha)
            best_settings = settings
            best_metrics = metrics
            best_metrics["batch_spearman"] = batch_spearman
            best_metrics["holdout_reward"] = selection[0]
            best_metrics["bot_recall"] = selection[2]
            best_scores = scores
    return best_alpha, best_settings, best_metrics, best_scores


def batch_groups_from_metadata(metadata: list[dict[str, Any]]) -> np.ndarray:
    chunk_ids = [str(row["chunk_id"]) for row in metadata]
    lookup = {chunk_id: idx for idx, chunk_id in enumerate(sorted(set(chunk_ids)))}
    return np.asarray([lookup[cid] for cid in chunk_ids], dtype=int)


def _predict_pos(model: Any, x: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x), dtype=float)
    return np.clip(proba[:, 1] if proba.ndim == 2 else proba, 0.0, 1.0)


def train_branch_oof(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    folds: int,
    transform_relative: bool,
) -> tuple[list[Any], np.ndarray]:
    weights = sample_weights(y)
    specs = make_base_models(seed)
    cv = GroupKFold(n_splits=max(2, min(folds, len(np.unique(groups)))))
    oof = np.zeros((len(y), len(specs)), dtype=float)

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(x, y, groups=groups), start=1):
        x_train = x[train_idx]
        x_valid = x[valid_idx]
        if transform_relative:
            x_train = transform_batch_relative_rows(x_train, groups[train_idx])
            x_valid = transform_batch_relative_rows(x_valid, groups[valid_idx])
        for model_idx, (_, model) in enumerate(specs):
            fitted = fit_model(clone(model), x_train, y[train_idx], weights[train_idx])
            oof[valid_idx, model_idx] = _predict_pos(fitted, x_valid)
        branch = "relative" if transform_relative else "absolute"
        print(f"Finished {branch} fold {fold_idx}/{cv.get_n_splits()}")

    final_models: list[Any] = []
    x_full = transform_batch_relative_rows(x, groups) if transform_relative else x
    for _, model in specs:
        final_models.append(fit_model(clone(model), x_full, y, weights))
    return final_models, np.mean(oof, axis=1)


def sweep_rank_boost(
    feature_dicts: list[dict[str, float]],
    raw_scores: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray,
    live_settings: dict[str, Any],
    max_positive_rate: float,
    holdout_mask: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    best_boost = 0.0
    best_metrics: dict[str, Any] = {}
    best_selection = (-1.0, -1.0, -1.0, -1.0)

    for boost in (0.0, 0.08, 0.12, 0.18, 0.25, 0.32):
        boosted = batch_rank_boost(feature_dicts, raw_scores.tolist(), blend=boost)
        live = _simulate_live_scores(
            np.asarray(boosted, dtype=float),
            live_settings,
            max_positive_rate=max_positive_rate,
        )
        selection = _live_selection_tuple(
            live,
            y,
            groups=groups,
            holdout_mask=holdout_mask,
        )
        print(
            f"rank_boost={boost:.2f} holdout={selection[0]:.4f} reward={selection[1]:.4f} "
            f"bot_recall={selection[2]:.4f} batch_spearman={selection[3]:.4f}"
        )
        if selection > best_selection:
            best_boost = float(boost)
            best_selection = selection
            _, meta = reward(live, y)
            best_metrics = {
                "reward": selection[1],
                "reward_meta": meta,
                "batch_spearman": selection[3],
                "holdout_reward": selection[0],
                "bot_recall": selection[2],
                "score_std": float(live.std()),
                "score_max": float(live.max()),
            }
    return best_boost, best_metrics


def evaluate_artifact_reward(
    artifact: dict[str, Any],
    feature_dicts: list[dict[str, float]],
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
) -> float:
    model = artifact["models"][0]
    feature_names = artifact["feature_names"]
    x = np.asarray(
        [[float(row.get(name, 0.0)) for name in feature_names] for row in feature_dicts],
        dtype=float,
    )
    if groups is None:
        groups = np.zeros(len(x), dtype=int)
    md = artifact.get("metadata") or {}
    boost = float(md.get("live_batch_rank_boost", 0.0) or 0.0)
    raw = np.zeros(len(x), dtype=float)
    for group_id in np.unique(groups):
        mask = groups == group_id
        batch_x = x[mask]
        batch_features = [feature_dicts[int(idx)] for idx in np.where(mask)[0]]
        if hasattr(model, "predict_chunk_scores"):
            raw[mask] = np.asarray(
                model.predict_chunk_scores(
                    [[] for _ in range(int(mask.sum()))],
                    feature_rows=[list(row) for row in batch_x],
                ),
                dtype=float,
            )
        else:
            raw[mask] = np.asarray(model.predict_proba(batch_x))[:, 1]
        if boost > 0.0:
            boosted = batch_rank_boost(
                batch_features,
                raw[mask].tolist(),
                blend=boost,
            )
            raw[mask] = np.asarray(boosted, dtype=float)
    live = simulate_live_miner_scores(
        raw,
        bias=float(md.get("score_logit_bias", 2.2) or 2.2),
        temperature=float(md.get("score_logit_temperature", 0.65) or 0.65),
        spread_blend=float(md.get("live_batch_spread_blend", 0.70) or 0.70),
        center_blend=float(md.get("live_batch_center_blend", 0.0) or 0.0),
        spread_low=md.get("live_batch_spread_low"),
        spread_high=md.get("live_batch_spread_high"),
        max_positive_rate=float(md.get("live_max_positive_rate", 0.10) or 0.10),
        logit_mode=str(md.get("live_logit_mode", "adaptive") or "adaptive"),
        target_median=float(md.get("live_logit_target_median", 0.24) or 0.24),
    )
    rew, _ = reward(live, y)
    return float(rew)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--deploy-path", type=Path, default=DEFAULT_DEPLOY_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-fpr", type=float, default=0.04)
    parser.add_argument("--max-positive-rate", type=float, default=0.10)
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()

    print("Fetching benchmark examples...")
    feature_dicts, y, metadata, benchmark_state = fetch_training_examples()
    x, feature_names = vectorize(feature_dicts)
    groups = batch_groups_from_metadata(metadata)
    holdout_mask = holdout_mask_from_metadata(metadata)
    holdout_count = int(holdout_mask.sum())
    print(
        f"Loaded {len(y)} samples | features={len(feature_names)} | "
        f"groups={len(np.unique(groups))} | holdout={holdout_count}"
    )

    abs_models, abs_oof = train_branch_oof(
        x,
        y,
        groups,
        seed=args.seed,
        folds=args.folds,
        transform_relative=False,
    )
    rel_models, rel_oof = train_branch_oof(
        x,
        y,
        groups,
        seed=args.seed + 11,
        folds=args.folds,
        transform_relative=True,
    )

    alpha, live_settings, live_metrics, blended_oof = sweep_blend(
        abs_oof,
        rel_oof,
        y,
        groups=groups,
        max_fpr=args.max_fpr,
        max_positive_rate=args.max_positive_rate,
        holdout_mask=holdout_mask,
    )

    rank_boost, rank_metrics = sweep_rank_boost(
        feature_dicts,
        blended_oof,
        y,
        groups=groups,
        live_settings=live_settings,
        max_positive_rate=args.max_positive_rate,
        holdout_mask=holdout_mask,
    )
    live_settings["live_batch_rank_boost"] = rank_boost
    live_metrics["rank_boost_metrics"] = rank_metrics
    live_metrics["batch_spearman"] = max(
        float(live_metrics.get("batch_spearman", 0.0) or 0.0),
        float(rank_metrics.get("batch_spearman", 0.0) or 0.0),
    )

    dual_model = DualBranchBatchAwareModel(
        absolute_models=abs_models,
        relative_models=rel_models,
        absolute_weight=alpha,
    )
    x_ap = average_precision_score(y, blended_oof)
    x_auc = roc_auc_score(y, blended_oof)
    diagnostics = {
        "absolute_oof_ap": float(average_precision_score(y, abs_oof)),
        "absolute_oof_auc": float(roc_auc_score(y, abs_oof)),
        "relative_oof_ap": float(average_precision_score(y, rel_oof)),
        "relative_oof_auc": float(roc_auc_score(y, rel_oof)),
        "blend_oof_ap": float(x_ap),
        "blend_oof_auc": float(x_auc),
        "absolute_weight": float(alpha),
        "relative_weight": float(1.0 - alpha),
        "cv_strategy": "groupkfold_dual_branch",
        "cv_folds": int(max(2, min(args.folds, len(np.unique(groups))))),
    }

    artifact = {
        "models": [dual_model],
        "model_weights": [1.0],
        "feature_names": feature_names,
        "metadata": {
            "model_name": "poker44-innovative-dual-branch",
            "model_version": MODEL_VERSION,
            "model_weights": [1.0],
            **live_settings,
            "score_remap": {},
            "metrics": live_metrics,
            "diagnostics": diagnostics,
            "training_objective": "dual_branch_absolute_relative_live_holdout",
        },
        "metrics": live_metrics,
        "training_samples": len(y),
        "metadata_rows": metadata,
        "benchmark_state": benchmark_state,
        "model_version": MODEL_VERSION,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    args.state_file.write_text(json.dumps(benchmark_state, indent=2) + "\n")
    print(json.dumps({"diagnostics": diagnostics, "live_metrics": live_metrics}, indent=2))
    print(f"Saved model to {args.output}")

    baseline_reward = 0.0
    if args.deploy_path.exists():
        baseline_artifact = joblib.load(args.deploy_path)
        baseline_reward = evaluate_artifact_reward(
            baseline_artifact,
            feature_dicts,
            y,
            groups=groups,
        )
    new_reward = evaluate_artifact_reward(
        artifact,
        feature_dicts,
        y,
        groups=groups,
    )
    print(f"Benchmark reward comparison | baseline={baseline_reward:.4f} new={new_reward:.4f}")

    if args.deploy and new_reward >= baseline_reward - 0.005:
        backup_path = args.deploy_path.with_name(
            f"{args.deploy_path.stem}.{MODEL_VERSION}_backup.joblib"
        )
        if args.deploy_path.exists():
            joblib.dump(joblib.load(args.deploy_path), backup_path)
        joblib.dump(artifact, args.deploy_path)
        print(f"DEPLOYED {args.deploy_path} backup={backup_path}")
    else:
        print("Skipped deploy (insufficient improvement or --deploy not set).")


if __name__ == "__main__":
    main()
