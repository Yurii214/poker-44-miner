#!/usr/bin/env python3
"""Train a hybrid classifier + batch-grouped LambdaRank stack."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    import lightgbm as lgb
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("lightgbm and xgboost are required for rank stack training.") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from poker44.score.scoring import reward
from poker44.validator.payload_view import prepare_hand_for_miner
from poker44_ml.calibration import BlendedQuantileCalibrator, simulate_live_miner_scores
from poker44_ml.features import chunk_features
from poker44_ml.inference import Poker44Model
from poker44_ml.rank_stack import (
    RankModelAdapter,
    batch_groups_from_metadata,
    fit_ranker,
    predict_rank_proba,
)
from poker44_ml.stacked import StackedEnsemble
from train_reference_stack import (  # noqa: E402
    DEFAULT_STATE_PATH,
    fetch_training_examples,
    fit_model,
    make_base_models,
    sample_weights,
    select_live_calibration,
    vectorize,
    _within_group_spearman,
)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

DEFAULT_OUTPUT = REPO_ROOT / "models" / "bot_detector_rank_stack.joblib"
DEFAULT_DEPLOY_PATH = REPO_ROOT / "models" / "bot_detector_v1.joblib"
MODEL_VERSION = "reference-rank-v2-hybrid-batch"
MIN_DEPLOY_REWARD = 0.40


def make_rank_models(seed: int) -> list[tuple[str, Any]]:
    return [
        (
            "lgb_rank",
            lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=700,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=5,
                min_child_samples=6,
                feature_fraction=0.8,
                bagging_fraction=0.85,
                bagging_freq=1,
                reg_alpha=0.5,
                reg_lambda=1.25,
                random_state=seed,
                verbose=-1,
                n_jobs=-1,
            ),
        ),
        (
            "xgb_rank",
            xgb.XGBRanker(
                objective="rank:pairwise",
                n_estimators=550,
                learning_rate=0.04,
                max_depth=4,
                min_child_weight=4,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.5,
                random_state=seed + 1,
                n_jobs=-1,
            ),
        ),
    ]


def _base_predict(model: Any, x: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x))
    return np.clip(proba[:, 1] if proba.ndim == 2 else proba, 0.0, 1.0)


def batch_cv_classifier_oof(
    x: np.ndarray,
    y: np.ndarray,
    metadata: list[dict],
    *,
    seed: int,
    folds: int,
) -> np.ndarray:
    batch_groups = batch_groups_from_metadata(metadata)
    weights = sample_weights(y)
    specs = make_base_models(seed)
    cv = GroupKFold(n_splits=max(2, min(folds, len(np.unique(batch_groups)))))
    oof = np.zeros((len(y), len(specs)), dtype=float)

    for fold_index, (train_idx, valid_idx) in enumerate(
        cv.split(x, y, groups=batch_groups),
        start=1,
    ):
        for model_index, (_, model) in enumerate(specs):
            fitted = fit_model(clone(model), x[train_idx], y[train_idx], weights[train_idx])
            oof[valid_idx, model_index] = _base_predict(fitted, x[valid_idx])
        print(f"Finished classifier batch-OOF fold {fold_index}/{cv.get_n_splits()}")
    return np.mean(oof, axis=1)


def train_classifier_stack(
    x: np.ndarray,
    y: np.ndarray,
    metadata: list[dict],
    *,
    seed: int,
    folds: int,
) -> tuple[StackedEnsemble, np.ndarray]:
    weights = sample_weights(y)
    specs = make_base_models(seed)
    batch_groups = batch_groups_from_metadata(metadata)
    classifier_oof = batch_cv_classifier_oof(x, y, metadata, seed=seed, folds=folds)
    stack_calibrator = BlendedQuantileCalibrator(blend=0.9).fit(classifier_oof)
    final_models = [fit_model(clone(model), x, y, weights) for _, model in specs]
    ensemble = StackedEnsemble(
        base_models=final_models,
        meta_model=None,
        calibrator=stack_calibrator,
        stack_mode="mean",
    )
    full_fit = np.asarray(ensemble.predict_proba(x))[:, 1]
    return ensemble, full_fit


def train_batch_rank_stack(
    x: np.ndarray,
    y: np.ndarray,
    metadata: list[dict],
    *,
    seed: int,
    folds: int,
) -> tuple[StackedEnsemble, np.ndarray, np.ndarray]:
    batch_groups = batch_groups_from_metadata(metadata)
    specs = make_rank_models(seed)
    cv = GroupKFold(n_splits=max(2, min(folds, len(np.unique(batch_groups)))))
    oof = np.zeros((len(y), len(specs)), dtype=float)

    for fold_index, (train_idx, valid_idx) in enumerate(
        cv.split(x, y, groups=batch_groups),
        start=1,
    ):
        train_groups = batch_groups[train_idx]
        for model_index, (_, model) in enumerate(specs):
            fitted = fit_ranker(clone(model), x[train_idx], y[train_idx], train_groups)
            oof[valid_idx, model_index] = predict_rank_proba(fitted, x[valid_idx])
        print(f"Finished batch-rank OOF fold {fold_index}/{cv.get_n_splits()}")

    rank_oof = np.mean(oof, axis=1)
    final_models = [
        RankModelAdapter(fit_ranker(clone(model), x, y, batch_groups))
        for _, model in specs
    ]
    ensemble = StackedEnsemble(
        base_models=final_models,
        meta_model=None,
        calibrator=None,
        stack_mode="mean",
    )
    full_fit = np.asarray(ensemble.predict_proba(x))[:, 1]
    return ensemble, rank_oof, full_fit


def select_blend_alpha(
    classifier_oof: np.ndarray,
    rank_oof: np.ndarray,
    labels: np.ndarray,
    *,
    max_fpr: float,
    max_positive_rate: float,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    best_alpha = 0.85
    best_payload: dict[str, Any] = {}
    best_metrics: dict[str, Any] = {}
    best_reward = -1.0

    for alpha in np.linspace(0.55, 0.95, 9):
        blended = alpha * classifier_oof + (1.0 - alpha) * rank_oof
        live_settings, live_metrics = select_live_calibration(
            blended,
            labels,
            max_fpr=max_fpr,
            max_positive_rate=max_positive_rate,
        )
        reward_value = float(live_metrics.get("reward", 0.0) or 0.0)
        if reward_value > best_reward:
            best_reward = reward_value
            best_alpha = float(alpha)
            best_payload = live_settings
            best_metrics = live_metrics
        print(
            f"Blend alpha={alpha:.2f} reward={reward_value:.4f} "
            f"fpr={live_metrics.get('reward_meta', {}).get('fpr', 0.0):.3f}"
        )
    return best_alpha, best_payload, best_metrics


def benchmark_reward_for_model(model_path: Path, labels: np.ndarray, rows: list[dict]) -> float:
    if not model_path.exists():
        return 0.0
    model = Poker44Model(model_path)
    feature_names = model.feature_names
    ensemble = model.models[0]
    x = np.asarray(
        [[float(row.get(name, 0.0)) for name in feature_names] for row in rows],
        dtype=float,
    )
    raw = np.asarray(ensemble.predict_proba(x))[:, 1]
    live = simulate_live_miner_scores(
        raw,
        bias=float(model.score_logit_bias),
        temperature=float(model.score_logit_temperature),
        spread_blend=float(model.live_batch_spread_blend),
        center_blend=float(model.live_batch_center_blend),
        max_positive_rate=float(model.metadata.get("live_max_positive_rate", 0.10) or 0.10),
    )
    rew, _ = reward(live, labels)
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
    parser.add_argument("--deploy", action="store_true", help="Copy to deploy path if reward improves.")
    args = parser.parse_args()

    print("Fetching benchmark examples...")
    feature_dicts, y, metadata, benchmark_state = fetch_training_examples()
    x, feature_names = vectorize(feature_dicts)
    batch_groups = batch_groups_from_metadata(metadata)
    print(
        f"Loaded {len(y)} chunks ({int(y.sum())} bot / {int(len(y) - y.sum())} human), "
        f"{len(feature_names)} robust features across {len(np.unique(batch_groups))} validator batches"
    )

    classifier_oof = batch_cv_classifier_oof(x, y, metadata, seed=args.seed, folds=args.folds)
    classifier_ensemble, classifier_full = train_classifier_stack(
        x,
        y,
        metadata,
        seed=args.seed,
        folds=args.folds,
    )
    rank_ensemble, rank_oof, rank_full = train_batch_rank_stack(
        x,
        y,
        metadata,
        seed=args.seed,
        folds=args.folds,
    )

    alpha, live_settings, live_metrics = select_blend_alpha(
        classifier_oof,
        rank_oof,
        y,
        max_fpr=args.max_fpr,
        max_positive_rate=args.max_positive_rate,
    )
    blended_oof = alpha * classifier_oof + (1.0 - alpha) * rank_oof
    blended_full = alpha * classifier_full + (1.0 - alpha) * rank_full

    diagnostics = {
        "blend_alpha_classifier": float(alpha),
        "blend_alpha_rank": float(1.0 - alpha),
        "classifier_batch_oof_ap": float(average_precision_score(y, classifier_oof)),
        "classifier_batch_oof_auc": float(roc_auc_score(y, classifier_oof)),
        "rank_batch_oof_ap": float(average_precision_score(y, rank_oof)),
        "rank_batch_oof_auc": float(roc_auc_score(y, rank_oof)),
        "hybrid_batch_oof_ap": float(average_precision_score(y, blended_oof)),
        "hybrid_batch_oof_auc": float(roc_auc_score(y, blended_oof)),
        "hybrid_batch_oof_spearman": _within_group_spearman(blended_oof, y, batch_groups),
        "hybrid_full_fit_ap": float(average_precision_score(y, blended_full)),
        "hybrid_full_fit_auc": float(roc_auc_score(y, blended_full)),
        "hybrid_full_fit_batch_spearman": _within_group_spearman(blended_full, y, batch_groups),
        "cv_strategy": "batch_group_kfold_hybrid",
        "batch_count": int(len(np.unique(batch_groups))),
    }
    print(json.dumps({"diagnostics": diagnostics, "live_settings": live_settings, "metrics": live_metrics}, indent=2))

    artifact = {
        "models": [classifier_ensemble, rank_ensemble],
        "model_weights": [float(alpha), float(1.0 - alpha)],
        "feature_names": feature_names,
        "metadata": {
            "model_name": "poker44-hybrid-rank-stack",
            "model_version": MODEL_VERSION,
            "model_weights": [float(alpha), float(1.0 - alpha)],
            "score_remap": live_settings.get("score_remap", {}),
            "live_batch_spread": live_settings.get("live_batch_spread", True),
            "live_batch_spread_blend": live_settings.get("live_batch_spread_blend", 0.85),
            "live_batch_center": live_settings.get("live_batch_center", False),
            "live_batch_center_blend": live_settings.get("live_batch_center_blend", 0.75),
            "live_max_positive_rate": live_settings.get("live_max_positive_rate", 0.10),
            "score_logit_bias": live_settings.get("score_logit_bias", 0.0),
            "score_logit_temperature": live_settings.get("score_logit_temperature", 1.0),
            "metrics": live_metrics,
            "diagnostics": diagnostics,
            "training_objective": "hybrid_classifier_batch_lambdarank",
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
    print(f"Saved model to {args.output}")

    baseline_reward = benchmark_reward_for_model(args.deploy_path, y, feature_dicts)
    new_reward = float(live_metrics.get("reward", 0.0) or 0.0)
    print(f"Benchmark reward | baseline={baseline_reward:.4f} hybrid_oof={new_reward:.4f}")

    if args.deploy and new_reward >= baseline_reward and new_reward >= MIN_DEPLOY_REWARD:
        backup_path = None
        if args.deploy_path.exists():
            backup_path = args.deploy_path.with_name(
                f"{args.deploy_path.stem}.{MODEL_VERSION.replace('.', '_')}_backup.joblib"
            )
            joblib.dump(joblib.load(args.deploy_path), backup_path)
        joblib.dump(artifact, args.deploy_path)
        print(f"Deployed to {args.deploy_path}" + (f"; backup={backup_path}" if backup_path else ""))
    else:
        print(
            "Skipped deploy "
            f"(hybrid={new_reward:.4f}, baseline={baseline_reward:.4f}, "
            f"min={MIN_DEPLOY_REWARD:.2f}, deploy_flag={args.deploy})"
        )


if __name__ == "__main__":
    main()
