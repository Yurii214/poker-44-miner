#!/usr/bin/env python3
"""Train an innovative dual-branch batch-aware Poker44 detector."""

from __future__ import annotations

import argparse
import gc
import json
import os
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
from poker44_ml.calibration import simulate_live_miner_scores, simulate_regime_live_miner_scores
from poker44_ml.innovative_model import (
    DualBranchBatchAwareModel,
    transform_batch_relative_rows,
)
from poker44_ml.live_augmentation import (
    PSEUDO_SOURCE,
    balance_pseudo_examples,
    build_pseudo_labeled_examples,
    merge_training_sets,
)
from poker44_ml.rank_stack import batch_rank_boost
from train_reference_stack import (  # noqa: E402
    DEFAULT_STATE_PATH,
    fetch_training_examples,
    fit_model,
    make_base_models,
    sample_weights,
    select_live_calibration,
    select_regime_calibration,
    vectorize,
)
from poker44_ml.anomaly_branch import HumanIsolationScorer, fuse_hybrid_scores
from poker44_ml.live_regime_tuning import load_regime_tune

DEFAULT_OUTPUT = REPO_ROOT / "models" / "bot_detector_innovative.joblib"
DEFAULT_DEPLOY_PATH = REPO_ROOT / "models" / "bot_detector_v1.joblib"
MODEL_VERSION = "reference-dualbranch-v6-rank-first"
HOLDOUT_SOURCE_DATES = 5
RANK_FIRST = True

TRAIN_PROFILES: dict[str, dict[str, Any]] = {
    "v6": {
        "model_version": "reference-dualbranch-v6-rank-first",
        "training_objective": "dual_branch_rank_first_spearman_v6",
        "max_fpr": 0.02,
        "max_positive_rate": 0.05,
        "spread_blend": 0.85,
        "target_medians": (0.12, 0.16, 0.20),
        "live_augment_default": True,
    },
    "v6.1": {
        "model_version": "reference-dualbranch-v6.1-rank-first",
        "training_objective": "dual_branch_rank_first_spearman_v6.1_stealth",
        "max_fpr": 0.01,
        "max_positive_rate": 0.03,
        "spread_blend": 0.88,
        "target_medians": (0.08, 0.10, 0.12, 0.14),
        "live_augment_default": True,
    },
    "v6.2": {
        "model_version": "reference-dualbranch-v6.2-rank-first",
        "training_objective": "dual_branch_rank_first_spearman_v6.2_balanced",
        "max_fpr": 0.01,
        "max_positive_rate": 0.025,
        "spread_blend": 0.90,
        "target_medians": (0.10, 0.12, 0.14, 0.16),
        "live_augment_default": True,
        "pseudo_balance": True,
        "pseudo_max_per_class": 50,
        "pseudo_min_bot_score": 0.68,
        "pseudo_max_human_score": 0.14,
        "pseudo_weight": 0.30,
        "benchmark_only_selection": True,
    },
    "v6.3": {
        "model_version": "reference-dualbranch-v6.3-rank-first",
        "training_objective": "dual_branch_rank_first_spearman_v6.3_stealth_rank",
        "max_fpr": 0.005,
        "max_positive_rate": 0.02,
        "spread_blend": 0.92,
        "target_medians": (0.14, 0.16, 0.18, 0.20, 0.22),
        "live_augment_default": False,
        "benchmark_only_selection": True,
        "live_score_ceiling": 0.49,
        "center_blends": (0.0,),
        "spread_bounds": (
            (None, None),
            (0.10, 0.42),
            (0.12, 0.45),
            (0.14, 0.48),
        ),
    },
    "v6.4": {
        "model_version": "reference-dualbranch-v6.4-hybrid-regime",
        "training_objective": "dual_branch_hybrid_iso_regime_v6.4",
        "max_fpr": 0.005,
        "max_positive_rate": 0.02,
        "spread_blend": 0.94,
        "live_augment_default": False,
        "benchmark_only_selection": True,
        "live_score_ceiling": 0.49,
        "hybrid_isolation": True,
        "regime_calibration": True,
    },
    "v6.5": {
        "model_version": "reference-dualbranch-v6.5-live-tuned-regime",
        "training_objective": "dual_branch_hybrid_live_tuned_regime_v6.5",
        "max_fpr": 0.005,
        "max_positive_rate": 0.02,
        "spread_blend": 0.94,
        "live_augment_default": False,
        "benchmark_only_selection": True,
        "live_score_ceiling": 0.49,
        "hybrid_isolation": True,
        "regime_calibration": True,
        "live_regime_tune_file": "models/live_regime_tune.json",
    },
    "v7": {
        "model_version": "reference-dualbranch-v7-benchmark-v112",
        "training_objective": "dual_branch_v7_benchmark_v112_generalize",
        "max_fpr": 0.005,
        "max_positive_rate": 0.02,
        "spread_blend": 0.94,
        "live_augment_default": False,
        "benchmark_only_selection": True,
        "live_score_ceiling": 0.49,
        "hybrid_isolation": True,
        "regime_calibration": True,
        "live_regime_tune_file": "models/live_regime_tune.json",
        "holdout_dates": 7,
        "center_blends": (0.0,),
        "spread_bounds": (
            (None, None),
            (0.10, 0.42),
            (0.12, 0.45),
            (0.14, 0.48),
        ),
    },
    "v8": {
        "model_version": "reference-dualbranch-v8-reward-recall",
        "training_objective": "dual_branch_v8_reward_recall_regime",
        "reward_first": True,
        "max_fpr": 0.005,
        "max_positive_rate": 0.05,
        "live_human_max_positive_rate": 0.0,
        "live_bot_max_positive_rate": 0.18,
        "live_human_score_ceiling": 0.49,
        "live_bot_score_ceiling": 0.58,
        "spread_blend": 0.94,
        "live_augment_default": False,
        "benchmark_only_selection": True,
        "live_score_ceiling": 0.49,
        "hybrid_isolation": True,
        "regime_calibration": True,
        "holdout_dates": 7,
        "min_bot_recall": 0.25,
        "center_blends": (0.0,),
        "spread_bounds": (
            (None, None),
            (0.10, 0.42),
            (0.12, 0.45),
            (0.14, 0.48),
        ),
    },
    "v9": {
        "model_version": "reference-dualbranch-v9-chunk-regime",
        "training_objective": "dual_branch_v9_chunk_regime_live80",
        "reward_first": True,
        "max_fpr": 0.005,
        "max_positive_rate": 0.05,
        "live_human_max_positive_rate": 0.0,
        "live_bot_max_positive_rate": 0.15,
        "live_human_score_ceiling": 0.49,
        "live_bot_score_ceiling": 0.58,
        "spread_blend": 0.94,
        "live_augment_default": False,
        "benchmark_only_selection": True,
        "live_score_ceiling": 0.49,
        "hybrid_isolation": True,
        "regime_calibration": True,
        "chunk_regime": True,
        "live_batch_size": 80,
        "holdout_dates": 7,
        "min_bot_recall": 0.30,
        "center_blends": (0.0,),
        "spread_bounds": (
            (None, None),
            (0.10, 0.42),
            (0.12, 0.45),
            (0.14, 0.48),
        ),
    },
    "v10": {
        "model_version": "reference-dualbranch-v10-holdout-recall",
        "training_objective": "dual_branch_v10_holdout_gated_recall",
        "reward_first": True,
        "holdout_gated_selection": True,
        "extended_regime_grid": True,
        "max_fpr": 0.005,
        "max_positive_rate": 0.05,
        "live_human_max_positive_rate": 0.0,
        "live_bot_max_positive_rate": 0.22,
        "live_human_score_ceiling": 0.49,
        "live_bot_score_ceiling": 0.62,
        "spread_blend": 0.94,
        "live_augment_default": False,
        "benchmark_only_selection": True,
        "live_score_ceiling": 0.49,
        "hybrid_isolation": True,
        "regime_calibration": True,
        "chunk_regime": True,
        "live_batch_size": 80,
        "holdout_dates": 7,
        "min_bot_recall": 0.35,
        "center_blends": (0.0,),
        "spread_bounds": (
            (None, None),
            (0.10, 0.42),
            (0.12, 0.45),
            (0.14, 0.48),
        ),
    },
    "v11": {
        "model_version": "reference-dualbranch-v11-rank-regime",
        "training_objective": "dual_branch_v11_rank_regime_dual_fpr",
        "reward_first": True,
        "holdout_gated_selection": True,
        "extended_regime_grid": True,
        "dual_fpr_selection": True,
        "live_regime_mode": "rank",
        "live_human_fraction": 0.35,
        "max_fpr": 0.005,
        "max_positive_rate": 0.05,
        "live_human_max_positive_rate": 0.0,
        "live_bot_max_positive_rate": 0.20,
        "live_human_score_ceiling": 0.49,
        "live_bot_score_ceiling": 0.62,
        "spread_blend": 0.94,
        "live_augment_default": False,
        "benchmark_only_selection": True,
        "live_score_ceiling": 0.49,
        "hybrid_isolation": True,
        "regime_calibration": True,
        "chunk_regime": True,
        "live_batch_size": 80,
        "holdout_dates": 7,
        "min_bot_recall": 0.38,
        "center_blends": (0.0,),
        "spread_bounds": (
            (None, None),
            (0.10, 0.42),
            (0.12, 0.45),
            (0.14, 0.48),
        ),
    },
    "v12": {
        "model_version": "reference-dualbranch-v12-r1-recall",
        "training_objective": "dual_branch_v12_absolute_regime_live_replay",
        "reward_first": True,
        "holdout_gated_selection": True,
        "extended_regime_grid": True,
        "dual_fpr_selection": True,
        "max_full_fpr": 0.015,
        "live_regime_mode": "absolute",
        "max_fpr": 0.005,
        "max_positive_rate": 0.05,
        "live_human_max_positive_rate": 0.0,
        "live_bot_max_positive_rate": 0.24,
        "live_human_score_ceiling": 0.49,
        "live_bot_score_ceiling": 0.66,
        "spread_blend": 0.94,
        "live_augment_default": False,
        "benchmark_only_selection": True,
        "live_score_ceiling": 0.49,
        "hybrid_isolation": True,
        "regime_calibration": True,
        "chunk_regime": True,
        "live_batch_size": 80,
        "holdout_dates": 8,
        "min_bot_recall": 0.45,
        "center_blends": (0.0,),
        "spread_bounds": (
            (None, None),
            (0.10, 0.42),
            (0.12, 0.45),
            (0.14, 0.48),
        ),
    },
}


def benchmark_holdout_mask(
    metadata: list[dict[str, Any]],
    *,
    holdout_dates: int = HOLDOUT_SOURCE_DATES,
) -> np.ndarray:
    """Date holdout on benchmark rows only (exclude live pseudo-label rows)."""
    bench_meta = [row for row in metadata if row.get("source") != PSEUDO_SOURCE]
    dates = sorted({str(row["source_date"]) for row in bench_meta})
    if len(dates) <= 1:
        return np.zeros(len(metadata), dtype=bool)
    keep = max(1, min(int(holdout_dates), len(dates) // 3 or 1))
    holdout = set(dates[-keep:])
    return np.asarray(
        [
            row.get("source") != PSEUDO_SOURCE and str(row["source_date"]) in holdout
            for row in metadata
        ],
        dtype=bool,
    )


def benchmark_rows_mask(metadata: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([row.get("source") != PSEUDO_SOURCE for row in metadata], dtype=bool)


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
    regime_overrides: dict[str, Any] | None = None,
    regime_scores: np.ndarray | None = None,
    live_batch_size: int = 80,
) -> np.ndarray:
    regime_overrides = regime_overrides or {}
    human_rate = settings.get(
        "live_human_max_positive_rate",
        regime_overrides.get("live_human_max_positive_rate"),
    )
    bot_rate = settings.get(
        "live_bot_max_positive_rate",
        regime_overrides.get("live_bot_max_positive_rate"),
    )
    human_ceiling = settings.get(
        "live_human_score_ceiling",
        regime_overrides.get("live_human_score_ceiling", settings.get("live_score_ceiling", 0.49)),
    )
    bot_ceiling = settings.get(
        "live_bot_score_ceiling",
        regime_overrides.get("live_bot_score_ceiling", 0.58),
    )
    chunk_regime = bool(settings.get("live_chunk_regime_enabled", True))
    if chunk_regime and bool(settings.get("live_regime_enabled", False)):
        return simulate_regime_live_miner_scores(
            scores,
            regime_threshold=float(settings.get("live_regime_threshold", 0.35) or 0.35),
            human_spread=tuple(settings.get("live_human_spread") or (0.04, 0.16)),
            bot_spread=tuple(settings.get("live_bot_spread") or (0.22, 0.48)),
            spread_blend=float(settings.get("live_batch_spread_blend", 0.70) or 0.70),
            batch_size=live_batch_size,
            max_positive_rate=max_positive_rate,
            human_max_positive_rate=float(human_rate) if human_rate is not None else None,
            bot_max_positive_rate=float(bot_rate) if bot_rate is not None else None,
            temperature=float(settings.get("score_logit_temperature", 0.55) or 0.55),
            hard_ceiling=settings.get("live_score_ceiling", 0.49),
            human_hard_ceiling=float(human_ceiling) if human_ceiling is not None else None,
            bot_hard_ceiling=float(bot_ceiling) if bot_ceiling is not None else None,
            regime_scores=regime_scores if regime_scores is not None else scores,
            chunk_regime=True,
            regime_mode=str(settings.get("live_regime_mode", "absolute") or "absolute"),
            human_fraction=float(settings.get("live_human_fraction", 0.35) or 0.35),
            apply_positive_cap=False,
            apply_miner_cap_replay=True,
        )
    return simulate_live_miner_scores(
        scores,
        bias=float(settings.get("score_logit_bias", 2.2) or 2.2),
        temperature=float(settings.get("score_logit_temperature", 0.65) or 0.65),
        spread_blend=float(settings.get("live_batch_spread_blend", 0.70) or 0.70),
        center_blend=float(settings.get("live_batch_center_blend", 0.0) or 0.0),
        spread_low=settings.get("live_batch_spread_low"),
        spread_high=settings.get("live_batch_spread_high"),
        max_positive_rate=max_positive_rate,
        human_max_positive_rate=float(human_rate) if human_rate is not None else None,
        bot_max_positive_rate=float(bot_rate) if bot_rate is not None else None,
        logit_mode=str(settings.get("live_logit_mode", "adaptive") or "adaptive"),
        target_median=float(settings.get("live_logit_target_median", 0.24) or 0.24),
        hard_ceiling=settings.get("live_score_ceiling", 0.49),
        human_hard_ceiling=float(human_ceiling) if human_ceiling is not None else None,
        bot_hard_ceiling=float(bot_ceiling) if bot_ceiling is not None else None,
        regime_enabled=bool(settings.get("live_regime_enabled", False)),
        regime_threshold=float(settings.get("live_regime_threshold", 0.35) or 0.35),
        human_spread=tuple(settings.get("live_human_spread") or (0.04, 0.16)),
        bot_spread=tuple(settings.get("live_bot_spread") or (0.22, 0.48)),
    )


def _live_selection_tuple(
    live_scores: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray,
    holdout_mask: np.ndarray | None = None,
    rank_first: bool = RANK_FIRST,
) -> tuple[float, float, float, float]:
    from train_reference_stack import _within_group_spearman

    rew, meta = reward(live_scores, y)
    spearman_full = float(_within_group_spearman(live_scores, y, groups).get("mean", 0.0))
    if holdout_mask is not None and bool(holdout_mask.any()):
        spearman = float(
            _within_group_spearman(
                live_scores[holdout_mask],
                y[holdout_mask],
                groups[holdout_mask],
            ).get("mean", 0.0)
        )
    else:
        spearman = spearman_full
    human_mask = y == 0
    cls_penalty = (
        float((live_scores[human_mask] >= 0.5).mean()) if human_mask.any() else 0.0
    )
    fpr = float(meta.get("fpr", 1.0) or 1.0)
    bot_recall = float(meta.get("bot_recall", 0.0) or 0.0)
    if rank_first:
        return (spearman, -cls_penalty, -fpr, bot_recall)
    holdout_rew = float(rew)
    if holdout_mask is not None and bool(holdout_mask.any()):
        holdout_rew, _ = reward(live_scores[holdout_mask], y[holdout_mask])
    return (holdout_rew, float(rew), bot_recall, spearman_full)


def sweep_blend(
    abs_oof: np.ndarray,
    rel_oof: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray,
    max_fpr: float,
    max_positive_rate: float,
    holdout_mask: np.ndarray | None = None,
    calibration_kwargs: dict[str, Any] | None = None,
    iso_scores: np.ndarray | None = None,
    use_regime: bool = False,
    fixed_regime: dict[str, Any] | None = None,
    regime_overrides: dict[str, Any] | None = None,
    reward_first: bool = RANK_FIRST,
    min_bot_recall: float = 0.25,
    supervised_oof: np.ndarray | None = None,
    chunk_regime: bool = True,
    live_batch_size: int = 80,
    selection_holdout_mask: np.ndarray | None = None,
    extended_regime_grid: bool = False,
    regime_mode: str = "absolute",
    human_fraction: float = 0.35,
    dual_fpr: bool = False,
    max_full_fpr: float | None = None,
) -> tuple[float, dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    calibration_kwargs = calibration_kwargs or {}
    regime_overrides = regime_overrides or {}
    best_alpha = 0.20 if reward_first is False else 0.75
    best_metrics: dict[str, Any] = {}
    best_settings: dict[str, Any] = {}
    best_scores = abs_oof
    best_supervised = abs_oof
    best_selection = (-1.0, -1.0, -1.0, -1.0)

    alpha_values = np.linspace(0.10, 0.30, 9) if reward_first is False else np.linspace(0.15, 0.55, 9)
    human_rate = regime_overrides.get("live_human_max_positive_rate")
    bot_rate = regime_overrides.get("live_bot_max_positive_rate")
    human_ceiling = regime_overrides.get(
        "live_human_score_ceiling",
        calibration_kwargs.get("hard_ceiling", 0.49),
    )
    bot_ceiling = regime_overrides.get("live_bot_score_ceiling", 0.58)
    for alpha in alpha_values:
        supervised = alpha * abs_oof + (1.0 - alpha) * rel_oof
        scores = supervised
        if iso_scores is not None:
            scores = fuse_hybrid_scores(scores, iso_scores, mode="max")
        if use_regime and fixed_regime is not None:
            from poker44_ml.calibration import simulate_regime_live_miner_scores

            threshold = float(fixed_regime["regime_threshold"])
            human_spread = tuple(fixed_regime["human_spread"])
            bot_spread = tuple(fixed_regime["bot_spread"])
            hard_ceiling = calibration_kwargs.get("hard_ceiling", 0.49)
            live = simulate_regime_live_miner_scores(
                scores,
                regime_threshold=threshold,
                human_spread=human_spread,
                bot_spread=bot_spread,
                spread_blend=float(calibration_kwargs.get("spread_blend", 0.94)),
                batch_size=live_batch_size,
                max_positive_rate=max_positive_rate,
                human_max_positive_rate=float(human_rate) if human_rate is not None else None,
                bot_max_positive_rate=float(bot_rate) if bot_rate is not None else None,
                hard_ceiling=hard_ceiling,
                human_hard_ceiling=float(human_ceiling) if human_ceiling is not None else None,
                bot_hard_ceiling=float(bot_ceiling) if bot_ceiling is not None else None,
                regime_scores=supervised,
                chunk_regime=chunk_regime,
                regime_mode=regime_mode,
                human_fraction=human_fraction,
                apply_positive_cap=False,
                apply_miner_cap_replay=True,
            )
            settings = {
                "live_batch_spread": True,
                "live_batch_spread_blend": float(calibration_kwargs.get("spread_blend", 0.94)),
                "live_batch_center": False,
                "live_max_positive_rate": float(max_positive_rate),
                "live_human_max_positive_rate": float(human_rate) if human_rate is not None else 0.0,
                "live_bot_max_positive_rate": float(bot_rate) if bot_rate is not None else max_positive_rate,
                "live_logit_mode": "regime",
                "live_regime_enabled": True,
                "live_chunk_regime_enabled": bool(chunk_regime),
                "live_regime_threshold": threshold,
                "live_human_spread": list(human_spread),
                "live_bot_spread": list(bot_spread),
                "live_score_ceiling": float(human_ceiling) if human_ceiling is not None else 0.49,
                "live_human_score_ceiling": float(human_ceiling) if human_ceiling is not None else 0.49,
                "live_bot_score_ceiling": float(bot_ceiling) if bot_ceiling is not None else 0.58,
                "score_logit_temperature": 0.55,
                "hybrid_fusion": "max",
            }
            rew, meta = reward(live, y)
            from train_reference_stack import _within_group_spearman

            batch_spearman = float(_within_group_spearman(live, y, groups).get("mean", 0.0))
            metrics = {
                "reward": float(rew),
                "reward_meta": meta,
                "batch_spearman": batch_spearman,
                "score_max": float(live.max()),
                "above_05": int((live >= 0.5).sum()),
                "calibration_choice": {
                    "regime_threshold": threshold,
                    "human_spread": list(human_spread),
                    "bot_spread": list(bot_spread),
                    "method": "live_tune_fixed",
                },
            }
        elif use_regime:
            settings, metrics = select_regime_calibration(
                scores,
                y,
                max_fpr=max_fpr,
                max_positive_rate=max_positive_rate,
                human_max_positive_rate=float(human_rate) if human_rate is not None else None,
                bot_max_positive_rate=float(bot_rate) if bot_rate is not None else None,
                groups=groups,
                batch_size=live_batch_size,
                spread_blend=float(calibration_kwargs.get("spread_blend", 0.92)),
                spearman_mask=calibration_kwargs.get("spearman_mask"),
                hard_ceiling=calibration_kwargs.get("hard_ceiling", 0.49),
                human_hard_ceiling=float(human_ceiling) if human_ceiling is not None else None,
                bot_hard_ceiling=float(bot_ceiling) if bot_ceiling is not None else None,
                reward_first=reward_first,
                min_bot_recall=min_bot_recall,
                regime_scores=supervised,
                chunk_regime=chunk_regime,
                holdout_mask=selection_holdout_mask,
                extended_grid=extended_regime_grid,
                regime_mode=regime_mode,
                human_fraction=human_fraction,
                dual_fpr=dual_fpr,
                max_full_fpr=max_full_fpr,
            )
        else:
            settings, metrics = select_live_calibration(
                scores,
                y,
                max_fpr=max_fpr,
                max_positive_rate=max_positive_rate,
                groups=groups,
                rank_first=not reward_first,
                **calibration_kwargs,
            )
        live = _simulate_live_scores(
            scores,
            settings,
            max_positive_rate=max_positive_rate,
            regime_overrides=regime_overrides,
            regime_scores=supervised,
            live_batch_size=live_batch_size,
        )
        selection = _live_selection_tuple(
            live,
            y,
            groups=groups,
            holdout_mask=holdout_mask,
            rank_first=not reward_first,
        )
        batch_spearman = selection[0] if not reward_first else selection[3]
        print(
            f"alpha={alpha:.2f} spearman={selection[0]:.4f} cls_pen={-selection[1]:.4f} "
            f"fpr={-selection[2]:.4f} bot_recall={selection[3]:.4f} "
            f"reward={metrics.get('reward', 0.0):.4f} batch_spearman={batch_spearman:.4f}"
        )
        if selection > best_selection:
            best_selection = selection
            best_alpha = float(alpha)
            best_settings = settings
            best_metrics = metrics
            best_metrics["batch_spearman"] = batch_spearman
            best_metrics["holdout_spearman"] = selection[0] if not reward_first else batch_spearman
            best_metrics["classification_penalty"] = float(-selection[1]) if not reward_first else 0.0
            best_metrics["holdout_reward"] = float(metrics.get("reward", 0.0))
            best_metrics["bot_recall"] = selection[3] if not reward_first else selection[2]
            best_scores = scores
            best_supervised = supervised
    return best_alpha, best_settings, best_metrics, best_scores, best_supervised


def batch_groups_from_metadata(metadata: list[dict[str, Any]]) -> np.ndarray:
    chunk_ids = [str(row["chunk_id"]) for row in metadata]
    lookup = {chunk_id: idx for idx, chunk_id in enumerate(sorted(set(chunk_ids)))}
    return np.asarray([lookup[cid] for cid in chunk_ids], dtype=int)


def _predict_pos(model: Any, x: np.ndarray) -> np.ndarray:
    import lightgbm as lgb
    if isinstance(model, lgb.LGBMRanker):
        raw = np.asarray(model.predict(x), dtype=float)
        raw = raw - raw.min()
        rng = raw.max()
        if rng > 1e-9:
            raw = raw / rng
        return np.clip(raw, 0.0, 1.0)
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
    weights: np.ndarray | None = None,
    n_jobs: int = 1,
) -> tuple[list[Any], np.ndarray]:
    weights = np.asarray(weights if weights is not None else sample_weights(y), dtype=float)
    specs = make_base_models(seed, n_jobs=n_jobs)
    cv = GroupKFold(n_splits=max(2, min(folds, len(np.unique(groups)))))
    oof = np.zeros((len(y), len(specs)), dtype=float)

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(x, y, groups=groups), start=1):
        x_train = x[train_idx]
        x_valid = x[valid_idx]
        if transform_relative:
            x_train = transform_batch_relative_rows(x_train, groups[train_idx])
            x_valid = transform_batch_relative_rows(x_valid, groups[valid_idx])
        for model_idx, (_, model) in enumerate(specs):
            fitted = fit_model(
                clone(model),
                x_train,
                y[train_idx],
                weights[train_idx],
                groups[train_idx],
            )
            oof[valid_idx, model_idx] = _predict_pos(fitted, x_valid)
        branch = "relative" if transform_relative else "absolute"
        print(f"Finished {branch} fold {fold_idx}/{cv.get_n_splits()}")

    final_models: list[Any] = []
    x_full = transform_batch_relative_rows(x, groups) if transform_relative else x
    for _, model in specs:
        final_models.append(fit_model(clone(model), x_full, y, weights, groups))
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
    regime_scores: np.ndarray | None = None,
    live_batch_size: int = 80,
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
            regime_scores=regime_scores,
            live_batch_size=live_batch_size,
        )
        selection = _live_selection_tuple(
            live,
            y,
            groups=groups,
            holdout_mask=holdout_mask,
            rank_first=RANK_FIRST,
        )
        print(
            f"rank_boost={boost:.2f} spearman={selection[0]:.4f} cls_pen={-selection[1]:.4f} "
            f"fpr={-selection[2]:.4f} bot_recall={selection[3]:.4f}"
        )
        if selection > best_selection:
            best_boost = float(boost)
            best_selection = selection
            _, meta = reward(live, y)
            best_metrics = {
                "reward": float(reward(live, y)[0]),
                "reward_meta": meta,
                "batch_spearman": selection[0],
                "holdout_spearman": selection[0],
                "classification_penalty": float(-selection[1]),
                "holdout_reward": float(reward(live, y)[0]),
                "bot_recall": selection[3],
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
    supervised = np.zeros(len(x), dtype=float)
    for group_id in np.unique(groups):
        mask = groups == group_id
        batch_x = x[mask]
        batch_features = [feature_dicts[int(idx)] for idx in np.where(mask)[0]]
        if hasattr(model, "predict_chunk_scores"):
            batch_raw = np.asarray(
                model.predict_chunk_scores(
                    [[] for _ in range(int(mask.sum()))],
                    feature_rows=[list(row) for row in batch_x],
                ),
                dtype=float,
            )
            raw[mask] = batch_raw
            if hasattr(model, "predict_supervised_raw_scores"):
                supervised[mask] = np.asarray(
                    model.predict_supervised_raw_scores(
                        [[] for _ in range(int(mask.sum()))],
                        feature_rows=[list(row) for row in batch_x],
                    ),
                    dtype=float,
                )
            else:
                supervised[mask] = batch_raw
        else:
            batch_proba = np.asarray(model.predict_proba(batch_x))[:, 1]
            raw[mask] = batch_proba
            supervised[mask] = batch_proba
        if boost > 0.0:
            boosted = batch_rank_boost(
                batch_features,
                raw[mask].tolist(),
                blend=boost,
            )
            raw[mask] = np.asarray(boosted, dtype=float)
            supervised[mask] = np.maximum(supervised[mask], raw[mask])
    max_rate = float(md.get("live_max_positive_rate", 0.10) or 0.10)
    live_batch_size = int(md.get("live_batch_size", 80) or 80)
    live = _simulate_live_scores(
        raw,
        md,
        max_positive_rate=max_rate,
        regime_scores=supervised,
        live_batch_size=live_batch_size,
    )
    rew, _ = reward(live, y)
    return float(rew)


def evaluate_artifact_spearman(
    artifact: dict[str, Any],
    feature_dicts: list[dict[str, float]],
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
) -> float:
    from train_reference_stack import _within_group_spearman

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
    return float(_within_group_spearman(live, y, groups).get("mean", 0.0))


def evaluate_artifact_spearman_masked(
    artifact: dict[str, Any],
    feature_dicts: list[dict[str, float]],
    y: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
) -> float:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0.0
    return evaluate_artifact_spearman(
        artifact,
        [feature_dicts[int(i)] for i in idx],
        y[idx],
        groups=groups[idx],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--deploy-path", type=Path, default=DEFAULT_DEPLOY_PATH)
    parser.add_argument(
        "--profile",
        choices=sorted(TRAIN_PROFILES),
        default="v6",
        help="Training preset (v6.1 = stealth rankQ: lower median, 3%% positive cap).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-fpr", type=float, default=None)
    parser.add_argument("--max-positive-rate", type=float, default=None)
    parser.add_argument(
        "--min-source-date",
        type=str,
        default="2026-06-13",
        help="Only use benchmark releases on or after this sourceDate (v1.12+).",
    )
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument(
        "--force-deploy",
        action="store_true",
        help="Deploy even if benchmark reward regresses (use with spearman gate).",
    )
    parser.add_argument("--live-augment", action="store_true")
    parser.add_argument(
        "--no-live-augment",
        action="store_true",
        help="Disable live pseudo-label augmentation.",
    )
    parser.add_argument(
        "--live-store-dir",
        type=Path,
        default=REPO_ROOT / "models" / "live_chunks",
    )
    parser.add_argument("--pseudo-weight", type=float, default=0.45)
    parser.add_argument("--pseudo-max-examples", type=int, default=400)
    parser.add_argument(
        "--pseudo-max-batches",
        type=int,
        default=120,
        help="Max logged validator batches to scan (streamed; not loaded all at once).",
    )
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    profile = TRAIN_PROFILES[args.profile]
    global RANK_FIRST
    reward_first_flag = bool(profile.get("reward_first", False))
    RANK_FIRST = not reward_first_flag
    if args.max_fpr is None:
        args.max_fpr = float(profile["max_fpr"])
    if args.max_positive_rate is None:
        args.max_positive_rate = float(profile["max_positive_rate"])
    use_live_augment = bool(profile.get("live_augment_default", False))
    if args.live_augment:
        use_live_augment = True
    if args.no_live_augment:
        use_live_augment = False
    calibration_kwargs = {
        "target_medians": tuple(profile.get("target_medians", (0.14, 0.16, 0.18))),
        "spread_blend": float(profile["spread_blend"]),
    }
    if "center_blends" in profile:
        calibration_kwargs["center_blends"] = tuple(profile["center_blends"])
    if "spread_bounds" in profile:
        calibration_kwargs["spread_bounds"] = tuple(profile["spread_bounds"])
    if "live_score_ceiling" in profile:
        calibration_kwargs["hard_ceiling"] = float(profile["live_score_ceiling"])
    regime_overrides = {
        key: profile[key]
        for key in (
            "live_human_max_positive_rate",
            "live_bot_max_positive_rate",
            "live_human_score_ceiling",
            "live_bot_score_ceiling",
        )
        if key in profile
    }
    min_bot_recall = float(profile.get("min_bot_recall", 0.25))
    chunk_regime = bool(profile.get("chunk_regime", True))
    live_batch_size = int(profile.get("live_batch_size", 80))
    extended_regime_grid = bool(profile.get("extended_regime_grid", False))
    regime_mode = str(profile.get("live_regime_mode", "absolute") or "absolute")
    human_fraction = float(profile.get("live_human_fraction", 0.35) or 0.35)
    dual_fpr = bool(profile.get("dual_fpr_selection", False))
    max_full_fpr = profile.get("max_full_fpr")
    max_full_fpr = float(max_full_fpr) if max_full_fpr is not None else None
    selection_holdout_mask: np.ndarray | None = None
    spearman_eval_mask: np.ndarray | None = None
    pseudo_weight = float(args.pseudo_weight)
    if profile.get("benchmark_only_selection"):
        selection_holdout_mask = None  # set after metadata load
        spearman_eval_mask = None
    training_objective = str(profile["training_objective"])
    model_version = str(profile["model_version"])
    training_weights: np.ndarray | None = None
    fixed_regime: dict[str, Any] | None = None
    if profile.get("live_regime_tune_file"):
        tune_path = REPO_ROOT / str(profile["live_regime_tune_file"])
        if not tune_path.is_file():
            raise FileNotFoundError(
                f"Missing live regime tune file: {tune_path}. "
                "Run scripts/tune_regime_from_live_chunks.py first."
            )
        fixed_regime = load_regime_tune(tune_path)
        print(
            "Using live-tuned regime | "
            f"threshold={fixed_regime['regime_threshold']:.4f} "
            f"human={fixed_regime['human_spread']} bot={fixed_regime['bot_spread']}"
        )

    print("Fetching benchmark examples...")
    feature_dicts, y, metadata, benchmark_state = fetch_training_examples(
        min_source_date=args.min_source_date or None,
    )
    if use_live_augment:
        pseudo_kwargs: dict[str, Any] = {
            "store_dir": args.live_store_dir,
            "model_path": args.deploy_path if args.deploy_path.exists() else DEFAULT_DEPLOY_PATH,
            "max_examples": args.pseudo_max_examples,
            "max_batches": args.pseudo_max_batches,
        }
        if "pseudo_min_bot_score" in profile:
            pseudo_kwargs["min_bot_score"] = float(profile["pseudo_min_bot_score"])
        if "pseudo_max_human_score" in profile:
            pseudo_kwargs["max_human_score"] = float(profile["pseudo_max_human_score"])
        pseudo_features, pseudo_labels, pseudo_metadata = build_pseudo_labeled_examples(
            **pseudo_kwargs,
        )
        if profile.get("pseudo_balance") and len(pseudo_labels):
            pseudo_features, pseudo_labels, pseudo_metadata = balance_pseudo_examples(
                pseudo_features,
                pseudo_labels,
                pseudo_metadata,
                max_per_class=int(profile.get("pseudo_max_per_class", 50)),
                seed=args.seed,
            )
        if "pseudo_weight" in profile:
            pseudo_weight = float(profile["pseudo_weight"])
        gc.collect()
        print(
            f"Live pseudo-labels | examples={len(pseudo_labels)} "
            f"bot={int(pseudo_labels.sum()) if len(pseudo_labels) else 0} "
            f"human={int(len(pseudo_labels) - pseudo_labels.sum()) if len(pseudo_labels) else 0}"
        )
        if len(pseudo_labels):
            feature_dicts, y, metadata, training_weights = merge_training_sets(
                feature_dicts,
                y,
                metadata,
                pseudo_features,
                pseudo_labels,
                pseudo_metadata,
                pseudo_weight=pseudo_weight,
            )
            benchmark_state["live_pseudo_examples"] = int(len(pseudo_labels))
            model_version = f"{model_version}-augmented"
        else:
            print("No high-confidence live pseudo labels yet; training on benchmark only.")

    x, feature_names = vectorize(feature_dicts)
    groups = batch_groups_from_metadata(metadata)
    if profile.get("benchmark_only_selection"):
        holdout_dates = int(profile.get("holdout_dates", HOLDOUT_SOURCE_DATES))
        selection_holdout_mask = benchmark_holdout_mask(metadata, holdout_dates=holdout_dates)
        spearman_eval_mask = benchmark_rows_mask(metadata)
        calibration_kwargs["spearman_mask"] = spearman_eval_mask
        holdout_mask = selection_holdout_mask
    else:
        holdout_mask = holdout_mask_from_metadata(metadata)
    holdout_count = int(holdout_mask.sum())
    bench_count = int(spearman_eval_mask.sum()) if spearman_eval_mask is not None else len(y)
    print(
        f"Loaded {len(y)} samples | features={len(feature_names)} | "
        f"groups={len(np.unique(groups))} | holdout={holdout_count} | benchmark={bench_count}"
    )

    abs_models, abs_oof = train_branch_oof(
        x,
        y,
        groups,
        seed=args.seed,
        folds=args.folds,
        transform_relative=False,
        weights=training_weights,
        n_jobs=args.n_jobs,
    )
    gc.collect()
    rel_models, rel_oof = train_branch_oof(
        x,
        y,
        groups,
        seed=args.seed + 11,
        folds=args.folds,
        transform_relative=True,
        weights=training_weights,
        n_jobs=args.n_jobs,
    )
    gc.collect()

    iso_scorer: HumanIsolationScorer | None = None
    iso_scores: np.ndarray | None = None
    if profile.get("hybrid_isolation"):
        print("Fitting human-only Isolation Forest anomaly branch...")
        iso_scorer = HumanIsolationScorer.fit(
            x,
            y,
            feature_names,
            seed=args.seed,
        )
        iso_scores = iso_scorer.predict_bot_scores(x)
        print(
            f"Anomaly branch | human_rows={int((y == 0).sum())} "
            f"iso_mean={float(iso_scores.mean()):.4f} iso_max={float(iso_scores.max()):.4f}"
        )

    alpha, live_settings, live_metrics, blended_oof, supervised_oof = sweep_blend(
        abs_oof,
        rel_oof,
        y,
        groups=groups,
        max_fpr=args.max_fpr,
        max_positive_rate=args.max_positive_rate,
        holdout_mask=selection_holdout_mask if selection_holdout_mask is not None else holdout_mask,
        calibration_kwargs=calibration_kwargs,
        iso_scores=iso_scores,
        use_regime=bool(profile.get("regime_calibration", False)),
        fixed_regime=fixed_regime,
        regime_overrides=regime_overrides,
        reward_first=reward_first_flag,
        min_bot_recall=min_bot_recall,
        chunk_regime=chunk_regime,
        live_batch_size=live_batch_size,
        selection_holdout_mask=selection_holdout_mask,
        extended_regime_grid=extended_regime_grid,
        regime_mode=regime_mode,
        human_fraction=human_fraction,
        dual_fpr=dual_fpr,
        max_full_fpr=max_full_fpr,
    )

    rank_boost, rank_metrics = sweep_rank_boost(
        feature_dicts,
        blended_oof,
        y,
        groups=groups,
        live_settings=live_settings,
        max_positive_rate=args.max_positive_rate,
        holdout_mask=selection_holdout_mask if selection_holdout_mask is not None else holdout_mask,
        regime_scores=supervised_oof,
        live_batch_size=live_batch_size,
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
        "anomaly_scorer": iso_scorer.to_dict() if iso_scorer is not None else None,
        "metadata": {
            "model_name": "poker44-innovative-dual-branch",
            "model_version": model_version,
            "model_weights": [1.0],
            **live_settings,
            "score_remap": {},
            "metrics": live_metrics,
            "diagnostics": diagnostics,
            "training_objective": training_objective,
        },
        "metrics": live_metrics,
        "training_samples": len(y),
        "metadata_rows": metadata,
        "benchmark_state": benchmark_state,
        "model_version": model_version,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    args.state_file.write_text(json.dumps(benchmark_state, indent=2) + "\n")
    print(json.dumps({"diagnostics": diagnostics, "live_metrics": live_metrics}, indent=2))
    print(f"Saved model to {args.output}")

    baseline_reward = 0.0
    baseline_spearman = 0.0
    baseline_holdout_spearman = 0.0
    baseline_artifact: dict[str, Any] | None = None
    bench_mask = benchmark_rows_mask(metadata)
    eval_holdout_mask = (
        selection_holdout_mask
        if selection_holdout_mask is not None and selection_holdout_mask.any()
        else holdout_mask
    )
    if args.deploy_path.exists():
        baseline_artifact = joblib.load(args.deploy_path)
        baseline_reward = evaluate_artifact_reward(
            baseline_artifact,
            feature_dicts,
            y,
            groups=groups,
        )
        baseline_spearman = evaluate_artifact_spearman_masked(
            baseline_artifact,
            feature_dicts,
            y,
            groups,
            bench_mask,
        )
        baseline_holdout_spearman = evaluate_artifact_spearman_masked(
            baseline_artifact,
            feature_dicts,
            y,
            groups,
            eval_holdout_mask,
        )
        del baseline_artifact
        gc.collect()
    new_reward = evaluate_artifact_reward(
        artifact,
        feature_dicts,
        y,
        groups=groups,
    )
    new_spearman = evaluate_artifact_spearman_masked(
        artifact,
        feature_dicts,
        y,
        groups,
        bench_mask,
    )
    new_holdout_spearman = evaluate_artifact_spearman_masked(
        artifact,
        feature_dicts,
        y,
        groups,
        eval_holdout_mask,
    )
    print(
        f"Benchmark comparison | baseline_reward={baseline_reward:.4f} new_reward={new_reward:.4f} "
        f"baseline_spearman={baseline_spearman:.4f} new_spearman={new_spearman:.4f} "
        f"holdout_spearman={new_holdout_spearman:.4f} baseline_holdout={baseline_holdout_spearman:.4f}"
    )

    reward_ok = new_reward >= baseline_reward - 0.005
    spearman_ok = new_holdout_spearman >= baseline_holdout_spearman - 0.005
    deploy_ok = reward_ok or spearman_ok or args.force_deploy
    if args.deploy and deploy_ok:
        backup_path = args.deploy_path.with_name(
            f"{args.deploy_path.stem}.{model_version}_backup.joblib"
        )
        if args.deploy_path.exists():
            joblib.dump(joblib.load(args.deploy_path), backup_path)
        joblib.dump(artifact, args.deploy_path)
        print(f"DEPLOYED {args.deploy_path} backup={backup_path}")
    else:
        reason = "insufficient improvement" if args.deploy else "--deploy not set"
        print(f"Skipped deploy ({reason}). reward_ok={reward_ok} spearman_ok={spearman_ok}")


if __name__ == "__main__":
    main()
