#!/usr/bin/env python3
"""Train a reference-style Poker44 LGB/XGB OOF stack."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

try:
    import lightgbm as lgb
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("lightgbm and xgboost are required for reference stack training.") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker44.score.scoring import reward
from poker44.validator.payload_view import prepare_hand_for_miner
from poker44_ml.calibration import (
    BlendedQuantileCalibrator,
    apply_score_logit_calibration,
    simulate_live_miner_scores,
)
from poker44_ml.features import chunk_features
from poker44_ml.robust_features import filter_robust_feature_names
from poker44_ml.stacked import StackedEnsemble

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

API_BASE = "https://api.poker44.net/api/v1/benchmark"
DEFAULT_OUTPUT = REPO_ROOT / "models" / "bot_detector_reference_stack.joblib"
DEFAULT_STATE_PATH = REPO_ROOT / "models" / "benchmark_state.json"
HUMAN_SAMPLE_WEIGHT = 1.75


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def fetch_benchmark_state() -> dict[str, Any]:
    payload = _get_json(f"{API_BASE}/releases")["data"]
    releases = payload.get("releases") or []
    return {
        "release_version": payload.get("releaseVersion"),
        "schema_version": payload.get("schemaVersion"),
        "source_dates": sorted({release["sourceDate"] for release in releases}),
        "release_ids": sorted({release["releaseId"] for release in releases}),
        "sample_count": sum(int(release.get("chunkCount", 0)) for release in releases),
    }


def fetch_training_examples(
    *,
    min_source_date: str | None = None,
) -> tuple[list[dict[str, float]], np.ndarray, list[dict], dict]:
    benchmark_state = fetch_benchmark_state()
    releases = _get_json(f"{API_BASE}/releases")["data"]["releases"]
    if min_source_date:
        releases = [
            release
            for release in releases
            if str(release.get("sourceDate", "")) >= str(min_source_date)
        ]
        benchmark_state["min_source_date"] = str(min_source_date)
        benchmark_state["filtered_release_count"] = int(len(releases))
    feature_dicts: list[dict[str, float]] = []
    labels: list[int] = []
    metadata: list[dict] = []

    for release in releases:
        source_date = release["sourceDate"]
        outer_chunks = _get_json(f"{API_BASE}/chunks?sourceDate={source_date}")["data"]["chunks"]
        for outer in outer_chunks:
            chunk_id = outer["chunkId"]
            detail = _get_json(f"{API_BASE}/chunks/{chunk_id}")["data"]
            ground_truth = detail.get("groundTruth") or []
            inner_chunks = detail.get("chunks") or []
            for index, inner_chunk in enumerate(inner_chunks):
                if index >= len(ground_truth):
                    continue
                visible_chunk = [
                    hand for hand in inner_chunk if isinstance(hand, dict)
                ]
                features = chunk_features(visible_chunk)
                features["hand_count"] = float(len(visible_chunk))
                feature_dicts.append(features)
                labels.append(int(ground_truth[index]))
                metadata.append(
                    {
                        "chunk_id": chunk_id,
                        "inner_index": index,
                        "source_date": source_date,
                        "hand_count": len(visible_chunk),
                    }
                )

    if not feature_dicts:
        raise RuntimeError("No benchmark training samples were fetched.")
    benchmark_state["training_chunks"] = int(len(labels))
    return feature_dicts, np.asarray(labels, dtype=int), metadata, benchmark_state


def vectorize(feature_dicts: list[dict[str, float]]) -> tuple[np.ndarray, list[str]]:
    all_names = sorted({name for row in feature_dicts for name in row})
    feature_names = filter_robust_feature_names(all_names)
    if not feature_names:
        raise RuntimeError("No robust feature names were generated.")
    x = np.asarray(
        [[float(row.get(name, 0.0)) for name in feature_names] for row in feature_dicts],
        dtype=float,
    )
    return x, feature_names


def sample_weights(labels: np.ndarray) -> np.ndarray:
    weights = np.ones(len(labels), dtype=float)
    weights[labels == 0] = HUMAN_SAMPLE_WEIGHT
    return weights


def make_base_models(seed: int, *, n_jobs: int = -1) -> list[tuple[str, Any]]:
    return [
        (
            "lgb",
            lgb.LGBMClassifier(
                n_estimators=900,
                learning_rate=0.018,
                num_leaves=31,
                max_depth=7,
                min_child_samples=10,
                feature_fraction=0.72,
                bagging_fraction=0.85,
                bagging_freq=1,
                reg_alpha=0.5,
                reg_lambda=1.2,
                class_weight="balanced",
                random_state=seed,
                verbose=-1,
                n_jobs=n_jobs,
            ),
        ),
        (
            "lgb_rank",
            lgb.LGBMRanker(
                n_estimators=600,
                learning_rate=0.022,
                num_leaves=23,
                max_depth=6,
                min_child_samples=8,
                feature_fraction=0.70,
                bagging_fraction=0.85,
                bagging_freq=1,
                reg_alpha=0.4,
                reg_lambda=1.0,
                random_state=seed + 2,
                verbose=-1,
                n_jobs=n_jobs,
            ),
        ),
        (
            "xgb",
            xgb.XGBClassifier(
                n_estimators=600,
                learning_rate=0.025,
                max_depth=5,
                min_child_weight=4,
                subsample=0.82,
                colsample_bytree=0.75,
                reg_alpha=0.5,
                reg_lambda=1.8,
                eval_metric="aucpr",
                random_state=seed + 1,
                n_jobs=n_jobs,
            ),
        ),
    ]


def fit_model(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray | None = None,
) -> Any:
    import lightgbm as lgb

    from poker44_ml.rank_stack import fit_ranker

    if isinstance(model, lgb.LGBMRanker):
        if groups is not None and len(np.unique(groups)) > 1:
            return fit_ranker(model, x, y, groups)
        group = np.ones(len(y), dtype=int)
        try:
            return model.fit(x, y, group=group, sample_weight=weights)
        except TypeError:
            return model.fit(x, y, group=group)
    try:
        return model.fit(x, y, sample_weight=weights)
    except TypeError:
        return model.fit(x, y)


def base_predict(model: Any, x: np.ndarray) -> np.ndarray:
    import lightgbm as lgb
    if isinstance(model, lgb.LGBMRanker):
        raw = np.asarray(model.predict(x), dtype=float)
        # convert ranking scores to [0,1] via sigmoid-like normalization
        raw = raw - raw.min()
        rng = raw.max()
        if rng > 1e-9:
            raw = raw / rng
        return np.clip(raw, 0.0, 1.0)
    proba = np.asarray(model.predict_proba(x))
    return np.clip(proba[:, 1] if proba.ndim == 2 else proba, 0.0, 1.0)


def _date_groups(metadata: list[dict]) -> np.ndarray:
    dates = sorted({row["source_date"] for row in metadata})
    lookup = {date: index for index, date in enumerate(dates)}
    return np.asarray([lookup[row["source_date"]] for row in metadata], dtype=int)


def _within_group_spearman(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    rhos: list[float] = []
    for group_id in np.unique(groups):
        mask = groups == group_id
        if int(mask.sum()) < 4:
            continue
        group_scores = scores[mask]
        group_labels = labels[mask]
        if float(group_scores.std()) < 1e-12 or float(group_labels.std()) < 1e-12:
            continue
        rank = float(
            __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(
                group_scores,
                group_labels,
            ).correlation
        )
        if rank == rank:
            rhos.append(rank)
    if not rhos:
        return {"mean": 0.0, "median": 0.0, "fraction_positive": 0.0}
    arr = np.asarray(rhos, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "fraction_positive": float((arr > 0.0).mean()),
    }


def train_oof_stack(
    x: np.ndarray,
    y: np.ndarray,
    metadata: list[dict],
    *,
    seed: int,
    folds: int,
) -> tuple[StackedEnsemble, np.ndarray, np.ndarray, dict[str, Any]]:
    weights = sample_weights(y)
    base_specs = make_base_models(seed)
    groups = _date_groups(metadata)
    unique_groups = len(np.unique(groups))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), len(base_specs)), dtype=float)

    for fold_index, (train_idx, valid_idx) in enumerate(cv.split(x, y), start=1):
        for model_index, (name, model) in enumerate(base_specs):
            fitted = fit_model(clone(model), x[train_idx], y[train_idx], weights[train_idx])
            oof[valid_idx, model_index] = base_predict(fitted, x[valid_idx])
        print(f"Finished OOF fold {fold_index}/{folds}")

    date_cv = GroupKFold(n_splits=max(2, min(folds, unique_groups)))
    date_oof = np.zeros((len(y), len(base_specs)), dtype=float)
    for train_idx, valid_idx in date_cv.split(x, y, groups=groups):
        for model_index, (_, model) in enumerate(base_specs):
            fitted = fit_model(clone(model), x[train_idx], y[train_idx], weights[train_idx])
            date_oof[valid_idx, model_index] = base_predict(fitted, x[valid_idx])
    date_base_mean = np.mean(date_oof, axis=1)

    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    meta.fit(oof, y, sample_weight=weights)
    meta_oof = np.asarray(meta.predict_proba(oof))[:, 1]
    base_mean_oof = np.mean(oof, axis=1)
    stack_calibrator = BlendedQuantileCalibrator(blend=0.9).fit(base_mean_oof)
    calibrated_oof = stack_calibrator.transform(base_mean_oof)

    final_base_models = [
        fit_model(clone(model), x, y, weights) for _, model in base_specs
    ]
    ensemble = StackedEnsemble(
        base_models=final_base_models,
        meta_model=meta,
        calibrator=stack_calibrator,
        stack_mode="mean",
    )
    full_fit_scores = base_predict(ensemble, x)
    diagnostics = {
        "base_model_names": [name for name, _ in base_specs],
        "oof_base_scores": {
            name: {
                "ap": float(average_precision_score(y, oof[:, index])),
                "auc": float(roc_auc_score(y, oof[:, index])),
            }
            for index, (name, _) in enumerate(base_specs)
        },
        "meta_oof_ap": float(average_precision_score(y, meta_oof)),
        "meta_oof_auc": float(roc_auc_score(y, meta_oof)),
        "base_mean_oof_ap": float(average_precision_score(y, base_mean_oof)),
        "base_mean_oof_auc": float(roc_auc_score(y, base_mean_oof)),
        "calibrated_oof_ap": float(average_precision_score(y, calibrated_oof)),
        "calibrated_oof_auc": float(roc_auc_score(y, calibrated_oof)),
        "date_holdout_ap": float(average_precision_score(y, date_base_mean)),
        "date_holdout_auc": float(roc_auc_score(y, date_base_mean)),
        "date_holdout_spearman": _within_group_spearman(date_base_mean, y, groups),
        "full_fit_ap": float(average_precision_score(y, full_fit_scores)),
        "full_fit_auc": float(roc_auc_score(y, full_fit_scores)),
        "full_fit_date_spearman": _within_group_spearman(full_fit_scores, y, groups),
        "cv_strategy": "stratified_with_date_holdout_report",
        "cv_folds": int(folds),
    }
    return ensemble, calibrated_oof, full_fit_scores, diagnostics


def select_live_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    max_fpr: float,
    batch_size: int = 40,
    max_positive_rate: float = 0.10,
    groups: np.ndarray | None = None,
    rank_first: bool = False,
    target_medians: tuple[float, ...] | None = None,
    spread_blend: float | None = None,
    spearman_mask: np.ndarray | None = None,
    center_blends: tuple[float, ...] | None = None,
    spread_bounds: tuple[tuple[float | None, float | None], ...] | None = None,
    hard_ceiling: float | None = 0.49,
) -> tuple[dict[str, float | str | bool], dict[str, Any]]:
    spread_blend = float(spread_blend if spread_blend is not None else (0.85 if rank_first else 0.70))
    if spread_bounds is None:
        spread_bounds = (
            (None, None),
            (0.08, 0.35),
            (0.10, 0.38),
            (0.12, 0.40),
        ) if rank_first else (
            (None, None),
            (0.08, 0.35),
            (0.10, 0.38),
            (0.12, 0.40),
            (0.15, 0.42),
            (0.18, 0.48),
        )
    if center_blends is None:
        center_blends = (0.0, 0.5, 0.75)
    if target_medians is None:
        target_medians = (0.12, 0.16, 0.20) if rank_first else (0.20, 0.24, 0.28)
    temperatures = (0.55, 0.65) if rank_first else (0.55, 0.65, 0.75)
    best: tuple[tuple[float, float, float, float], dict[str, Any], np.ndarray] | None = None
    fallback: tuple[tuple[float, float, float, float], dict[str, Any], np.ndarray] | None = None

    for spread_low, spread_high in spread_bounds:
        for center_blend in center_blends:
            for target_median in target_medians:
                for temperature in temperatures:
                    live_scores = simulate_live_miner_scores(
                        scores,
                        bias=2.2,
                        temperature=float(temperature),
                        batch_size=batch_size,
                        spread_blend=spread_blend,
                        center_blend=float(center_blend),
                        spread_low=spread_low,
                        spread_high=spread_high,
                        max_positive_rate=max_positive_rate,
                        logit_mode="adaptive",
                        target_median=float(target_median),
                        hard_ceiling=hard_ceiling,
                    )
                    rew, meta = reward(live_scores, labels)
                    batch_spearman = 0.0
                    spearman_labels = labels
                    spearman_scores = live_scores
                    spearman_groups = groups
                    if spearman_mask is not None and bool(spearman_mask.any()):
                        spearman_labels = labels[spearman_mask]
                        spearman_scores = live_scores[spearman_mask]
                        spearman_groups = groups[spearman_mask] if groups is not None else None
                    if spearman_groups is not None:
                        batch_spearman = float(
                            _within_group_spearman(
                                spearman_scores,
                                spearman_labels,
                                spearman_groups,
                            ).get(
                                "mean",
                                0.0,
                            )
                        )
                    human_mask = labels == 0
                    cls_penalty = (
                        float((live_scores[human_mask] >= 0.5).mean())
                        if human_mask.any()
                        else 0.0
                    )
                    if rank_first:
                        candidate = (
                            float(batch_spearman),
                            -cls_penalty,
                            -float(meta.get("fpr", 1.0)),
                            float(rew),
                        )
                    else:
                        candidate = (
                            float(batch_spearman),
                            float(rew),
                            float(average_precision_score(labels, live_scores)),
                            -float(meta.get("fpr", 1.0)),
                        )
                    payload = {
                        "temperature": float(temperature),
                        "center_blend": float(center_blend),
                        "target_median": float(target_median),
                        "spread_low": spread_low,
                        "spread_high": spread_high,
                        "reward": float(rew),
                        "reward_meta": meta,
                        "batch_spearman": batch_spearman,
                    }
                    packed = (candidate, payload, live_scores)
                    if fallback is None or float(rew) > fallback[1]["reward"]:
                        fallback = packed
                    if float(meta.get("fpr", 1.0)) > max_fpr:
                        continue
                    if rank_first and cls_penalty > 0.0:
                        continue
                    if best is None or candidate > best[0]:
                        best = packed

    chosen = best or fallback
    if chosen is None:
        raise RuntimeError("Could not select live calibration.")
    _, payload, live_scores = chosen
    center_blend = float(payload.get("center_blend", 0.0))
    live_settings = {
        "live_batch_spread": True,
        "live_batch_spread_blend": spread_blend,
        "live_batch_center": center_blend > 0.0,
        "live_batch_center_blend": center_blend,
        "live_max_positive_rate": float(max_positive_rate),
        "live_logit_mode": "adaptive",
        "live_logit_target_median": float(payload.get("target_median", 0.24)),
        "score_logit_bias": 2.2,
        "score_logit_temperature": float(payload["temperature"]),
        "score_remap": {},
    }
    if payload.get("spread_low") is not None and payload.get("spread_high") is not None:
        live_settings["live_batch_spread_low"] = float(payload["spread_low"])
        live_settings["live_batch_spread_high"] = float(payload["spread_high"])
    if hard_ceiling is not None:
        live_settings["live_score_ceiling"] = float(hard_ceiling)
    metrics = {
        "reward": float(payload["reward"]),
        "reward_meta": payload["reward_meta"],
        "average_precision": float(average_precision_score(labels, live_scores)),
        "roc_auc": float(roc_auc_score(labels, live_scores)),
        "score_min": float(live_scores.min()),
        "score_mean": float(live_scores.mean()),
        "score_max": float(live_scores.max()),
        "score_std": float(live_scores.std()),
        "above_05": int((live_scores >= 0.5).sum()),
        "calibration_choice": payload,
    }
    return live_settings, metrics


def select_regime_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    max_fpr: float,
    batch_size: int = 40,
    max_positive_rate: float = 0.02,
    human_max_positive_rate: float | None = None,
    bot_max_positive_rate: float | None = None,
    groups: np.ndarray | None = None,
    spread_blend: float = 0.92,
    spearman_mask: np.ndarray | None = None,
    hard_ceiling: float | None = 0.49,
    human_hard_ceiling: float | None = None,
    bot_hard_ceiling: float | None = None,
    reward_first: bool = False,
    min_bot_recall: float = 0.25,
    regime_scores: np.ndarray | None = None,
    chunk_regime: bool = True,
    holdout_mask: np.ndarray | None = None,
    extended_grid: bool = False,
) -> tuple[dict[str, float | str | bool | list[float]], dict[str, Any]]:
    from poker44_ml.calibration import simulate_regime_live_miner_scores

    thresholds = (
        (0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.44)
        if extended_grid
        else (0.32, 0.36, 0.40, 0.44, 0.48)
    )
    human_spreads = ((0.03, 0.14), (0.04, 0.16), (0.05, 0.18))
    bot_spreads = (
        (0.22, 0.50),
        (0.24, 0.54),
        (0.26, 0.58),
        (0.28, 0.60),
        (0.30, 0.62),
    ) if extended_grid else (
        (0.22, 0.50),
        (0.24, 0.54),
        (0.26, 0.58),
        (0.28, 0.62),
    )
    bot_ceilings = (0.58, 0.60, 0.62) if extended_grid else (bot_hard_ceiling or 0.58,)
    bot_rates = (
        (0.15, 0.18, 0.20, 0.22)
        if extended_grid and bot_max_positive_rate is not None
        else (float(bot_max_positive_rate) if bot_max_positive_rate is not None else max_positive_rate,)
    )
    if human_hard_ceiling is None and hard_ceiling is not None:
        human_hard_ceiling = float(hard_ceiling)
    if bot_hard_ceiling is None:
        bot_hard_ceiling = 0.58
    best: tuple[tuple[float, float, float, float], dict[str, Any], np.ndarray] | None = None
    fallback: tuple[tuple[float, float, float, float], dict[str, Any], np.ndarray] | None = None

    for threshold in thresholds:
        for human_spread in human_spreads:
            for bot_spread in bot_spreads:
                for bot_ceiling in bot_ceilings:
                    for bot_rate in bot_rates:
                        live_scores = simulate_regime_live_miner_scores(
                            scores,
                            regime_threshold=float(threshold),
                            human_spread=tuple(human_spread),
                            bot_spread=tuple(bot_spread),
                            spread_blend=spread_blend,
                            batch_size=batch_size,
                            max_positive_rate=max_positive_rate,
                            human_max_positive_rate=human_max_positive_rate,
                            bot_max_positive_rate=float(bot_rate),
                            hard_ceiling=hard_ceiling,
                            human_hard_ceiling=human_hard_ceiling,
                            bot_hard_ceiling=float(bot_ceiling),
                            regime_scores=regime_scores,
                            chunk_regime=chunk_regime,
                        )
                        sel_mask = (
                            holdout_mask
                            if holdout_mask is not None and bool(holdout_mask.any())
                            else np.ones(len(labels), dtype=bool)
                        )
                        sel_scores = live_scores[sel_mask]
                        sel_labels = labels[sel_mask]
                        rew, meta = reward(sel_scores, sel_labels)
                        batch_spearman = 0.0
                        spearman_labels = labels
                        spearman_scores = live_scores
                        spearman_groups = groups
                        if spearman_mask is not None and bool(spearman_mask.any()):
                            spearman_labels = labels[spearman_mask]
                            spearman_scores = live_scores[spearman_mask]
                            spearman_groups = groups[spearman_mask] if groups is not None else None
                        if spearman_groups is not None:
                            batch_spearman = float(
                                _within_group_spearman(
                                    spearman_scores,
                                    spearman_labels,
                                    spearman_groups,
                                ).get("mean", 0.0)
                            )
                        human_mask = sel_labels == 0
                        cls_penalty = (
                            float((sel_scores[human_mask] >= 0.5).mean())
                            if human_mask.any()
                            else 0.0
                        )
                        bot_recall = float(meta.get("bot_recall", 0.0) or 0.0)
                        if reward_first:
                            candidate = (
                                float(rew),
                                bot_recall,
                                -float(meta.get("fpr", 1.0)),
                                float(batch_spearman),
                            )
                        else:
                            candidate = (
                                float(batch_spearman),
                                -cls_penalty,
                                -float(meta.get("fpr", 1.0)),
                                float(rew),
                            )
                        full_rew, full_meta = reward(live_scores, labels)
                        payload = {
                            "regime_threshold": float(threshold),
                            "human_spread": list(human_spread),
                            "bot_spread": list(bot_spread),
                            "bot_max_positive_rate": float(bot_rate),
                            "bot_hard_ceiling": float(bot_ceiling),
                            "reward": float(full_rew),
                            "holdout_reward": float(rew),
                            "reward_meta": full_meta,
                            "holdout_reward_meta": meta,
                            "batch_spearman": batch_spearman,
                            "above_05": int((live_scores >= 0.5).sum()),
                            "score_max": float(live_scores.max()),
                        }
                        packed = (candidate, payload, live_scores)
                        if fallback is None or float(rew) > fallback[1].get("holdout_reward", fallback[1]["reward"]):
                            fallback = packed
                        if float(meta.get("fpr", 1.0)) > max_fpr:
                            continue
                        if cls_penalty > 0.0:
                            continue
                        if reward_first and bot_recall < min_bot_recall:
                            continue
                        if best is None or candidate > best[0]:
                            best = packed

    chosen = best or fallback
    if chosen is None:
        raise RuntimeError("Could not select regime calibration.")
    _, payload, live_scores = chosen
    live_settings = {
        "live_batch_spread": True,
        "live_batch_spread_blend": spread_blend,
        "live_batch_center": False,
        "live_batch_center_blend": 0.0,
        "live_max_positive_rate": float(max_positive_rate),
        "live_human_max_positive_rate": float(
            max_positive_rate if human_max_positive_rate is None else human_max_positive_rate
        ),
        "live_bot_max_positive_rate": float(
            payload.get("bot_max_positive_rate", bot_max_positive_rate)
            if bot_max_positive_rate is not None
            else max_positive_rate
        ),
        "live_logit_mode": "regime",
        "live_regime_enabled": True,
        "live_chunk_regime_enabled": bool(chunk_regime),
        "live_regime_threshold": float(payload["regime_threshold"]),
        "live_human_spread": list(payload["human_spread"]),
        "live_bot_spread": list(payload["bot_spread"]),
        "live_score_ceiling": float(human_hard_ceiling) if human_hard_ceiling is not None else 0.49,
        "live_human_score_ceiling": float(human_hard_ceiling) if human_hard_ceiling is not None else 0.49,
        "live_bot_score_ceiling": float(
            payload.get("bot_hard_ceiling", bot_hard_ceiling) if bot_hard_ceiling is not None else 0.58
        ),
        "score_logit_bias": 2.2,
        "score_logit_temperature": 0.55,
        "score_remap": {},
        "hybrid_fusion": "max",
    }
    metrics = {
        "reward": float(payload["reward"]),
        "reward_meta": payload["reward_meta"],
        "average_precision": float(average_precision_score(labels, live_scores)),
        "roc_auc": float(roc_auc_score(labels, live_scores)),
        "score_min": float(live_scores.min()),
        "score_mean": float(live_scores.mean()),
        "score_max": float(live_scores.max()),
        "score_std": float(live_scores.std()),
        "above_05": int((live_scores >= 0.5).sum()),
        "calibration_choice": payload,
        "batch_spearman": float(payload.get("batch_spearman", 0.0)),
    }
    return live_settings, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-fpr", type=float, default=0.04)
    parser.add_argument("--max-positive-rate", type=float, default=0.10)
    args = parser.parse_args()

    print("Fetching benchmark examples...")
    feature_dicts, y, metadata, benchmark_state = fetch_training_examples()
    x, feature_names = vectorize(feature_dicts)
    print(
        f"Loaded {len(y)} chunks ({int(y.sum())} bot / {int(len(y) - y.sum())} human), "
        f"{len(feature_names)} robust features"
    )

    ensemble, calibrated_oof, full_fit_scores, diagnostics = train_oof_stack(
        x,
        y,
        metadata,
        seed=args.seed,
        folds=args.folds,
    )
    calibration_scores = full_fit_scores
    live_settings, live_metrics = select_live_calibration(
        calibration_scores,
        y,
        max_fpr=args.max_fpr,
        max_positive_rate=args.max_positive_rate,
    )
    print(json.dumps({"diagnostics": diagnostics, "live_settings": live_settings, "metrics": live_metrics}, indent=2))

    artifact = {
        "models": [ensemble],
        "model_weights": [1.0],
        "feature_names": feature_names,
        "metadata": {
            "model_name": "poker44-reference-stack",
            "model_version": "reference-stack-v5.1-behavioral",
            "model_weights": [1.0],
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
            "human_sample_weight": HUMAN_SAMPLE_WEIGHT,
        },
        "metrics": live_metrics,
        "training_samples": len(y),
        "metadata_rows": metadata,
        "benchmark_state": benchmark_state,
        "model_version": "reference-stack-v5.1-behavioral",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    args.state_file.write_text(json.dumps(benchmark_state, indent=2) + "\n")
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()

