"""Tune dual-regime calibration from logged validator batches + benchmark holdout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture

from poker44.score.scoring import reward
from poker44_ml.calibration import simulate_regime_live_miner_scores
from poker44_ml.inference import Poker44Model
from poker44_ml.live_chunk_store import iter_logged_batches

DEFAULT_TUNE_PATH = Path(__file__).resolve().parents[1] / "models" / "live_regime_tune.json"


def replay_live_batch_priors(
    model: Poker44Model,
    *,
    store_dir: str | Path | None = None,
    max_batches: int | None = 300,
) -> list[dict[str, Any]]:
    """Replay stored validator batches; return per-batch supervised-raw stats."""
    rows: list[dict[str, Any]] = []
    for batch in iter_logged_batches(store_dir, max_batches=max_batches):
        chunks = batch.get("chunks") or []
        if not chunks:
            continue
        supervised_raw = model.predict_supervised_raw_scores(chunks)
        if not supervised_raw:
            continue
        arr = np.asarray(supervised_raw, dtype=float)
        rows.append(
            {
                "ts": batch.get("ts"),
                "chunk_count": len(chunks),
                "prior_mean": float(arr.mean()),
                "prior_median": float(np.median(arr)),
                "prior_max": float(arr.max()),
                "prior_std": float(arr.std()),
                "prior_p90": float(np.quantile(arr, 0.9)),
            }
        )
    return rows


def estimate_threshold_from_priors(priors: np.ndarray) -> dict[str, float]:
    """Split live batch priors into human-like / bot-like clusters via 2-GMM."""
    values = np.asarray(priors, dtype=float).reshape(-1, 1)
    if len(values) < 20:
        return {
            "regime_threshold": 0.28,
            "low_cluster_mean": float(np.quantile(values, 0.25)),
            "high_cluster_mean": float(np.quantile(values, 0.75)),
            "method": "quantile_fallback",
        }
    gmm = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(values)
    order = np.argsort(gmm.means_.ravel())
    low = float(gmm.means_.ravel()[order[0]])
    high = float(gmm.means_.ravel()[order[1]])
    threshold = float((low + high) / 2.0)
    threshold = float(np.clip(threshold, 0.20, 0.45))
    if float(values.std()) < 0.015:
        threshold = float(np.clip(np.quantile(values, 0.55), 0.22, 0.38))
    return {
        "regime_threshold": threshold,
        "low_cluster_mean": low,
        "high_cluster_mean": high,
        "method": "gmm_2component",
    }


def refine_threshold_on_benchmark(
    hybrid_scores: np.ndarray,
    labels: np.ndarray,
    *,
    groups: np.ndarray,
    center: float,
    max_fpr: float = 0.005,
    max_positive_rate: float = 0.02,
    spread_blend: float = 0.94,
    hard_ceiling: float = 0.49,
    human_spreads: tuple[tuple[float, float], ...] = (
        (0.03, 0.14),
        (0.04, 0.16),
        (0.05, 0.18),
    ),
    bot_spreads: tuple[tuple[float, float], ...] = (
        (0.20, 0.45),
        (0.22, 0.48),
        (0.24, 0.49),
    ),
    radius: float = 0.10,
    steps: int = 13,
) -> dict[str, Any]:
    """Grid around live-estimated threshold; pick best benchmark spearman with cls=0."""
    from train_reference_stack import _within_group_spearman

    thresholds = np.linspace(
        max(0.12, center - radius),
        min(0.55, center + radius),
        max(3, int(steps)),
    )
    best: tuple[tuple[float, float, float, float], dict[str, Any]] | None = None
    fallback: tuple[tuple[float, float, float, float], dict[str, Any]] | None = None

    for threshold in thresholds:
        for human_spread in human_spreads:
            for bot_spread in bot_spreads:
                live = simulate_regime_live_miner_scores(
                    hybrid_scores,
                    regime_threshold=float(threshold),
                    human_spread=tuple(human_spread),
                    bot_spread=tuple(bot_spread),
                    spread_blend=spread_blend,
                    max_positive_rate=max_positive_rate,
                    hard_ceiling=hard_ceiling,
                )
                rew, meta = reward(live, labels)
                spearman = float(
                    _within_group_spearman(live, labels, groups).get("mean", 0.0)
                )
                human_mask = labels == 0
                cls_penalty = (
                    float((live[human_mask] >= 0.5).mean()) if human_mask.any() else 0.0
                )
                candidate = (spearman, -cls_penalty, -float(meta.get("fpr", 1.0)), float(rew))
                payload = {
                    "regime_threshold": float(threshold),
                    "human_spread": list(human_spread),
                    "bot_spread": list(bot_spread),
                    "batch_spearman": spearman,
                    "reward": float(rew),
                    "score_max": float(live.max()),
                    "score_mean": float(live.mean()),
                    "above_05": int((live >= 0.5).sum()),
                }
                packed = (candidate, payload)
                if fallback is None or float(rew) > fallback[1]["reward"]:
                    fallback = packed
                if float(meta.get("fpr", 1.0)) > max_fpr:
                    continue
                if cls_penalty > 0.0:
                    continue
                if float(live.max()) < 0.28:
                    continue
                if float(live.std()) < 0.06:
                    continue
                if best is None or candidate > best[0]:
                    best = packed

    chosen = best or fallback
    if chosen is None:
        raise RuntimeError("Could not refine regime threshold on benchmark.")
    _, payload = chosen
    payload["method"] = "benchmark_refine"
    payload["center_threshold"] = float(center)
    return payload


def tune_regime_from_live(
    *,
    model_path: str | Path,
    store_dir: str | Path | None = None,
    max_live_batches: int = 300,
    hybrid_scores: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    output_path: str | Path = DEFAULT_TUNE_PATH,
) -> dict[str, Any]:
    model = Poker44Model(model_path=model_path)
    live_rows = replay_live_batch_priors(
        model,
        store_dir=store_dir,
        max_batches=max_live_batches,
    )
    priors = np.asarray([row["prior_mean"] for row in live_rows], dtype=float)
    live_estimate = estimate_threshold_from_priors(priors)

    result: dict[str, Any] = {
        "model_version": model.model_version,
        "live_batches_replayed": len(live_rows),
        "live_prior_stats": {
            "mean": float(priors.mean()) if len(priors) else 0.0,
            "std": float(priors.std()) if len(priors) else 0.0,
            "p10": float(np.quantile(priors, 0.10)) if len(priors) else 0.0,
            "p50": float(np.quantile(priors, 0.50)) if len(priors) else 0.0,
            "p90": float(np.quantile(priors, 0.90)) if len(priors) else 0.0,
        },
        "live_estimate": live_estimate,
    }

    if hybrid_scores is not None and labels is not None and groups is not None:
        refined = refine_threshold_on_benchmark(
            hybrid_scores,
            labels,
            groups=groups,
            center=float(live_estimate["regime_threshold"]),
        )
        result["calibration"] = refined
    else:
        result["calibration"] = {
            "regime_threshold": float(live_estimate["regime_threshold"]),
            "human_spread": [0.04, 0.16],
            "bot_spread": [0.22, 0.48],
            "method": "live_gmm_only",
        }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def load_regime_tune(path: str | Path = DEFAULT_TUNE_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    cal = dict(payload.get("calibration") or {})
    return {
        "regime_threshold": float(cal.get("regime_threshold", 0.28)),
        "human_spread": tuple(cal.get("human_spread") or (0.04, 0.16)),
        "bot_spread": tuple(cal.get("bot_spread") or (0.22, 0.48)),
        "source": payload,
    }
