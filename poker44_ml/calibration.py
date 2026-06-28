from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.preprocessing import QuantileTransformer


class BlendedQuantileCalibrator:
    """Monotone score spreader for collapsed stacked probabilities."""

    def __init__(self, blend: float = 0.9, max_quantiles: int = 256) -> None:
        self.blend = float(max(0.0, min(1.0, blend)))
        self.max_quantiles = int(max(8, max_quantiles))
        self._qt: Optional[QuantileTransformer] = None

    def fit(self, scores: np.ndarray) -> "BlendedQuantileCalibrator":
        values = np.asarray(scores, dtype=float).reshape(-1, 1)
        n_quantiles = int(max(8, min(self.max_quantiles, len(values))))
        qt = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="uniform",
            subsample=max(len(values), 1000),
            random_state=42,
        )
        qt.fit(values)
        self._qt = qt
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=float).reshape(-1, 1)
        if self._qt is None:
            return np.clip(values.ravel(), 0.0, 1.0)
        uniformized = self._qt.transform(values).ravel()
        base = np.clip(values.ravel(), 0.0, 1.0)
        mixed = self.blend * uniformized + (1.0 - self.blend) * base
        return np.clip(mixed, 0.0, 1.0)


def apply_threshold_logit_remap(
    scores: np.ndarray,
    *,
    threshold: float,
    temperature: float,
) -> np.ndarray:
    values = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
    adjusted = (values - float(threshold)) / max(float(temperature), 1e-6)
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(adjusted, -40.0, 40.0))), 0.0, 1.0)


def apply_batch_median_center(
    scores: np.ndarray,
    *,
    blend: float = 0.75,
) -> np.ndarray:
    """Re-center a live batch so its median sits near 0.5 before spreading."""
    values = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    if values.size <= 1:
        return values
    median = float(np.median(values))
    centered = np.clip(values - median + 0.5, 0.0, 1.0)
    mix = float(max(0.0, min(1.0, blend)))
    return np.clip(mix * centered + (1.0 - mix) * values, 0.0, 1.0)


def apply_batch_quantile_spread(
    scores: np.ndarray,
    *,
    blend: float = 0.85,
    min_std: float = 0.02,
    spread_low: float | None = None,
    spread_high: float | None = None,
) -> np.ndarray:
    """Spread scores within a validator request while preserving rank order.

    Uses a rank-based linear spread instead of fitting a QuantileTransformer
    each call, which reduces latency from ~700ms to <1ms for a 40-chunk batch.
    """
    values = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    if values.size <= 1:
        return values
    if float(values.std()) < min_std:
        return values

    n = len(values)
    order = np.argsort(values, kind="stable")
    rank_uniform = np.empty(n, dtype=float)
    rank_uniform[order] = np.linspace(0.0, 1.0, n)
    spread = np.clip(float(blend) * rank_uniform + (1.0 - float(blend)) * values, 0.0, 1.0)

    if spread_low is None and spread_high is None:
        return spread
    low = float(spread_low if spread_low is not None else 0.0)
    high = float(spread_high if spread_high is not None else 1.0)
    if high < low:
        low, high = high, low
    return np.clip(low + (high - low) * spread, 0.0, 1.0)


def simulate_live_batch_scores(
    scores: np.ndarray,
    *,
    batch_size: int = 40,
    blend: float = 0.85,
    center_blend: float = 0.0,
    spread_low: float | None = None,
    spread_high: float | None = None,
) -> np.ndarray:
    """Evaluate calibration as validators see it: one request batch at a time."""
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return values
    batch_size = max(1, int(batch_size))
    output = np.zeros_like(values)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        batch = values[start:stop]
        if center_blend > 0.0:
            batch = apply_batch_median_center(batch, blend=center_blend)
        output[start:stop] = apply_batch_quantile_spread(
            batch,
            blend=blend,
            spread_low=spread_low,
            spread_high=spread_high,
        )
    return output


def apply_score_logit_calibration(
    scores: np.ndarray,
    *,
    bias: float = 0.0,
    temperature: float = 1.0,
) -> np.ndarray:
    """Conservative logit shift used by both training and live inference."""
    values = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(values / (1.0 - values))
    adjusted = (logits - float(bias)) / max(float(temperature), 1e-6)
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(adjusted, -40.0, 40.0))), 0.0, 1.0)


def apply_batch_adaptive_logit(
    scores: np.ndarray,
    *,
    target_median: float = 0.30,
    temperature: float = 0.85,
) -> np.ndarray:
    """Map each live batch median to a safe target while preserving rank spread."""
    values = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
    if values.size <= 1:
        return apply_score_logit_calibration(values, bias=0.0, temperature=temperature)
    median = float(np.median(values))
    target = float(max(0.05, min(0.45, target_median)))
    median_logit = np.log(median / (1.0 - median))
    target_logit = np.log(target / (1.0 - target))
    shift = median_logit - target_logit
    logits = np.log(values / (1.0 - values)) - shift
    adjusted = logits / max(float(temperature), 1e-6)
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(adjusted, -40.0, 40.0))), 0.0, 1.0)


def _resolve_score_ceiling(
    *,
    hard_ceiling: float | None,
    human_hard_ceiling: float | None,
    bot_hard_ceiling: float | None,
    batch_regime: str | None,
) -> float | None:
    if batch_regime == "bot" and bot_hard_ceiling is not None:
        return float(bot_hard_ceiling)
    if batch_regime == "human" and human_hard_ceiling is not None:
        return float(human_hard_ceiling)
    if hard_ceiling is not None:
        return float(hard_ceiling)
    return None


def apply_live_positive_cap(
    scores: np.ndarray,
    *,
    max_positive_rate: float = 0.10,
    score_cap_epsilon: float = 1e-6,
    hard_ceiling: float | None = 0.49,
    human_hard_ceiling: float | None = None,
    bot_hard_ceiling: float | None = None,
    batch_regime: str | None = None,
) -> np.ndarray:
    """Mirror miner-side cap: limit hard flags at 0.5 while preserving order."""
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return values
    max_positive = int(values.size * max_positive_rate)
    positive_count = int((values >= 0.5).sum())
    if positive_count > max_positive:
        sorted_scores = np.sort(values)[::-1]
        if max_positive <= 0:
            scale = (0.5 - score_cap_epsilon) / max(float(sorted_scores[0]), score_cap_epsilon)
            values = np.clip(values * scale, 0.0, 0.5 - score_cap_epsilon)
        else:
            cutoff = float(sorted_scores[max_positive - 1])
            scale = (0.5 - score_cap_epsilon) / max(cutoff, score_cap_epsilon)
            values = np.where(values >= cutoff, values, values * scale)
            values = np.clip(values, 0.0, 1.0)
    ceiling = _resolve_score_ceiling(
        hard_ceiling=hard_ceiling,
        human_hard_ceiling=human_hard_ceiling,
        bot_hard_ceiling=bot_hard_ceiling,
        batch_regime=batch_regime,
    )
    if ceiling is not None:
        values = np.clip(values, 0.0, ceiling)
    return values


def apply_regime_batch_spread(
    scores: np.ndarray,
    *,
    batch_prior: float,
    regime_threshold: float,
    human_spread: tuple[float, float],
    bot_spread: tuple[float, float],
    blend: float = 0.92,
) -> tuple[np.ndarray, float]:
    """Dual-regime spread: tight low band for human-like batches, wide high band for bot-like."""
    if batch_prior >= float(regime_threshold):
        low, high = bot_spread
        target_median = float(bot_spread[0] + (bot_spread[1] - bot_spread[0]) * 0.55)
    else:
        low, high = human_spread
        target_median = float(human_spread[0] + (human_spread[1] - human_spread[0]) * 0.35)
    spread = apply_batch_quantile_spread(
        scores,
        blend=blend,
        spread_low=low,
        spread_high=high,
    )
    return spread, target_median


def simulate_regime_live_miner_scores(
    raw_scores: np.ndarray,
    *,
    regime_threshold: float = 0.35,
    human_spread: tuple[float, float] = (0.04, 0.16),
    bot_spread: tuple[float, float] = (0.22, 0.48),
    spread_blend: float = 0.92,
    batch_size: int = 40,
    max_positive_rate: float = 0.02,
    human_max_positive_rate: float | None = None,
    bot_max_positive_rate: float | None = None,
    temperature: float = 0.55,
    hard_ceiling: float | None = 0.49,
    human_hard_ceiling: float | None = None,
    bot_hard_ceiling: float | None = None,
) -> np.ndarray:
    """Regime-aware live path: batch prior selects spread band + target median."""
    values = np.asarray(raw_scores, dtype=float)
    if values.size == 0:
        return values
    batch_size = max(1, int(batch_size))
    human_rate = float(max_positive_rate if human_max_positive_rate is None else human_max_positive_rate)
    bot_rate = float(max_positive_rate if bot_max_positive_rate is None else bot_max_positive_rate)
    if human_hard_ceiling is None and hard_ceiling is not None:
        human_hard_ceiling = float(hard_ceiling)
    output = np.zeros_like(values)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        batch_raw = values[start:stop]
        prior = float(np.mean(batch_raw))
        is_bot_batch = prior >= float(regime_threshold)
        spread, target_median = apply_regime_batch_spread(
            batch_raw,
            batch_prior=prior,
            regime_threshold=regime_threshold,
            human_spread=human_spread,
            bot_spread=bot_spread,
            blend=spread_blend,
        )
        calibrated = apply_batch_adaptive_logit(
            spread,
            target_median=target_median,
            temperature=temperature,
        )
        output[start:stop] = apply_live_positive_cap(
            calibrated,
            max_positive_rate=bot_rate if is_bot_batch else human_rate,
            hard_ceiling=None,
            human_hard_ceiling=human_hard_ceiling,
            bot_hard_ceiling=bot_hard_ceiling,
            batch_regime="bot" if is_bot_batch else "human",
        )
    return output


def simulate_live_miner_scores(
    raw_scores: np.ndarray,
    *,
    bias: float,
    temperature: float,
    batch_size: int = 40,
    spread_blend: float = 0.85,
    center_blend: float = 0.0,
    spread_low: float | None = None,
    spread_high: float | None = None,
    max_positive_rate: float = 0.10,
    logit_mode: str = "fixed",
    target_median: float = 0.28,
    hard_ceiling: float | None = 0.49,
    human_hard_ceiling: float | None = None,
    bot_hard_ceiling: float | None = None,
    human_max_positive_rate: float | None = None,
    bot_max_positive_rate: float | None = None,
    regime_enabled: bool = False,
    regime_threshold: float = 0.35,
    human_spread: tuple[float, float] = (0.04, 0.16),
    bot_spread: tuple[float, float] = (0.22, 0.48),
) -> np.ndarray:
    """End-to-end live path: batch spread -> logit calibration -> positive cap."""
    if regime_enabled:
        return simulate_regime_live_miner_scores(
            raw_scores,
            regime_threshold=regime_threshold,
            human_spread=human_spread,
            bot_spread=bot_spread,
            spread_blend=spread_blend,
            batch_size=batch_size,
            max_positive_rate=max_positive_rate,
            human_max_positive_rate=human_max_positive_rate,
            bot_max_positive_rate=bot_max_positive_rate,
            temperature=temperature,
            hard_ceiling=hard_ceiling,
            human_hard_ceiling=human_hard_ceiling,
            bot_hard_ceiling=bot_hard_ceiling,
        )
    spread = simulate_live_batch_scores(
        raw_scores,
        batch_size=batch_size,
        blend=spread_blend,
        center_blend=center_blend,
        spread_low=spread_low,
        spread_high=spread_high,
    )
    output = np.zeros_like(spread)
    for start in range(0, len(spread), batch_size):
        stop = min(start + batch_size, len(spread))
        batch = spread[start:stop]
        mode = str(logit_mode or "fixed").lower()
        if mode in {"auto", "adaptive"}:
            if mode == "auto":
                median = float(np.median(batch))
                batch_std = float(batch.std())
                use_adaptive = median >= 0.52 and batch_std <= 0.04
            else:
                use_adaptive = True
            if use_adaptive:
                calibrated = apply_batch_adaptive_logit(
                    batch,
                    target_median=target_median,
                    temperature=temperature,
                )
            else:
                calibrated = apply_score_logit_calibration(
                    batch,
                    bias=bias,
                    temperature=temperature,
                )
        else:
            calibrated = apply_score_logit_calibration(
                batch,
                bias=bias,
                temperature=temperature,
            )
        output[start:stop] = apply_live_positive_cap(
            calibrated,
            max_positive_rate=max_positive_rate,
            hard_ceiling=hard_ceiling,
        )
    return output

