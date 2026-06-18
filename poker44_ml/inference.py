from __future__ import annotations

import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

from poker44_ml.calibration import (
    apply_batch_adaptive_logit,
    apply_batch_median_center,
    apply_batch_quantile_spread,
    apply_score_logit_calibration,
)
from poker44_ml.features import chunk_features
from poker44_ml.rank_stack import batch_rank_boost

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "bot_detector_v1.joblib"
SCORE_LOG_DECIMALS = 8


class Poker44Model:
    """Runtime wrapper for reference-compatible Poker44 artifacts."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        if joblib is None:
            raise RuntimeError("joblib is required to load Poker44 models.")
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")

        artifact = joblib.load(self.model_path)
        self.models = list(artifact.get("models") or [])
        if not self.models and artifact.get("model") is not None:
            self.models = [artifact["model"]]
        if not self.models:
            raise RuntimeError("Model artifact contains no models.")

        self.feature_names = list(artifact.get("feature_names") or [])
        self.metadata = dict(artifact.get("metadata") or {})
        self.metrics = dict(artifact.get("metrics") or self.metadata.get("metrics") or {})
        self.model_version = str(
            artifact.get("model_version")
            or self.metadata.get("model_version")
            or "reference-stack"
        )
        self.calibrator = artifact.get("calibrator")
        self.score_logit_bias = float(self.metadata.get("score_logit_bias", 0.0) or 0.0)
        self.score_logit_temperature = max(
            float(self.metadata.get("score_logit_temperature", 1.0) or 1.0),
            1e-6,
        )
        score_remap = self.metadata.get("score_remap")
        if isinstance(score_remap, dict) and score_remap.get("kind"):
            self.score_remap: dict[str, Any] = score_remap
        elif (
            isinstance(self.calibrator, dict)
            and self.calibrator.get("kind") == "threshold_logit_v1"
        ):
            self.score_remap = dict(self.calibrator)
            self.calibrator = None
        else:
            self.score_remap = {}
        self.live_batch_spread = bool(self.metadata.get("live_batch_spread", True))
        self.live_batch_spread_blend = float(
            self.metadata.get("live_batch_spread_blend", 0.85) or 0.85
        )
        spread_low = self.metadata.get("live_batch_spread_low")
        spread_high = self.metadata.get("live_batch_spread_high")
        self.live_batch_spread_low = (
            float(spread_low) if spread_low is not None else None
        )
        self.live_batch_spread_high = (
            float(spread_high) if spread_high is not None else None
        )
        self.live_batch_center = bool(self.metadata.get("live_batch_center", False))
        self.live_batch_center_blend = float(
            self.metadata.get("live_batch_center_blend", 0.75) or 0.75
        )
        self.live_logit_mode = str(self.metadata.get("live_logit_mode", "auto") or "auto")
        if "live_adaptive_logit" in self.metadata:
            self.live_logit_mode = (
                "adaptive" if bool(self.metadata.get("live_adaptive_logit")) else "fixed"
            )
        self.live_logit_target_median = float(
            self.metadata.get("live_logit_target_median", 0.28) or 0.28
        )
        self.live_logit_collapse_median = float(
            self.metadata.get("live_logit_collapse_median", 0.52) or 0.52
        )
        self.live_logit_collapse_std = float(
            self.metadata.get("live_logit_collapse_std", 0.04) or 0.04
        )
        self.live_batch_rank_boost = float(
            self.metadata.get("live_batch_rank_boost", 0.0) or 0.0
        )
        self.model_weights = list(
            artifact.get("model_weights")
            or self.metadata.get("model_weights")
            or [1.0 for _ in self.models]
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-40.0, min(40.0, float(value)))
        return 1.0 / (1.0 + math.exp(-value))

    def _aligned_rows(self, chunks: list[list[dict[str, Any]]]) -> list[list[float]]:
        rows: list[list[float]] = []
        for chunk in chunks:
            features = chunk_features(chunk)
            features["hand_count"] = float(len(chunk))
            if not self.feature_names:
                self.feature_names = sorted(features)
            rows.append([float(features.get(name, 0.0)) for name in self.feature_names])
        return rows

    def _raw_model_scores(
        self,
        rows: list[list[float]],
        chunks: list[list[dict[str, Any]]] | None = None,
    ) -> list[float]:
        per_model: list[list[float]] = []
        for model in self.models:
            if (
                chunks is not None
                and hasattr(model, "predict_chunk_scores")
                and not isinstance(model, type(self))
            ):
                raw = model.predict_chunk_scores(chunks, feature_rows=rows)
                per_model.append([self._clamp01(float(value)) for value in raw])
                continue
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(rows)
                per_model.append([self._clamp01(row[1]) for row in probabilities])
            elif hasattr(model, "decision_function"):
                decisions = model.decision_function(rows)
                per_model.append([self._sigmoid(value) for value in decisions])
            else:
                raw_vals = list(model.predict(rows))
                # LGBMRanker produces unbounded ranking scores — normalise per batch
                raw_arr = [float(v) for v in raw_vals]
                lo = min(raw_arr)
                hi = max(raw_arr)
                rng = hi - lo
                if rng > 1e-9:
                    raw_arr = [(v - lo) / rng for v in raw_arr]
                per_model.append([self._clamp01(v) for v in raw_arr])

        weights = [max(0.0, float(value)) for value in self.model_weights[: len(per_model)]]
        if len(weights) != len(per_model) or sum(weights) <= 0.0:
            weights = [1.0 for _ in per_model]
        total_weight = sum(weights)

        scores: list[float] = []
        for row_index in range(len(rows)):
            value = sum(
                weight * model_scores[row_index]
                for weight, model_scores in zip(weights, per_model)
            ) / total_weight
            scores.append(self._clamp01(value))
        return scores

    def _apply_calibrator(self, scores: list[float]) -> list[float]:
        if not scores or self.calibrator is None:
            return [self._clamp01(value) for value in scores]
        if hasattr(self.calibrator, "predict_proba"):
            calibrated = self.calibrator.predict_proba([[float(value)] for value in scores])
            return [self._clamp01(row[1]) for row in calibrated]
        if hasattr(self.calibrator, "transform"):
            return [self._clamp01(value) for value in self.calibrator.transform(scores)]
        return [self._clamp01(value) for value in scores]

    def _apply_batch_spread(self, scores: list[float]) -> list[float]:
        if not scores or not self.live_batch_spread:
            return [self._clamp01(value) for value in scores]
        values = np.asarray(scores, dtype=float)
        if self.live_batch_center:
            values = apply_batch_median_center(
                values,
                blend=self.live_batch_center_blend,
            )
        spread = apply_batch_quantile_spread(
            values,
            blend=self.live_batch_spread_blend,
            spread_low=self.live_batch_spread_low,
            spread_high=self.live_batch_spread_high,
        )
        return [self._clamp01(float(value)) for value in spread]

    def _apply_score_remap(self, scores: list[float]) -> list[float]:
        if not scores or not self.score_remap:
            return [self._clamp01(value) for value in scores]
        if self.score_remap.get("kind") == "threshold_logit_v1":
            # Benchmark-only remap; it collapses live batches when raw scores sit below
            # the tuned threshold.
            return [self._clamp01(value) for value in scores]
        return [self._clamp01(value) for value in scores]

    def _should_use_adaptive_logit(self, values: np.ndarray) -> bool:
        mode = self.live_logit_mode.lower()
        if mode == "adaptive":
            return True
        if mode == "fixed":
            return False
        if values.size <= 1:
            return False
        median = float(np.median(values))
        spread = float(values.std())
        return (
            median >= self.live_logit_collapse_median
            and spread <= self.live_logit_collapse_std
        )

    def _apply_score_logit(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        values = np.asarray(scores, dtype=float)
        if self._should_use_adaptive_logit(values):
            calibrated = apply_batch_adaptive_logit(
                values,
                target_median=self.live_logit_target_median,
                temperature=self.score_logit_temperature,
            )
            return [self._clamp01(float(value)) for value in calibrated]
        if abs(self.score_logit_bias) < 1e-12 and abs(self.score_logit_temperature - 1.0) < 1e-12:
            return [self._clamp01(value) for value in scores]
        calibrated = apply_score_logit_calibration(
            values,
            bias=self.score_logit_bias,
            temperature=self.score_logit_temperature,
        )
        return [self._clamp01(float(value)) for value in calibrated]

    def _feature_dicts(self, chunks: list[list[dict[str, Any]]]) -> list[dict[str, float]]:
        feature_rows: list[dict[str, float]] = []
        for chunk in chunks:
            features = chunk_features(chunk)
            features["hand_count"] = float(len(chunk))
            feature_rows.append(features)
        return feature_rows

    def predict_chunk_scores(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        if not chunks:
            return []
        rows = self._aligned_rows(chunks)
        feature_rows = self._feature_dicts(chunks)
        raw_scores = self._raw_model_scores(rows, chunks=chunks)
        if self.live_batch_rank_boost > 0.0:
            raw_scores = batch_rank_boost(
                feature_rows,
                raw_scores,
                blend=self.live_batch_rank_boost,
            )
        calibrated_scores = self._apply_calibrator(raw_scores)
        spread_scores = self._apply_batch_spread(calibrated_scores)
        remapped_scores = self._apply_score_remap(spread_scores)
        logit_scores = self._apply_score_logit(remapped_scores)
        return [round(self._clamp01(value), 6) for value in logit_scores]

    def predict_chunk_score(self, chunk: list[dict[str, Any]]) -> float:
        scores = self.predict_chunk_scores([chunk])
        return scores[0] if scores else 0.5

    def score_chunk(self, chunk: list[dict[str, Any]]) -> float:
        return self.predict_chunk_score(chunk)

    def _round_score_log_values(self, scores: list[float]) -> list[float]:
        return [round(float(value), int(SCORE_LOG_DECIMALS)) for value in scores]

    def debug_score_components(
        self,
        chunks: list[list[dict[str, Any]]],
    ) -> dict[str, list[float]]:
        if not chunks:
            return {}
        rows = self._aligned_rows(chunks)
        feature_rows = self._feature_dicts(chunks)
        raw_scores = self._raw_model_scores(rows, chunks=chunks)
        if self.live_batch_rank_boost > 0.0:
            raw_scores = batch_rank_boost(
                feature_rows,
                raw_scores,
                blend=self.live_batch_rank_boost,
            )
        calibrated_scores = self._apply_calibrator(raw_scores)
        spread_scores = self._apply_batch_spread(calibrated_scores)
        remapped_scores = self._apply_score_remap(spread_scores)
        logit_scores = self._apply_score_logit(remapped_scores)
        return {
            "raw_scores": self._round_score_log_values(raw_scores),
            "spread_scores": self._round_score_log_values(spread_scores),
            "remapped_scores": self._round_score_log_values(remapped_scores),
            "final_scores": self._round_score_log_values(logit_scores),
        }

    def benchmark_latency(
        self,
        chunks: list[list[dict[str, Any]]],
        repeats: int = 5,
    ) -> dict[str, float]:
        if not chunks:
            return {"latency_per_chunk_ms": 0.0, "total_latency_ms": 0.0}
        repeats = max(1, int(repeats))
        started = time.perf_counter()
        for _ in range(repeats):
            self.predict_chunk_scores(chunks)
        elapsed_ms = (time.perf_counter() - started) * 1000.0 / repeats
        return {
            "latency_per_chunk_ms": elapsed_ms / max(len(chunks), 1),
            "total_latency_ms": elapsed_ms,
        }

