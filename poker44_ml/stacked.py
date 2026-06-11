"""Picklable stacked ensemble compatible with the Poker44Model loader."""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


class StackedEnsemble:
    """OOF-stacked base models plus optional monotone post-calibration."""

    def __init__(
        self,
        base_models: Sequence[Any],
        meta_model: Any,
        calibrator: Optional[Any] = None,
        feature_indices: Optional[Sequence[int]] = None,
        score_shift: float = 0.0,
        chunk_models: Optional[Sequence[Any]] = None,
        stack_mode: str = "mean",
    ) -> None:
        self.base_models: List[Any] = list(base_models)
        self.chunk_models: List[Any] = list(chunk_models or [])
        self.meta_model = meta_model
        self.stack_mode = str(stack_mode or "mean")
        self.calibrator = calibrator
        self.feature_indices: Optional[np.ndarray] = (
            np.asarray(list(feature_indices), dtype=np.int64)
            if feature_indices is not None
            else None
        )
        self.score_shift = float(score_shift)

    def _select_features(self, x: np.ndarray) -> np.ndarray:
        if self.feature_indices is None:
            return x
        return x[:, self.feature_indices]

    def _base_probs(self, x: np.ndarray) -> np.ndarray:
        cols: List[np.ndarray] = []
        for model in self.base_models:
            if hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(x))
                col = proba[:, 1] if proba.ndim == 2 else proba
            elif hasattr(model, "decision_function"):
                raw = np.asarray(model.decision_function(x), dtype=float)
                col = 1.0 / (1.0 + np.exp(-np.clip(raw, -40.0, 40.0)))
            else:
                col = np.asarray(model.predict(x), dtype=float)
            cols.append(np.clip(np.asarray(col, dtype=float), 0.0, 1.0))
        return np.stack(cols, axis=1)

    def _chunk_probs(self, chunks: Sequence[Any]) -> np.ndarray:
        if not self.chunk_models:
            return np.zeros((len(chunks), 0), dtype=float)
        cols: List[np.ndarray] = []
        for model in self.chunk_models:
            if hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(chunks))
                col = proba[:, 1] if proba.ndim == 2 else proba
            elif hasattr(model, "predict_chunk_scores"):
                col = np.asarray(model.predict_chunk_scores(chunks), dtype=float)
            else:
                raise RuntimeError(
                    f"Chunk model {type(model).__name__} exposes neither "
                    "predict_proba nor predict_chunk_scores."
                )
            cols.append(np.clip(np.asarray(col, dtype=float), 0.0, 1.0))
        return np.stack(cols, axis=1)

    def base_score_matrix(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        return self._base_probs(self._select_features(x_arr))

    def _combine_base_scores(self, z: np.ndarray) -> np.ndarray:
        if z.size == 0:
            return np.zeros(z.shape[0], dtype=float)
        if self.stack_mode == "meta" and self.meta_model is not None:
            meta_proba = np.asarray(self.meta_model.predict_proba(z))
            return meta_proba[:, 1] if meta_proba.ndim == 2 else meta_proba
        return np.mean(z, axis=1)

    def _meta_to_output(self, z: np.ndarray) -> np.ndarray:
        p1 = np.asarray(self._combine_base_scores(z), dtype=float)
        if self.calibrator is not None:
            if hasattr(self.calibrator, "transform"):
                p1 = np.asarray(self.calibrator.transform(p1), dtype=float)
            elif hasattr(self.calibrator, "predict"):
                p1 = np.asarray(self.calibrator.predict(p1), dtype=float)
        if self.score_shift:
            p1 = self._logit_shift(p1, self.score_shift)
        return np.clip(p1, 0.0, 1.0)

    def predict_proba(self, x: Any) -> np.ndarray:
        if self.chunk_models:
            raise RuntimeError(
                "StackedEnsemble has chunk-based learners; use "
                "predict_chunk_scores(chunks) instead of predict_proba(rows)."
            )
        x_arr = np.asarray(x, dtype=np.float64)
        z = self._base_probs(self._select_features(x_arr))
        p1 = self._meta_to_output(z)
        return np.stack([1.0 - p1, p1], axis=1)

    def predict_chunk_scores(
        self,
        chunks: Sequence[Any],
        feature_rows: Any,
    ) -> List[float]:
        x_arr = np.asarray(feature_rows, dtype=np.float64)
        feature_probs = (
            self._base_probs(self._select_features(x_arr))
            if self.base_models
            else np.zeros((len(chunks), 0), dtype=float)
        )
        chunk_probs = self._chunk_probs(list(chunks))
        if feature_probs.size == 0 and chunk_probs.size == 0:
            raise RuntimeError("No base or chunk models are available for scoring.")
        if chunk_probs.size == 0:
            stacked = feature_probs
        elif feature_probs.size == 0:
            stacked = chunk_probs
        else:
            stacked = np.concatenate([feature_probs, chunk_probs], axis=1)
        return [float(value) for value in self._meta_to_output(stacked)]

    def predict(self, x: Any) -> np.ndarray:
        proba = self.predict_proba(x)[:, 1]
        return (proba >= 0.5).astype(int)

    @staticmethod
    def _logit_shift(values: np.ndarray, shift: float) -> np.ndarray:
        clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)) + float(shift)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))

