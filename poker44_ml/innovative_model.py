from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def _predict_pos_proba(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x), dtype=float)
        if proba.ndim == 2:
            return _clip01(proba[:, 1])
        return _clip01(proba)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -40.0, 40.0)))
    return _clip01(np.asarray(model.predict(x), dtype=float))


def transform_batch_relative_rows(rows: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """Per-group robust normalization + rank percentile features.

    Fully vectorised: no Python-level column loops. Computes all 556
    argsorts simultaneously via scipy.stats.rankdata when available,
    otherwise uses a vectorised argsort approach.
    """
    x = np.asarray(rows, dtype=float)
    if x.size == 0:
        return x
    if groups is None:
        groups = np.zeros(len(x), dtype=int)
    groups = np.asarray(groups, dtype=int)
    out = np.zeros((len(x), x.shape[1] * 2), dtype=float)

    for group_id in np.unique(groups):
        mask = groups == group_id
        batch = x[mask]
        n, d = batch.shape
        if n == 0:
            continue

        # Robust normalization (vectorised over all columns at once)
        median = np.median(batch, axis=0)
        q75 = np.percentile(batch, 75, axis=0)
        q25 = np.percentile(batch, 25, axis=0)
        iqr = np.where((q75 - q25) < 1e-8, 1.0, (q75 - q25))
        robust = (batch - median) / iqr

        # Rank features — vectorised double-argsort trick
        if n == 1:
            ranks = np.full((n, d), 0.5)
        else:
            # argsort of argsort gives rank; normalise to [0, 1]
            order = np.argsort(batch, axis=0, kind="stable")        # (n, d)
            rank_mat = np.argsort(order, axis=0, kind="stable").astype(float)
            ranks = rank_mat / (n - 1)  # [0, 1]

        out[mask] = np.concatenate([robust, ranks], axis=1)
    return out


class DualBranchBatchAwareModel:
    """Blend absolute and within-batch relative classifiers."""

    def __init__(
        self,
        *,
        absolute_model: Any | None = None,
        relative_model: Any | None = None,
        absolute_models: Sequence[Any] | None = None,
        relative_models: Sequence[Any] | None = None,
        absolute_weight: float = 0.7,
    ) -> None:
        self.absolute_models = list(absolute_models or ([absolute_model] if absolute_model is not None else []))
        self.relative_models = list(relative_models or ([relative_model] if relative_model is not None else []))
        if not self.absolute_models or not self.relative_models:
            raise ValueError("DualBranchBatchAwareModel requires both absolute and relative models.")
        self.absolute_weight = float(max(0.0, min(1.0, absolute_weight)))

    def _combined(self, rows: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        abs_cols = [_predict_pos_proba(model, rows) for model in self.absolute_models]
        abs_p = np.mean(np.stack(abs_cols, axis=0), axis=0)
        rel_rows = transform_batch_relative_rows(rows, groups=groups)
        rel_cols = [_predict_pos_proba(model, rel_rows) for model in self.relative_models]
        rel_p = np.mean(np.stack(rel_cols, axis=0), axis=0)
        mixed = self.absolute_weight * abs_p + (1.0 - self.absolute_weight) * rel_p
        return _clip01(mixed)

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> np.ndarray:
        x = np.asarray(rows, dtype=float)
        p1 = self._combined(x, groups=None)
        return np.stack([1.0 - p1, p1], axis=1)

    def predict(self, rows: Sequence[Sequence[float]]) -> np.ndarray:
        return (self.predict_proba(rows)[:, 1] >= 0.5).astype(int)

    def predict_chunk_scores(
        self,
        chunks: Sequence[Any],
        feature_rows: Sequence[Sequence[float]],
    ) -> list[float]:
        x = np.asarray(feature_rows, dtype=float)
        # In live miner calls all rows are one validator batch.
        groups = np.zeros(len(x), dtype=int)
        scores = self._combined(x, groups=groups)
        return [float(value) for value in scores]
