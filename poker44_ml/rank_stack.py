"""Batch-grouped learning-to-rank adapters for Poker44 inference."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class RankModelAdapter:
    """Expose ranker raw scores as bounded probabilities for StackedEnsemble."""

    def __init__(self, ranker: Any) -> None:
        self.ranker = ranker

    def predict(self, x: Any) -> np.ndarray:
        return _sigmoid(np.asarray(self.ranker.predict(x), dtype=float))

    def predict_proba(self, x: Any) -> np.ndarray:
        positive = self.predict(x)
        return np.column_stack([1.0 - positive, positive])


def batch_groups_from_metadata(metadata: Sequence[dict]) -> np.ndarray:
    chunk_ids = [str(row["chunk_id"]) for row in metadata]
    lookup = {chunk_id: index for index, chunk_id in enumerate(sorted(set(chunk_ids)))}
    return np.asarray([lookup[chunk_id] for chunk_id in chunk_ids], dtype=int)


def sort_rows_by_groups(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    _, group_sizes = np.unique(sorted_groups, return_counts=True)
    return x[order], y[order], sorted_groups, group_sizes


def fit_ranker(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> Any:
    x_sorted, y_sorted, _, group_sizes = sort_rows_by_groups(x, y, groups)
    return model.fit(x_sorted, y_sorted, group=group_sizes)


def predict_rank_scores(model: Any, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(x), dtype=float)


def predict_rank_proba(model: Any, x: np.ndarray) -> np.ndarray:
    return RankModelAdapter(model).predict(x)


_BATCH_RANK_SIGNALS: tuple[tuple[str, float], ...] = (
    ("schema_aggression_share_mean", 1.0),
    ("schema_raise_share_mean", 1.0),
    ("schema_high_aggression_hand_rate", 1.0),
    ("schema_call_to_share_min", -1.0),
    ("schema_check_share_mean", -1.0),
    ("schema_action_entropy_mean", -1.0),
)


def batch_rank_boost(
    feature_rows: Sequence[dict[str, float]],
    raw_scores: Sequence[float],
    *,
    blend: float = 0.12,
) -> list[float]:
    """Nudge raw scores using within-batch feature ranks."""
    blend = float(max(0.0, min(0.5, blend)))
    if blend <= 0.0 or len(feature_rows) <= 1:
        return [float(value) for value in raw_scores]

    import numpy as np

    base = np.clip(np.asarray(raw_scores, dtype=float), 0.0, 1.0)
    rank_parts: list[np.ndarray] = []
    for name, direction in _BATCH_RANK_SIGNALS:
        values = np.asarray(
            [float(row.get(name, 0.0)) for row in feature_rows],
            dtype=float,
        )
        if float(values.std()) < 1e-12:
            continue
        order = values.argsort(kind="stable")
        ranks = np.empty_like(values)
        ranks[order] = np.linspace(0.0, 1.0, len(values))
        if direction < 0:
            ranks = 1.0 - ranks
        rank_parts.append(ranks)
    if not rank_parts:
        return [float(value) for value in base]
    rank_score = np.mean(np.stack(rank_parts, axis=0), axis=0)
    mixed = (1.0 - blend) * base + blend * rank_score
    return [float(value) for value in np.clip(mixed, 0.0, 1.0)]
