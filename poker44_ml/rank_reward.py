"""The current SN126 rank-first reward, reproduced for local model selection.

Mirrors origin/main ``poker44/score/scoring.py``:
    reward = 0.75 * average_precision + 0.25 * recall_at_fpr(max_fpr=0.05)
Both terms are purely rank-based; there is no FPR cliff or positive-rate penalty.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def recall_at_fpr(y_score: np.ndarray, y_true: np.ndarray, *, max_fpr: float = 0.05) -> tuple[float, float]:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))
    if pos <= 0 or neg <= 0 or scores.size == 0:
        return 0.0, 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    recall = tp / max(pos, 1)
    fpr = fp / max(neg, 1)
    allowed = fpr <= float(max_fpr)
    if not np.any(allowed):
        return 0.0, 0.0
    idx = np.flatnonzero(allowed)
    best = int(idx[np.argmax(recall[allowed])])
    return float(recall[best]), float(fpr[best])


def rank_reward(y_score: np.ndarray, y_true: np.ndarray) -> tuple[float, dict]:
    scores = np.asarray(y_score, dtype=float)
    labels = np.asarray(y_true, dtype=int)
    ap = float(average_precision_score(labels, scores)) if scores.size and np.any(labels == 1) else 0.0
    recall, fpr = recall_at_fpr(scores, labels, max_fpr=0.05)
    reward = 0.75 * ap + 0.25 * recall
    return reward, {"reward": reward, "average_precision": ap, "recall_at_fpr5": recall, "fpr": fpr}
