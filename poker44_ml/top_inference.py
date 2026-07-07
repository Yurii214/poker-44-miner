"""Serving head for the rank-first top-miner ensemble.

The subnet reward is ``0.75*AP + 0.25*recall@FPR<=5%`` and is purely rank-based.
So the serving score for each chunk is its **within-batch rank** among all
chunks in the same validator request, rescaled to [0, 1]:

* monotone in the model's raw probability -> cannot reduce AP;
* tie-free -> avoids the averaged-rank AP penalty that compressed scores cause;
* immune to calibration drift -> only order matters, and order is preserved.

This deliberately replaces the old positive-rate caps / regime ceilings, which
were built for the retired multiplicative-FPR-penalty reward and actively hurt
average precision by collapsing the score distribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOP_MODEL_PATH = REPO_ROOT / "models" / "bot_detector_top.joblib"


def within_batch_rank_scores(raw: Sequence[float]) -> list[float]:
    """Map raw scores to their within-batch rank in [0, 1] (tie-free)."""
    n = len(raw)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    order = np.argsort(np.asarray(raw, dtype=float), kind="mergesort")  # ascending
    out = [0.0] * n
    for pos, i in enumerate(order):
        out[int(i)] = pos / (n - 1)
    return out


class TopMinerModel:
    """Loads the ``TopEnsemble`` artifact and serves rank-normalised scores."""

    def __init__(self, model_path: str | Path = DEFAULT_TOP_MODEL_PATH):
        import joblib

        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path)
        self.feature_names: list[str] = list(self.model.feature_names)
        self.metadata: dict[str, Any] = dict(getattr(self.model, "metadata", {}) or {})
        # Compatibility surface expected by neurons/miner.py.
        self.model_version: str = str(self.metadata.get("model_version", "top-v1"))
        self.model_name: str = str(self.metadata.get("model_name", "poker44-rankfirst-ensemble"))
        self.metrics: dict[str, Any] = dict(self.metadata.get("metrics", {}) or {})
        self.live_regime_enabled: bool = False

    def _features(self, chunks: Sequence[Sequence[dict]]) -> list[dict[str, float]]:
        from poker44_ml.rank_features import top_chunk_features

        # Live chunks arrive already miner-visible (validator-sanitised).
        return [top_chunk_features(chunk, already_prepared=True) for chunk in chunks]

    def predict_raw_scores(self, chunks: Sequence[Sequence[dict]]) -> list[float]:
        if not chunks:
            return []
        feats = self._features(chunks)
        return [float(v) for v in self.model.predict_from_dicts(feats)]

    def predict_chunk_scores(self, chunks: Sequence[Sequence[dict]]) -> list[float]:
        """Return one within-batch-rank-normalised bot-risk score per chunk."""
        raw = self.predict_raw_scores(chunks)
        ranked = within_batch_rank_scores(raw)
        return [round(float(v), 6) for v in ranked]
