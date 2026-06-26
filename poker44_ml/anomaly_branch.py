"""One-class human baseline via Isolation Forest — catches novel bots (SN126 hybrid pattern)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class HumanIsolationScorer:
    """Train on humans only; high bot probability when anomaly score is low."""

    forest: IsolationForest
    p_min: float
    p1: float
    p_max: float
    feature_names: list[str]

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
        *,
        seed: int = 42,
        contamination: float = 0.02,
    ) -> "HumanIsolationScorer":
        human_mask = y == 0
        x_human = np.asarray(x[human_mask], dtype=float)
        if len(x_human) < 20:
            raise ValueError("Need at least 20 human rows to fit isolation forest.")
        forest = IsolationForest(
            n_estimators=200,
            max_samples=min(256, len(x_human)),
            contamination=float(contamination),
            random_state=int(seed),
            n_jobs=1,
        )
        forest.fit(x_human)
        raw = forest.score_samples(x_human)
        p_min = float(np.min(raw))
        p1 = float(np.quantile(raw, 0.01))
        p_max = float(np.max(raw))
        return cls(
            forest=forest,
            p_min=p_min,
            p1=p1,
            p_max=p_max,
            feature_names=list(feature_names),
        )

    def _anomaly_to_botprob(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        span_normal = max(0.001, self.p_max - self.p1)
        span_anom = max(0.001, self.p1 - self.p_min)
        bp_normal = 0.5 - (scores - self.p1) / span_normal * 0.5
        bp_anom = 0.5 + (self.p1 - scores) / span_anom * 0.5
        return np.where(
            scores >= self.p1,
            np.clip(bp_normal, 0.0, 0.5),
            np.clip(bp_anom, 0.5, 1.0),
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        raw = self.forest.score_samples(x)
        return self._anomaly_to_botprob(raw)

    def predict_bot_scores(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self.predict_proba(x), 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "forest": self.forest,
            "p_min": self.p_min,
            "p1": self.p1,
            "p_max": self.p_max,
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HumanIsolationScorer":
        return cls(
            forest=payload["forest"],
            p_min=float(payload["p_min"]),
            p1=float(payload["p1"]),
            p_max=float(payload["p_max"]),
            feature_names=list(payload.get("feature_names") or []),
        )


def fuse_hybrid_scores(
    supervised: np.ndarray,
    anomaly: np.ndarray,
    *,
    mode: str = "max",
    anomaly_weight: float = 1.0,
) -> np.ndarray:
    sup = np.clip(np.asarray(supervised, dtype=float), 0.0, 1.0)
    ano = np.clip(np.asarray(anomaly, dtype=float), 0.0, 1.0)
    if mode == "blend":
        w = float(max(0.0, min(1.0, anomaly_weight)))
        return np.clip((1.0 - w) * sup + w * ano, 0.0, 1.0)
    return np.maximum(sup, ano)
