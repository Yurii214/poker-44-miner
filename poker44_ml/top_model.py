"""Diverse OOF-stacked ensemble for rank-first SN126 bot detection.

Design (mirrors the leading open-source miners, tuned for the current reward
``0.75*average_precision + 0.25*recall@FPR<=0.05`` which is purely rank-based):

* Heterogeneous base learners across 3 tree families (LightGBM x3 with distinct
  regularisation, XGBoost, ExtraTrees, RandomForest) for ranking stability.
* Out-of-fold (grouped-by-date) stacking into a LogisticRegression meta-learner
  so the blend never sees a chunk's own fold.
* Serving emits the raw meta probability; the *within-batch rank-normalisation*
  that maximises the rank-first reward is applied at inference time
  (``poker44_ml.top_inference``), not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

import lightgbm as lgb
import xgboost as xgb


def _base_learners() -> dict[str, Any]:
    """Return a fresh, diverse set of base classifiers.

    Deliberately REGULARIZED (shallow trees, high min_child, strong L1/L2,
    feature/row subsampling) to avoid memorizing the ~700-chunk public benchmark
    -- unregularized deep trees achieved perfect benchmark separation
    (human std == 0) and then collapsed to ~0.01 on the shifted live feed.
    """
    return {
        "lgbm_a": lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=4,
            min_child_samples=40, subsample=0.8, colsample_bytree=0.6,
            reg_lambda=5.0, reg_alpha=1.0, random_state=42, n_jobs=-1, verbose=-1,
        ),
        "lgbm_b": lgb.LGBMClassifier(
            n_estimators=250, learning_rate=0.03, num_leaves=31,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
            reg_lambda=3.0, random_state=123, n_jobs=-1, verbose=-1,
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=4,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
            reg_lambda=3.0, eval_metric="logloss", random_state=31, n_jobs=-1,
        ),
        "extratrees": ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=8, max_features=0.3,
            random_state=17, n_jobs=-1,
        ),
        "randomforest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=8, max_features=0.4,
            random_state=71, n_jobs=-1,
        ),
    }


@dataclass
class TopEnsemble:
    """Trained OOF-stacked ensemble. Persist/reload via joblib."""

    feature_names: list[str]
    base_models: dict[str, Any]
    meta: LogisticRegression
    metadata: dict[str, Any] = field(default_factory=dict)

    def _stack(self, x: np.ndarray) -> np.ndarray:
        cols = [np.clip(m.predict_proba(x)[:, 1], 0.0, 1.0) for m in self.base_models.values()]
        return np.column_stack(cols)

    def predict_proba_raw(self, x: np.ndarray) -> np.ndarray:
        """Raw bot probability per row (rank-faithful, uncalibrated)."""
        stacked = self._stack(np.asarray(x, dtype=float))
        return np.clip(self.meta.predict_proba(stacked)[:, 1], 0.0, 1.0)

    def predict_from_dicts(self, feature_dicts: list[dict[str, float]]) -> np.ndarray:
        x = np.asarray(
            [[float(d.get(n, 0.0)) for n in self.feature_names] for d in feature_dicts],
            dtype=float,
        )
        return self.predict_proba_raw(x)


def train_top_ensemble(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    *,
    sample_weight: np.ndarray | None = None,
    n_splits: int = 5,
) -> TopEnsemble:
    """Fit base learners with grouped-OOF stacking, then a LogisticRegression meta."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))

    base_names = list(_base_learners().keys())
    oof = np.zeros((len(y), len(base_names)), dtype=float)
    gkf = GroupKFold(n_splits=n_splits)

    for train_idx, val_idx in gkf.split(x, y, groups):
        learners = _base_learners()
        fold_w = None if sample_weight is None else sample_weight[train_idx]
        for j, (name, model) in enumerate(learners.items()):
            _fit(model, x[train_idx], y[train_idx], fold_w)
            oof[val_idx, j] = np.clip(model.predict_proba(x[val_idx])[:, 1], 0.0, 1.0)

    meta = LogisticRegression(C=0.8, max_iter=2000)
    meta.fit(oof, y, sample_weight=sample_weight)

    # Refit base learners on all data for production.
    prod = _base_learners()
    for name, model in prod.items():
        _fit(model, x, y, sample_weight)

    return TopEnsemble(
        feature_names=list(feature_names),
        base_models=prod,
        meta=meta,
        metadata={"n_splits": n_splits, "base_learners": base_names},
    )


def _fit(model: Any, x: np.ndarray, y: np.ndarray, weight: np.ndarray | None) -> None:
    if weight is None:
        model.fit(x, y)
    else:
        model.fit(x, y, sample_weight=weight)
