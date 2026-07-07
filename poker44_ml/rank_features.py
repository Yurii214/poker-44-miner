"""Supplementary scale-invariant chunk features for the rank-first competition.

These add the high-signal families the dual-branch model lacked, mirroring what
the leading open-source SN126 miners rely on:

* temporal-consistency / autocorrelation of a player's per-hand behaviour
  (bots hold a fixed strategy -> high autocorrelation, ~0 drift);
* intra-session drift (early-vs-late quartile behaviour);
* pot-fraction bet-size clustering (bots snap bets to modal pot fractions).

Everything here is deliberately scale-invariant (shares, CVs, entropies,
autocorrelations) so it does not degrade on the validator-sanitised live feed.
All keys are prefixed ``xseq_`` and merge on top of ``chunk_features``.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

_STD_POT_FRACTIONS = (0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.5, 2.0)
_POT_CLUSTER_TOL = 0.08


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _cv(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if abs(mean) < 1e-9:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(max(0.0, var)) / abs(mean)


def _autocorr(seq: Sequence[float], lag: int) -> float:
    """Pearson autocorrelation of a sequence at the given lag, in [-1, 1]."""
    n = len(seq)
    if n <= lag + 1:
        return 0.0
    mean = sum(seq) / n
    denom = sum((v - mean) ** 2 for v in seq)
    if denom < 1e-12:
        return 0.0
    num = sum((seq[i] - mean) * (seq[i + lag] - mean) for i in range(n - lag))
    return max(-1.0, min(1.0, num / denom))


def _quartile_drift(seq: Sequence[float]) -> float:
    """Mean(last quarter) - mean(first quarter). ~0 for a fixed strategy."""
    n = len(seq)
    if n < 4:
        return 0.0
    q = max(1, n // 4)
    first = sum(seq[:q]) / q
    last = sum(seq[-q:]) / q
    return last - first


def _hand_behaviour(hand: dict[str, Any]) -> tuple[float, float, float, list[float]]:
    """Return (aggression_share, fold_share, size_cv, bet_pot_ratios) for one hand."""
    actions = hand.get("actions") or []
    types: list[str] = []
    amounts: list[float] = []
    ratios: list[float] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        atype = str(action.get("action_type") or "").lower().strip()
        types.append(atype)
        amt = max(0.0, _safe_float(action.get("normalized_amount_bb")))
        if atype in ("bet", "raise"):
            amounts.append(amt)
            before = max(0.0, _safe_float(action.get("pot_before")) / 0.02)
            if before > 1e-6:
                ratios.append(amt / before)
    counts = Counter(types)
    meaningful = max(
        counts.get("call", 0)
        + counts.get("check", 0)
        + counts.get("bet", 0)
        + counts.get("raise", 0)
        + counts.get("fold", 0),
        1,
    )
    aggression = (counts.get("bet", 0) + counts.get("raise", 0)) / meaningful
    fold = counts.get("fold", 0) / meaningful
    return aggression, fold, _cv(amounts), ratios


def extra_sequence_features(chunk: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Compute ``xseq_*`` temporal + pot-fraction features for a prepared chunk."""
    hands = [h for h in chunk if isinstance(h, dict)]
    if not hands:
        return {}

    aggr_seq: list[float] = []
    fold_seq: list[float] = []
    size_seq: list[float] = []
    all_ratios: list[float] = []
    for hand in hands:
        aggression, fold, size_cv, ratios = _hand_behaviour(hand)
        aggr_seq.append(aggression)
        fold_seq.append(fold)
        size_seq.append(size_cv)
        all_ratios.extend(ratios)

    feats: dict[str, float] = {}
    for name, seq in (("aggr", aggr_seq), ("fold", fold_seq), ("size", size_seq)):
        for lag in (1, 2, 3):
            feats[f"xseq_{name}_autocorr_l{lag}"] = _autocorr(seq, lag)
        feats[f"xseq_{name}_drift"] = _quartile_drift(seq)
        feats[f"xseq_{name}_cross_cv"] = _cv(seq)

    # Pot-fraction clustering: bots snap bet sizes to modal pot fractions.
    if all_ratios:
        near = 0
        for r in all_ratios:
            if any(abs(r - f) <= _POT_CLUSTER_TOL for f in _STD_POT_FRACTIONS):
                near += 1
        feats["xseq_bet_pot_cluster_frac"] = near / len(all_ratios)
        feats["xseq_bet_pot_ratio_cv"] = _cv(all_ratios)
        # Shannon entropy over standard-fraction buckets (low = templated bot).
        bucket_counts: Counter[int] = Counter()
        for r in all_ratios:
            best = min(range(len(_STD_POT_FRACTIONS)), key=lambda i: abs(r - _STD_POT_FRACTIONS[i]))
            bucket_counts[best] += 1
        total = sum(bucket_counts.values())
        ent = -sum((c / total) * math.log(c / total) for c in bucket_counts.values() if c)
        feats["xseq_bet_pot_bucket_entropy"] = ent / math.log(len(_STD_POT_FRACTIONS))
        feats["xseq_bet_pot_modal_frac"] = max(bucket_counts.values()) / total
    else:
        feats["xseq_bet_pot_cluster_frac"] = 0.0
        feats["xseq_bet_pot_ratio_cv"] = 0.0
        feats["xseq_bet_pot_bucket_entropy"] = 0.0
        feats["xseq_bet_pot_modal_frac"] = 0.0

    return feats


def top_chunk_features(
    chunk: Sequence[dict[str, Any]],
    *,
    already_prepared: bool = False,
) -> dict[str, float]:
    """``chunk_features`` (robust behavioural + collision signatures) plus ``xseq_*``.

    Applies ``prepare_hand_for_miner`` unless ``already_prepared`` so training
    and serving see the exact same payload (no train/serve skew).
    """
    from poker44_ml.features import chunk_features

    if not chunk:
        return {"hand_count": 0.0}
    if already_prepared:
        prepared = [h for h in chunk if isinstance(h, dict)]
    else:
        from poker44.validator.payload_view import prepare_hand_for_miner

        prepared = [prepare_hand_for_miner(h) for h in chunk if isinstance(h, dict)]

    feats = chunk_features(prepared, already_prepared=True)
    feats.update(extra_sequence_features(prepared))
    feats["hand_count"] = float(len(prepared))
    return feats
