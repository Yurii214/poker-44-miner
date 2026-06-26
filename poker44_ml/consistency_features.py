"""Cross-hand consistency features — strong bot tells from public SN126 miners."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from poker44_ml.features import _safe_div, _safe_float, _safe_int


def cross_hand_consistency_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    """Chunk-level signals: low decision entropy and bet-size quantization imply bots."""
    if not chunk:
        return {}

    decisions: dict[tuple[str, int], list[str]] = {}
    pot_fracs: list[float] = []
    bet_sizes_bb: list[float] = []
    bigrams: Counter[tuple[str, str]] = Counter()
    actions_total = 0

    for hand in chunk:
        if not isinstance(hand, dict):
            continue
        for action in hand.get("actions") or []:
            if not isinstance(action, dict):
                continue
            at = str(action.get("action_type") or "").lower().strip()
            seat = _safe_int(action.get("actor_seat"), 0)
            street = str(action.get("street") or "").lower().strip()
            key = (street, seat)
            decisions.setdefault(key, []).append(at)
            amt = max(0.0, _safe_float(action.get("normalized_amount_bb")))
            pot_before = _safe_float(action.get("pot_before"))
            pot_bb = pot_before / 0.02 if pot_before > 0 else 0.0
            if amt > 0:
                bet_sizes_bb.append(amt)
            if pot_bb > 0 and amt > 0:
                pot_fracs.append(amt / pot_bb)
            actions_total += 1
        types = [
            str((a or {}).get("action_type") or "").lower().strip()
            for a in hand.get("actions") or []
        ]
        for bg in zip(types[:-1], types[1:]):
            bigrams[bg] += 1

    feats: dict[str, float] = {"sig_actions_total": float(actions_total)}
    ents: list[float] = []
    bucket_sizes: list[int] = []
    for lst in decisions.values():
        counts = Counter(lst)
        n = sum(counts.values())
        bucket_sizes.append(n)
        if n <= 1:
            continue
        p = np.asarray(list(counts.values()), dtype=float) / n
        ents.append(float(-np.sum(p * np.log(p + 1e-12))))
    feats["sig_decision_buckets"] = float(len(decisions))
    feats["sig_decision_ent_mean"] = float(np.mean(ents)) if ents else 0.0
    feats["sig_decision_ent_std"] = float(np.std(ents)) if len(ents) > 1 else 0.0
    feats["sig_decision_bucket_size_mean"] = _safe_div(sum(bucket_sizes), len(bucket_sizes))

    if bet_sizes_bb:
        bs = np.asarray(bet_sizes_bb, dtype=float)
        feats["sig_betbb_cv"] = float(np.std(bs) / (np.mean(bs) + 1e-9))
        feats["sig_betbb_uniq_round"] = float(len({round(b, 1) for b in bs}))
    else:
        feats["sig_betbb_cv"] = 0.0
        feats["sig_betbb_uniq_round"] = 0.0

    if pot_fracs:
        pf = np.asarray(pot_fracs, dtype=float)
        feats["sig_potfrac_cv"] = float(np.std(pf) / (np.mean(pf) + 1e-9))
        snap = [round(p * 20) / 20 for p in pf]
        feats["sig_potfrac_snap_uniq"] = float(len(set(snap)))
        for target in (0.5, 0.66, 0.75, 1.0):
            feats[f"sig_potfrac_near_{int(target * 100):03d}"] = float(
                np.mean([abs(p - target) < 0.07 for p in pf])
            )
    else:
        feats["sig_potfrac_cv"] = 0.0
        feats["sig_potfrac_snap_uniq"] = 0.0
        for target in (0.5, 0.66, 0.75, 1.0):
            feats[f"sig_potfrac_near_{int(target * 100):03d}"] = 0.0

    feats["sig_bigram_uniq"] = float(len(bigrams))
    if bigrams:
        counts = np.asarray(list(bigrams.values()), dtype=float)
        p = counts / counts.sum()
        feats["sig_bigram_ent"] = float(-np.sum(p * np.log(p + 1e-12)))
    else:
        feats["sig_bigram_ent"] = 0.0

    return feats
