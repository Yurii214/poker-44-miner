#!/usr/bin/env python3
"""Train the rank-first top-miner ensemble on the public Poker44 benchmark.

Pipeline:
  1. Pull all benchmark releases (api.poker44.net/api/v1/benchmark), cached to disk.
  2. Featurise every inner chunk on the EXACT miner-visible payload
     (prepare_hand_for_miner) -> scale-invariant behavioural + collision +
     temporal + pot-fraction features (no train/serve skew).
  3. Weight humans x HUMAN_W and the most-recent dates x RECENCY_W.
  4. Hold out the N most recent dates; report the true rank-first reward on them.
  5. Refit on ALL data and save a serving artifact.

Usage:
  python scripts/train_top_miner.py [--min-source-date YYYY-MM-DD]
      [--holdout-dates 2] [--human-weight 20] [--recency-weight 6]
      [--recency-days 8] [--output models/bot_detector_top.joblib]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker44_ml.rank_features import top_chunk_features
from poker44_ml.rank_reward import rank_reward
from poker44_ml.robust_features import filter_robust_feature_names
from poker44_ml.top_model import TopEnsemble, train_top_ensemble

API_BASE = "https://api.poker44.net/api/v1/benchmark"
CACHE_DIR = REPO_ROOT / "models" / "benchmark_cache"
DEFAULT_OUTPUT = REPO_ROOT / "models" / "bot_detector_top.joblib"


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def _cached_chunk_detail(chunk_id: str) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{chunk_id}.json.gz"
    if path.is_file():
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    detail = _get_json(f"{API_BASE}/chunks/{chunk_id}")["data"]
    with gzip.open(path, "wt") as fh:
        json.dump(detail, fh)
    return detail


def build_dataset(min_source_date: str | None):
    releases = _get_json(f"{API_BASE}/releases")["data"]["releases"]
    if min_source_date:
        releases = [r for r in releases if str(r.get("sourceDate", "")) >= str(min_source_date)]
    releases.sort(key=lambda r: str(r.get("sourceDate", "")))

    feats: list[dict[str, float]] = []
    labels: list[int] = []
    dates: list[str] = []

    for ri, release in enumerate(releases, 1):
        sd = release["sourceDate"]
        outer = _get_json(f"{API_BASE}/chunks?sourceDate={sd}")["data"]["chunks"]
        for outer_chunk in outer:
            detail = _cached_chunk_detail(outer_chunk["chunkId"])
            gt = detail.get("groundTruth") or []
            inner = detail.get("chunks") or []
            for idx, inner_chunk in enumerate(inner):
                if idx >= len(gt):
                    continue
                hands = [h for h in inner_chunk if isinstance(h, dict)]
                if not hands:
                    continue
                feats.append(top_chunk_features(hands))  # applies miner view
                labels.append(int(gt[idx]))
                dates.append(sd)
        print(f"  [{ri}/{len(releases)}] {sd}: cumulative chunks={len(labels)}", flush=True)

    if not feats:
        raise RuntimeError("No benchmark training samples fetched.")
    return feats, np.asarray(labels, dtype=int), np.asarray(dates)


def select_feature_names(feature_dicts: list[dict[str, float]]) -> list[str]:
    all_names = sorted({n for d in feature_dicts for n in d})
    robust = set(filter_robust_feature_names(all_names))
    # Always include our scale-invariant supplementary families.
    robust.update(n for n in all_names if n.startswith("xseq_"))
    robust.discard("hand_count")  # absolute count is not scale-invariant
    return sorted(robust)


def vectorize(feature_dicts, names):
    return np.asarray([[float(d.get(n, 0.0)) for n in names] for d in feature_dicts], dtype=float)


def make_weights(labels, dates, *, human_w, recency_w, recency_days):
    uniq = sorted(set(dates.tolist()))
    recent = set(uniq[-recency_days:]) if recency_days > 0 else set()
    w = np.ones(len(labels), dtype=float)
    w[labels == 0] *= float(human_w)          # protect humans -> sharpen FPR term
    for i, d in enumerate(dates):
        if d in recent:
            w[i] *= float(recency_w)          # track drifting live bots
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-source-date", default=None)
    ap.add_argument("--holdout-dates", type=int, default=2)
    ap.add_argument("--human-weight", type=float, default=20.0)
    ap.add_argument("--recency-weight", type=float, default=6.0)
    ap.add_argument("--recency-days", type=int, default=8)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    print("Fetching + featurising benchmark data...", flush=True)
    feats, labels, dates = build_dataset(args.min_source_date)
    names = select_feature_names(feats)
    x = vectorize(feats, names)
    print(f"Dataset: {len(labels)} chunks, {int((labels==1).sum())} bot / "
          f"{int((labels==0).sum())} human, {len(names)} features, "
          f"{len(set(dates.tolist()))} dates.", flush=True)

    uniq = sorted(set(dates.tolist()))
    holdout = set(uniq[-args.holdout_dates:]) if args.holdout_dates > 0 else set()
    is_hold = np.array([d in holdout for d in dates])
    weights = make_weights(labels, dates, human_w=args.human_weight,
                           recency_w=args.recency_weight, recency_days=args.recency_days)

    # --- Temporal-holdout evaluation: train on past, score the newest dates ---
    if holdout and is_hold.any() and (~is_hold).any():
        print(f"Holdout dates: {sorted(holdout)} ({int(is_hold.sum())} chunks)", flush=True)
        ens = train_top_ensemble(
            x[~is_hold], labels[~is_hold], dates[~is_hold], names,
            sample_weight=weights[~is_hold],
        )
        hold_scores = ens.predict_proba_raw(x[is_hold])
        reward, detail = rank_reward(hold_scores, labels[is_hold])
        print("\n=== HOLDOUT (rank-first reward) ===", flush=True)
        print(f"  reward            = {detail['reward']:.4f}", flush=True)
        print(f"  average_precision = {detail['average_precision']:.4f}", flush=True)
        print(f"  recall@FPR<=5%    = {detail['recall_at_fpr5']:.4f}", flush=True)

    # --- Production model: refit on ALL data ---
    print("\nTraining production ensemble on all data...", flush=True)
    prod = train_top_ensemble(x, labels, dates, names, sample_weight=weights)
    prod.metadata.update({
        "model_name": "poker44-rankfirst-ensemble",
        "model_version": "top-v1",
        "n_chunks": int(len(labels)),
        "n_features": len(names),
        "source_dates": uniq,
        "human_weight": args.human_weight,
        "recency_weight": args.recency_weight,
        "recency_days": args.recency_days,
        "serving": "within_batch_rank_normalized",
    })

    import joblib
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(prod, args.output)
    print(f"Saved artifact -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
