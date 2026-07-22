"""Production trainer: 58 stable features + stability-filtered action n-grams.

This extends the proven benchmark-supervised recipe (scripts/train_top_miner.py)
with one new signal family: fixed-vocabulary, per-hand-normalised action n-grams
(poker44_ml/ngram_features.py). Scripted policies replay characteristic action
skeletons that marginal-rate aggregates cannot express.

The whole point is *transfer*, not benchmark score. Raw n-gram columns overfit
the benchmark and collapse on live (every chunk flagged). So we keep only the
columns that (a) shift little benchmark->live (z <= ZMAX against captured live
chunks) and (b) actually separate bots on the benchmark (bot-AP >= APMIN). That
filter is the same idea top competitors apply in their robust_features stage.

Reproducible from the committed repo: benchmark via the public API (as the
existing trainer does) and live chunks from models/live_chunks/. The resulting
frozen vocabulary is written into the artifact metadata so serving featurises
identically. Config below is fixed from the offline sweep (best HEALTHY config
that also beat baseline on the corrected upstream reward).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_top_miner as T  # noqa: E402
from poker44_ml.rank_features import top_chunk_features  # noqa: E402
from poker44_ml.ngram_features import chunk_ngram_counts  # noqa: E402
from poker44_ml.top_model import train_top_ensemble  # noqa: E402
from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402

# Frozen config from the offline sweep (see memory: competitor-synthesis).
ZMAX = 1.5
APMIN = 0.55
NGRAM_MIN_CHUNK_SUPPORT = 50
DEFAULT_OUT = REPO_ROOT / "models" / "bot_detector_top.joblib"


def _load_live_ngram_counts(vocab_probe_hands: bool = True, cap: int = 300):
    """Content-deduped live chunks -> list of (ngram Counter, n_hands)."""
    seen: set[int] = set()
    rows = []
    files = sorted(glob.glob(str(REPO_ROOT / "models/live_chunks/chunks/*.jsonl.gz")))
    for f in files[-8:]:
        with gzip.open(f, "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                for c in (json.loads(line).get("chunks") or []):
                    if not (isinstance(c, list) and c):
                        continue
                    key = hash(json.dumps(c, sort_keys=True)[:3000])
                    if key in seen:
                        continue
                    seen.add(key)
                    hands = [prepare_hand_for_miner(x) for x in c if isinstance(x, dict)]
                    rows.append((chunk_ngram_counts(hands), len(hands)))
                    if len(rows) >= cap:
                        return rows
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--human-weight", type=float, default=3.0)
    ap.add_argument("--recency-weight", type=float, default=6.0)
    ap.add_argument("--recency-days", type=int, default=8)
    ap.add_argument("--version", default="top-ngram-2026-07-22")
    args = ap.parse_args()

    # ---- benchmark: base feats + raw n-gram counts + labels/dates ----
    releases = T._get_json(f"{T.API_BASE}/releases")["data"]["releases"]
    releases.sort(key=lambda r: str(r.get("sourceDate", "")))
    base, counters, nhands, labels, dates = [], [], [], [], []
    for ri, rel in enumerate(releases, 1):
        sd = rel["sourceDate"]
        for oc in T._get_json(f"{T.API_BASE}/chunks?sourceDate={sd}")["data"]["chunks"]:
            detail = T._cached_chunk_detail(oc["chunkId"])
            gt = detail.get("groundTruth") or []
            for idx, inner in enumerate(detail.get("chunks") or []):
                if idx >= len(gt):
                    continue
                hands = [prepare_hand_for_miner(h) for h in inner if isinstance(h, dict)]
                if not hands:
                    continue
                base.append(top_chunk_features(hands, already_prepared=True))
                counters.append(chunk_ngram_counts(hands))
                nhands.append(len(hands))
                labels.append(int(gt[idx]))
                dates.append(sd)
        print(f"  [{ri}/{len(releases)}] {sd}: {len(labels)}", flush=True)

    labels = np.asarray(labels, dtype=int)
    dates = np.asarray(dates)
    names58 = T.select_feature_names(base)
    X58 = T.vectorize(base, names58)
    print(f"base features: {len(names58)}  chunks: {len(labels)}", flush=True)

    # ---- mine vocabulary (support >= N chunks) over ALL benchmark chunks ----
    support: dict[str, int] = {}
    for c in counters:
        for t in set(c):
            support[t] = support.get(t, 0) + 1
    vocab_all = sorted(t for t, n in support.items() if n >= NGRAM_MIN_CHUNK_SUPPORT)
    print(f"candidate vocab (support>={NGRAM_MIN_CHUNK_SUPPORT}): {len(vocab_all)}", flush=True)

    NG = np.zeros((len(labels), len(vocab_all)))
    for i, c in enumerate(counters):
        n = max(nhands[i], 1)
        for j, t in enumerate(vocab_all):
            NG[i, j] = c.get(t, 0) / n

    # ---- live n-gram matrix on the same vocab, for the z-shift filter ----
    live = _load_live_ngram_counts()
    LNG = np.zeros((len(live), len(vocab_all)))
    for i, (c, nh) in enumerate(live):
        n = max(nh, 1)
        for j, t in enumerate(vocab_all):
            LNG[i, j] = c.get(t, 0) / n
    print(f"live chunks for z-filter: {len(live)}", flush=True)

    # ---- filter: keep transferable (z<=ZMAX) AND bot-informative (ap>=APMIN) ----
    keep = []
    for j, t in enumerate(vocab_all):
        col = NG[:, j]
        sd_ = col.std()
        if sd_ < 1e-9:
            continue
        z = abs(col.mean() - LNG[:, j].mean()) / sd_
        apv = max(average_precision_score(labels, col), average_precision_score(labels, -col))
        if z <= ZMAX and apv >= APMIN:
            keep.append(t)
    print(f"kept n-gram tokens (z<={ZMAX}, ap>={APMIN}): {len(keep)}", flush=True)
    print(f"  {keep}", flush=True)

    keep_idx = [vocab_all.index(t) for t in keep]
    ng_names = [f"ng_{t}" for t in keep]
    X = np.hstack([X58, NG[:, keep_idx]])
    names = names58 + ng_names

    # ---- train the production ensemble on ALL data ----
    w = T.make_weights(
        labels, dates,
        human_w=args.human_weight,
        recency_w=args.recency_weight,
        recency_days=args.recency_days,
    )
    ens = train_top_ensemble(X, labels, dates, names, sample_weight=w)
    ens.metadata = dict(getattr(ens, "metadata", {}) or {})
    ens.metadata.update({
        "model_name": "poker44-innovative-dual-branch",
        "model_version": args.version,
        "serving": "raw_probability",
        "ngram_vocab": keep,
        "ngram_config": {"zmax": ZMAX, "apmin": APMIN, "min_support": NGRAM_MIN_CHUNK_SUPPORT},
        "human_weight": args.human_weight,
        "n_features": len(names),
        "n_base_features": len(names58),
        "n_ngram_features": len(ng_names),
        "n_chunks": int(len(labels)),
        "source_dates": sorted(set(dates.tolist())),
    })
    out = Path(args.out)
    joblib.dump(ens, out)
    print(f"\nsaved {out}  ({len(names)} features = {len(names58)} base + {len(ng_names)} n-gram)", flush=True)


if __name__ == "__main__":
    main()
