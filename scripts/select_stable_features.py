#!/usr/bin/env python3
"""Select domain-STABLE, bot-informative features and write models/stable_features.json.

The public benchmark and the live eval feed are a different distribution
(adversarial AUC=1.0). A feature is kept only if it (a) separates bots on the
benchmark (bot AP >= --ap-min) AND (b) does not shift much benchmark->live
(relative mean shift <= --shift-max), measured against the archived real eval
batches in models/live_chunks/. This forces the model onto transferable signal.

Memory-safe: live chunks are featurised in a streaming, capped fashion.
Run periodically as live_chunks grows (e.g. weekly)."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from train_top_miner import build_dataset, vectorize
from poker44_ml.robust_features import filter_robust_feature_names
from poker44.validator.payload_view import prepare_hand_for_miner
from poker44_ml.rank_features import top_chunk_features
from sklearn.metrics import average_precision_score


def load_live_rows(names, cap=600, recent_files=10):
    files = sorted((REPO / "models" / "live_chunks" / "chunks").glob("*.jsonl.gz"))[-recent_files:]
    rows = []
    for f in reversed(files):
        if len(rows) >= cap:
            break
        with gzip.open(f, "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                for c in (json.loads(line).get("chunks") or []):
                    if not (isinstance(c, list) and c):
                        continue
                    fd = top_chunk_features([prepare_hand_for_miner(h) for h in c if isinstance(h, dict)],
                                            already_prepared=True)
                    rows.append([float(fd.get(n, 0.0)) for n in names])
                    if len(rows) >= cap:
                        break
                if len(rows) >= cap:
                    break
    return np.asarray(rows, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ap-min", type=float, default=0.58)
    ap.add_argument("--shift-max", type=float, default=0.30)
    args = ap.parse_args()

    print("featurising benchmark...", flush=True)
    feats, labels, dates = build_dataset(None)
    names = sorted({n for d in feats for n in d})
    # only consider behaviourally-robust names to begin with
    names = sorted(set(filter_robust_feature_names(names)) | {n for n in names if n.startswith("xseq_")})
    names = [n for n in names if n != "hand_count"]
    Xb = vectorize(feats, names)
    print("featurising live (streaming)...", flush=True)
    Xl = load_live_rows(names)
    print(f"benchmark {Xb.shape}, live {Xl.shape}", flush=True)

    chosen = []
    for i, n in enumerate(names):
        v = Xb[:, i]
        if v.std() < 1e-9:
            continue
        apv = max(average_precision_score(labels, v), average_precision_score(labels, -v))
        shift = abs(Xb[:, i].mean() - Xl[:, i].mean()) / (abs(Xb[:, i].mean()) + abs(Xl[:, i].mean()) + 1e-9)
        if apv >= args.ap_min and shift <= args.shift_max:
            chosen.append((n, float(apv), float(shift)))
    chosen.sort(key=lambda r: -r[1])
    out = {
        "features": [n for n, _, _ in chosen],
        "detail": [{"name": n, "bot_ap": a, "rel_shift": s} for n, a, s in chosen],
        "selection": {"ap_min": args.ap_min, "shift_max": args.shift_max,
                      "source": "benchmark vs live_chunks adversarial stability"},
        "n": len(chosen),
    }
    (REPO / "models" / "stable_features.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {len(chosen)} stable features -> models/stable_features.json", flush=True)


if __name__ == "__main__":
    main()
