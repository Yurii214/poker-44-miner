"""Live-health gate for model deploys.

The public benchmark is a different distribution from the live eval (adversarial
AUC=1.0), so benchmark holdout does NOT predict live performance. A model can
look great on benchmark yet be DEGENERATE on the real live feed:
  * collapsed-low  -> scores everything ~0 (thinks all humans); no signal
  * compressed-high -> scores everything ~1 (thinks all bots); no separation

This gate scores a candidate on a recent sample of the archived real validator
batches (models/live_chunks/) and rejects degenerate distributions. It is the
guard that must run before any auto-deploy — the benchmark guard once deployed a
compressed-high model (live mean 0.72, std 0.10) over a rank-3 model.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Healthy live score distribution: centered, well-spread (a real classifier).
HEALTHY_MEAN = (0.30, 0.70)
HEALTHY_STD_MIN = 0.10


def sample_live_chunks(cap: int = 250, recent_files: int = 5) -> list[list[dict]]:
    from poker44.validator.payload_view import prepare_hand_for_miner

    files = sorted((REPO / "models" / "live_chunks" / "chunks").glob("*.jsonl.gz"))[-recent_files:]
    out: list[list[dict]] = []
    for f in reversed(files):
        if len(out) >= cap:
            break
        with gzip.open(f, "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                for c in (json.loads(line).get("chunks") or []):
                    if isinstance(c, list) and c:
                        out.append([prepare_hand_for_miner(h) for h in c if isinstance(h, dict)])
                        if len(out) >= cap:
                            break
                if len(out) >= cap:
                    break
    return out


def live_health(model_or_path: Any, chunks: list[list[dict]] | None = None) -> dict:
    """Return {'healthy': bool, 'mean':.., 'std':.., 'reason':..} for a model.

    ``model_or_path`` may be a TopMinerModel, a path, or an object exposing
    ``predict_raw_scores``.
    """
    from poker44_ml.top_inference import TopMinerModel

    model = model_or_path
    if isinstance(model_or_path, (str, Path)):
        model = TopMinerModel(model_or_path)
    if chunks is None:
        chunks = sample_live_chunks()
    if not chunks:
        return {"healthy": False, "mean": None, "std": None, "reason": "no_live_chunks"}

    raw = np.asarray(model.predict_raw_scores(chunks), dtype=float)
    mean, std = float(raw.mean()), float(raw.std())
    reasons = []
    if not (HEALTHY_MEAN[0] <= mean <= HEALTHY_MEAN[1]):
        reasons.append(f"mean {mean:.3f} outside {HEALTHY_MEAN} (collapsed-low or compressed-high)")
    if std < HEALTHY_STD_MIN:
        reasons.append(f"std {std:.3f} < {HEALTHY_STD_MIN} (no separation)")
    healthy = not reasons
    return {"healthy": healthy, "mean": mean, "std": std, "n": len(chunks),
            "reason": "ok" if healthy else "; ".join(reasons)}
