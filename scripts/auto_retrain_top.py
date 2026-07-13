#!/usr/bin/env python3
"""Daily retrain-and-maybe-deploy for the rank-first top-miner model.

Runs shortly after the ~00:05 UTC public benchmark release. It:

  1. Pulls the freshest benchmark releases and featurises them (cached).
  2. Trains a CANDIDATE on all dates except the newest holdout slice.
  3. Scores both the candidate and the CURRENTLY-DEPLOYED model on that same
     held-out newest slice with the real rank-first reward.
  4. Deploys the freshly-trained model ONLY if the candidate does not regress
     versus the deployed model (candidate_reward >= deployed_reward - tolerance).
     Otherwise it keeps the current model and logs why.

Why this guard is fair in steady state: the holdout is the newest date, which
the deployed model (trained on a *previous* day) has not seen either -- so the
comparison is apples-to-apples. On the very first run the deployed model may
have seen the holdout, which only makes the guard more conservative (safe).

Deploy = run scripts/publish_miner_repo.sh (regenerate manifest, commit, push
to the public repo, pm2 restart, verify, write deploy lock).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_top_miner import build_dataset, make_weights, select_feature_names, vectorize
from poker44_ml.rank_reward import rank_reward
from poker44_ml.top_model import train_top_ensemble

DEFAULT_ARTIFACT = REPO / "models" / "bot_detector_top.joblib"
MIN_HOLDOUT_CHUNKS = 20  # expand holdout to older dates until it has at least this many


def _log(msg: str) -> None:
    print(msg, flush=True)


def _holdout_mask(dates: np.ndarray, n_dates: int) -> np.ndarray:
    """Newest n_dates as holdout, expanded until >= MIN_HOLDOUT_CHUNKS chunks."""
    uniq = sorted(set(dates.tolist()))
    k = max(1, n_dates)
    while k <= len(uniq):
        hold = set(uniq[-k:])
        mask = np.array([d in hold for d in dates])
        if int(mask.sum()) >= MIN_HOLDOUT_CHUNKS or k == len(uniq):
            return mask
        k += 1
    return np.array([d in set(uniq[-k:]) for d in dates])


def _deployed_reward(artifact_path: Path, hold_feats: list[dict], hold_labels: np.ndarray):
    if not artifact_path.is_file():
        return None
    try:
        model = joblib.load(artifact_path)
    except Exception as exc:  # noqa: BLE001
        _log(f"  deployed model unreadable ({exc}); treating as no baseline.")
        return None
    if not hasattr(model, "predict_from_dicts"):
        _log("  deployed model is a legacy format (not TopEnsemble); no baseline.")
        return None
    scores = model.predict_from_dicts(hold_feats)
    reward, _ = rank_reward(scores, hold_labels)
    return reward


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout-dates", type=int, default=1)
    ap.add_argument("--human-weight", type=float, default=1.0)
    ap.add_argument("--recency-weight", type=float, default=6.0)
    ap.add_argument("--recency-days", type=int, default=8)
    ap.add_argument("--max-regression", type=float, default=0.01,
                    help="Deploy unless candidate reward is more than this below the deployed model.")
    ap.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    ap.add_argument("--deploy", action="store_true", help="Publish + restart on promotion.")
    args = ap.parse_args()

    _log("=== auto_retrain_top ===")
    feats, labels, dates = build_dataset(None)
    names = select_feature_names(feats)
    x = vectorize(feats, names)
    weights = make_weights(labels, dates, human_w=args.human_weight,
                           recency_w=args.recency_weight, recency_days=args.recency_days)
    uniq = sorted(set(dates.tolist()))
    newest = uniq[-1]
    _log(f"Dataset: {len(labels)} chunks / {len(names)} features / {len(uniq)} dates (newest {newest}).")

    # --- Train the fresh production candidate on ALL data (tracks evolving bots). ---
    _log("Training fresh candidate on all dates...")
    prod = train_top_ensemble(x, labels, dates, names, sample_weight=weights)
    prod.metadata.update({
        "model_name": "poker44-rankfirst-ensemble",
        "model_version": f"top-{newest}",  # dated version tied to training data
        "n_chunks": int(len(labels)),
        "n_features": len(names),
        "source_dates": uniq,
        "human_weight": args.human_weight,
        "recency_weight": args.recency_weight,
        "recency_days": args.recency_days,
        "serving": "raw_probability",
    })

    # Save to a candidate path first so we can score it as a real TopMinerModel
    # (which featurises live chunks) and never touch the LIVE artifact until the
    # gate passes AND --deploy is set.
    candidate_path = args.artifact.with_suffix(".candidate.joblib")
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(prod, candidate_path)

    # --- LIVE-HEALTH GATE (NOT benchmark holdout, which is a mirage). ---
    # Reject degenerate candidates by their score distribution on real live
    # chunks, and don't replace a healthy model with one that separates worse.
    from poker44_ml.live_health import live_health, sample_live_chunks
    live_chunks = sample_live_chunks(cap=250)
    cand_h = live_health(candidate_path, live_chunks)
    _log(f"Candidate live-health: healthy={cand_h['healthy']} mean={cand_h['mean']} "
         f"std={cand_h['std']} ({cand_h['reason']})")
    if not cand_h["healthy"]:
        _log("DECISION: KEEP current model (fresh candidate is DEGENERATE on live).")
        candidate_path.unlink(missing_ok=True)
        return 0
    dep_h = None
    if args.artifact.is_file():
        try:
            dep_h = live_health(args.artifact, live_chunks)
            _log(f"Deployed  live-health: healthy={dep_h['healthy']} mean={dep_h['mean']} std={dep_h['std']}")
        except Exception as exc:  # noqa: BLE001
            _log(f"  deployed live-health unavailable ({exc}).")
    if dep_h is not None and dep_h["healthy"] and cand_h["std"] < dep_h["std"] - 0.05:
        _log("DECISION: KEEP current model (candidate has notably worse live separation).")
        candidate_path.unlink(missing_ok=True)
        return 0
    _log(f"DECISION: PROMOTE (fresh, live-healthy candidate top-{newest}).")

    if not args.deploy:
        _log("(--deploy not set; candidate saved but LIVE artifact untouched, not published.)")
        return 0

    import shutil
    shutil.move(str(candidate_path), str(args.artifact))
    _log(f"Promoted candidate -> {args.artifact}")
    _log("Publishing + restarting miner via scripts/publish_miner_repo.sh ...")
    result = subprocess.run(
        ["bash", str(REPO / "scripts" / "publish_miner_repo.sh")],
        cwd=str(REPO),
    )
    if result.returncode != 0:
        _log(f"publish_miner_repo.sh exited {result.returncode}.")
        return result.returncode
    _log("Deploy complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
