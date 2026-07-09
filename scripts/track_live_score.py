#!/usr/bin/env python3
"""Append UID 166's LIVE competition standing to models/live_score_history.jsonl.

The public benchmark is a different distribution from the live eval (adversarial
AUC=1.0), so benchmark reward does NOT predict rank. The leaderboard composite is
the only valid fitness signal. Run this periodically (e.g. hourly) to build a
history of live composite vs the currently-served model version, so model changes
can be A/B-judged on REAL live performance instead of the mirage benchmark.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # so joblib can unpickle TopEnsemble for served_version
HISTORY = REPO / "models" / "live_score_history.jsonl"
LEADERBOARD = "https://api.poker44.net/api/v1/competition/leaderboard"


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def served_version() -> str:
    try:
        import joblib
        art = joblib.load(REPO / "models" / "bot_detector_top.joblib")
        return str(getattr(art, "metadata", {}).get("model_version", "unknown"))
    except Exception:
        return "unknown"


def main(uid: int = 166) -> None:
    lb = _get(LEADERBOARD)["data"]
    ep = lb.get("epoch", {})
    rows = lb.get("rows", [])
    row = next((r for r in rows if int(r.get("uid", -1)) == uid), None)
    ranks = sorted((int(r["rank"]) for r in rows if r.get("rank") is not None))
    top10_cut = None
    for r in rows:
        if r.get("rank") == 10:
            top10_cut = r.get("compositeScore")
    rounds = {}
    if row:
        for w in (row.get("windowCompositeScores") or []):
            rounds[f"R{w['roundIndex']}"] = w.get("compositeScore")
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "epoch": ep.get("epochId"),
        "served_version": served_version(),
        "uid": uid,
        "rank": (row or {}).get("rank"),
        "composite": (row or {}).get("compositeScore"),
        "rounds": rounds,
        "total_scored": len(rows),
        "top1_composite": max((r.get("compositeScore") or 0) for r in rows) if rows else None,
        "top10_cutoff": top10_cut,
    }
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    uid = int(sys.argv[sys.argv.index("--uid") + 1]) if "--uid" in sys.argv else 166
    main(uid)
