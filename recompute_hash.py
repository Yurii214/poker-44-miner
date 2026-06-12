#!/usr/bin/env python3
"""Refresh models/model_manifest.json commit and implementation hash."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "models" / "model_manifest.json"


def main() -> int:
    if not MANIFEST_PATH.is_file():
        print(f"missing manifest: {MANIFEST_PATH}")
        return 1

    sys.path.insert(0, str(ROOT))
    from poker44.utils.model_manifest import build_local_model_manifest, sha256_file

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    files = [ROOT / str(item) for item in manifest.get("implementation_files", [])]
    repo_url = str(manifest.get("repo_url", "https://github.com/Yurii214/poker-44-miner")).strip()
    artifact_rel = str(manifest.get("artifact_url", "models/bot_detector_v1.joblib")).strip()
    artifact_path = ROOT / artifact_rel
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()

    refreshed = build_local_model_manifest(
        repo_root=ROOT,
        implementation_files=files,
        defaults={
            **manifest,
            "repo_url": repo_url,
            "repo_commit": head,
            "artifact_url": artifact_rel,
            "artifact_sha256": sha256_file(artifact_path) if artifact_path.is_file() else "",
            "model_card_url": f"{repo_url.rstrip('/')}/blob/main/README.md",
        },
    )

    MANIFEST_PATH.write_text(json.dumps(refreshed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"updated {MANIFEST_PATH}")
    print(f"repo_commit={head}")
    print(f"implementation_sha256={refreshed.get('implementation_sha256')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
