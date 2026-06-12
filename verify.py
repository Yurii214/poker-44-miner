#!/usr/bin/env python3
"""Verify repo commit and implementation hash for manifest review."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "models" / "model_manifest.json"


def _sha256_for_files(repo_root: Path, relative_files: list[str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(relative_files):
        path = repo_root / rel
        digest.update(rel.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST_PATH.is_file():
        print(f"missing manifest: {MANIFEST_PATH}")
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    files = [str(item) for item in manifest.get("implementation_files", []) if str(item).strip()]
    if not files:
        print("implementation_files is empty")
        return 1

    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    recomputed = _sha256_for_files(ROOT, files)
    manifest_commit = str(manifest.get("repo_commit", "")).strip()
    manifest_hash = str(manifest.get("implementation_sha256", "")).strip()
    artifact_path = ROOT / str(manifest.get("artifact_url", "models/bot_detector_v1.joblib"))
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest() if artifact_path.is_file() else ""

    checks = {
        "repo_commit_matches_head": manifest_commit == head,
        "implementation_sha256_matches": manifest_hash == recomputed,
        "artifact_present": artifact_path.is_file(),
        "artifact_sha256_matches": str(manifest.get("artifact_sha256", "")).strip() == artifact_sha256,
        "data_attestation_present": bool(str(manifest.get("data_attestation", "")).strip()),
    }

    print(f"HEAD: {head}")
    print(f"manifest repo_commit: {manifest_commit}")
    print(f"manifest implementation_sha256: {manifest_hash}")
    print(f"recomputed implementation_sha256: {recomputed}")
    print(f"artifact: {artifact_path.name} present={checks['artifact_present']}")
    if artifact_sha256:
        print(f"artifact_sha256: {artifact_sha256}")

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1

    print("OK: manifest matches repository")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
