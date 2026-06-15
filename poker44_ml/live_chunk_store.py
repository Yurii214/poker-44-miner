"""Persist validator request batches for live distribution analysis and training."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker44.ml.features import payload_chunk_signature

DEFAULT_STORE_DIR = Path(__file__).resolve().parents[1] / "models" / "live_chunks"
_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chunk_signature(chunk: list[dict[str, Any]]) -> str:
    if not chunk:
        return "empty"
    try:
        sig = payload_chunk_signature(chunk)
        payload = json.dumps(sig, separators=(",", ":"), sort_keys=True)
    except Exception:
        payload = json.dumps(
            {"hand_count": len(chunk)},
            separators=(",", ":"),
            sort_keys=True,
        )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _day_path(store_dir: Path) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return store_dir / "chunks" / f"{day}.jsonl.gz"


def _manifest_path(store_dir: Path) -> Path:
    return store_dir / "manifest.json"


def _load_manifest(store_dir: Path) -> dict[str, Any]:
    path = _manifest_path(store_dir)
    if not path.exists():
        return {
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "total_batches": 0,
            "total_chunks": 0,
            "unique_signatures": 0,
            "seen_signatures": {},
        }
    return json.loads(path.read_text())


def _save_manifest(store_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    _manifest_path(store_dir).write_text(json.dumps(manifest, indent=2) + "\n")


def is_logging_enabled() -> bool:
    value = os.getenv("POKER44_LOG_LIVE_CHUNKS", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def log_validator_batch(
    *,
    chunks: list[list[dict[str, Any]]],
    raw_scores: list[float] | None = None,
    final_scores: list[float] | None = None,
    validator_hotkey: str | None = None,
    uid: int | None = None,
    store_dir: str | Path | None = None,
    max_chunks_per_batch: int | None = None,
) -> bool:
    """Append one validator request batch if logging is enabled."""
    if not is_logging_enabled() or not chunks:
        return False

    store = Path(store_dir or os.getenv("POKER44_LIVE_CHUNK_DIR", DEFAULT_STORE_DIR))
    store.mkdir(parents=True, exist_ok=True)
    (store / "chunks").mkdir(parents=True, exist_ok=True)

    limit = int(
        max_chunks_per_batch
        or os.getenv("POKER44_LIVE_CHUNK_MAX_PER_BATCH", "80")
        or 80
    )
    trimmed = chunks[: max(1, limit)]

    signatures = [_chunk_signature(chunk) for chunk in trimmed]
    record = {
        "ts": _utc_now(),
        "uid": uid,
        "validator_hotkey": validator_hotkey,
        "chunk_count": len(trimmed),
        "hand_counts": [len(chunk) for chunk in trimmed],
        "chunk_signatures": signatures,
        "raw_scores": [round(float(v), 6) for v in (raw_scores or [])[: len(trimmed)]],
        "final_scores": [round(float(v), 6) for v in (final_scores or [])[: len(trimmed)]],
        "chunks": trimmed,
    }

    with _LOCK:
        manifest = _load_manifest(store)
        seen: dict[str, int] = dict(manifest.get("seen_signatures") or {})
        novel = 0
        for signature in signatures:
            if signature not in seen:
                novel += 1
            seen[signature] = int(seen.get(signature, 0)) + 1
        manifest["seen_signatures"] = seen
        manifest["unique_signatures"] = len(seen)
        manifest["total_batches"] = int(manifest.get("total_batches", 0)) + 1
        manifest["total_chunks"] = int(manifest.get("total_chunks", 0)) + len(trimmed)
        manifest["last_batch_at"] = record["ts"]
        manifest["last_novel_signatures"] = novel

        line = json.dumps(record, separators=(",", ":"), ensure_ascii=True)
        with gzip.open(_day_path(store), "at", encoding="utf-8") as handle:
            handle.write(line + "\n")
        _save_manifest(store, manifest)
    return True


def iter_logged_batches(
    store_dir: str | Path | None = None,
    *,
    max_batches: int | None = None,
) -> list[dict[str, Any]]:
    """Load logged validator batches newest files first."""
    store = Path(store_dir or os.getenv("POKER44_LIVE_CHUNK_DIR", DEFAULT_STORE_DIR))
    chunk_dir = store / "chunks"
    if not chunk_dir.exists():
        return []

    files = sorted(chunk_dir.glob("*.jsonl.gz"), reverse=True)
    batches: list[dict[str, Any]] = []
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                batches.append(json.loads(line))
                if max_batches is not None and len(batches) >= max_batches:
                    return batches
    return batches
