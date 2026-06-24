"""Build pseudo-labeled training rows from logged live validator batches."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np

from poker44_ml.features import chunk_features
from poker44_ml.inference import Poker44Model
from poker44_ml.live_chunk_store import iter_logged_batches

PSEUDO_SOURCE = "live_pseudo"
DEFAULT_PSEUDO_WEIGHT = 0.35
DEFAULT_MAX_BATCHES = 120


def build_pseudo_labeled_examples(
    *,
    store_dir: str | Path | None = None,
    teacher: Poker44Model | None = None,
    model_path: str | Path | None = None,
    min_bot_score: float = 0.62,
    max_human_score: float = 0.18,
    max_examples: int = 400,
    max_batches: int | None = DEFAULT_MAX_BATCHES,
) -> tuple[list[dict[str, float]], np.ndarray, list[dict[str, Any]]]:
    """High-confidence teacher labels on novel live chunks only.

    Streams logged batches one at a time so large live-chunk archives do not
    need to fit in memory (each batch can be several MB of hand payloads).
    """
    lazy_teacher = teacher
    feature_dicts: list[dict[str, float]] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    batches_scanned = 0

    for batch in iter_logged_batches(store_dir, max_batches=max_batches):
        batches_scanned += 1
        chunks = batch.get("chunks") or []
        if not chunks:
            continue
        final_scores = batch.get("final_scores") or []
        if len(final_scores) != len(chunks):
            if lazy_teacher is None:
                lazy_teacher = teacher or Poker44Model(model_path=model_path)
            final_scores = lazy_teacher.predict_chunk_scores(chunks)
        signatures = batch.get("chunk_signatures") or []
        for index, chunk in enumerate(chunks):
            if len(feature_dicts) >= max_examples:
                if lazy_teacher is not teacher:
                    del lazy_teacher
                gc.collect()
                return (
                    feature_dicts,
                    np.asarray(labels, dtype=int),
                    metadata,
                )
            signature = signatures[index] if index < len(signatures) else f"idx-{index}"
            if signature in seen_signatures:
                continue
            score = float(final_scores[index]) if index < len(final_scores) else 0.5
            if score >= min_bot_score:
                label = 1
            elif score <= max_human_score:
                label = 0
            else:
                continue
            features = chunk_features(chunk)
            features["hand_count"] = float(len(chunk))
            feature_dicts.append(features)
            labels.append(label)
            metadata.append(
                {
                    "source": PSEUDO_SOURCE,
                    "chunk_id": f"live-{signature}",
                    "inner_index": index,
                    "source_date": str(batch.get("ts", ""))[:10],
                    "hand_count": len(chunk),
                    "pseudo_score": round(score, 6),
                    "validator_hotkey": batch.get("validator_hotkey"),
                }
            )
            seen_signatures.add(signature)

    if lazy_teacher is not teacher:
        del lazy_teacher
    gc.collect()
    return feature_dicts, np.asarray(labels, dtype=int), metadata


def balance_pseudo_examples(
    feature_dicts: list[dict[str, float]],
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    max_per_class: int = 50,
    seed: int = 42,
) -> tuple[list[dict[str, float]], np.ndarray, list[dict[str, Any]]]:
    """Downsample the majority pseudo-label class to reduce live-augment skew."""
    if len(labels) == 0:
        return feature_dicts, labels, metadata
    bot_idx = np.flatnonzero(labels == 1)
    human_idx = np.flatnonzero(labels == 0)
    if len(bot_idx) == 0 or len(human_idx) == 0:
        return feature_dicts, labels, metadata
    rng = np.random.default_rng(seed)
    keep_n = min(len(bot_idx), len(human_idx), int(max_per_class))
    picked = np.concatenate([
        rng.choice(bot_idx, size=keep_n, replace=False),
        rng.choice(human_idx, size=keep_n, replace=False),
    ])
    picked.sort()
    return (
        [feature_dicts[int(i)] for i in picked],
        labels[picked],
        [metadata[int(i)] for i in picked],
    )


def merge_training_sets(
    benchmark_features: list[dict[str, float]],
    benchmark_labels: np.ndarray,
    benchmark_metadata: list[dict[str, Any]],
    pseudo_features: list[dict[str, float]],
    pseudo_labels: np.ndarray,
    pseudo_metadata: list[dict[str, Any]],
    *,
    pseudo_weight: float = DEFAULT_PSEUDO_WEIGHT,
) -> tuple[list[dict[str, float]], np.ndarray, list[dict[str, Any]], np.ndarray]:
    """Concatenate benchmark and pseudo-labeled live rows with sample weights."""
    merged_features = list(benchmark_features) + list(pseudo_features)
    merged_labels = np.concatenate(
        [np.asarray(benchmark_labels, dtype=int), np.asarray(pseudo_labels, dtype=int)]
    )
    merged_metadata = list(benchmark_metadata) + list(pseudo_metadata)
    weights = np.ones(len(merged_labels), dtype=float)
    benchmark_count = len(benchmark_labels)
    weights[:benchmark_count] = 1.0
    if len(pseudo_labels):
        weights[benchmark_count:] = float(max(0.05, min(1.0, pseudo_weight)))
    return merged_features, merged_labels, merged_metadata, weights
