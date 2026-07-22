"""Fixed-vocabulary action n-gram features.

Scripted policies replay characteristic action skeletons: the *sequence* of
(street, action, pot-relative size) decisions repeats in ways that marginal
rates cannot express. Our existing ``xseq_`` family measures autocorrelation and
clustering of action magnitudes; it says nothing about which action sequences
actually occur. This module fills that gap.

Two properties matter more than the token design itself:

* **Frozen vocabulary.** The token set is mined once from the benchmark and
  pinned to ``models/ngram_vocab.json``. Emitting a column per token seen at
  train time would make the feature schema drift between training runs and
  inference, so unseen tokens are dropped rather than added.
* **Per-hand normalization.** Raw counts scale with the number of hands, and
  live chunks run 80-100 hands against the benchmark's 30-40 (2.83x measured).
  Counts are divided by the hand count at emission so the columns stay
  comparable across that gap instead of encoding chunk size.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

_VOCAB_PATH = Path(__file__).resolve().parent.parent / "models" / "ngram_vocab.json"

# Single-char action codes keep tokens short enough to read in feature dumps.
_ACTION_CODE = {
    "fold": "F",
    "check": "K",
    "call": "C",
    "bet": "B",
    "raise": "R",
    "all_in": "A",
    "allin": "A",
    "post": "P",
}

# Pot-relative size buckets. Deliberately NOT absolute big-blind magnitudes:
# live pots run about half benchmark size, so any _bb-scaled bucket is 2-11
# sigma out of distribution. A ratio to the pot is scale-free.
_SIZE_EDGES = ((0.4, "s"), (0.9, "m"), (1.5, "p"))
_MIN_CHUNK_SUPPORT = 50


def _size_bucket(amount: float, pot_before: float) -> str:
    """Bucket a wager by its fraction of the pot it was made into."""
    if amount <= 0.0:
        return "-"
    if pot_before <= 0.0:
        return "o"
    ratio = amount / pot_before
    for edge, label in _SIZE_EDGES:
        if ratio < edge:
            return label
    return "o"


def hand_tokens(hand: dict[str, Any]) -> list[str]:
    """Token stream for one hand, in action order."""
    tokens: list[str] = []
    for action in hand.get("actions") or []:
        if not isinstance(action, dict):
            continue
        street = str(action.get("street") or "?")
        code = _ACTION_CODE.get(str(action.get("action_type") or "").lower(), "X")
        try:
            amount = float(action.get("amount") or 0.0)
            pot_before = float(action.get("pot_before") or 0.0)
        except (TypeError, ValueError):
            amount, pot_before = 0.0, 0.0
        tokens.append(f"{street[:1]}{code}{_size_bucket(amount, pot_before)}")
    return tokens


def hand_ngrams(tokens: Sequence[str]) -> Iterable[str]:
    """Unigrams, bigrams and trigrams over one hand's token stream."""
    n = len(tokens)
    for i in range(n):
        yield tokens[i]
        if i + 1 < n:
            yield f"{tokens[i]}>{tokens[i + 1]}"
        if i + 2 < n:
            yield f"{tokens[i]}>{tokens[i + 1]}>{tokens[i + 2]}"


def chunk_ngram_counts(chunk: Sequence[dict[str, Any]]) -> Counter:
    """Raw (un-normalized) n-gram counts across every hand in a chunk."""
    counts: Counter = Counter()
    for hand in chunk:
        if isinstance(hand, dict):
            counts.update(hand_ngrams(hand_tokens(hand)))
    return counts


def load_vocab(path: Path | None = None) -> list[str]:
    """Load the frozen vocabulary; empty list disables the family entirely."""
    p = Path(path) if path is not None else _VOCAB_PATH
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return []
    return list(data.get("tokens") or [])


def build_vocab(
    chunks: Iterable[Sequence[dict[str, Any]]],
    min_chunk_support: int = _MIN_CHUNK_SUPPORT,
) -> list[str]:
    """Mine the vocabulary: tokens present in >= ``min_chunk_support`` chunks.

    Support is counted per *chunk*, not per occurrence, so a token that fires
    hundreds of times inside one unusual chunk does not earn a column.
    """
    support: Counter = Counter()
    for chunk in chunks:
        support.update(set(chunk_ngram_counts(chunk)))
    return sorted(t for t, c in support.items() if c >= min_chunk_support)


def repeat_pair_rate(chunk: Sequence[dict[str, Any]]) -> float:
    """Unbiased probability that two distinct hands share an action skeleton.

    ``sum c(c-1) / n(n-1)`` is length-invariant, unlike a unique-share ratio
    which decays mechanically as the chunk grows — the exact failure mode the
    benchmark-vs-live hand-count gap would otherwise induce.
    """
    sigs = Counter()
    for hand in chunk:
        if isinstance(hand, dict):
            sigs[">".join(hand_tokens(hand))] += 1
    n = sum(sigs.values())
    if n < 2:
        return 0.0
    return sum(c * (c - 1) for c in sigs.values()) / (n * (n - 1))


def ngram_features(
    chunk: Sequence[dict[str, Any]],
    vocab: Sequence[str] | None = None,
) -> dict[str, float]:
    """Per-hand-normalized n-gram features for one chunk.

    Returns a column for every vocabulary token (zero when absent) so the
    schema is identical at train and serve time regardless of chunk content.
    """
    tokens = list(vocab) if vocab is not None else load_vocab()
    if not tokens:
        return {}
    hands = [h for h in chunk if isinstance(h, dict)]
    n_hands = max(len(hands), 1)
    counts = chunk_ngram_counts(hands)
    feats = {f"ng_{t}": counts.get(t, 0.0) / n_hands for t in tokens}
    feats["ng_repeat_pair_rate"] = repeat_pair_rate(hands)
    feats["ng_vocab_coverage"] = (
        sum(1 for t in tokens if t in counts) / len(tokens) if tokens else 0.0
    )
    return feats
