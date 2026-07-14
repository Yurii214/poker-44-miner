"""Format-invariant bot-regularity index + live pseudo-labeling.

The public benchmark is a different distribution from the live eval (AUC=1.0), so
benchmark-trained models suffer concept shift. But bots are behaviourally
REGULAR in ANY format: few distinct hand-scripts, low action entropy, tight
bet-size clustering, long repeated runs. This index combines those signals
(bot-direction known a priori, INDEPENDENT of any trained model), and lets us
pseudo-label the confident tails of the REAL live feed to train on the live
distribution directly.

Validated: on the labelled benchmark this index alone gives bot AP ~0.71.
"""

from __future__ import annotations

import numpy as np

# feature name -> sign (bot direction: +1 = higher is more bot, -1 = lower is more bot)
REGULARITY_FEATURES = {
    "chunk_action_signature_unique_share": -1,
    "chunk_street_signature_unique_share": -1,
    "chunk_amount_bucket_signature_unique_share": -1,
    "schema_street_entropy_std": -1,
    "action_entropy_mean": -1,
    "schema_action_entropy_q50": -1,
    "actor_entropy_q90": -1,
    "sig_decision_ent_mean": -1,
    "action_run_max_share_mean": +1,
    "action_run_max_share_q10": +1,
    "xseq_bet_pot_modal_frac": +1,
    "xseq_bet_pot_cluster_frac": +1,
}


def regularity_index(X: np.ndarray, names: list[str]) -> np.ndarray:
    """Per-row bot-regularity score (z-scored within X, oriented bot-high)."""
    idx = {n: i for i, n in enumerate(names)}
    use = [(idx[n], s) for n, s in REGULARITY_FEATURES.items() if n in idx]
    if not use:
        return np.zeros(len(X))
    z = np.zeros(len(X), dtype=float)
    for i, s in use:
        v = X[:, i].astype(float)
        sd = v.std() + 1e-9
        z += s * (v - v.mean()) / sd
    return z / len(use)


def pseudo_label_live(Xl: np.ndarray, names: list[str], *, tail: float = 0.25):
    """Pseudo-label live rows by regularity tails.

    Returns (labels, mask): the top ``tail`` fraction by regularity -> bot(1),
    the bottom ``tail`` -> human(0); the ambiguous middle is masked out.
    """
    r = regularity_index(Xl, names)
    n = len(r)
    order = np.argsort(r)
    k = max(1, int(n * tail))
    labels = np.full(n, -1, dtype=int)
    labels[order[:k]] = 0          # lowest regularity -> human
    labels[order[-k:]] = 1         # highest regularity -> bot
    mask = labels >= 0
    return labels, mask
