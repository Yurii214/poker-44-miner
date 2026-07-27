"""Hierarchical Set Transformer over raw action sequences.

Our tree ensemble consumes ~74 per-chunk aggregate statistics that discard
action *order*. Scripted bots are most distinguishable in their temporal
structure — sizing rhythms, postflop response sequences, preflop priors — which
aggregates wash out. This module learns a representation directly from the
(action_type, street, actor_role, size-bucket, pot-flow) token stream of each
hand and contributes one extra column to the stacked ensemble.

Structure (hierarchical, permutation-invariant across hands):
  action tokens --[Transformer x2]--> attention-pool --> hand embedding
  hand embeddings --[Transformer x1]--> attention-pool --> chunk embedding --> MLP --> logit

Everything is derived from validator-visible fields only. Amounts are bucketed
by big-blind magnitude *and* fed as log1p continuous channels; pot flow is a
relative (pot_after - pot_before) bucket so it survives the benchmark->live pot
compression. Chunks longer than the cap are span-sampled (not head-truncated)
so the ~80-100-hand live chunks are represented across their whole span.

Independent implementation; technique studied from open-source competitors but
no code copied. sklearn-style wrapper so it drops into the OOF stack; pickles as
(state_dict + config); CPU-only by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch required for the sequence model: "
        "pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc

# --- tokenizer -------------------------------------------------------------

_ACTION_VOCAB = {"<pad>": 0, "check": 1, "call": 2, "bet": 3, "raise": 4, "fold": 5, "all_in": 6}
_STREET_VOCAB = {"<pad>": 0, "preflop": 1, "flop": 2, "turn": 3, "river": 4, "": 5}
_STREET_IDX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
# BB-magnitude buckets for the wager size (0 = pad, 1 = no chips in).
_SIZE_EDGES = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0)
_SIZE_VOCAB_SIZE = len(_SIZE_EDGES) + 2
_POTFLOW_VOCAB_SIZE = 5  # pad, flat, small, medium, large
MAX_ACTIONS = 12
MAX_HANDS = 48
CONT_DIM = 3  # log1p(amount_bb), log1p(pot_after_bb), log1p(pot_delta_bb)


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return d if v is None else float(v)
    except (TypeError, ValueError):
        return d


def _action_id(v: Any) -> int:
    r = str(v or "").strip().lower()
    if r in _ACTION_VOCAB:
        return _ACTION_VOCAB[r]
    for key in ("raise", "bet", "call", "check", "fold"):
        if key in r:
            return _ACTION_VOCAB[key]
    return _ACTION_VOCAB["check"]


def _size_bucket(amount_bb: float) -> int:
    if amount_bb <= 0.0:
        return 1
    for i, edge in enumerate(_SIZE_EDGES):
        if amount_bb < edge:
            return i + 1
    return _SIZE_VOCAB_SIZE - 1


def _potflow_id(delta_bb: float) -> int:
    if delta_bb <= 1e-6:
        return 1
    if delta_bb <= 1.0:
        return 2
    if delta_bb <= 4.0:
        return 3
    return 4


def _span_sample(total: int, limit: int) -> List[int]:
    if total <= limit:
        return list(range(total))
    if limit <= 1:
        return [total // 2]
    last = total - 1
    idx = sorted({int(round(i * last / (limit - 1))) for i in range(limit)})
    i = 0
    while len(idx) < limit:  # fill if rounding collapsed positions
        if i not in idx:
            idx.append(i)
        i += 1
    return sorted(idx)[:limit]


def encode_chunk(chunk: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Pad one chunk into (hand x action) token tensors + masks."""
    at = np.zeros((MAX_HANDS, MAX_ACTIONS), np.int64)
    st = np.zeros((MAX_HANDS, MAX_ACTIONS), np.int64)
    role = np.zeros((MAX_HANDS, MAX_ACTIONS), np.int64)
    size = np.zeros((MAX_HANDS, MAX_ACTIONS), np.int64)
    flow = np.zeros((MAX_HANDS, MAX_ACTIONS), np.int64)
    cont = np.zeros((MAX_HANDS, MAX_ACTIONS, CONT_DIM), np.float32)
    amask = np.zeros((MAX_HANDS, MAX_ACTIONS), np.bool_)
    hmask = np.zeros(MAX_HANDS, np.bool_)

    hands = [h for h in chunk if isinstance(h, dict)]
    for hi, si in enumerate(_span_sample(len(hands), MAX_HANDS)):
        hand = hands[si]
        md = hand.get("metadata") or {}
        try:
            hero = int(md.get("hero_seat") or 0)
        except (TypeError, ValueError):
            hero = 0
        bb = _f(md.get("bb"), 0.02) or 0.02
        actions = hand.get("actions") or []
        n = min(len(actions), MAX_ACTIONS)
        for ai in range(n):
            a = actions[ai]
            if not isinstance(a, dict):
                continue
            at[hi, ai] = _action_id(a.get("action_type"))
            st[hi, ai] = _STREET_VOCAB.get(str(a.get("street") or "").strip().lower(), 5)
            try:
                seat = int(a.get("actor_seat") or 0)
            except (TypeError, ValueError):
                seat = 0
            role[hi, ai] = 1 if (hero and seat == hero) else 2
            amt_bb = _f(a.get("normalized_amount_bb")) or (_f(a.get("amount")) / bb)
            pot_after_bb = _f(a.get("pot_after")) / bb
            pot_before_bb = _f(a.get("pot_before")) / bb
            delta_bb = max(pot_after_bb - pot_before_bb, 0.0)
            size[hi, ai] = _size_bucket(amt_bb)
            flow[hi, ai] = _potflow_id(delta_bb)
            cont[hi, ai, 0] = math.log1p(max(amt_bb, 0.0))
            cont[hi, ai, 1] = math.log1p(max(pot_after_bb, 0.0))
            cont[hi, ai, 2] = math.log1p(delta_bb)
            amask[hi, ai] = True
        hmask[hi] = bool(amask[hi].any())
    return {"at": at, "st": st, "role": role, "size": size, "flow": flow,
            "cont": cont, "amask": amask, "hmask": hmask}


class _DS(Dataset):
    def __init__(self, chunks, y=None, w=None):
        self.enc = [encode_chunk(c) for c in chunks]
        self.y = np.asarray(y, np.float32) if y is not None else None
        self.w = np.asarray(w, np.float32) if w is not None else None

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        return (self.enc[i],
                float(self.y[i]) if self.y is not None else 0.0,
                float(self.w[i]) if self.w is not None else 1.0)


def _collate(batch):
    out = {}
    for k in ("at", "st", "role", "size", "flow", "cont", "amask", "hmask"):
        s = np.stack([b[0][k] for b in batch], 0)
        if k == "cont":
            out[k] = torch.from_numpy(s).float()
        elif k in ("amask", "hmask"):
            out[k] = torch.from_numpy(s).bool()
        else:
            out[k] = torch.from_numpy(s).long()
    out["label"] = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    out["weight"] = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return out


# --- model -----------------------------------------------------------------

@dataclass
class SeqConfig:
    d_model: int = 48
    n_heads: int = 4
    n_action_layers: int = 2
    n_hand_layers: int = 1
    dropout: float = 0.1
    ff_mult: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in
                ("d_model", "n_heads", "n_action_layers", "n_hand_layers", "dropout", "ff_mult")}


class _AttnPool(nn.Module):
    """Single learned-query attention pool (Set Transformer PMA, k=1)."""

    def __init__(self, d, h, p):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, kpm=None):
        b = x.size(0)
        safe = kpm
        allpad = kpm.all(dim=1) if kpm is not None else None
        if safe is not None and allpad is not None and allpad.any():
            safe = safe.clone()
            safe[allpad, 0] = False  # avoid NaN on fully-padded rows
        o, _ = self.attn(self.q.expand(b, 1, -1), x, x, key_padding_mask=safe, need_weights=False)
        o = self.norm(o.squeeze(1))
        if allpad is not None and allpad.any():
            o = o.masked_fill(allpad.unsqueeze(-1), 0.0)
        return o


class ChunkSetTransformer(nn.Module):
    def __init__(self, cfg: SeqConfig):
        super().__init__()
        d = cfg.d_model
        self.at_emb = nn.Embedding(len(_ACTION_VOCAB), d, padding_idx=0)
        self.st_emb = nn.Embedding(len(_STREET_VOCAB), d, padding_idx=0)
        self.role_emb = nn.Embedding(3, d, padding_idx=0)
        self.size_emb = nn.Embedding(_SIZE_VOCAB_SIZE, d, padding_idx=0)
        self.flow_emb = nn.Embedding(_POTFLOW_VOCAB_SIZE, d, padding_idx=0)
        self.pos_emb = nn.Embedding(MAX_ACTIONS, d)
        self.cont_proj = nn.Linear(CONT_DIM, d)
        self.in_norm = nn.LayerNorm(d)
        self.in_drop = nn.Dropout(cfg.dropout)

        def enc(n):
            layer = nn.TransformerEncoderLayer(
                d, cfg.n_heads, dim_feedforward=d * cfg.ff_mult,
                dropout=cfg.dropout, batch_first=True, activation="gelu")
            return nn.TransformerEncoder(layer, num_layers=n)

        self.action_enc = enc(cfg.n_action_layers)
        self.action_pool = _AttnPool(d, cfg.n_heads, cfg.dropout)
        self.hand_enc = enc(cfg.n_hand_layers)
        self.chunk_pool = _AttnPool(d, cfg.n_heads, cfg.dropout)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                  nn.Dropout(cfg.dropout), nn.Linear(d, 1))

    def forward(self, at, st, role, size, flow, cont, amask, hmask):
        b, h, a = at.shape
        pos = torch.arange(a, device=at.device).unsqueeze(0).expand(b * h, a)
        r = lambda t: t.reshape(b * h, a)
        emb = (self.at_emb(r(at)) + self.st_emb(r(st)) + self.role_emb(r(role))
               + self.size_emb(r(size)) + self.flow_emb(r(flow)) + self.pos_emb(pos)
               + self.cont_proj(cont.reshape(b * h, a, -1)))
        emb = self.in_drop(self.in_norm(emb))
        akpm = ~r(amask)
        hand_emb = self.action_pool(self.action_enc(emb, src_key_padding_mask=akpm), kpm=akpm)
        hand_emb = hand_emb.reshape(b, h, -1).masked_fill(~hmask.unsqueeze(-1), 0.0)
        hkpm = ~hmask
        chunk_emb = self.chunk_pool(self.hand_enc(hand_emb, src_key_padding_mask=hkpm), kpm=hkpm)
        return self.head(chunk_emb).squeeze(-1)


# --- sklearn-style wrapper for the OOF stack -------------------------------

@dataclass
class SequenceModelWrapper:
    config: SeqConfig = field(default_factory=SeqConfig)
    n_epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.12
    early_stopping_patience: int = 3
    seed: int = 42
    device: str = "cpu"
    verbose: bool = False
    _state: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def __post_init__(self):
        if not isinstance(self.config, SeqConfig):
            self.config = SeqConfig(**dict(self.config))

    def fit(self, chunks, y, sample_weight=None):
        y = np.asarray(y, np.float32)
        w = (np.asarray(sample_weight, np.float32) if sample_weight is not None
             else np.ones(len(chunks), np.float32))
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(chunks))
        vn = max(int(round(self.val_fraction * len(chunks))), 1)
        vi, ti = order[:vn], order[vn:]
        tl = DataLoader(_DS([chunks[i] for i in ti], y[ti], w[ti]),
                        batch_size=self.batch_size, shuffle=True, collate_fn=_collate)
        vl = DataLoader(_DS([chunks[i] for i in vi], y[vi], w[vi]),
                        batch_size=self.batch_size, shuffle=False, collate_fn=_collate)

        torch.manual_seed(self.seed)
        model = ChunkSetTransformer(self.config).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.learning_rate,
                                weight_decay=self.weight_decay)
        lossf = nn.BCEWithLogitsLoss(reduction="none")
        # SELECT ON VAL LOSS, not AP: benchmark classes are trivially separable,
        # so AP pegs ~1.0 from epoch 0 and would freeze a near-untrained net that
        # collapses on live. Val loss keeps improving as the net actually learns.
        best, best_state, patience = float("inf"), None, 0
        for ep in range(self.n_epochs):
            model.train()
            for batch in tl:
                inp = {k: batch[k].to(self.device) for k in
                       ("at", "st", "role", "size", "flow", "cont", "amask", "hmask")}
                logit = model(**inp)
                raw = lossf(logit, batch["label"].to(self.device))
                wt = batch["weight"].to(self.device)
                loss = (raw * wt).sum() / wt.sum().clamp(min=1e-6)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            vloss = self._val_loss(model, vl, lossf)
            if self.verbose:
                print(f"    seq epoch {ep+1}/{self.n_epochs} val_loss={vloss:.4f}", flush=True)
            if vloss + 1e-5 < best:
                best, patience = vloss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self._state = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                       "config": self.config.to_dict()}
        return self

    def _val_loss(self, model, loader, lossf):
        if len(loader.dataset) == 0:
            return float("inf")
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                inp = {k: batch[k].to(self.device) for k in
                       ("at", "st", "role", "size", "flow", "cont", "amask", "hmask")}
                raw = lossf(model(**inp), batch["label"].to(self.device))
                wt = batch["weight"].to(self.device)
                tot += float((raw * wt).sum().item())
                cnt += int(wt.sum().item())
        return tot / max(cnt, 1)

    def _predict(self, model, chunks):
        if not chunks:
            return np.zeros(0)
        loader = DataLoader(_DS(list(chunks)), batch_size=self.batch_size,
                            shuffle=False, collate_fn=_collate)
        model.eval()
        out: List[float] = []
        with torch.no_grad():
            for batch in loader:
                inp = {k: batch[k].to(self.device) for k in
                       ("at", "st", "role", "size", "flow", "cont", "amask", "hmask")}
                out.extend(model(**inp).detach().cpu().tolist())
        a = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(out, np.float64), -40, 40)))
        return np.clip(a, 1e-6, 1 - 1e-6)

    def predict_proba(self, chunks):
        if self._state is None:
            raise RuntimeError("predict_proba before fit")
        model = ChunkSetTransformer(SeqConfig(**self._state["config"])).to(self.device)
        model.load_state_dict(self._state["state_dict"])
        p = self._predict(model, chunks)
        return np.column_stack([1 - p, p])

    def predict_chunk_scores(self, chunks) -> List[float]:
        return self.predict_proba(chunks)[:, 1].tolist()

    def __getstate__(self):
        d = {k: getattr(self, k) for k in
             ("n_epochs", "batch_size", "learning_rate", "weight_decay",
              "val_fraction", "early_stopping_patience", "seed", "device", "verbose")}
        d["config"] = self.config.to_dict()
        d["_state"] = self._state
        return d

    def __setstate__(self, s):
        self.config = SeqConfig(**s.pop("config"))
        self._state = s.pop("_state")
        for k, v in s.items():
            setattr(self, k, v)
        # A CUDA-trained artifact must still load on this CPU-only miner.
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
