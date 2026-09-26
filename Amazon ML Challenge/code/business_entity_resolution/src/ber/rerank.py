"""Optional multilingual cross-encoder reranker for borderline pairs (v5).

A bi-encoder (embed.py) compresses each record to one vector, which is fast but lossy.
A CROSS-encoder reads the two texts jointly and is more accurate at deciding "same or
not", but far more expensive -- so we run it ONLY on the pairs the LightGBM matcher is
unsure about (score within a band of the threshold) and blend its probability with the
matcher's. This spends the expensive model exactly where it can flip a decision.

COMPLIANCE: an Apache-2.0 cross-encoder run locally, well under the 8B-parameter cap --
the same basis as embed.py. Experimental: default OFF and not validated at full scale.

The encoder is pluggable, like embed.py: 'sentence-transformers' for real runs, 'stub'
(a deterministic character-overlap score) so the reranking + merge plumbing can be
tested fully offline.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def compose_texts(core_name, norm_addr, country_key) -> List[str]:
    """Compose the record text the cross-encoder reads, IDENTICALLY at train and
    inference time. Includes the address, because the same-name/different-branch
    disambiguation (the 0.41-precision trap) is only decidable from the address."""
    n = len(core_name)
    out = [""] * n
    for i in range(n):
        nm = core_name[i] or ""; ad = norm_addr[i] or ""; co = country_key[i] or ""
        out[i] = f"{nm} | {ad} | {co}".strip()
    return out


class StubReranker:
    """Deterministic offline stand-in: a cheap character-bigram Jaccard on the pair.
    Not semantic -- only for exercising the reranking plumbing without a model."""
    def _bigrams(self, s: str):
        s = s or ""
        return {s[i:i + 2] for i in range(len(s) - 1)} or {s}

    def score(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        out = np.zeros(len(pairs), dtype=np.float32)
        for i, (a, b) in enumerate(pairs):
            ba, bb = self._bigrams(a), self._bigrams(b)
            u = len(ba | bb)
            out[i] = (len(ba & bb) / u) if u else 0.0
        return out


class CrossEncoderReranker:
    """Wraps a sentence-transformers CrossEncoder; returns a 0..1 match probability."""
    def __init__(self, model_name: str, device: str = None, enc_batch: int = 128):
        from sentence_transformers import CrossEncoder  # imported lazily
        try:
            import torch
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            device = device or "cpu"
        self.model = CrossEncoder(model_name, device=device)
        self.enc_batch = enc_batch

    def score(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        s = np.asarray(self.model.predict(pairs, batch_size=self.enc_batch,
                                          show_progress_bar=False), dtype=np.float32).ravel()
        # squash logits to 0..1 if the model returns raw scores
        if s.size and (s.min() < 0.0 or s.max() > 1.0):
            s = 1.0 / (1.0 + np.exp(-s))
        return s.astype(np.float32)


def load_reranker(cfg):
    backend = getattr(cfg, "cross_encoder_backend", "sentence-transformers")
    if backend == "stub":
        return StubReranker()
    return CrossEncoderReranker(cfg.cross_encoder_model, device=cfg.embed_device,
                                enc_batch=cfg.cross_encoder_enc_batch)


def blend(lgbm: np.ndarray, ce: np.ndarray, weight: float) -> np.ndarray:
    """Convex blend of the LightGBM probability and the cross-encoder probability."""
    return ((1.0 - weight) * np.asarray(lgbm, np.float32)
            + weight * np.asarray(ce, np.float32)).astype(np.float32)
