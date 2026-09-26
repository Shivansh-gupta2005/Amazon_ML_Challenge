"""Local multilingual embeddings: a semantic blocker + feature (v5).

WHY: character n-grams match on shared *letters*. They cannot tell that two French
names are semantically the same when they share few character chunks (synonyms,
different word forms, translation-like variation). A multilingual sentence embedding
captures meaning, which is exactly the France gap the earlier approaches leave open.

COMPLIANCE (important): this uses a pretrained, permissively-licensed sentence
transformer -- `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, ~118M parameters,
far under the 8B cap) -- run ENTIRELY LOCALLY. The model weights are downloaded once
and then used offline; the model is never queried as an external service and holds no
knowledge of the specific businesses in the dataset. That is a pretrained backbone,
not external-data augmentation, so it is within the rules (no external DB/API/internet
lookup about the entities themselves).

DESIGN:
  * Encoding is the expensive step and benefits from a GPU; it is done once and cached
    to a float32 memmap on disk (12.5M x 384 floats ~ 19 GB -- too big for RAM, so a
    memmap is required).
  * FAISS ANN search runs on CPU, per country+source subset -- so each index is built
    over a small gallery, mirroring how the TF-IDF blockers already work.
  * The encoder is pluggable: the real runs use `sentence-transformers`; a deterministic
    `HashingEncoder` stands in for offline testing where the model can't be downloaded.
"""
from __future__ import annotations

import os
import zlib
from typing import List, Tuple

import numpy as np

try:
    import faiss
    _HAVE_FAISS = True
except Exception:                       # pragma: no cover
    _HAVE_FAISS = False


# ----------------------------------------------------------------- encoders
class HashingEncoder:
    """Deterministic char-n-gram hashing encoder (no model download).

    Not semantic -- it exists so the FAISS blocking + feature plumbing can be tested
    fully offline. Similar strings still map to similar vectors (shared n-grams hash to
    shared dimensions), so it also exercises the blocker's ability to retrieve
    near-duplicates. The real runs use STEncoder instead.
    """
    def __init__(self, dim: int = 256, ngram: Tuple[int, int] = (3, 4)):
        self.dim = dim
        self.ngram = ngram

    def encode(self, texts: List[str], batch_size: int = 256, **_) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        lo, hi = self.ngram
        for i, t in enumerate(texts):
            s = "^" + (t or "") + "$"
            for n in range(lo, hi + 1):
                for j in range(len(s) - n + 1):
                    h = zlib.crc32(s[j:j + n].encode("utf-8"))
                    out[i, h % self.dim] += 1.0 if (h >> 20) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)


class STEncoder:
    """Wraps a sentence-transformers model; returns L2-normalised float32 vectors."""
    def __init__(self, model_name: str, device: str = None):
        from sentence_transformers import SentenceTransformer  # imported lazily
        try:
            import torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            self.device = device or "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: List[str], batch_size: int = 256, **_) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def load_encoder(cfg):
    backend = getattr(cfg, "embed_backend", "sentence-transformers")
    if backend == "hashing":
        return HashingEncoder(dim=getattr(cfg, "embed_hash_dim", 256))
    return STEncoder(cfg.embed_model, device=cfg.embed_device)


# ------------------------------------------------------------- encode to disk
def encode_to_memmap(encoder, texts: List[str], path: str,
                     batch: int = 100_000, enc_batch: int = 256,
                     log=lambda m: None) -> np.ndarray:
    """Encode all texts once, streaming into a float32 memmap addressed by row id."""
    n, d = len(texts), encoder.dim
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(n, max(d, 1)))
    for s in range(0, n, batch):
        e = min(s + batch, n)
        mm[s:e] = encoder.encode(texts[s:e], batch_size=enc_batch)
        if (s // batch) % 5 == 0:
            log(f"  embedded {e}/{n}")
    mm.flush()
    return mm


# --------------------------------------------------------------- ANN blocker
def faiss_block(query: np.ndarray, gallery: np.ndarray, k: int, min_cosine: float,
                hnsw_max_exact: int = 200_000, M: int = 32,
                ef_construction: int = 200, ef_search: int = 64):
    """Top-k gallery rows per query row by embedding cosine (inner product on
    L2-normalised vectors). Exact IndexFlatIP for small galleries, HNSW for large.
    Returns (query_local, gallery_local) positional index arrays."""
    nq, ng = len(query), len(gallery)
    if nq == 0 or ng == 0 or not _HAVE_FAISS:
        return np.zeros(0, int), np.zeros(0, int)
    query = np.ascontiguousarray(query, dtype=np.float32)
    gallery = np.ascontiguousarray(gallery, dtype=np.float32)
    d = gallery.shape[1]
    k = min(k, ng)
    if ng <= hnsw_max_exact:
        index = faiss.IndexFlatIP(d)
        index.add(gallery)
    else:
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.add(gallery)
        index.hnsw.efSearch = ef_search
    sims, idx = index.search(query, k)
    mask = (sims >= min_cosine) & (idx >= 0)
    q_local = np.repeat(np.arange(nq), k)[mask.ravel()]
    g_local = idx.ravel()[mask.ravel()]
    return q_local.astype(int), g_local.astype(int)


def paired_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine for dense L2-normalised vectors == row-wise dot product."""
    if len(A) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.einsum("ij,ij->i", A.astype(np.float32), B.astype(np.float32)).astype(np.float32)
