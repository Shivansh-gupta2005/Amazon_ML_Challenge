"""Character n-gram TF-IDF index with scalable sparse top-n blocking.

The vectorisers are fit once over all records; full matrices are never stored.
Blocking uses `sparse_dot_topn.sp_matmul_topn`, which computes A @ Bᵀ keeping only
the top-n entries per row -- so the (n_query x n_gallery) product is never
densified, and millions x millions blocking runs in bounded memory. When the
library is unavailable it falls back to a memory-capped scipy path.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sparse_dot_topn import sp_matmul_topn
    _HAVE_SDT = True
except Exception:                        # pragma: no cover
    _HAVE_SDT = False


class TextIndex:
    def __init__(self, cfg):
        self.cfg = cfg
        self.name_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=cfg.name_ngram_range,
            min_df=cfg.min_df, max_df=cfg.name_max_df, max_features=cfg.max_features_name,
            lowercase=False, dtype=np.float32)
        self.addr_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=cfg.addr_ngram_range,
            min_df=cfg.min_df, max_df=cfg.addr_max_df, max_features=cfg.max_features_addr,
            lowercase=False, dtype=np.float32)
        # plain 'char' (not char_wb) on the space-removed name, for domain/handle forms
        self.squish_vec = TfidfVectorizer(
            analyzer="char", ngram_range=cfg.squish_ngram_range,
            min_df=cfg.min_df, max_df=cfg.squish_max_df, max_features=cfg.max_features_name,
            lowercase=False, dtype=np.float32)

    def fit(self, name_texts, addr_texts, squish_texts=None) -> "TextIndex":
        self.name_vec.fit(name_texts)
        self.addr_vec.fit(addr_texts)
        self.squish_vec.fit(squish_texts if squish_texts is not None else name_texts)
        return self

    # transforms (bounded, on demand) ------------------------------------
    def transform_name(self, texts) -> sp.csr_matrix:
        return self.name_vec.transform(texts).tocsr()

    def transform_addr(self, texts) -> sp.csr_matrix:
        return self.addr_vec.transform(texts).tocsr()

    def transform_squish(self, texts) -> sp.csr_matrix:
        return self.squish_vec.transform(texts).tocsr()

    # paired-row cosine (L2-normalised rows -> dot == cosine) -------------
    @staticmethod
    def paired_cosine(A: sp.csr_matrix, B: sp.csr_matrix) -> np.ndarray:
        if A.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32)

    # top-k neighbours by name cosine ------------------------------------
    def topk(self, query: sp.csr_matrix, gallery: sp.csr_matrix, k: int,
             min_cosine: float, n_threads=None):
        """Return (q_local, g_local, sim) flat arrays; indices are row positions."""
        nq, ng = query.shape[0], gallery.shape[0]
        if nq == 0 or ng == 0:
            return (np.zeros(0, int), np.zeros(0, int), np.zeros(0, np.float32))
        k_eff = min(k, ng)

        if _HAVE_SDT:
            C = sp_matmul_topn(query, gallery.T.tocsr(), top_n=k_eff,
                               threshold=min_cosine, sort=False, n_threads=n_threads)
            C = C.tocsr()
            counts = np.diff(C.indptr)
            q_local = np.repeat(np.arange(nq), counts)
            return q_local.astype(int), C.indices.astype(int), C.data.astype(np.float32)

        # fallback: chunked dense top-k with a hard cap on gallery size
        return self._topk_scipy(query, gallery, k_eff, min_cosine)

    def _topk_scipy(self, query, gallery, k_eff, min_cosine, chunk=256):
        GT = gallery.T.tocsr()
        out_q, out_g, out_s = [], [], []
        for start in range(0, query.shape[0], chunk):
            Q = query[start:start + chunk]
            sims = (Q @ GT).toarray()
            for r in range(sims.shape[0]):
                row = sims[r]
                cand = np.argpartition(row, -k_eff)[-k_eff:] if k_eff < len(row) else np.arange(len(row))
                cand = cand[row[cand] >= min_cosine]
                if cand.size:
                    out_q.append(np.full(cand.size, start + r, dtype=int))
                    out_g.append(cand.astype(int))
                    out_s.append(row[cand].astype(np.float32))
        if not out_q:
            return (np.zeros(0, int), np.zeros(0, int), np.zeros(0, np.float32))
        return np.concatenate(out_q), np.concatenate(out_g), np.concatenate(out_s)
