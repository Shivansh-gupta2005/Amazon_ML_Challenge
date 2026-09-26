"""Candidate generation (blocking): name + address cosine top-n, unioned.

Per country and target source we run two sparse cosine top-n blockers, plus postal:
  * NAME char n-gram cosine top-n (robust to typos and word-order).
  * ADDRESS char n-gram cosine top-n -- the key recall driver on real data, where a
    true match often shares the address but has a completely different name (rebrand,
    domain/handle form, heavy typo). Name-only blocking can never see those.
High-document-frequency n-grams are pruned (name_max_df / addr_max_df) so each sparse
top-n product only touches rare, discriminative n-grams and stays cheap at tens-of-
millions scale. Postal-code matches are unioned in (no name gate -- postal is an
address signal). `name_cos` is then computed for every surviving candidate as a feature.
Everything is returned as flat numpy arrays keyed by global row id.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import embed
from .vectorize import TextIndex

try:
    from sparse_dot_topn import sp_matmul_topn
    _HAVE_SDT = True
except Exception:                        # pragma: no cover
    _HAVE_SDT = False


def _topn(Q: sp.csr_matrix, GT: sp.csr_matrix, k: int, thr: float, n_threads):
    """Top-k columns per row of Q @ GT; returns (row_local, col_local)."""
    nq = Q.shape[0]
    if nq == 0 or GT.shape[1] == 0:
        return np.zeros(0, int), np.zeros(0, int)
    k = min(k, GT.shape[1])
    if _HAVE_SDT:
        C = sp_matmul_topn(Q, GT, top_n=k, threshold=thr, sort=False, n_threads=n_threads).tocsr()
        return np.repeat(np.arange(nq), np.diff(C.indptr)).astype(int), C.indices.astype(int)
    out_r, out_c = [], []
    for r in range(nq):
        row = (Q[r] @ GT).toarray().ravel()
        cand = np.argpartition(row, -k)[-k:] if k < len(row) else np.arange(len(row))
        cand = cand[row[cand] >= thr]
        if cand.size:
            out_r.append(np.full(cand.size, r)); out_c.append(cand)
    if not out_r:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(out_r).astype(int), np.concatenate(out_c).astype(int)


def _cosine_block(transform, texts_s1, texts_tg, s1_rows, tg_rows, k, thr, chunk, n_threads):
    """Top-k gallery rows per Source-1 row by cosine of `transform`-ed text."""
    if len(s1_rows) == 0 or len(tg_rows) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    GT = transform(list(texts_tg)).T.tocsr()
    out_s1, out_cn = [], []
    for s in range(0, len(s1_rows), chunk):
        e = min(s + chunk, len(s1_rows))
        ql, gl = _topn(transform(list(texts_s1[s:e])), GT, k, thr, n_threads)
        if len(ql):
            out_s1.append(s1_rows[s:e][ql]); out_cn.append(tg_rows[gl])
    if not out_s1:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(out_s1), np.concatenate(out_cn)


def _postal(cfg, s1g, tgg):
    """Same-postal pairs (capped). No name gate -- postal is an address signal."""
    if not cfg.use_postal_blocking:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    left = s1g.loc[s1g["postal"] != "", ["row", "postal"]]
    right = tgg.loc[tgg["postal"] != "", ["row", "postal"]]
    if len(left) == 0 or len(right) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    vc = right["postal"].value_counts()
    ok = set(vc[vc <= cfg.postal_max_per_key].index)
    left = left[left["postal"].isin(ok)]; right = right[right["postal"].isin(ok)]
    if len(left) == 0 or len(right) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    m = left.merge(right, on="postal", suffixes=("_s1", "_tg"))
    return m["row_s1"].to_numpy(np.int64), m["row_tg"].to_numpy(np.int64)


def _dedup_pairs(s1r, cnr):
    if len(s1r) == 0:
        return s1r, cnr
    key = s1r.astype(np.int64) * (int(cnr.max()) + 1) + cnr.astype(np.int64)
    _, idx = np.unique(key, return_index=True)
    return s1r[idx], cnr[idx]


def _block_country_source(cfg, index, s1g, tgg, embed_mat=None):
    s1_rows = s1g["row"].to_numpy(np.int64)
    tg_rows = tgg["row"].to_numpy(np.int64)
    parts_s1, parts_cn = [], []

    a, b = _cosine_block(index.transform_name, s1g["name_key"].to_numpy(dtype=object),
                         tgg["name_key"].to_numpy(dtype=object), s1_rows, tg_rows,
                         cfg.knn_name, cfg.knn_min_cosine, cfg.knn_chunk_rows, cfg.sdt_threads)
    parts_s1.append(a); parts_cn.append(b)

    if cfg.use_addr_blocking:
        a2, b2 = _cosine_block(index.transform_addr, s1g["norm_addr"].to_numpy(dtype=object),
                               tgg["norm_addr"].to_numpy(dtype=object), s1_rows, tg_rows,
                               cfg.knn_addr, cfg.addr_min_cosine, cfg.knn_chunk_rows, cfg.sdt_threads)
        parts_s1.append(a2); parts_cn.append(b2)

    if cfg.use_squish_blocking:
        a3, b3 = _cosine_block(index.transform_squish, s1g["name_squish"].to_numpy(dtype=object),
                               tgg["name_squish"].to_numpy(dtype=object), s1_rows, tg_rows,
                               cfg.knn_squish, cfg.squish_min_cosine, cfg.knn_chunk_rows, cfg.sdt_threads)
        parts_s1.append(a3); parts_cn.append(b3)

    # SEMANTIC embedding blocker (v5): FAISS top-k on this group's embedding rows.
    # Catches matches that share meaning but few character n-grams (the France gap).
    if cfg.use_embeddings and embed_mat is not None:
        q = np.asarray(embed_mat[s1_rows]); g = np.asarray(embed_mat[tg_rows])
        el, gl = embed.faiss_block(q, g, cfg.knn_embed, cfg.embed_min_cosine,
                                   hnsw_max_exact=cfg.embed_hnsw_max_exact)
        if len(el):
            parts_s1.append(s1_rows[el]); parts_cn.append(tg_rows[gl])

    p1, p2 = _postal(cfg, s1g, tgg)
    parts_s1.append(p1); parts_cn.append(p2)
    return _dedup_pairs(np.concatenate(parts_s1), np.concatenate(parts_cn))


def generate_candidates(cfg, index: TextIndex, s1: pd.DataFrame, s2: pd.DataFrame,
                        s3: pd.DataFrame, name_key_by_row: np.ndarray,
                        log=lambda m: None, embed_mat=None) -> Dict[str, np.ndarray]:
    parts_s1, parts_cn, parts_src = [], [], []
    for tg, tag in ((s2, 2), (s3, 3)):
        if cfg.block_within_country:
            countries = sorted(set(s1["country_key"]).intersection(set(tg["country_key"])))
            groups = [(c, s1[s1["country_key"] == c], tg[tg["country_key"] == c]) for c in countries]
        else:
            groups = [("__all__", s1, tg)]
        for c, s1g, tgg in groups:
            if len(s1g) == 0 or len(tgg) == 0:
                continue
            a, b = _block_country_source(cfg, index, s1g, tgg, embed_mat=embed_mat)
            if len(a):
                parts_s1.append(a); parts_cn.append(b)
                parts_src.append(np.full(len(a), tag, dtype=np.int8))
            log(f"  S{tag} country='{c}': s1={len(s1g)} tgt={len(tgg)} pairs={len(a)}")

    if not parts_s1:
        z = np.zeros(0, np.int64)
        return {"s1_row": z, "cand_row": z, "cand_source": np.zeros(0, np.int8),
                "name_cos": np.zeros(0, np.float32)}
    s1_row = np.concatenate(parts_s1); cand_row = np.concatenate(parts_cn)
    cand_source = np.concatenate(parts_src)

    # name cosine for every surviving candidate (a model feature), in batches
    name_cos = np.zeros(len(s1_row), dtype=np.float32)
    bs = 1_000_000
    for s in range(0, len(s1_row), bs):
        e = min(s + bs, len(s1_row))
        A = index.transform_name(name_key_by_row[s1_row[s:e]].tolist())
        B = index.transform_name(name_key_by_row[cand_row[s:e]].tolist())
        name_cos[s:e] = TextIndex.paired_cosine(A, B)
    return {"s1_row": s1_row, "cand_row": cand_row,
            "cand_source": cand_source, "name_cos": name_cos}


def blocking_recall(cand: Dict[str, np.ndarray], id_by_row: np.ndarray,
                    ground_truth: Dict[str, set], s1_row_of: Dict[str, int]) -> dict:
    interest = {s1_row_of[s]: s for s in ground_truth if s in s1_row_of}
    got: Dict[str, set] = {}
    s1r, cnr = cand["s1_row"], cand["cand_row"]
    for i in range(len(s1r)):
        sid = interest.get(int(s1r[i]))
        if sid is not None:
            got.setdefault(sid, set()).add(id_by_row[cnr[i]])
    total = found = 0
    for sid, truth in ground_truth.items():
        if not truth:
            continue
        total += len(truth)
        found += len(truth & got.get(sid, set()))
    return {"candidate_pairs": int(len(s1r)), "pair_recall": (found / total) if total else 1.0}
