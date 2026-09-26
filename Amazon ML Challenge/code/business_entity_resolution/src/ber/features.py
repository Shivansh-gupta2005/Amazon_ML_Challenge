"""Pairwise features, computed in bounded streaming batches.

Cheap features (equality, lengths, postal, numbers, cosines) are vectorised over
the batch with numpy / scipy; only the fuzzy string ratios use a per-pair rapidfuzz
call (C-level). Nothing here materialises more than one batch of pairs at a time,
so memory stays flat regardless of the total candidate count.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from rapidfuzz import fuzz

from .vectorize import TextIndex

FEATURE_NAMES: List[str] = [
    "name_cos", "addr_cos",
    "rf_name_token_set", "rf_name_token_sort", "rf_name_partial", "rf_addr_token_set",
    "first_token_match", "acronym_match", "name_len_ratio",
    "postal_both_present", "postal_equal", "num_overlap", "num_any_match",
    "country_match", "cand_is_s3",
    # v2 additions
    "squish_cos", "num_exact", "addr_containment", "region_match",
    # v4 additions (cheap, precision-friendly, language-agnostic)
    "postal_prefix_match", "shared_long_token",
    # v5 addition (semantic; 0 when embeddings are disabled)
    "embed_cos",
]
N_FEATURES = len(FEATURE_NAMES)


def build_record_arrays(allrec) -> Dict[str, np.ndarray]:
    """Compact per-row arrays addressed by global row id (positional)."""
    acr = allrec["acronym"].to_numpy(dtype=object)
    return {
        "core_name": allrec["core_name"].to_numpy(dtype=object),
        "norm_addr": allrec["norm_addr"].to_numpy(dtype=object),
        "first_token": allrec["first_token"].to_numpy(dtype=object),
        "acronym": acr,
        "acr_len": np.array([len(x) for x in acr], dtype=np.int32),
        "core_len": allrec["core_len"].to_numpy(dtype=np.int32),
        "postal": allrec["postal"].to_numpy(dtype=object),
        "nums": allrec["nums"].to_numpy(dtype=object),
        "country_key": allrec["country_key"].to_numpy(dtype=object),
    }


def _num_overlap(nums_l: List[str], nums_r: List[str]):
    jac = np.zeros(len(nums_l), dtype=np.float32)
    anym = np.zeros(len(nums_l), dtype=np.float32)
    for i, (a, b) in enumerate(zip(nums_l, nums_r)):
        if not a or not b:
            continue
        sa, sb = set(a.split()), set(b.split())
        inter = len(sa & sb)
        if inter:
            anym[i] = 1.0
            jac[i] = inter / len(sa | sb)
    return jac, anym


def compute_batch(addr_mat, arr: Dict[str, np.ndarray],
                  s1_rows: np.ndarray, cand_rows: np.ndarray,
                  cand_src: np.ndarray, name_cos: np.ndarray,
                  squish_mat=None, embed_mat=None) -> np.ndarray:
    n = len(s1_rows)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    if n == 0:
        return X

    # address cosine via the precomputed address matrix (row lookup, no re-transform)
    X[:, 1] = TextIndex.paired_cosine(addr_mat[s1_rows], addr_mat[cand_rows])
    X[:, 0] = name_cos

    # fuzzy ratios (per-pair, C-level)
    lc = arr["core_name"][s1_rows].tolist()
    rc = arr["core_name"][cand_rows].tolist()
    la_l = arr["norm_addr"][s1_rows].tolist()
    ra_l = arr["norm_addr"][cand_rows].tolist()
    X[:, 2] = [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(lc, rc)]
    X[:, 3] = [fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(lc, rc)]
    X[:, 4] = [fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(lc, rc)]
    X[:, 5] = [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(la_l, ra_l)]

    # vectorised cheap features
    ft_l, ft_r = arr["first_token"][s1_rows], arr["first_token"][cand_rows]
    X[:, 6] = ((ft_l == ft_r) & (ft_l != "")).astype(np.float32)
    ac_l, ac_r = arr["acronym"][s1_rows], arr["acronym"][cand_rows]
    X[:, 7] = ((ac_l == ac_r) & (arr["acr_len"][s1_rows] >= 2)).astype(np.float32)
    cl, cr = arr["core_len"][s1_rows], arr["core_len"][cand_rows]
    mx = np.maximum(cl, cr)
    X[:, 8] = np.where(mx > 0, np.minimum(cl, cr) / np.maximum(mx, 1), 0.0)

    p_l, p_r = arr["postal"][s1_rows], arr["postal"][cand_rows]
    both = (p_l != "") & (p_r != "")
    X[:, 9] = both.astype(np.float32)
    X[:, 10] = (both & (p_l == p_r)).astype(np.float32)

    jac, anym = _num_overlap(arr["nums"][s1_rows].tolist(), arr["nums"][cand_rows].tolist())
    X[:, 11] = jac
    X[:, 12] = anym

    X[:, 13] = (arr["country_key"][s1_rows] == arr["country_key"][cand_rows]).astype(np.float32)
    X[:, 14] = (cand_src == 3).astype(np.float32)

    # v2 features
    if squish_mat is not None:
        X[:, 15] = TextIndex.paired_cosine(squish_mat[s1_rows], squish_mat[cand_rows])

    # single per-row pass computes all string-token features (v2 + v4) at once:
    #   num_exact (16), addr_containment (17), region_match (18)  -- v2
    #   postal_prefix_match (19), shared_long_token (20)          -- v4
    nums_l = arr["nums"][s1_rows].tolist(); nums_r = arr["nums"][cand_rows].tolist()
    pl = arr["postal"][s1_rows].tolist(); pr = arr["postal"][cand_rows].tolist()
    ne = np.zeros(n, np.float32); ac = np.zeros(n, np.float32); rm = np.zeros(n, np.float32)
    pp = np.zeros(n, np.float32); slt = np.zeros(n, np.float32)
    for i in range(n):
        na, nb = nums_l[i], nums_r[i]
        if na and nb and set(na.split()) == set(nb.split()):
            ne[i] = 1.0
        ta, tb = la_l[i].split(), ra_l[i].split()
        if ta and tb:
            sa, sb = set(ta), set(tb)
            m = min(len(sa), len(sb))
            ac[i] = (len(sa & sb) / m) if m else 0.0
            if ta[-1] == tb[-1]:
                rm[i] = 1.0
        # v4: postal region prefix (first 2 chars) -- FR departments / IN PIN regions
        a2, b2 = pl[i], pr[i]
        if a2 and b2 and a2[:2] == b2[:2]:
            pp[i] = 1.0
        # v4: a shared distinctive (len>=5) name token survives suffix noise / rebrands
        long_a = {t for t in lc[i].split() if len(t) >= 5}
        if long_a:
            if long_a & {t for t in rc[i].split() if len(t) >= 5}:
                slt[i] = 1.0
    X[:, 16] = ne; X[:, 17] = ac; X[:, 18] = rm; X[:, 19] = pp; X[:, 20] = slt

    # v5: semantic embedding cosine (0 when embeddings are disabled -> a constant
    # column the tree simply ignores, so feature width stays fixed at N_FEATURES)
    if embed_mat is not None:
        from .embed import paired_cosine as _emb_cos
        X[:, 21] = _emb_cos(np.asarray(embed_mat[s1_rows]), np.asarray(embed_mat[cand_rows]))
    return X
