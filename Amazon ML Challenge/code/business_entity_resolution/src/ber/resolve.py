"""Decision / conflict-resolution layer.

Turns pairwise scores into the final matches. Two ideas, both aimed at precision
(the F0.5 objective punishes false merges 2x harder than misses):

  1. Threshold: keep only pairs with score >= threshold.
  2. Target uniqueness: Source1 is the deduplicated reference, so a given Source2/
     Source3 record refers to exactly one real business and may therefore be claimed
     by at most one Source1 entity. When several Source1 entities score the same
     target above threshold, only the single highest-scoring one keeps it.

A Source1 entity may still match many Source2/Source3 records (one-to-many on the
Source1 side), so we never cap the number of matches per Source1 entity.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def assign_matches(cfg, cand: pd.DataFrame, scores: np.ndarray,
                   threshold: float) -> pd.DataFrame:
    """Return the surviving (s1_id, cand_id, score) rows after threshold + uniqueness."""
    df = cand.copy()
    df["score"] = scores
    kept = df[df["score"] >= threshold].copy()
    if len(kept) == 0:
        return kept

    if cfg.enforce_target_uniqueness:
        # each candidate id kept only for its best-scoring Source1 entity
        kept = kept.sort_values("score", ascending=False)
        kept = kept.drop_duplicates(subset=["cand_id"], keep="first")
    return kept


def matches_by_s1(kept: pd.DataFrame, all_s1_ids: List[str]) -> List[tuple]:
    """One row per Source1 id (empty list for singletons / no-match entities)."""
    if len(kept):
        grouped: Dict[str, List[str]] = (
            kept.sort_values("score", ascending=False)
            .groupby("s1_id")["cand_id"].apply(list).to_dict()
        )
    else:
        grouped = {}
    return [(sid, grouped.get(sid, [])) for sid in all_s1_ids]


def predictions_dict(kept: pd.DataFrame, all_s1_ids: List[str]) -> Dict[str, set]:
    out = {sid: set() for sid in all_s1_ids}
    if len(kept):
        for sid, cid in zip(kept["s1_id"], kept["cand_id"]):
            out.setdefault(sid, set()).add(cid)
    return out
