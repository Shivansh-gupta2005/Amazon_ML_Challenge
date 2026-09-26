"""Robust TSV I/O and ground-truth parsing.

All challenge files are tab-separated; business addresses and the ID-list columns
contain commas, so we always read with sep='\t'. We also force dtype=str and
keep_default_na=False so that empty match lists stay '' rather than becoming NaN.
"""
from __future__ import annotations

import os
from typing import Dict, List, Set

import numpy as np
import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def _read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        quoting=3,          # csv.QUOTE_NONE -- IDs/addresses are never quoted
    )


def read_source(path: str) -> pd.DataFrame:
    """Read a *_source{1,2,3}.tsv file and guarantee the expected columns exist."""
    df = _read_tsv(path)
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; found {list(df.columns)}")
    for c in SOURCE_COLUMNS:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df = df[df["entity_id"] != ""].reset_index(drop=True)
    return df[SOURCE_COLUMNS].copy()


def parse_id_list(cell: str) -> List[str]:
    """Parse a comma-separated ID list cell into a de-duplicated, ordered list."""
    if cell is None:
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for tok in str(cell).split(","):
        tok = tok.strip()
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def read_ground_truth(path: str) -> Dict[str, Set[str]]:
    """Return {source1_entity_id -> set(matched entity ids)} (empty set = singleton).

    Vectorised: iterates two aligned columns with zip rather than DataFrame.iterrows,
    which is orders of magnitude faster on millions of rows.
    """
    df = _read_tsv(path)
    cols = {c.lower(): c for c in df.columns}
    s1_col = cols.get("source1_entity_id", df.columns[0])
    m_col = cols.get("matched_entity_ids", df.columns[1] if len(df.columns) > 1 else None)
    s1_vals = df[s1_col].astype(str).str.strip().to_numpy()
    m_vals = (df[m_col].astype(str).to_numpy() if m_col is not None
              else np.array([""] * len(df)))
    gt: Dict[str, Set[str]] = {}
    for s1, cell in zip(s1_vals, m_vals):
        if s1:
            gt[s1] = set(parse_id_list(cell))
    return gt


def write_id_list_tsv(path: str, id_col: str, list_col: str,
                      rows: List[tuple]) -> None:
    """Write a two-column TSV (id, comma-joined list) with no quoting.

    `rows` is an iterable of (source1_id, list_of_ids). Empty lists become ''.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"{id_col}\t{list_col}\n")
        for s1_id, ids in rows:
            # de-duplicate while preserving order, drop empties
            seen: Set[str] = set()
            clean: List[str] = []
            for i in ids:
                i = str(i).strip()
                if i and i not in seen:
                    seen.add(i)
                    clean.append(i)
            fh.write(f"{s1_id}\t{','.join(clean)}\n")


def resolve_source_paths(cfg, split: str) -> Dict[str, str]:
    """Locate the three source files (and ground truth for train) for a split.

    Tries the documented `dataset/<split>/<split>_sourceN.tsv` layout first, then a
    few tolerant fallbacks (no prefix, alternate directory) so the pipeline runs on
    slightly different folder conventions without edits.
    """
    prefix = cfg.train_prefix if split == "train" else cfg.test_prefix
    subdir = cfg.train_subdir if split == "train" else cfg.test_subdir
    base = os.path.join(cfg.data_dir, subdir)

    def find(kind_name: str) -> str:
        candidates = [
            os.path.join(base, prefix + kind_name),   # dataset/train/train_source1.tsv
            os.path.join(base, kind_name),            # dataset/train/source1.tsv
            os.path.join(cfg.data_dir, prefix + kind_name),
            os.path.join(cfg.data_dir, kind_name),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        raise FileNotFoundError(
            f"Could not find '{kind_name}' for split '{split}'. Tried: {candidates}"
        )

    paths = {
        "source1": find(cfg.source1_name),
        "source2": find(cfg.source2_name),
        "source3": find(cfg.source3_name),
    }
    if split == "train":
        paths["ground_truth"] = find(cfg.ground_truth_name)
    return paths
