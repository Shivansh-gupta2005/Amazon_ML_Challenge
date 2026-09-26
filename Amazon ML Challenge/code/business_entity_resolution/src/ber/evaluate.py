"""Evaluation: macro-averaged F0.5 over Source1 entities, and threshold tuning.

Scoring follows the challenge spec exactly, including singletons:
  * true empty & pred empty -> 1.0
  * true empty & pred non-empty -> 0.0  (false merge on a singleton)
  * pred empty & true non-empty -> 0.0
  * otherwise F_beta with beta = 0.5
The final number is the simple mean across every Source1 entity in the eval set.
"""
from __future__ import annotations

from typing import Dict, List, Set

import numpy as np
import pandas as pd

from . import resolve


def f_beta_entity(pred: Set[str], true: Set[str], beta: float = 0.5) -> float:
    if not true and not pred:
        return 1.0
    if not true and pred:
        return 0.0
    if not pred and true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(true)
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1.0 + b2) * precision * recall / denom if denom else 0.0


def macro_f_beta(pred: Dict[str, Set[str]], truth: Dict[str, Set[str]],
                 s1_ids: List[str], beta: float = 0.5) -> float:
    if not s1_ids:
        return 0.0
    total = 0.0
    for sid in s1_ids:
        total += f_beta_entity(pred.get(sid, set()), truth.get(sid, set()), beta)
    return total / len(s1_ids)


def detailed_scores(pred: Dict[str, Set[str]], truth: Dict[str, Set[str]],
                    s1_ids: List[str], beta: float = 0.5) -> dict:
    f_vals, p_vals, r_vals = [], [], []
    for sid in s1_ids:
        p = pred.get(sid, set())
        t = truth.get(sid, set())
        f_vals.append(f_beta_entity(p, t, beta))
        if p and t:
            tp = len(p & t)
            p_vals.append(tp / len(p))
            r_vals.append(tp / len(t))
    return {
        "macro_f_beta": float(np.mean(f_vals)) if f_vals else 0.0,
        "mean_precision_on_overlap": float(np.mean(p_vals)) if p_vals else 0.0,
        "mean_recall_on_overlap": float(np.mean(r_vals)) if r_vals else 0.0,
        "n_entities": len(s1_ids),
    }


def tune_threshold(cfg, cand: pd.DataFrame, scores: np.ndarray,
                   truth: Dict[str, Set[str]], s1_ids: List[str]):
    """Grid-search the decision threshold to maximise macro F0.5 on the eval set."""
    grid = np.linspace(0.05, 0.95, cfg.threshold_grid_steps)
    best_t, best_f = cfg.min_threshold, -1.0
    for t in grid:
        if t < cfg.min_threshold:
            continue
        kept = resolve.assign_matches(cfg, cand, scores, t)
        pred = resolve.predictions_dict(kept, s1_ids)
        f = macro_f_beta(pred, truth, s1_ids, beta=cfg.beta)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def split_s1_ids(s1_ids: List[str], val_fraction: float, seed: int):
    """Deterministic train/val split of Source1 ids (pairs never cross the split)."""
    rng = np.random.default_rng(seed)
    ids = list(s1_ids)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_fraction))
    val = set(ids[:n_val])
    train = [i for i in ids if i not in val]
    return train, list(val)
