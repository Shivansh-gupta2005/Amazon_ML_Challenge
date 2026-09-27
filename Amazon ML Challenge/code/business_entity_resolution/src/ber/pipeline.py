"""End-to-end orchestration, scaled for tens of millions of records.

Training runs on a subsample of Source-1 entities (a sample trains an equally good
pairwise matcher). Inference covers every Source-1 record, streaming candidate pairs
through featurisation + scoring in bounded batches and writing both output TSVs
incrementally so nothing large is held in memory.
"""
from __future__ import annotations

import os
import pickle
import time
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from . import blocking, embed, evaluate, features, io_utils, model, rerank, resolve
from .config import Config
from .normalize import add_normalized_columns
from .vectorize import TextIndex

_t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time()-_t0:6.0f}s] {msg}", flush=True)


# --------------------------------------------------------------------- context
class Context:
    """Normalised records, TF-IDF index, and fast row/id lookup structures."""

    def __init__(self, cfg: Config, s1_raw, s2_raw, s3_raw):
        log("normalising records (vectorised) ...")
        s1 = add_normalized_columns(s1_raw)
        s2 = add_normalized_columns(s2_raw)
        s3 = add_normalized_columns(s3_raw)
        self.n1, self.n2, self.n3 = len(s1), len(s2), len(s3)
        self.allrec = pd.concat([s1, s2, s3], ignore_index=True)
        self.allrec["row"] = np.arange(len(self.allrec))
        del s1, s2, s3

        log("fitting TF-IDF index ...")
        self.index = TextIndex(cfg).fit(self.allrec["name_key"].tolist(),
                                        self.allrec["norm_addr"].tolist(),
                                        self.allrec["name_squish"].tolist())
        self.N = len(self.allrec)
        self.arr = features.build_record_arrays(self.allrec)
        self.id_by_row = self.allrec["entity_id"].to_numpy(dtype=object)
        self.name_key_by_row = self.allrec["name_key"].to_numpy(dtype=object)
        self.id_index = pd.Index(self.allrec["entity_id"])
        cols = ["row", "country_key", "name_key", "postal", "norm_addr", "name_squish"]
        self.s1n = self.allrec.iloc[:self.n1][cols].copy()
        self.s2n = self.allrec.iloc[self.n1:self.n1 + self.n2][cols].copy()
        self.s3n = self.allrec.iloc[self.n1 + self.n2:][cols].copy()
        log("precomputing address + squish matrices ...")
        self.addr_mat = self.index.transform_addr(self.allrec["norm_addr"].tolist())
        self.squish_mat = self.index.transform_squish(self.allrec["name_squish"].tolist())

        # v5: semantic embeddings (optional). Encoded once, cached to a float32 memmap
        # addressed by global row id, reused by the embedding blocker and the feature.
        self.embed_mat = None
        if cfg.use_embeddings:
            enc = embed.load_encoder(cfg)
            texts = self.allrec[cfg.embed_text_col].tolist()
            emb_dir = cfg.model_dir or "."
            emb_path = os.path.join(emb_dir, f"emb_{len(self.allrec)}_{cfg.embed_text_col}.f32")
            log(f"encoding embeddings [{cfg.embed_backend}] -> {emb_path} ...")
            self.embed_mat = embed.encode_to_memmap(
                enc, texts, emb_path, batch=cfg.embed_batch_size,
                enc_batch=cfg.embed_enc_batch, log=log)
            log(f"embeddings ready: {self.embed_mat.shape}")

        del self.allrec                       # free the big concatenated frame

    def s1_ids(self) -> List[str]:
        return self.id_by_row[:self.n1].tolist()

    def rows_for_ids(self, ids) -> np.ndarray:
        return self.id_index.get_indexer(pd.Index(list(ids)))

    @classmethod
    def get_or_create(cls, cfg: Config, split: str = "train") -> "Context":
        cache_file = None
        if getattr(cfg, "use_cache", True) and getattr(cfg, "cache_dir", None):
            os.makedirs(cfg.cache_dir, exist_ok=True)
            cache_file = os.path.join(cfg.cache_dir, f"context_{split}.pkl")
            if os.path.isfile(cache_file):
                log(f"found cached {split} context -> {cache_file}")
                try:
                    t_start = time.time()
                    with open(cache_file, "rb") as fh:
                        ctx = pickle.load(fh)
                    log(f"loaded cached {split} context in {time.time() - t_start:.1f}s (N={ctx.N})")
                    return ctx
                except Exception as e:
                    log(f"warning: failed to load cache ({e}); recomputing from raw sources ...")

        paths = io_utils.resolve_source_paths(cfg, split)
        subdir = getattr(cfg, f"{split}_subdir")
        log(f"loading {split} sources from {os.path.join(cfg.data_dir, subdir)}")
        s1 = io_utils.read_source(paths["source1"])
        s2 = io_utils.read_source(paths["source2"])
        s3 = io_utils.read_source(paths["source3"])
        log(f"{split} records: S1={len(s1)} S2={len(s2)} S3={len(s3)}")
        ctx = cls(cfg, s1, s2, s3)
        del s1, s2, s3

        if getattr(cfg, "use_cache", True) and cache_file:
            log(f"saving {split} context to cache: {cache_file} ...")
            try:
                t_save = time.time()
                with open(cache_file, "wb") as fh:
                    pickle.dump(ctx, fh, protocol=pickle.HIGHEST_PROTOCOL)
                mb = os.path.getsize(cache_file) / (1024 * 1024)
                log(f"saved {split} context cache ({mb:.1f} MB in {time.time() - t_save:.1f}s)")
            except Exception as e:
                log(f"warning: failed to save cache ({e})")

        return ctx


# ------------------------------------------------------------------- labelling
def label_pairs(ctx: Context, cand: Dict[str, np.ndarray], gt: Dict[str, Set[str]]) -> np.ndarray:
    s1_ids = ctx.id_by_row[cand["s1_row"]]
    cand_ids = ctx.id_by_row[cand["cand_row"]]
    empty: Set[str] = set()
    y = np.zeros(len(s1_ids), dtype=np.int8)
    for i in range(len(s1_ids)):
        if cand_ids[i] in gt.get(s1_ids[i], empty):
            y[i] = 1
    return y


_EMPTY_CAND = {"s1_row": np.zeros(0, np.int64), "cand_row": np.zeros(0, np.int64),
               "cand_source": np.zeros(0, np.int8), "name_cos": np.zeros(0, np.float32)}


def injected_only(ctx: Context, cand: Dict[str, np.ndarray], gt: Dict[str, Set[str]],
                  s1_rows) -> Dict[str, np.ndarray]:
    """Return ONLY the true-positive pairs blocking missed for the given Source-1 rows."""
    N = ctx.N
    have = set((cand["s1_row"].astype(np.int64) * N + cand["cand_row"].astype(np.int64)).tolist())
    add_s1, add_mid = [], []
    for r in s1_rows:
        for mid in gt.get(ctx.id_by_row[r], ()):
            add_s1.append(int(r)); add_mid.append(mid)
    if not add_s1:
        return dict(_EMPTY_CAND)
    mid_rows = ctx.rows_for_ids(add_mid)
    add_s1 = np.array(add_s1, dtype=np.int64)
    keep = mid_rows >= 0
    add_s1, mid_rows = add_s1[keep], mid_rows[keep].astype(np.int64)
    fresh = np.array([k not in have for k in (add_s1 * N + mid_rows)], dtype=bool)
    add_s1, mid_rows = add_s1[fresh], mid_rows[fresh]
    if len(add_s1) == 0:
        return dict(_EMPTY_CAND)
    src = np.where(np.char.startswith(ctx.id_by_row[mid_rows].astype(str), "S3"), 3, 2).astype(np.int8)
    cos = TextIndex.paired_cosine(ctx.index.transform_name(ctx.name_key_by_row[add_s1].tolist()),
                                  ctx.index.transform_name(ctx.name_key_by_row[mid_rows].tolist()))
    return {"s1_row": add_s1, "cand_row": mid_rows, "cand_source": src, "name_cos": cos}


# ----------------------------------------------------------------- featurise
def featurize_all(cfg: Config, ctx: Context, cand: Dict[str, np.ndarray]) -> np.ndarray:
    n = len(cand["s1_row"])
    X = np.zeros((n, features.N_FEATURES), dtype=np.float32)
    bs = cfg.feature_batch_size
    for s in range(0, n, bs):
        e = min(s + bs, n)
        X[s:e] = features.compute_batch(ctx.addr_mat, ctx.arr,
                                        cand["s1_row"][s:e], cand["cand_row"][s:e],
                                        cand["cand_source"][s:e], cand["name_cos"][s:e],
                                        squish_mat=ctx.squish_mat, embed_mat=ctx.embed_mat)
    return X


def _subset(cand: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[mask] for k, v in cand.items()}


# ------------------------------------------------------- cold-country calibration
def estimate_cold_threshold(cfg: Config, ctx: Context, cand: Dict[str, np.ndarray],
                            X_base: np.ndarray, y_base: np.ndarray,
                            X_inj: np.ndarray, y_inj: np.ndarray, inj: Dict[str, np.ndarray],
                            gt: Dict[str, Set[str]], sample_rows: np.ndarray,
                            fallback: float):
    """Leave-one-country-out (LOCO) estimate of the threshold for an UNSEEN country.

    The test set contains France, which never appears in training. A threshold tuned
    on the in-distribution (US+India) validation split is therefore calibrated for
    the wrong distribution. To estimate the right bar for a genuinely unseen country
    we, for each training country c: train a fold model on every OTHER country's
    pairs, then tune a threshold on country c -- which that fold model has never seen,
    exactly the situation France is in. The median of those LOCO thresholds is applied
    to any unseen test country. This measures the correct direction (higher OR lower)
    instead of assuming precision is the only problem. Fold models reuse the already
    computed features, so this only costs a few fast LightGBM refits, not re-blocking.
    """
    base_country = ctx.arr["country_key"][cand["s1_row"]]
    inj_country = (ctx.arr["country_key"][inj["s1_row"]]
                   if len(inj["s1_row"]) else np.zeros(0, dtype=object))
    sample_country = ctx.arr["country_key"][sample_rows]

    usable = [c for c in sorted(set(base_country.tolist()))
              if int(y_base[base_country == c].sum()) >= cfg.cold_min_positive_eval]
    if len(usable) < 2:
        log("cold-country: <2 usable training countries -> using in-distribution threshold")
        return fallback, []

    fold_cfg = Config(**{**cfg.__dict__})
    fold_cfg.lgbm_params = {**cfg.lgbm_params, "n_estimators": cfg.cold_fold_estimators}
    ts, plog = [], []
    for c in usable:
        tr_b = base_country != c
        Xc, yc = X_base[tr_b], y_base[tr_b]
        if len(X_inj):
            tr_i = inj_country != c
            Xc = np.vstack([Xc, X_inj[tr_i]]); yc = np.concatenate([yc, y_inj[tr_i]])
        if len(np.unique(yc)) < 2:
            continue
        fold = model.train_matcher(fold_cfg, Xc, yc)
        ev = base_country == c
        ev_scores = model.predict_scores(fold, X_base[ev])
        ev_df = pd.DataFrame({"s1_id": ctx.id_by_row[cand["s1_row"][ev]],
                              "cand_id": ctx.id_by_row[cand["cand_row"][ev]]})
        ev_ids = [ctx.id_by_row[r] for r, cc in zip(sample_rows, sample_country) if cc == c]
        t_c, f_c = evaluate.tune_threshold(cfg, ev_df, ev_scores, gt, ev_ids)
        ts.append(t_c); plog.append((c, float(t_c), float(f_c)))
    if not ts:
        return fallback, []
    cold = float(np.clip(np.median(ts), cfg.cold_threshold_floor, cfg.cold_threshold_cap))
    return cold, plog


# ------------------------------------------------------------------- training
def train_and_validate(cfg: Config):
    paths = io_utils.resolve_source_paths(cfg, "train")
    gt = io_utils.read_ground_truth(paths["ground_truth"])
    log(f"gt entities={len(gt)}")

    ctx = Context.get_or_create(cfg, "train")

    # subsample Source-1 entities for training
    rng = np.random.default_rng(cfg.lgbm_params["random_state"])
    s1_rows_all = np.arange(ctx.n1)
    if cfg.max_train_entities and ctx.n1 > cfg.max_train_entities:
        sample_rows = np.sort(rng.choice(s1_rows_all, cfg.max_train_entities, replace=False))
        log(f"subsampling {cfg.max_train_entities} of {ctx.n1} Source-1 entities for training")
    else:
        sample_rows = s1_rows_all
    s1n_s = ctx.s1n.loc[ctx.s1n["row"].isin(set(sample_rows.tolist()))]

    log("blocking (train sample) ...")
    cand = blocking.generate_candidates(cfg, ctx.index, s1n_s, ctx.s2n, ctx.s3n,
                                        ctx.name_key_by_row, log=lambda m: None,
                                        embed_mat=ctx.embed_mat)
    log(f"train-sample candidate pairs: {len(cand['s1_row'])}")

    # train / val split by Source-1 row
    val_mask_ids = set(evaluate.split_s1_ids(
        ctx.id_by_row[sample_rows].tolist(), cfg.val_fraction, cfg.lgbm_params["random_state"])[1])
    val_rows = set(int(r) for r in sample_rows if ctx.id_by_row[r] in val_mask_ids)
    is_val = np.array([int(r) in val_rows for r in cand["s1_row"]], dtype=bool)
    # blocking recall on the validation sample
    val_ids = [ctx.id_by_row[r] for r in sample_rows if int(r) in val_rows]
    s1_row_of = {ctx.id_by_row[r]: int(r) for r in sample_rows if int(r) in val_rows}
    rec = blocking.blocking_recall(_subset(cand, is_val), ctx.id_by_row,
                                   {k: gt.get(k, set()) for k in val_ids}, s1_row_of)
    log(f"blocking (val): pair_recall={rec['pair_recall']:.4f} pairs={rec['candidate_pairs']}")

    # featurise the blocked candidates ONCE; reuse for the val model and the refit
    log("featurising candidates ...")
    X_base = featurize_all(cfg, ctx, cand); y_base = label_pairs(ctx, cand, gt)

    # injected missed positives (true positives blocking missed) -- featurised once too
    if cfg.inject_missed_positives:
        inj = injected_only(ctx, cand, gt, sample_rows)
        X_inj = featurize_all(cfg, ctx, inj)
        y_inj = np.ones(len(inj["s1_row"]), dtype=np.int8)
        inj_val = np.array([int(r) in val_rows for r in inj["s1_row"]], dtype=bool)
    else:
        inj = dict(_EMPTY_CAND)
        X_inj = np.zeros((0, features.N_FEATURES), np.float32); y_inj = np.zeros(0, np.int8)
        inj_val = np.zeros(0, bool)

    Xtr = np.vstack([X_base[~is_val], X_inj[~inj_val]])
    ytr = np.concatenate([y_base[~is_val], y_inj[~inj_val]])
    Xva, yva = X_base[is_val], y_base[is_val]
    log(f"train pairs={len(ytr)} (pos={int(ytr.sum())}) | val pairs={len(yva)} (pos={int(yva.sum())})")

    log("training LightGBM matcher ...")
    clf = model.train_matcher(cfg, Xtr, ytr, Xva, yva)

    cand_va = _subset(cand, is_val)
    va_df = pd.DataFrame({"s1_id": ctx.id_by_row[cand_va["s1_row"]],
                          "cand_id": ctx.id_by_row[cand_va["cand_row"]]})
    va_scores = model.predict_scores(clf, Xva)
    threshold, val_f = evaluate.tune_threshold(cfg, va_df, va_scores, gt, val_ids)
    kept = resolve.assign_matches(cfg, va_df, va_scores, threshold)
    det = evaluate.detailed_scores(resolve.predictions_dict(kept, val_ids), gt, val_ids, cfg.beta)
    log(f"VALIDATION macro_F0.5={val_f:.4f} threshold={threshold:.3f} "
        f"prec@overlap={det['mean_precision_on_overlap']:.3f} rec@overlap={det['mean_recall_on_overlap']:.3f}")
    log("top features: " + ", ".join(f"{n}={v}" for n, v in
                                     model.feature_importance(clf, features.FEATURE_NAMES)[:8]))

    # cold-country threshold (v4): the bar to apply to unseen test countries (France).
    train_countries = sorted(set(ctx.arr["country_key"].tolist()))
    if cfg.cold_threshold_override is not None:
        cold_threshold = float(cfg.cold_threshold_override)
        log(f"cold-country: using override threshold={cold_threshold:.3f}")
    elif cfg.use_cold_country_threshold:
        cold_threshold, plog = estimate_cold_threshold(
            cfg, ctx, cand, X_base, y_base, X_inj, y_inj, inj, gt, sample_rows, threshold)
        if plog:
            log("cold-country LOCO folds: " +
                ", ".join(f"{c}: t={t:.3f} F0.5={f:.3f}" for c, t, f in plog))
        log(f"cold-country: threshold for UNSEEN countries = {cold_threshold:.3f} "
            f"(in-distribution = {threshold:.3f})")
    else:
        cold_threshold = threshold
        log("cold-country: disabled -> unseen countries use the in-distribution threshold")

    # refit final model on all sampled candidates + injected positives (reuse features)
    log("refitting final model on full training sample ...")
    best_iter = getattr(clf, "best_iteration_", None)
    configured = cfg.lgbm_params["n_estimators"]
    n_trees = min(max(int(best_iter), cfg.min_final_estimators), configured) if best_iter else configured
    final_cfg = Config(**{**cfg.__dict__}); final_cfg.lgbm_params = {**cfg.lgbm_params, "n_estimators": n_trees}
    Xall = np.vstack([X_base, X_inj]); yall = np.concatenate([y_base, y_inj])
    final_model = model.train_matcher(final_cfg, Xall, yall)
    log(f"final model trees={n_trees}")
    return final_model, threshold, {
        "val_macro_f_beta": val_f, "threshold": threshold,
        "cold_threshold": cold_threshold, "train_countries": train_countries,
        "blocking": rec,
    }


# ----------------------------------------------------------------- prediction
def predict_test(cfg: Config, clf, threshold: float,
                 cold_threshold: Optional[float] = None,
                 train_countries: Optional[List[str]] = None) -> dict:
    ctx = Context.get_or_create(cfg, "test")
    all_s1_ids = ctx.s1_ids()

    log("blocking (all test entities) ...")
    cand = blocking.generate_candidates(cfg, ctx.index, ctx.s1n, ctx.s2n, ctx.s3n,
                                        ctx.name_key_by_row, log=lambda m: None,
                                        embed_mat=ctx.embed_mat)
    log(f"test candidate pairs: {len(cand['s1_row'])}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    cand_path = os.path.join(cfg.output_dir, "candidate_pairs.tsv")
    _write_grouped(cand_path, "candidate_entity_ids", ctx, cand["s1_row"], cand["cand_row"],
                   all_s1_ids)
    log(f"wrote {cand_path}")

    # per-row decision bar: unseen test countries (France) use the cold-calibrated
    # threshold; countries seen in training keep the in-distribution threshold.
    seen = set(train_countries or [])
    use_cold = (cfg.use_cold_country_threshold and cold_threshold is not None
                and cold_threshold != threshold and len(seen) > 0)
    if use_cold:
        log(f"applying cold threshold={cold_threshold:.3f} to test countries not in "
            f"training ({len(seen)} seen); seen countries use {threshold:.3f}")

    # stream featurise + score, keep only survivors
    log("scoring candidates (streamed) ...")
    ce_on = cfg.use_cross_encoder
    band = cfg.cross_encoder_band
    n = len(cand["s1_row"]); bs = cfg.feature_batch_size
    keep_s1, keep_cn, keep_sc = [], [], []
    bl_s1, bl_cn, bl_sc, bl_thr = [], [], [], []   # borderline pairs (cross-encoder path)
    for s in range(0, n, bs):
        e = min(s + bs, n)
        s1_slice = cand["s1_row"][s:e]; cn_slice = cand["cand_row"][s:e]
        X = features.compute_batch(ctx.addr_mat, ctx.arr, s1_slice, cn_slice,
                                   cand["cand_source"][s:e], cand["name_cos"][s:e],
                                   squish_mat=ctx.squish_mat, embed_mat=ctx.embed_mat)
        sc = model.predict_scores(clf, X)
        if use_cold:
            is_cold = ~np.isin(ctx.arr["country_key"][s1_slice], list(seen))
            thr = np.where(is_cold, cold_threshold, threshold).astype(np.float32)
        else:
            thr = np.full(len(sc), threshold, dtype=np.float32)

        if ce_on:
            # confident matches kept now; the uncertain band is deferred to the
            # cross-encoder (it can flip a decision either way).
            m = sc >= (thr + band)
            bmask = (sc >= (thr - band)) & (sc < (thr + band))
            if bmask.any():
                bl_s1.append(s1_slice[bmask]); bl_cn.append(cn_slice[bmask])
                bl_sc.append(sc[bmask]); bl_thr.append(thr[bmask])
        else:
            m = sc >= thr
            # optional corroboration tier: rescue borderline pairs with a strong address
            # signal (same postal, or high address cosine) without lowering the global bar.
            if cfg.corroboration_delta > 0:
                corr = (sc >= (thr - cfg.corroboration_delta)) & \
                       ((X[:, 10] > 0) | (X[:, 1] >= cfg.corroboration_addr_cos))
                m = m | corr
        if m.any():
            keep_s1.append(s1_slice[m]); keep_cn.append(cn_slice[m])
            keep_sc.append(sc[m])
        if (s // bs) % 20 == 0:
            log(f"  scored {e}/{n}")

    # cross-encoder rerank of the collected borderline pairs (optional)
    if ce_on and bl_s1:
        b1 = np.concatenate(bl_s1); b2 = np.concatenate(bl_cn)
        bsc = np.concatenate(bl_sc); bth = np.concatenate(bl_thr)
        log(f"cross-encoder reranking {len(b1)} borderline pairs [{cfg.cross_encoder_backend}] ...")
        # compose name|address|country text (identical to training) for each side
        cn = ctx.arr["core_name"]; ad = ctx.arr["norm_addr"]; co = ctx.arr["country_key"]
        ta = rerank.compose_texts(cn[b1], ad[b1], co[b1])
        tb = rerank.compose_texts(cn[b2], ad[b2], co[b2])
        pairs = list(zip(ta, tb))
        reranker = rerank.load_reranker(cfg)
        ce = reranker.score(pairs)
        blended = rerank.blend(bsc, ce, cfg.cross_encoder_weight)
        take = blended >= bth
        if take.any():
            keep_s1.append(b1[take]); keep_cn.append(b2[take]); keep_sc.append(blended[take])
        log(f"cross-encoder kept {int(take.sum())}/{len(b1)} borderline pairs")

    if keep_s1:
        ks1 = np.concatenate(keep_s1); kcn = np.concatenate(keep_cn); ksc = np.concatenate(keep_sc)
    else:
        ks1 = np.zeros(0, int); kcn = np.zeros(0, int); ksc = np.zeros(0, np.float32)

    # target uniqueness: each candidate id kept for its best-scoring Source-1
    if cfg.enforce_target_uniqueness and len(ks1):
        order = np.argsort(-ksc, kind="stable")
        ks1, kcn, ksc = ks1[order], kcn[order], ksc[order]
        _, first = np.unique(kcn, return_index=True)
        ks1, kcn, ksc = ks1[first], kcn[first], ksc[first]

    match_path = os.path.join(cfg.output_dir, "matching_results.tsv")
    _write_grouped(match_path, "matched_entity_ids", ctx, ks1, kcn, all_s1_ids, scores=ksc)
    n_matched = len(np.unique(ks1)) if len(ks1) else 0
    log(f"wrote {match_path} ({n_matched} entities matched)")
    return {"candidate_pairs": cand_path, "matching_results": match_path,
            "n_s1": len(all_s1_ids), "n_pairs": int(n), "n_matched_entities": int(n_matched)}


def _write_grouped(path: str, list_col: str, ctx: Context, s1_rows: np.ndarray,
                   cand_rows: np.ndarray, all_s1_ids: List[str],
                   scores: Optional[np.ndarray] = None) -> None:
    """Stream a one-row-per-Source-1 TSV without materialising all lists in memory.

    Candidates are sorted by Source-1 row (then score desc), then swept once with
    searchsorted boundaries -- so even 100M candidate pairs write in bounded memory.
    """
    n1 = len(all_s1_ids)
    starts = ends = None
    cand_ids = None
    if len(s1_rows):
        order = np.lexsort((-scores, s1_rows)) if scores is not None else np.argsort(s1_rows, kind="stable")
        s1_rows = s1_rows[order]
        cand_ids = ctx.id_by_row[cand_rows[order]]
        arange = np.arange(n1)
        starts = np.searchsorted(s1_rows, arange, side="left")
        ends = np.searchsorted(s1_rows, arange, side="right")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"source1_entity_id\t{list_col}\n")
        for r in range(n1):
            if starts is not None and ends[r] > starts[r]:
                seen: Set[str] = set(); clean = []
                for i in cand_ids[starts[r]:ends[r]]:
                    if i not in seen:
                        seen.add(i); clean.append(i)
                fh.write(f"{all_s1_ids[r]}\t{','.join(clean)}\n")
            else:
                fh.write(f"{all_s1_ids[r]}\t\n")


# ------------------------------------------------------------------ full runs
def run_all(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    final_model, threshold, metrics = train_and_validate(cfg)
    cold_threshold = metrics.get("cold_threshold", threshold)
    train_countries = metrics.get("train_countries", [])
    model_path = os.path.join(cfg.model_dir, "matcher.pkl")
    model.save_model(model_path, final_model, threshold, features.FEATURE_NAMES,
                     extra={"cold_threshold": cold_threshold, "train_countries": train_countries})
    log(f"saved model -> {model_path}")
    test_s1 = os.path.join(cfg.data_dir, cfg.test_subdir)
    if os.path.isdir(test_s1) or os.path.exists(os.path.join(cfg.data_dir, cfg.test_prefix + cfg.source1_name)):
        try:
            metrics["prediction"] = predict_test(cfg, final_model, threshold,
                                                 cold_threshold=cold_threshold,
                                                 train_countries=train_countries)
        except FileNotFoundError as e:
            log(f"skipping test prediction: {e}")
    else:
        log("no test split found -- trained + validated only.")
    return metrics


def predict_only(cfg: Config):
    saved = model.load_model(os.path.join(cfg.model_dir, "matcher.pkl"))
    extra = saved.get("extra", {})
    return predict_test(cfg, saved["model"], saved["threshold"],
                        cold_threshold=extra.get("cold_threshold"),
                        train_countries=extra.get("train_countries"))
