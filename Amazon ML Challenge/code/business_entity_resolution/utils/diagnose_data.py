#!/usr/bin/env python3
"""Diagnose WHERE the score is hiding, using the labelled TRAIN split.

The question this answers: are top scores (0.98+) reachable because the data has a
strong EXACT key (phone / email / domain / registration number) that resolves most
matches deterministically at high precision -- something a fuzzy-first pipeline
under-exploits? Or is the remaining signal genuinely fuzzy?

It measures, on ground-truth pairs:
  1. Match structure   -- singletons vs matched, match-count distribution, S2 vs S3.
  2. Per exact key      -- COVERAGE (what % of true pairs share the key) and a bounded
                           deterministic-rule PRECISION/RECALL (match iff the key is
                           shared, restricted to small clusters so precision is real).
  3. Deterministic reach-- % of true pairs recoverable by the union of high-precision keys.
  4. Residual hardness  -- for true pairs NO exact key catches, how similar are the names
                           (are they near-duplicates fuzzy matching can get, or truly divergent).

Everything is computed from the provided files only -- no external data. Read-only.

Usage:
    python3 utils/diagnose_data.py --data-dir "<dataset>"        # uses the train split
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ber import io_utils                       # noqa: E402
from ber.normalize import add_normalized_columns  # noqa: E402

BOUND = 40          # only trust a key value whose total cluster is <= BOUND for precision
HIGH_PREC = 0.90    # a key counts toward "deterministic reach" if its rule precision >= this


def _blob(name: pd.Series, addr: pd.Series) -> pd.Series:
    return (name.fillna("") + " " + addr.fillna("")).str.lower()


def _extract(blob: pd.Series):
    """Vectorised regex extraction of set-valued raw keys."""
    phones = blob.str.replace(r"[^0-9]", " ", regex=True).str.findall(r"\d{10,12}")
    phones = phones.apply(lambda xs: {x[-10:] for x in xs})           # last 10 digits
    emails = blob.str.findall(r"[a-z0-9._%+-]+@[a-z0-9-]+\.[a-z0-9.-]+").apply(set)
    domains = blob.str.findall(r"[a-z0-9][a-z0-9-]*\.(?:com|in|co|org|net|io|biz|info|us|fr)").apply(set)
    longnums = blob.str.findall(r"\d{7,}").apply(set)
    return phones, emails, domains, longnums


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args(argv)

    from ber.config import Config
    cfg = Config(); cfg.data_dir = args.data_dir
    paths = io_utils.resolve_source_paths(cfg, args.split)
    print(f"loading {args.split} sources from {os.path.dirname(paths['source1'])} ...", flush=True)
    s1 = io_utils.read_source(paths["source1"])
    s2 = io_utils.read_source(paths["source2"])
    s3 = io_utils.read_source(paths["source3"])
    gt = io_utils.read_ground_truth(paths["ground_truth"])

    allrec = pd.concat([s1, s2, s3], ignore_index=True)
    norm = add_normalized_columns(allrec)
    eid = allrec["entity_id"].to_numpy(dtype=object)
    row_of = {e: i for i, e in enumerate(eid)}
    src = np.array([1] * len(s1) + [2] * len(s2) + [3] * len(s3))

    print("extracting raw keys (phone/email/domain/number) ...", flush=True)
    phones, emails, domains, longnums = _extract(_blob(allrec["business_name"], allrec["business_address"]))

    # scalar keys -> per-row string arrays (empty string == absent)
    scal = {
        "exact_core_name":  norm["core_name"].to_numpy(dtype=object),
        "exact_squish_name": norm["name_squish"].to_numpy(dtype=object),
        "postal":           norm["postal"].to_numpy(dtype=object),
        "first_token":      norm["first_token"].to_numpy(dtype=object),
        "name+postal":      np.array([f"{n}|{p}" if n and p else "" for n, p in
                                      zip(norm["core_name"], norm["postal"])], dtype=object),
    }
    setk = {"phone": phones.tolist(), "email": emails.tolist(),
            "domain": domains.tolist(), "long_number": longnums.tolist()}

    # ---- ground-truth pair list (row_a in S1, row_b in target) ----
    pa, pb = [], []
    n_singleton = n_matched = 0
    mcount = []
    src_of_match = {2: 0, 3: 0}
    for s1_id, matched in gt.items():
        a = row_of.get(s1_id)
        if a is None:
            continue
        if not matched:
            n_singleton += 1; mcount.append(0); continue
        n_matched += 1; k = 0
        for mid in matched:
            b = row_of.get(mid)
            if b is not None:
                pa.append(a); pb.append(b); k += 1
                src_of_match[src[b]] = src_of_match.get(src[b], 0) + 1
        mcount.append(k)
    pa = np.array(pa); pb = np.array(pb)
    total_pairs = len(pa)

    n_s1 = n_singleton + n_matched
    print("\n" + "=" * 66)
    print("1. MATCH STRUCTURE")
    print("=" * 66)
    print(f"Source1 entities        : {n_s1}")
    print(f"  singletons (0 matches): {n_singleton}  ({100*n_singleton/max(n_s1,1):.1f}%)  <- scored 1.0 if predicted empty")
    print(f"  with >=1 match        : {n_matched}  ({100*n_matched/max(n_s1,1):.1f}%)")
    mc = np.array(mcount)
    print(f"matches per entity      : mean={mc.mean():.2f}  max={mc.max()}  "
          f"(1:{int((mc==1).sum())}, 2:{int((mc==2).sum())}, 3+:{int((mc>=3).sum())})")
    print(f"true matched pairs      : {total_pairs}   (to S2: {src_of_match.get(2,0)}, to S3: {src_of_match.get(3,0)})")

    # ---- per-key coverage + bounded deterministic precision/recall ----
    print("\n" + "=" * 66)
    print("2. EXACT-KEY ANALYSIS  (per key: coverage of true pairs, and a")
    print("   deterministic 'match-iff-shared' rule's precision/recall)")
    print("=" * 66)
    print(f"{'key':<18}{'coverage':>10}{'rule_prec':>11}{'rule_rec':>10}")
    print("-" * 66)

    N = len(eid)
    is_s1 = (src == 1)
    gt_keys = np.sort(pa.astype(np.int64) * N + pb.astype(np.int64))   # sorted for searchsorted
    covered_any_hp = np.zeros(total_pairs, dtype=bool)                 # GT pairs a high-prec key catches

    def _longform(values_per_row):
        """(value, row, is_s1) long-form for non-empty values -> two DataFrames (s1 side, target side)."""
        rows = np.repeat(np.arange(N), [len(v) for v in values_per_row])
        vals = np.array([x for v in values_per_row for x in v], dtype=object)
        if len(rows) == 0:
            empty = pd.DataFrame({"v": [], "row": []})
            return empty, empty
        s1m = is_s1[rows]
        df = pd.DataFrame({"v": vals, "row": rows})
        return df[s1m].copy(), df[~s1m].copy()

    def analyse(name, shared_pair_mask, values_per_row):
        cov = float(shared_pair_mask.mean()) if total_pairs else 0.0
        la, lb = _longform(values_per_row)
        prec, rec = float("nan"), 0.0
        if len(la) and len(lb):
            # drop values whose cluster is large on either side (ambiguous -> low precision)
            va = la["v"].value_counts(); vb = lb["v"].value_counts()
            okv = set(va.index[va.values <= BOUND]).intersection(vb.index[vb.values <= BOUND])
            la = la[la["v"].isin(okv)]; lb = lb[lb["v"].isin(okv)]
            if len(la) and len(lb):
                m = la.merge(lb, on="v")                       # all shared (s1_row, target_row) pairs
                mk = m["row_x"].to_numpy(np.int64) * N + m["row_y"].to_numpy(np.int64)
                pos = np.zeros(len(mk), bool)
                idx = np.searchsorted(gt_keys, mk)
                inb = idx < len(gt_keys)
                pos[inb] = gt_keys[np.clip(idx, 0, len(gt_keys) - 1)][inb] == mk[inb]
                tp = int(pos.sum()); fp = int((~pos).sum())
                prec = tp / (tp + fp) if (tp + fp) else float("nan")
                rec = len(np.unique(mk[pos])) / total_pairs if total_pairs else 0.0
                if prec == prec and prec >= HIGH_PREC and tp:
                    covered_any_hp[np.searchsorted(gt_keys, np.unique(mk[pos]))] = True
        print(f"{name:<18}{cov:>9.1%}{prec:>11.3f}{rec:>10.1%}")

    # scalar keys
    for name, arr in scal.items():
        va, vb = arr[pa], arr[pb]
        shared = (va == vb) & (va != "")
        analyse(name, np.asarray(shared), [([v] if v else []) for v in arr])
    # set keys
    for name, lst in setk.items():
        shared = np.array([bool(lst[a] & lst[b]) for a, b in zip(pa, pb)]) if total_pairs else np.zeros(0, bool)
        analyse(name, shared, [list(s) for s in lst])

    print("-" * 66)
    print("coverage = % of true pairs that share the key (a recall ceiling for that key)")
    print(f"rule_prec/rec = precision & recall of 'match iff shared', on clusters <= {BOUND}")

    # ---- deterministic reach (union of high-precision keys) ----
    reach = covered_any_hp.mean() if total_pairs else 0.0
    print("\n" + "=" * 66)
    print("3. DETERMINISTIC REACH")
    print("=" * 66)
    print(f"true pairs recoverable by the UNION of high-precision keys (>= {HIGH_PREC:.0%}): {reach:.1%}")
    print("  -> if this is high, an exact-key stage can capture most matches at high")
    print("     precision, which is very likely how the top scores are reached.")

    # ---- residual hardness ----
    resid_keys = gt_keys[~covered_any_hp]          # keys are indexed in sorted gt_keys order
    print("\n" + "=" * 66)
    print("4. RESIDUAL (true pairs NO high-precision key catches)")
    print("=" * 66)
    if len(resid_keys) == 0:
        print("none -- every true pair is caught by an exact key.")
    else:
        cn = scal["exact_core_name"]
        sample = (resid_keys if len(resid_keys) <= 20000
                  else np.random.default_rng(0).choice(resid_keys, 20000, replace=False))
        def bigrams(s):
            s = s or ""
            return {s[i:i+2] for i in range(len(s)-1)} or {s}
        js = []
        for kkey in sample:
            a, b = cn[int(kkey) // N], cn[int(kkey) % N]
            ba, bb = bigrams(a), bigrams(b)
            u = len(ba | bb); js.append(len(ba & bb)/u if u else 0.0)
        js = np.array(js)
        print(f"residual true pairs     : {len(resid_keys)}  ({100*len(resid_keys)/total_pairs:.1f}% of all true pairs)")
        print(f"name bigram-Jaccard      : mean={js.mean():.2f}  "
              f">=0.6 (near-dup): {100*(js>=0.6).mean():.0f}%   "
              f"<0.3 (divergent): {100*(js<0.3).mean():.0f}%")
        print("  -> high near-dup share = fuzzy name matching can still get these.")
        print("     high divergent share = these need address/semantic signals (embeddings).")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
