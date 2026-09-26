#!/usr/bin/env python3
"""Validate matching_results.tsv and candidate_pairs.tsv against the challenge rules.

Stdlib only -- no third-party dependencies. Mirrors the checks described in the
challenge brief so a formatting problem can be caught locally instead of costing a
submission. It reads only the two output files and the test source files; it does
NOT compute your F0.5 score.

Usage:
    python3 utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test

Exit code 0 and "PASS" when safe to submit; exit 1 and a numbered issue list otherwise.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Set, Tuple


def _read_tsv_rows(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _find_source(test_dir: str, kind: str) -> str:
    for name in (f"test_{kind}.tsv", f"{kind}.tsv"):
        p = os.path.join(test_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"could not find test {kind} in {test_dir}")


def _load_ids(path: str) -> List[str]:
    header, rows = _read_tsv_rows(path)
    try:
        idx = [c.strip() for c in header].index("entity_id")
    except ValueError:
        idx = 0
    return [r[idx].strip() for r in rows if r and r[idx].strip()]


def _parse_list(cell: str) -> List[str]:
    return [t.strip() for t in cell.split(",") if t.strip()]


def _validate_file(path: str, list_col: str, valid_s1: Set[str],
                   valid_targets: Set[str], issues: List[str]) -> Dict[str, List[str]]:
    label = os.path.basename(path)
    header, rows = _read_tsv_rows(path)
    expected = ["source1_entity_id", list_col]
    if [c.strip() for c in header] != expected:
        issues.append(f"[{label}] header is {header}, expected {expected}")

    parsed: Dict[str, List[str]] = {}
    seen_s1: Set[str] = set()
    for ln, r in enumerate(rows, start=2):
        if len(r) == 1:      # empty list -> a trailing tab may be dropped by writers
            r = [r[0], ""]
        if len(r) != 2:
            issues.append(f"[{label}] line {ln}: expected 2 tab-separated columns, got {len(r)}")
            continue
        s1, cell = r[0].strip(), r[1]
        if s1 in seen_s1:
            issues.append(f"[{label}] duplicate source1_entity_id row: {s1}")
        seen_s1.add(s1)
        if s1 not in valid_s1:
            issues.append(f"[{label}] line {ln}: {s1} is not a test Source1 id")
        ids = _parse_list(cell)
        if len(ids) != len(set(ids)):
            dups = sorted({x for x in ids if ids.count(x) > 1})
            issues.append(f"[{label}] {s1}: duplicate ids within list: {dups}")
        for x in ids:
            if x.startswith("S1-"):
                issues.append(f"[{label}] {s1}: self-match to Source1 id {x} not allowed")
            elif not (x.startswith("S2-") or x.startswith("S3-")):
                issues.append(f"[{label}] {s1}: id {x} is not an S2-/S3- id")
            elif x not in valid_targets:
                issues.append(f"[{label}] {s1}: id {x} does not exist in the test set")
        parsed[s1] = ids

    missing = valid_s1 - seen_s1
    if missing:
        issues.append(f"[{label}] missing {len(missing)} Source1 entities, e.g. "
                      f"{sorted(missing)[:5]}")
    return parsed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args(argv)

    issues: List[str] = []
    s1_ids = set(_load_ids(_find_source(args.test_dir, "source1")))
    s2_ids = set(_load_ids(_find_source(args.test_dir, "source2")))
    s3_ids = set(_load_ids(_find_source(args.test_dir, "source3")))
    targets = s2_ids | s3_ids

    match = _validate_file(args.matching, "matched_entity_ids", s1_ids, targets, issues)
    cand = _validate_file(args.candidate, "candidate_entity_ids", s1_ids, targets, issues)

    # every match should be a subset of that entity's candidates
    for s1, ids in match.items():
        cset = set(cand.get(s1, []))
        stray = [x for x in ids if x not in cset]
        if stray:
            issues.append(f"[subset] {s1}: matched ids not in candidate set: {stray[:5]}"
                          + (" ..." if len(stray) > 5 else ""))

    if issues:
        print(f"FAIL -- {len(issues)} issue(s):")
        for i, msg in enumerate(issues, 1):
            print(f"  {i}. {msg}")
        return 1
    print("PASS -- both files satisfy all validation rules.")
    print(f"  Source1 entities: {len(s1_ids)} | matched rows: {len(match)} | candidate rows: {len(cand)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
