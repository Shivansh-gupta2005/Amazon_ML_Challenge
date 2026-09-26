#!/usr/bin/env python3
"""Generate synthetic 3-source business data that mimics the challenge's noise.

This is ONLY for exercising / smoke-testing the pipeline end-to-end and getting a
self-measured F0.5 -- it is not the real challenge data and encodes no external
knowledge. It reproduces the documented noise patterns: name abbreviations & legal
suffix swaps, '&' vs 'and', typos, word-order transposition; address abbreviations,
dropped components, landmark references, reordering. Training covers US + India;
the test set additionally introduces France (an unseen country label).

Usage:
    python3 utils/make_synthetic_data.py --out-dir dataset --seed 42
"""
from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Tuple

ROOTS = ["Sunrise", "Global", "Apex", "Ganga", "Himalaya", "Star", "Blue", "Green",
         "Metro", "Prime", "Royal", "Silver", "Golden", "Pioneer", "United",
         "National", "Crescent", "Orchid", "Lotus", "Pearl", "Summit", "Vertex",
         "Nova", "Zenith", "Falcon", "Maple", "Cedar", "Delta", "Omega", "Indus"]

# syllables used to synthesise a large, diverse pool of brand-like tokens so that
# names stay reasonably unique at scale (real business names are highly diverse).
_SYL = ["ka", "ro", "tex", "via", "lon", "mar", "san", "del", "cor", "vin", "tri",
        "bel", "nor", "quo", "zen", "ther", "flux", "pol", "ryn", "vex", "sol",
        "andi", "oro", "esta", "lume", "cris", "dyna", "orbi", "veda", "aster"]


def _make_brands(rng: random.Random, n: int):
    seen, out = set(), []
    while len(out) < n:
        w = "".join(rng.choice(_SYL) for _ in range(rng.choice([2, 2, 3]))).capitalize()
        if w not in seen:
            seen.add(w); out.append(w)
    return out
SECTORS = ["Textiles", "Technologies", "Foods", "Motors", "Traders", "Solutions",
           "Industries", "Pharma", "Logistics", "Steel", "Constructions", "Exports",
           "Chemicals", "Systems", "Electronics", "Retail", "Ventures", "Agro"]
US_SUFFIX = ["Inc", "LLC", "Corp", "Co", "Incorporated"]
IN_SUFFIX = ["Pvt Ltd", "Ltd", "Private Limited"]
FR_SUFFIX = ["SA", "SARL", "SAS"]

US_STREETS = ["Main St", "Oak Ave", "Maple Rd", "Elm Blvd", "Pine Ln", "Cedar Dr",
              "Washington Ave", "Lincoln Blvd", "Park St", "Lake Rd"]
US_CITIES = [("New York", "NY"), ("Austin", "TX"), ("Denver", "CO"),
             ("Seattle", "WA"), ("Miami", "FL"), ("Chicago", "IL")]
IN_STREETS = ["MG Road", "Nehru Nagar", "Gandhi Marg", "Station Road", "Ring Road",
              "Link Road", "Church Street", "Brigade Road", "SV Road", "FC Road"]
IN_CITIES = [("Mumbai", "MH"), ("Bengaluru", "KA"), ("Delhi", "DL"),
             ("Pune", "MH"), ("Chennai", "TN"), ("Hyderabad", "TS")]
FR_STREETS = ["Rue de la Paix", "Avenue Victor Hugo", "Boulevard Saint Germain",
              "Rue du Commerce", "Avenue des Champs"]
FR_CITIES = [("Paris", "IDF"), ("Lyon", "ARA"), ("Marseille", "PAC"), ("Lille", "HDF")]
LANDMARKS = ["Near SBI ATM", "Opp Bus Stand", "Behind City Mall", "Next to Metro Station",
             "Near Central Park", "Opposite Post Office"]

NAME_ABBR = {"Corporation": "Corp", "Incorporated": "Inc", "Company": "Co",
             "Limited": "Ltd", "Private": "Pvt", "and": "&", "International": "Intl"}
ADDR_ABBR = {"Road": "Rd", "Street": "St", "Avenue": "Ave", "Boulevard": "Blvd",
             "Lane": "Ln", "Drive": "Dr", "North": "N", "South": "S"}


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    op = rng.random()
    if op < 0.34:                                  # transpose
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    elif op < 0.67:                                # delete
        return word[:i] + word[i + 1:]
    else:                                          # duplicate
        return word[:i] + word[i] + word[i:]


def _abbr_swap(text: str, table: Dict[str, str], rng: random.Random, p: float) -> str:
    out = []
    for tok in text.split():
        key = tok
        if key in table and rng.random() < p:
            out.append(table[key])
        else:
            # occasional reverse swap (expanded <-> abbreviated)
            rev = {v: k for k, v in table.items()}
            if tok in rev and rng.random() < p * 0.5:
                out.append(rev[tok])
            else:
                out.append(tok)
    return " ".join(out)


def canonical_name(rng: random.Random, suffixes: List[str], brands=None) -> str:
    pool = brands if brands else ROOTS
    n_roots = rng.choice([1, 1, 2])
    roots = rng.sample(pool, n_roots)
    sector = rng.choice(SECTORS)
    connect = " and " if (n_roots == 2 and rng.random() < 0.3) else " "
    base = connect.join(roots) if n_roots == 2 else roots[0]
    suffix = rng.choice(suffixes)
    return f"{base} {sector} {suffix}"


def canonical_address(rng: random.Random, country: str) -> Tuple[str, str]:
    if country == "US":
        street = rng.choice(US_STREETS)
        city, state = rng.choice(US_CITIES)
        pin = f"{rng.randint(10000, 99999)}"
        num = rng.randint(1, 9999)
        return f"{num} {street}, {city}, {state} {pin}", pin
    elif country == "India":
        street = rng.choice(IN_STREETS)
        city, state = rng.choice(IN_CITIES)
        pin = f"{rng.randint(100000, 999999)}"
        num = rng.randint(1, 200)
        return f"{num} {street}, {city}, {state} {pin}", pin
    else:  # France
        street = rng.choice(FR_STREETS)
        city, state = rng.choice(FR_CITIES)
        pin = f"{rng.randint(10000, 99999)}"
        num = rng.randint(1, 300)
        return f"{num} {street}, {city} {pin}", pin


def noisy_name(name: str, rng: random.Random) -> str:
    s = _abbr_swap(name, NAME_ABBR, rng, p=0.6)
    if rng.random() < 0.3:                          # word-order transposition
        toks = s.split()
        if len(toks) >= 3:
            i = rng.randrange(len(toks) - 1)
            toks[i], toks[i + 1] = toks[i + 1], toks[i]
            s = " ".join(toks)
    if rng.random() < 0.4:                           # a typo in one token
        toks = s.split()
        j = rng.randrange(len(toks))
        toks[j] = _typo(toks[j], rng)
        s = " ".join(toks)
    if rng.random() < 0.15:                          # drop legal suffix
        toks = s.split()
        if len(toks) > 2:
            s = " ".join(toks[:-1])
    return s


def match_name(name: str, rng: random.Random, brands=None) -> str:
    """A matching record's name: usually a noisy variant, but sometimes a domain/handle
    form or an outright rebrand (different name, same address) -- the hard real-world
    cases that name-only blocking cannot catch but address blocking can."""
    r = rng.random()
    if r < 0.15:                                   # domain form
        return name.replace(" ", "").lower() + ".com"
    if r < 0.22:                                   # handle form
        return "@" + name.replace(" ", "").lower()
    if r < 0.32:                                   # rebrand: a completely different name
        return canonical_name(rng, ["Ltd"], brands)
    return noisy_name(name, rng)                    # normal noisy variant


def noisy_address(addr: str, pin: str, rng: random.Random) -> str:
    s = _abbr_swap(addr, ADDR_ABBR, rng, p=0.6)
    if rng.random() < 0.35:                          # drop the PIN/ZIP
        s = s.replace(pin, "").strip().rstrip(",").strip()
    if rng.random() < 0.3:                           # drop the state / trailing comp
        parts = [p.strip() for p in s.split(",")]
        if len(parts) > 2:
            parts = parts[:-1]
        s = ", ".join(parts)
    if rng.random() < 0.25:                           # add a landmark reference
        s = f"{rng.choice(LANDMARKS)}, {s}"
    if rng.random() < 0.2:                            # reorder components
        parts = [p.strip() for p in s.split(",") if p.strip()]
        rng.shuffle(parts)
        s = ", ".join(parts)
    return s


def build_split(rng: random.Random, n_entities: int, countries: List[str],
                start_idx: Dict[str, int], brands=None):
    s1_rows, s2_rows, s3_rows = [], [], []
    gt: List[Tuple[str, List[str]]] = []
    c2 = start_idx["S2"]
    c3 = start_idx["S3"]

    for e in range(n_entities):
        country = rng.choice(countries)
        suffixes = {"US": US_SUFFIX, "India": IN_SUFFIX, "France": FR_SUFFIX}[country]
        name = canonical_name(rng, suffixes, brands)
        addr, pin = canonical_address(rng, country)
        s1_id = f"S1-{start_idx['S1'] + e:05d}"
        s1_rows.append((s1_id, name, addr, country))

        matched: List[str] = []
        # number of S2 / S3 matches (0..2); ~25% singletons
        n2 = rng.choices([0, 1, 2], weights=[0.35, 0.5, 0.15])[0]
        n3 = rng.choices([0, 1, 2], weights=[0.45, 0.45, 0.10])[0]
        for _ in range(n2):
            sid = f"S2-{c2:05d}"; c2 += 1
            s2_rows.append((sid, match_name(name, rng, brands), noisy_address(addr, pin, rng), country))
            matched.append(sid)
        for _ in range(n3):
            sid = f"S3-{c3:05d}"; c3 += 1
            s3_rows.append((sid, match_name(name, rng, brands), noisy_address(addr, pin, rng), country))
            matched.append(sid)

        # HARD NEGATIVE: same/near-same name but a DIFFERENT address (a different
        # branch or a different business entirely). Must NOT be matched -- this is the
        # false-merge trap that the precision-heavy F0.5 metric punishes.
        if rng.random() < 0.35:
            other_addr, other_pin = canonical_address(rng, country)
            while other_pin == pin:
                other_addr, other_pin = canonical_address(rng, country)
            if rng.random() < 0.5:
                sid = f"S2-{c2:05d}"; c2 += 1
                s2_rows.append((sid, noisy_name(name, rng),
                                noisy_address(other_addr, other_pin, rng), country))
            else:
                sid = f"S3-{c3:05d}"; c3 += 1
                s3_rows.append((sid, noisy_name(name, rng),
                                noisy_address(other_addr, other_pin, rng), country))
        gt.append((s1_id, matched))

    # distractor (non-matching) records to make blocking non-trivial
    n_distract = int(n_entities * 0.6)
    for _ in range(n_distract):
        country = rng.choice(countries)
        suffixes = {"US": US_SUFFIX, "India": IN_SUFFIX, "France": FR_SUFFIX}[country]
        name = canonical_name(rng, suffixes, brands)
        addr, pin = canonical_address(rng, country)
        if rng.random() < 0.5:
            sid = f"S2-{c2:05d}"; c2 += 1
            s2_rows.append((sid, noisy_name(name, rng), noisy_address(addr, pin, rng), country))
        else:
            sid = f"S3-{c3:05d}"; c3 += 1
            s3_rows.append((sid, noisy_name(name, rng), noisy_address(addr, pin, rng), country))

    return s1_rows, s2_rows, s3_rows, gt, {"S2": c2, "S3": c3}


def _write_source(path: str, rows: List[Tuple[str, str, str, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")


def _write_gt(path: str, gt: List[Tuple[str, List[str]]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for s1, ids in gt:
            fh.write(f"{s1}\t{','.join(ids)}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=300)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    brands = _make_brands(rng, max(4000, args.n_train // 4))

    # TRAIN: US + India
    tr_s1, tr_s2, tr_s3, tr_gt, nxt = build_split(
        rng, args.n_train, ["US", "India"], {"S1": 0, "S2": 0, "S3": 0}, brands)
    _write_source(os.path.join(args.out_dir, "train", "train_source1.tsv"), tr_s1)
    _write_source(os.path.join(args.out_dir, "train", "train_source2.tsv"), tr_s2)
    _write_source(os.path.join(args.out_dir, "train", "train_source3.tsv"), tr_s3)
    _write_gt(os.path.join(args.out_dir, "train", "train_ground_truth.tsv"), tr_gt)

    # TEST: US + India + France (unseen country label)
    te_s1, te_s2, te_s3, te_gt, _ = build_split(
        rng, args.n_test, ["US", "India", "France"],
        {"S1": 100000, "S2": 100000, "S3": 100000}, brands)
    _write_source(os.path.join(args.out_dir, "test", "test_source1.tsv"), te_s1)
    _write_source(os.path.join(args.out_dir, "test", "test_source2.tsv"), te_s2)
    _write_source(os.path.join(args.out_dir, "test", "test_source3.tsv"), te_s3)
    # keep a hidden gt copy so we can self-score the smoke test (not part of a real submission)
    _write_gt(os.path.join(args.out_dir, "test", "_test_ground_truth_hidden.tsv"), te_gt)

    print(f"train: S1={len(tr_s1)} S2={len(tr_s2)} S3={len(tr_s3)}")
    print(f"test : S1={len(te_s1)} S2={len(te_s2)} S3={len(te_s3)} (+France)")
    print(f"written under {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
