"""Vectorised name / address normalisation.

Every operation is a whole-column pandas string op (C-level), so normalising tens
of millions of rows takes seconds-to-minutes instead of the hours a per-row Python
loop would cost. We deliberately store only compact scalar/string columns -- never a
Python set per row -- so memory stays bounded at scale. Token-level features are
recomputed from these strings inside small streaming batches later.

Handles the documented noise: name abbreviations / legal-suffix inconsistencies,
`&`/`and`, punctuation, transliteration (unicode NFKD ascii fold), address
abbreviations, and postal-code / street-number extraction.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------- vocab tables
NAME_ABBREV = {
    "pvt": "private", "ltd": "limited", "corp": "corporation", "inc": "incorporated",
    "co": "company", "intl": "international", "natl": "national", "assn": "association",
    "assoc": "association", "bros": "brothers", "mfg": "manufacturing",
    "dept": "department", "svc": "service", "svcs": "services", "mktg": "marketing",
    "dist": "distributors", "ent": "enterprises", "grp": "group",
}
LEGAL_SUFFIX_TOKENS = [
    "corporation", "incorporated", "company", "limited", "private", "llc", "llp",
    "plc", "gmbh", "ag", "sa", "sas", "srl", "bv", "nv", "pte", "pty", "ltda",
    "corp", "inc", "co", "ltd", "pvt", "kg", "oy", "ab", "spa",
    # French legal suffixes (test set contains France)
    "sarl", "sasu", "eurl", "sci", "scp", "sca", "snc",
]
CONNECTOR_TOKENS = ["and", "the", "of", "for"]

ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "ln": "lane", "dr": "drive", "hwy": "highway", "sq": "square", "apt": "apartment",
    "bldg": "building", "fl": "floor", "flr": "floor", "ste": "suite",
    "opp": "opposite", "nr": "near", "jn": "junction", "jnc": "junction",
    "mkt": "market", "ngr": "nagar", "cly": "colony", "sec": "sector",
    "no": "number",
}

_SUFFIX_RE = r"\b(?:" + "|".join(LEGAL_SUFFIX_TOKENS + CONNECTOR_TOKENS) + r")\b"


# --------------------------------------------------------- scalar helpers (tests)
def _ascii_fold_scalar(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_name(raw: str) -> str:
    s = _ascii_fold_scalar(raw).lower().replace("&", " and ")
    s = re.sub(r"[^0-9a-z]+", " ", s).strip()
    toks = [NAME_ABBREV.get(t, t) for t in s.split()]
    return " ".join(toks)


def core_name(raw: str) -> str:
    full = normalize_name(raw)
    core = re.sub(_SUFFIX_RE, " ", full)
    core = re.sub(r"\s+", " ", core).strip()
    return core if core else full


# ------------------------------------------------------------- vectorised path
def _clean_series(s: pd.Series) -> pd.Series:
    """Fold accents, lowercase, '&'->and, strip punctuation, collapse whitespace."""
    s = s.fillna("").astype("string")
    s = s.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    s = s.str.lower().str.replace("&", " and ", regex=False)
    s = s.str.replace(r"[^0-9a-z]+", " ", regex=True).str.strip()
    return s.str.replace(r"\s+", " ", regex=True)


def _expand(s: pd.Series, table: dict) -> pd.Series:
    for k, v in table.items():
        s = s.str.replace(rf"\b{k}\b", v, regex=True)
    return s


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach compact normalised columns (all vectorised, no per-row Python)."""
    out = pd.DataFrame({"entity_id": df["entity_id"].astype("string")})
    out["country_key"] = _clean_series(df["country"])

    # names -- strip domain/handle artefacts first (@acme, acme.com, www.) so those
    # forms normalise close to the spaced name
    name_raw = df["business_name"].fillna("").astype("string").str.lower()
    name_raw = name_raw.str.replace(r"https?://", " ", regex=True)
    name_raw = name_raw.str.replace(r"www\.", " ", regex=True)
    name_raw = name_raw.str.replace("@", " ", regex=False)
    name_raw = name_raw.str.replace(r"\.(com|in|co|org|net|io|biz|info|us|fr)\b", " ", regex=True)
    norm = _expand(_clean_series(name_raw), NAME_ABBREV)
    core = norm.str.replace(_SUFFIX_RE, " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    core = core.where(core.str.len() > 0, norm)          # fall back if stripped empty
    out["norm_name"] = norm
    out["core_name"] = core
    out["name_key"] = core.where(core.str.len() > 0, norm)
    out["name_squish"] = out["name_key"].str.replace(r"\s+", "", regex=True)
    out["first_token"] = core.str.extract(r"^(\S+)", expand=False).fillna("")
    out["acronym"] = (core.str.replace(r"([a-z0-9])[a-z0-9]*", r"\1", regex=True)
                          .str.replace(r"\s+", "", regex=True).fillna(""))
    out["core_len"] = core.str.len().fillna(0).astype("int32")

    # addresses
    addr_fold = df["business_address"].fillna("").astype("string")
    addr_fold = addr_fold.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii").str.lower()
    out["norm_addr"] = _expand(_clean_series(df["business_address"]), ADDR_ABBREV)
    out["postal"] = addr_fold.str.extract(r"(?s).*?(\d{5,6})(?:\D*$)", expand=False).fillna("")
    out["nums"] = addr_fold.str.replace(r"[^0-9]+", " ", regex=True).str.strip().fillna("")

    # to plain python str dtype for fast object-array indexing during batching
    for c in ["entity_id", "country_key", "norm_name", "core_name", "name_key",
              "name_squish", "first_token", "acronym", "norm_addr", "postal", "nums"]:
        out[c] = out[c].astype(object)
    return out


def postal_series(addr: pd.Series) -> pd.Series:
    a = addr.fillna("").astype("string")
    a = a.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    return a.str.extract(r"(?s).*?(\d{5,6})(?:\D*$)", expand=False).fillna("")
