"""
features.py – Feature engineering for entity pair matching.

For every (S1 entity, candidate S2/S3 entity) pair, computes:

NAME features:
  - exact_name_match        : normalized names are identical
  - rapidfuzz_ratio         : RapidFuzz token_sort_ratio
  - levenshtein_sim         : 1 - (edit_dist / max_len)
  - jaccard_name            : token Jaccard on name tokens
  - tfidf_cosine_name       : TF-IDF cosine on normalized names
  - token_overlap_name      : |intersection| / |union| of name tokens
  - name_len_diff           : |len(s1_name) - len(s2_name)| / max_len

ADDRESS features:
  - exact_address_match     : normalized addresses are identical
  - jaccard_address         : token Jaccard on address tokens
  - tfidf_cosine_address    : TF-IDF cosine on normalized addresses
  - token_overlap_address   : |intersection| / |union| of address tokens
  - numeric_token_overlap   : overlap of digit-only tokens
  - address_sim             : RapidFuzz ratio on normalized addresses

OTHER features:
  - country_match           : country strings are equal
  - common_token_count      : |name_tokens ∩ candidate_name_tokens|
  - number_overlap          : overlap of all numeric substrings
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Feature column names (order matters – matches model training)
FEATURE_COLS = [
    # Name
    "exact_name_match",
    "rapidfuzz_ratio",
    "levenshtein_sim",
    "jaccard_name",
    "tfidf_cosine_name",
    "token_overlap_name",
    "name_len_diff",
    # Address
    "exact_address_match",
    "jaccard_address",
    "tfidf_cosine_address",
    "token_overlap_address",
    "numeric_token_overlap",
    "address_sim",
    # Other
    "country_match",
    "common_token_count",
    "number_overlap",
]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val).strip()


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _token_overlap(a: Set[str], b: Set[str]) -> float:
    """|intersection| / |union| — same as Jaccard, alias kept for clarity."""
    return _jaccard(a, b)


def _levenshtein_sim(s1: str, s2: str) -> float:
    """Normalised Levenshtein similarity using rapidfuzz."""
    if not s1 and not s2:
        return 1.0
    ratio = fuzz.ratio(s1, s2) / 100.0  # rapidfuzz ratio = Levenshtein-based
    return ratio


def _numeric_tokens(text: str) -> Set[str]:
    return set(re.findall(r"\b\d+\b", text))


def _len_diff(s1: str, s2: str) -> float:
    max_len = max(len(s1), len(s2), 1)
    return abs(len(s1) - len(s2)) / max_len


# ---------------------------------------------------------------------------
# TF-IDF vectorizer (fitted on demand, cached)
# ---------------------------------------------------------------------------

class _TfidfCache:
    """Lazy TF-IDF vectorizers for name and address columns."""

    def __init__(self) -> None:
        self._name_vec:    Optional[TfidfVectorizer] = None
        self._address_vec: Optional[TfidfVectorizer] = None

    def fit(
        self,
        name_corpus: List[str],
        address_corpus: List[str],
    ) -> None:
        self._name_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), min_df=1
        )
        self._name_vec.fit(name_corpus)

        self._address_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), min_df=1
        )
        self._address_vec.fit(address_corpus)

    def name_cosine(self, a: str, b: str) -> float:
        if self._name_vec is None:
            return 0.0
        v = self._name_vec.transform([a, b])
        return float(cosine_similarity(v[0], v[1])[0, 0])

    def address_cosine(self, a: str, b: str) -> float:
        if self._address_vec is None:
            return 0.0
        v = self._address_vec.transform([a, b])
        return float(cosine_similarity(v[0], v[1])[0, 0])


# Module-level cache instance
_tfidf_cache = _TfidfCache()


def fit_tfidf(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
) -> None:
    """Fit the global TF-IDF vectorizers on all available data."""
    name_corpus    = []
    address_corpus = []
    for df in (df_s1, df_s2, df_s3):
        if df is None or df.empty:
            continue
        name_corpus    += df["norm_name"].fillna("").tolist()
        address_corpus += df["norm_address"].fillna("").tolist()
    _tfidf_cache.fit(name_corpus, address_corpus)


# ---------------------------------------------------------------------------
# Per-pair feature computation
# ---------------------------------------------------------------------------

def compute_pair_features(
    s1_row: pd.Series,
    cand_row: pd.Series,
) -> Dict[str, float]:
    """
    Compute all features for a single (S1, candidate) pair.

    Both rows must have: norm_name, norm_address, name_tokens,
    address_tokens, country  (added by normalizer.normalize_dataframe).
    """
    # ---- Retrieve and normalise field values ----------------------------
    s1_name    = _safe_str(s1_row.get("norm_name", ""))
    cand_name  = _safe_str(cand_row.get("norm_name", ""))
    s1_addr    = _safe_str(s1_row.get("norm_address", ""))
    cand_addr  = _safe_str(cand_row.get("norm_address", ""))
    s1_ctry    = _safe_str(s1_row.get("country", ""))
    cand_ctry  = _safe_str(cand_row.get("country", ""))

    s1_ntok    = set(s1_row.get("name_tokens", frozenset()))
    cand_ntok  = set(cand_row.get("name_tokens", frozenset()))
    s1_atok    = set(s1_row.get("address_tokens", frozenset()))
    cand_atok  = set(cand_row.get("address_tokens", frozenset()))

    # ---- NAME features -------------------------------------------------
    exact_name      = float(s1_name == cand_name and s1_name != "")
    rf_ratio        = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    lev_sim         = _levenshtein_sim(s1_name, cand_name)
    jaccard_name    = _jaccard(s1_ntok, cand_ntok)
    tfidf_name      = _tfidf_cache.name_cosine(s1_name, cand_name)
    tok_ovlp_name   = _token_overlap(s1_ntok, cand_ntok)
    len_diff        = _len_diff(s1_name, cand_name)

    # ---- ADDRESS features ----------------------------------------------
    exact_addr      = float(s1_addr == cand_addr and s1_addr != "")
    jaccard_addr    = _jaccard(s1_atok, cand_atok)
    tfidf_addr      = _tfidf_cache.address_cosine(s1_addr, cand_addr)
    tok_ovlp_addr   = _token_overlap(s1_atok, cand_atok)
    num_s1          = _numeric_tokens(s1_addr)
    num_cand        = _numeric_tokens(cand_addr)
    num_ovlp        = _jaccard(num_s1, num_cand)
    addr_sim        = fuzz.ratio(s1_addr, cand_addr) / 100.0

    # ---- OTHER features ------------------------------------------------
    ctry_match      = float(s1_ctry.lower() == cand_ctry.lower() and s1_ctry != "")
    common_toks     = float(len(s1_ntok & cand_ntok))
    num_all_s1      = _numeric_tokens(s1_name + " " + s1_addr)
    num_all_cand    = _numeric_tokens(cand_name + " " + cand_addr)
    number_ovlp     = _jaccard(num_all_s1, num_all_cand)

    return {
        "exact_name_match":      exact_name,
        "rapidfuzz_ratio":       rf_ratio,
        "levenshtein_sim":       lev_sim,
        "jaccard_name":          jaccard_name,
        "tfidf_cosine_name":     tfidf_name,
        "token_overlap_name":    tok_ovlp_name,
        "name_len_diff":         len_diff,
        "exact_address_match":   exact_addr,
        "jaccard_address":       jaccard_addr,
        "tfidf_cosine_address":  tfidf_addr,
        "token_overlap_address": tok_ovlp_addr,
        "numeric_token_overlap": num_ovlp,
        "address_sim":           addr_sim,
        "country_match":         ctry_match,
        "common_token_count":    common_toks,
        "number_overlap":        number_ovlp,
    }


# ---------------------------------------------------------------------------
# Batch feature computation
# ---------------------------------------------------------------------------

def build_feature_matrix(
    pairs: List[Tuple[str, str]],        # [(s1_id, cand_id), ...]
    s1_lookup:   Dict[str, pd.Series],   # entity_id → row
    cand_lookup: Dict[str, pd.Series],   # entity_id → row (S2 + S3 combined)
) -> pd.DataFrame:
    """
    Build a feature DataFrame for a list of (s1_id, cand_id) pairs.

    Returns a DataFrame with columns = FEATURE_COLS, index = range(len(pairs)).
    Missing entities produce all-zero rows.
    """
    records = []
    for s1_id, cand_id in pairs:
        s1_row   = s1_lookup.get(s1_id)
        cand_row = cand_lookup.get(cand_id)

        if s1_row is None or cand_row is None:
            records.append({col: 0.0 for col in FEATURE_COLS})
            continue

        feat = compute_pair_features(s1_row, cand_row)
        records.append(feat)

    df = pd.DataFrame(records, columns=FEATURE_COLS)
    # Fill any remaining NaN with 0
    df = df.fillna(0.0)
    return df


def make_lookups(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    """Build entity_id → row lookups for fast access."""
    s1_lookup: Dict[str, pd.Series] = {
        row["entity_id"]: row for _, row in df_s1.iterrows()
    }
    cand_lookup: Dict[str, pd.Series] = {}
    for df in (df_s2, df_s3):
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                cand_lookup[row["entity_id"]] = row
    return s1_lookup, cand_lookup
