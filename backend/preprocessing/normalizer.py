"""
normalizer.py – Business name and address normalization.

Applies:
  - Unicode normalization (NFKD → ASCII)
  - lowercase
  - punctuation removal / normalization
  - whitespace normalization
  - '&' → 'and'
  - common legal suffix normalization
  - common business / address abbreviations
  - token set for blocking
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Set

import pandas as pd

# ---------------------------------------------------------------------------
# Legal suffix mappings  (order matters – longer first)
# ---------------------------------------------------------------------------
LEGAL_SUFFIXES: dict[str, str] = {
    r"\bltd\b": "limited",
    r"\bllc\b": "llc",
    r"\bl\.l\.c\b": "llc",
    r"\bllp\b": "llp",
    r"\bl\.l\.p\b": "llp",
    r"\binc\b": "inc",
    r"\bincorporated\b": "inc",
    r"\bcorp\b": "corp",
    r"\bcorporation\b": "corp",
    r"\bco\b": "co",
    r"\bcompany\b": "co",
    r"\bplc\b": "plc",
    r"\bgmbh\b": "gmbh",
    r"\bsarl\b": "sarl",
    r"\bsa\b": "sa",
    r"\bsas\b": "sas",
    r"\bsrl\b": "srl",
    r"\bse\b": "se",
    r"\bag\b": "ag",
    r"\bab\b": "ab",
    r"\bpvt\b": "pvt",
    r"\bprivate\b": "pvt",
    r"\bpty\b": "pty",
    r"\bnv\b": "nv",
    r"\bbv\b": "bv",
    r"\bgroup\b": "grp",
    r"\bgrp\b": "grp",
    r"\bholdings\b": "hldg",
    r"\bholding\b": "hldg",
    r"\bhldg\b": "hldg",
    r"\benterprises\b": "ent",
    r"\benterprise\b": "ent",
    r"\bindustries\b": "ind",
    r"\bindustry\b": "ind",
    r"\bservices\b": "svc",
    r"\bservice\b": "svc",
    r"\bsolutions\b": "sol",
    r"\btechnologies\b": "tech",
    r"\btechnology\b": "tech",
    r"\btech\b": "tech",
    r"\binternational\b": "intl",
    r"\bintl\b": "intl",
    r"\bnational\b": "natl",
    r"\bglobal\b": "global",
    r"\bsystems\b": "sys",
    r"\bsystem\b": "sys",
    r"\bassociates\b": "assoc",
    r"\bassociate\b": "assoc",
    r"\bpartners\b": "ptnr",
    r"\bpartner\b": "ptnr",
    r"\bconsulting\b": "cnslt",
    r"\bconsultants\b": "cnslt",
}

# ---------------------------------------------------------------------------
# Common address abbreviation mappings
# ---------------------------------------------------------------------------
ADDRESS_ABBREV: dict[str, str] = {
    r"\bstreet\b": "st",
    r"\bstr\b": "st",
    r"\bavenue\b": "ave",
    r"\bav\b": "ave",
    r"\bboulevard\b": "blvd",
    r"\bblvd\b": "blvd",
    r"\broad\b": "rd",
    r"\bdrive\b": "dr",
    r"\blane\b": "ln",
    r"\bcourt\b": "ct",
    r"\bplace\b": "pl",
    r"\bsquare\b": "sq",
    r"\bsuite\b": "ste",
    r"\bfloor\b": "fl",
    r"\bnorth\b": "n",
    r"\bsouth\b": "s",
    r"\beast\b": "e",
    r"\bwest\b": "w",
    r"\bnortheast\b": "ne",
    r"\bnorthwest\b": "nw",
    r"\bsoutheast\b": "se",
    r"\bsouthwest\b": "sw",
    r"\bbuilding\b": "bldg",
    r"\bdepartment\b": "dept",
    r"\bparkway\b": "pkwy",
    r"\bhighway\b": "hwy",
    r"\bterrace\b": "ter",
}

# Stop words to exclude from token sets for blocking
STOP_WORDS: Set[str] = {
    "the", "a", "an", "of", "and", "or", "in", "at", "on", "for", "to",
    "by", "with", "de", "la", "le", "les", "du", "des", "et",  # French
    "der", "die", "das", "und",  # German
    "pvt", "ltd", "llc", "inc", "co", "corp",
}


# ---------------------------------------------------------------------------
# Core normalization functions
# ---------------------------------------------------------------------------

def _unicode_normalize(text: str) -> str:
    """Convert to ASCII via NFKD decomposition."""
    nfkd = unicodedata.normalize("NFKD", text)
    return nfkd.encode("ascii", errors="ignore").decode("ascii")


def _remove_punctuation(text: str, keep_numbers: bool = False) -> str:
    """Remove punctuation; optionally keep digits."""
    if keep_numbers:
        # Keep alphanumeric and spaces
        return re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"[^a-z\s]", " ", text)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _apply_mappings(text: str, mappings: dict[str, str]) -> str:
    for pattern, replacement in mappings.items():
        text = re.sub(pattern, replacement, text)
    return text


def normalize_business_name(name: str) -> str:
    """Full pipeline for business name normalization.

    Steps:
      1. Unicode → ASCII
      2. Lowercase
      3. '&' → 'and'
      4. Remove punctuation (keep only alpha + space)
      5. Apply legal suffix abbreviations
      6. Whitespace normalization
    """
    if not isinstance(name, str) or not name.strip():
        return ""

    text = _unicode_normalize(name)
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)          # remove punctuation
    text = re.sub(r"[_]", " ", text)               # underscores → space
    text = _apply_mappings(text, LEGAL_SUFFIXES)
    text = _normalize_whitespace(text)
    return text


def normalize_address(address: str) -> str:
    """Full pipeline for address normalization.

    Steps:
      1. Unicode → ASCII
      2. Lowercase
      3. '&' → 'and'
      4. Remove punctuation (preserve digits – important for street numbers)
      5. Apply address abbreviations
      6. Whitespace normalization
    """
    if not isinstance(address, str) or not address.strip():
        return ""

    text = _unicode_normalize(address)
    text = text.lower()
    text = text.replace("&", " and ")
    # Keep digits for street numbers and zip codes
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"[_]", " ", text)
    text = _apply_mappings(text, ADDRESS_ABBREV)
    text = _normalize_whitespace(text)
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Split normalized text into tokens, optionally removing stop words."""
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens


def get_name_tokens(name: str) -> Set[str]:
    """Return the set of meaningful tokens from a normalized business name."""
    normalized = normalize_business_name(name)
    return set(tokenize(normalized, remove_stopwords=True))


def get_address_tokens(address: str) -> Set[str]:
    """Return the set of meaningful tokens from a normalized address."""
    normalized = normalize_address(address)
    return set(tokenize(normalized, remove_stopwords=False))


def extract_postal_token(address: str) -> str | None:
    """Try to extract a postal-code-like token from an address string.

    Supports common formats:
      - 5-digit US ZIP  (e.g. 94105)
      - UK postcode     (e.g. SW1A 2AA)
      - French CP       (e.g. 75008)
      - Indian PIN      (e.g. 110001)
      - German PLZ      (e.g. 10115)
    Returns the first match or None.
    """
    patterns = [
        r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b",   # UK
        r"\b\d{5,6}\b",                                    # US / France / India / Germany
        r"\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b",                 # Canada
    ]
    addr_upper = address.upper()
    for pat in patterns:
        m = re.search(pat, addr_upper)
        if m:
            return m.group(0).replace(" ", "").lower()
    return None


def normalize_dataframe(df, name_col: str = "business_name",
                         address_col: str = "business_address") -> None:
    """Add ``norm_name`` and ``norm_address`` columns to *df* in-place.

    Handles empty DataFrames gracefully.
    """
    if df.empty or name_col not in df.columns:
        # Add empty columns so downstream code doesn't break
        for col in ("norm_name", "norm_address", "postal_token",
                    "name_tokens", "address_tokens"):
            if col not in df.columns:
                df[col] = pd.Series(dtype=object)
        return

    df["norm_name"] = df[name_col].apply(normalize_business_name)
    df["norm_address"] = df[address_col].apply(normalize_address)
    df["postal_token"] = df[address_col].apply(extract_postal_token)
    df["name_tokens"] = df["norm_name"].apply(
        lambda x: frozenset(tokenize(x, remove_stopwords=True))
    )
    df["address_tokens"] = df["norm_address"].apply(
        lambda x: frozenset(tokenize(x, remove_stopwords=False))
    )
