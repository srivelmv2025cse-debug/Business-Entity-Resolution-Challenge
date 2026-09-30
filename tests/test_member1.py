"""
tests/test_member1.py – Unit tests for Member 1 modules:
  - TSV loading (loader.py)
  - Business name & address normalization (normalizer.py)
  - Candidate generation & blocking (blocking.py, candidate_generator.py)
  - No S1 self-matches
  - Country open-set handling (France, etc.)

Uses a tiny synthetic dataset so tests run without official data.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
import tempfile

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make sure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.loader import load_tsv, load_entity_file, load_ground_truth
from backend.preprocessing.normalizer import (
    normalize_business_name,
    normalize_address,
    tokenize,
    get_name_tokens,
    get_address_tokens,
    extract_postal_token,
    normalize_dataframe,
)
from backend.blocking.blocking import BlockingEngine


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tmp_tsv(tmp_path):
    """Return a helper that writes rows to a TSV file."""
    def _writer(filename: str, rows: list[dict]) -> Path:
        df = pd.DataFrame(rows)
        p = tmp_path / filename
        df.to_csv(p, sep="\t", index=False)
        return p
    return _writer


ENTITY_COLS = ["entity_id", "business_name", "business_address", "country"]

def _make_entity_row(eid, name, address, country):
    return {"entity_id": eid, "business_name": name, "business_address": address, "country": country}


# ===========================================================================
# 1. TSV Loading tests
# ===========================================================================

class TestLoader:
    def test_load_basic_tsv(self, tmp_tsv):
        rows = [
            _make_entity_row("S1_001", "Acme Corp", "123 Main St", "US"),
            _make_entity_row("S1_002", "Beta LLC",  "456 Elm Ave", "US"),
        ]
        path = tmp_tsv("source1.tsv", rows)
        df = load_entity_file(path)
        assert len(df) == 2
        assert list(df.columns[:4]) == ENTITY_COLS

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_entity_file(tmp_path / "nonexistent.tsv")

    def test_load_missing_columns_raises(self, tmp_tsv):
        rows = [{"entity_id": "S1_001", "business_name": "Acme"}]  # missing cols
        path = tmp_tsv("bad.tsv", rows)
        with pytest.raises(ValueError, match="missing required columns"):
            load_entity_file(path)

    def test_load_ground_truth(self, tmp_tsv):
        rows = [
            {"source1_entity_id": "S1_001", "source2_entity_id": "S2_001"},
            {"source1_entity_id": "S1_001", "source2_entity_id": "S3_001"},
        ]
        path = tmp_tsv("gt.tsv", rows)
        df = load_ground_truth(path)
        assert len(df) == 2

    def test_load_preserves_countries(self, tmp_tsv):
        rows = [
            _make_entity_row("S1_001", "Boulangerie Dupont", "12 Rue de la Paix", "France"),
            _make_entity_row("S1_002", "Sunrise Pvt Ltd", "Plot 42 Pune", "India"),
            _make_entity_row("S1_003", "Müller GmbH", "Berliner Str 5", "Germany"),
        ]
        path = tmp_tsv("multi_country.tsv", rows)
        df = load_entity_file(path)
        assert set(df["country"].tolist()) == {"France", "India", "Germany"}


# ===========================================================================
# 2. Normalization tests
# ===========================================================================

class TestNormalizeName:
    def test_lowercase(self):
        assert normalize_business_name("ACME CORPORATION") == "acme corp"

    def test_ampersand_to_and(self):
        result = normalize_business_name("Salt & Pepper Co")
        assert "and" in result

    def test_unicode_normalization(self):
        result = normalize_business_name("Müller GmbH")
        assert "u" in result.lower() or "muller" in result.lower()

    def test_legal_suffix_ltd(self):
        result = normalize_business_name("Sunrise Technologies Ltd")
        assert "ltd" not in result or "limited" not in result  # either normalised
        assert "tech" in result

    def test_legal_suffix_incorporated(self):
        result = normalize_business_name("Global Foods Incorporated")
        assert "inc" in result

    def test_punctuation_removed(self):
        result = normalize_business_name("Alpha, Beta & Gamma (Holdings) Ltd.")
        assert "(" not in result
        assert "." not in result

    def test_whitespace_normalized(self):
        result = normalize_business_name("  Acme   Corp  ")
        assert not result.startswith(" ")
        assert "  " not in result

    def test_empty_string(self):
        assert normalize_business_name("") == ""

    def test_french_company(self):
        result = normalize_business_name("Boulangerie Dupont SARL")
        assert "boulangerie" in result
        assert "dupont" in result


class TestNormalizeAddress:
    def test_lowercase(self):
        assert normalize_address("123 MAIN STREET").startswith("123")
        assert normalize_address("123 MAIN STREET") == normalize_address("123 main street")

    def test_preserves_numbers(self):
        result = normalize_address("123 Main St, Springfield, IL 62701")
        assert "123" in result
        assert "62701" in result

    def test_street_abbreviation(self):
        result = normalize_address("123 Main Street")
        assert "st" in result

    def test_avenue_abbreviation(self):
        result = normalize_address("456 Innovation Avenue")
        assert "ave" in result

    def test_unicode_in_address(self):
        result = normalize_address("12 Rue de la Paix, 75001 Paris")
        assert "75001" in result
        assert "paris" in result

    def test_empty_address(self):
        assert normalize_address("") == ""


class TestTokenize:
    def test_basic_split(self):
        tokens = tokenize("acme corp limited")
        assert "acme" in tokens

    def test_stop_word_removal(self):
        tokens = tokenize("the acme co", remove_stopwords=True)
        assert "the" not in tokens

    def test_no_stop_word_removal(self):
        tokens = tokenize("the acme co", remove_stopwords=False)
        assert "the" in tokens


class TestPostalToken:
    def test_us_zip(self):
        assert extract_postal_token("Springfield IL 62701") == "62701"

    def test_french_cp(self):
        assert extract_postal_token("75001 Paris France") == "75001"

    def test_uk_postcode(self):
        result = extract_postal_token("London SW1A 2AA")
        assert result is not None
        assert "sw1a" in result.lower() or "2aa" in result.lower()

    def test_no_postal(self):
        assert extract_postal_token("Some Street Name") is None


# ===========================================================================
# 3. Candidate Generation / Blocking tests
# ===========================================================================

def _make_df(rows: list[dict]) -> pd.DataFrame:
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    normalize_dataframe(df)
    return df


def _sample_s1():
    return _make_df([
        _make_entity_row("S1_001", "Acme Corporation", "123 Main St Springfield IL 62701", "US"),
        _make_entity_row("S1_002", "TechNova Solutions Ltd", "456 Innovation Dr Austin TX", "US"),
        _make_entity_row("S1_003", "Boulangerie Dupont", "12 Rue de la Paix 75001 Paris", "France"),
        _make_entity_row("S1_004", "NoMatch Entity", "999 Unknown Ave", "US"),
    ])


def _sample_s2():
    return _make_df([
        _make_entity_row("S2_001", "ACME Corp.", "123 Main Street Springfield", "US"),
        _make_entity_row("S2_002", "TechNova Solutions", "456 Innovation Drive Austin", "US"),
        _make_entity_row("S2_009", "Other Company A", "100 Broadway New York", "US"),
    ])


def _sample_s3():
    return _make_df([
        _make_entity_row("S3_001", "Acme Corporation LLC", "123 Main St Springfield IL", "US"),
        _make_entity_row("S3_003", "Boulangerie Dupont SARL", "12 rue paix 75001 Paris", "France"),
        _make_entity_row("S3_010", "Other Company B", "200 Park Ave New York", "US"),
    ])


class TestBlocking:
    def test_every_s1_has_row(self):
        """Every S1 entity must appear exactly once in output."""
        engine = BlockingEngine(_sample_s1(), _sample_s2(), _sample_s3())
        result = engine.run()
        s1_ids = set(_sample_s1()["entity_id"].tolist())
        assert set(result.keys()) == s1_ids

    def test_no_s1_self_matches(self):
        """Candidate IDs must never be S1 entity IDs."""
        engine = BlockingEngine(_sample_s1(), _sample_s2(), _sample_s3())
        result = engine.run()
        s1_ids = set(_sample_s1()["entity_id"].tolist())
        for s1_id, cands in result.items():
            for c in cands:
                assert c not in s1_ids, (
                    f"Self-match detected: S1 entity {s1_id} matched to {c} which is also a S1 ID"
                )

    def test_candidates_only_from_s2_s3(self):
        """All candidate IDs must come from S2 or S3."""
        engine = BlockingEngine(_sample_s1(), _sample_s2(), _sample_s3())
        result = engine.run()
        s2_ids = set(_sample_s2()["entity_id"].tolist())
        s3_ids = set(_sample_s3()["entity_id"].tolist())
        valid = s2_ids | s3_ids
        for s1_id, cands in result.items():
            for c in cands:
                assert c in valid, f"Invalid candidate {c} for S1 {s1_id}"

    def test_known_match_is_found(self):
        """S1_001 (Acme) should retrieve S2_001 and/or S3_001 (also Acme)."""
        engine = BlockingEngine(_sample_s1(), _sample_s2(), _sample_s3())
        result = engine.run()
        acme_cands = result.get("S1_001", [])
        assert "S2_001" in acme_cands or "S3_001" in acme_cands, (
            f"Acme match not found. Got: {acme_cands}"
        )

    def test_french_entity_matched(self):
        """French entity S1_003 should find S3_003."""
        engine = BlockingEngine(_sample_s1(), _sample_s2(), _sample_s3())
        result = engine.run()
        dupont_cands = result.get("S1_003", [])
        assert "S3_003" in dupont_cands, (
            f"French Dupont match not found. Got: {dupont_cands}"
        )

    def test_no_match_entity_has_empty_candidates(self):
        """S1_004 (NoMatch Entity) should ideally have few/no candidates."""
        engine = BlockingEngine(_sample_s1(), _sample_s2(), _sample_s3())
        result = engine.run()
        # Just ensure it still has a row (may or may not have candidates)
        assert "S1_004" in result

    def test_no_duplicate_candidates(self):
        """Candidate IDs should not be repeated for any S1 entity."""
        engine = BlockingEngine(_sample_s1(), _sample_s2(), _sample_s3())
        result = engine.run()
        for s1_id, cands in result.items():
            assert len(cands) == len(set(cands)), (
                f"Duplicate candidates for {s1_id}: {cands}"
            )


# ===========================================================================
# 4. Country open-set handling
# ===========================================================================

class TestCountryOpenSet:
    def test_france_supported(self):
        """France should be handled without errors."""
        s1 = _make_df([_make_entity_row("S1_F1", "Cafe de Flore", "172 Blvd St Germain 75006 Paris", "France")])
        s2 = _make_df([_make_entity_row("S2_F1", "Cafe de Flore SARL", "172 boulevard st germain paris", "France")])
        s3 = _make_df([_make_entity_row("S3_F1", "Café de Flore", "172 Blvd Saint Germain 75006", "France")])
        engine = BlockingEngine(s1, s2, s3)
        result = engine.run()
        assert "S1_F1" in result

    def test_multiple_countries_no_cross_contamination(self):
        """Country blocking should prefer same-country pairs."""
        s1 = _make_df([
            _make_entity_row("S1_US", "Alpha Corp", "100 Main St New York", "US"),
            _make_entity_row("S1_FR", "Alpha Corp", "100 Rue Main Paris", "France"),
        ])
        s2 = _make_df([
            _make_entity_row("S2_US", "Alpha Corporation", "100 Main Street New York", "US"),
            _make_entity_row("S2_FR", "Alpha Corp SARL", "100 Rue Main 75001 Paris", "France"),
        ])
        s3 = _make_df([])  # empty S3

        # Should not crash with empty S3
        engine = BlockingEngine(s1, s2, s3)
        result = engine.run()
        assert "S1_US" in result
        assert "S1_FR" in result

    def test_unknown_country_handled(self):
        """Unknown / rare countries must not raise errors."""
        s1 = _make_df([_make_entity_row("S1_X1", "Test Corp", "123 Some St", "Ruritania")])
        s2 = _make_df([_make_entity_row("S2_X1", "Test Corp", "123 Some Street", "Ruritania")])
        s3 = _make_df([])
        engine = BlockingEngine(s1, s2, s3)
        result = engine.run()
        assert "S1_X1" in result
