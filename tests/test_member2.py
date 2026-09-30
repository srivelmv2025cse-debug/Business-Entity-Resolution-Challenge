"""
test_member2.py – Tests for ML model, F0.5 evaluation, and API endpoints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---- Ensure project root on path ----------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# F0.5 evaluation tests
# ---------------------------------------------------------------------------

from backend.evaluation.f05 import (
    entity_f05,
    macro_f05,
    f05_score,
    find_best_threshold,
    parse_ground_truth,
    parse_predictions,
)


class TestF05Score:
    def test_perfect(self):
        assert f05_score(1.0, 1.0) == pytest.approx(1.0)

    def test_zero_recall(self):
        assert f05_score(1.0, 0.0) == pytest.approx(0.0)

    def test_zero_precision(self):
        assert f05_score(0.0, 1.0) == pytest.approx(0.0)

    def test_both_zero(self):
        assert f05_score(0.0, 0.0) == pytest.approx(0.0)

    def test_precision_favoured(self):
        """F0.5 should favour precision over recall."""
        # High precision, low recall
        f_high_p = f05_score(1.0, 0.5)
        # Low precision, high recall
        f_high_r = f05_score(0.5, 1.0)
        assert f_high_p > f_high_r

    def test_formula(self):
        p, r = 0.8, 0.6
        expected = (1.25 * p * r) / (0.25 * p + r)
        assert f05_score(p, r) == pytest.approx(expected)


class TestEntityF05:
    def test_singleton_correct_abstention(self):
        p, r, f = entity_f05(set(), set())
        assert p == 1.0 and r == 1.0 and f == 1.0

    def test_singleton_false_merge(self):
        p, r, f = entity_f05(set(), {"S2_001"})
        assert p == 0.0 and r == 0.0 and f == 0.0

    def test_perfect_match(self):
        true_ids = {"S2_001", "S3_001"}
        pred_ids = {"S2_001", "S3_001"}
        p, r, f = entity_f05(true_ids, pred_ids)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f == pytest.approx(1.0)

    def test_partial_match(self):
        true_ids = {"S2_001", "S2_002"}
        pred_ids = {"S2_001"}
        p, r, f = entity_f05(true_ids, pred_ids)
        assert p == pytest.approx(1.0)      # 1/1 predicted is correct
        assert r == pytest.approx(0.5)      # 1/2 truth found
        assert f == pytest.approx(f05_score(1.0, 0.5))

    def test_no_prediction_with_truth(self):
        true_ids = {"S2_001"}
        pred_ids = set()
        p, r, f = entity_f05(true_ids, pred_ids)
        assert f == pytest.approx(0.0)

    def test_false_positive(self):
        true_ids = {"S2_001"}
        pred_ids = {"S2_001", "S2_999"}
        p, r, f = entity_f05(true_ids, pred_ids)
        assert p == pytest.approx(0.5)
        assert r == pytest.approx(1.0)


class TestMacroF05:
    def test_all_perfect(self):
        gt   = {"S1_001": {"S2_001"}, "S1_002": {"S2_002"}}
        pred = {"S1_001": {"S2_001"}, "S1_002": {"S2_002"}}
        m = macro_f05(gt, pred)
        assert m["f05"] == pytest.approx(1.0)

    def test_singleton_included(self):
        gt   = {"S1_001": {"S2_001"}, "S1_002": set()}  # S1_002 = singleton
        pred = {"S1_001": {"S2_001"}, "S1_002": set()}  # correct abstention
        m = macro_f05(gt, pred)
        assert m["f05"] == pytest.approx(1.0)

    def test_missing_s1_treated_as_empty_prediction(self):
        gt   = {"S1_001": {"S2_001"}}
        pred = {}   # S1_001 not in predictions
        m = macro_f05(gt, pred)
        assert m["f05"] == pytest.approx(0.0)

    def test_macro_average(self):
        gt = {
            "S1_A": {"S2_001"},
            "S1_B": {"S2_002"},
        }
        pred = {
            "S1_A": {"S2_001"},   # perfect
            "S1_B": set(),        # missed
        }
        m = macro_f05(gt, pred)
        # Entity A: F0.5=1.0; Entity B: F0.5=0.0 → macro = 0.5
        assert m["f05"] == pytest.approx(0.5)


class TestParseHelpers:
    def test_parse_ground_truth(self):
        df = pd.DataFrame({
            "source1_entity_id": ["S1_001", "S1_001", "S1_002"],
            "source2_entity_id": ["S2_001", "S3_001", "S2_002"],
        })
        gt = parse_ground_truth(df)
        assert gt["S1_001"] == {"S2_001", "S3_001"}
        assert gt["S1_002"] == {"S2_002"}

    def test_parse_predictions(self):
        df = pd.DataFrame({
            "source1_entity_id":  ["S1_001", "S1_002"],
            "matched_entity_ids": ["S2_001,S3_001", ""],
        })
        preds = parse_predictions(df)
        assert preds["S1_001"] == {"S2_001", "S3_001"}
        assert preds["S1_002"] == set()


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

from backend.models.features import (
    compute_pair_features,
    FEATURE_COLS,
    build_feature_matrix,
    make_lookups,
)


def _make_row(entity_id, norm_name, norm_address, country, name_tokens=None, address_tokens=None):
    return pd.Series({
        "entity_id":      entity_id,
        "norm_name":      norm_name,
        "norm_address":   norm_address,
        "country":        country,
        "name_tokens":    frozenset(name_tokens or norm_name.split()),
        "address_tokens": frozenset(address_tokens or norm_address.split()),
    })


class TestFeatures:
    def test_exact_name_match(self):
        s1  = _make_row("S1_001", "acme corp", "123 main st", "US")
        s2  = _make_row("S2_001", "acme corp", "123 main st", "US")
        feat = compute_pair_features(s1, s2)
        assert feat["exact_name_match"] == 1.0
        assert feat["exact_address_match"] == 1.0

    def test_different_names(self):
        s1  = _make_row("S1_001", "acme corp", "123 main st", "US")
        s2  = _make_row("S2_099", "xyz ltd",   "999 oak ave", "UK")
        feat = compute_pair_features(s1, s2)
        assert feat["exact_name_match"] == 0.0
        assert feat["country_match"] == 0.0

    def test_country_match(self):
        s1  = _make_row("S1_001", "alpha co", "1 main st", "US")
        s2  = _make_row("S2_001", "beta co",  "2 oak rd",  "US")
        feat = compute_pair_features(s1, s2)
        assert feat["country_match"] == 1.0

    def test_all_feature_keys_present(self):
        s1  = _make_row("S1_001", "acme corp", "123 main st", "US")
        s2  = _make_row("S2_001", "acme corp", "123 main st", "US")
        feat = compute_pair_features(s1, s2)
        for col in FEATURE_COLS:
            assert col in feat, f"Missing feature: {col}"

    def test_build_feature_matrix(self):
        s1_df = pd.DataFrame([{
            "entity_id": "S1_001", "norm_name": "acme corp",
            "norm_address": "123 main st", "country": "US",
            "name_tokens": frozenset(["acme", "corp"]),
            "address_tokens": frozenset(["123", "main", "st"]),
        }])
        s2_df = pd.DataFrame([{
            "entity_id": "S2_001", "norm_name": "acme corporation",
            "norm_address": "123 main street", "country": "US",
            "name_tokens": frozenset(["acme", "corporation"]),
            "address_tokens": frozenset(["123", "main", "street"]),
        }])
        s1_lookup, cand_lookup = make_lookups(s1_df, s2_df, pd.DataFrame())
        X = build_feature_matrix([("S1_001", "S2_001")], s1_lookup, cand_lookup)
        assert list(X.columns) == FEATURE_COLS
        assert len(X) == 1
        assert X["jaccard_name"].iloc[0] > 0.0   # "acme" in common


# ---------------------------------------------------------------------------
# Threshold sweep test
# ---------------------------------------------------------------------------

class TestThresholdSweep:
    def test_find_best_threshold(self):
        """Verify the threshold sweep returns valid values."""
        pair_scores = {
            ("S1_001", "S2_001"): 0.9,
            ("S1_001", "S2_002"): 0.2,
            ("S1_002", "S2_003"): 0.85,
        }
        gt = {
            "S1_001": {"S2_001"},
            "S1_002": {"S2_003"},
        }
        best_thr, best_metrics = find_best_threshold(
            s1_ids=["S1_001", "S1_002"],
            pair_scores=pair_scores,
            ground_truth=gt,
            thresholds=[0.50, 0.75, 0.95],
        )
        assert 0.50 <= best_thr <= 0.95
        assert 0.0 <= best_metrics["f05"] <= 1.0


# ---------------------------------------------------------------------------
# Model exists sanity check
# ---------------------------------------------------------------------------

class TestModelFiles:
    def test_model_file_exists(self):
        model_path = PROJECT_ROOT / "models" / "entity_match_model.pkl"
        assert model_path.exists(), "entity_match_model.pkl must exist after training"

    def test_threshold_file_valid(self):
        thr_path = PROJECT_ROOT / "models" / "threshold.json"
        assert thr_path.exists(), "threshold.json must exist after training"
        with open(thr_path) as f:
            data = json.load(f)
        assert "threshold" in data
        assert 0.0 < data["threshold"] <= 1.0
        assert "val_f05" in data


# ---------------------------------------------------------------------------
# Output file tests
# ---------------------------------------------------------------------------

class TestOutputFiles:
    def test_matching_results_exists(self):
        results_path = PROJECT_ROOT / "output" / "matching_results.tsv"
        assert results_path.exists(), "matching_results.tsv must exist"

    def test_matching_results_schema(self):
        results_path = PROJECT_ROOT / "output" / "matching_results.tsv"
        if not results_path.exists():
            pytest.skip("matching_results.tsv not yet generated")
        df = pd.read_csv(results_path, sep="\t", dtype=str, keep_default_na=False)
        assert "source1_entity_id" in df.columns
        assert "matched_entity_ids" in df.columns

    def test_no_s1_in_matched_ids(self):
        results_path = PROJECT_ROOT / "output" / "matching_results.tsv"
        if not results_path.exists():
            pytest.skip("matching_results.tsv not yet generated")
        df = pd.read_csv(results_path, sep="\t", dtype=str, keep_default_na=False)
        for _, row in df.iterrows():
            s1_id = row["source1_entity_id"]
            matched = str(row["matched_entity_ids"])
            ids = [x.strip() for x in matched.split(",") if x.strip()]
            assert s1_id not in ids, f"Self-match detected: {s1_id}"

    def test_all_matched_ids_are_s2_or_s3(self):
        results_path = PROJECT_ROOT / "output" / "matching_results.tsv"
        if not results_path.exists():
            pytest.skip("matching_results.tsv not yet generated")
        df = pd.read_csv(results_path, sep="\t", dtype=str, keep_default_na=False)
        for _, row in df.iterrows():
            matched = str(row["matched_entity_ids"])
            ids = [x.strip() for x in matched.split(",") if x.strip()]
            for eid in ids:
                assert eid.startswith("S2_") or eid.startswith("S3_"), \
                    f"Invalid match ID: {eid} (must be S2_ or S3_)"
