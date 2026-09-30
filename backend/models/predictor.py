"""
predictor.py – Load trained model and generate matching_results.tsv.

Rules:
  - Only predict entities that appear in candidate_pairs.tsv
  - Every Source-1 test entity must have exactly one row
  - Only S2 and S3 IDs are allowed as matches
  - Allow multiple matches per S1 entity
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.preprocessing.normalizer import normalize_dataframe
from backend.models.features import (
    fit_tfidf, build_feature_matrix, make_lookups, FEATURE_COLS, _tfidf_cache
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
MODEL_DIR         = PROJECT_ROOT / "models"
MODEL_PATH        = MODEL_DIR / "entity_match_model.pkl"
THRESH_PATH       = MODEL_DIR / "threshold.json"
CANDIDATE_PATH    = PROJECT_ROOT / "output" / "candidate_pairs.tsv"
RESULTS_PATH      = PROJECT_ROOT / "output" / "matching_results.tsv"


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

def predict(
    candidate_pairs_path: Optional[Path] = None,
    output_path:          Optional[Path] = None,
    model_path:           Optional[Path] = None,
    threshold_path:       Optional[Path] = None,
    df_s1: Optional[pd.DataFrame] = None,
    df_s2: Optional[pd.DataFrame] = None,
    df_s3: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Generate matching_results.tsv from candidate_pairs.tsv.

    Parameters can be overridden for API usage.
    Returns a summary dict.
    """
    # ---- Resolve paths ---------------------------------------------------
    cp_path    = Path(candidate_pairs_path) if candidate_pairs_path else CANDIDATE_PATH
    out_path   = Path(output_path)          if output_path          else RESULTS_PATH
    mdl_path   = Path(model_path)           if model_path           else MODEL_PATH
    thr_path   = Path(threshold_path)       if threshold_path       else THRESH_PATH

    if not mdl_path.exists():
        raise FileNotFoundError(f"Model not found at {mdl_path}. Train first.")
    if not thr_path.exists():
        raise FileNotFoundError(f"Threshold file not found at {thr_path}. Train first.")
    if not cp_path.exists():
        raise FileNotFoundError(f"Candidate pairs not found at {cp_path}.")

    # ---- Load model + threshold ------------------------------------------
    logger.info("Loading model from %s", mdl_path)
    clf = joblib.load(mdl_path)
    with open(thr_path) as f:
        thr_data  = json.load(f)
    threshold = float(thr_data["threshold"])
    logger.info("Threshold: %.4f", threshold)

    # ---- Load candidate pairs --------------------------------------------
    logger.info("Loading candidate pairs from %s", cp_path)
    cp_df = pd.read_csv(cp_path, sep="\t", dtype=str, keep_default_na=False)
    if "source1_entity_id" not in cp_df.columns:
        raise ValueError("candidate_pairs.tsv must have 'source1_entity_id' column")

    # ---- Load entity data (if not provided) ------------------------------
    if df_s1 is None or df_s2 is None or df_s3 is None:
        df_s1, df_s2, df_s3 = _load_test_data()

    # ---- Normalize -------------------------------------------------------
    logger.info("Normalizing entity data …")
    normalize_dataframe(df_s1)
    normalize_dataframe(df_s2)
    normalize_dataframe(df_s3)

    # ---- Fit TF-IDF (refit on test data if needed) -----------------------
    if _tfidf_cache._name_vec is None:
        logger.info("Fitting TF-IDF on test data …")
        fit_tfidf(df_s1, df_s2, df_s3)

    # ---- Build lookups ---------------------------------------------------
    s1_lookup, cand_lookup = make_lookups(df_s1, df_s2, df_s3)
    valid_cand_ids: Set[str] = set(cand_lookup.keys())

    # ---- Build pairs from candidate_pairs.tsv ----------------------------
    results: Dict[str, List[str]] = {}

    for _, row in cp_df.iterrows():
        s1_id    = str(row["source1_entity_id"]).strip()
        cand_str = str(row.get("candidate_entity_ids", "")).strip()

        if not cand_str or cand_str.lower() in ("nan", "none", ""):
            results[s1_id] = []
            continue

        cand_ids = [c.strip() for c in cand_str.split(",") if c.strip()]
        # Filter only valid candidate IDs
        cand_ids = [c for c in cand_ids if c in valid_cand_ids]

        if not cand_ids:
            results[s1_id] = []
            continue

        # Build feature matrix
        pairs_to_score = [(s1_id, cid) for cid in cand_ids]
        X = build_feature_matrix(pairs_to_score, s1_lookup, cand_lookup)

        # Score
        scores = clf.predict_proba(X[FEATURE_COLS])[:, 1]

        # Apply threshold
        matched = [cid for cid, sc in zip(cand_ids, scores) if sc >= threshold]
        results[s1_id] = matched

    # ---- Ensure all S1 entities from candidate_pairs.tsv have a row -----
    for s1_id in cp_df["source1_entity_id"].tolist():
        if s1_id not in results:
            results[s1_id] = []

    # ---- Write output ----------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for s1_id, matched_ids in results.items():
        out_rows.append({
            "source1_entity_id":  s1_id,
            "matched_entity_ids": ",".join(sorted(matched_ids)),
        })
    out_df = pd.DataFrame(out_rows, columns=["source1_entity_id", "matched_entity_ids"])
    out_df.to_csv(out_path, sep="\t", index=False)
    logger.info("matching_results.tsv written → %s (%d rows)", out_path, len(out_df))

    # ---- Summary ---------------------------------------------------------
    match_count     = sum(len(v) for v in results.values())
    singleton_count = sum(1 for v in results.values() if len(v) == 0)

    return {
        "output_path":    str(out_path),
        "threshold":      threshold,
        "match_count":    match_count,
        "singleton_count": singleton_count,
        "n_entities":     len(results),
    }


def _load_test_data():
    """Load test source files or fall back to synthetic test data."""
    from backend.data.loader import get_test_paths, load_all_sources, get_train_paths
    test_paths = get_test_paths()
    if Path(test_paths["source1"]).exists():
        df_s1, df_s2, df_s3, _ = load_all_sources(
            test_paths["source1"], test_paths["source2"], test_paths["source3"]
        )
        return df_s1, df_s2, df_s3

    # Fall back to train data if test data not present
    logger.warning("Test data not found. Using train data for entity lookup.")
    train_paths = get_train_paths()
    if Path(train_paths["source1"]).exists():
        df_s1, df_s2, df_s3, _ = load_all_sources(
            train_paths["source1"], train_paths["source2"], train_paths["source3"]
        )
        return df_s1, df_s2, df_s3

    # Absolute fallback – use synthetic data that trainer wrote
    synthetic_dir = PROJECT_ROOT / "dataset" / "train"
    df_s1, df_s2, df_s3, _ = load_all_sources(
        synthetic_dir / "train_source1.tsv",
        synthetic_dir / "train_source2.tsv",
        synthetic_dir / "train_source3.tsv",
    )
    return df_s1, df_s2, df_s3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )
    summary = predict()
    print("\nPrediction complete:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
