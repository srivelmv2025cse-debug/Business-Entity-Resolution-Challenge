"""
trainer.py – Train, validate and persist the entity-matching model.

Steps:
  1. Load training data
  2. Normalize (via Member 1's normalizer)
  3. Build training pairs (positives + hard negatives)
  4. Compute features
  5. Train LightGBM (fallback: HistGradientBoosting / RandomForest)
  6. Threshold sweep → pick best macro F0.5
  7. Save model and threshold
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Project root on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.loader           import load_all_sources, get_train_paths
from backend.preprocessing.normalizer import normalize_dataframe
from backend.models.features       import (
    fit_tfidf, build_feature_matrix, make_lookups, FEATURE_COLS
)
from backend.models.training_data  import build_training_pairs
from backend.evaluation.f05        import (
    find_best_threshold, macro_f05, parse_ground_truth, print_metrics
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR   = PROJECT_ROOT / "models"
MODEL_PATH  = MODEL_DIR / "entity_match_model.pkl"
THRESH_PATH = MODEL_DIR / "threshold.json"

THRESHOLDS_TO_TRY = [
    0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.92,
    0.94, 0.95, 0.97,
]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _get_model():
    """Return the best available gradient-boosting classifier."""
    try:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        logger.info("Using LightGBM classifier")
        return clf
    except ImportError:
        pass

    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=4,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        logger.info("Using XGBoost classifier")
        return clf
    except ImportError:
        pass

    from sklearn.ensemble import HistGradientBoostingClassifier
    logger.info("Falling back to HistGradientBoostingClassifier")
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=42,
        class_weight="balanced",
    )


# ---------------------------------------------------------------------------
# Synthetic training data (when official train files are absent)
# ---------------------------------------------------------------------------

def _make_synthetic_train(base_dir: Path) -> None:
    """Create a richer synthetic training set for dev/testing."""
    base_dir.mkdir(parents=True, exist_ok=True)
    s1_rows = [
        ["S1_001", "Acme Corporation",            "123 Main St Springfield IL 62701",    "US"],
        ["S1_002", "TechNova Solutions Ltd",       "456 Innovation Dr Austin TX 78701",   "US"],
        ["S1_003", "Global Freight & Logistics",   "78 Harbor Rd Los Angeles CA 90001",   "US"],
        ["S1_004", "Boulangerie Dupont",           "12 Rue de la Paix 75001 Paris",       "France"],
        ["S1_005", "Müller Enterprises GmbH",      "Berliner Str 5 10115 Berlin",         "Germany"],
        ["S1_006", "Sunrise Technologies Pvt Ltd", "Plot 42 MIDC Pune 411018",            "India"],
        ["S1_007", "Summit Partners LLC",          "789 Wall Street New York NY 10005",   "US"],
        ["S1_008", "NoMatch Entity Corp",          "999 Unknown Ave Nowhere",             "US"],
    ]
    s2_rows = [
        ["S2_001", "ACME Corp.",                  "123 Main Street Springfield Illinois", "US"],
        ["S2_002", "TechNova Solutions",           "456 Innovation Drive Austin Texas",    "US"],
        ["S2_003", "Global Freight and Logistics", "78 Harbor Road Los Angeles CA",        "US"],
        ["S2_004", "Boulangerie Dupont SARL",      "12 rue de la paix 75001 Paris",        "France"],
        ["S2_005", "Muller Enterprises GmbH",      "Berliner Strasse 5 10115 Berlin",      "Germany"],
        ["S2_009", "Other Company Alpha",          "100 Broadway New York NY",             "US"],
        ["S2_010", "Random Firm Inc",              "55 Oak Ave Chicago IL",               "US"],
        ["S2_011", "Pacific Trade Corp",           "900 Harbor View Blvd Los Angeles",     "US"],
    ]
    s3_rows = [
        ["S3_001", "Acme Corporation LLC",         "123 Main St Springfield IL 62701",    "US"],
        ["S3_002", "Technova Sol Ltd",             "456 Innovation Dr Austin TX",          "US"],
        ["S3_006", "Sunrise Tech Pvt Ltd",         "Plot 42 MIDC Pune 411018",             "India"],
        ["S3_007", "Summit Partners",              "789 Wall St New York NY 10005",        "US"],
        ["S3_010", "Other Company Beta",           "200 Park Ave New York NY",             "US"],
        ["S3_011", "Pacific Trading Corporation",  "900 Harbor View Boulevard Los Angeles","US"],
        ["S3_012", "Totally Different Business",   "1 Random Rd Atlanta GA",              "US"],
    ]
    gt_rows = [
        ["S1_001", "S2_001"], ["S1_001", "S3_001"],
        ["S1_002", "S2_002"], ["S1_002", "S3_002"],
        ["S1_003", "S2_003"],
        ["S1_004", "S2_004"],
        ["S1_005", "S2_005"],
        ["S1_006", "S3_006"],
        ["S1_007", "S3_007"],
        # S1_008 → no match (singleton)
    ]
    cols = ["entity_id", "business_name", "business_address", "country"]
    pd.DataFrame(s1_rows, columns=cols).to_csv(base_dir/"train_source1.tsv", sep="\t", index=False)
    pd.DataFrame(s2_rows, columns=cols).to_csv(base_dir/"train_source2.tsv", sep="\t", index=False)
    pd.DataFrame(s3_rows, columns=cols).to_csv(base_dir/"train_source3.tsv", sep="\t", index=False)
    pd.DataFrame(gt_rows, columns=["source1_entity_id","source2_entity_id"]).to_csv(
        base_dir/"train_ground_truth.tsv", sep="\t", index=False
    )
    logger.info("Synthetic training data written to %s", base_dir)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train(
    train_s1_path: Optional[Path] = None,
    train_s2_path: Optional[Path] = None,
    train_s3_path: Optional[Path] = None,
    train_gt_path: Optional[Path] = None,
) -> Dict:
    """
    Full training pipeline.

    Returns dict with: f05, precision, recall, threshold, match_count,
    singleton_count, model_path, threshold_path.
    """
    # ---- Resolve paths ---------------------------------------------------
    if train_s1_path is None:
        default_paths = get_train_paths()
        train_s1_path = default_paths["source1"]
        train_s2_path = default_paths["source2"]
        train_s3_path = default_paths["source3"]
        train_gt_path = default_paths["ground_truth"]

    # Fall back to synthetic data if official data not found
    if not Path(train_s1_path).exists():
        logger.warning("Training data not found. Creating synthetic data …")
        _make_synthetic_train(Path(train_s1_path).parent)

    # ---- 1. Load ---------------------------------------------------------
    logger.info("Loading training data …")
    df_s1, df_s2, df_s3, df_gt = load_all_sources(
        train_s1_path, train_s2_path, train_s3_path, train_gt_path
    )

    # ---- 2. Normalize ----------------------------------------------------
    logger.info("Normalizing …")
    normalize_dataframe(df_s1)
    normalize_dataframe(df_s2)
    normalize_dataframe(df_s3)

    # ---- 3. Fit TF-IDF ---------------------------------------------------
    logger.info("Fitting TF-IDF vectorizers …")
    fit_tfidf(df_s1, df_s2, df_s3)

    # ---- 4. Build lookups ------------------------------------------------
    s1_lookup, cand_lookup = make_lookups(df_s1, df_s2, df_s3)

    # ---- 5. Build training pairs -----------------------------------------
    logger.info("Building training pairs …")
    pairs, labels = build_training_pairs(df_s1, df_s2, df_s3, df_gt)

    if not pairs:
        raise RuntimeError("No training pairs could be built – check ground truth file.")

    # ---- 6. Compute features ---------------------------------------------
    logger.info("Computing features for %d pairs …", len(pairs))
    X = build_feature_matrix(pairs, s1_lookup, cand_lookup)
    y = np.array(labels)

    # ---- 7. Train/val split (group by S1 entity to avoid leakage) --------
    groups = np.array([p[0] for p in pairs])
    unique_groups = np.unique(groups)

    if len(unique_groups) >= 4:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups))
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx],      y[val_idx]
        val_pairs  = [pairs[i] for i in val_idx]
        val_groups = groups[val_idx]
        logger.info("Train: %d pairs, Val: %d pairs", len(train_idx), len(val_idx))
    else:
        # Too few groups for split — train on all, eval on all
        X_train, X_val = X, X
        y_train, y_val = y, y
        val_pairs  = pairs
        val_groups = groups
        logger.warning("Too few S1 entities for split – evaluating on training data")

    # ---- 8. Fit model ----------------------------------------------------
    logger.info("Training model …")
    clf = _get_model()
    clf.fit(X_train[FEATURE_COLS], y_train)

    # ---- 9. Predict scores on validation set -----------------------------
    val_proba = clf.predict_proba(X_val[FEATURE_COLS])[:, 1]

    # Build pair_scores dict
    pair_scores: Dict[tuple, float] = {}
    for (s1_id, cand_id), score in zip(val_pairs, val_proba):
        pair_scores[(s1_id, cand_id)] = float(score)

    # ---- 10. Build ground truth for validation S1 IDs --------------------
    gt_all   = parse_ground_truth(df_gt)
    val_s1_ids = list(set(val_groups))
    gt_val   = {sid: gt_all.get(sid, set()) for sid in val_s1_ids}

    # ---- 11. Threshold sweep ---------------------------------------------
    logger.info("Sweeping thresholds …")
    best_thr, best_metrics = find_best_threshold(
        s1_ids=val_s1_ids,
        pair_scores=pair_scores,
        ground_truth=gt_val,
        thresholds=THRESHOLDS_TO_TRY,
    )
    logger.info(
        "Best threshold: %.2f  →  F0.5=%.4f  P=%.4f  R=%.4f",
        best_thr, best_metrics["f05"], best_metrics["precision"], best_metrics["recall"],
    )

    # ---- 12. Retrain on ALL data with best threshold ---------------------
    logger.info("Retraining on full dataset …")
    clf.fit(X[FEATURE_COLS], y)

    # ---- 13. Save model + threshold --------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    threshold_data = {
        "threshold": best_thr,
        "val_f05":   best_metrics["f05"],
        "val_precision": best_metrics["precision"],
        "val_recall":    best_metrics["recall"],
    }
    with open(THRESH_PATH, "w") as f:
        json.dump(threshold_data, f, indent=2)
    logger.info("Model saved → %s", MODEL_PATH)
    logger.info("Threshold saved → %s", THRESH_PATH)

    # ---- 14. Summary stats -----------------------------------------------
    per      = best_metrics.get("per_entity", {})
    false_merges    = sum(1 for v in per.values()
                          if v["true_count"] == 0 and v["pred_count"] > 0)
    singleton_count = sum(1 for v in per.values() if v["true_count"] == 0)
    match_count     = sum(v["pred_count"] for v in per.values())

    result = {
        "f05":             best_metrics["f05"],
        "precision":       best_metrics["precision"],
        "recall":          best_metrics["recall"],
        "threshold":       best_thr,
        "match_count":     match_count,
        "singleton_count": singleton_count,
        "false_merges":    false_merges,
        "model_path":      str(MODEL_PATH),
        "threshold_path":  str(THRESH_PATH),
        "n_train_pairs":   len(pairs),
        "n_features":      len(FEATURE_COLS),
    }
    return result


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    metrics = train()
    print_metrics(
        {
            "f05":        metrics["f05"],
            "precision":  metrics["precision"],
            "recall":     metrics["recall"],
            "n_entities": metrics.get("singleton_count", 0),
            "per_entity": {},
        },
        threshold=metrics["threshold"],
    )
    print(f"  False merges     : {metrics['false_merges']}")
    print(f"  Singleton count  : {metrics['singleton_count']}")
    print(f"  Match count      : {metrics['match_count']}")
    print(f"\n  Model  -> {metrics['model_path']}")
    print(f"  Thresh -> {metrics['threshold_path']}\n")
