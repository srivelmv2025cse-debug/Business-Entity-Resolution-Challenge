"""
f05.py – F0.5 evaluation for business entity resolution.

F0.5 formula:
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Singleton rule:
    - If true matches = empty AND predicted = empty → F0.5 = 1.0
    - If true matches = empty AND predicted non-empty → F0.5 = 0.0

Final score = macro average of per-entity F0.5.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set
import numpy as np


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def f05_score(precision: float, recall: float) -> float:
    """Compute F0.5 from precision and recall."""
    denom = (0.25 * precision) + recall
    if denom == 0.0:
        return 0.0
    return (1.25 * precision * recall) / denom


def entity_f05(
    true_ids: Set[str],
    pred_ids: Set[str],
) -> tuple[float, float, float]:
    """
    Compute precision, recall and F0.5 for a single Source-1 entity.

    Parameters
    ----------
    true_ids : set of ground-truth matched entity IDs (S2/S3)
    pred_ids : set of predicted matched entity IDs (S2/S3)

    Returns
    -------
    (precision, recall, f05)
    """
    # Singleton rule
    if len(true_ids) == 0:
        if len(pred_ids) == 0:
            return 1.0, 1.0, 1.0   # correct abstention
        else:
            return 0.0, 0.0, 0.0   # false merge

    # Normal case
    tp = len(true_ids & pred_ids)
    precision = tp / len(pred_ids) if pred_ids else 0.0
    recall    = tp / len(true_ids)
    score     = f05_score(precision, recall)
    return precision, recall, score


def macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions:  Dict[str, Set[str]],
) -> Dict[str, float]:
    """
    Compute macro-averaged F0.5 over all Source-1 entities.

    Parameters
    ----------
    ground_truth : {s1_id → set of true matched IDs}
    predictions  : {s1_id → set of predicted matched IDs}

    Returns
    -------
    dict with keys: precision, recall, f05, per_entity
    """
    precisions: List[float] = []
    recalls:    List[float] = []
    f05s:       List[float] = []
    per_entity: Dict[str, Dict] = {}

    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())
        p, r, f = entity_f05(true_ids, pred_ids)
        precisions.append(p)
        recalls.append(r)
        f05s.append(f)
        per_entity[s1_id] = {
            "precision": p,
            "recall":    r,
            "f05":       f,
            "true_count": len(true_ids),
            "pred_count": len(pred_ids),
            "tp": len(true_ids & pred_ids),
        }

    macro_p  = float(np.mean(precisions)) if precisions else 0.0
    macro_r  = float(np.mean(recalls))    if recalls    else 0.0
    macro_f  = float(np.mean(f05s))       if f05s       else 0.0

    return {
        "precision":  macro_p,
        "recall":     macro_r,
        "f05":        macro_f,
        "per_entity": per_entity,
        "n_entities": len(ground_truth),
    }


# ---------------------------------------------------------------------------
# Ground-truth parsing helpers
# ---------------------------------------------------------------------------

def parse_ground_truth(df_gt) -> Dict[str, Set[str]]:
    """
    Convert a ground-truth DataFrame into {s1_id → set of matched IDs}.

    Expects columns: source1_entity_id, source2_entity_id
    (Member 1's loader schema).  Treats empty / NaN as no match.
    """
    gt: Dict[str, Set[str]] = {}
    for _, row in df_gt.iterrows():
        s1_id   = str(row["source1_entity_id"]).strip()
        s2_id   = str(row.get("source2_entity_id", "")).strip()
        if s1_id not in gt:
            gt[s1_id] = set()
        if s2_id and s2_id.lower() not in ("nan", "", "none"):
            gt[s1_id].add(s2_id)
    return gt


def parse_predictions(df_pred) -> Dict[str, Set[str]]:
    """
    Convert a predictions DataFrame into {s1_id → set of predicted IDs}.

    Expects columns: source1_entity_id, matched_entity_ids
    (comma-separated).
    """
    preds: Dict[str, Set[str]] = {}
    for _, row in df_pred.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw   = str(row.get("matched_entity_ids", "")).strip()
        ids: Set[str] = set()
        if raw and raw.lower() not in ("nan", "", "none"):
            ids = {x.strip() for x in raw.split(",") if x.strip()}
        preds[s1_id] = ids
    return preds


# ---------------------------------------------------------------------------
# Threshold search helper
# ---------------------------------------------------------------------------

def find_best_threshold(
    s1_ids: List[str],
    pair_scores: Dict[tuple, float],       # {(s1_id, cand_id): score}
    ground_truth: Dict[str, Set[str]],
    thresholds: Optional[List[float]] = None,
) -> tuple[float, Dict[str, float]]:
    """
    Sweep thresholds and return (best_threshold, metrics_at_best).

    Parameters
    ----------
    s1_ids       : all S1 entity IDs
    pair_scores  : model probability scores per candidate pair
    ground_truth : {s1_id → set of true match IDs}
    thresholds   : list of thresholds to test (default predefined set)
    """
    if thresholds is None:
        thresholds = [
            0.50, 0.55, 0.60, 0.65, 0.70,
            0.75, 0.80, 0.85, 0.90, 0.92,
            0.94, 0.95, 0.97,
        ]

    best_thr  = 0.50
    best_f05  = -1.0
    best_metrics: Dict[str, float] = {}

    for thr in thresholds:
        predictions: Dict[str, Set[str]] = {s1_id: set() for s1_id in s1_ids}
        for (s1_id, cand_id), score in pair_scores.items():
            if score >= thr:
                predictions[s1_id].add(cand_id)

        metrics = macro_f05(ground_truth, predictions)
        if metrics["f05"] > best_f05:
            best_f05     = metrics["f05"]
            best_thr     = thr
            best_metrics = metrics

    return best_thr, best_metrics


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_metrics(metrics: Dict, threshold: Optional[float] = None) -> None:
    """Print macro metrics in a human-readable table."""
    print("\n" + "=" * 52)
    print("  Entity Resolution – Evaluation Results")
    print("=" * 52)
    if threshold is not None:
        print(f"  Threshold          : {threshold:.4f}")
    print(f"  Macro F0.5         : {metrics['f05']:.4f}")
    print(f"  Macro Precision    : {metrics['precision']:.4f}")
    print(f"  Macro Recall       : {metrics['recall']:.4f}")
    print(f"  Total S1 entities  : {metrics.get('n_entities', '?')}")

    per = metrics.get("per_entity", {})
    if per:
        false_merges   = sum(1 for v in per.values()
                             if v["true_count"] == 0 and v["pred_count"] > 0)
        singleton_ok   = sum(1 for v in per.values()
                             if v["true_count"] == 0 and v["pred_count"] == 0)
        singleton_total = false_merges + singleton_ok
        print(f"  Singleton entities : {singleton_total}")
        print(f"  False merges       : {false_merges}")
        match_count = sum(v["pred_count"] for v in per.values())
        print(f"  Total match count  : {match_count}")
    print("=" * 52 + "\n")
