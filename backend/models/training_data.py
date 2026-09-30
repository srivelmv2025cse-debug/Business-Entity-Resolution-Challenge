"""
training_data.py – Build positive + hard-negative training examples.

Positive examples:  pairs from ground truth
Hard negatives:
  1. Same-name-token candidates that are NOT true matches
  2. Same-country candidates with shared name tokens
  3. Address-token overlapping candidates that are NOT true matches
  4. Similar business words but different locations
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def build_training_pairs(
    df_s1:        pd.DataFrame,
    df_s2:        pd.DataFrame,
    df_s3:        pd.DataFrame,
    df_gt:        pd.DataFrame,
    neg_ratio:    int = 4,
    max_per_s1:   int = 30,
    random_seed:  int = 42,
) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    Generate (s1_id, cand_id) pairs and binary labels.

    Parameters
    ----------
    df_s1, df_s2, df_s3 : preprocessed source DataFrames
    df_gt               : ground-truth DataFrame
    neg_ratio           : hard negatives per positive pair
    max_per_s1          : cap on negative examples per S1 entity
    random_seed         : reproducibility

    Returns
    -------
    pairs  : list of (s1_id, cand_id) tuples
    labels : parallel list of 0/1 labels
    """
    random.seed(random_seed)

    # ---------------------------------------------------------------
    # Build indexes
    # ---------------------------------------------------------------
    # Ground truth: s1_id → set of true match IDs
    gt_map: Dict[str, Set[str]] = defaultdict(set)
    for _, row in df_gt.iterrows():
        s1_id  = str(row["source1_entity_id"]).strip()
        s2_id  = str(row.get("source2_entity_id", "")).strip()
        if s2_id and s2_id.lower() not in ("nan", "", "none"):
            gt_map[s1_id].add(s2_id)

    # Combined candidate pool: entity_id → row for S2 + S3
    cand_rows: Dict[str, pd.Series] = {}
    for df in (df_s2, df_s3):
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                cand_rows[row["entity_id"]] = row

    # Inverted index: name_token → list[cand_id]
    token_to_cand: Dict[str, List[str]] = defaultdict(list)
    for cid, crow in cand_rows.items():
        for tok in crow.get("name_tokens", frozenset()):
            token_to_cand[tok].append(cid)

    # Country → list[cand_id]
    country_to_cand: Dict[str, List[str]] = defaultdict(list)
    for cid, crow in cand_rows.items():
        ctry = str(crow.get("country", "")).strip().lower()
        if ctry:
            country_to_cand[ctry].append(cid)

    # Address token → list[cand_id]
    addr_tok_to_cand: Dict[str, List[str]] = defaultdict(list)
    for cid, crow in cand_rows.items():
        for tok in crow.get("address_tokens", frozenset()):
            if len(tok) > 2:
                addr_tok_to_cand[tok].append(cid)

    # ---------------------------------------------------------------
    # Generate pairs
    # ---------------------------------------------------------------
    pairs:  List[Tuple[str, str]] = []
    labels: List[int]             = []

    for _, s1_row in df_s1.iterrows():
        s1_id      = s1_row["entity_id"]
        true_ids   = gt_map.get(s1_id, set())
        s1_country = str(s1_row.get("country", "")).strip().lower()
        s1_ntoks   = set(s1_row.get("name_tokens", frozenset()))
        s1_atoks   = set(s1_row.get("address_tokens", frozenset()))

        # ---- Positives ----
        for tid in true_ids:
            if tid in cand_rows:
                pairs.append((s1_id, tid))
                labels.append(1)

        # ---- Hard negative pool ----
        neg_pool: Set[str] = set()

        # Strategy 1: Same name token but not a true match
        for tok in s1_ntoks:
            for cid in token_to_cand.get(tok, []):
                if cid not in true_ids:
                    neg_pool.add(cid)

        # Strategy 2: Same country
        for cid in country_to_cand.get(s1_country, []):
            if cid not in true_ids:
                neg_pool.add(cid)

        # Strategy 3: Address token overlap
        for tok in s1_atoks:
            if len(tok) > 2:
                for cid in addr_tok_to_cand.get(tok, []):
                    if cid not in true_ids:
                        neg_pool.add(cid)

        # Strategy 4: Random negatives (diversify the pool)
        all_cand_ids = list(cand_rows.keys())
        random_negs = random.sample(
            all_cand_ids,
            min(len(all_cand_ids), max(5, neg_ratio * max(len(true_ids), 1)))
        )
        for cid in random_negs:
            if cid not in true_ids:
                neg_pool.add(cid)

        # Sample negatives (cap to avoid extreme imbalance)
        n_pos  = max(len(true_ids), 1)
        n_neg  = min(len(neg_pool), n_pos * neg_ratio, max_per_s1)

        neg_sample = random.sample(sorted(neg_pool), n_neg)
        for cid in neg_sample:
            pairs.append((s1_id, cid))
            labels.append(0)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    logger.info(
        "Training pairs built: %d total  (pos=%d, neg=%d, ratio=1:%.1f)",
        len(pairs), n_pos, n_neg, n_neg / max(n_pos, 1),
    )
    return pairs, labels
