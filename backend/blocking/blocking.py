"""
blocking.py – Multi-strategy candidate blocking.

Strategies used (in order, each adds to the candidate pool):
  1. Exact normalized name match
  2. Name token overlap (≥ 1 common token)
  3. Country + name-token overlap
  4. Address token overlap
  5. Postal-code token match
  6. TF-IDF cosine similarity on normalized names

Output contract:
  - Only S1 → S2 and S1 → S3 pairs
  - NEVER S1 → S1
  - Every S1 entity has exactly one row in output
  - Candidate IDs are de-duplicated
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Minimum token-overlap count to include a candidate pair
MIN_TOKEN_OVERLAP = 1
# TF-IDF cosine threshold  (lower → more recall, higher → more precision)
TFIDF_THRESHOLD = 0.25
# Maximum candidates per S1 entity (cap to keep downstream tractable)
MAX_CANDIDATES = 200


# ---------------------------------------------------------------------------
# Blocking engine
# ---------------------------------------------------------------------------

class BlockingEngine:
    """Multi-strategy blocking to generate S1→S2 and S1→S3 candidate pairs.

    Parameters
    ----------
    df_s1, df_s2, df_s3:
        DataFrames **already preprocessed** by
        ``normalizer.normalize_dataframe()``.  Must have columns:
        ``entity_id``, ``norm_name``, ``norm_address``,
        ``name_tokens``, ``address_tokens``, ``country``, ``postal_token``.
    """

    def __init__(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame,
    ) -> None:
        self.df_s1 = df_s1.reset_index(drop=True)
        self.df_s2 = df_s2.reset_index(drop=True)
        self.df_s3 = df_s3.reset_index(drop=True)

        # candidates[s1_id] = set of candidate entity IDs (s2 or s3 only)
        self._candidates: Dict[str, Set[str]] = defaultdict(set)

        # Build indexes for fast lookup
        self._s2_ids: Set[str] = set(df_s2["entity_id"].tolist())
        self._s3_ids: Set[str] = set(df_s3["entity_id"].tolist())

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, List[str]]:
        """Execute all blocking strategies and return candidate dict."""
        logger.info("=== Running blocking strategies ===")

        self._strategy_exact_name()
        self._strategy_name_token_overlap()
        self._strategy_country_name_token()
        self._strategy_address_token_overlap()
        self._strategy_postal_token()
        self._strategy_tfidf()

        result = self._finalize()
        logger.info(
            "Blocking done. Total S1 entities: %d, total candidate pairs: %d",
            len(result),
            sum(len(v) for v in result.values()),
        )
        return result

    # ------------------------------------------------------------------
    # Internal strategies
    # ------------------------------------------------------------------

    def _add_pairs(self, s1_id: str, candidate_ids: List[str], source: str) -> int:
        """Add valid S2/S3 candidates, rejecting any S1 IDs."""
        added = 0
        for cid in candidate_ids:
            if cid in self._s2_ids or cid in self._s3_ids:
                self._candidates[s1_id].add(cid)
                added += 1
        return added

    # ------------------------------------------------------------------
    # Strategy 1 – Exact normalized name
    # ------------------------------------------------------------------
    def _strategy_exact_name(self) -> None:
        logger.info("  [1/6] Exact normalized name match")

        # Build lookup: norm_name → list[entity_id]  for S2 and S3
        s2_name_idx: Dict[str, List[str]] = defaultdict(list)
        s3_name_idx: Dict[str, List[str]] = defaultdict(list)

        for _, row in self.df_s2.iterrows():
            if row["norm_name"]:
                s2_name_idx[row["norm_name"]].append(row["entity_id"])
        for _, row in self.df_s3.iterrows():
            if row["norm_name"]:
                s3_name_idx[row["norm_name"]].append(row["entity_id"])

        count = 0
        for _, row in self.df_s1.iterrows():
            s1_id = row["entity_id"]
            nn = row["norm_name"]
            if nn:
                count += self._add_pairs(
                    s1_id, s2_name_idx.get(nn, []) + s3_name_idx.get(nn, []), "exact"
                )
        logger.info("    → %d pairs added", count)

    # ------------------------------------------------------------------
    # Strategy 2 – Name token overlap (≥ 1 shared token)
    # ------------------------------------------------------------------
    def _strategy_name_token_overlap(self) -> None:
        logger.info("  [2/6] Name token overlap (≥%d tokens)", MIN_TOKEN_OVERLAP)

        # Inverted index: token → list[entity_id] for S2 and S3
        token_to_s2: Dict[str, List[str]] = defaultdict(list)
        token_to_s3: Dict[str, List[str]] = defaultdict(list)

        for _, row in self.df_s2.iterrows():
            for tok in row["name_tokens"]:
                token_to_s2[tok].append(row["entity_id"])
        for _, row in self.df_s3.iterrows():
            for tok in row["name_tokens"]:
                token_to_s3[tok].append(row["entity_id"])

        count = 0
        for _, row in self.df_s1.iterrows():
            s1_id = row["entity_id"]
            for tok in row["name_tokens"]:
                count += self._add_pairs(
                    s1_id,
                    token_to_s2.get(tok, []) + token_to_s3.get(tok, []),
                    "token",
                )
        logger.info("    → cumulative unique pairs (approx): %d",
                    sum(len(v) for v in self._candidates.values()))

    # ------------------------------------------------------------------
    # Strategy 3 – Country + name token
    # ------------------------------------------------------------------
    def _strategy_country_name_token(self) -> None:
        logger.info("  [3/6] Country + name token overlap")

        # (country, token) → list[entity_id]
        ct_to_s2: Dict[tuple, List[str]] = defaultdict(list)
        ct_to_s3: Dict[tuple, List[str]] = defaultdict(list)

        for _, row in self.df_s2.iterrows():
            c = row.get("country", "")
            for tok in row["name_tokens"]:
                ct_to_s2[(c, tok)].append(row["entity_id"])
        for _, row in self.df_s3.iterrows():
            c = row.get("country", "")
            for tok in row["name_tokens"]:
                ct_to_s3[(c, tok)].append(row["entity_id"])

        count = 0
        for _, row in self.df_s1.iterrows():
            s1_id = row["entity_id"]
            c = row.get("country", "")
            for tok in row["name_tokens"]:
                count += self._add_pairs(
                    s1_id,
                    ct_to_s2.get((c, tok), []) + ct_to_s3.get((c, tok), []),
                    "country_token",
                )
        logger.info("    → cumulative unique pairs (approx): %d",
                    sum(len(v) for v in self._candidates.values()))

    # ------------------------------------------------------------------
    # Strategy 4 – Address token overlap
    # ------------------------------------------------------------------
    def _strategy_address_token_overlap(self) -> None:
        logger.info("  [4/6] Address token overlap")

        # Inverted index on address tokens
        addr_to_s2: Dict[str, List[str]] = defaultdict(list)
        addr_to_s3: Dict[str, List[str]] = defaultdict(list)

        for _, row in self.df_s2.iterrows():
            for tok in row["address_tokens"]:
                if len(tok) > 2:  # skip very short tokens
                    addr_to_s2[tok].append(row["entity_id"])
        for _, row in self.df_s3.iterrows():
            for tok in row["address_tokens"]:
                if len(tok) > 2:
                    addr_to_s3[tok].append(row["entity_id"])

        count = 0
        for _, row in self.df_s1.iterrows():
            s1_id = row["entity_id"]
            for tok in row["address_tokens"]:
                if len(tok) > 2:
                    count += self._add_pairs(
                        s1_id,
                        addr_to_s2.get(tok, []) + addr_to_s3.get(tok, []),
                        "address",
                    )
        logger.info("    → cumulative unique pairs (approx): %d",
                    sum(len(v) for v in self._candidates.values()))

    # ------------------------------------------------------------------
    # Strategy 5 – Postal-code token
    # ------------------------------------------------------------------
    def _strategy_postal_token(self) -> None:
        logger.info("  [5/6] Postal-code token match")

        postal_to_s2: Dict[str, List[str]] = defaultdict(list)
        postal_to_s3: Dict[str, List[str]] = defaultdict(list)

        for _, row in self.df_s2.iterrows():
            pt = row.get("postal_token")
            if pt:
                postal_to_s2[pt].append(row["entity_id"])
        for _, row in self.df_s3.iterrows():
            pt = row.get("postal_token")
            if pt:
                postal_to_s3[pt].append(row["entity_id"])

        count = 0
        for _, row in self.df_s1.iterrows():
            s1_id = row["entity_id"]
            pt = row.get("postal_token")
            if pt:
                count += self._add_pairs(
                    s1_id,
                    postal_to_s2.get(pt, []) + postal_to_s3.get(pt, []),
                    "postal",
                )
        logger.info("    → cumulative unique pairs (approx): %d",
                    sum(len(v) for v in self._candidates.values()))

    # ------------------------------------------------------------------
    # Strategy 6 – TF-IDF cosine similarity
    # ------------------------------------------------------------------
    def _strategy_tfidf(self) -> None:
        logger.info(
            "  [6/6] TF-IDF cosine similarity (threshold=%.2f)", TFIDF_THRESHOLD
        )

        s1_names = self.df_s1["norm_name"].fillna("").tolist()
        s2_names = self.df_s2["norm_name"].fillna("").tolist() if not self.df_s2.empty else []
        s3_names = self.df_s3["norm_name"].fillna("").tolist() if not self.df_s3.empty else []

        all_names = s1_names + s2_names + s3_names

        if not any(n.strip() for n in all_names) or not s1_names:
            logger.warning("    All names empty – skipping TF-IDF")
            return

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        try:
            vectorizer.fit(all_names)
        except ValueError:
            logger.warning("    TF-IDF fit failed – skipping")
            return

        s1_mat = vectorizer.transform(s1_names)
        s2_ids = self.df_s2["entity_id"].tolist() if not self.df_s2.empty else []
        s3_ids = self.df_s3["entity_id"].tolist() if not self.df_s3.empty else []

        count = 0
        # Process in batches to limit memory
        batch = 50
        s1_ids = self.df_s1["entity_id"].tolist()

        for start in range(0, len(s1_ids), batch):
            end = min(start + batch, len(s1_ids))
            s1_batch = s1_mat[start:end]

            # S2
            if s2_names:
                s2_mat = vectorizer.transform(s2_names)
                sim_s2 = cosine_similarity(s1_batch, s2_mat)
                for i, sims in enumerate(sim_s2):
                    s1_id = s1_ids[start + i]
                    hits = [s2_ids[j] for j, sc in enumerate(sims) if sc >= TFIDF_THRESHOLD]
                    count += self._add_pairs(s1_id, hits, "tfidf_s2")

            # S3
            if s3_names:
                s3_mat = vectorizer.transform(s3_names)
                sim_s3 = cosine_similarity(s1_batch, s3_mat)
                for i, sims in enumerate(sim_s3):
                    s1_id = s1_ids[start + i]
                    hits = [s3_ids[j] for j, sc in enumerate(sims) if sc >= TFIDF_THRESHOLD]
                    count += self._add_pairs(s1_id, hits, "tfidf_s3")

        logger.info("    → cumulative unique pairs (approx): %d",
                    sum(len(v) for v in self._candidates.values()))


    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize(self) -> Dict[str, List[str]]:
        """Build final dict ensuring every S1 entity has an entry."""
        result: Dict[str, List[str]] = {}
        s1_ids = self.df_s1["entity_id"].tolist()
        for s1_id in s1_ids:
            cands = self._candidates.get(s1_id, set())
            # Cap to MAX_CANDIDATES to keep downstream feasible
            cand_list = sorted(cands)[:MAX_CANDIDATES]
            result[s1_id] = cand_list
        return result
