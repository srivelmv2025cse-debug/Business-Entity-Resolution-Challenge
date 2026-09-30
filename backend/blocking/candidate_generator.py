"""
candidate_generator.py – End-to-end pipeline that:

  1. Loads train OR test data (whichever is available)
  2. Normalizes all three sources
  3. Runs multi-strategy blocking
  4. Writes output/candidate_pairs.tsv

Usage:
    python -m backend.blocking.candidate_generator            # uses test data
    python -m backend.blocking.candidate_generator --train    # uses train data
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on PYTHONPATH when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.loader import (
    load_all_sources,
    get_test_paths,
    get_train_paths,
)
from backend.preprocessing.normalizer import normalize_dataframe
from backend.blocking.blocking import BlockingEngine

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_FILE = OUTPUT_DIR / "candidate_pairs.tsv"


# ---------------------------------------------------------------------------
# Synthetic dev data (used when official data is absent)
# ---------------------------------------------------------------------------

def _create_synthetic_data(base_dir: Path) -> None:
    """Write tiny synthetic TSV files under *base_dir* for dev/testing."""
    base_dir.mkdir(parents=True, exist_ok=True)
    s1_rows = [
        ["S1_001", "Acme Corporation", "123 Main St, Springfield, IL 62701", "US"],
        ["S1_002", "TechNova Solutions Ltd", "456 Innovation Dr, Austin TX 78701", "US"],
        ["S1_003", "Global Freight & Logistics Inc", "78 Harbor Rd, Los Angeles CA 90001", "US"],
        ["S1_004", "Boulangerie Dupont", "12 Rue de la Paix, 75001 Paris", "France"],
        ["S1_005", "Müller Enterprises GmbH", "Berliner Strasse 5, 10115 Berlin", "Germany"],
        ["S1_006", "Sunrise Technologies Pvt Ltd", "Plot 42 MIDC Pune 411018", "India"],
        ["S1_007", "Summit Partners LLC", "789 Wall Street New York NY 10005", "US"],
        ["S1_008", "NoMatch Entity", "999 Unknown Ave", "US"],
    ]
    s2_rows = [
        ["S2_001", "ACME Corp.", "123 Main Street Springfield Illinois", "US"],
        ["S2_002", "TechNova Solutions", "456 Innovation Drive Austin Texas", "US"],
        ["S2_003", "Global Freight and Logistics", "78 Harbor Road Los Angeles CA", "US"],
        ["S2_004", "Boulangerie Dupont SARL", "12 rue de la paix 75001 Paris", "France"],
        ["S2_005", "Muller Enterprises", "Berliner Str 5 10115 Berlin", "Germany"],
        ["S2_009", "Other Company A", "100 Broadway New York", "US"],
    ]
    s3_rows = [
        ["S3_001", "Acme Corporation LLC", "123 Main St Springfield IL 62701", "US"],
        ["S3_002", "Technova Sol Ltd", "456 Innovation Dr Austin TX", "US"],
        ["S3_006", "Sunrise Tech Pvt Ltd", "Plot 42 MIDC Pune 411018", "India"],
        ["S3_007", "Summit Partners", "789 Wall St New York NY 10005", "US"],
        ["S3_010", "Other Company B", "200 Park Ave New York", "US"],
    ]
    cols = ["entity_id", "business_name", "business_address", "country"]
    pd.DataFrame(s1_rows, columns=cols).to_csv(base_dir / "test_source1.tsv", sep="\t", index=False)
    pd.DataFrame(s2_rows, columns=cols).to_csv(base_dir / "test_source2.tsv", sep="\t", index=False)
    pd.DataFrame(s3_rows, columns=cols).to_csv(base_dir / "test_source3.tsv", sep="\t", index=False)
    logger.info("Synthetic dev data written to %s", base_dir)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(use_train: bool = False) -> Path:
    """Run the full preprocessing + blocking pipeline.

    Parameters
    ----------
    use_train:
        If ``True``, load from ``dataset/train/``; otherwise ``dataset/test/``.

    Returns
    -------
    Path
        Path to the generated ``candidate_pairs.tsv``.
    """
    paths = get_train_paths() if use_train else get_test_paths()

    # Check if official data exists; fall back to synthetic dev set
    s1_path = paths["source1"]
    if not s1_path.exists():
        logger.warning(
            "Official data not found at %s. "
            "Creating synthetic dev dataset for testing.",
            s1_path,
        )
        synthetic_dir = PROJECT_ROOT / "dataset" / "test"
        _create_synthetic_data(synthetic_dir)
        paths = get_test_paths()

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    gt_path = paths.get("ground_truth")
    df_s1, df_s2, df_s3, df_gt = load_all_sources(
        paths["source1"],
        paths["source2"],
        paths["source3"],
        gt_path if (gt_path and Path(gt_path).exists()) else None,
    )

    # ------------------------------------------------------------------
    # 2. Normalize
    # ------------------------------------------------------------------
    logger.info("Normalizing source dataframes …")
    normalize_dataframe(df_s1)
    normalize_dataframe(df_s2)
    normalize_dataframe(df_s3)

    # ------------------------------------------------------------------
    # 3. Block
    # ------------------------------------------------------------------
    engine = BlockingEngine(df_s1, df_s2, df_s3)
    candidates: dict[str, list[str]] = engine.run()

    # ------------------------------------------------------------------
    # 4. Write output
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id in df_s1["entity_id"].tolist():
        cand_list = candidates.get(s1_id, [])
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for c in cand_list:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        rows.append(
            {
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(deduped),
            }
        )

    out_df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    out_df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    logger.info("candidate_pairs.tsv written → %s (%d rows)", OUTPUT_FILE, len(out_df))

    # Quick sanity check
    _sanity_check(out_df, df_s1, df_s2, df_s3)
    return OUTPUT_FILE


def _sanity_check(
    out_df: pd.DataFrame,
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
) -> None:
    """Run basic assertions on the output file."""
    s1_ids = set(df_s1["entity_id"].tolist())
    s2_ids = set(df_s2["entity_id"].tolist())
    s3_ids = set(df_s3["entity_id"].tolist())
    valid_candidate_ids = s2_ids | s3_ids

    out_s1_ids = set(out_df["source1_entity_id"].tolist())
    missing = s1_ids - out_s1_ids
    if missing:
        logger.error("SANITY FAIL: %d S1 entities missing from output: %s", len(missing), missing)
    else:
        logger.info("SANITY OK: all %d S1 entities present in output", len(s1_ids))

    s1_self_matches = 0
    invalid_cands = 0
    for _, row in out_df.iterrows():
        s1_id = row["source1_entity_id"]
        cand_str = row["candidate_entity_ids"]
        if not cand_str:
            continue
        cands = cand_str.split(",")
        for c in cands:
            if c in s1_ids:
                s1_self_matches += 1
                logger.error("  SELF-MATCH: %s → %s", s1_id, c)
            if c not in valid_candidate_ids:
                invalid_cands += 1
                logger.error("  INVALID CAND ID: %s → %s", s1_id, c)

    if s1_self_matches == 0 and invalid_cands == 0:
        logger.info("SANITY OK: no self-matches, no invalid candidate IDs")
    else:
        logger.error(
            "SANITY ISSUES – self-matches: %d  invalid IDs: %d",
            s1_self_matches, invalid_cands,
        )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run candidate pair generation")
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Use training data instead of test data",
    )
    args = parser.parse_args()
    output_path = run_pipeline(use_train=args.train)
    print(f"\nDone! Output: {output_path}")
