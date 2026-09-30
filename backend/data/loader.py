"""
loader.py – TSV data loading and validation for all entity sources.

Supports Source 1, Source 2, Source 3 and ground-truth files.
Always uses tab-separated format.  Does NOT hard-code any country list.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required columns for entity files and ground-truth
# ---------------------------------------------------------------------------
ENTITY_REQUIRED_COLS = {"entity_id", "business_name", "business_address", "country"}
GROUND_TRUTH_REQUIRED_COLS = {"source1_entity_id", "source2_entity_id"}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_tsv(path: str | Path, required_cols: Optional[set] = None) -> pd.DataFrame:
    """Read a TSV file into a DataFrame and validate required columns.

    Parameters
    ----------
    path:
        Absolute or relative path to the ``.tsv`` file.
    required_cols:
        Set of column names that must be present.  Raises ``ValueError`` if any
        are missing.

    Returns
    -------
    pd.DataFrame
        Loaded and lightly validated DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading TSV: %s", path)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    # Strip leading/trailing whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    if required_cols:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"File '{path.name}' is missing required columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

    logger.info("  → %d rows, %d columns", len(df), len(df.columns))
    return df


def load_entity_file(path: str | Path) -> pd.DataFrame:
    """Load an entity TSV (source1 / source2 / source3)."""
    df = load_tsv(path, required_cols=ENTITY_REQUIRED_COLS)
    _validate_entity_df(df, path)
    return df


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """Load a ground-truth TSV file."""
    df = load_tsv(path, required_cols=GROUND_TRUTH_REQUIRED_COLS)
    logger.info("Ground truth loaded: %d pairs", len(df))
    return df


def load_all_sources(
    source1_path: str | Path,
    source2_path: str | Path,
    source3_path: str | Path,
    ground_truth_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Convenience wrapper: load all three sources (+ optional ground truth).

    Returns
    -------
    (df_s1, df_s2, df_s3, df_gt)
        ``df_gt`` is ``None`` when *ground_truth_path* is not provided.
    """
    df_s1 = load_entity_file(source1_path)
    df_s2 = load_entity_file(source2_path)
    df_s3 = load_entity_file(source3_path)

    df_gt = None
    if ground_truth_path is not None:
        df_gt = load_ground_truth(ground_truth_path)

    _log_source_stats(df_s1, df_s2, df_s3)
    return df_s1, df_s2, df_s3, df_gt


# ---------------------------------------------------------------------------
# Default path resolution helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the project root directory (two levels up from this file)."""
    return Path(__file__).resolve().parent.parent.parent


def get_train_paths() -> dict:
    """Return default training-data file paths."""
    root = _project_root()
    base = root / "dataset" / "train"
    return {
        "source1": base / "train_source1.tsv",
        "source2": base / "train_source2.tsv",
        "source3": base / "train_source3.tsv",
        "ground_truth": base / "train_ground_truth.tsv",
    }


def get_test_paths() -> dict:
    """Return default test-data file paths."""
    root = _project_root()
    base = root / "dataset" / "test"
    return {
        "source1": base / "test_source1.tsv",
        "source2": base / "test_source2.tsv",
        "source3": base / "test_source3.tsv",
    }


def load_train_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all training data using default paths."""
    paths = get_train_paths()
    df_s1, df_s2, df_s3, df_gt = load_all_sources(
        paths["source1"], paths["source2"], paths["source3"], paths["ground_truth"]
    )
    return df_s1, df_s2, df_s3, df_gt


def load_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all test data using default paths."""
    paths = get_test_paths()
    df_s1, df_s2, df_s3, _ = load_all_sources(
        paths["source1"], paths["source2"], paths["source3"]
    )
    return df_s1, df_s2, df_s3


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_entity_df(df: pd.DataFrame, path: Path) -> None:
    """Warn about common data quality issues without raising errors."""
    n_nulls = df["entity_id"].isnull().sum() + (df["entity_id"] == "").sum()
    if n_nulls:
        logger.warning(
            "  [%s] %d rows with missing entity_id", path.name, n_nulls
        )

    n_dup = df["entity_id"].duplicated().sum()
    if n_dup:
        logger.warning(
            "  [%s] %d duplicate entity_ids detected", path.name, n_dup
        )

    countries = df["country"].unique().tolist()
    logger.info("  [%s] countries present: %s", path.name, countries)


def _log_source_stats(
    df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame
) -> None:
    logger.info(
        "Source sizes – S1: %d  S2: %d  S3: %d",
        len(df_s1), len(df_s2), len(df_s3),
    )
