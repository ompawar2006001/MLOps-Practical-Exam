
"""Validation module for the Breast Cancer ML pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

INPUT_PATH: Path = (
    PROJECT_ROOT / "feature_store" / "cancer_features.parquet"
)

TARGET: str = "target"

REQUIRED_COLUMNS = [
    "patient_id",
    "event_timestamp",
    "target",
]


def run_validation() -> bool:
    """Validate the processed Breast Cancer feature dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found at {INPUT_PATH}. "
            "Run 'python -m src.feature_engineering' first."
        )

    df = pd.read_parquet(INPUT_PATH)

    logger.info("Loaded feature dataset with shape %s", df.shape)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Feature dataset is empty.")

    if df.isnull().any().any():
        raise ValueError(
            "Dataset contains missing values."
        )

    if df.duplicated(subset=["patient_id"]).any():
        raise ValueError(
            "Duplicate patient IDs found."
        )

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "Target must contain exactly two classes."
        )

    feature_columns = [
        column
        for column in df.columns
        if column not in REQUIRED_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No model features found.")

    if not all(
        pd.api.types.is_numeric_dtype(df[column])
        for column in feature_columns
    ):
        raise ValueError(
            "All model features must be numeric."
        )

    numeric_features = df[feature_columns].to_numpy()

    if not np.isfinite(numeric_features).all():
        raise ValueError(
            "Infinite values found in model features."
        )

    logger.info("Validation completed successfully.")
    logger.info("Total records: %d", len(df))
    logger.info("Total features: %d", len(feature_columns))
    logger.info("Target classes: %s", sorted(df[TARGET].unique()))

    return True


def main() -> None:
    """CLI entry point."""
    run_validation()


if __name__ == "__main__":
    main()
