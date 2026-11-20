"""
Feature engineering and offline Parquet feature-store materialization.

Reads ``data/breast_cancer_cleaned.csv``, engineers clinical ratio/product
features, scales all model features with ``RobustScaler``, persists the
scaler artifact, materializes the feature store as Parquet, and writes
``feature_store/registry.json`` describing the feature group.

Run from the repository root with::

    python -m src.feature_engineering
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

INPUT_PATH: Path = PROJECT_ROOT / "data" / "breast_cancer_cleaned.csv"

FEATURE_STORE_DIR: Path = PROJECT_ROOT / "feature_store"

SCALER_PATH: Path = FEATURE_STORE_DIR / "cancer_scaler.pkl"

STORE_PATH: Path = FEATURE_STORE_DIR / "cancer_features.parquet"

REGISTRY_PATH: Path = FEATURE_STORE_DIR / "registry.json"

ENTITY_KEY: str = "patient_id"
TIMESTAMP_KEY: str = "event_timestamp"
TARGET: str = "target"

FEATURE_GROUP: str = "breast_cancer_features"
VERSION: str = "v1.0"


# ---------------------------------------------------------------------------
# Feature engineering pipeline
# ---------------------------------------------------------------------------

def run_feature_engineering() -> dict[str, Any]:
    """
    Engineer features, scale them, and materialize the Parquet store.

    Returns:
        The feature-store registry dictionary written to registry.json.
    """

    # Validate input dataset
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {INPUT_PATH}. "
            "Run 'python -m src.preprocessing' first."
        )

    # Create feature-store directory
    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)

    # Load cleaned dataset
    df: pd.DataFrame = pd.read_csv(INPUT_PATH)

    logger.info("Loaded cleaned dataset with shape %s", df.shape)

    # -----------------------------------------------------------------------
    # Clinical ratio / product feature engineering
    # -----------------------------------------------------------------------

    # Epsilon guards against division by zero
    df["area_to_perimeter_ratio"] = (
        df["mean_area"] / (df["mean_perimeter"] + 1e-5)
    )

    df["worst_to_mean_radius_ratio"] = (
        df["worst_radius"] / (df["mean_radius"] + 1e-5)
    )

    df["concavity_compactness_product"] = (
        df["mean_concavity"] * df["mean_compactness"]
    )

    # -----------------------------------------------------------------------
    # Separate metadata and model features
    # -----------------------------------------------------------------------

    meta_cols = [
        ENTITY_KEY,
        TIMESTAMP_KEY,
        TARGET,
    ]

    feature_cols: list[str] = [
        column
        for column in df.columns
        if column not in meta_cols
    ]

    # -----------------------------------------------------------------------
    # Feature scaling
    # -----------------------------------------------------------------------

    scaler = RobustScaler()

    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Persist fitted scaler
    joblib.dump(scaler, SCALER_PATH)

    logger.info("Persisted fitted scaler to %s", SCALER_PATH)

    # -----------------------------------------------------------------------
    # Materialize feature store as Parquet
    # -----------------------------------------------------------------------

    df.to_parquet(STORE_PATH, index=False)

    logger.info(
        "Materialized %d features to %s",
        len(feature_cols),
        STORE_PATH,
    )

    # -----------------------------------------------------------------------
    # Feature-store registry
    # -----------------------------------------------------------------------

    registry: dict[str, Any] = {
        "feature_group": FEATURE_GROUP,
        "version": VERSION,
        "entity_key": ENTITY_KEY,
        "timestamp_key": TIMESTAMP_KEY,
        "target": TARGET,
        "features": feature_cols,
        "storage_path": str(
            STORE_PATH.relative_to(PROJECT_ROOT)
        ),
    }

    # Write registry metadata
    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=4),
        encoding="utf-8",
    )

    logger.info(
        "Wrote feature-store registry to %s",
        REGISTRY_PATH,
    )

    return registry


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    run_feature_engineering()


if __name__ == "__main__":
    main()