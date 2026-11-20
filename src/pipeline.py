
"""End-to-end ML pipeline with MLflow (SQLite tracking backend).
Reads features from the Parquet feature store, trains three candidate
classifiers, logs params/metrics/artifacts to ``sqlite:///mlflow.db``
(with explicit model signatures and run tags), selects the winner by F1,
and persists it to ``models/best_cancer_model.pkl``.
Run from the repository root with::
python -m src.pipeline
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.feature_engineering import run_feature_engineering
from src.preprocessing import run_preprocessing
from src.validation import run_validation


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REGISTRY_PATH: Path = PROJECT_ROOT / "feature_store" / "registry.json"
MODELS_DIR: Path = PROJECT_ROOT / "models"
BEST_MODEL_PATH: Path = MODELS_DIR / "best_cancer_model.pkl"

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

TRACKING_URI: str = "sqlite:///mlflow.db"
EXPERIMENT_NAME: str = "Breast_Cancer_Classification_Team_G2"
TEAM_TAG: str = "Group_G2"
STAGE_TAG: str = "candidate_training"


def _load_registry() -> dict[str, Any]:
    """Load the feature-store registry.

    Raises:
        RuntimeError: If the registry file is missing.
    """

    if not REGISTRY_PATH.exists():
        raise RuntimeError(
            f"Feature-store registry not found at {REGISTRY_PATH}. "
            "Run 'python -m src.feature_engineering' first."
        )

    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def run_training_and_selection() -> tuple[str, float]:
    """Train candidate models, log them to MLflow, and save the winner.

    Returns:
        Tuple of (winning model name, F1 score).
    """

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    registry = _load_registry()

    df: pd.DataFrame = pd.read_parquet(
        PROJECT_ROOT / registry["storage_path"]
    )

    X: pd.DataFrame = df[registry["features"]]
    y: pd.Series = df[registry["target"]]

    logger.info(
        "Loaded feature store with shape %s (%d features)",
        df.shape,
        X.shape[1],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models: dict[str, Any] = {
        "SVC_RBF": SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE,
        ),
    }

    best_f1: float = -1.0
    best_model: Any = None
    best_name: str = ""

    for name, model in models.items():

        with mlflow.start_run(run_name=f"Cancer_{name}"):

            mlflow.set_tag("team", TEAM_TAG)
            mlflow.set_tag("stage", STAGE_TAG)

            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            recall = recall_score(y_test, preds)
            roc_auc = roc_auc_score(y_test, probs)

            mlflow.log_param("model_type", name)
            mlflow.log_param("feature_group", registry["feature_group"])

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("roc_auc", roc_auc)

            mlflow.log_artifact(str(REGISTRY_PATH))

            signature = infer_signature(
                X_train,
                model.predict(X_train),
            )

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.iloc[:5],
            )

            logger.info(
                "%s | Acc: %.4f | F1: %.4f",
                name,
                acc,
                f1,
            )

            if f1 > best_f1:
                best_f1, best_model, best_name = f1, model, name

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, BEST_MODEL_PATH)

    logger.info(
        "Winner: %s (F1: %.4f) -> %s",
        best_name,
        best_f1,
        BEST_MODEL_PATH,
    )

    return best_name, best_f1


def main() -> None:
    """Run the full pipeline: preprocess -> features -> validate -> train."""

    run_preprocessing()
    run_feature_engineering()
    run_validation()
    run_training_and_selection()


if __name__ == "__main__":
    main()
