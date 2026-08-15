from pathlib import Path

import joblib
import mlflow
from mlflow.models import infer_signature


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

MLFLOW_TRACKING_URI = "file:./mlruns"

MODEL_NAME = "ecommerce-hybrid-recommender"


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("REGISTERING CHAMPION MODEL")
    print("=" * 60)

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        "ecommerce-recommendation"
    )

    # --------------------------------------------------------
    # Load saved artifacts
    # --------------------------------------------------------

    print("\nLoading model artifacts...")

    artifacts = {
        "cf_model": joblib.load(
            MODEL_DIR / "cf_model.joblib"
        ),

        "popularity_scores": joblib.load(
            MODEL_DIR / "popularity_scores.joblib"
        ),

        "user_histories": joblib.load(
            MODEL_DIR / "user_histories.joblib"
        ),

        "similar_items": joblib.load(
            MODEL_DIR / "similar_items.joblib"
        ),

        "mappings": joblib.load(
            MODEL_DIR / "mappings.joblib"
        ),

        "config": joblib.load(
            MODEL_DIR / "config.joblib"
        ),
    }

    config = artifacts["config"]

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="champion_hybrid_model"
    ):

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        mlflow.log_params({
            "model_type": "hybrid",
            "cf_weight": config["cf_weight"],
            "popularity_weight": config[
                "popularity_weight"
            ],
            "n_similar_items": config[
                "n_similar_items"
            ],
        })

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        mlflow.log_metrics({
            "precision_at_10": 0.007040,
            "recall_at_10": 0.039526,
            "ndcg_at_10": 0.028895,
        })

        # ----------------------------------------------------
        # Log artifacts
        # ----------------------------------------------------

        for file in MODEL_DIR.glob(
            "*.joblib"
        ):

            mlflow.log_artifact(
                str(file),
                artifact_path="model_artifacts",
            )

        # ----------------------------------------------------
        # Tags
        # ----------------------------------------------------

        mlflow.set_tags({
            "model_stage": "candidate",
            "dataset": "ecommerce-events",
            "model_type": "hybrid-recommender",
        })

        run_id = mlflow.active_run().info.run_id

        print(
            f"\nMLflow Run ID: {run_id}"
        )

    print("\n" + "=" * 60)
    print("MODEL REGISTERED AS EXPERIMENT")
    print("=" * 60)