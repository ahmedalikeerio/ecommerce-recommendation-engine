import logging
import os
import time
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import RecommendationService


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "ecommerce-hybrid-recommender"
MODEL_ALIAS = "champion"

EXPERIMENT_ID = "320293588776741204"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5001",
)

ARTIFACT_ROOT = os.getenv(
    "MODEL_ARTIFACT_ROOT",
    "/app/mlruns",
)

CF_WEIGHT = 0.4
POPULARITY_WEIGHT = 0.6


# ============================================================
# MLflow
# ============================================================

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Load Champion Model Metadata
# ============================================================

print("Loading champion model...")

client = MlflowClient()

try:

    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS,
    )

    MODEL_VERSION = champion.version
    RUN_ID = champion.run_id

    print(
        f"Champion version: {MODEL_VERSION}"
    )

    print(
        f"Champion run: {RUN_ID}"
    )

except Exception as e:

    logger.exception(
        "Failed to load champion model from MLflow."
    )

    raise RuntimeError(
        f"Could not load champion model "
        f"'{MODEL_NAME}@{MODEL_ALIAS}'"
    ) from e


# ============================================================
# Artifact Location
# ============================================================

artifact_dir = (
    Path(ARTIFACT_ROOT)
    / EXPERIMENT_ID
    / RUN_ID
    / "artifacts"
    / "model_artifacts"
)

print(
    f"Loading inference artifacts from: "
    f"{artifact_dir}"
)


# ============================================================
# Inference Service
# ============================================================

try:

    recommendation_service = RecommendationService(
        artifact_dir=artifact_dir,
        cf_weight=CF_WEIGHT,
        popularity_weight=POPULARITY_WEIGHT,
    )

    logger.info(
        "Recommendation inference service initialized."
    )

except Exception as e:

    logger.exception(
        "Failed to initialize recommendation service."
    )

    raise RuntimeError(
        "Could not initialize recommendation service."
    ) from e


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="E-commerce Recommendation API",
    description=(
        "Production recommendation API using "
        "a hybrid collaborative filtering and "
        "popularity model."
    ),
    version="1.0.0",
)


# ============================================================
# Request Schema
# ============================================================

class RecommendationRequest(BaseModel):

    user_id: int = Field(
        ...,
        gt=0,
        description="Unique user ID",
        examples=[829044],
    )

    n_recommendations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recommendations",
    )


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "ecommerce-recommendation-api",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }


# ============================================================
# Recommendation Endpoint
# ============================================================

@app.post("/recommend")
def recommend(
    request: RecommendationRequest,
):

    start_time = time.perf_counter()

    try:

        recommendations = (
            recommendation_service.recommend(
                user_id=request.user_id,
                n_recommendations=(
                    request.n_recommendations
                ),
            )
        )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        if not recommendations:

            logger.warning(
                "No recommendations found | user_id=%s",
                request.user_id,
            )

        else:

            logger.info(
                (
                    "Recommendation request | "
                    "user_id=%s | "
                    "recommendations=%d | "
                    "latency_ms=%.2f"
                ),
                request.user_id,
                len(recommendations),
                latency_ms,
            )

        return {
            "user_id": request.user_id,
            "recommendations": recommendations,
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "latency_ms": round(
                latency_ms,
                2,
            ),
        }

    except Exception as e:

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        logger.exception(
            (
                "Recommendation request failed | "
                "user_id=%s | "
                "latency_ms=%.2f"
            ),
            request.user_id,
            latency_ms,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate recommendations.",
        ) from e