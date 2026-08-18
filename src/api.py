from pathlib import Path
import logging
import time

import joblib
import mlflow
from mlflow import MlflowClient

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MLFLOW_TRACKING_URI = "file:./mlruns"

MODEL_NAME = "ecommerce-hybrid-recommender"
MODEL_ALIAS = "champion"


# ============================================================
# Logging Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("recommendation-api")


# ============================================================
# MLflow Champion Model
# ============================================================

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

client = MlflowClient()

print("Loading champion model...")

champion = client.get_model_version_by_alias(
    MODEL_NAME,
    MODEL_ALIAS,
)

RUN_ID = champion.run_id

print(
    f"Champion version: {champion.version}"
)

print(
    f"Champion run: {RUN_ID}"
)


# ============================================================
# Download Champion Artifacts
# ============================================================

artifact_dir = mlflow.artifacts.download_artifacts(
    run_id=RUN_ID,
    artifact_path="model_artifacts",
)

print(
    f"Artifacts loaded from: {artifact_dir}"
)


# ============================================================
# Load Recommendation Artifacts
# ============================================================

user_histories = joblib.load(
    Path(artifact_dir) / "user_histories.joblib"
)

similar_items = joblib.load(
    Path(artifact_dir) / "similar_items.joblib"
)

popularity_scores = joblib.load(
    Path(artifact_dir) / "popularity_scores.joblib"
)

print(
    "Champion artifacts loaded successfully."
)


logger.info(
    "Champion model loaded | "
    "model=%s | version=%s | run_id=%s",
    MODEL_NAME,
    champion.version,
    RUN_ID,
)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="E-commerce Recommendation API",
    description=(
        "Production-style hybrid recommendation API "
        "using collaborative filtering and popularity ranking."
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
    )

    n_recommendations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recommendations to return",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 829044,
                "n_recommendations": 10,
            }
        }


# ============================================================
# Recommendation Logic
# ============================================================

def recommend(
    user_id,
    n_recommendations=10,
):

    history = user_histories.get(
        user_id,
        [],
    )

    if not history:
        return []

    interacted_items = set(history)

    candidate_scores = {}

    # --------------------------------------------------------
    # Collaborative filtering scores
    # --------------------------------------------------------

    for item_id in history:

        if item_id not in similar_items:
            continue

        for similar_item, similarity in (
            similar_items[item_id]
        ):

            if similar_item in interacted_items:
                continue

            candidate_scores[similar_item] = (
                candidate_scores.get(
                    similar_item,
                    0.0,
                )
                + similarity
            )

    if not candidate_scores:
        return []

    # --------------------------------------------------------
    # Normalize CF scores
    # --------------------------------------------------------

    max_cf = max(
        candidate_scores.values()
    )

    if max_cf > 0:

        candidate_scores = {
            item: score / max_cf
            for item, score
            in candidate_scores.items()
        }

    # --------------------------------------------------------
    # Hybrid ranking
    # --------------------------------------------------------

    recommendations = []

    for item_id, cf_score in (
        candidate_scores.items()
    ):

        popularity_score = (
            popularity_scores.get(
                item_id,
                0.0,
            )
        )

        final_score = (
            0.4 * cf_score
            + 0.6 * popularity_score
        )

        recommendations.append(
            (
                item_id,
                final_score,
            )
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    recommendations.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        int(item_id)
        for item_id, score
        in recommendations[:n_recommendations]
    ]


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "recommendation-api",
        "model": MODEL_NAME,
        "model_version": champion.version,
    }


# ============================================================
# Recommendation Endpoint
# ============================================================

@app.post("/recommend")
def get_recommendations(
    request: RecommendationRequest,
):

    start_time = time.perf_counter()

    try:

        recommendations = recommend(
            user_id=request.user_id,
            n_recommendations=request.n_recommendations,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # ----------------------------------------------------
        # No recommendations
        # ----------------------------------------------------

        if not recommendations:

            logger.warning(
                "No recommendations | "
                "user=%s | model=%s | version=%s",
                request.user_id,
                MODEL_NAME,
                champion.version,
            )

            raise HTTPException(
                status_code=404,
                detail=(
                    "No recommendations found "
                    "for this user."
                ),
            )

        # ----------------------------------------------------
        # Successful request
        # ----------------------------------------------------

        logger.info(
            "Recommendation request | "
            "user=%s | recommendations=%s | "
            "model=%s | version=%s | latency_ms=%.2f",
            request.user_id,
            len(recommendations),
            MODEL_NAME,
            champion.version,
            latency_ms,
        )

        return {
            "user_id": request.user_id,
            "recommendations": recommendations,
            "model": MODEL_NAME,
            "model_version": champion.version,
            "latency_ms": round(
                latency_ms,
                2,
            ),
        }

    except HTTPException:
        raise

    except Exception as error:

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.exception(
            "Recommendation error | "
            "user=%s | latency_ms=%.2f | error=%s",
            request.user_id,
            latency_ms,
            error,
        )

        raise HTTPException(
            status_code=500,
            detail="Internal recommendation service error.",
        )