import logging
import os
import time
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "ecommerce-hybrid-recommender"
MODEL_ALIAS = "champion"

EXPERIMENT_ID = "320293588776741204"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5001"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Load Champion Model
# ============================================================

print("Loading champion model...")

client = MlflowClient()

try:
    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS
    )

    VERSION = champion.version
    RUN_ID = champion.run_id

    print(f"Champion version: {VERSION}")
    print(f"Champion run: {RUN_ID}")

except Exception as e:
    logger.exception("Failed to load champion model from MLflow.")
    raise RuntimeError(
        f"Could not load champion model '{MODEL_NAME}@{MODEL_ALIAS}'"
    ) from e


# ============================================================
# Artifact Location
# ============================================================

# The MLflow registry contains a legacy File Store artifact URI:
#
# runs:/<RUN_ID>/model_artifacts
#
# In Docker Compose, both MLflow and FastAPI share ./mlruns,
# but they mount it at different paths.
#
# Therefore, the API loads the champion artifacts directly
# from the shared /app/mlruns directory.

artifact_dir = (
    Path("/app/mlruns")
    / EXPERIMENT_ID
    / RUN_ID
    / "artifacts"
    / "model_artifacts"
)

print(f"Loading artifacts from: {artifact_dir}")

if not artifact_dir.exists():
    raise FileNotFoundError(
        f"Model artifacts not found at: {artifact_dir}"
    )


# ============================================================
# Load Model Artifacts
# ============================================================

try:

    user_histories = joblib.load(
        artifact_dir / "user_histories.joblib"
    )

    similar_items = joblib.load(
        artifact_dir / "similar_items.joblib"
    )

    popularity_scores = joblib.load(
        artifact_dir / "popularity_scores.joblib"
    )

    logger.info("Champion artifacts loaded successfully.")

except Exception as e:

    logger.exception("Failed to load model artifacts.")

    raise RuntimeError(
        "Could not load champion model artifacts."
    ) from e


print("Champion artifacts loaded successfully.")


# ============================================================
# Recommendation Configuration
# ============================================================

CF_WEIGHT = 0.4
POPULARITY_WEIGHT = 0.6


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="E-commerce Recommendation API",
    description="Hybrid recommendation API using collaborative filtering and popularity ranking.",
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
        "model_version": VERSION,
    }


# ============================================================
# Recommendation Function
# ============================================================

def recommend_for_user(
    user_id: int,
    n_recommendations: int = 10,
):

    history = user_histories.get(user_id)

    if history is None:
        return []

    # --------------------------------------------------------
    # Collaborative Filtering Score
    # --------------------------------------------------------

    cf_scores = {}

    for item_id in history:

        neighbors = similar_items.get(item_id, [])

        for neighbor_item, similarity in neighbors:

            if neighbor_item in history:
                continue

            cf_scores[neighbor_item] = (
                cf_scores.get(neighbor_item, 0.0)
                + similarity
            )

    # --------------------------------------------------------
    # Normalize CF scores
    # --------------------------------------------------------

    if cf_scores:

        max_cf_score = max(cf_scores.values())

        if max_cf_score > 0:

            cf_scores = {
                item: score / max_cf_score
                for item, score in cf_scores.items()
            }

    # --------------------------------------------------------
    # Hybrid Scoring
    # --------------------------------------------------------

    hybrid_scores = {}

    candidate_items = set(cf_scores.keys())

    for item_id in popularity_scores.keys():

        if item_id not in history:
            candidate_items.add(item_id)

    for item_id in candidate_items:

        cf_score = cf_scores.get(item_id, 0.0)

        popularity_score = popularity_scores.get(
            item_id,
            0.0
        )

        final_score = (
            CF_WEIGHT * cf_score
            + POPULARITY_WEIGHT * popularity_score
        )

        hybrid_scores[item_id] = final_score

    # --------------------------------------------------------
    # Sort Recommendations
    # --------------------------------------------------------

    recommendations = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    recommendations = recommendations[
        :n_recommendations
    ]

    return [
        int(item_id)
        for item_id, _ in recommendations
    ]


# ============================================================
# Recommendation Endpoint
# ============================================================

@app.post("/recommend")
def recommend(request: RecommendationRequest):

    start_time = time.perf_counter()

    try:

        recommendations = recommend_for_user(
            user_id=request.user_id,
            n_recommendations=request.n_recommendations,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        if not recommendations:

            logger.warning(
                "No recommendations found | user_id=%s",
                request.user_id,
            )

        else:

            logger.info(
                "Recommendation request | user_id=%s | recommendations=%d | latency_ms=%.2f",
                request.user_id,
                len(recommendations),
                latency_ms,
            )

        return {
            "user_id": request.user_id,
            "recommendations": recommendations,
            "model": MODEL_NAME,
            "model_version": VERSION,
            "latency_ms": round(latency_ms, 2),
        }

    except Exception as e:

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.exception(
            "Recommendation request failed | user_id=%s | latency_ms=%.2f",
            request.user_id,
            latency_ms,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate recommendations.",
        ) from e