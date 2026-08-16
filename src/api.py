from pathlib import Path
import mlflow
from mlflow import MlflowClient

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

import time


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# MLflow Champion Model
# ============================================================

MLFLOW_TRACKING_URI = "file:./mlruns"
MODEL_NAME = "ecommerce-hybrid-recommender"
MODEL_ALIAS = "champion"


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

# Download artifacts belonging to champion run
artifact_dir = mlflow.artifacts.download_artifacts(
    run_id=RUN_ID,
    artifact_path="model_artifacts",
)

print(
    f"Artifacts loaded from: {artifact_dir}"
)

user_histories = joblib.load(
    Path(artifact_dir) / "user_histories.joblib"
)

similar_items = joblib.load(
    Path(artifact_dir) / "similar_items.joblib"
)

popularity_scores = joblib.load(
    Path(artifact_dir) / "popularity_scores.joblib"
)

print("Champion artifacts loaded successfully.")

# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="E-commerce Recommendation API",
    description="Hybrid recommendation system using collaborative filtering and popularity ranking.",
    version="1.0.0",
)


# ============================================================
# Request Schema
# ============================================================

class RecommendationRequest(BaseModel):
    user_id: int
    n_recommendations: int = 10

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 829044,
                "n_recommendations": 10
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
        []
    )

    if not history:
        return []

    interacted_items = set(history)

    candidate_scores = {}

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
                    0.0
                )
                + similarity
            )

    if not candidate_scores:
        return []

    max_cf = max(
        candidate_scores.values()
    )

    recommendations = []

    for item_id, cf_score in (
        candidate_scores.items()
    ):

        cf_score = (
            cf_score / max_cf
            if max_cf > 0
            else 0
        )

        popularity_score = (
            popularity_scores.get(
                item_id,
                0.0
            )
        )

        final_score = (
            0.4 * cf_score
            + 0.6 * popularity_score
        )

        recommendations.append(
            (
                item_id,
                final_score
            )
        )

    recommendations.sort(
        key=lambda x: x[1],
        reverse=True
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
    }


# ============================================================
# Recommendation Endpoint
# ============================================================

@app.post("/recommend")
def get_recommendations(
    request: RecommendationRequest,
):
    start_time = time.perf_counter()

    recommendations = recommend(
        user_id=request.user_id,
        n_recommendations=request.n_recommendations,
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail="No recommendations found for this user.",
        )

    print(
        f"Recommendation request | "
        f"user={request.user_id} | "
        f"recommendations={len(recommendations)} | "
        f"latency={latency_ms:.2f}ms"
    )

    return {
        "user_id": request.user_id,
        "recommendations": recommendations,
        "model": MODEL_NAME,
        "model_version": champion.version,
        "latency_ms": round(latency_ms, 2),
    }
