from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"


# ============================================================
# Load Model Artifacts
# ============================================================

print("Loading recommendation artifacts...")

user_histories = joblib.load(
    MODEL_DIR / "user_histories.joblib"
)

similar_items = joblib.load(
    MODEL_DIR / "similar_items.joblib"
)

popularity_scores = joblib.load(
    MODEL_DIR / "popularity_scores.joblib"
)

print("Artifacts loaded successfully.")


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
        item_id
        for item_id, score
        in recommendations[
            :n_recommendations
        ]
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

    recommendations = recommend(
        user_id=request.user_id,
        n_recommendations=request.n_recommendations,
    )

    if not recommendations:

        raise HTTPException(
            status_code=404,
            detail="No recommendations found for this user.",
        )

    return {
        "user_id": request.user_id,
        "recommendations": recommendations,
    }