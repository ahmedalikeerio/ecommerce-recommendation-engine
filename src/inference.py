from pathlib import Path

import joblib


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"


# ============================================================
# Load Model Artifacts
# ============================================================

def load_artifacts():

    print("Loading model artifacts...")

    cf_model = joblib.load(
        MODEL_DIR / "cf_model.joblib"
    )

    popularity_scores = joblib.load(
        MODEL_DIR / "popularity_scores.joblib"
    )

    user_histories = joblib.load(
        MODEL_DIR / "user_histories.joblib"
    )

    similar_items = joblib.load(
        MODEL_DIR / "similar_items.joblib"
    )

    mappings = joblib.load(
        MODEL_DIR / "mappings.joblib"
    )

    config = joblib.load(
        MODEL_DIR / "config.joblib"
    )

    print("All artifacts loaded successfully.")

    return {
        "cf_model": cf_model,
        "popularity_scores": popularity_scores,
        "user_histories": user_histories,
        "similar_items": similar_items,
        "user_to_index": mappings["user_to_index"],
        "item_to_index": mappings["item_to_index"],
        "config": config,
    }


# ============================================================
# Recommendation
# ============================================================

def recommend(
    user_id,
    artifacts,
    n_recommendations=10,
):

    user_histories = artifacts[
        "user_histories"
    ]

    similar_items = artifacts[
        "similar_items"
    ]

    popularity_scores = artifacts[
        "popularity_scores"
    ]

    config = artifacts["config"]

    history = user_histories.get(
        user_id,
        []
    )

    if not history:
        return []

    interacted_items = set(history)

    candidate_scores = {}

    # --------------------------------------------------------
    # Collaborative filtering
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
                    0.0
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
    # Hybrid scoring
    # --------------------------------------------------------

    hybrid_scores = {}

    for item_id, cf_score in (
        candidate_scores.items()
    ):

        popularity_score = (
            popularity_scores.get(
                item_id,
                0.0
            )
        )

        hybrid_score = (
            config["cf_weight"] * cf_score
            +
            config["popularity_weight"]
            * popularity_score
        )

        hybrid_scores[item_id] = (
            hybrid_score
        )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    recommendations = sorted(
        hybrid_scores.items(),
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
# Test Inference
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RECOMMENDATION INFERENCE")
    print("=" * 60)

    artifacts = load_artifacts()

    # Test user
    example_user = 829044

    recommendations = recommend(
        example_user,
        artifacts,
        n_recommendations=10,
    )

    print(
        f"\nUser: {example_user}"
    )

    print(
        "\nRecommendations:"
    )

    print(recommendations)