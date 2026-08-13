from pathlib import Path

import numpy as np
import pandas as pd

from collaborative import (
    load_data,
    prepare_matrix,
    train_model,
)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

N_SIMILAR_ITEMS = 40
N_RECOMMENDATIONS = 10

# Best configuration from experiments
CF_WEIGHT = 0.4
POPULARITY_WEIGHT = 0.6


# ============================================================
# Popularity Scores
# ============================================================

def build_popularity_scores(train):

    item_popularity = (
        train.groupby("item_id")["interaction_count"]
        .sum()
    )

    popularity_scores = np.log1p(item_popularity)

    min_score = popularity_scores.min()
    max_score = popularity_scores.max()

    if max_score > min_score:
        popularity_scores = (
            (popularity_scores - min_score)
            / (max_score - min_score)
        )

    return popularity_scores.to_dict()


# ============================================================
# User Histories
# ============================================================

def build_user_histories(train):

    return (
        train.groupby("user_id")["item_id"]
        .agg(list)
        .to_dict()
    )


# ============================================================
# Similar Item Graph
# ============================================================

def build_similar_items(
    model,
    item_user_matrix,
    item_to_index,
):

    index_to_item = {
        index: item_id
        for item_id, index in item_to_index.items()
    }

    distances, indices = model.kneighbors(
        item_user_matrix,
        n_neighbors=N_SIMILAR_ITEMS + 1
    )

    similar_items = {}

    for item_index in range(
        item_user_matrix.shape[0]
    ):

        item_id = index_to_item[item_index]

        neighbors = []

        for distance, neighbor_index in zip(
            distances[item_index][1:],
            indices[item_index][1:]
        ):

            neighbor_item = index_to_item[
                neighbor_index
            ]

            similarity = 1 - distance

            neighbors.append(
                (
                    neighbor_item,
                    similarity
                )
            )

        similar_items[item_id] = neighbors

    return similar_items


# ============================================================
# Hybrid Recommendation
# ============================================================

def recommend_hybrid(
    user_id,
    user_histories,
    similar_items,
    popularity_scores,
    n_recommendations=10,
    cf_weight=CF_WEIGHT,
    popularity_weight=POPULARITY_WEIGHT,
):

    history = user_histories.get(user_id, [])

    if not history:
        return []

    interacted_items = set(history)

    candidate_scores = {}

    # --------------------------------------------------------
    # Collaborative Filtering
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

    max_cf = max(candidate_scores.values())

    if max_cf > 0:

        candidate_scores = {
            item: score / max_cf
            for item, score in candidate_scores.items()
        }

    # --------------------------------------------------------
    # Hybrid Score
    # --------------------------------------------------------

    hybrid_scores = {}

    for item_id, cf_score in (
        candidate_scores.items()
    ):

        popularity_score = popularity_scores.get(
            item_id,
            0.0
        )

        hybrid_scores[item_id] = (
            cf_weight * cf_score
            + popularity_weight * popularity_score
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
        for item_id, _ in recommendations[
            :n_recommendations
        ]
    ]


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("Loading training data...")

    train = load_data()

    print(
        f"Training interactions: {len(train):,}"
    )

    # --------------------------------------------------------
    # Prepare Matrix
    # --------------------------------------------------------

    print("\nPreparing matrix...")

    (
        train,
        user_item_matrix,
        item_user_matrix,
        user_to_index,
        item_to_index,
    ) = prepare_matrix(
        train,
        view_weight=1,
        cart_weight=3,
        purchase_weight=5,
    )

    print(
        f"Users: {user_item_matrix.shape[0]:,}"
    )

    print(
        f"Items: {item_user_matrix.shape[0]:,}"
    )

    # --------------------------------------------------------
    # Train Collaborative Filtering
    # --------------------------------------------------------

    print("\nTraining CF model...")

    model = train_model(
        item_user_matrix,
        n_similar_items=N_SIMILAR_ITEMS,
    )

    # --------------------------------------------------------
    # Popularity
    # --------------------------------------------------------

    print("\nBuilding popularity scores...")

    popularity_scores = build_popularity_scores(train)

    # --------------------------------------------------------
    # User Histories
    # --------------------------------------------------------

    print("\nBuilding user histories...")

    user_histories = build_user_histories(train)

    # --------------------------------------------------------
    # Similar Items
    # --------------------------------------------------------

    print("\nBuilding similar-item graph...")

    similar_items = build_similar_items(
        model,
        item_user_matrix,
        item_to_index,
    )

    # --------------------------------------------------------
    # Example Recommendation
    # --------------------------------------------------------

    example_user = train["user_id"].iloc[0]

    recommendations = recommend_hybrid(
        user_id=example_user,
        user_histories=user_histories,
        similar_items=similar_items,
        popularity_scores=popularity_scores,
        n_recommendations=N_RECOMMENDATIONS,
    )

    print("\n" + "=" * 60)
    print("HYBRID RECOMMENDATIONS")
    print("=" * 60)

    print(f"\nUser: {example_user}")

    print("\nRecommended Items:")
    print(recommendations)