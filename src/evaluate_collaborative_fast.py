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

TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"

K = 10
MAX_USERS = 10_000
N_SIMILAR_ITEMS = 20


# ============================================================
# Metrics
# ============================================================

def precision_at_k(recommended, relevant, k=10):

    recommended = recommended[:k]

    if not recommended:
        return 0.0

    hits = len(set(recommended) & set(relevant))

    return hits / k


def recall_at_k(recommended, relevant, k=10):

    if not relevant:
        return 0.0

    recommended = recommended[:k]

    hits = len(set(recommended) & set(relevant))

    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k=10):

    recommended = recommended[:k]

    relevant = set(relevant)

    dcg = 0.0

    for rank, item in enumerate(recommended, start=1):

        if item in relevant:
            dcg += 1 / np.log2(rank + 1)

    ideal_hits = min(len(relevant), k)

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg


# ============================================================
# Build User Histories
# ============================================================

def build_user_histories(train):

    print("Building user histories...")

    user_histories = (
        train
        .groupby("user_id")["item_id"]
        .agg(list)
        .to_dict()
    )

    return user_histories


# ============================================================
# Select Evaluation Users
# ============================================================

def select_evaluation_users(
    test,
    user_to_index,
    max_users=10_000,
):

    test = test[
        test["user_id"].isin(user_to_index)
    ].copy()

    user_test_items = (
        test
        .groupby("user_id")["item_id"]
        .apply(set)
    )

    if len(user_test_items) > max_users:

        user_test_items = user_test_items.sample(
            n=max_users,
            random_state=42
        )

    return user_test_items


# ============================================================
# Precompute Similar Items
# ============================================================

def precompute_similar_items(
    item_user_matrix,
    item_to_index,
    user_histories,
    evaluation_users,
    model,
    n_similar_items=20,
):

    print("Finding unique history items...")

    unique_items = set()

    for user_id in evaluation_users.index:

        history = user_histories.get(user_id, [])

        for item_id in history:

            if item_id in item_to_index:
                unique_items.add(item_id)

    unique_items = list(unique_items)

    print(
        f"Unique items requiring similarity search: "
        f"{len(unique_items):,}"
    )

    # Convert item IDs to matrix indices
    query_indices = [
        item_to_index[item_id]
        for item_id in unique_items
    ]

    print("Computing item similarities in batch...")

    distances, indices = model.kneighbors(
        item_user_matrix[query_indices],
        n_neighbors=n_similar_items + 1,
    )

    index_to_item = {
        index: item_id
        for item_id, index in item_to_index.items()
    }

    similar_items = {}

    for row, item_id in enumerate(unique_items):

        neighbors = []

        for distance, neighbor_index in zip(
            distances[row],
            indices[row]
        ):

            neighbor_item = index_to_item[neighbor_index]

            similarity = 1 - distance

            neighbors.append(
                (neighbor_item, similarity)
            )

        similar_items[item_id] = neighbors

    return similar_items


# ============================================================
# Fast Recommendation
# ============================================================

def recommend_for_user_fast(
    user_id,
    user_histories,
    similar_items,
    item_to_index,
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

        for similar_item, similarity in similar_items[item_id]:

            # Don't recommend previously interacted items
            if similar_item in interacted_items:
                continue

            candidate_scores[similar_item] = (
                candidate_scores.get(
                    similar_item,
                    0.0
                )
                + similarity
            )

    recommendations = sorted(
        candidate_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        item_id
        for item_id, score in recommendations[
            :n_recommendations
        ]
    ]


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    evaluation_users,
    user_histories,
    similar_items,
    item_to_index,
    k=10,
):

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    evaluated_users = 0

    print("Evaluating users...")

    for user_id, relevant_items in evaluation_users.items():

        recommendations = recommend_for_user_fast(
            user_id=user_id,
            user_histories=user_histories,
            similar_items=similar_items,
            item_to_index=item_to_index,
            n_recommendations=k,
        )

        if not recommendations:
            continue

        precision_scores.append(
            precision_at_k(
                recommendations,
                relevant_items,
                k
            )
        )

        recall_scores.append(
            recall_at_k(
                recommendations,
                relevant_items,
                k
            )
        )

        ndcg_scores.append(
            ndcg_at_k(
                recommendations,
                relevant_items,
                k
            )
        )

        evaluated_users += 1

    return {
        "Precision@10": np.mean(
            precision_scores
        ),
        "Recall@10": np.mean(
            recall_scores
        ),
        "NDCG@10": np.mean(
            ndcg_scores
        ),
        "Evaluated Users": evaluated_users,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("OPTIMIZED COLLABORATIVE FILTERING EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading training data...")

    train = load_data()

    print(
        f"Training interactions: {len(train):,}"
    )

    print("\nLoading test data...")

    test = pd.read_csv(TEST_FILE)

    # --------------------------------------------------------
    # Prepare matrix
    # --------------------------------------------------------

    print("\nPreparing sparse matrix...")

    (
        train,
        user_item_matrix,
        item_user_matrix,
        user_to_index,
        item_to_index,
    ) = prepare_matrix(train)

    print(
        f"Users: {user_item_matrix.shape[0]:,}"
    )

    print(
        f"Items: {item_user_matrix.shape[0]:,}"
    )

    print(
        f"Interactions: {user_item_matrix.nnz:,}"
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print("\nTraining model...")

    model = train_model(
        item_user_matrix
    )

    # --------------------------------------------------------
    # User histories
    # --------------------------------------------------------

    user_histories = build_user_histories(
        train
    )

    # --------------------------------------------------------
    # Evaluation users
    # --------------------------------------------------------

    evaluation_users = select_evaluation_users(
        test,
        user_to_index,
        MAX_USERS,
    )

    print(
        f"Evaluation users: "
        f"{len(evaluation_users):,}"
    )

    # --------------------------------------------------------
    # Precompute similarities
    # --------------------------------------------------------

    similar_items = precompute_similar_items(
        item_user_matrix=item_user_matrix,
        item_to_index=item_to_index,
        user_histories=user_histories,
        evaluation_users=evaluation_users,
        model=model,
        n_similar_items=N_SIMILAR_ITEMS,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = evaluate(
        evaluation_users=evaluation_users,
        user_histories=user_histories,
        similar_items=similar_items,
        item_to_index=item_to_index,
        k=K,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("OPTIMIZED RESULTS")
    print("=" * 60)

    for metric, value in results.items():

        if isinstance(value, float):
            print(
                f"{metric}: {value:.6f}"
            )
        else:
            print(
                f"{metric}: {value:,}"
            )