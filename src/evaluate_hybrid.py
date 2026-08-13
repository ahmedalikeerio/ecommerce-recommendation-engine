from pathlib import Path

import numpy as np
import pandas as pd

from hybrid import (
    build_popularity_scores,
    build_user_histories,
    build_similar_items,
    recommend_hybrid,
)

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
N_SIMILAR_ITEMS = 40

CF_WEIGHT = 0.4
POPULARITY_WEIGHT = 0.6


# ============================================================
# Metrics
# ============================================================

def precision_at_k(recommended, relevant, k=10):

    recommended = recommended[:k]

    hits = len(
        set(recommended) & set(relevant)
    )

    return hits / k


def recall_at_k(recommended, relevant, k=10):

    if not relevant:
        return 0.0

    hits = len(
        set(recommended) & set(relevant)
    )

    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k=10):

    recommended = recommended[:k]
    relevant = set(relevant)

    dcg = 0.0

    for rank, item in enumerate(
        recommended,
        start=1
    ):

        if item in relevant:

            dcg += (
                1 /
                np.log2(rank + 1)
            )

    ideal_hits = min(
        len(relevant),
        k
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1 /
        np.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    return dcg / idcg


# ============================================================
# Evaluation Users
# ============================================================

def select_evaluation_users(
    test,
    user_to_index,
    max_users=10_000,
):

    test = test[
        test["user_id"].isin(
            user_to_index
        )
    ]

    user_test_items = (
        test
        .groupby("user_id")["item_id"]
        .apply(set)
    )

    if len(user_test_items) > max_users:

        user_test_items = (
            user_test_items.sample(
                n=max_users,
                random_state=42
            )
        )

    return user_test_items


# ============================================================
# Evaluation
# ============================================================

def evaluate_hybrid(
    evaluation_users,
    user_histories,
    similar_items,
    popularity_scores,
):

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    evaluated_users = 0

    print("Evaluating users...")

    for user_id, relevant_items in (
        evaluation_users.items()
    ):

        recommendations = (
            recommend_hybrid(
                user_id=user_id,
                user_histories=user_histories,
                similar_items=similar_items,
                popularity_scores=popularity_scores,
                n_recommendations=K,
                cf_weight=CF_WEIGHT,
                popularity_weight=POPULARITY_WEIGHT,
            )
        )

        if not recommendations:
            continue

        precision_scores.append(
            precision_at_k(
                recommendations,
                relevant_items,
                K
            )
        )

        recall_scores.append(
            recall_at_k(
                recommendations,
                relevant_items,
                K
            )
        )

        ndcg_scores.append(
            ndcg_at_k(
                recommendations,
                relevant_items,
                K
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
    print("HYBRID RECOMMENDATION EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading training data...")

    train = load_data()

    print(
        f"Training interactions: "
        f"{len(train):,}"
    )

    test = pd.read_csv(
        TEST_FILE
    )

    # --------------------------------------------------------
    # Prepare matrix
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

    # --------------------------------------------------------
    # Train CF model
    # --------------------------------------------------------

    print("\nTraining CF model...")

    model = train_model(
        item_user_matrix,
        n_similar_items=N_SIMILAR_ITEMS,
    )

    # --------------------------------------------------------
    # Build histories
    # --------------------------------------------------------

    print(
        "\nBuilding user histories..."
    )

    user_histories = (
        build_user_histories(train)
    )

    # --------------------------------------------------------
    # Popularity
    # --------------------------------------------------------

    print(
        "\nBuilding popularity scores..."
    )

    popularity_scores = (
        build_popularity_scores(train)
    )

    # --------------------------------------------------------
    # Similar items
    # --------------------------------------------------------

    print(
        "\nBuilding similar-item graph..."
    )

    similar_items = (
        build_similar_items(
            model,
            item_user_matrix,
            item_to_index,
        )
    )

    # --------------------------------------------------------
    # Evaluation users
    # --------------------------------------------------------

    evaluation_users = (
        select_evaluation_users(
            test,
            user_to_index,
            MAX_USERS,
        )
    )

    print(
        f"\nEvaluation users: "
        f"{len(evaluation_users):,}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = evaluate_hybrid(
        evaluation_users,
        user_histories,
        similar_items,
        popularity_scores,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("HYBRID RESULTS")
    print("=" * 60)

    for metric, value in results.items():

        if isinstance(value, float):

            print(
                f"{metric}: "
                f"{value:.6f}"
            )

        else:

            print(
                f"{metric}: "
                f"{value:,}"
            )