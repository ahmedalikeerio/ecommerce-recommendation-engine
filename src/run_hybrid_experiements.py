from pathlib import Path

import mlflow
import pandas as pd

from collaborative import (
    load_data,
    prepare_matrix,
    train_model,
)

from hybrid import (
    build_popularity_scores,
    build_user_histories,
    build_similar_items,
    recommend_hybrid,
)

from evaluate_hybrid import (
    select_evaluation_users,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
)


BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"

EXPERIMENT_NAME = "ecommerce-hybrid-recommendation"

K = 10
MAX_USERS = 10_000
N_SIMILAR_ITEMS = 40


# ============================================================
# Hybrid configurations
# ============================================================

EXPERIMENTS = [
    {
        "name": "hybrid_50_50",
        "cf_weight": 0.5,
        "popularity_weight": 0.5,
    },
    {
        "name": "hybrid_40_60",
        "cf_weight": 0.4,
        "popularity_weight": 0.6,
    },
    {
        "name": "hybrid_30_70",
        "cf_weight": 0.3,
        "popularity_weight": 0.7,
    },
    {
        "name": "hybrid_20_80",
        "cf_weight": 0.2,
        "popularity_weight": 0.8,
    },
]


# ============================================================
# MLflow
# ============================================================

mlflow.set_tracking_uri(
    f"file://{BASE_DIR / 'mlruns'}"
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# Load data ONCE
# ============================================================

print("Loading training data...")

train_raw = load_data()

print("Loading test data...")

test = pd.read_csv(TEST_FILE)


# ============================================================
# Prepare model ONCE
# ============================================================

print("\nPreparing matrix...")

train, user_item_matrix, item_user_matrix, user_to_index, item_to_index = (
    prepare_matrix(
        train_raw.copy(),
        view_weight=1,
        cart_weight=3,
        purchase_weight=5,
    )
)

print("\nTraining CF model...")

model = train_model(
    item_user_matrix,
    n_similar_items=N_SIMILAR_ITEMS,
)

print("\nBuilding user histories...")

user_histories = build_user_histories(train)

print("\nBuilding popularity scores...")

popularity_scores = build_popularity_scores(train)

print("\nBuilding similar-item graph...")

similar_items = build_similar_items(
    model,
    item_user_matrix,
    item_to_index,
)

print("\nSelecting evaluation users...")

evaluation_users = select_evaluation_users(
    test,
    user_to_index,
    MAX_USERS,
)

print(
    f"Evaluation users: "
    f"{len(evaluation_users):,}"
)


# ============================================================
# Run experiments
# ============================================================

for experiment in EXPERIMENTS:

    name = experiment["name"]

    cf_weight = experiment["cf_weight"]
    popularity_weight = experiment["popularity_weight"]

    print("\n")
    print("=" * 70)
    print(f"RUNNING: {name}")
    print("=" * 70)

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    evaluated_users = 0

    with mlflow.start_run(
        run_name=name
    ):

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        mlflow.log_params({
            "model": "hybrid_recommender",
            "cf_weight": cf_weight,
            "popularity_weight": popularity_weight,
            "view_weight": 1,
            "cart_weight": 3,
            "purchase_weight": 5,
            "n_similar_items": N_SIMILAR_ITEMS,
            "interaction_scaling": "log1p",
            "similarity": "cosine",
            "k": K,
            "evaluation_users": MAX_USERS,
        })

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        print("Evaluating...")

        for user_id, relevant_items in (
            evaluation_users.items()
        ):

            recommendations = recommend_hybrid(
                user_id=user_id,
                user_histories=user_histories,
                similar_items=similar_items,
                popularity_scores=popularity_scores,
                n_recommendations=K,
                cf_weight=cf_weight,
                popularity_weight=popularity_weight,
            )

            if not recommendations:
                continue

            precision_scores.append(
                precision_at_k(
                    recommendations,
                    relevant_items,
                    K,
                )
            )

            recall_scores.append(
                recall_at_k(
                    recommendations,
                    relevant_items,
                    K,
                )
            )

            ndcg_scores.append(
                ndcg_at_k(
                    recommendations,
                    relevant_items,
                    K,
                )
            )

            evaluated_users += 1

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        precision = sum(
            precision_scores
        ) / len(precision_scores)

        recall = sum(
            recall_scores
        ) / len(recall_scores)

        ndcg = sum(
            ndcg_scores
        ) / len(ndcg_scores)

        mlflow.log_metrics({
            "precision_at_10": precision,
            "recall_at_10": recall,
            "ndcg_at_10": ndcg,
        })

        mlflow.set_tags({
            "model_type": "hybrid",
            "dataset": "retailrocket",
            "experiment_type": "hybrid_weight_tuning",
        })

        print("\nResults:")

        print(
            f"Precision@10: {precision:.6f}"
        )

        print(
            f"Recall@10: {recall:.6f}"
        )

        print(
            f"NDCG@10: {ndcg:.6f}"
        )

        print(
            f"Evaluated Users: "
            f"{evaluated_users:,}"
        )

        print(
            f"\nMLflow run logged: {name}"
        )


print("\n")
print("=" * 70)
print("ALL HYBRID EXPERIMENTS COMPLETED")
print("=" * 70)