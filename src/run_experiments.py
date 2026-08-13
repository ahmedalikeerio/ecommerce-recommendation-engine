from pathlib import Path

import mlflow
import pandas as pd

from collaborative import (
    load_data,
    prepare_matrix,
    train_model,
)

from evaluate_collaborative_fast import (
    build_user_histories,
    select_evaluation_users,
    precompute_similar_items,
    evaluate,
)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"

EXPERIMENT_NAME = "ecommerce-recommendation"

K = 10
MAX_USERS = 10_000
MAX_ITEMS = 50_000
MIN_ITEM_INTERACTIONS = 5



# ============================================================
# Experiments
# ============================================================

EXPERIMENTS = [
    {
        "name": "neighbors_20",
        "view_weight": 1,
        "cart_weight": 3,
        "purchase_weight": 5,
        "n_similar_items": 20,
    },
    {
        "name": "neighbors_30",
        "view_weight": 1,
        "cart_weight": 3,
        "purchase_weight": 5,
        "n_similar_items": 30,
    },
    {
        "name": "neighbors_40",
        "view_weight": 1,
        "cart_weight": 3,
        "purchase_weight": 5,
        "n_similar_items": 40,
    },
    {
        "name": "neighbors_50",
        "view_weight": 1,
        "cart_weight": 3,
        "purchase_weight": 5,
        "n_similar_items": 50,
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
# Load Data Once
# ============================================================

print("Loading training data...")

train_raw = load_data()

print("Loading test data...")

test = pd.read_csv(TEST_FILE)


# ============================================================
# Run Experiments
# ============================================================

for experiment in EXPERIMENTS:

    name = experiment["name"]

    print("\n")
    print("=" * 70)
    print(f"RUNNING EXPERIMENT: {name}")
    print("=" * 70)

    view_weight = experiment["view_weight"]
    cart_weight = experiment["cart_weight"]
    purchase_weight = experiment["purchase_weight"]
    n_similar_items = experiment["n_similar_items"]

    with mlflow.start_run(
        run_name=name
    ):

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        mlflow.log_params({

            "model": "item-item-collaborative-filtering",

            "view_weight": view_weight,

            "cart_weight": cart_weight,

            "purchase_weight": purchase_weight,

            "n_similar_items": n_similar_items,

            "interaction_scaling": "log1p",

            "similarity": "cosine",

            "max_items": MAX_ITEMS,

            "min_item_interactions":
                MIN_ITEM_INTERACTIONS,

            "k": K,

            "evaluation_users":
                MAX_USERS,
        })

        # ----------------------------------------------------
        # Prepare Matrix
        # ----------------------------------------------------

        train = train_raw.copy()

        (
            train,
            user_item_matrix,
            item_user_matrix,
            user_to_index,
            item_to_index,
        ) = prepare_matrix(

            train,

            view_weight=view_weight,

            cart_weight=cart_weight,

            purchase_weight=purchase_weight,
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        print("Training model...")

        model = train_model(
            item_user_matrix,
            n_similar_items=n_similar_items,
        )

        # ----------------------------------------------------
        # User histories
        # ----------------------------------------------------

        user_histories = (
            build_user_histories(train)
        )

        # ----------------------------------------------------
        # Evaluation users
        # ----------------------------------------------------

        evaluation_users = (
            select_evaluation_users(
                test,
                user_to_index,
                MAX_USERS,
            )
        )

        # ----------------------------------------------------
        # Similar items
        # ----------------------------------------------------

        similar_items = (
            precompute_similar_items(

                item_user_matrix=
                    item_user_matrix,

                item_to_index=
                    item_to_index,

                user_histories=
                    user_histories,

                evaluation_users=
                    evaluation_users,

                model=
                    model,

                n_similar_items=
                    n_similar_items,
            )
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        results = evaluate(

            evaluation_users=
                evaluation_users,

            user_histories=
                user_histories,

            similar_items=
                similar_items,

            item_to_index=
                item_to_index,

            k=K,
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        mlflow.log_metrics({

            "precision_at_10":
                float(results["Precision@10"]),

            "recall_at_10":
                float(results["Recall@10"]),

            "ndcg_at_10":
                float(results["NDCG@10"]),
        })

        mlflow.set_tags({

            "experiment_type":
                "hyperparameter_experiment",

            "model_type":
                "item_item_cf",

            "dataset":
                "retailrocket",

        })

        print("\nResults:")

        print(
            f"Precision@10: "
            f"{results['Precision@10']:.6f}"
        )

        print(
            f"Recall@10: "
            f"{results['Recall@10']:.6f}"
        )

        print(
            f"NDCG@10: "
            f"{results['NDCG@10']:.6f}"
        )

        print(
            f"Evaluated Users: "
            f"{results['Evaluated Users']:,}"
        )

        print(
            f"\nMLflow run logged: {name}"
        )


print("\n")
print("=" * 70)
print("ALL EXPERIMENTS COMPLETED")
print("=" * 70)