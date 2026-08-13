from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from collaborative import (
    load_data,
    prepare_matrix,
    train_model,
)

from evaluate_collaborative import (
    evaluate_collaborative,
)

VIEW_WEIGHT = 1.0
CART_WEIGHT = 3.0
PURCHASE_WEIGHT = 10.0

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"

EXPERIMENT_NAME = "ecommerce-recommendation"

K = 10
MAX_USERS = 10_000

MIN_ITEM_INTERACTIONS = 5
MAX_ITEMS = 50_000
N_SIMILAR_ITEMS = 20


# ============================================================
# MLflow Setup
# ============================================================

mlflow.set_tracking_uri(
    f"file://{BASE_DIR / 'mlruns'}"
)

mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# Load Data
# ============================================================

print("Loading data...")

train = load_data()

test = pd.read_csv(TEST_FILE)


# ============================================================
# Prepare Matrix
# ============================================================

print("Preparing matrix...")

(
    train,
    user_item_matrix,
    item_user_matrix,
    user_to_index,
    item_to_index,
) = prepare_matrix(train,
                   view_weight=VIEW_WEIGHT,
    cart_weight=CART_WEIGHT,
    purchase_weight=PURCHASE_WEIGHT,)


print(
    f"Users: {user_item_matrix.shape[0]:,}"
)

print(
    f"Items: {item_user_matrix.shape[0]:,}"
)

print(
    f"Interactions: {user_item_matrix.nnz:,}"
)


# ============================================================
# Train + Evaluate
# ============================================================

with mlflow.start_run(
    run_name="collaborative_v3_purchase_10"
):

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    mlflow.log_params({
        "model": "item-item-collaborative-filtering",
        "similarity": "cosine",
        "n_similar_items": N_SIMILAR_ITEMS,
        "min_item_interactions": MIN_ITEM_INTERACTIONS,
        "max_items": MAX_ITEMS,
        "k": K,
        "max_evaluation_users": MAX_USERS,

        # Current interaction weights
        "view_weight": VIEW_WEIGHT,
        "cart_weight": CART_WEIGHT,
        "purchase_weight": PURCHASE_WEIGHT,

        # Scaling
        "interaction_scaling": "log1p",
    })

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining model...")

    model = train_model(
        item_user_matrix
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\nEvaluating...")

    results = evaluate_collaborative(
        train=train,
        test=test,
        model=model,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        k=K,
        max_users=MAX_USERS,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mlflow.log_metrics({
        "precision_at_10": float(
            results["Precision@10"]
        ),
        "recall_at_10": float(
            results["Recall@10"]
        ),
        "ndcg_at_10": float(
            results["NDCG@10"]
        ),
        "evaluated_users": float(
            results["Evaluated Users"]
        ),
    })

    # --------------------------------------------------------
    # Tags
    # --------------------------------------------------------

    mlflow.set_tags({
        "model_type": "collaborative_filtering",
        "feedback_type": "implicit",
        "similarity_metric": "cosine",
        "dataset": "retailrocket",
    })

    print("\nResults:")

    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nMLflow run logged successfully.")