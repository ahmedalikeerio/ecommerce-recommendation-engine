from pathlib import Path

import joblib

from collaborative import (
    load_data,
    prepare_matrix,
    train_model,
)

from hybrid import (
    build_popularity_scores,
    build_user_histories,
    build_similar_items,
)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

N_SIMILAR_ITEMS = 40


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("Loading training data...")

    train = load_data()

    print(
        f"Training interactions: "
        f"{len(train):,}"
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

    print(
        f"Users: {user_item_matrix.shape[0]:,}"
    )

    print(
        f"Items: {item_user_matrix.shape[0]:,}"
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
    # Build popularity scores
    # --------------------------------------------------------

    print("\nBuilding popularity scores...")

    popularity_scores = build_popularity_scores(
        train
    )

    # --------------------------------------------------------
    # Build user histories
    # --------------------------------------------------------

    print("\nBuilding user histories...")

    user_histories = build_user_histories(
        train
    )

    # --------------------------------------------------------
    # Build similarity graph
    # --------------------------------------------------------

    print("\nBuilding similar-item graph...")

    similar_items = build_similar_items(
        model,
        item_user_matrix,
        item_to_index,
    )

    # --------------------------------------------------------
    # Create model directory
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

    print("\nSaving model artifacts...")

    joblib.dump(
        model,
        MODEL_DIR / "cf_model.joblib"
    )

    joblib.dump(
        popularity_scores,
        MODEL_DIR / "popularity_scores.joblib"
    )

    joblib.dump(
        user_histories,
        MODEL_DIR / "user_histories.joblib"
    )

    joblib.dump(
        similar_items,
        MODEL_DIR / "similar_items.joblib"
    )

    joblib.dump(
        {
            "user_to_index": user_to_index,
            "item_to_index": item_to_index,
        },
        MODEL_DIR / "mappings.joblib"
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    joblib.dump(
        {
            "n_similar_items": N_SIMILAR_ITEMS,
            "cf_weight": 0.4,
            "popularity_weight": 0.6,
        },
        MODEL_DIR / "config.joblib"
    )

    print("\n" + "=" * 60)
    print("MODEL ARTIFACTS SAVED")
    print("=" * 60)

    for file in MODEL_DIR.iterdir():

        print(
            f"{file.name}: "
            f"{file.stat().st_size / (1024 ** 2):.2f} MB"
        )