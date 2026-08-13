from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"


# Minimum number of interactions required for an item
MIN_ITEM_INTERACTIONS = 5

# Number of similar items to retrieve
N_SIMILAR_ITEMS = 20

# Maximum number of items used by the first model
MAX_ITEMS = 50_000

INTERACTION_WEIGHTS = {
    "viewed": 1.0,
    "added_to_cart": 3.0,
    "purchased": 5.0
}


# ============================================================
# Load Data
# ============================================================

def load_data():

    train = pd.read_csv(TRAIN_FILE)

    return train


# ============================================================
# Prepare Item-User Matrix
# ============================================================

def prepare_matrix(
        train,
        view_weight=1.0,
        cart_weight=3.0,
     purchase_weight=5.0
     ):

    # Remove extremely rare items
    item_counts = train["item_id"].value_counts()

    valid_items = item_counts[
        item_counts >= MIN_ITEM_INTERACTIONS
    ].index

    train = train[
        train["item_id"].isin(valid_items)
    ].copy()

    # Keep the most active items for the first version
    
    top_items = (
        train
        .groupby("item_id")["interaction_count"]
        .sum()
        .sort_values(ascending=False)
        .head(MAX_ITEMS)
        .index
        )

    train = train[
        train["item_id"].isin(top_items)
    ].copy()

    # Create integer indices
    user_ids = train["user_id"].unique()
    item_ids = train["item_id"].unique()

    user_to_index = {
        user_id: index
        for index, user_id in enumerate(user_ids)
    }

    item_to_index = {
        item_id: index
        for index, item_id in enumerate(item_ids)
    }

    # Convert IDs to matrix indices
    rows = train["user_id"].map(user_to_index)
    cols = train["item_id"].map(item_to_index)

    raw_score = (
    train["viewed"] * view_weight
    + train["added_to_cart"] * cart_weight
    + train["purchased"] * purchase_weight
    )

    values = np.log1p(raw_score)


    # User-item matrix
    user_item_matrix = csr_matrix(
        (
            values,
            (rows, cols)
        ),
        shape=(
            len(user_ids),
            len(item_ids)
        )
    )

    # Convert to item-user matrix
    item_user_matrix = user_item_matrix.T.tocsr()

    return (
        train,
        user_item_matrix,
        item_user_matrix,
        user_to_index,
        item_to_index,
    )


# ============================================================
# Train Item-Item Model
# ============================================================

def train_model(item_user_matrix, n_similar_items =20):

    model = NearestNeighbors(
        n_neighbors=n_similar_items + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1
    )

    model.fit(item_user_matrix)

    return model


# ============================================================
# Build Similar Item Graph
# ============================================================

def build_similar_items(
    model,
    item_user_matrix,
    item_to_index,
    n_similar_items=N_SIMILAR_ITEMS
):

    index_to_item = {
        index: item_id
        for item_id, index in item_to_index.items()
    }

    similar_items = {}

    print("Building similar-item graph...")

    for item_index in range(
        item_user_matrix.shape[0]
    ):

        distances, indices = model.kneighbors(
            item_user_matrix[item_index],
            n_neighbors=n_similar_items + 1
        )

        item_id = index_to_item[item_index]

        similar_items[item_id] = []

        for distance, similar_index in zip(
            distances[0][1:],
            indices[0][1:]
        ):

            similar_item_id = (
                index_to_item[similar_index]
            )

            similarity = 1 - distance

            similar_items[item_id].append(
                (
                    similar_item_id,
                    similarity
                )
            )

    return similar_items
# ============================================================
# Recommend Items
# ============================================================

def recommend_for_user(
    user_id,
    train,
    model,
    user_to_index,
    item_to_index,
    n_recommendations=10
):

    if user_id not in user_to_index:
        return []

    # Reverse item mapping
    index_to_item = {
        index: item_id
        for item_id, index in item_to_index.items()
    }

    user_index = user_to_index[user_id]

    # Items user has already interacted with
    user_vector = train[
        train["user_id"] == user_id
    ]

    interacted_items = (
        user_vector["item_id"]
        .unique()
        .tolist()
    )

    candidate_scores = {}

    # Find similar products for each interacted item
    for item_id in interacted_items:

        if item_id not in item_to_index:
            continue

        item_index = item_to_index[item_id]

        distances, indices = model.kneighbors(
            model._fit_X[item_index],
            n_neighbors=N_SIMILAR_ITEMS + 1
        )

        for distance, similar_index in zip(
            distances[0],
            indices[0]
        ):

            similar_item = index_to_item[similar_index]

            # Don't recommend something user already interacted with
            if similar_item in interacted_items:
                continue

            similarity = 1 - distance

            candidate_scores[similar_item] = (
                candidate_scores.get(similar_item, 0)
                + similarity
            )

    # Rank candidates
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
# Main
# ============================================================

if __name__ == "__main__":

    print("Loading training data...")

    train = load_data()

    print(f"Original interactions: {len(train):,}")

    print("Preparing sparse matrix...")

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
        f"Matrix shape: {user_item_matrix.shape}"
    )

    print(
        f"Non-zero interactions: "
        f"{user_item_matrix.nnz:,}"
    )

    print("Training item-item model...")

    model = train_model(item_user_matrix)

    model = train_model(item_user_matrix)

# ============================================================
# Build Similar Item Graph
# ============================================================

    similar_items = build_similar_items(
        model,
        item_user_matrix,
        item_to_index,
        N_SIMILAR_ITEMS
    )

    # ============================================================
    # Save Similarity Graph for LTR
    # ============================================================

    print("Saving similarity graph...")

    similarity_rows = []

    for item_id, similar_list in similar_items.items():

        for similar_item_id, similarity in similar_list:

            similarity_rows.append({
                "item_id": item_id,
                "similar_item_id": similar_item_id,
                "similarity": similarity,
            })

    similarity_df = pd.DataFrame(
        similarity_rows
    )

    similarity_file = (
        BASE_DIR
        / "data"
        / "processed"
        / "similar_items.csv"
    )

    similarity_df.to_csv(
        similarity_file,
        index=False
    )

    print(
        f"Saved similarity graph to: "
        f"{similarity_file}"
    )

    # ============================================================
    # Test with one user
    # ============================================================

    # Test with one user
    example_user = train["user_id"].iloc[0]

    recommendations = recommend_for_user(
        example_user,
        train,
        model,
        user_to_index,
        item_to_index,
        n_recommendations=10
    )

    print("\nExample User:")
    print(example_user)

    print("\nRecommended Items:")
    print(recommendations)