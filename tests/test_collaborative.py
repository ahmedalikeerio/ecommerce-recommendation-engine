import numpy as np
import pandas as pd

from src.collaborative import (
    load_data,
    prepare_matrix,
    train_model,
)


def test_prepare_matrix():
    train = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "item_id": [101, 101, 101, 101, 101, 102, 102, 102, 102, 102],
        "interaction_count": [1] * 10,
        "viewed": [1] * 10,
        "added_to_cart": [0] * 10,
        "purchased": [0] * 10,
    })

    (
        filtered_train,
        user_item_matrix,
        item_user_matrix,
        user_to_index,
        item_to_index,
    ) = prepare_matrix(train)

    assert user_item_matrix.shape == (5, 2)
    assert item_user_matrix.shape == (2, 5)

    assert len(user_to_index) == 5
    assert len(item_to_index) == 2

    assert user_item_matrix.nnz == 10


def test_train_model():
    matrix = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ])

    model = train_model(
        matrix,
        n_similar_items=2,
    )

    assert model.n_neighbors == 3
    assert model.metric == "cosine"