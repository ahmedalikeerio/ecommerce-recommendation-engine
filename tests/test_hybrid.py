from src.hybrid import (
    build_popularity_scores,
    build_user_histories,
    recommend_hybrid,
)

import pandas as pd


def test_build_user_histories():
    train = pd.DataFrame({
        "user_id": [1, 1, 2],
        "item_id": [101, 102, 103],
    })

    result = build_user_histories(train)

    assert result == {
        1: [101, 102],
        2: [103],
    }


def test_build_popularity_scores():
    train = pd.DataFrame({
        "item_id": [101, 101, 102],
        "interaction_count": [10, 5, 2],
    })

    result = build_popularity_scores(train)

    assert set(result.keys()) == {101, 102}
    assert result[101] == 1.0
    assert result[102] == 0.0


def test_recommend_hybrid_excludes_interacted_items():
    user_histories = {
        1: [101, 102]
    }

    similar_items = {
        101: [
            (102, 0.9),
            (103, 0.8),
            (104, 0.7),
        ],
        102: [
            (101, 0.9),
            (104, 0.8),
            (105, 0.7),
        ],
    }

    popularity_scores = {
        103: 0.5,
        104: 1.0,
        105: 0.2,
    }

    result = recommend_hybrid(
        user_id=1,
        user_histories=user_histories,
        similar_items=similar_items,
        popularity_scores=popularity_scores,
        n_recommendations=3,
    )

    assert len(result) == 3
    assert 101 not in result
    assert 102 not in result