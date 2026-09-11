import pandas as pd

from src.baseline import (
    most_viewed_baseline,
    most_engaged_baseline,
    most_purchased_baseline,
    most_added_to_cart_baseline,
)


def test_most_viewed_baseline():
    train = pd.DataFrame({
        "item_id": [1, 1, 2, 2, 3],
        "most_viewed": [10, 5, 20, 2, 8],
    })

    result = most_viewed_baseline(train, k=2)

    assert result == [2, 1]


def test_most_engaged_baseline():
    train = pd.DataFrame({
        "item_id": [1, 1, 2, 2, 3],
        "interaction_score": [5, 3, 12, 2, 4],
    })

    result = most_engaged_baseline(train, k=2)

    assert result == [2, 1]


def test_most_purchased_baseline():
    train = pd.DataFrame({
        "item_id": [1, 2, 2, 3, 3],
        "purchased": [1, 2, 3, 1, 4],
    })

    result = most_purchased_baseline(train, k=2)

    assert result == [2, 3]


def test_most_added_to_cart_baseline():
    train = pd.DataFrame({
        "item_id": [1, 1, 2, 2, 3],
        "added_to_cart": [2, 3, 5, 1, 2],
    })

    result = most_added_to_cart_baseline(train, k=2)

    assert result == [2, 1]