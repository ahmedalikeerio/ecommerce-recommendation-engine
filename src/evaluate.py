from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"


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


def evaluate_global_recommender(
    recommendations,
    train_data,
    test_data,
    k=10
):

    # Only evaluate users known during training
    train_users = set(train_data["user_id"])

    test_data = test_data[
        test_data["user_id"].isin(train_users)
    ]

    user_items = (
        test_data
        .groupby("user_id")["item_id"]
        .apply(set)
    )

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for relevant_items in user_items:

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

    return {
        "Precision@10": np.mean(precision_scores),
        "Recall@10": np.mean(recall_scores),
        "NDCG@10": np.mean(ndcg_scores),
        "Evaluated Users": len(user_items)
    }


if __name__ == "__main__":

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    # Baseline: most viewed
    recommendations = (
        train
        .groupby("item_id")["viewed"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .index
        .tolist()
    )

    results = evaluate_global_recommender(
        recommendations,
        train,
        test,
        k=10
    )

    print(results)