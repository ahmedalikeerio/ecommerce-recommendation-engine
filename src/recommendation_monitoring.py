from pathlib import Path
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = (
    BASE_DIR
    / "mlruns"
    / "320293588776741204"
    / "da3f7735781e4bada249729fe989f522"
    / "artifacts"
    / "model_artifacts"
)

OUTPUT_FILE = BASE_DIR / "monitoring" / "reports" / "recommendation_behavior.txt"


def load_artifacts():
    user_histories = joblib.load(
        ARTIFACT_DIR / "user_histories.joblib"
    )
    similar_items = joblib.load(
        ARTIFACT_DIR / "similar_items.joblib"
    )
    popularity_scores = joblib.load(
        ARTIFACT_DIR / "popularity_scores.joblib"
    )

    return user_histories, similar_items, popularity_scores


def generate_recommendations(
    user_histories,
    similar_items,
    popularity_scores,
    cf_weight=0.4,
    popularity_weight=0.6,
    n_recommendations=10,
):
    recommendations = {}

    for user_id, history in user_histories.items():

        if not history:
            recommendations[user_id] = []
            continue

        interacted_items = set(history)
        candidate_scores = {}

        for item_id in history:

            neighbors = similar_items.get(item_id, [])

            for similar_item, similarity in neighbors:

                if similar_item in interacted_items:
                    continue

                candidate_scores[similar_item] = (
                    candidate_scores.get(similar_item, 0.0)
                    + similarity
                )

        if not candidate_scores:
            recommendations[user_id] = []
            continue

        max_cf_score = max(candidate_scores.values())

        if max_cf_score > 0:
            candidate_scores = {
                item: score / max_cf_score
                for item, score in candidate_scores.items()
            }

        hybrid_scores = {}

        for item_id, cf_score in candidate_scores.items():

            popularity_score = popularity_scores.get(item_id, 0.0)

            hybrid_scores[item_id] = (
                cf_weight * cf_score
                + popularity_weight * popularity_score
            )

        ranked = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        recommendations[user_id] = [
            int(item_id)
            for item_id, _ in ranked[:n_recommendations]
        ]

    return recommendations


def calculate_metrics(recommendations, popularity_scores):

    total_users = len(recommendations)

    users_with_recommendations = sum(
        bool(items)
        for items in recommendations.values()
    )

    all_recommended_items = [
        item
        for items in recommendations.values()
        for item in items
    ]

    unique_recommended_items = set(all_recommended_items)

    catalog_size = len(popularity_scores)

    coverage = (
        len(unique_recommended_items) / catalog_size
        if catalog_size
        else 0
    )

    recommendation_counts = np.array(
        [len(items) for items in recommendations.values()]
    )

    average_recommendations = (
        recommendation_counts.mean()
        if len(recommendation_counts)
        else 0
    )

    cold_start_users = sum(
        not history
        for history in recommendations.values()
    )

    # How concentrated recommendations are among popular items.
    popularity_values = [
        popularity_scores.get(item, 0)
        for item in all_recommended_items
    ]

    average_popularity = (
        np.mean(popularity_values)
        if popularity_values
        else 0
    )

    return {
        "total_users": total_users,
        "users_with_recommendations": users_with_recommendations,
        "recommendation_coverage": coverage,
        "average_recommendations_per_user": average_recommendations,
        "cold_start_users": cold_start_users,
        "unique_recommended_items": len(unique_recommended_items),
        "catalog_size": catalog_size,
        "average_recommended_item_popularity": average_popularity,
    }


def save_report(metrics):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_FILE, "w") as f:

        f.write("Recommendation Behavior Monitoring\n")
        f.write("=" * 45 + "\n\n")

        f.write(
            f"Total users evaluated: "
            f"{metrics['total_users']}\n"
        )

        f.write(
            f"Users with recommendations: "
            f"{metrics['users_with_recommendations']}\n"
        )

        f.write(
            f"Recommendation coverage: "
            f"{metrics['recommendation_coverage']:.2%}\n"
        )

        f.write(
            f"Average recommendations per user: "
            f"{metrics['average_recommendations_per_user']:.2f}\n"
        )

        f.write(
            f"Cold-start users: "
            f"{metrics['cold_start_users']}\n"
        )

        f.write(
            f"Unique recommended items: "
            f"{metrics['unique_recommended_items']}\n"
        )

        f.write(
            f"Catalog size: "
            f"{metrics['catalog_size']}\n"
        )

        f.write(
            f"Average recommended item popularity: "
            f"{metrics['average_recommended_item_popularity']:.4f}\n"
        )


if __name__ == "__main__":

    print("Loading recommendation artifacts...")

    user_histories, similar_items, popularity_scores = (
        load_artifacts()
    )

    print("Generating recommendations...")

    recommendations = generate_recommendations(
        user_histories,
        similar_items,
        popularity_scores,
    )

    print("Calculating monitoring metrics...")

    metrics = calculate_metrics(
        recommendations,
        popularity_scores,
    )

    save_report(metrics)

    print("\nRecommendation monitoring completed.")
    print(f"Report saved to: {OUTPUT_FILE}")

    for name, value in metrics.items():
        print(f"{name}: {value}")