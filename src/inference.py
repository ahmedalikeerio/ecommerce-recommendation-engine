from pathlib import Path

import joblib


class RecommendationService:
    """
    Production inference service for the e-commerce
    hybrid recommendation model.
    """

    def __init__(
        self,
        artifact_dir,
        cf_weight=0.4,
        popularity_weight=0.6,
    ):
        self.artifact_dir = Path(artifact_dir)

        self.cf_weight = cf_weight
        self.popularity_weight = popularity_weight

        self.user_histories = None
        self.similar_items = None
        self.popularity_scores = None

        self.load_artifacts()

    # ========================================================
    # Load Artifacts
    # ========================================================

    def load_artifacts(self):
        """Load trained recommendation artifacts once."""

        if not self.artifact_dir.exists():
            raise FileNotFoundError(
                f"Model artifacts not found at: "
                f"{self.artifact_dir}"
            )

        self.user_histories = joblib.load(
            self.artifact_dir / "user_histories.joblib"
        )

        self.similar_items = joblib.load(
            self.artifact_dir / "similar_items.joblib"
        )

        self.popularity_scores = joblib.load(
            self.artifact_dir / "popularity_scores.joblib"
        )

    # ========================================================
    # Recommendation
    # ========================================================

    def recommend(
        self,
        user_id,
        n_recommendations=10,
    ):
        """
        Generate hybrid recommendations for a user.
        """

        history = self.user_histories.get(
            user_id,
            []
        )

        if not history:
            return []

        interacted_items = set(history)

        candidate_scores = {}

        # ----------------------------------------------------
        # Collaborative Filtering
        # ----------------------------------------------------

        for item_id in history:

            neighbors = self.similar_items.get(
                item_id,
                []
            )

            for similar_item, similarity in neighbors:

                if similar_item in interacted_items:
                    continue

                candidate_scores[similar_item] = (
                    candidate_scores.get(
                        similar_item,
                        0.0
                    )
                    + similarity
                )

        if not candidate_scores:
            return []

        # ----------------------------------------------------
        # Normalize CF scores
        # ----------------------------------------------------

        max_cf_score = max(
            candidate_scores.values()
        )

        if max_cf_score > 0:

            candidate_scores = {
                item: score / max_cf_score
                for item, score
                in candidate_scores.items()
            }

        # ----------------------------------------------------
        # Hybrid scoring
        # ----------------------------------------------------

        hybrid_scores = {}

        for item_id, cf_score in (
            candidate_scores.items()
        ):

            popularity_score = (
                self.popularity_scores.get(
                    item_id,
                    0.0
                )
            )

            hybrid_scores[item_id] = (
                self.cf_weight * cf_score
                +
                self.popularity_weight
                * popularity_score
            )

        # ----------------------------------------------------
        # Rank
        # ----------------------------------------------------

        recommendations = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            int(item_id)
            for item_id, _ in recommendations[
                :n_recommendations
            ]
        ]