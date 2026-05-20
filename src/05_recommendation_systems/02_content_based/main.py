from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class ContentBasedRecommender:
    """Content-based recommender for vessel route / port recommendations.

    Builds item profiles from feature vectors and recommends items similar
    to those a user has previously rated highly.

    Parameters
    ----------
    top_n : int
        Number of similar items returned per query. Default ``5``.
    """

    def __init__(self, top_n: int = 5):
        self.top_n = top_n
        self._item_profiles: np.ndarray | None = None
        self._sim_matrix: np.ndarray | None = None

    def fit(self, item_features: np.ndarray) -> "ContentBasedRecommender":
        """Build normalised item profiles and pre-compute similarity matrix.

        Parameters
        ----------
        item_features : np.ndarray, shape (n_items, n_features)
            Feature matrix — one row per item (e.g. port: avg_draught,
            avg_loa, cargo_type_flags, region_encoding …).
        """
        self._item_profiles = normalize(item_features.astype(float))
        self._sim_matrix    = cosine_similarity(self._item_profiles)
        return self

    def recommend(self, user_ratings: np.ndarray, n: int | None = None) -> list[int]:
        """Return item indices recommended for a user.

        Constructs a user profile as the rating-weighted average of item
        profiles, then ranks unrated items by cosine similarity to that
        profile.

        Parameters
        ----------
        user_ratings : np.ndarray, shape (n_items,)
            Known ratings for each item; 0 means unrated.
        n : int or None
            Number of recommendations. Defaults to ``self.top_n``.
        """
        n = n or self.top_n
        rated = user_ratings > 0
        if not rated.any():
            # No history — fall back to globally most popular (highest index)
            return list(range(min(n, len(user_ratings))))

        weights = user_ratings[rated] / user_ratings[rated].sum()
        user_profile = (self._item_profiles[rated] * weights[:, None]).sum(axis=0)
        scores = self._item_profiles @ user_profile

        # Mask already-rated items
        scores[rated] = -np.inf
        return list(np.argsort(scores)[::-1][:n])

    def similar_items(self, item_id: int, n: int | None = None) -> list[int]:
        """Return the ``n`` most similar items to ``item_id``."""
        n = n or self.top_n
        scores = self._sim_matrix[item_id].copy()
        scores[item_id] = -1  # exclude self
        return list(np.argsort(scores)[::-1][:n])

    def evaluate(self, rating_matrix: np.ndarray, test_matrix: np.ndarray, n: int = 5) -> dict:
        """Precision@N and Recall@N averaged over all users.

        Parameters
        ----------
        rating_matrix : np.ndarray, shape (n_users, n_items)
            Training ratings (used to build user profiles).
        test_matrix : np.ndarray, shape (n_users, n_items)
            Held-out ratings used as ground truth.
        """
        precisions, recalls = [], []
        for uid in range(len(rating_matrix)):
            relevant = set(np.where(test_matrix[uid] > 0)[0])
            if not relevant:
                continue
            recs = set(self.recommend(rating_matrix[uid], n=n))
            hits = len(recs & relevant)
            precisions.append(hits / n)
            recalls.append(hits / len(relevant))
        return {
            f"precision@{n}": round(float(np.mean(precisions)), 4),
            f"recall@{n}":    round(float(np.mean(recalls)), 4),
        }


if __name__ == "__main__":
    np.random.seed(42)

    N_USERS, N_ITEMS, N_FEATURES = 50, 20, 8

    # Item features: simulate port attributes
    # (avg draught, avg LOA, container flag, tanker flag, bulk flag,
    #  ferry flag, lat-bin, lon-bin)
    item_features = np.random.rand(N_ITEMS, N_FEATURES)

    # User–item rating matrix (0 = unvisited)
    matrix = (np.random.rand(N_USERS, N_ITEMS) > 0.7).astype(float)
    matrix *= np.random.randint(1, 6, size=matrix.shape)

    # Train / test split: hold out 20 % of each user's ratings
    train = matrix.copy()
    test  = np.zeros_like(matrix)
    for uid in range(N_USERS):
        rated = np.where(matrix[uid] > 0)[0]
        if len(rated) > 1:
            holdout = np.random.choice(rated, size=max(1, len(rated) // 5), replace=False)
            train[uid, holdout] = 0
            test[uid, holdout]  = matrix[uid, holdout]

    rec = ContentBasedRecommender(top_n=5)
    rec.fit(item_features)

    print("metrics:                  ", rec.evaluate(train, test, n=5))
    print("recommendations vessel 0: ", rec.recommend(train[0]))
    print("ports similar to port 3:  ", rec.similar_items(3))
