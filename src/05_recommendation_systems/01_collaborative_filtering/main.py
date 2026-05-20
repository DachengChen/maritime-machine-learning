from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCF:
    """User-based collaborative filtering for voyage/route recommendations.

    Builds a user–item rating matrix and recommends items to a user based
    on ratings given by the most similar users (nearest neighbours).

    Parameters
    ----------
    k : int
        Number of similar users to consider. Default ``5``.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self._matrix: np.ndarray | None = None
        self._sim: np.ndarray | None = None

    def fit(self, matrix: np.ndarray) -> "UserBasedCF":
        """Fit on a user–item rating matrix.

        Parameters
        ----------
        matrix : np.ndarray, shape (n_users, n_items)
            Observed ratings; 0 means unrated.
        """
        self._matrix = matrix.astype(float)
        self._sim = cosine_similarity(matrix)
        return self

    def recommend(self, user_id: int, n: int = 5) -> list[int]:
        """Return top-``n`` item indices recommended for ``user_id``."""
        sim_scores = self._sim[user_id].copy()
        sim_scores[user_id] = -1  # exclude self

        top_k = np.argsort(sim_scores)[::-1][: self.k]
        scores = self._sim[user_id, top_k] @ self._matrix[top_k]

        # Mask already-rated items
        already_rated = self._matrix[user_id] > 0
        scores[already_rated] = -np.inf

        return list(np.argsort(scores)[::-1][:n])

    def evaluate(self, test_matrix: np.ndarray, n: int = 5) -> dict:
        """Precision@N and Recall@N averaged over all users."""
        precisions, recalls = [], []
        for uid in range(len(test_matrix)):
            relevant = set(np.where(test_matrix[uid] > 0)[0])
            if not relevant:
                continue
            recs = set(self.recommend(uid, n))
            hits = len(recs & relevant)
            precisions.append(hits / n)
            recalls.append(hits / len(relevant))
        return {
            f"precision@{n}": round(float(np.mean(precisions)), 4),
            f"recall@{n}":    round(float(np.mean(recalls)), 4),
        }


if __name__ == "__main__":
    np.random.seed(42)
    # 50 vessels × 20 routes; ratings 1–5, 0 = unvisited
    matrix = (np.random.rand(50, 20) > 0.7).astype(float)
    matrix *= np.random.randint(1, 6, size=matrix.shape)

    train = matrix.copy()
    test  = np.zeros_like(matrix)
    # Hold out 20% of ratings for evaluation
    for uid in range(len(matrix)):
        rated = np.where(matrix[uid] > 0)[0]
        if len(rated) > 1:
            holdout = np.random.choice(rated, size=max(1, len(rated) // 5), replace=False)
            train[uid, holdout] = 0
            test[uid, holdout]  = matrix[uid, holdout]

    cf = UserBasedCF(k=5)
    cf.fit(train)
    print(cf.evaluate(test, n=5))
    print("recommendations for vessel 0:", cf.recommend(0, n=5))
