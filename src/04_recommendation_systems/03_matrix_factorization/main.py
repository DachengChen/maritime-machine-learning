from __future__ import annotations

import numpy as np


class MatrixFactorization:
    """SVD-based matrix factorization for vessel route recommendations.

    Decomposes the user–item matrix R ≈ U · Σ · Vt using truncated SVD,
    then reconstructs it to predict missing ratings.

    Parameters
    ----------
    n_factors : int
        Number of latent factors (rank of the decomposition). Default ``10``.
    """

    def __init__(self, n_factors: int = 10):
        self.n_factors = n_factors
        self._pred: np.ndarray | None = None
        self._matrix: np.ndarray | None = None

    def fit(self, matrix: np.ndarray) -> "MatrixFactorization":
        """Decompose the rating matrix using truncated SVD.

        Parameters
        ----------
        matrix : np.ndarray, shape (n_users, n_items)
            Observed ratings; 0 means unrated.
        """
        self._matrix = matrix.astype(float)
        U, sigma, Vt = np.linalg.svd(self._matrix, full_matrices=False)
        k = min(self.n_factors, len(sigma))
        self._pred = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        return float(self._pred[user_id, item_id])

    def recommend(self, user_id: int, n: int = 5) -> list[int]:
        """Return top-``n`` unrated item indices for ``user_id``."""
        scores = self._pred[user_id].copy()
        already_rated = self._matrix[user_id] > 0
        scores[already_rated] = -np.inf
        return list(np.argsort(scores)[::-1][:n])

    def evaluate(self, test_matrix: np.ndarray) -> dict:
        """RMSE over observed test ratings."""
        mask = test_matrix > 0
        diff = (self._pred[mask] - test_matrix[mask]) ** 2
        return {"rmse": round(float(np.sqrt(diff.mean())), 4)}


if __name__ == "__main__":
    np.random.seed(42)
    matrix = (np.random.rand(50, 20) > 0.7).astype(float)
    matrix *= np.random.randint(1, 6, size=matrix.shape)

    train = matrix.copy()
    test  = np.zeros_like(matrix)
    for uid in range(len(matrix)):
        rated = np.where(matrix[uid] > 0)[0]
        if len(rated) > 1:
            holdout = np.random.choice(rated, size=max(1, len(rated) // 5), replace=False)
            train[uid, holdout] = 0
            test[uid, holdout]  = matrix[uid, holdout]

    mf = MatrixFactorization(n_factors=10)
    mf.fit(train)
    print(mf.evaluate(test))
    print("recommendations for vessel 0:", mf.recommend(0, n=5))
