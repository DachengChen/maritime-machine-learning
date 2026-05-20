from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs


class KMeansClustering:
    """k-means clustering for vessel trajectory segmentation.

    Parameters
    ----------
    n_clusters : int
        Number of clusters *k*. Default ``5``.
    random_state : int
        Reproducibility seed. Default ``42``.
    """

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._model = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init="auto"
        )

    def fit(self, X) -> "KMeansClustering":
        self._model.fit(X)
        return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def fit_predict(self, X) -> np.ndarray:
        return self._model.fit_predict(X)

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self._model.cluster_centers_

    @property
    def inertia_(self) -> float:
        return float(self._model.inertia_)

    def evaluate(self, X) -> dict:
        """Return inertia and silhouette score for the fitted clustering."""
        labels = self.predict(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")
        return {
            "inertia": round(self.inertia_, 4),
            "silhouette": round(float(sil), 4),
        }


if __name__ == "__main__":
    X, _ = make_blobs(n_samples=500, centers=5, n_features=4, random_state=42)

    km = KMeansClustering(n_clusters=5)
    km.fit(X)
    print(km.evaluate(X))
