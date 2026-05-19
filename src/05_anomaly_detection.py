"""
anomaly_detection.py
====================
Identify anomalous vessel behaviour in AIS position streams.

Two complementary approaches are provided:

1. **Isolation Forest** – unsupervised, density-based anomaly scorer.
   Works well for detecting unusual combinations of speed, course, and
   position without requiring labelled training data.

2. **DBSCAN Trajectory Clustering** – groups position reports into dense
   spatial clusters.  Points labelled ``-1`` (noise) are flagged as
   anomalous.  Useful for detecting vessels deviating from established
   route corridors.

Usage example
-------------
>>> from src.anomaly_detection import IsolationForestDetector
>>> detector = IsolationForestDetector(contamination=0.02)
>>> detector.fit(X_train)
>>> labels = detector.predict(X_test)   # -1 = anomaly, 1 = normal
>>> scores = detector.score_samples(X_test)   # lower = more anomalous
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Isolation Forest – general purpose anomaly detection
# ---------------------------------------------------------------------------

class IsolationForestDetector:
    """Anomaly detector based on Isolation Forest.

    Parameters
    ----------
    contamination:
        Expected proportion of anomalies in the data (0 < contamination ≤ 0.5).
    n_estimators:
        Number of isolation trees.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._pipeline: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("detector", model),
        ])

    def fit(self, X: pd.DataFrame | np.ndarray) -> "IsolationForestDetector":
        """Fit the Isolation Forest on *X*.

        Parameters
        ----------
        X:
            Feature matrix representing normal (or mixed) vessel behaviour.

        Returns
        -------
        self
        """
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict anomaly labels.

        Returns
        -------
        np.ndarray
            Array of ``1`` (normal) or ``-1`` (anomaly) for each row.
        """
        if self._pipeline is None:
            raise RuntimeError("Detector has not been fitted yet. Call fit() first.")
        return self._pipeline.predict(X)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return the raw anomaly score for each sample.

        Lower (more negative) scores indicate greater anomalousness.
        """
        if self._pipeline is None:
            raise RuntimeError("Detector has not been fitted yet. Call fit() first.")
        scaler = self._pipeline.named_steps["scaler"]
        detector = self._pipeline.named_steps["detector"]
        X_scaled = scaler.transform(X)
        return detector.score_samples(X_scaled)

    def flag_anomalies(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "anomaly",
        score_col: str = "anomaly_score",
    ) -> pd.DataFrame:
        """Add anomaly label and score columns to *df* in-place.

        Parameters
        ----------
        df:
            Data frame containing *feature_cols*.
        feature_cols:
            Feature columns to feed into the model.
        label_col:
            Name of the column that will hold the ``1`` / ``-1`` labels.
        score_col:
            Name of the column that will hold the anomaly scores.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with *label_col* and *score_col* appended.
        """
        df = df.copy()
        X = df[feature_cols].dropna()
        df.loc[X.index, label_col] = self.predict(X)
        df.loc[X.index, score_col] = self.score_samples(X)
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Serialise the fitted detector pipeline to *filepath*."""
        if self._pipeline is None:
            raise RuntimeError("Nothing to save – detector has not been fitted.")
        joblib.dump(self._pipeline, filepath)

    @classmethod
    def load(cls, filepath: str) -> "IsolationForestDetector":
        """Restore a saved detector from *filepath*."""
        instance = cls.__new__(cls)
        instance._pipeline = joblib.load(filepath)
        return instance


# ---------------------------------------------------------------------------
# DBSCAN Trajectory Clustering – route-corridor anomaly detection
# ---------------------------------------------------------------------------

class DBSCANAnomalyDetector:
    """Spatial anomaly detection via DBSCAN clustering.

    Positions that fall outside dense route corridors (DBSCAN label ``-1``)
    are flagged as anomalous.

    Parameters
    ----------
    eps:
        Maximum distance between two samples for one to be considered in the
        neighbourhood of the other (in degrees by default; scale the
        coordinates if you want metric distances).
    min_samples:
        Minimum number of samples in a neighbourhood to form a core point.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 10) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self._scaler: StandardScaler | None = None
        self._dbscan: DBSCAN | None = None

    def fit_predict(
        self,
        df: pd.DataFrame,
        lat_col: str = "lat",
        lon_col: str = "lon",
        label_col: str = "cluster",
    ) -> pd.DataFrame:
        """Cluster position reports and flag noise as anomalies.

        Parameters
        ----------
        df:
            Data frame with latitude and longitude columns.
        lat_col, lon_col:
            Column names for coordinates.
        label_col:
            Column name to store DBSCAN cluster labels (``-1`` = anomaly).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with *label_col* appended.
        """
        df = df.copy()
        coords = df[[lat_col, lon_col]].dropna().values

        self._scaler = StandardScaler()
        coords_scaled = self._scaler.fit_transform(coords)

        self._dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = self._dbscan.fit_predict(coords_scaled)

        valid_idx = df[[lat_col, lon_col]].dropna().index
        df.loc[valid_idx, label_col] = labels
        return df

    @staticmethod
    def is_anomaly(df: pd.DataFrame, label_col: str = "cluster") -> pd.Series:
        """Return a boolean mask where ``True`` means the point is anomalous."""
        return df[label_col] == -1
