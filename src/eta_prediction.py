"""
eta_prediction.py
=================
Estimate the Estimated Time of Arrival (ETA) for a vessel voyage.

Approach
--------
ETA is framed as a **regression** problem: given a snapshot of the current
voyage state, predict the remaining travel time (in seconds) until arrival.

The default model is a Gradient Boosted Regressor (XGBoost) wrapped in a
scikit-learn ``Pipeline`` that standardises numerical features before fitting.

Usage example
-------------
>>> from src.eta_prediction import ETAPredictor
>>> model = ETAPredictor()
>>> model.fit(X_train, y_train)   # y: remaining seconds to arrival
>>> y_pred = model.predict(X_test)
>>> model.save("models/eta_model.joblib")
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ETAPredictor:
    """Gradient-boosted regression model for ETA prediction.

    Parameters
    ----------
    n_estimators:
        Number of boosting stages.
    learning_rate:
        Shrinkage factor applied to each tree.
    max_depth:
        Maximum depth of individual regression estimators.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self._pipeline: Pipeline | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> Pipeline:
        regressor = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "ETAPredictor":
        """Fit the ETA model.

        Parameters
        ----------
        X:
            Feature matrix (rows = position snapshots).
        y:
            Target vector of remaining travel time in **seconds**.

        Returns
        -------
        self
        """
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict remaining travel time (seconds) for each row in *X*."""
        if self._pipeline is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self._pipeline.predict(X)

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> dict[str, float]:
        """Return MAE, RMSE, and R² on the given data split.

        Returns
        -------
        dict
            ``{"mae": ..., "rmse": ..., "r2": ...}`` – all values in seconds
            except R² which is dimensionless.
        """
        y_pred = self.predict(X)
        y = np.asarray(y)
        return {
            "mae": mean_absolute_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "r2": r2_score(y, y_pred),
        }

    def cross_validate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        cv: int = 5,
    ) -> dict[str, float]:
        """Run k-fold cross-validation and return mean ± std of MAE.

        Parameters
        ----------
        X, y:
            Full dataset.
        cv:
            Number of folds.

        Returns
        -------
        dict
            ``{"cv_mae_mean": ..., "cv_mae_std": ...}``
        """
        pipeline = self._build_pipeline()
        scores = cross_val_score(
            pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        return {
            "cv_mae_mean": -scores.mean(),
            "cv_mae_std": scores.std(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Serialise the fitted pipeline to *filepath* using joblib."""
        if self._pipeline is None:
            raise RuntimeError("Nothing to save – model has not been fitted.")
        joblib.dump(self._pipeline, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ETAPredictor":
        """Load a previously saved ETA model from *filepath*.

        Returns
        -------
        ETAPredictor
            Instance with the restored pipeline attached.
        """
        instance = cls.__new__(cls)
        instance._pipeline = joblib.load(filepath)
        return instance


# ---------------------------------------------------------------------------
# Target-engineering helper
# ---------------------------------------------------------------------------

def compute_remaining_time(df: pd.DataFrame, arrival_col: str = "arrival_time") -> pd.Series:
    """Compute the remaining travel time for each position report.

    Parameters
    ----------
    df:
        Data frame with ``timestamp`` and *arrival_col* columns (both
        ``datetime64`` / timezone-aware).
    arrival_col:
        Column name holding the known arrival timestamp.

    Returns
    -------
    pd.Series
        Remaining time in **seconds** (float).  Negative values (arrival
        already passed) are set to 0.
    """
    remaining = (df[arrival_col] - df["timestamp"]).dt.total_seconds()
    return remaining.clip(lower=0.0)
