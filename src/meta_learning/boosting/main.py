from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


def _clf_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


def _reg_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(rmse, 4),
        "r2": round(r2_score(y_true, y_pred), 4),
    }


class BoostingClassification:
    """Gradient Boosting classifier.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds. Default ``100``.
    learning_rate : float
        Shrinkage factor. Default ``0.1``.
    max_depth : int
        Maximum tree depth. Default ``3``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )

    def fit(self, X, y) -> "BoostingClassification":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _clf_metrics(y, self.predict(X))


class BoostingRegression:
    """Gradient Boosting regressor.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds. Default ``100``.
    learning_rate : float
        Shrinkage factor. Default ``0.1``.
    max_depth : int
        Maximum tree depth. Default ``3``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self._model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )

    def fit(self, X, y) -> "BoostingRegression":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _reg_metrics(y, self.predict(X))


if __name__ == "__main__":
    # Classification demo
    Xc, yc = make_classification(n_samples=500, n_features=10, random_state=42)
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=42)
    clf = BoostingClassification(n_estimators=100, learning_rate=0.1)
    clf.fit(Xc_tr, yc_tr)
    print("classifier:", clf.evaluate(Xc_te, yc_te))

    # Regression demo
    Xr, yr = make_regression(n_samples=500, n_features=10, noise=0.5, random_state=42)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    reg = BoostingRegression(n_estimators=100, learning_rate=0.1)
    reg.fit(Xr_tr, yr_tr)
    print("regressor: ", reg.evaluate(Xr_te, yr_te))
