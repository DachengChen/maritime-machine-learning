"""
09_meta_learning.py
===================
Dummy reference implementations for meta-learning (ensemble) algorithms
applied to maritime AIS classification and numeric-prediction tasks.

All classes wrap scikit-learn estimators and expose the project-standard
``fit`` / ``predict`` / ``evaluate`` interface.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Chapter 11 – Bagging (dual use)
# ---------------------------------------------------------------------------

class BaggingClassification:
    """Bootstrap aggregating (Bagging) classifier (Chapter 11).

    Parameters
    ----------
    n_estimators : int
        Number of base estimators. Default ``10``.

    Example
    -------
    >>> from src.meta_learning import BaggingClassification
    >>> clf = BaggingClassification(n_estimators=50)
    >>> clf.fit(X_train, y_train)
    >>> print(clf.evaluate(X_test, y_test))
    """

    def __init__(self, n_estimators: int = 10, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = BaggingClassifier(
            n_estimators=n_estimators, random_state=random_state
        )

    def fit(self, X, y) -> "BaggingClassification":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _clf_metrics(y, self.predict(X))


class BaggingRegression:
    """Bootstrap aggregating (Bagging) regressor (Chapter 11).

    Example
    -------
    >>> from src.meta_learning import BaggingRegression
    >>> reg = BaggingRegression(n_estimators=50)
    >>> reg.fit(X_train, y_train)
    >>> print(reg.evaluate(X_test, y_test))
    """

    def __init__(self, n_estimators: int = 10, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = BaggingRegressor(
            n_estimators=n_estimators, random_state=random_state
        )

    def fit(self, X, y) -> "BaggingRegression":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _reg_metrics(y, self.predict(X))


# ---------------------------------------------------------------------------
# Chapter 11 – Boosting (dual use)
# ---------------------------------------------------------------------------

class BoostingClassification:
    """Gradient Boosting classifier (Chapter 11).

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        Shrinkage factor applied to each tree.

    Example
    -------
    >>> from src.meta_learning import BoostingClassification
    >>> clf = BoostingClassification(n_estimators=100, learning_rate=0.1)
    >>> clf.fit(X_train, y_train)
    >>> print(clf.evaluate(X_test, y_test))
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
    """Gradient Boosting regressor (Chapter 11).

    Example
    -------
    >>> from src.meta_learning import BoostingRegression
    >>> reg = BoostingRegression(n_estimators=100, learning_rate=0.1)
    >>> reg.fit(X_train, y_train)
    >>> print(reg.evaluate(X_test, y_test))
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


# ---------------------------------------------------------------------------
# Chapter 11 – Random Forests (dual use)
# ---------------------------------------------------------------------------

class RandomForestClassification:
    """Random Forest classifier (Chapter 11).

    Parameters
    ----------
    n_estimators : int
        Number of trees. Default ``100``.

    Example
    -------
    >>> from src.meta_learning import RandomForestClassification
    >>> clf = RandomForestClassification(n_estimators=200)
    >>> clf.fit(X_train, y_train)
    >>> print(clf.evaluate(X_test, y_test))
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )

    def fit(self, X, y) -> "RandomForestClassification":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _clf_metrics(y, self.predict(X))

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_


class RandomForestRegression:
    """Random Forest regressor (Chapter 11).

    Example
    -------
    >>> from src.meta_learning import RandomForestRegression
    >>> reg = RandomForestRegression(n_estimators=200)
    >>> reg.fit(X_train, y_train)
    >>> print(reg.evaluate(X_test, y_test))
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )

    def fit(self, X, y) -> "RandomForestRegression":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _reg_metrics(y, self.predict(X))

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_
