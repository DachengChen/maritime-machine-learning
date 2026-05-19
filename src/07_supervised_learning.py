"""
07_supervised_learning.py
=========================
Dummy reference implementations for common supervised learning algorithms
applied to maritime AIS classification and numeric-prediction tasks.

Each class wraps a scikit-learn estimator and exposes the project-standard
``fit`` / ``predict`` / ``evaluate`` interface.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.base import BaseEstimator


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
# Chapter 3 – Nearest Neighbour Classification
# ---------------------------------------------------------------------------

class NearestNeighborClassifier:
    """k-Nearest Neighbours classifier (Chapter 3).

    Parameters
    ----------
    k : int
        Number of neighbours. Default ``5``.

    Example
    -------
    >>> from src.supervised_learning import NearestNeighborClassifier
    >>> clf = NearestNeighborClassifier(k=5)
    >>> clf.fit(X_train, y_train)
    >>> print(clf.evaluate(X_test, y_test))
    """

    def __init__(self, k: int = 5):
        self.k = k
        self._model = KNeighborsClassifier(n_neighbors=k)

    def fit(self, X, y) -> "NearestNeighborClassifier":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _clf_metrics(y, self.predict(X))


# ---------------------------------------------------------------------------
# Chapter 4 – Naïve Bayes Classification
# ---------------------------------------------------------------------------

class NaiveBayesClassifier:
    """Gaussian Naïve Bayes classifier (Chapter 4).

    Example
    -------
    >>> from src.supervised_learning import NaiveBayesClassifier
    >>> clf = NaiveBayesClassifier()
    >>> clf.fit(X_train, y_train)
    >>> print(clf.evaluate(X_test, y_test))
    """

    def __init__(self):
        self._model = GaussianNB()

    def fit(self, X, y) -> "NaiveBayesClassifier":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _clf_metrics(y, self.predict(X))


# ---------------------------------------------------------------------------
# Chapter 5 – Decision Trees
# ---------------------------------------------------------------------------

class DecisionTreeClassification:
    """CART decision-tree classifier (Chapter 5).

    Parameters
    ----------
    max_depth : int or None
        Maximum tree depth. ``None`` grows fully.

    Example
    -------
    >>> from src.supervised_learning import DecisionTreeClassification
    >>> clf = DecisionTreeClassification(max_depth=6)
    >>> clf.fit(X_train, y_train)
    >>> print(clf.evaluate(X_test, y_test))
    """

    def __init__(self, max_depth: int | None = None):
        self.max_depth = max_depth
        self._model = DecisionTreeClassifier(max_depth=max_depth)

    def fit(self, X, y) -> "DecisionTreeClassification":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _clf_metrics(y, self.predict(X))


# ---------------------------------------------------------------------------
# Chapter 5 – Classification Rule Learners (1R / RIPPER stub)
# ---------------------------------------------------------------------------

class ClassificationRuleLearner:
    """Simple one-rule (1R) classification rule learner (Chapter 5).

    Trains a depth-1 decision tree as a stand-in for a proper rule-induction
    algorithm such as RIPPER.  Replace ``self._model`` with a ``wittgenstein``
    ``RIPPER`` instance for production use.

    Example
    -------
    >>> from src.supervised_learning import ClassificationRuleLearner
    >>> clf = ClassificationRuleLearner()
    >>> clf.fit(X_train, y_train)
    >>> print(clf.evaluate(X_test, y_test))
    """

    def __init__(self):
        self._model = DecisionTreeClassifier(max_depth=1)

    def fit(self, X, y) -> "ClassificationRuleLearner":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _clf_metrics(y, self.predict(X))


# ---------------------------------------------------------------------------
# Chapter 6 – Linear Regression
# ---------------------------------------------------------------------------

class LinearRegressionModel:
    """Ordinary least-squares linear regression (Chapter 6).

    Example
    -------
    >>> from src.supervised_learning import LinearRegressionModel
    >>> reg = LinearRegressionModel()
    >>> reg.fit(X_train, y_train)
    >>> print(reg.evaluate(X_test, y_test))
    """

    def __init__(self):
        self._model = LinearRegression()

    def fit(self, X, y) -> "LinearRegressionModel":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _reg_metrics(y, self.predict(X))


# ---------------------------------------------------------------------------
# Chapter 6 – Regression Trees
# ---------------------------------------------------------------------------

class RegressionTree:
    """CART regression tree (Chapter 6).

    Parameters
    ----------
    max_depth : int or None
        Maximum tree depth. Default ``None``.

    Example
    -------
    >>> from src.supervised_learning import RegressionTree
    >>> reg = RegressionTree(max_depth=5)
    >>> reg.fit(X_train, y_train)
    >>> print(reg.evaluate(X_test, y_test))
    """

    def __init__(self, max_depth: int | None = None):
        self.max_depth = max_depth
        self._model = DecisionTreeRegressor(max_depth=max_depth)

    def fit(self, X, y) -> "RegressionTree":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _reg_metrics(y, self.predict(X))


# ---------------------------------------------------------------------------
# Chapter 6 – Model Trees (M5-style stub)
# ---------------------------------------------------------------------------

class ModelTree:
    """Model tree stub (Chapter 6).

    Uses a shallow regression tree as a stand-in for a true M5 model tree
    (piecewise-linear leaves).  Swap in ``m5py.M5Prime`` for production use.

    Example
    -------
    >>> from src.supervised_learning import ModelTree
    >>> reg = ModelTree(max_depth=4)
    >>> reg.fit(X_train, y_train)
    >>> print(reg.evaluate(X_test, y_test))
    """

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self._model = DecisionTreeRegressor(max_depth=max_depth)

    def fit(self, X, y) -> "ModelTree":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _reg_metrics(y, self.predict(X))


# ---------------------------------------------------------------------------
# Chapter 7 – Neural Networks (dual use)
# ---------------------------------------------------------------------------

class NeuralNetworkClassifier:
    """Multi-layer perceptron classifier (Chapter 7).

    Example
    -------
    >>> from src.supervised_learning import NeuralNetworkClassifier
    >>> clf = NeuralNetworkClassifier(hidden_layer_sizes=(64, 32))
    >>> clf.fit(X_train, y_train)
    >>> print(clf.evaluate(X_test, y_test))
    """

    def __init__(self, hidden_layer_sizes: tuple = (100,), max_iter: int = 200):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self._model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42
        )

    def fit(self, X, y) -> "NeuralNetworkClassifier":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _clf_metrics(y, self.predict(X))


class NeuralNetworkRegressor:
    """Multi-layer perceptron regressor (Chapter 7).

    Example
    -------
    >>> from src.supervised_learning import NeuralNetworkRegressor
    >>> reg = NeuralNetworkRegressor(hidden_layer_sizes=(64, 32))
    >>> reg.fit(X_train, y_train)
    >>> print(reg.evaluate(X_test, y_test))
    """

    def __init__(self, hidden_layer_sizes: tuple = (100,), max_iter: int = 200):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self._model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42
        )

    def fit(self, X, y) -> "NeuralNetworkRegressor":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _reg_metrics(y, self.predict(X))


# ---------------------------------------------------------------------------
# Chapter 7 – Support Vector Machines (dual use)
# ---------------------------------------------------------------------------

class SVMClassifier:
    """Support Vector Machine classifier (Chapter 7).

    Example
    -------
    >>> from src.supervised_learning import SVMClassifier
    >>> clf = SVMClassifier(kernel="rbf", C=1.0)
    >>> clf.fit(X_train, y_train)
    >>> print(clf.evaluate(X_test, y_test))
    """

    def __init__(self, kernel: str = "rbf", C: float = 1.0):
        self.kernel = kernel
        self.C = C
        self._model = SVC(kernel=kernel, C=C)

    def fit(self, X, y) -> "SVMClassifier":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _clf_metrics(y, self.predict(X))


class SVMRegressor:
    """Support Vector Machine regressor (Chapter 7).

    Example
    -------
    >>> from src.supervised_learning import SVMRegressor
    >>> reg = SVMRegressor(kernel="rbf", C=1.0)
    >>> reg.fit(X_train, y_train)
    >>> print(reg.evaluate(X_test, y_test))
    """

    def __init__(self, kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self._model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    def fit(self, X, y) -> "SVMRegressor":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _reg_metrics(y, self.predict(X))
