from __future__ import annotations

import numpy as np
from sklearn.svm import SVC, SVR
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


class SVMClassifier:
    """Support Vector Machine classifier.

    Parameters
    ----------
    kernel : str
        Kernel type (``'rbf'``, ``'linear'``, …). Default ``'rbf'``.
    C : float
        Regularisation parameter. Default ``1.0``.
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
    """Support Vector Machine regressor.

    Parameters
    ----------
    kernel : str
        Kernel type. Default ``'rbf'``.
    C : float
        Regularisation parameter. Default ``1.0``.
    epsilon : float
        Epsilon-tube width. Default ``0.1``.
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


if __name__ == "__main__":
    # Classification demo
    Xc, yc = make_classification(n_samples=300, n_features=10, random_state=42)
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=42)
    clf = SVMClassifier(kernel="rbf", C=1.0)
    clf.fit(Xc_tr, yc_tr)
    print("classifier:", clf.evaluate(Xc_te, yc_te))

    # Regression demo
    Xr, yr = make_regression(n_samples=300, n_features=10, noise=0.5, random_state=42)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    reg = SVMRegressor(kernel="rbf", C=1.0)
    reg.fit(Xr_tr, yr_tr)
    print("regressor: ", reg.evaluate(Xr_te, yr_te))
