from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


def _reg_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(rmse, 4),
        "r2": round(r2_score(y_true, y_pred), 4),
    }


class LinearRegressionModel:
    """Ordinary least-squares linear regression."""

    def __init__(self):
        self._model = LinearRegression()

    def fit(self, X, y) -> "LinearRegressionModel":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _reg_metrics(y, self.predict(X))


if __name__ == "__main__":
    X, y = make_regression(n_samples=500, n_features=10, noise=0.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = LinearRegressionModel()
    reg.fit(X_train, y_train)
    print(reg.evaluate(X_test, y_test))
