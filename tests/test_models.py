"""
test_models.py
==============
Smoke tests for ETA prediction, destination prediction, anomaly detection,
and model evaluation.
"""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eta_prediction import ETAPredictor, compute_remaining_time
from src.destination_prediction import DestinationPredictor
from src.anomaly_detection import IsolationForestDetector, DBSCANAnomalyDetector
from src.model_evaluation import (
    regression_metrics,
    classification_metrics,
    plot_regression_residuals,
    plot_confusion_matrix,
    plot_feature_importances,
    cross_validate_regression,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)
N = 200


@pytest.fixture()
def regression_data():
    X = RNG.standard_normal((N, 5))
    y = 3 * X[:, 0] - 2 * X[:, 1] + RNG.standard_normal(N) * 0.5
    return X, y


@pytest.fixture()
def classification_data():
    X = RNG.standard_normal((N, 4))
    # Two classes: above / below mean of first feature
    labels = np.where(X[:, 0] > 0, "ROTTERDAM", "HAMBURG")
    return X, labels


# ---------------------------------------------------------------------------
# ETA Predictor
# ---------------------------------------------------------------------------

class TestETAPredictor:
    def test_fit_predict(self, regression_data):
        X, y = regression_data
        model = ETAPredictor(n_estimators=50)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (N,)

    def test_evaluate_returns_keys(self, regression_data):
        X, y = regression_data
        model = ETAPredictor(n_estimators=50)
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert {"mae", "rmse", "r2"} <= set(metrics.keys())

    def test_r2_positive_on_train(self, regression_data):
        X, y = regression_data
        model = ETAPredictor(n_estimators=50)
        model.fit(X, y)
        assert model.evaluate(X, y)["r2"] > 0

    def test_predict_before_fit_raises(self):
        model = ETAPredictor()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 5)))

    def test_save_load(self, regression_data, tmp_path):
        X, y = regression_data
        model = ETAPredictor(n_estimators=50)
        model.fit(X, y)
        path = str(tmp_path / "eta.joblib")
        model.save(path)
        loaded = ETAPredictor.load(path)
        np.testing.assert_allclose(model.predict(X), loaded.predict(X))

    def test_cross_validate(self, regression_data):
        X, y = regression_data
        model = ETAPredictor(n_estimators=50)
        result = model.cross_validate(X, y, cv=3)
        assert "cv_mae_mean" in result and result["cv_mae_mean"] >= 0


class TestComputeRemainingTime:
    def test_positive_remaining(self):
        df = pd.DataFrame({
            "timestamp":    pd.to_datetime(["2024-01-01 08:00"], utc=True),
            "arrival_time": pd.to_datetime(["2024-01-01 12:00"], utc=True),
        })
        result = compute_remaining_time(df)
        assert result.iloc[0] == pytest.approx(4 * 3600)

    def test_clipped_to_zero(self):
        df = pd.DataFrame({
            "timestamp":    pd.to_datetime(["2024-01-01 14:00"], utc=True),
            "arrival_time": pd.to_datetime(["2024-01-01 12:00"], utc=True),
        })
        result = compute_remaining_time(df)
        assert result.iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Destination Predictor
# ---------------------------------------------------------------------------

class TestDestinationPredictor:
    def test_fit_predict_string_labels(self, classification_data):
        X, y = classification_data
        model = DestinationPredictor(n_estimators=50)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (N,)
        assert set(preds).issubset({"ROTTERDAM", "HAMBURG"})

    def test_evaluate_returns_keys(self, classification_data):
        X, y = classification_data
        model = DestinationPredictor(n_estimators=50)
        model.fit(X, y)
        result = model.evaluate(X, y)
        assert {"accuracy", "f1_macro", "report"} <= set(result.keys())

    def test_accuracy_above_chance(self, classification_data):
        X, y = classification_data
        model = DestinationPredictor(n_estimators=50)
        model.fit(X, y)
        assert model.evaluate(X, y)["accuracy"] > 0.5

    def test_predict_proba_shape(self, classification_data):
        X, y = classification_data
        model = DestinationPredictor(n_estimators=50)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (N, 2)

    def test_classes_attribute(self, classification_data):
        X, y = classification_data
        model = DestinationPredictor(n_estimators=50)
        model.fit(X, y)
        assert set(model.classes_) == {"ROTTERDAM", "HAMBURG"}

    def test_save_load(self, classification_data, tmp_path):
        X, y = classification_data
        model = DestinationPredictor(n_estimators=50)
        model.fit(X, y)
        path = str(tmp_path / "dest.joblib")
        model.save(path)
        loaded = DestinationPredictor.load(path)
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))

    def test_feature_importances(self, classification_data):
        X, y = classification_data
        model = DestinationPredictor(n_estimators=50)
        model.fit(X, y)
        names = ["f0", "f1", "f2", "f3"]
        imp = model.feature_importances(names)
        assert len(imp) == 4
        assert imp.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------

class TestIsolationForestDetector:
    def test_fit_predict_labels(self):
        X = RNG.standard_normal((100, 3))
        det = IsolationForestDetector(contamination=0.05, n_estimators=50)
        det.fit(X)
        labels = det.predict(X)
        assert set(labels).issubset({-1, 1})

    def test_score_samples_shape(self):
        X = RNG.standard_normal((50, 3))
        det = IsolationForestDetector(n_estimators=50)
        det.fit(X)
        scores = det.score_samples(X)
        assert scores.shape == (50,)

    def test_fit_before_predict(self):
        det = IsolationForestDetector()
        with pytest.raises(RuntimeError):
            det.predict(np.zeros((5, 3)))

    def test_flag_anomalies(self):
        df = pd.DataFrame(RNG.standard_normal((60, 3)), columns=["a", "b", "c"])
        det = IsolationForestDetector(contamination=0.1, n_estimators=50)
        det.fit(df[["a", "b", "c"]].values)
        result = det.flag_anomalies(df, feature_cols=["a", "b", "c"])
        assert "anomaly" in result.columns
        assert "anomaly_score" in result.columns

    def test_save_load(self, tmp_path):
        X = RNG.standard_normal((60, 3))
        det = IsolationForestDetector(n_estimators=50)
        det.fit(X)
        path = str(tmp_path / "det.joblib")
        det.save(path)
        loaded = IsolationForestDetector.load(path)
        np.testing.assert_array_equal(det.predict(X), loaded.predict(X))


class TestDBSCANAnomalyDetector:
    def test_fit_predict_adds_cluster_col(self):
        df = pd.DataFrame({
            "lat": np.linspace(51, 54, 30),
            "lon": np.linspace(4, 10, 30),
        })
        det = DBSCANAnomalyDetector(eps=0.3, min_samples=3)
        result = det.fit_predict(df)
        assert "cluster" in result.columns

    def test_is_anomaly_returns_bool_series(self):
        df = pd.DataFrame({
            "lat": np.linspace(51, 54, 20),
            "lon": np.linspace(4, 10, 20),
            "cluster": [1] * 18 + [-1, -1],
        })
        mask = DBSCANAnomalyDetector.is_anomaly(df)
        assert mask.sum() == 2


# ---------------------------------------------------------------------------
# Model Evaluation utilities
# ---------------------------------------------------------------------------

class TestRegressionMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = regression_metrics(y, y)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)

    def test_keys_present(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = regression_metrics(y, y + 0.1)
        assert {"mae", "rmse", "r2", "mape"} <= set(metrics.keys())

    def test_prefix(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = regression_metrics(y, y, prefix="train_")
        assert "train_mae" in metrics


class TestClassificationMetrics:
    def test_perfect_prediction(self):
        y = np.array([0, 1, 0, 1])
        metrics = classification_metrics(y, y)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_keys_present(self):
        y = np.array([0, 1, 0, 1])
        metrics = classification_metrics(y, y)
        assert {"accuracy", "precision", "recall", "f1"} <= set(metrics.keys())


class TestPlotHelpers:
    def test_plot_regression_residuals(self, regression_data):
        import matplotlib
        matplotlib.use("Agg")
        X, y = regression_data
        ax = plot_regression_residuals(y, y + 0.1)
        assert ax is not None

    def test_plot_confusion_matrix(self):
        import matplotlib
        matplotlib.use("Agg")
        y = np.array(["A", "B", "A", "B", "A"])
        ax = plot_confusion_matrix(y, y)
        assert ax is not None

    def test_plot_feature_importances(self):
        import matplotlib
        matplotlib.use("Agg")
        imp = pd.Series({"f1": 0.4, "f2": 0.3, "f3": 0.2, "f4": 0.1})
        ax = plot_feature_importances(imp, top_n=3)
        assert ax is not None


class TestCrossValidateRegression:
    def test_returns_dataframe(self, regression_data):
        from sklearn.linear_model import Ridge
        X, y = regression_data
        result = cross_validate_regression(Ridge(), X, y, cv=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_fold_index(self, regression_data):
        from sklearn.linear_model import Ridge
        X, y = regression_data
        result = cross_validate_regression(Ridge(), X, y, cv=4)
        assert list(result.index) == [1, 2, 3, 4]
