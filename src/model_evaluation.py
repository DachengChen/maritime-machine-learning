"""
model_evaluation.py
===================
Shared evaluation utilities for regression and classification models.

Functions
---------
regression_metrics        – MAE, RMSE, R², and MAPE for a regression task.
classification_metrics    – Accuracy, Precision, Recall, F1 for a classifier.
plot_regression_residuals – Scatter plot of predicted vs. actual values.
plot_confusion_matrix     – Heatmap confusion matrix.
plot_feature_importances  – Horizontal bar chart of feature importances.
cross_validate_regression – k-Fold CV wrapper returning mean ± std of key metrics.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """Compute standard regression evaluation metrics.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Model predictions.
    prefix:
        Optional string prepended to each metric key (e.g. ``"train_"``).

    Returns
    -------
    dict
        Keys: ``mae``, ``rmse``, ``r2``, ``mape`` (with optional prefix).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error – guard against zero division
    nonzero = y_true != 0
    if nonzero.any():
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    else:
        mape = float("nan")

    return {
        f"{prefix}mae": mae,
        f"{prefix}rmse": rmse,
        f"{prefix}r2": r2,
        f"{prefix}mape": mape,
    }


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    average: str = "macro",
    prefix: str = "",
) -> dict[str, float]:
    """Compute standard classification evaluation metrics.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    average:
        Averaging strategy for multi-class metrics (``"macro"`` / ``"weighted"``).
    prefix:
        Optional string prepended to each metric key.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1`` (with optional prefix).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        f"{prefix}f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_regression_residuals(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter plot of predicted versus actual values with a perfect-fit line.

    Parameters
    ----------
    y_true, y_pred:
        Ground-truth and predicted values.
    title, xlabel, ylabel:
        Axis labels.
    ax:
        Existing axes to draw on; a new figure is created if ``None``.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ax.scatter(y_true, y_pred, alpha=0.4, edgecolors="none", s=20)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    labels: list | None = None,
    title: str = "Confusion Matrix",
    ax: plt.Axes | None = None,
    normalize: bool = True,
) -> plt.Axes:
    """Plot a (optionally normalised) confusion matrix as a seaborn heatmap.

    Parameters
    ----------
    y_true, y_pred:
        Ground-truth and predicted labels.
    labels:
        List of class label names for the axes.
    title:
        Figure title.
    ax:
        Existing axes to draw on.
    normalize:
        If ``True``, normalise by the true-label count (row-wise).

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fmt = "d"
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.where(row_sums == 0, 0.0, cm / row_sums.astype(float))
        fmt = ".2f"

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        xticklabels=labels if labels is not None else "auto",
        yticklabels=labels if labels is not None else "auto",
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    return ax


def plot_feature_importances(
    importances: pd.Series,
    top_n: int = 20,
    title: str = "Feature Importances",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Horizontal bar chart of top *top_n* feature importances.

    Parameters
    ----------
    importances:
        :class:`pd.Series` with feature names as index and importance as values.
    top_n:
        Number of features to display.
    title:
        Chart title.
    ax:
        Existing axes to draw on.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, top_n // 2)))

    top = importances.nlargest(top_n).sort_values()
    ax.barh(top.index, top.values)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=9)
    return ax


# ---------------------------------------------------------------------------
# Cross-validation wrapper
# ---------------------------------------------------------------------------

def cross_validate_regression(
    estimator,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    cv: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Manual k-fold cross-validation returning per-fold regression metrics.

    Parameters
    ----------
    estimator:
        A scikit-learn compatible estimator with ``fit`` and ``predict``.
    X, y:
        Feature matrix and target vector.
    cv:
        Number of folds.
    shuffle:
        Whether to shuffle data before splitting.
    random_state:
        Random seed used when *shuffle* is ``True``.

    Returns
    -------
    pd.DataFrame
        One row per fold, columns: ``fold``, ``mae``, ``rmse``, ``r2``, ``mape``.
    """
    X = np.asarray(X) if isinstance(X, pd.DataFrame) else X
    y = np.asarray(y)

    kf = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    records = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        estimator.fit(X_tr, y_tr)
        y_pred = estimator.predict(X_val)
        metrics = regression_metrics(y_val, y_pred)
        metrics["fold"] = fold_idx
        records.append(metrics)

    df = pd.DataFrame(records).set_index("fold")
    return df
