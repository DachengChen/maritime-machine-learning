"""
destination_prediction.py
=========================
Predict the next port-of-call (destination) for a vessel.

Approach
--------
Destination prediction is framed as a **multi-class classification** problem.
The default model is a Random Forest Classifier that maps a voyage-state
feature vector to one of the known destination labels.

An optional label-encoding step handles the free-text destination strings
produced by :func:`~src.data_preprocessing.normalise_destination`.

Usage example
-------------
>>> from src.destination_prediction import DestinationPredictor
>>> model = DestinationPredictor()
>>> model.fit(X_train, y_train)   # y: destination label strings or integers
>>> y_pred = model.predict(X_test)
>>> model.save("models/destination_model.joblib")
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DestinationPredictor:
    """Random-forest multi-class classifier for destination prediction.

    Parameters
    ----------
    n_estimators:
        Number of trees in the forest.
    max_depth:
        Maximum depth of each tree (``None`` = fully grown).
    min_samples_leaf:
        Minimum number of samples required to be at a leaf node.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._pipeline: Pipeline | None = None
        self._label_encoder: LabelEncoder | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> Pipeline:
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "DestinationPredictor":
        """Fit the destination classifier.

        String labels are automatically encoded with
        :class:`sklearn.preprocessing.LabelEncoder`.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Target destination labels (strings or integers).

        Returns
        -------
        self
        """
        y = np.asarray(y)
        if y.dtype.kind in ("U", "O"):   # string labels
            self._label_encoder = LabelEncoder()
            y_enc = self._label_encoder.fit_transform(y)
        else:
            y_enc = y

        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X, y_enc)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict destination labels.

        If labels were string-encoded during :meth:`fit`, the returned array
        contains the original string labels.
        """
        if self._pipeline is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        y_enc = self._pipeline.predict(X)
        if self._label_encoder is not None:
            return self._label_encoder.inverse_transform(y_enc)
        return y_enc

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return class probabilities for each input row."""
        if self._pipeline is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self._pipeline.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        """Array of class labels known to the model."""
        if self._pipeline is None:
            raise RuntimeError("Model has not been fitted yet.")
        enc_classes = self._pipeline.named_steps["classifier"].classes_
        if self._label_encoder is not None:
            return self._label_encoder.inverse_transform(enc_classes)
        return enc_classes

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> dict:
        """Return accuracy, macro-F1, and a full classification report.

        Returns
        -------
        dict
            ``{"accuracy": ..., "f1_macro": ..., "report": <string>}``
        """
        y_pred = self.predict(X)
        y_true = np.asarray(y)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "report": classification_report(y_true, y_pred, zero_division=0),
        }

    def cross_validate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        cv: int = 5,
    ) -> dict[str, float]:
        """Run k-fold cross-validation and return mean ± std of accuracy.

        Returns
        -------
        dict
            ``{"cv_accuracy_mean": ..., "cv_accuracy_std": ...}``
        """
        y = np.asarray(y)
        if y.dtype.kind in ("U", "O"):
            le = LabelEncoder()
            y = le.fit_transform(y)

        pipeline = self._build_pipeline()
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        return {
            "cv_accuracy_mean": scores.mean(),
            "cv_accuracy_std": scores.std(),
        }

    def feature_importances(self, feature_names: list[str]) -> pd.Series:
        """Return feature importances from the Random Forest.

        Parameters
        ----------
        feature_names:
            Names matching the columns used during training.

        Returns
        -------
        pd.Series
            Importances sorted descending.
        """
        if self._pipeline is None:
            raise RuntimeError("Model has not been fitted yet.")
        clf = self._pipeline.named_steps["classifier"]
        return (
            pd.Series(clf.feature_importances_, index=feature_names)
            .sort_values(ascending=False)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Serialise the fitted model to *filepath* using joblib."""
        if self._pipeline is None:
            raise RuntimeError("Nothing to save – model has not been fitted.")
        payload = {"pipeline": self._pipeline, "label_encoder": self._label_encoder}
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: str) -> "DestinationPredictor":
        """Load a saved destination model from *filepath*.

        Returns
        -------
        DestinationPredictor
            Instance with the restored pipeline and encoder attached.
        """
        payload = joblib.load(filepath)
        instance = cls.__new__(cls)
        instance._pipeline = payload["pipeline"]
        instance._label_encoder = payload["label_encoder"]
        return instance
