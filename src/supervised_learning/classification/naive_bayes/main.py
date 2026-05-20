from __future__ import annotations

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def _clf_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


class NaiveBayesClassifier:
    """Gaussian Naïve Bayes classifier."""

    def __init__(self):
        self._model = GaussianNB()

    def fit(self, X, y) -> "NaiveBayesClassifier":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _clf_metrics(y, self.predict(X))


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = NaiveBayesClassifier()
    clf.fit(X_train, y_train)
    print(clf.evaluate(X_test, y_test))
