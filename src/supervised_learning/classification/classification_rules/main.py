from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def _clf_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


class ClassificationRuleLearner:
    """Simple one-rule (1R) classification rule learner.

    Trains a depth-1 decision tree as a stand-in for a proper rule-induction
    algorithm such as RIPPER. Replace ``self._model`` with a ``wittgenstein``
    ``RIPPER`` instance for production use.
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


if __name__ == "__main__":
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = ClassificationRuleLearner()
    clf.fit(X_train, y_train)
    print(clf.evaluate(X_test, y_test))
