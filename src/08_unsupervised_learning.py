"""
08_unsupervised_learning.py
===========================
Dummy reference implementations for unsupervised learning algorithms
applied to maritime AIS pattern detection and clustering tasks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import combinations


# ---------------------------------------------------------------------------
# Chapter 8 – Association Rules (Apriori stub)
# ---------------------------------------------------------------------------

class AssociationRuleLearner:
    """Brute-force frequent-itemset miner with association rule extraction
    (Chapter 8).

    For production use install ``mlxtend`` and replace with
    ``mlxtend.frequent_patterns.apriori`` + ``association_rules``.

    Parameters
    ----------
    min_support : float
        Minimum support threshold (0–1).
    min_confidence : float
        Minimum confidence threshold (0–1).

    Example
    -------
    >>> from src.unsupervised_learning import AssociationRuleLearner
    >>> arl = AssociationRuleLearner(min_support=0.3, min_confidence=0.6)
    >>> # transactions: list of sets, e.g. [{'Hamburg', 'Rotterdam'}, ...]
    >>> arl.fit(transactions)
    >>> print(arl.rules_[:3])
    """

    def __init__(self, min_support: float = 0.3, min_confidence: float = 0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rules_: list[dict] = []
        self._freq_itemsets: dict = {}

    # ------------------------------------------------------------------
    def fit(self, transactions: list[set]) -> "AssociationRuleLearner":
        """Mine frequent itemsets and generate association rules.

        Parameters
        ----------
        transactions : list of set
            Each set is one transaction (e.g. ports visited per voyage).
        """
        n = len(transactions)
        items: set = set()
        for t in transactions:
            items.update(t)

        freq: dict = {}

        # --- 1-itemsets
        for item in items:
            sup = sum(1 for t in transactions if item in t) / n
            if sup >= self.min_support:
                freq[frozenset([item])] = sup

        # --- k-itemsets (k >= 2)
        k = 2
        prev_keys = list(freq.keys())
        while prev_keys:
            candidates = set()
            for a, b in combinations(prev_keys, 2):
                union = a | b
                if len(union) == k:
                    candidates.add(union)
            new_keys = []
            for cand in candidates:
                sup = sum(1 for t in transactions if cand.issubset(t)) / n
                if sup >= self.min_support:
                    freq[cand] = sup
                    new_keys.append(cand)
            prev_keys = new_keys
            k += 1

        self._freq_itemsets = freq

        # --- Generate rules
        rules = []
        for itemset, sup in freq.items():
            if len(itemset) < 2:
                continue
            for size in range(1, len(itemset)):
                for antecedent in map(frozenset, combinations(itemset, size)):
                    consequent = itemset - antecedent
                    ant_sup = freq.get(antecedent, 0)
                    conf = sup / ant_sup if ant_sup > 0 else 0
                    if conf >= self.min_confidence:
                        rules.append({
                            "antecedent": set(antecedent),
                            "consequent": set(consequent),
                            "support": round(sup, 4),
                            "confidence": round(conf, 4),
                            "lift": round(conf / freq.get(consequent, 1), 4),
                        })
        self.rules_ = rules
        return self

    def transform(self) -> pd.DataFrame:
        """Return mined rules as a DataFrame."""
        return pd.DataFrame(self.rules_)


# ---------------------------------------------------------------------------
# Chapter 9 – k-means Clustering
# ---------------------------------------------------------------------------

class KMeansClustering:
    """k-means clustering for vessel trajectory segmentation (Chapter 9).

    Parameters
    ----------
    n_clusters : int
        Number of clusters *k*.
    random_state : int
        Reproducibility seed.

    Example
    -------
    >>> from src.unsupervised_learning import KMeansClustering
    >>> km = KMeansClustering(n_clusters=5)
    >>> km.fit(X)
    >>> labels = km.predict(X)
    >>> print(km.evaluate(X))
    """

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._model = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init="auto"
        )

    def fit(self, X) -> "KMeansClustering":
        self._model.fit(X)
        return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def fit_predict(self, X) -> np.ndarray:
        return self._model.fit_predict(X)

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self._model.cluster_centers_

    @property
    def inertia_(self) -> float:
        return float(self._model.inertia_)

    def evaluate(self, X) -> dict:
        """Return inertia and silhouette score for the fitted clustering."""
        labels = self.predict(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")
        return {
            "inertia": round(self.inertia_, 4),
            "silhouette": round(float(sil), 4),
        }
