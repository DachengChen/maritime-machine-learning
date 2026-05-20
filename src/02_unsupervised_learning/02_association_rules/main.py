from __future__ import annotations

import pandas as pd
from itertools import combinations


class AssociationRuleLearner:
    """Brute-force frequent-itemset miner with association rule extraction.

    For production use install ``mlxtend`` and replace with
    ``mlxtend.frequent_patterns.apriori`` + ``association_rules``.

    Parameters
    ----------
    min_support : float
        Minimum support threshold (0–1). Default ``0.3``.
    min_confidence : float
        Minimum confidence threshold (0–1). Default ``0.6``.
    """

    def __init__(self, min_support: float = 0.3, min_confidence: float = 0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rules_: list[dict] = []
        self._freq_itemsets: dict = {}

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

        # 1-itemsets
        for item in items:
            sup = sum(1 for t in transactions if item in t) / n
            if sup >= self.min_support:
                freq[frozenset([item])] = sup

        # k-itemsets (k >= 2)
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

        # Generate rules
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


if __name__ == "__main__":
    # Synthetic port-call transactions
    transactions = [
        {"Hamburg", "Rotterdam", "Antwerp"},
        {"Hamburg", "Rotterdam"},
        {"Rotterdam", "Antwerp"},
        {"Hamburg", "Antwerp"},
        {"Hamburg", "Rotterdam", "Antwerp", "Bremen"},
        {"Rotterdam", "Bremen"},
        {"Hamburg", "Bremen"},
        {"Antwerp", "Bremen"},
        {"Hamburg", "Rotterdam", "Bremen"},
        {"Rotterdam", "Antwerp", "Bremen"},
    ]

    learner = AssociationRuleLearner(min_support=0.3, min_confidence=0.6)
    learner.fit(transactions)
    df = learner.transform()
    print(df.to_string(index=False))
