from __future__ import annotations

import pathlib

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = pathlib.Path(__file__).parents[4] / "data" / "sample" / "01_ais_data.csv"
FEATURES = ["length", "width"]
TARGET = "shiptype"
KEEP_TYPES = ["Cargo", "Tanker", "Fishing", "Pleasure", "Passenger", "Sailing"]

# Mapping from full ship type names to short labels for the confusion matrix
SHORT_LABELS = {
    "Cargo": "Cargo",
    "Tanker": "Tanker",
    "Fishing": "Fishing",
    "Pleasure": "Pleasure",
    "Passenger": "Passngr",
    "Sailing": "Sailing",
}


# ---------------------------------------------------------------------------
# Step 1 – Load & clean
# ---------------------------------------------------------------------------

def load_and_clean(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.dropna(subset=FEATURES + [TARGET])
    df = df[df[TARGET].str.strip() != ""]
    df = df[df[TARGET].isin(KEEP_TYPES)]
    df[FEATURES] = df[FEATURES].astype(float)
    return df


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _clf_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


class NaiveBayesClassifier:
    """Gaussian Naïve Bayes classifier."""

    def __init__(self, var_smoothing: float = 1e-9):
        self._model = GaussianNB(var_smoothing=var_smoothing)

    def fit(self, X, y) -> "NaiveBayesClassifier":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        return _clf_metrics(y, self.predict(X))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    console = Console()

    df = load_and_clean(DATA_PATH)
    console.print(f"Rows after cleaning: [bold]{len(df)}[/bold]")
    console.print(f"Ship types: {sorted(df[TARGET].unique())}\n")

    le = LabelEncoder()
    X = df[FEATURES].values
    y = le.fit_transform(df[TARGET].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []
    for var_smoothing in [1e-11, 1e-9, 1e-7, 1e-5, 1e-3]:
        clf = NaiveBayesClassifier(var_smoothing=var_smoothing)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        m = _clf_metrics(y_test, y_pred)
        results.append((var_smoothing, m["accuracy"], m["f1_macro"]))

        # Confusion matrix
        labels = le.classes_
        cm = confusion_matrix(y_test, y_pred)
        mid_idx = len(labels) // 2

        tbl = Table(
            title=f"Confusion matrix  var_smoothing={var_smoothing:.0e}",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            title_style="bold",
        )
        tbl.add_column("True \\ Predicted", style="bold", justify="right")
        for lb in labels:
            tbl.add_column(SHORT_LABELS.get(lb, lb), justify="right")

        for i, row in enumerate(cm):
            row_label = (
                f"[bold magenta]{labels[i]}[/bold magenta]" if i == mid_idx else labels[i]
            )
            cells = []
            for j, v in enumerate(row):
                style = "bold green" if i == j else ("bold red" if v > 0 else "dim")
                cells.append(f"[{style}]{v}[/{style}]")
            tbl.add_row(row_label, *cells)

        console.print(tbl)
        console.print()

    # Summary table
    summary = Table(
        title="Gaussian Naïve Bayes Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title_style="bold",
    )
    summary.add_column("var_smoothing", justify="center", style="bold")
    summary.add_column("Accuracy", justify="right")
    summary.add_column("F1 macro", justify="right")

    best = max(results, key=lambda r: r[2])
    for vs, acc, f1 in results:
        style = "bold green" if vs == best[0] else ""
        summary.add_row(f"{vs:.0e}", f"{acc:.4f}", f"{f1:.4f}", style=style)

    console.print(summary)
    console.print(
        f"\n[bold green]Best var_smoothing by f1_macro:[/bold green] "
        f"{best[0]:.0e}  accuracy={best[1]}  f1_macro={best[2]}"
    )
