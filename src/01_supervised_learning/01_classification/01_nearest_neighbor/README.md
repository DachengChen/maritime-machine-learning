# k-Nearest Neighbours — Classification

## How It Works
kNN stores every labelled training example. To classify a new vessel, it computes the distance (e.g. Euclidean) from that vessel to every stored example, selects the *k* nearest, and assigns the most frequent class among them. There is no explicit training phase — all computation happens at prediction time. The choice of *k* controls the bias-variance trade-off: small *k* is flexible but noisy, large *k* is stable but may blur class boundaries.

## Use Case — Vessel Type Identification from Physical Dimensions

Classify the ship type (e.g. Cargo, Tanker, Fishing, Pleasure) of an AIS-transmitting vessel using only its **length** and **width**.

**Data:** `data/sample/01_ais_data.csv`

| Feature | Column | Description |
|---|---|---|
| `length` | col 9 | Overall vessel length (metres) |
| `width` | col 8 | Overall vessel beam/width (metres) |

**Label:** `shiptype` (col 7) — 6 classes kept after filtering:

| Class | Approx. test rows |
|---|---|
| Cargo | 1749 |
| Tanker | 655 |
| Pleasure | 418 |
| Fishing | 331 |
| Sailing | 232 |
| Passenger | 131 |

All other vessel types (Tug, SAR, Military, HSC, etc.) are excluded.

### Step 1 — Clean the Data

Rows where `length` or `width` is missing cannot be used as feature vectors and are dropped:

```bash
# Quick inspection — ship types that have both dimensions present
cat data/sample/01_ais_data.csv \
  | tail -n +2 \
  | cut -d',' -f2,7,8,9 \
  | sort | uniq \
  | cut -d',' -f2- \
  | grep -v ',,'
```

In Python (`load_and_clean` in `main.py`):
1. Read CSV with `pandas`.
2. Drop rows where `length` **or** `width` **or** `shiptype` is `NaN` / empty.
3. Keep only rows whose `shiptype` is one of: `Cargo`, `Tanker`, `Fishing`, `Pleasure`, `Passenger`, `Sailing`.
4. Cast both feature columns to `float`.

### Step 2 — Train & Evaluate

- 80 / 20 stratified train-test split.
- `KNeighborsClassifier` run for k ∈ {1, 3, 5, 7}.
- For each k a **confusion matrix** is printed with rows labelled **True** (actual class) and columns labelled **Predicted** (model output).
- Summary table at the end reports accuracy and macro-averaged F1 across all k values, and highlights the best k by F1.

```
── Confusion matrix  k=5 ──
                       Predicted
              Cargo  Dredging  Fishing  ...
True   Cargo  36449        45       74  ...
    Dredging     43       945       30  ...
     Fishing      6        12     4010  ...
         ...

+------+------------+------------+
|    k |   accuracy |   f1_macro |
+------+------------+------------+
|    1 |     0.8393 |     0.6175 |
|    3 |     0.8853 |     0.5945 |
|    5 |     0.8882 |     0.6239 |
|    7 |     0.8895 |     0.6121 |
+------+------------+------------+
Best k by f1_macro: k=5
```

## Notes
- Physical dimensions alone are a coarse signal; large overlap exists between ship-type classes of similar size.
- Choose *k* via cross-validation; larger *k* reduces noise sensitivity.

## Run
```bash
python main.py
```
