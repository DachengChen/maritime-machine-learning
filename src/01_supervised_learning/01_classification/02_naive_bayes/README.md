# Gaussian Naïve Bayes — Classification

## How It Works
Naïve Bayes applies Bayes' theorem with the "naïve" assumption that every feature is conditionally independent of every other given the class label. For each class it computes the prior probability and the likelihood of each feature value, then picks the class with the highest posterior probability. The **Gaussian** variant models each feature's likelihood as a normal distribution, estimating the mean and variance per class from training data. Despite the unrealistic independence assumption, it often performs surprisingly well and is extremely fast to train and predict.

## Use Case — Vessel Type Identification from Physical Dimensions

Classify the ship type (e.g. Cargo, Tanker, Fishing, Pleasure) of an AIS-transmitting vessel using only its **length** and **width**.

**Data:** `data/sample/01_ais_data.csv`

| Feature | Column | Description |
|---|---|---|
| `length` | col 9 | Overall vessel length (metres) |
| `width` | col 8 | Overall vessel beam/width (metres) |

**Label:** `shiptype` — 6 classes kept after filtering:

| Class | Approx. test rows |
|---|---|
| Cargo | ~37 900 |
| Tanker | ~15 700 |
| Fishing | ~4 800 |
| Pleasure | ~780 |
| Sailing | ~490 |
| Passenger | ~3 500 |

All other vessel types are excluded.

### Step 1 — Clean the Data

Rows where `length` or `width` is missing cannot be used as feature vectors and are dropped. In Python (`load_and_clean` in `main.py`):

1. Read CSV with `pandas`.
2. Drop rows where `length` **or** `width` **or** `shiptype` is `NaN` / empty.
3. Keep only rows whose `shiptype` is one of: `Cargo`, `Tanker`, `Fishing`, `Pleasure`, `Passenger`, `Sailing`.
4. Cast both feature columns to `float`.

### Step 2 — Train & Evaluate

- 80 / 20 stratified train-test split.
- `GaussianNB` run for `var_smoothing` ∈ {1e-11, 1e-9, 1e-7, 1e-5, 1e-3}.
- For each setting a **confusion matrix** is printed with rows labelled **True** (actual class) and columns labelled **Predicted** (model output). Correct predictions are highlighted in green, errors in red.
- Summary table at the end reports accuracy and macro-averaged F1 for each `var_smoothing` value and highlights the best by F1.

```
Confusion matrix  var_smoothing=1e-03
╭──────────────────┬───────┬─────────┬─────────┬──────────┬─────────┬────────╮
│ True \ Predicted │ Cargo │ Fishing │ Passngr │ Pleasure │ Sailing │ Tanker │
├──────────────────┼───────┼─────────┼─────────┼──────────┼─────────┼────────┤
│            Cargo │ 32766 │     201 │      11 │       78 │       0 │   4893 │
│          Fishing │   822 │    2867 │       0 │     1109 │       0 │      0 │
│        Passenger │  2223 │     273 │       0 │       10 │       0 │   1032 │
│         Pleasure │     1 │     215 │       0 │      566 │       0 │      0 │
│          Sailing │    17 │     127 │       0 │      343 │       0 │      0 │
│           Tanker │ 11929 │      53 │       0 │        0 │       0 │   3724 │
╰──────────────────┴───────┴─────────┴─────────┴──────────┴─────────┴────────╯

Gaussian Naïve Bayes Summary
╭───────────────┬──────────┬──────────╮
│ var_smoothing │ Accuracy │ F1 macro │
├───────────────┼──────────┼──────────┤
│     1e-11     │   0.6221 │   0.3360 │
│     1e-09     │   0.6221 │   0.3360 │
│     1e-07     │   0.6221 │   0.3360 │
│     1e-05     │   0.6221 │   0.3360 │
│     1e-03     │   0.6311 │   0.3537 │  ← best
╰───────────────┴──────────┴──────────╯
```

## Notes
- Very fast; suitable for streaming AIS data with millions of rows.
- `var_smoothing` adds a fraction of the largest per-feature variance to every class variance — larger values regularise more.
- Length and width are correlated, which violates the independence assumption and caps F1. Compare with k-NN which reaches higher F1 on the same features.
- Passenger and Sailing vessels are most often misclassified as Cargo because their size overlap is significant.

## Run
```bash
python main.py
```
