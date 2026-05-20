# Bagging — Meta-Learning

## How It Works
Bagging (Bootstrap Aggregating) trains *B* independent base estimators on different bootstrap samples (random samples with replacement) of the training set. For classification, predictions are combined by majority vote; for regression, by averaging. Because each model sees a slightly different dataset, their errors are uncorrelated, and averaging reduces variance without increasing bias. Random Forests are the most well-known extension of bagging applied to decision trees.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Robust vessel type classification** | Aggregate many decision trees trained on different AIS subsets to reduce misclassification from noisy or missing fields |
| **ETA variance reduction** | Average multiple regression trees to produce stable arrival-time estimates even with incomplete data |
| **AIS data quality scoring** | Train a bagged classifier on labelled clean/corrupt messages; ensemble reduces false-positive rate |
| **Out-of-bag anomaly detection** | Use OOB (out-of-bag) error estimates to identify AIS records that are consistently hard to classify |

## AIS Features Typically Used
- `sog`, `cog`, `heading` — Kinematic state
- `length`, `width`, `draught` — Physical dimensions
- `navigationalstatus`, `shiptype` — Labels / secondary features

## Notes
- Directly reduces overfitting of individual trees with no hyperparameter tuning beyond *B* (number of estimators).
- Parallel training makes bagging fast on large AIS datasets.

## Run
```bash
python main.py
```
