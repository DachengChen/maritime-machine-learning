# Maritime Machine Learning

> Python starter project for AIS-based vessel analytics — data preprocessing,
> feature engineering, ETA prediction, destination prediction, and anomaly detection.

---

## Project Structure

```
maritime-machine-learning/
├── data/
│   ├── generate_sample_data.py   # Synthetic AIS data generator
│   ├── raw/                      # Place raw AIS CSV files here
│   ├── processed/                # Output of preprocessing pipeline
│   └── sample/                   # Synthetic sample dataset (ais_sample.csv)
├── models/                       # Saved model artefacts (.joblib)
├── src/
│   ├── __init__.py
│   ├── 01_data_preprocessing.py     # AIS loading, cleaning, normalisation
│   ├── 02_feature_engineering.py    # Temporal, kinematic & spatial features
│   ├── 03_eta_prediction.py         # Gradient-boosted ETA regressor
│   ├── 04_destination_prediction.py # Random-forest destination classifier
│   ├── 05_anomaly_detection.py      # Isolation Forest & DBSCAN detectors
│   ├── 06_model_evaluation.py       # Shared metrics & visualisation helpers
│   ├── 07_supervised_learning.py    # KNN, NB, DT, Rules, LinReg, NN, SVM
│   ├── 08_unsupervised_learning.py  # Association Rules, k-means
│   └── 09_meta_learning.py          # Bagging, Boosting, Random Forests
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py
│   └── test_models.py
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the sample dataset

```bash
python data/generate_sample_data.py
# → data/sample/ais_sample.csv  (~1 000 synthetic AIS records)
```

### 3. Run the tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Module Overview

### `src/01_data_preprocessing.py`

| Function | Description |
|---|---|
| `load_ais_csv(filepath)` | Load a raw AIS CSV file |
| `preprocess(df)` | Full cleaning pipeline (invalid positions, SOG/COG, deduplication, …) |
| `data_summary(df)` | Per-column statistics including null counts |

### `src/02_feature_engineering.py`

| Function | Description |
|---|---|
| `add_temporal_features(df)` | Cyclical hour / day-of-week / month encodings |
| `add_kinematic_features(df)` | ΔSOG, ΔCOG, step distance, elapsed time |
| `add_spatial_features(df)` | Cyclical lat/lon encodings |
| `add_voyage_aggregate_features(df)` | Expanding mean/std of SOG and COG |
| `build_features(df)` | Runs all of the above |
| `get_feature_matrix(df)` | Returns a model-ready feature matrix |

### `src/03_eta_prediction.py`

`ETAPredictor` — Gradient Boosted Regressor wrapped in a scikit-learn `Pipeline`.

```python
from src.eta_prediction import ETAPredictor

model = ETAPredictor(n_estimators=200)
model.fit(X_train, y_train)           # y: remaining seconds to arrival
print(model.evaluate(X_test, y_test)) # {"mae": ..., "rmse": ..., "r2": ...}
model.save("models/eta_model.joblib")
```

### `src/04_destination_prediction.py`

`DestinationPredictor` — Random Forest multi-class classifier.

```python
from src.destination_prediction import DestinationPredictor

model = DestinationPredictor()
model.fit(X_train, y_train)           # y: destination label strings
print(model.evaluate(X_test, y_test)) # {"accuracy": ..., "f1_macro": ...}
model.save("models/destination_model.joblib")
```

### `src/05_anomaly_detection.py`

`IsolationForestDetector` — unsupervised anomaly scoring.  
`DBSCANAnomalyDetector` — route-corridor clustering; noise points are anomalies.

```python
from src.anomaly_detection import IsolationForestDetector

det = IsolationForestDetector(contamination=0.03)
det.fit(X_train)
df_scored = det.flag_anomalies(df, feature_cols=[...])
```

### `src/06_model_evaluation.py`

Shared helpers for both regression and classification:

```python
from src.model_evaluation import regression_metrics, plot_regression_residuals

metrics = regression_metrics(y_true, y_pred)
ax = plot_regression_residuals(y_true, y_pred)
```

---

## ML Algorithms Reference

The table below maps each algorithm to its learning task and numbered source file.

### Supervised Learning — [`src/07_supervised_learning.py`](src/07_supervised_learning.py)

| Algorithm | Learning Task |
|---|---|
| Nearest Neighbor | Classification |
| Naive Bayes | Classification |
| Decision Trees | Classification |
| Classification Rule Learners | Classification |
| Linear Regression | Numeric prediction |
| Regression Trees | Numeric prediction |
| Model Trees | Numeric prediction |
| Neural Networks | Dual use |
| Support Vector Machines | Dual use |

### Unsupervised Learning — [`src/08_unsupervised_learning.py`](src/08_unsupervised_learning.py)

| Algorithm | Learning Task |
|---|---|
| Association Rules | Pattern detection |
| k-means Clustering | Clustering |

### Meta-Learning — [`src/09_meta_learning.py`](src/09_meta_learning.py)

| Algorithm | Learning Task |
|---|---|
| Bagging | Dual use |
| Boosting | Dual use |
| Random Forests | Dual use |

All implementations follow the same `fit` / `predict` / `evaluate` interface used by the rest of the project.
Imports still work using the clean alias, e.g. `from src.supervised_learning import NearestNeighborClassifier`.

---

## Expected AIS Columns

| Column | Type | Description |
|---|---|---|
| `mmsi` | int | Maritime Mobile Service Identity |
| `timestamp` | datetime | UTC position report time |
| `lat` | float | Latitude (−90 … 90) |
| `lon` | float | Longitude (−180 … 180) |
| `sog` | float | Speed Over Ground (knots) |
| `cog` | float | Course Over Ground (degrees) |
| `heading` | float | True heading (0–360; 511 = unavailable) |
| `vessel_type` | int | AIS vessel-type code |
| `destination` | str | Free-text destination field |
| `draught` | float | Draught (metres) |
| `length` | int | Overall vessel length (metres) |

---

## License

MIT
