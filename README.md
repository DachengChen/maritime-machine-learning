# Maritime Machine Learning

> Python project for AIS-based vessel analytics using machine learning.

---

## Project Structure

```
maritime-machine-learning/
├── data/
│   ├── raw/                          # Place raw AIS CSV files here
│   ├── processed/                    # Output of preprocessing pipeline
│   └── sample/                       # Real AIS sample data from Kaggle
│       └── 01_ais_data.csv
├── models/                           # Saved model artefacts (.joblib)
├── src/
│   ├── 07_supervised_learning.py    # KNN, NB, DT, Rules, LinReg, NN, SVM
│   ├── 08_unsupervised_learning.py  # Association Rules, k-means
│   └── 09_meta_learning.py          # Bagging, Boosting, Random Forests
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Sample dataset

`data/sample/01_ais_data.csv` is a real AIS dataset downloaded from Kaggle:

> **[AIS Dataset — Emin Serkan Erdonmez](https://www.kaggle.com/datasets/eminserkanerdonmez/ais-dataset)**

---

## Data

### Sample data — `data/sample/01_ais_data.csv`

This file is a real-world AIS (Automatic Identification System) dataset sourced from Kaggle:

| | |
|---|---|
| **Source** | [https://www.kaggle.com/datasets/eminserkanerdonmez/ais-dataset](https://www.kaggle.com/datasets/eminserkanerdonmez/ais-dataset) |
| **Author** | Emin Serkan Erdonmez |
| **Contents** | Vessel position reports including MMSI, navigational status, SOG, COG, heading, ship type, dimensions, and draught |

Place additional raw AIS CSV files in `data/raw/`.

---

## ML Algorithms Reference

Each algorithm lives in its own directory and can be run directly:

```bash
python src/supervised_learning/classification/nearest_neighbor/main.py
python src/supervised_learning/classification/naive_bayes/main.py
python src/supervised_learning/classification/decision_trees/main.py
python src/supervised_learning/classification/classification_rules/main.py
python src/supervised_learning/numeric_prediction/linear_regression/main.py
python src/supervised_learning/numeric_prediction/regression_trees/main.py
python src/supervised_learning/numeric_prediction/model_trees/main.py
python src/supervised_learning/dual_use/neural_networks/main.py
python src/supervised_learning/dual_use/svm/main.py
python src/unsupervised_learning/association_rules/main.py
python src/unsupervised_learning/kmeans/main.py
python src/meta_learning/bagging/main.py
python src/meta_learning/boosting/main.py
python src/meta_learning/random_forests/main.py
```

All implementations expose the same `fit` / `predict` / `evaluate` interface.

### Supervised Learning

| Path | Algorithm | Task |
|---|---|---|
| `supervised_learning/classification/nearest_neighbor/` | k-Nearest Neighbours | Classification |
| `supervised_learning/classification/naive_bayes/` | Gaussian Naïve Bayes | Classification |
| `supervised_learning/classification/decision_trees/` | CART Decision Tree | Classification |
| `supervised_learning/classification/classification_rules/` | 1R Rule Learner | Classification |
| `supervised_learning/numeric_prediction/linear_regression/` | Linear Regression | Numeric prediction |
| `supervised_learning/numeric_prediction/regression_trees/` | Regression Tree | Numeric prediction |
| `supervised_learning/numeric_prediction/model_trees/` | Model Tree (M5 stub) | Numeric prediction |
| `supervised_learning/dual_use/neural_networks/` | MLP (classifier + regressor) | Dual use |
| `supervised_learning/dual_use/svm/` | SVM (classifier + regressor) | Dual use |

### Unsupervised Learning

| Path | Algorithm | Task |
|---|---|---|
| `unsupervised_learning/association_rules/` | Apriori / Association Rules | Pattern detection |
| `unsupervised_learning/kmeans/` | k-means Clustering | Clustering |

### Meta-Learning

| Path | Algorithm | Task |
|---|---|---|
| `meta_learning/bagging/` | Bagging (classifier + regressor) | Dual use |
| `meta_learning/boosting/` | Gradient Boosting (classifier + regressor) | Dual use |
| `meta_learning/random_forests/` | Random Forest (classifier + regressor) | Dual use |

---

## License

MIT
