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
│   ├── supervised_learning/
│   │   ├── classification/
│   │   │   ├── nearest_neighbor/main.py
│   │   │   ├── naive_bayes/main.py
│   │   │   ├── decision_trees/main.py
│   │   │   └── classification_rules/main.py
│   │   ├── numeric_prediction/
│   │   │   ├── linear_regression/main.py
│   │   │   ├── regression_trees/main.py
│   │   │   └── model_trees/main.py
│   │   └── dual_use/
│   │       ├── neural_networks/main.py
│   │       └── svm/main.py
│   ├── unsupervised_learning/
│   │   ├── association_rules/main.py
│   │   └── kmeans/main.py
│   ├── meta_learning/
│   │   ├── bagging/main.py
│   │   ├── boosting/main.py
│   │   └── random_forests/main.py
│   ├── deep_learning/
│   │   ├── lstm/main.py
│   │   ├── gru/main.py
│   │   ├── cnn/main.py
│   │   ├── transformer/main.py
│   │   └── autoencoder/main.py
│   ├── recommendation_systems/
│   │   ├── collaborative_filtering/main.py
│   │   ├── content_based/main.py
│   │   ├── matrix_factorization/main.py
│   │   └── neural_cf/main.py
│   └── reinforcement_learning/
│       ├── q_learning/main.py
│       ├── deep_q_network/main.py
│       └── policy_gradient/main.py
└── requirements.txt
```

---

## Quick Start

### 1. Set up the environment

Using pyenv is recommended — see [Environment Setup with pyenv](#environment-setup-with-pyenv) at the bottom of this file.

Or install directly into your active environment:

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
python src/deep_learning/lstm/main.py
python src/deep_learning/gru/main.py
python src/deep_learning/cnn/main.py
python src/deep_learning/transformer/main.py
python src/deep_learning/autoencoder/main.py
python src/recommendation_systems/collaborative_filtering/main.py
python src/recommendation_systems/content_based/main.py
python src/recommendation_systems/matrix_factorization/main.py
python src/recommendation_systems/neural_cf/main.py
python src/reinforcement_learning/q_learning/main.py
python src/reinforcement_learning/deep_q_network/main.py
python src/reinforcement_learning/policy_gradient/main.py
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

### Deep Learning

| Path | Algorithm | Task |
|---|---|---|
| `deep_learning/lstm/` | LSTM sequence predictor | Regression / forecasting |
| `deep_learning/gru/` | GRU sequence predictor | Regression / forecasting |
| `deep_learning/cnn/` | 1-D CNN sequence model | Regression / classification |
| `deep_learning/transformer/` | Transformer encoder | Regression / classification |
| `deep_learning/autoencoder/` | LSTM Autoencoder | Anomaly detection |

### Recommendation Systems

| Path | Algorithm | Task |
|---|---|---|
| `recommendation_systems/collaborative_filtering/` | User-based CF | Route / port recommendations |
| `recommendation_systems/content_based/` | Content-based filtering | Item-feature similarity |
| `recommendation_systems/matrix_factorization/` | SVD Matrix Factorisation | Latent-factor recommendations |
| `recommendation_systems/neural_cf/` | Neural Collaborative Filtering | Deep recommendation |

### Reinforcement Learning

| Path | Algorithm | Task |
|---|---|---|
| `reinforcement_learning/q_learning/` | Tabular Q-learning | Route optimisation |
| `reinforcement_learning/deep_q_network/` | DQN (neural Q-network) | Continuous state control |
| `reinforcement_learning/policy_gradient/` | REINFORCE | Policy gradient optimisation |

---

## Environment Setup with pyenv

### 1. Install Python and create a virtual environment

```bash
# Install Python 3.13.1
pyenv install 3.13.1

# Pin the version locally (.python-version already set)
pyenv local 3.13.1

# Create a venv inside the project
python -m venv .venv

# Activate
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run any algorithm

```bash
# General pattern
python src/<category>/<algorithm>/main.py

# Examples
python src/supervised_learning/classification/nearest_neighbor/main.py
python src/deep_learning/lstm/main.py
python src/recommendation_systems/content_based/main.py
python src/reinforcement_learning/deep_q_network/main.py
```

### Useful pyenv commands

| Command | Description |
|---|---|
| `pyenv versions` | List all installed Python versions |
| `pyenv local 3.13.1` | Pin Python version for the project |
| `pyenv which python` | Show the path of the active Python binary |
| `source .venv/bin/activate` | Activate the project venv |
| `deactivate` | Deactivate the venv |

---

## License

MIT
