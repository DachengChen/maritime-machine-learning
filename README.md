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
│   ├── 01_supervised_learning/
│   │   ├── 01_classification/
│   │   │   ├── 01_nearest_neighbor/main.py
│   │   │   ├── 02_naive_bayes/main.py
│   │   │   ├── 03_decision_trees/main.py
│   │   │   └── 04_classification_rules/main.py
│   │   ├── 02_numeric_prediction/
│   │   │   ├── 01_linear_regression/main.py
│   │   │   ├── 02_regression_trees/main.py
│   │   │   └── 03_model_trees/main.py
│   │   └── 03_dual_use/
│   │       ├── 01_neural_networks/main.py
│   │       └── 02_svm/main.py
│   ├── 02_unsupervised_learning/
│   │   ├── 01_kmeans/main.py
│   │   └── 02_association_rules/main.py
│   ├── 03_meta_learning/
│   │   ├── 01_bagging/main.py
│   │   ├── 02_boosting/main.py
│   │   └── 03_random_forests/main.py
│   ├── 04_recommendation_systems/
│   │   ├── 01_collaborative_filtering/main.py
│   │   ├── 02_content_based/main.py
│   │   ├── 03_matrix_factorization/main.py
│   │   └── 04_neural_cf/main.py
│   ├── 05_reinforcement_learning/
│   │   ├── 01_q_learning/main.py
│   │   ├── 02_deep_q_network/main.py
│   │   └── 03_policy_gradient/main.py
│   └── 06_deep_learning/
│       ├── 01_rnn/main.py
│       ├── 02_cnn/main.py
│       ├── 03_lstm/main.py
│       ├── 04_gru/main.py
│       ├── 05_autoencoder/main.py
│       ├── 06_transformer/main.py
│       ├── 07_brnn/main.py
│       └── 08_drnn/main.py
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
python src/01_supervised_learning/01_classification/01_nearest_neighbor/main.py
python src/01_supervised_learning/01_classification/02_naive_bayes/main.py
python src/01_supervised_learning/01_classification/03_decision_trees/main.py
python src/01_supervised_learning/01_classification/04_classification_rules/main.py
python src/01_supervised_learning/02_numeric_prediction/01_linear_regression/main.py
python src/01_supervised_learning/02_numeric_prediction/02_regression_trees/main.py
python src/01_supervised_learning/02_numeric_prediction/03_model_trees/main.py
python src/01_supervised_learning/03_dual_use/01_neural_networks/main.py
python src/01_supervised_learning/03_dual_use/02_svm/main.py
python src/02_unsupervised_learning/01_kmeans/main.py
python src/02_unsupervised_learning/02_association_rules/main.py
python src/03_meta_learning/01_bagging/main.py
python src/03_meta_learning/02_boosting/main.py
python src/03_meta_learning/03_random_forests/main.py
python src/04_recommendation_systems/01_collaborative_filtering/main.py
python src/04_recommendation_systems/02_content_based/main.py
python src/04_recommendation_systems/03_matrix_factorization/main.py
python src/04_recommendation_systems/04_neural_cf/main.py
python src/05_reinforcement_learning/01_q_learning/main.py
python src/05_reinforcement_learning/02_deep_q_network/main.py
python src/05_reinforcement_learning/03_policy_gradient/main.py
python src/06_deep_learning/01_rnn/main.py
python src/06_deep_learning/02_cnn/main.py
python src/06_deep_learning/03_lstm/main.py
python src/06_deep_learning/04_gru/main.py
python src/06_deep_learning/05_autoencoder/main.py
python src/06_deep_learning/06_transformer/main.py
python src/06_deep_learning/07_brnn/main.py
python src/06_deep_learning/08_drnn/main.py
```

All implementations expose the same `fit` / `predict` / `evaluate` interface.

### Supervised Learning

| Path | Algorithm | Task |
|---|---|---|
| `01_supervised_learning/01_classification/01_nearest_neighbor/` | k-Nearest Neighbours | Classification |
| `01_supervised_learning/01_classification/02_naive_bayes/` | Gaussian Naïve Bayes | Classification |
| `01_supervised_learning/01_classification/03_decision_trees/` | CART Decision Tree | Classification |
| `01_supervised_learning/01_classification/04_classification_rules/` | 1R Rule Learner | Classification |
| `01_supervised_learning/02_numeric_prediction/01_linear_regression/` | Linear Regression | Numeric prediction |
| `01_supervised_learning/02_numeric_prediction/02_regression_trees/` | Regression Tree | Numeric prediction |
| `01_supervised_learning/02_numeric_prediction/03_model_trees/` | Model Tree (M5 stub) | Numeric prediction |
| `01_supervised_learning/03_dual_use/01_neural_networks/` | MLP (classifier + regressor) | Dual use |
| `01_supervised_learning/03_dual_use/02_svm/` | SVM (classifier + regressor) | Dual use |

### Unsupervised Learning

| Path | Algorithm | Task |
|---|---|---|
| `02_unsupervised_learning/01_kmeans/` | k-means Clustering | Clustering |
| `02_unsupervised_learning/02_association_rules/` | Apriori / Association Rules | Pattern detection |

### Meta-Learning

| Path | Algorithm | Task |
|---|---|---|
| `03_meta_learning/01_bagging/` | Bagging (classifier + regressor) | Dual use |
| `03_meta_learning/02_boosting/` | Gradient Boosting (classifier + regressor) | Dual use |
| `03_meta_learning/03_random_forests/` | Random Forest (classifier + regressor) | Dual use |

### Recommendation Systems

| Path | Algorithm | Task |
|---|---|---|
| `04_recommendation_systems/01_collaborative_filtering/` | User-based CF | Route / port recommendations |
| `04_recommendation_systems/02_content_based/` | Content-based filtering | Item-feature similarity |
| `04_recommendation_systems/03_matrix_factorization/` | SVD Matrix Factorisation | Latent-factor recommendations |
| `04_recommendation_systems/04_neural_cf/` | Neural Collaborative Filtering | Deep recommendation |

### Reinforcement Learning

| Path | Algorithm | Task |
|---|---|---|
| `05_reinforcement_learning/01_q_learning/` | Tabular Q-learning | Route optimisation |
| `05_reinforcement_learning/02_deep_q_network/` | DQN (neural Q-network) | Continuous state control |
| `05_reinforcement_learning/03_policy_gradient/` | REINFORCE | Policy gradient optimisation |

### Deep Learning

| Path | Algorithm | Task |
|---|---|---|
| `06_deep_learning/01_rnn/` | Vanilla RNN sequence predictor | Regression / forecasting |
| `06_deep_learning/02_cnn/` | 1-D CNN sequence model | Regression / classification |
| `06_deep_learning/03_lstm/` | LSTM sequence predictor | Regression / forecasting |
| `06_deep_learning/04_gru/` | GRU sequence predictor | Regression / forecasting |
| `06_deep_learning/05_autoencoder/` | LSTM Autoencoder | Anomaly detection |
| `06_deep_learning/06_transformer/` | Transformer encoder | Regression / classification |
| `06_deep_learning/07_brnn/` | Bidirectional RNN predictor | Regression / classification |
| `06_deep_learning/08_drnn/` | Deep RNN (stacked layers) | Regression / forecasting |

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
python src/01_supervised_learning/01_classification/01_nearest_neighbor/main.py
python src/06_deep_learning/03_lstm/main.py
python src/04_recommendation_systems/02_content_based/main.py
python src/05_reinforcement_learning/02_deep_q_network/main.py
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
