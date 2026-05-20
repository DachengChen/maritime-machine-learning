# Gradient Boosting — Meta-Learning

## How It Works
Boosting trains estimators sequentially: each new model focuses on the examples that previous models got wrong. Gradient Boosting frames this as gradient descent in function space — each new tree is fit to the negative gradient (pseudo-residuals) of the loss function. The final prediction is an additive combination of all trees, each scaled by a learning rate. Shrinkage (small learning rate) and subsampling (stochastic gradient boosting) act as regularisation. Popular implementations include XGBoost, LightGBM, and scikit-learn's `GradientBoostingClassifier`.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **High-accuracy ship type classification** | Sequentially correct misclassifications of ambiguous vessel types (e.g. cargo vs. vehicle carrier) from AIS features |
| **ETA prediction with complex interactions** | Learn non-linear interactions between speed, route, weather proxies, and vessel class |
| **AIS spoofing detection** | Detect subtle manipulation patterns (small speed inconsistencies, impossible course changes) that weak learners alone miss |
| **Port congestion forecasting** | Predict berth wait time by boosting over traffic volume, vessel priority, and tidal features |

## AIS Features Typically Used
- `sog`, `cog`, `heading` — Kinematic state
- `length`, `width`, `draught` — Physical dimensions
- `shiptype`, `navigationalstatus` — Categorical features (encoded)
- Speed change rate, heading variance (engineered)

## Notes
- More accurate than single trees and bagging on structured AIS tabular data.
- Sensitive to learning rate and tree depth — tune via cross-validation.

## Run
```bash
python main.py
```
