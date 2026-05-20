# Linear Regression — Numeric Prediction

## How It Works
Linear Regression models the target variable as a weighted sum of input features plus a bias term: `y = w₁x₁ + w₂x₂ + … + b`. Weights are fitted by minimising the mean squared error (MSE) between predictions and observed values via the ordinary least-squares (OLS) closed-form solution or gradient descent. It is the simplest continuous-output predictor and provides an interpretable baseline; each weight directly quantifies the marginal effect of its feature on the target.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **ETA estimation** | Predict time-to-arrival from remaining distance and average SOG |
| **Speed-over-ground forecasting** | Estimate expected SOG from heading, route segment, and ship type |
| **Fuel consumption proxy** | Approximate fuel burn as a linear function of speed and displacement |
| **Draught change prediction** | Relate cargo load to change in draught across a voyage |

## AIS Features Typically Used
- `sog` — Target or predictor depending on the task
- `cog`, `heading` — Direction of travel
- `length`, `width`, `draught` — Physical characteristics correlated with drag and fuel use
- Distance to destination (engineered feature)

## Notes
- Assumes a linear relationship; will underfit complex non-linear dynamics.
- Use Lasso (L1) or Ridge (L2) regularisation when features are collinear.

## Run
```bash
python main.py
```
