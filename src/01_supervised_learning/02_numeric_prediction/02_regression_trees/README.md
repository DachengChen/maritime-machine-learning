# Regression Tree — Numeric Prediction

## How It Works
A Regression Tree partitions the feature space with binary splits — identical to a classification tree — but predicts the mean target value of the training samples that fall into each leaf. At each node, the split is chosen to minimise the total residual sum of squares (RSS) across the two child nodes. The result is a piecewise-constant function: each leaf returns one constant prediction. Tree depth controls model complexity; shallow trees underfit, deep trees overfit.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Non-linear ETA prediction** | Capture regime changes (e.g. port approach reduces speed) that linear models miss |
| **SOG prediction by ship class** | Automatically learn separate speed profiles for cargo, tanker, and fishing vessels |
| **Waiting time at anchor** | Predict anchorage waiting time from port congestion features and ship priority |
| **Draught-dependent fuel estimation** | Model the non-linear relationship between laden draught and fuel consumption |

## AIS Features Typically Used
- `sog`, `cog`, `heading` — Kinematic state
- `shiptype`, `length`, `draught` — Vessel characteristics
- `navigationalstatus` — Operational context

## Notes
- More flexible than linear regression for AIS data, which often exhibits regime-switching behaviour.
- Prone to high variance; pair with bagging or random forests for production use.

## Run
```bash
python main.py
```
