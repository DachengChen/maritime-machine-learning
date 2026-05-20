# Random Forest — Meta-Learning

## How It Works
Random Forest extends bagging by adding feature randomisation: at each node split, only a random subset of features (typically √p for classification, p/3 for regression) is considered as split candidates. This decorrelates the individual trees more than pure bagging, further reducing ensemble variance. The combination of bootstrap sampling and feature subsampling produces diverse, robust trees whose averaged predictions are consistently better than any single tree. The OOB samples also provide a free internal cross-validation estimate.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Vessel behaviour classification** | Classify ships as fishing, anchored, underway, or drifting with high accuracy and built-in feature importance scores |
| **SOG/COG anomaly detection** | Flag kinematic outliers by comparing predictions against the ensemble's confidence interval |
| **Feature importance for AIS fields** | Identify which AIS attributes (SOG, heading, length, draught…) most drive ship type prediction |
| **Voyage fuel regression** | Robustly estimate fuel consumption across diverse vessel types and route conditions |

## AIS Features Typically Used
- `sog`, `cog`, `heading` — Kinematic state
- `length`, `width`, `draught` — Physical dimensions
- `navigationalstatus`, `shiptype` — Labels / secondary features
- Engineered: acceleration, heading change rate, time-in-region

## Notes
- Provides `feature_importances_` — valuable for maritime domain understanding.
- Robust to missing values with imputation; handles mixed feature types well.

## Run
```bash
python main.py
```
