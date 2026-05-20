# Support Vector Machine — Classification & Regression

## How It Works
An SVM finds the hyperplane that maximises the margin between classes (classification) or that fits training points within a tube of width ε while minimising model complexity (SVR for regression). The kernel trick (RBF, polynomial) implicitly maps inputs to a high-dimensional space, allowing non-linear decision boundaries without explicitly computing the transformation. Only the training points closest to the boundary ("support vectors") influence the model, making SVMs memory-efficient and robust to outliers outside the margin.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **AIS spoofing detection** | Classify genuine vs. manipulated AIS messages in a high-dimensional feature space; SVMs excel at separating rare anomalous events |
| **Vessel type classification** | RBF-kernel SVM learns non-linear boundaries between ship categories |
| **Speed deviation regression (SVR)** | Predict expected SOG; deviations from predicted value flag potential data quality issues |
| **Dark vessel identification** | Binary SVC on behavioural features to distinguish legitimate transponder-off events from intentional AIS shutdowns |

## AIS Features Typically Used
- `sog`, `cog`, `heading` — Kinematic state
- `length`, `width`, `draught` — Vessel dimensions
- Derived features: speed change rate, heading variance over window

## Notes
- Scale all features before training (RBF kernel is distance-sensitive).
- Training complexity scales poorly with dataset size (O(n²–n³)); use `LinearSVC` or subsampling for large AIS archives.

## Run
```bash
python main.py
```
