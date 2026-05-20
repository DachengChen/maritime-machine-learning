# CART Decision Tree — Classification

## How It Works
A Decision Tree recursively partitions the feature space by choosing the split (feature + threshold) that maximises information gain (or minimises Gini impurity) at each node. The result is a binary tree of if-then rules that can be visualised and interpreted directly. CART (Classification and Regression Trees) handles both classification and regression; here it is used for classification. Pruning or depth limits prevent overfitting.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Vessel behaviour classification** | Learn explicit speed/heading thresholds to classify vessels as fishing, anchored, or underway |
| **Regulatory compliance screening** | Build interpretable rules (e.g. "if SOG > 5 knots AND draught > 8 m → flag for inspection") for port state control |
| **Ship type identification** | Split on dimensions (length, width) and speed to categorise vessel types |
| **Collision-risk labelling** | Classify encounter scenarios (safe, caution, danger) based on CPA and TCPA features |

## AIS Features Typically Used
- `sog`, `cog`, `heading` — Kinematic state
- `length`, `width`, `draught` — Physical dimensions
- `navigationalstatus`, `shiptype` — Labels / secondary features

## Notes
- Fully interpretable: produced rules can be reviewed by maritime domain experts.
- Prone to overfitting on small AIS datasets; control via `max_depth` or `min_samples_leaf`.

## Run
```bash
python main.py
```
