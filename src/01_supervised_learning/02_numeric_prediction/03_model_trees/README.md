# Model Tree (M5 stub) — Numeric Prediction

## How It Works
A Model Tree extends a regression tree by fitting a linear regression model at each leaf instead of storing a constant mean. The tree structure segments the input space into regions, and within each region a local linear model provides fine-grained predictions. The M5 algorithm (and its variants) builds the tree greedily, then prunes it by replacing subtrees with leaf linear models whenever doing so reduces expected prediction error. The result combines the non-linear segmentation power of trees with the extrapolation ability of linear models.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Piecewise ETA modelling** | Different linear ETA equations for open-sea, coastal, and port-approach legs |
| **Speed profile by route segment** | Separate linear speed models for different navigational contexts |
| **Fuel estimation across vessel classes** | One branch per ship type, each with its own linear consumption equation |
| **Tide-adjusted draught prediction** | Linear draught–load relationships that differ by port / tidal region |

## AIS Features Typically Used
- `sog`, `cog`, `heading` — Kinematic inputs
- `shiptype`, `length`, `draught` — Vessel characteristics
- Route segment or port zone (engineered feature)

## Notes
- More accurate than plain regression trees for continuous targets with local linearity.
- The current implementation is a shallow CART stub; replace the leaf predictor with `LinearRegression` for a full M5 model.

## Run
```bash
python main.py
```
