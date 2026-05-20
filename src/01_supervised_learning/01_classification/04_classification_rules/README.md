# 1R Rule Learner — Classification

## How It Works
1R (One Rule) finds the single feature that, when split into discrete intervals, produces the fewest classification errors on the training set. A simple lookup table maps each interval of the chosen feature to the majority class. It is the simplest possible rule-based classifier and serves as an interpretable baseline; more powerful rule sets (RIPPER, CN2) extend this idea by chaining multiple rules.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Speed-based vessel screening** | One rule such as "if SOG < 1 knot → anchored/moored" captures the most discriminative signal with zero complexity |
| **Rapid baseline for ship type** | Identify that vessel length is the dominant discriminator between cargo ships and fishing vessels |
| **Alert rule generation** | Produce a single, auditable rule for bridge watch officers or VTS operators |
| **Feature importance proxy** | Discover which AIS attribute alone is most predictive before building complex models |

## AIS Features Typically Used
- `sog` — Often the single best discriminator for navigational status
- `length` or `shiptype` — Strong single-feature predictors of vessel category
- `draught` — Distinguishes laden vs. ballast voyages

## Notes
- Accuracy is low; its value is interpretability and speed.
- Use as a sanity-check baseline before deploying kNN, Decision Trees, or ensemble methods.

## Run
```bash
python main.py
```
