# Gaussian Naïve Bayes — Classification

## How It Works
Naïve Bayes applies Bayes' theorem with the "naïve" assumption that every feature is conditionally independent of every other given the class label. For each class, it computes the prior probability and the likelihood of each feature value, then picks the class with the highest posterior probability. The Gaussian variant models each feature's likelihood as a normal distribution, estimating the mean and variance per class from training data. Despite the unrealistic independence assumption, it often performs surprisingly well and is extremely fast to train and predict.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Ship type classification** | Given SOG, COG, heading, and dimensions, estimate whether a vessel is cargo, tanker, fishing, or passenger |
| **Navigational status detection** | Probabilistically classify vessels as anchored, moored, or underway from sensor readings |
| **Anomalous message filtering** | Quickly flag AIS messages with unusual feature combinations that are unlikely under normal class distributions |
| **Real-time triage** | Fast onboard or shore-side screening of incoming AIS messages before applying heavier models |

## AIS Features Typically Used
- `sog` — Speed over ground
- `cog` — Course over ground
- `heading` — True heading
- `length`, `width`, `draught` — Physical vessel dimensions
- `shiptype` — Label for supervised training
- `navigationalstatus` — Encoded operational state

## Notes
- Very fast; suitable for streaming AIS data.
- Degrades when features are strongly correlated (e.g. length and width).

## Run
```bash
python main.py
```
