# k-Nearest Neighbours — Classification

## How It Works
kNN stores every labelled training example. To classify a new vessel, it computes the distance (e.g. Euclidean) from that vessel to every stored example, selects the *k* nearest, and assigns the most frequent class among them. There is no explicit training phase — all computation happens at prediction time. The choice of *k* controls the bias-variance trade-off: small *k* is flexible but noisy, large *k* is stable but may blur class boundaries.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Vessel type identification** | Classify unknown ships (cargo, tanker, fishing, passenger) by comparing SOG, COG, heading, length, and width against a labelled fleet database |
| **Navigational status detection** | Determine whether a vessel is anchored, moored, or underway based on its movement profile |
| **AIS track matching** | Match an anonymous AIS track to a known vessel profile using trajectory similarity |
| **Port congestion prediction** | Find historically similar traffic snapshots to estimate current congestion levels |

## AIS Features Typically Used
- `sog` — Speed over ground
- `cog` — Course over ground
- `heading` — True heading
- `length`, `width`, `draught` — Physical vessel dimensions
- `navigationalstatus` — Encoded label for supervised training

## Notes
- Works well with small, clean datasets; performance degrades with high-dimensional or noisy AIS data.
- Choose *k* via cross-validation; larger *k* reduces noise sensitivity.

## Run
```bash
python main.py
```
