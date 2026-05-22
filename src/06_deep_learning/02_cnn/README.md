# 1-D CNN — Deep Learning

## How It Works
A 1-D Convolutional Neural Network applies learnable filters along the time axis of a sequence. Each filter slides over consecutive time steps, computing a dot product to produce a feature map that captures local patterns (e.g. a sudden speed change spanning 3 steps). Multiple convolutional layers stack to detect increasingly abstract temporal motifs. Pooling layers (max or average) reduce dimensionality and provide translation invariance. The final representation is flattened and passed to dense layers for classification or regression. 1-D CNNs train faster than RNNs and parallelise well.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Manoeuvre detection** | Detect short local patterns in AIS tracks — acceleration bursts, sharp turns, U-turns — using convolutional filters over a sliding window of position reports |
| **Vessel activity classification** | Classify a fixed-length track window as fishing, transiting, anchoring, or port approach |
| **AIS message quality screening** | Identify sequences of consecutive implausible reports (impossible speed jumps) with local pattern filters |
| **Short-term speed forecasting** | Predict next-step SOG from a recent fixed-length speed window |

## AIS Features Typically Used
- Sequential `sog` — Speed time series (primary input channel)
- Sequential `cog`, `heading` — Course / heading channels
- Multi-channel input: stack SOG, COG, heading as separate channels for each time step

## Notes
- Best for detecting **local** temporal patterns; use LSTM/Transformer for long-range dependencies.
- Input must be a fixed-length sequence; segment and pad raw AIS tracks accordingly.

## Run
```bash
python main.py
```
