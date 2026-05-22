# LSTM Autoencoder — Deep Learning

## How It Works
An Autoencoder learns to compress input data into a low-dimensional latent representation (encoding) and then reconstruct the original input from it (decoding). The LSTM Autoencoder applies this idea to sequences: an encoder LSTM reads the input sequence and produces a fixed-size latent vector; a decoder LSTM receives that vector and reconstructs the sequence step by step. During training, only *normal* examples are used, so the model learns what normal looks like. At inference time, anomalies — patterns the model has never seen — produce high reconstruction error, which serves as an anomaly score.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **AIS spoofing detection** | Flag vessels whose reported track has unusually high reconstruction error, indicating impossible kinematics (teleportation, impossible turns) |
| **Dark vessel behaviour** | Detect anomalous movement patterns after vessels re-emerge following an AIS transponder shutdown |
| **Fishing vessel poaching detection** | Reconstruct normal fishing patterns; deviations (loitering near protected zones) trigger high error |
| **Equipment malfunction flagging** | Identify sensor faults producing systematic drift in reported SOG or heading values |

## AIS Features Typically Used
- Time-ordered `sog`, `cog`, `heading` per MMSI (sequence input)
- Reconstruction error threshold set from the 95th percentile of training-set errors
- `shiptype` stratification (train separate models per vessel type for tighter baselines)

## Notes
- Only normal data is needed for training — no anomaly labels required.
- The reconstruction error threshold controls precision vs. recall; tune per operational context.

## Run
```bash
python main.py
```
