# GRU — Deep Learning

## How It Works
A Gated Recurrent Unit (GRU) is a streamlined variant of the LSTM that uses only two gates: the **reset gate** controls how much of the previous hidden state to forget when computing the candidate state, and the **update gate** blends the old hidden state with the new candidate. Removing the separate cell state reduces the number of parameters by ~25% compared to an LSTM, leading to faster training and lower memory use with similar sequence-modelling capability. GRUs are preferred when computational resources are limited or datasets are moderate in size.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Real-time vessel tracking** | Efficiently predict the next AIS position report in a live stream with lower latency than an LSTM |
| **Onboard / edge deployment** | Lightweight enough to run on vessel computers or VTS edge servers for real-time anomaly flagging |
| **Short-to-medium voyage forecasting** | Model SOG and COG sequences over coastal passages where long-range memory is less critical |
| **AIS gap filling** | Predict missing position reports during communication blackouts using recent track history |

## AIS Features Typically Used
- Time-ordered `sog`, `cog`, `heading` sequences per MMSI
- Time delta between consecutive messages (handles irregular sampling)
- `shiptype` as a static embedding

## Notes
- Equivalent performance to LSTM on most maritime sequence tasks with fewer parameters.
- Prefer LSTM if sequences are very long (> 1000 steps) and training data is abundant.

## Run
```bash
python main.py
```
