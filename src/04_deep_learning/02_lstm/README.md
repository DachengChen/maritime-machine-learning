# LSTM — Deep Learning

## How It Works
A Long Short-Term Memory (LSTM) is a recurrent neural network cell designed to capture long-range dependencies in sequences. Each LSTM cell maintains a hidden state *h* and a cell state *c* controlled by three gates: the **forget gate** decides which information to discard, the **input gate** decides what new information to store, and the **output gate** determines what the cell state contributes to the output. This gating mechanism prevents the vanishing-gradient problem that cripples vanilla RNNs on long sequences, making LSTMs well-suited to variable-length time series.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Trajectory prediction** | Predict the next *n* positions of a vessel by learning its long-range movement pattern from a full voyage track |
| **Multi-step ETA forecasting** | Model the entire voyage sequence — port departures, sea passages, congestion delays — to forecast arrival time |
| **Speed profile modelling** | Learn how a vessel's SOG evolves over an entire voyage, capturing slow-steaming and acceleration phases |
| **Behaviour change detection** | Identify the time step at which a vessel's behaviour deviates from its learned historical pattern |

## AIS Features Typically Used
- Time-ordered `sog`, `cog`, `heading` sequences per MMSI
- Time delta between consecutive AIS messages
- Latitude / longitude sequences (if available)
- `shiptype`, `navigationalstatus` as static context features

## Notes
- Requires sequences sorted by MMSI and timestamp; handle irregular sampling with time-delta features.
- Stacked LSTM layers improve representation capacity but increase training time.

## Run
```bash
python main.py
```
