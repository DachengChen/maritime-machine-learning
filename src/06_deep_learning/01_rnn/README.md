# RNN — Deep Learning

## How It Works
A vanilla Recurrent Neural Network (RNN) processes sequences one timestep at a time, passing a hidden state forward to capture temporal dependencies. At each step the hidden state is updated as $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$. While simple and computationally cheap, vanilla RNNs suffer from the **vanishing-gradient problem** on long sequences, making it difficult to learn dependencies that span many timesteps. They are a useful baseline before escalating to LSTM or GRU architectures.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Short-horizon trajectory prediction** | Predict the next position of a vessel over a short window where long-range dependencies are less critical |
| **Speed / course change detection** | Learn typical SOG/COG transitions to flag abrupt deviations in near real-time |
| **Port entry sequence modelling** | Capture the ordered sequence of navigational status changes during a port approach |
| **Lightweight anomaly baseline** | Serve as a fast, low-cost anomaly detector where latency matters more than accuracy |

## AIS Features Typically Used
- Time-ordered `sog`, `cog`, `heading` sequences per MMSI
- Time delta between consecutive AIS messages
- Latitude / longitude sequences (if available)
- `navigationalstatus` as a categorical context feature

## Notes
- Prone to vanishing gradients on sequences longer than ~20 timesteps; prefer LSTM/GRU for long voyages.
- Stacking layers adds capacity but amplifies gradient issues; keep `num_layers` low (1–2).

## Run
```bash
python main.py
```
