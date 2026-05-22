# BRNN — Deep Learning

## How It Works
A Bidirectional Recurrent Neural Network (BRNN) runs two vanilla RNNs over the same sequence — one in the **forward** direction and one in the **backward** direction — and concatenates their hidden states at each timestep. This gives the model access to both past and future context, improving classification and labelling tasks where the full sequence is available at inference time. The output feature dimension is doubled ($2 \times \text{hidden\_size}$) compared to a unidirectional RNN.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Voyage segment classification** | Label each AIS point as *anchoring*, *transit*, *manoeuvring*, etc., using both preceding and following context |
| **Post-hoc anomaly labelling** | Identify anomalous segments in a completed voyage where the full track is available |
| **Port stay detection** | Determine the exact start and end of a port call by reading the sequence in both directions |
| **Activity recognition** | Classify fishing, towing, or cargo operations using the full AIS track of a trip |

## AIS Features Typically Used
- Time-ordered `sog`, `cog`, `heading` sequences per MMSI
- Time delta between consecutive AIS messages
- Latitude / longitude sequences (if available)
- `navigationalstatus` as a categorical context feature

## Notes
- Not suitable for real-time prediction because the backward pass requires the full future sequence.
- Like vanilla RNNs, susceptible to vanishing gradients on very long sequences; consider Bi-LSTM or Bi-GRU for long voyages.
- Inference cost is roughly double that of a unidirectional RNN.

## Run
```bash
python main.py
```
