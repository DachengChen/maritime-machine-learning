# DRNN — Deep Learning

## How It Works
A Deep Recurrent Neural Network (DRNN) stacks multiple RNN layers where the full hidden-state sequence output of layer $l$ becomes the input sequence to layer $l+1$. With $L$ layers the representation hierarchy looks like:

$$h_t^{(l)} = \tanh\!\left(W_h^{(l)} h_{t-1}^{(l)} + W_x^{(l)} h_t^{(l-1)} + b^{(l)}\right)$$

Lower layers capture fine-grained short-range patterns (e.g. sudden speed changes) while upper layers encode longer-range dynamics (e.g. voyage phases). Dropout is applied between intermediate layers to regularise the deeper network.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Multi-scale trajectory modelling** | Lower layers capture short bursts of manoeuvring; upper layers encode voyage-level structure |
| **Hierarchical behaviour recognition** | Distinguish fishing micro-manoeuvres (lower layers) from overall fishing trip patterns (upper layers) |
| **Long-range ETA forecasting** | Deep hierarchy allows the model to learn both local speed fluctuations and global routing patterns |
| **Anomaly scoring** | Rich multi-layer representations improve discrimination between normal and abnormal voyage segments |

## AIS Features Typically Used
- Time-ordered `sog`, `cog`, `heading` sequences per MMSI
- Time delta between consecutive AIS messages
- Latitude / longitude sequences (if available)
- `navigationalstatus` as a categorical context feature

## Notes
- Deeper networks amplify the vanishing-gradient problem; apply gradient clipping (`torch.nn.utils.clip_grad_norm_`) during training.
- Dropout between layers is critical to prevent overfitting with many parameters.
- For sequences longer than ~30 timesteps, prefer Deep LSTM or Deep GRU to mitigate gradient issues.

## Run
```bash
python main.py
```
