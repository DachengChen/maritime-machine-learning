# Transformer — Deep Learning

## How It Works
The Transformer replaces recurrence with **self-attention**: every position in the sequence directly attends to every other position, computing a weighted sum of value vectors based on query–key similarity. Multi-head attention runs several attention operations in parallel, each learning different relational patterns. Positional encodings inject order information since the architecture is otherwise permutation-invariant. Stacked encoder layers build increasingly abstract global representations of the entire sequence simultaneously, enabling massive parallelisation during training and strong long-range dependency modelling.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Long-voyage trajectory modelling** | Attend to distant past events (departure port, weather window) that are relevant to the current vessel state — something LSTMs struggle with |
| **Multi-vessel interaction modelling** | Apply cross-attention between the tracks of multiple vessels in the same sea area to predict collision risk or traffic flow |
| **Voyage intent classification** | Classify the overall purpose of a full voyage (ballast, laden, port-hopping) from the complete AIS track |
| **Route recommendation** | Encode historical route sequences and decode optimal next-waypoint recommendations |

## AIS Features Typically Used
- Full-voyage `sog`, `cog`, `heading` sequences per MMSI
- Positional encoding over timestamps (absolute or relative time deltas)
- `shiptype` as a learnable embedding prepended to the sequence
- Multi-vessel: encode each track independently, then cross-attend

## Notes
- Requires significantly more data than LSTM to generalise well.
- For short sequences (< 50 steps) a GRU is likely more efficient; Transformer shines on long, complex routes.

## Run
```bash
python main.py
```
