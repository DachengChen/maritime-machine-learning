# Neural Collaborative Filtering — Recommendation Systems

## How It Works
Neural Collaborative Filtering (NCF) replaces the dot-product interaction of matrix factorisation with a neural network that can learn arbitrary non-linear user–item interactions. It combines two paths: a **Generalised Matrix Factorisation (GMF)** branch that element-wise multiplies user and item embeddings (preserving linear MF), and an **MLP branch** that concatenates them and passes them through several dense layers. The outputs of both branches are fused and passed to a final prediction layer. Training uses binary cross-entropy (implicit feedback) or MSE (explicit ratings). Learnable embeddings mean both paths improve jointly end-to-end.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Personalised port recommendation** | Learn complex non-linear vessel–port affinities that SVD misses — e.g. a vessel may prefer a port only when its draught is above a threshold |
| **Route personalisation** | Model how subtle vessel characteristics (speed profile, flag state, operator) influence route preferences |
| **Dynamic berth allocation** | Learn berth–vessel compatibility from historical allocation data including implicit signals (turnaround time, revisit rate) |
| **Fleet scheduling assistance** | Suggest optimal next-port assignments that balance operational history with real-time port availability |

## AIS / Interaction Data
- Vessel–port visit matrix (implicit: 1 = visited, 0 = not sampled)
- Vessel embedding: initialised from `shiptype`, `length`, `draught` features
- Port embedding: initialised from port depth, cargo type, region

## Notes
- Outperforms plain SVD on complex, non-linear interaction patterns with sufficient data.
- Requires more data and tuning than classical MF; start with SVD as the baseline.

## Run
```bash
python main.py
```
