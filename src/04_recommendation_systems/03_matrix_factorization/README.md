# Matrix Factorisation (SVD) — Recommendation Systems

## How It Works
Matrix Factorisation decomposes the sparse user–item interaction matrix **R** (shape: users × items) into two low-rank matrices: a user embedding matrix **U** and an item embedding matrix **V**, such that **R ≈ U × Vᵀ**. Each row of **U** is a latent vector representing a user's preferences in a compressed space; each row of **V** is a latent vector representing an item's properties. Truncated SVD (used here) computes the top-*k* singular values/vectors. Missing entries (unseen interactions) are filled by the product **u · vᵀ**, providing predictions for unrated items.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Latent route pattern discovery** | Decompose the vessel–port visit matrix to uncover hidden trade patterns (e.g. tanker circuit, feeder loop) encoded in latent dimensions |
| **Port affinity scoring** | Predict the affinity of a vessel for an unvisited port by reconstructing its row in the interaction matrix |
| **Fleet segmentation** | Cluster vessels in the latent embedding space to group them by hidden operational similarity |
| **Sparse AIS data imputation** | Fill missing voyage history for vessels with limited AIS records by leveraging global patterns in the factorised matrix |

## AIS / Interaction Data
- Vessel–port visit frequency matrix (vessel = user, port = item)
- `shiptype`, `draught` — Used for post-hoc analysis of latent dimensions
- Number of visits or dwell time as the rating signal

## Notes
- SVD requires a fully observed matrix for exact computation; use alternating least squares (ALS) or SGD for very sparse matrices.
- Latent dimensions can be inspected to understand which vessel and port clusters they capture.

## Run
```bash
python main.py
```
