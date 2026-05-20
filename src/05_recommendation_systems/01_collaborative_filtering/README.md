# User-Based Collaborative Filtering — Recommendation Systems

## How It Works
Collaborative Filtering recommends items based on the preferences of *similar users*, without using any item content features. User-based CF computes a similarity score (e.g. cosine similarity or Pearson correlation) between every pair of users from their rating/interaction histories. To predict a target user's preference for an unseen item, it identifies the *k* most similar users who have interacted with that item and takes a weighted average of their ratings. The core assumption is that users who agreed in the past will agree in the future.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Port-of-call recommendation** | Suggest the next port for a vessel based on the historical port sequences of similar vessels (same type, trade route, flag) |
| **Anchorage zone recommendation** | Recommend waiting anchorages based on what similar vessels have chosen under similar traffic conditions |
| **Fuel-stop scheduling** | Identify preferred bunkering ports from vessels with similar routes and consumption profiles |
| **Route option ranking** | Surface alternative routes preferred by historically similar vessels during adverse weather or congestion |

## AIS Features / Interaction Data
- Vessel–port visit history (vessel = user, port = item, visit frequency = implicit rating)
- `shiptype`, `length`, `draught` — Used to define vessel similarity groups
- `navigationalstatus` at port arrival/departure

## Notes
- Suffers from cold-start problem for new vessels with no voyage history.
- Sparsity of vessel–port interaction matrices can be addressed with matrix factorisation (see `03_matrix_factorization`).

## Run
```bash
python main.py
```
