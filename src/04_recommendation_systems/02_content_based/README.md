# Content-Based Filtering — Recommendation Systems

## How It Works
Content-Based Filtering recommends items similar to those a user has liked before, using only the features of items (and optionally users) rather than other users' preferences. Each item is represented as a feature vector (its "profile"), and a user profile is constructed as the weighted mean of the feature vectors of items the user has rated. Recommendations are the unseen items whose feature vectors are most similar (cosine similarity) to the user profile. The method is fully explainable: recommendations can be justified by specific feature matches.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Berth assignment recommendation** | Recommend berths whose physical characteristics (depth, length, cargo handling equipment) match the vessel's requirements |
| **Vessel-to-route matching** | Suggest routes with environmental conditions (distance, current, depth) that match the vessel's historical preferences |
| **Similar vessel identification** | Find vessels with similar operational profiles (speed range, route geography, ship type) for benchmarking |
| **Cold-start port recommendation** | For a new vessel with no history, recommend ports that match its physical profile (draught, LOA, cargo type) |

## AIS / Item Features Typically Used
- `length`, `width`, `draught` — Vessel / berth physical compatibility
- `shiptype` — One-hot encoded category
- `sog` range, `cog` distribution — Operational profile features
- Port characteristics: depth, berth length, cargo type handled (external data)

## Notes
- No other-user data required; works from day one for any vessel with known characteristics.
- Limited to recommending items similar to what the vessel has already experienced ("filter bubble").

## Run
```bash
python main.py
```
