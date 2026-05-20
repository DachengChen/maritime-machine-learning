# k-means Clustering — Unsupervised Learning

## How It Works
k-means partitions *n* data points into *k* clusters by iterating two steps: (1) assign each point to the nearest centroid, and (2) recompute each centroid as the mean of its assigned points. This continues until assignments stop changing. The algorithm minimises the within-cluster sum of squares (inertia). Because initialisation matters, k-means++ seeds centroids far apart to improve convergence. The number of clusters *k* must be chosen in advance — the elbow method or silhouette scores help select it.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Sea lane discovery** | Cluster vessel tracks by COG and position to automatically identify dominant trade routes |
| **Behavioural profiling** | Group vessels by movement patterns (slow-steaming cargo, fast ferry, drifting fishing) without any labels |
| **Anchorage zone detection** | Cluster near-zero-SOG position reports to map waiting areas around busy ports |
| **Traffic density segmentation** | Segment ocean regions into high, medium, and low traffic zones for maritime surveillance |

## AIS Features Typically Used
- `sog`, `cog` — Movement signature
- `heading` — Direction
- Latitude, longitude (from raw AIS if available) — Spatial clustering
- `shiptype` — Optional post-hoc cluster labelling

## Notes
- Sensitive to feature scale — always normalise before clustering.
- k-means assumes spherical clusters; use DBSCAN for irregular-shaped sea lane shapes.

## Run
```bash
python main.py
```
