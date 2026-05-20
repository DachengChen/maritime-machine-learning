# Association Rule Learning — Unsupervised Learning

## How It Works
Association rule mining discovers co-occurrence patterns of the form "if {A, B} then {C}" in transaction-like datasets. The Apriori algorithm first enumerates frequent itemsets (combinations that appear in at least *min_support* fraction of transactions), then generates rules from those itemsets that exceed a *min_confidence* threshold. The lift metric measures how much more often A and C co-occur than expected by chance. Pruning via the anti-monotone property (a superset cannot be more frequent than its subset) makes the search tractable.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Port calling pattern mining** | Discover that vessels calling at Port A almost always also call at Port B within the same voyage |
| **Vessel behaviour co-occurrence** | Identify that high draught + low SOG + specific heading frequently co-occur (laden vessel approaching port) |
| **Anomaly rule extraction** | Find rare but significant patterns: AIS-off + re-emergence near a sanctioned port |
| **Cargo type inference** | Associate ship dimensions + route with implied cargo type from historical records |

## AIS Features Typically Used
- `navigationalstatus` — Discretised operational state
- `shiptype` — Vessel category
- `sog` binned (e.g. anchored, slow, normal, fast)
- Port / region visited (engineered from position)

## Notes
- Requires discretisation of continuous AIS features (SOG, COG, draught) into bins.
- The current implementation is a brute-force Apriori stub; use `mlxtend` for production datasets.

## Run
```bash
python main.py
```
