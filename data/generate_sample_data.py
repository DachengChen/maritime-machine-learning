"""
generate_sample_data.py
=======================
Generate a small synthetic AIS dataset suitable for smoke-testing the
maritime-machine-learning pipeline.

Run directly to produce ``data/sample/ais_sample.csv``:

    python data/generate_sample_data.py
"""

import os
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_VESSELS = 20
REPORTS_PER_VESSEL = (30, 80)   # (min, max) position reports per vessel

# Synthetic ports (name, lat, lon)
PORTS = [
    ("ROTTERDAM", 51.9, 4.5),
    ("HAMBURG", 53.5, 9.9),
    ("ANTWERP", 51.2, 4.4),
    ("FELIXSTOWE", 51.9, 1.3),
    ("LE_HAVRE", 49.5, 0.1),
    ("BREMERHAVEN", 53.5, 8.6),
    ("AMSTERDAM", 52.4, 4.9),
    ("ZEEBRUGGE", 51.3, 3.2),
]

VESSEL_TYPES = [70, 71, 72, 80, 89]   # Cargo & tanker type codes

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)


def _random_port(exclude_idx: int | None = None):
    choices = [i for i in range(len(PORTS)) if i != exclude_idx]
    return random.choice(choices)


def _interpolate_route(
    src_lat: float, src_lon: float,
    dst_lat: float, dst_lon: float,
    n_points: int,
    noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate a route with small Gaussian noise."""
    t = np.linspace(0, 1, n_points)
    lats = src_lat + t * (dst_lat - src_lat) + rng.normal(0, noise, n_points)
    lons = src_lon + t * (dst_lon - src_lon) + rng.normal(0, noise, n_points)
    return lats, lons


def generate_ais_sample(
    n_vessels: int = N_VESSELS,
    reports_range: tuple[int, int] = REPORTS_PER_VESSEL,
    add_anomalies: bool = True,
) -> pd.DataFrame:
    """Generate a synthetic AIS dataset.

    Parameters
    ----------
    n_vessels:
        Number of unique MMSI identifiers to generate.
    reports_range:
        (min, max) number of position reports per vessel.
    add_anomalies:
        If ``True``, inject a handful of anomalous reports (high speed,
        unusual position).

    Returns
    -------
    pd.DataFrame
        Synthetic AIS data with the following columns:
        mmsi, timestamp, lat, lon, sog, cog, heading,
        vessel_type, destination, draught, length.
    """
    rows = []
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    for vessel_idx in range(n_vessels):
        mmsi = 200000000 + vessel_idx * 1000 + rng.integers(1, 999)
        src_idx = _random_port()
        dst_idx = _random_port(exclude_idx=src_idx)
        src = PORTS[src_idx]
        dst = PORTS[dst_idx]

        n_reports = int(rng.integers(*reports_range))
        lats, lons = _interpolate_route(src[1], src[2], dst[1], dst[2], n_reports)

        # Spread reports over the voyage duration (6-48 hours)
        voyage_hours = rng.uniform(6, 48)
        timestamps = [
            base_time + timedelta(hours=float(t))
            for t in np.linspace(0, voyage_hours, n_reports)
        ]
        # Offset base time so vessels don't all start at the same moment
        base_time += timedelta(hours=rng.uniform(0.5, 4))

        sog_mean = rng.uniform(8, 18)
        vessel_type = int(rng.choice(VESSEL_TYPES))
        draught = round(float(rng.uniform(4, 14)), 1)
        length = int(rng.integers(100, 400))

        for i in range(n_reports):
            if i == 0:
                cog = float(rng.uniform(0, 360))
            else:
                # COG roughly follows the heading towards destination
                dlat = lats[min(i + 1, n_reports - 1)] - lats[i]
                dlon = lons[min(i + 1, n_reports - 1)] - lons[i]
                cog = (np.degrees(np.arctan2(dlon, dlat)) + 360) % 360

            sog = float(max(0, rng.normal(sog_mean, 2)))
            heading = (cog + rng.uniform(-5, 5)) % 360

            rows.append({
                "mmsi": int(mmsi),
                "timestamp": timestamps[i].isoformat(),
                "lat": round(lats[i], 6),
                "lon": round(lons[i], 6),
                "sog": round(sog, 1),
                "cog": round(cog, 1),
                "heading": round(heading, 1),
                "vessel_type": vessel_type,
                "destination": dst[0],
                "draught": draught,
                "length": length,
            })

    df = pd.DataFrame(rows)

    if add_anomalies:
        # Inject 5 anomalous records (very high speed, unusual position)
        anomaly_rows = []
        for _ in range(5):
            base_row = df.sample(1, random_state=int(rng.integers(9999))).iloc[0].to_dict()
            base_row["sog"] = round(float(rng.uniform(60, 100)), 1)   # implausibly fast
            base_row["lat"] = round(float(rng.uniform(30, 35)), 6)     # off-route
            base_row["lon"] = round(float(rng.uniform(-10, -5)), 6)
            anomaly_rows.append(base_row)
        df = pd.concat([df, pd.DataFrame(anomaly_rows)], ignore_index=True)

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "sample")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ais_sample.csv")

    df = generate_ais_sample()
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")
