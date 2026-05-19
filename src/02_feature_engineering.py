"""
feature_engineering.py
=======================
Transform cleaned AIS data into model-ready features.

Features produced
-----------------
Temporal
    hour_sin, hour_cos          – cyclical encoding of hour-of-day
    dow_sin, dow_cos            – cyclical encoding of day-of-week
    month_sin, month_cos        – cyclical encoding of month

Kinematic
    delta_sog                   – change in SOG between consecutive reports
    delta_cog                   – change in COG (shortest angular distance)
    distance_nm                 – great-circle distance to previous report (nm)
    elapsed_seconds             – seconds since previous report for same vessel

Positional
    lat_sin, lat_cos            – cyclical latitude encoding
    lon_sin, lon_cos            – cyclical longitude encoding

Aggregated (per-voyage statistics)
    mean_sog, std_sog           – SOG statistics over the voyage so far
    mean_cog, std_cog           – COG statistics over the voyage so far
"""

import numpy as np
import pandas as pd
from geopy.distance import great_circle


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time-of-day, day-of-week, and month features.

    Requires a ``timestamp`` column with ``datetime64`` dtype.
    """
    df = df.copy()
    ts = df["timestamp"]

    hour = ts.dt.hour + ts.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    dow = ts.dt.dayofweek.astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    month = (ts.dt.month - 1).astype(float)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    return df


# ---------------------------------------------------------------------------
# Kinematic features
# ---------------------------------------------------------------------------

def _angular_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    """Shortest signed angular difference *a – b* in [-180, 180] degrees."""
    diff = (a - b + 180) % 360 - 180
    return diff


def add_kinematic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-step kinematic delta features.

    Computes differences between consecutive position reports for the same
    vessel (grouped by ``mmsi``).  The first report for each vessel gets
    ``NaN`` deltas.
    """
    df = df.copy()
    grp = df.groupby("mmsi", sort=False)

    df["delta_sog"] = grp["sog"].diff()
    df["delta_cog"] = _angular_diff(df["cog"], grp["cog"].shift(1))

    # Elapsed seconds since last report
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        prev_ts = grp["timestamp"].shift(1)
        df["elapsed_seconds"] = (df["timestamp"] - prev_ts).dt.total_seconds()
    else:
        df["elapsed_seconds"] = np.nan

    # Great-circle distance to previous report (nautical miles)
    prev_lat = grp["lat"].shift(1)
    prev_lon = grp["lon"].shift(1)

    def _haversine_nm(row):
        if pd.isna(row["prev_lat"]):
            return np.nan
        return great_circle(
            (row["lat"], row["lon"]),
            (row["prev_lat"], row["prev_lon"])
        ).nautical

    tmp = df[["lat", "lon"]].copy()
    tmp["prev_lat"] = prev_lat.values
    tmp["prev_lon"] = prev_lon.values
    df["distance_nm"] = tmp.apply(_haversine_nm, axis=1)

    return df


# ---------------------------------------------------------------------------
# Positional / spatial features
# ---------------------------------------------------------------------------

def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical latitude and longitude encodings."""
    df = df.copy()
    df["lat_sin"] = np.sin(np.deg2rad(df["lat"]))
    df["lat_cos"] = np.cos(np.deg2rad(df["lat"]))
    df["lon_sin"] = np.sin(np.deg2rad(df["lon"]))
    df["lon_cos"] = np.cos(np.deg2rad(df["lon"]))
    return df


# ---------------------------------------------------------------------------
# Aggregated voyage features
# ---------------------------------------------------------------------------

def add_voyage_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add expanding-window voyage statistics (mean/std of SOG and COG)."""
    df = df.copy()
    grp = df.groupby("mmsi", sort=False)

    df["mean_sog"] = grp["sog"].transform(lambda x: x.expanding().mean())
    df["std_sog"] = grp["sog"].transform(lambda x: x.expanding().std())
    df["mean_cog"] = grp["cog"].transform(lambda x: x.expanding().mean())
    df["std_cog"] = grp["cog"].transform(lambda x: x.expanding().std())

    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering pipeline.

    Parameters
    ----------
    df:
        Cleaned AIS data frame (output of :func:`~src.data_preprocessing.preprocess`).

    Returns
    -------
    pd.DataFrame
        Data frame with all engineered features appended.
    """
    df = add_temporal_features(df)
    df = add_kinematic_features(df)
    df = add_spatial_features(df)
    df = add_voyage_aggregate_features(df)
    return df


# ---------------------------------------------------------------------------
# Feature-matrix helpers
# ---------------------------------------------------------------------------

#: Default feature columns used by the predictive models.
DEFAULT_FEATURE_COLS = [
    "lat", "lon",
    "sog", "cog",
    "lat_sin", "lat_cos", "lon_sin", "lon_cos",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "delta_sog", "delta_cog",
    "distance_nm", "elapsed_seconds",
    "mean_sog", "std_sog",
    "mean_cog", "std_cog",
]


def get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Return a feature matrix ready for scikit-learn estimators.

    Parameters
    ----------
    df:
        Feature-engineered AIS data frame.
    feature_cols:
        Columns to include.  Defaults to :data:`DEFAULT_FEATURE_COLS`.
    dropna:
        Drop rows that contain any ``NaN`` in the selected columns.

    Returns
    -------
    pd.DataFrame
        Feature matrix (subset of *df*).
    """
    if feature_cols is None:
        feature_cols = [c for c in DEFAULT_FEATURE_COLS if c in df.columns]
    X = df[feature_cols]
    if dropna:
        X = X.dropna()
    return X
