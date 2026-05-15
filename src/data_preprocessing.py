"""
data_preprocessing.py
=====================
Utilities for loading, cleaning, and normalising raw AIS data.

Expected raw AIS columns (subset used throughout this project):
    mmsi          – Maritime Mobile Service Identity (vessel identifier)
    timestamp     – UTC timestamp of position report
    lat           – Latitude  (-90 … 90)
    lon           – Longitude (-180 … 180)
    sog           – Speed Over Ground (knots)
    cog           – Course Over Ground (degrees, 0-360)
    heading       – True heading (degrees, 0-360 or 511 = not available)
    vessel_type   – Numeric vessel-type code (AIS field 8)
    destination   – Free-text destination field
    draught       – Static draught (metres)
    length        – Overall vessel length (metres)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Physically implausible value ranges – records outside these are dropped
VALID_LAT = (-90.0, 90.0)
VALID_LON = (-180.0, 180.0)
VALID_SOG = (0.0, 102.3)   # 102.3 kn is the AIS max encoded value
VALID_COG = (0.0, 360.0)
VALID_HEADING = (0, 360)   # 511 means "not available"

# Minimum number of position reports required to keep a trajectory
MIN_TRAJECTORY_POINTS = 3


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_ais_csv(filepath: str, parse_dates: bool = True) -> pd.DataFrame:
    """Load a CSV file containing AIS data.

    Parameters
    ----------
    filepath:
        Path to the CSV file.
    parse_dates:
        If *True* (default), attempt to parse the ``timestamp`` column as
        UTC-aware ``datetime64``.

    Returns
    -------
    pd.DataFrame
        Raw AIS data frame.
    """
    date_cols = ["timestamp"] if parse_dates else []
    df = pd.read_csv(filepath, parse_dates=date_cols)
    if parse_dates and "timestamp" in df.columns:
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df


# ---------------------------------------------------------------------------
# Validation / cleaning helpers
# ---------------------------------------------------------------------------

def remove_invalid_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with physically impossible latitude/longitude values."""
    lat_ok = df["lat"].between(*VALID_LAT)
    lon_ok = df["lon"].between(*VALID_LON)
    return df[lat_ok & lon_ok].copy()


def remove_invalid_sog(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where Speed Over Ground is outside the valid AIS range."""
    return df[df["sog"].between(*VALID_SOG)].copy()


def remove_invalid_cog(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where Course Over Ground is outside [0, 360]."""
    return df[df["cog"].between(*VALID_COG)].copy()


def replace_unavailable_heading(df: pd.DataFrame, fill_value: float = np.nan) -> pd.DataFrame:
    """Replace heading value 511 (AIS 'not available') with *fill_value*.

    Parameters
    ----------
    df:
        Data frame containing a ``heading`` column.
    fill_value:
        Replacement value (default ``NaN``).
    """
    df = df.copy()
    df.loc[df["heading"] == 511, "heading"] = fill_value
    return df


def drop_duplicate_reports(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate position reports (same MMSI + timestamp)."""
    return df.drop_duplicates(subset=["mmsi", "timestamp"]).copy()


def remove_short_trajectories(df: pd.DataFrame, min_points: int = MIN_TRAJECTORY_POINTS) -> pd.DataFrame:
    """Remove vessels with fewer than *min_points* position reports."""
    counts = df.groupby("mmsi")["mmsi"].transform("count")
    return df[counts >= min_points].copy()


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise_destination(df: pd.DataFrame) -> pd.DataFrame:
    """Upper-case and strip whitespace from the ``destination`` field."""
    if "destination" not in df.columns:
        return df
    df = df.copy()
    df["destination"] = df["destination"].astype(str).str.strip().str.upper()
    df.loc[df["destination"].isin(["NAN", "", "0", "NONE"]), "destination"] = np.nan
    return df


def sort_by_vessel_time(df: pd.DataFrame) -> pd.DataFrame:
    """Sort data by MMSI then timestamp (ascending) and reset index."""
    return df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, min_points: int = MIN_TRAJECTORY_POINTS) -> pd.DataFrame:
    """Run the full preprocessing pipeline on a raw AIS data frame.

    Steps
    -----
    1. Remove invalid positions.
    2. Remove invalid SOG values.
    3. Remove invalid COG values.
    4. Replace unavailable heading (511) with NaN.
    5. Drop exact duplicate (mmsi, timestamp) pairs.
    6. Normalise destination strings.
    7. Remove vessels with fewer than *min_points* position reports.
    8. Sort by vessel and time, reset index.

    Parameters
    ----------
    df:
        Raw AIS data frame (output of :func:`load_ais_csv`).
    min_points:
        Minimum trajectory length to retain.

    Returns
    -------
    pd.DataFrame
        Cleaned AIS data frame.
    """
    df = remove_invalid_positions(df)
    df = remove_invalid_sog(df)
    df = remove_invalid_cog(df)
    df = replace_unavailable_heading(df)
    df = drop_duplicate_reports(df)
    df = normalise_destination(df)
    df = remove_short_trajectories(df, min_points=min_points)
    df = sort_by_vessel_time(df)
    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a concise statistical summary of the AIS data frame.

    Returns a :class:`pd.DataFrame` with per-column statistics including
    count, null count, mean, min, and max for numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = df[numeric_cols].describe().T
    stats["null_count"] = df[numeric_cols].isna().sum()
    return stats
