"""
test_data_preprocessing.py
===========================
Unit tests for src/data_preprocessing.py.
"""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_preprocessing import (
    remove_invalid_positions,
    remove_invalid_sog,
    remove_invalid_cog,
    replace_unavailable_heading,
    drop_duplicate_reports,
    remove_short_trajectories,
    normalise_destination,
    sort_by_vessel_time,
    preprocess,
    data_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df():
    """Minimal valid AIS data frame for testing."""
    return pd.DataFrame({
        "mmsi":        [111, 111, 111, 222, 222, 222, 333, 333, 333],
        "timestamp":   pd.to_datetime([
            "2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00",
            "2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00",
            "2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00",
        ], utc=True),
        "lat":         [51.5, 51.6, 51.7,  53.5, 53.6, 53.7,  49.0, 49.1, 49.2],
        "lon":         [4.0,  4.1,  4.2,   9.9, 10.0, 10.1,   0.1,  0.2,  0.3],
        "sog":         [10.0, 11.0, 12.0,   8.0,  8.5,  9.0,  14.0, 13.5, 13.0],
        "cog":         [45.0, 46.0, 47.0,  90.0, 91.0, 92.0, 180.0, 181.0, 182.0],
        "heading":     [45.0, 46.0, 47.0,  90.0, 91.0, 92.0, 180.0, 181.0, 182.0],
        "destination": ["ROTTERDAM", "ROTTERDAM", "ROTTERDAM",
                        "HAMBURG",   "HAMBURG",   "HAMBURG",
                        "LE_HAVRE",  "LE_HAVRE",  "LE_HAVRE"],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRemoveInvalidPositions:
    def test_removes_out_of_range_lat(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "lat"] = 99.0          # invalid
        result = remove_invalid_positions(df)
        assert len(result) == len(sample_df) - 1

    def test_removes_out_of_range_lon(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "lon"] = 200.0         # invalid
        result = remove_invalid_positions(df)
        assert len(result) == len(sample_df) - 1

    def test_keeps_valid_rows(self, sample_df):
        result = remove_invalid_positions(sample_df)
        assert len(result) == len(sample_df)


class TestRemoveInvalidSog:
    def test_removes_negative_sog(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "sog"] = -1.0
        result = remove_invalid_sog(df)
        assert len(result) == len(sample_df) - 1

    def test_removes_over_limit_sog(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "sog"] = 110.0
        result = remove_invalid_sog(df)
        assert len(result) == len(sample_df) - 1


class TestReplaceUnavailableHeading:
    def test_replaces_511(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "heading"] = 511
        result = replace_unavailable_heading(df)
        assert pd.isna(result.loc[0, "heading"])

    def test_leaves_valid_headings(self, sample_df):
        result = replace_unavailable_heading(sample_df)
        assert result["heading"].notna().all()


class TestDropDuplicateReports:
    def test_drops_exact_duplicates(self, sample_df):
        df = pd.concat([sample_df, sample_df.iloc[:2]], ignore_index=True)
        result = drop_duplicate_reports(df)
        assert len(result) == len(sample_df)


class TestRemoveShortTrajectories:
    def test_removes_short_vessel(self, sample_df):
        # Add a vessel with only 1 report
        new_row = sample_df.iloc[0:1].copy()
        new_row["mmsi"] = 999
        df = pd.concat([sample_df, new_row], ignore_index=True)
        result = remove_short_trajectories(df, min_points=3)
        assert 999 not in result["mmsi"].values

    def test_keeps_long_enough_vessels(self, sample_df):
        result = remove_short_trajectories(sample_df, min_points=3)
        assert len(result) == len(sample_df)


class TestNormaliseDestination:
    def test_uppercases_and_strips(self):
        df = pd.DataFrame({"destination": [" rotterdam ", "Hamburg", "  LE HAVRE  "]})
        result = normalise_destination(df)
        assert list(result["destination"]) == ["ROTTERDAM", "HAMBURG", "LE HAVRE"]

    def test_replaces_zero_string(self):
        df = pd.DataFrame({"destination": ["0", "NaN", "NONE"]})
        result = normalise_destination(df)
        assert result["destination"].isna().all()


class TestPreprocess:
    def test_returns_dataframe(self, sample_df):
        result = preprocess(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_sorted_by_mmsi_timestamp(self, sample_df):
        shuffled = sample_df.sample(frac=1, random_state=0).reset_index(drop=True)
        result = preprocess(shuffled)
        assert (result["mmsi"].diff().fillna(0) >= 0).all()


class TestDataSummary:
    def test_returns_dataframe(self, sample_df):
        result = data_summary(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_null_count_column(self, sample_df):
        result = data_summary(sample_df)
        assert "null_count" in result.columns
