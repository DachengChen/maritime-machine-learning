"""
test_feature_engineering.py
============================
Unit tests for src/feature_engineering.py.
"""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feature_engineering import (
    add_temporal_features,
    add_kinematic_features,
    add_spatial_features,
    add_voyage_aggregate_features,
    build_features,
    get_feature_matrix,
    DEFAULT_FEATURE_COLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ais_df():
    """Small preprocessed AIS data frame."""
    return pd.DataFrame({
        "mmsi": [111, 111, 111, 222, 222, 222],
        "timestamp": pd.to_datetime([
            "2024-03-15 06:00", "2024-03-15 07:00", "2024-03-15 08:00",
            "2024-03-15 12:00", "2024-03-15 13:00", "2024-03-15 14:00",
        ], utc=True),
        "lat": [51.0, 51.1, 51.2, 53.0, 53.1, 53.2],
        "lon": [4.0,  4.1,  4.2,  9.9, 10.0, 10.1],
        "sog": [10.0, 11.0, 10.5,  8.0,  8.5,  9.0],
        "cog": [45.0, 46.0, 47.0, 90.0, 91.0, 92.0],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTemporalFeatures:
    def test_adds_hour_features(self, ais_df):
        result = add_temporal_features(ais_df)
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns

    def test_adds_dow_features(self, ais_df):
        result = add_temporal_features(ais_df)
        assert "dow_sin" in result.columns and "dow_cos" in result.columns

    def test_adds_month_features(self, ais_df):
        result = add_temporal_features(ais_df)
        assert "month_sin" in result.columns and "month_cos" in result.columns

    def test_sin_cos_range(self, ais_df):
        result = add_temporal_features(ais_df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            assert result[col].between(-1, 1).all(), f"{col} out of [-1, 1]"


class TestKinematicFeatures:
    def test_adds_delta_sog(self, ais_df):
        result = add_kinematic_features(ais_df)
        assert "delta_sog" in result.columns

    def test_adds_delta_cog(self, ais_df):
        result = add_kinematic_features(ais_df)
        assert "delta_cog" in result.columns

    def test_adds_distance_nm(self, ais_df):
        result = add_kinematic_features(ais_df)
        assert "distance_nm" in result.columns

    def test_first_report_delta_is_nan(self, ais_df):
        result = add_kinematic_features(ais_df)
        # First report per vessel should have NaN delta (use nth(0) to get the
        # literal first row, not the first non-null value returned by first())
        first_rows = result.groupby("mmsi", sort=False).nth(0)
        assert first_rows["delta_sog"].isna().all()

    def test_distance_positive(self, ais_df):
        result = add_kinematic_features(ais_df)
        valid = result["distance_nm"].dropna()
        assert (valid >= 0).all()


class TestSpatialFeatures:
    def test_adds_lat_lon_sin_cos(self, ais_df):
        result = add_spatial_features(ais_df)
        for col in ["lat_sin", "lat_cos", "lon_sin", "lon_cos"]:
            assert col in result.columns

    def test_range(self, ais_df):
        result = add_spatial_features(ais_df)
        for col in ["lat_sin", "lat_cos", "lon_sin", "lon_cos"]:
            assert result[col].between(-1, 1).all()


class TestVoyageAggregateFeatures:
    def test_adds_mean_std_sog(self, ais_df):
        result = add_voyage_aggregate_features(ais_df)
        assert "mean_sog" in result.columns and "std_sog" in result.columns

    def test_mean_sog_monotone_group(self, ais_df):
        result = add_voyage_aggregate_features(ais_df)
        # mean_sog for first report of each vessel equals that vessel's first SOG
        first_rows = result.groupby("mmsi").first()
        for mmsi, row in first_rows.iterrows():
            expected = ais_df[ais_df["mmsi"] == mmsi]["sog"].iloc[0]
            assert abs(row["mean_sog"] - expected) < 1e-6


class TestBuildFeatures:
    def test_all_feature_groups_present(self, ais_df):
        result = build_features(ais_df)
        for col in ["hour_sin", "lat_sin", "delta_sog", "mean_sog"]:
            assert col in result.columns

    def test_no_new_rows(self, ais_df):
        result = build_features(ais_df)
        assert len(result) == len(ais_df)


class TestGetFeatureMatrix:
    def test_returns_subset_of_cols(self, ais_df):
        featured = build_features(ais_df)
        X = get_feature_matrix(featured, feature_cols=["lat", "lon", "sog"])
        assert list(X.columns) == ["lat", "lon", "sog"]

    def test_dropna_removes_nan_rows(self, ais_df):
        featured = build_features(ais_df)
        # First rows have NaN kinematic features; dropna should reduce count
        X_all = get_feature_matrix(featured, dropna=False)
        X_clean = get_feature_matrix(featured, dropna=True)
        assert len(X_clean) <= len(X_all)
