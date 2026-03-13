"""
Tests for etl/feature_engineering.py

Verifies that each feature-engineering function:
  1. Produces correct output types and shapes
  2. Handles edge cases (NaN, zeros, missing data)
  3. Is deterministic (same seed → same result)
  4. Produces values within expected ranges

Run:
    cd /Users/wangzimeiyi/Desktop/Practicum
    /usr/bin/python3 -m pytest tests/test_feature_engineering.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from etl.feature_engineering import (
    DEFAULT_CPI_TABLE,
    DEFAULT_PPI_TABLE,
    QUANTITY_BIN_LABELS,
    adjust_cost_by_ppi,
    assign_geo_cluster,
    assign_nearest_acf,
    assign_scope_cluster,
    build_acf_state_norm_lookup,
    compute_acf_state_norm,
    compute_geo_clusters,
    compute_inflation_factor,
    compute_quantity_bins,
    compute_scope_clusters,
    engineer_advanced_features,
    stratified_sample,
)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "Previous work"
INFLATION_CSV = (
    DATA_DIR / "ACF" / "inflation_ACF" / "data" / "projects_clusters_log_outliers.csv"
)


# ---------------------------------------------------------------------------
# 1. Inflation / PPI adjustment
# ---------------------------------------------------------------------------


class TestComputeInflationFactor:
    def test_same_year_returns_one(self):
        assert compute_inflation_factor(2025, base_year=2025) == pytest.approx(1.0)

    def test_older_year_returns_greater_than_one(self):
        factor = compute_inflation_factor(2010, base_year=2025)
        assert factor > 1.0

    def test_newer_year_returns_less_than_one(self):
        factor = compute_inflation_factor(2025, base_year=2010)
        assert factor < 1.0

    def test_missing_year_returns_one(self):
        factor = compute_inflation_factor(1800, base_year=2025)
        assert factor == pytest.approx(1.0)

    def test_custom_cpi_table(self):
        custom = {2020: 100.0, 2025: 120.0}
        factor = compute_inflation_factor(2020, base_year=2025, cpi_table=custom)
        assert factor == pytest.approx(1.2)

    def test_known_value(self):
        factor = compute_inflation_factor(2020, base_year=2024)
        expected = DEFAULT_CPI_TABLE[2024] / DEFAULT_CPI_TABLE[2020]
        assert factor == pytest.approx(expected)


class TestAdjustCostByPPI:
    def test_same_year_returns_original(self):
        assert adjust_cost_by_ppi(1_000_000, 2025, base_year=2025) == pytest.approx(
            1_000_000
        )

    def test_older_year_inflates(self):
        adjusted = adjust_cost_by_ppi(1_000_000, 2010, base_year=2025)
        assert adjusted > 1_000_000

    def test_missing_year_returns_original(self):
        adjusted = adjust_cost_by_ppi(500_000, 1800, base_year=2025)
        assert adjusted == pytest.approx(500_000)

    def test_custom_ppi_table(self):
        custom = {2020: 200.0, 2025: 300.0}
        adjusted = adjust_cost_by_ppi(1_000_000, 2020, base_year=2025, ppi_table=custom)
        assert adjusted == pytest.approx(1_500_000)


# ---------------------------------------------------------------------------
# 2. ACF state normalization
# ---------------------------------------------------------------------------


class TestComputeACFStateNorm:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "project_state": ["CA", "CA", "TX", "TX", "NY", "NY"],
                "acf": [1.2, 1.3, 0.9, 0.8, 1.0, 1.1],
            }
        )

    def test_returns_series(self, sample_df):
        result = compute_acf_state_norm(sample_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_df)

    def test_mean_approximately_one(self, sample_df):
        result = compute_acf_state_norm(sample_df)
        assert result.mean() == pytest.approx(1.0, abs=0.15)

    def test_high_acf_state_above_one(self, sample_df):
        result = compute_acf_state_norm(sample_df)
        # CA has highest ACF (1.25 mean) → should be above overall norm
        ca_norm = result.iloc[0]  # first CA row
        tx_norm = result.iloc[2]  # first TX row
        assert ca_norm > tx_norm

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(ValueError, match="acf"):
            compute_acf_state_norm(df)


class TestBuildACFStateNormLookup:
    def test_returns_dict(self):
        df = pd.DataFrame(
            {
                "project_state": ["CA", "CA", "TX", "TX"],
                "acf": [1.2, 1.3, 0.9, 0.8],
            }
        )
        lookup = build_acf_state_norm_lookup(df)
        assert isinstance(lookup, dict)
        assert "CA" in lookup
        assert "TX" in lookup

    def test_values_sum_to_n_states(self):
        df = pd.DataFrame(
            {
                "project_state": ["CA", "TX"],
                "acf": [1.0, 1.0],
            }
        )
        lookup = build_acf_state_norm_lookup(df)
        assert all(v == pytest.approx(1.0) for v in lookup.values())


# ---------------------------------------------------------------------------
# 3. Quantity binning
# ---------------------------------------------------------------------------


class TestComputeQuantityBins:
    def test_returns_categorical_series(self):
        quantities = pd.Series([5, 50, 500, 5000, 50000, 500000])
        result = compute_quantity_bins(quantities)
        assert len(result) == 6

    def test_correct_bin_assignments(self):
        quantities = pd.Series([5, 50, 500, 5000, 50000, 500000])
        result = compute_quantity_bins(quantities)
        expected = ["0-10", "10-100", "100-1,000", "1,000-10,000", "10,000-100,000", "100,000+"]
        assert list(result.astype(str)) == expected

    def test_edge_value_10(self):
        result = compute_quantity_bins(pd.Series([10]))
        assert str(result.iloc[0]) == "0-10"

    def test_nan_produces_nan(self):
        result = compute_quantity_bins(pd.Series([np.nan]))
        assert pd.isna(result.iloc[0])

    def test_custom_bins(self):
        result = compute_quantity_bins(
            pd.Series([5, 15]),
            bins=[0, 10, 20],
            labels=["small", "medium"],
        )
        assert str(result.iloc[0]) == "small"
        assert str(result.iloc[1]) == "medium"

    def test_mismatched_bins_labels_raises(self):
        with pytest.raises(ValueError, match="len"):
            compute_quantity_bins(pd.Series([1]), bins=[0, 10], labels=["a", "b"])


# ---------------------------------------------------------------------------
# 4. Scope clustering
# ---------------------------------------------------------------------------


class TestComputeScopeClusters:
    @pytest.fixture
    def sample_df(self):
        rng = np.random.RandomState(42)
        n = 200
        return pd.DataFrame(
            {
                "total_mat_lab_equip": rng.uniform(100_000, 50_000_000, n),
                "median_quantity_most_common_unit": rng.uniform(1, 10_000, n),
            }
        )

    def test_returns_series_and_model(self, sample_df):
        labels, km = compute_scope_clusters(sample_df, n_clusters=5)
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(sample_df)
        assert hasattr(km, "predict")

    def test_correct_number_of_clusters(self, sample_df):
        labels, _ = compute_scope_clusters(sample_df, n_clusters=5)
        assert labels.nunique() <= 5

    def test_deterministic(self, sample_df):
        labels1, _ = compute_scope_clusters(sample_df, n_clusters=5)
        labels2, _ = compute_scope_clusters(sample_df, n_clusters=5)
        assert (labels1 == labels2).all()

    def test_handles_zero_quantity(self):
        df = pd.DataFrame(
            {
                "total_mat_lab_equip": [100_000, 200_000, 300_000, 400_000],
                "median_quantity_most_common_unit": [0, 0, 100, 200],
            }
        )
        labels, _ = compute_scope_clusters(df, n_clusters=2)
        assert len(labels) == 4
        assert not labels.isna().any()


class TestAssignScopeCluster:
    def test_assigns_integer(self):
        df = pd.DataFrame(
            {
                "total_mat_lab_equip": [100_000, 1_000_000, 10_000_000],
                "median_quantity_most_common_unit": [100, 1_000, 10_000],
            }
        )
        _, km = compute_scope_clusters(df, n_clusters=2)
        label = assign_scope_cluster(500_000, 500, km)
        assert isinstance(label, int)
        assert 0 <= label < 2

    def test_zero_quantity_handled(self):
        df = pd.DataFrame(
            {
                "total_mat_lab_equip": [100_000, 1_000_000],
                "median_quantity_most_common_unit": [100, 1_000],
            }
        )
        _, km = compute_scope_clusters(df, n_clusters=2)
        label = assign_scope_cluster(100_000, 0, km)
        assert isinstance(label, int)


# ---------------------------------------------------------------------------
# 5. Geographic clustering
# ---------------------------------------------------------------------------


class TestComputeGeoClusters:
    @pytest.fixture
    def coords(self):
        return (
            pd.Series([34.0, 40.7, 41.8, 29.7, 47.6]),  # lat
            pd.Series([-118.2, -74.0, -87.6, -95.3, -122.3]),  # lon
        )

    def test_returns_series_and_model(self, coords):
        labels, km = compute_geo_clusters(*coords, n_clusters=2)
        assert isinstance(labels, pd.Series)
        assert len(labels) == 5
        assert hasattr(km, "predict")

    def test_labels_are_region_strings(self, coords):
        labels, _ = compute_geo_clusters(*coords, n_clusters=2)
        assert all(lbl.startswith("Region_") for lbl in labels)

    def test_deterministic(self, coords):
        labels1, _ = compute_geo_clusters(*coords, n_clusters=2)
        labels2, _ = compute_geo_clusters(*coords, n_clusters=2)
        assert (labels1 == labels2).all()

    def test_handles_nan_coords(self):
        lat = pd.Series([34.0, np.nan, 41.8, 29.7])
        lon = pd.Series([-118.2, -74.0, np.nan, -95.3])
        labels, _ = compute_geo_clusters(lat, lon, n_clusters=2)
        assert len(labels) == 4
        assert not labels.isna().any()


class TestAssignGeoCluster:
    def test_returns_region_string(self):
        lat = pd.Series([34.0, 40.7, 41.8])
        lon = pd.Series([-118.2, -74.0, -87.6])
        _, km = compute_geo_clusters(lat, lon, n_clusters=2)
        label = assign_geo_cluster(37.0, -122.0, km)
        assert isinstance(label, str)
        assert label.startswith("Region_")


# ---------------------------------------------------------------------------
# 6. Nearest-neighbor ACF
# ---------------------------------------------------------------------------


class TestAssignNearestACF:
    def test_single_neighbor(self):
        project_coords = np.array([[40.0, -75.0]])
        ref_coords = np.array([[40.0, -74.0], [34.0, -118.0]])
        ref_acf = np.array([1.1, 1.3])
        result = assign_nearest_acf(project_coords, ref_coords, ref_acf)
        assert result[0] == pytest.approx(1.1)

    def test_multiple_projects(self):
        project_coords = np.array([[40.0, -75.0], [34.0, -117.0]])
        ref_coords = np.array([[40.0, -74.0], [34.0, -118.0]])
        ref_acf = np.array([1.1, 1.3])
        result = assign_nearest_acf(project_coords, ref_coords, ref_acf)
        assert len(result) == 2
        assert result[0] == pytest.approx(1.1)
        assert result[1] == pytest.approx(1.3)

    def test_k_neighbors_average(self):
        project_coords = np.array([[40.0, -75.0]])
        ref_coords = np.array([[40.0, -74.0], [40.0, -76.0], [34.0, -118.0]])
        ref_acf = np.array([1.0, 1.2, 1.5])
        result = assign_nearest_acf(project_coords, ref_coords, ref_acf, n_neighbors=2)
        assert result[0] == pytest.approx(1.1)  # mean of 1.0 and 1.2

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError, match="empty"):
            assign_nearest_acf(
                np.array([[40.0, -75.0]]),
                np.array([]).reshape(0, 2),
                np.array([]),
            )

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            assign_nearest_acf(
                np.array([[40.0, -75.0]]),
                np.array([[40.0, -74.0]]),
                np.array([1.0, 1.1]),
            )


# ---------------------------------------------------------------------------
# 7. Stratified sampling
# ---------------------------------------------------------------------------


class TestStratifiedSample:
    @pytest.fixture
    def sample_df(self):
        rng = np.random.RandomState(42)
        n = 500
        return pd.DataFrame(
            {
                "region": rng.choice(["R0", "R1", "R2", "R3"], n),
                "quantity_bin": rng.choice(["small", "medium", "large"], n),
                "cost": rng.uniform(10000, 1000000, n),
            }
        )

    def test_reduces_row_count(self, sample_df):
        result = stratified_sample(sample_df, ["region", "quantity_bin"], max_per_stratum=10)
        assert len(result) < len(sample_df)

    def test_preserves_rare_categories(self):
        # Create a df where "RARE" appears only twice
        df = pd.DataFrame(
            {
                "region": ["RARE", "RARE"] + ["COMMON"] * 100,
                "quantity_bin": ["small"] * 102,
                "cost": range(102),
            }
        )
        result = stratified_sample(df, ["region"], max_per_stratum=10, min_frequency=5)
        rare_count = (result["region"] == "RARE").sum()
        assert rare_count == 2

    def test_deterministic(self, sample_df):
        r1 = stratified_sample(sample_df, ["region"], max_per_stratum=10)
        r2 = stratified_sample(sample_df, ["region"], max_per_stratum=10)
        assert r1.index.equals(r2.index)

    def test_no_duplicates_in_result(self, sample_df):
        result = stratified_sample(sample_df, ["region", "quantity_bin"], max_per_stratum=10)
        assert not result.index.duplicated().any()


# ---------------------------------------------------------------------------
# 8. engineer_advanced_features (integration)
# ---------------------------------------------------------------------------


class TestEngineerAdvancedFeatures:
    @pytest.fixture
    def sample_df(self):
        rng = np.random.RandomState(42)
        n = 100
        return pd.DataFrame(
            {
                "project_state": rng.choice(["CA", "TX", "NY", "PA"], n),
                "acf": rng.uniform(0.8, 1.3, n),
                "total_mat_lab_equip": rng.uniform(100_000, 10_000_000, n),
                "median_quantity_most_common_unit": rng.uniform(1, 10_000, n),
                "project_latitude": rng.uniform(30, 45, n),
                "project_longitude": rng.uniform(-120, -70, n),
                "project_year": rng.choice([2015, 2018, 2020, 2022, 2024], n),
            }
        )

    def test_adds_expected_columns(self, sample_df):
        result, _ = engineer_advanced_features(sample_df)
        for col in [
            "inflation_factor",
            "acf_state_norm",
            "quantity_bin",
            "scope_cluster",
            "region",
            "ppi_adjusted_cost",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_returns_artifacts_dict(self, sample_df):
        _, artifacts = engineer_advanced_features(sample_df)
        assert "scope_km" in artifacts
        assert "geo_km" in artifacts
        assert "acf_state_norm_lookup" in artifacts

    def test_row_count_unchanged(self, sample_df):
        result, _ = engineer_advanced_features(sample_df)
        assert len(result) == len(sample_df)

    def test_artifacts_are_reusable(self, sample_df):
        _, artifacts = engineer_advanced_features(sample_df)
        scope_km = artifacts["scope_km"]
        geo_km = artifacts["geo_km"]
        label = assign_scope_cluster(1_000_000, 500, scope_km)
        assert isinstance(label, int)
        region = assign_geo_cluster(40.0, -75.0, geo_km)
        assert region.startswith("Region_")

    def test_inflation_factor_positive(self, sample_df):
        result, _ = engineer_advanced_features(sample_df)
        assert (result["inflation_factor"] > 0).all()


# ---------------------------------------------------------------------------
# 9. Integration with real CSV data
# ---------------------------------------------------------------------------


class TestIntegrationWithRealData:
    @pytest.fixture(scope="class")
    def real_df(self):
        if not INFLATION_CSV.exists():
            pytest.skip(f"CSV not found: {INFLATION_CSV}")
        return pd.read_csv(INFLATION_CSV)

    def test_quantity_bins_on_real_data(self, real_df):
        if "median_quantity_most_common_unit" not in real_df.columns:
            pytest.skip("Column not in CSV")
        result = compute_quantity_bins(
            real_df["median_quantity_most_common_unit"].fillna(0)
        )
        assert len(result) == len(real_df)
        assert not result.isna().all()

    def test_scope_clusters_on_real_data(self, real_df):
        cost_col = "total_mat_lab_equip"
        qty_col = "median_quantity_most_common_unit"
        if cost_col not in real_df.columns:
            pytest.skip(f"Column '{cost_col}' not in CSV")
        if qty_col not in real_df.columns:
            pytest.skip(f"Column '{qty_col}' not in CSV")
        labels, km = compute_scope_clusters(
            real_df, cost_col=cost_col, quantity_col=qty_col, n_clusters=15
        )
        assert labels.nunique() <= 15
        assert len(labels) == len(real_df)
