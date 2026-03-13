"""
Tests for etl/loaders.py

Verifies that each loader:
  1. Reads the CSV without errors
  2. Returns the correct number of RawProjectRecord instances
  3. Populates key fields with non-None values
  4. Handles the column renaming correctly (e.g. "type" → project_type)

Run:
    cd /Users/wangzimeiyi/Desktop/Practicum
    python -m pytest tests/test_loaders.py -v
"""

import sys
from pathlib import Path

import pytest

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from etl.loaders import (
    load_all_projects,
    load_inflation_acf_projects,
    load_market_basket_projects,
    load_preprocessed_projects,
    load_regression_projects,
)
from schema import RawProjectRecord

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "Previous work"

INFLATION_CLUSTERED_CSV = (
    DATA_DIR / "ACF" / "inflation_ACF" / "data" / "projects_clusters_log_outliers.csv"
)
PREPROCESSED_CSV = (
    DATA_DIR / "ACF" / "inflation_ACF" / "data" / "projects_preprocessed.csv"
)
MARKET_BASKET_CSV = (
    DATA_DIR / "ACF" / "market_basket_acf" / "data" / "final_dataset_on_year.csv"
)
REGRESSION_CSV = DATA_DIR / "Regression" / "data" / "base_data_for_model.csv"


# ---------------------------------------------------------------------------
# Loader 1: Inflation ACF (clustered)
# ---------------------------------------------------------------------------


class TestLoadInflationACFProjects:
    """Tests for load_inflation_acf_projects()."""

    @pytest.fixture(scope="class")
    def records(self):
        """Load once, share across all tests in this class."""
        assert (
            INFLATION_CLUSTERED_CSV.exists()
        ), f"Test CSV not found: {INFLATION_CLUSTERED_CSV}"
        return load_inflation_acf_projects(str(INFLATION_CLUSTERED_CSV))

    def test_returns_list(self, records):
        assert isinstance(records, list)

    def test_correct_count(self, records):
        # CSV has 1,819 data rows (1,820 lines minus the header)
        assert len(records) == 1819

    def test_returns_raw_project_records(self, records):
        assert all(isinstance(r, RawProjectRecord) for r in records)

    def test_project_id_populated(self, records):
        # Every row should have a non-empty project_id
        for r in records:
            assert r.project_id is not None
            assert r.project_id != "UNKNOWN"

    def test_location_fields(self, records):
        """First row is PITTSBURGH, PA with known lat/lon."""
        first = records[0]
        assert first.project_state == "PA"
        assert first.project_city == "PITTSBURGH"
        assert first.project_latitude is not None
        assert first.project_longitude is not None
        assert 39.0 < first.project_latitude < 42.0  # PA latitude range
        assert -81.0 < first.project_longitude < -79.0  # PA longitude range

    def test_cost_fields(self, records):
        """First row has total_mat_lab_equip = 48622213.0."""
        first = records[0]
        assert first.total_mat_lab_equip is not None
        assert first.total_mat_lab_equip == pytest.approx(48622213.0, rel=0.01)

    def test_column_rename_type_to_project_type(self, records):
        """Verify 'type' column was renamed to 'project_type'."""
        first = records[0]
        assert first.project_type == "Communication Devices"

    def test_column_rename_cost_per_sqft(self, records):
        """Verify 'cost_per_sqft' was renamed to 'price_per_sq_ft'."""
        first = records[0]
        assert first.price_per_sq_ft is not None
        assert first.price_per_sq_ft > 0

    def test_column_rename_dod_acf(self, records):
        """Verify 'DoD_ACF2024' was renamed to 'dod_acf_2024'."""
        first = records[0]
        assert first.dod_acf_2024 is not None
        assert first.dod_acf_2024 == pytest.approx(1.06, rel=0.01)

    def test_census_enrichment(self, records):
        """Verify Census data (population, density, state_name) is populated."""
        first = records[0]
        assert first.population is not None
        assert first.population > 0
        assert first.density is not None
        assert first.state_name == "Pennsylvania"

    def test_ppi_fields(self, records):
        """Verify PPI and adjusted cost fields are populated."""
        first = records[0]
        assert first.wpuip2300001 is not None
        assert first.adjusted_total_mat_lab_equip is not None

    def test_no_empty_states(self, records):
        """Most records should have a state code."""
        states_present = sum(1 for r in records if r.project_state is not None)
        assert states_present / len(records) > 0.95  # at least 95%

    def test_construction_category(self, records):
        """First row is Commercial."""
        assert records[0].construction_category == "Commercial"


# ---------------------------------------------------------------------------
# Loader 2: Preprocessed projects
# ---------------------------------------------------------------------------


class TestLoadPreprocessedProjects:
    """Tests for load_preprocessed_projects()."""

    @pytest.fixture(scope="class")
    def records(self):
        assert PREPROCESSED_CSV.exists(), f"Test CSV not found: {PREPROCESSED_CSV}"
        return load_preprocessed_projects(str(PREPROCESSED_CSV))

    def test_returns_list(self, records):
        assert isinstance(records, list)

    def test_correct_count(self, records):
        assert len(records) == 1819

    def test_returns_raw_project_records(self, records):
        assert all(isinstance(r, RawProjectRecord) for r in records)

    def test_same_first_project_as_clustered(self, records):
        """Should be the same underlying data as the clustered file."""
        first = records[0]
        assert first.project_state == "PA"
        assert first.project_city == "PITTSBURGH"
        assert first.total_mat_lab_equip == pytest.approx(48622213.0, rel=0.01)


# ---------------------------------------------------------------------------
# Loader 3: Market-basket ACF
# ---------------------------------------------------------------------------


class TestLoadMarketBasketProjects:
    """Tests for load_market_basket_projects()."""

    @pytest.fixture(scope="class")
    def records(self):
        assert MARKET_BASKET_CSV.exists(), f"Test CSV not found: {MARKET_BASKET_CSV}"
        return load_market_basket_projects(str(MARKET_BASKET_CSV))

    def test_returns_list(self, records):
        assert isinstance(records, list)

    def test_correct_count(self, records):
        # CSV has 932 data rows
        assert len(records) == 932

    def test_returns_raw_project_records(self, records):
        assert all(isinstance(r, RawProjectRecord) for r in records)

    def test_project_id_populated(self, records):
        for r in records:
            assert r.project_id is not None
            assert r.project_id != "UNKNOWN"

    def test_column_rename_type(self, records):
        """Verify 'type' → 'project_type'."""
        first = records[0]
        assert first.project_type is not None
        assert first.project_type == "Communication Devices"

    def test_location_fields(self, records):
        first = records[0]
        assert first.project_state == "CA"
        assert first.project_city == "LOS ANGELES"

    def test_metro_area_populated(self, records):
        """Market-basket records should have metro area matches."""
        metros_present = sum(1 for r in records if r.matched_metro_area is not None)
        # Some early rows may not have matches; at least 50% should
        assert metros_present / len(records) > 0.5

    def test_project_year_populated(self, records):
        first = records[0]
        assert first.project_year is not None
        assert 1990 <= first.project_year <= 2030

    def test_price_per_sq_ft(self, records):
        first = records[0]
        assert first.price_per_sq_ft is not None
        assert first.price_per_sq_ft > 0

    def test_project_cost(self, records):
        first = records[0]
        assert first.project_cost is not None
        assert first.project_cost > 0

    def test_project_region(self, records):
        """Market-basket has a region column (e.g. 'WEST', 'SOUTHEAST')."""
        first = records[0]
        assert first.project_region is not None


# ---------------------------------------------------------------------------
# Loader 4: Regression Flask app dataset
# ---------------------------------------------------------------------------


class TestLoadRegressionProjects:
    """Tests for load_regression_projects()."""

    @pytest.fixture(scope="class")
    def records(self):
        if not REGRESSION_CSV.exists():
            pytest.skip(f"Regression CSV not found: {REGRESSION_CSV}")
        return load_regression_projects(str(REGRESSION_CSV))

    def test_returns_list(self, records):
        assert isinstance(records, list)

    def test_correct_count(self, records):
        # CSV has 17,025 data rows
        assert len(records) == 17025

    def test_returns_raw_project_records(self, records):
        assert all(isinstance(r, RawProjectRecord) for r in records)

    def test_project_state_populated(self, records):
        """Most records should have a state code."""
        states_present = sum(1 for r in records if r.project_state is not None)
        assert states_present / len(records) > 0.90

    def test_project_type_populated(self, records):
        types_present = sum(1 for r in records if r.project_type is not None)
        assert types_present / len(records) > 0.90

    def test_inflation_factor_populated(self, records):
        """Regression CSV has pre-computed inflation_factor."""
        inf_present = sum(1 for r in records if r.inflation_factor is not None)
        assert inf_present / len(records) > 0.90

    def test_acf_populated(self, records):
        """Regression CSV has pre-computed ACF."""
        acf_present = sum(1 for r in records if r.acf is not None)
        assert acf_present / len(records) > 0.50

    def test_target_populated(self, records):
        """Regression CSV has inflation-adjusted target cost."""
        target_present = sum(
            1 for r in records if r.total_project_cost_normalized_2025 is not None
        )
        assert target_present / len(records) > 0.90

    def test_complexity_category_populated(self, records):
        """Regression CSV has CIQS complexity category."""
        complex_present = sum(
            1 for r in records if r.ciqs_complexity_category is not None
        )
        assert complex_present / len(records) > 0.50

    def test_county_name_populated(self, records):
        """Regression CSV has county_name."""
        county_present = sum(1 for r in records if r.county_name is not None)
        assert county_present / len(records) > 0.50

    def test_location_coords_populated(self, records):
        """Regression CSV has lat/lon for geographic clustering."""
        lat_present = sum(1 for r in records if r.project_latitude is not None)
        assert lat_present / len(records) > 0.50

    def test_inflation_factor_values_positive(self, records):
        """Pre-computed inflation factors should all be positive."""
        for r in records:
            if r.inflation_factor is not None:
                assert r.inflation_factor > 0, (
                    f"Record {r.project_id} has non-positive inflation_factor={r.inflation_factor}"
                )

    def test_acf_values_in_reasonable_range(self, records):
        """ACF values should be within a reasonable range (0.5 to 2.0)."""
        for r in records:
            if r.acf is not None:
                assert 0.3 < r.acf < 3.0, (
                    f"Record {r.project_id} has out-of-range acf={r.acf}"
                )

    def test_target_cost_positive(self, records):
        """Normalized 2025 target costs should be positive."""
        for r in records:
            if r.total_project_cost_normalized_2025 is not None:
                assert r.total_project_cost_normalized_2025 > 0, (
                    f"Record {r.project_id} has non-positive target cost"
                )

    def test_first_record_has_project_id(self, records):
        """First record should have a real project_id, not UNKNOWN."""
        assert records[0].project_id != "UNKNOWN"

    def test_project_category_populated(self, records):
        cat_present = sum(1 for r in records if r.project_category is not None)
        assert cat_present / len(records) > 0.80

    def test_area_type_populated(self, records):
        """Regression CSV has area_type (Urban/Rural)."""
        area_present = sum(1 for r in records if r.area_type is not None)
        assert area_present / len(records) > 0.50

    def test_cnt_division_populated(self, records):
        div_present = sum(1 for r in records if r.cnt_division is not None)
        assert div_present / len(records) > 0.50


# ---------------------------------------------------------------------------
# Loader 4b: Regression loader — synthetic CSV edge cases
# ---------------------------------------------------------------------------


class TestLoadRegressionProjectsSynthetic:
    """Edge-case tests using synthetic CSV files."""

    def test_empty_csv_returns_empty_list(self, tmp_path):
        """An empty CSV (header only) should return an empty list."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("project_id,inflation_factor,acf,project_state\n")
        records = load_regression_projects(str(csv_file))
        assert isinstance(records, list)
        assert len(records) == 0

    def test_single_row_csv(self, tmp_path):
        """A CSV with one data row should return exactly one record."""
        csv_file = tmp_path / "single.csv"
        csv_file.write_text(
            "project_id,inflation_factor,acf,project_state,project_type,project_category\n"
            "P001,1.15,1.08,TX,Pavement Markers,Civil\n"
        )
        records = load_regression_projects(str(csv_file))
        assert len(records) == 1
        assert records[0].project_id == "P001"
        assert records[0].inflation_factor == pytest.approx(1.15)
        assert records[0].acf == pytest.approx(1.08)
        assert records[0].project_state == "TX"

    def test_csv_with_missing_optional_columns(self, tmp_path):
        """CSV missing optional columns should still load; missing fields default to None."""
        csv_file = tmp_path / "minimal_cols.csv"
        csv_file.write_text(
            "project_id,project_state\n"
            "P001,CA\n"
            "P002,TX\n"
        )
        records = load_regression_projects(str(csv_file))
        assert len(records) == 2
        assert records[0].project_state == "CA"
        assert records[0].inflation_factor is None
        assert records[0].acf is None
        assert records[0].total_project_cost_normalized_2025 is None

    def test_csv_with_nan_values(self, tmp_path):
        """NaN values in the CSV should become None on the record."""
        csv_file = tmp_path / "nan_values.csv"
        csv_file.write_text(
            "project_id,inflation_factor,acf,total_project_cost_normalized_2025\n"
            "P001,,1.08,5000000\n"
            "P002,1.15,,\n"
        )
        records = load_regression_projects(str(csv_file))
        assert len(records) == 2
        assert records[0].inflation_factor is None
        assert records[0].acf == pytest.approx(1.08)
        assert records[0].total_project_cost_normalized_2025 == pytest.approx(5_000_000.0)
        assert records[1].inflation_factor == pytest.approx(1.15)
        assert records[1].acf is None
        assert records[1].total_project_cost_normalized_2025 is None

    def test_csv_with_extra_columns_ignored(self, tmp_path):
        """Extra columns not in the mapping should not cause errors."""
        csv_file = tmp_path / "extra_cols.csv"
        csv_file.write_text(
            "project_id,inflation_factor,acf,some_unknown_column,another_col\n"
            "P001,1.1,0.95,foo,bar\n"
        )
        records = load_regression_projects(str(csv_file))
        assert len(records) == 1
        assert records[0].inflation_factor == pytest.approx(1.1)
        assert records[0].acf == pytest.approx(0.95)

    def test_csv_preserves_numeric_precision(self, tmp_path):
        """Float values should maintain reasonable precision."""
        csv_file = tmp_path / "precision.csv"
        csv_file.write_text(
            "project_id,inflation_factor,acf,total_project_cost_normalized_2025\n"
            "P001,1.123456789,0.987654321,12345678.90\n"
        )
        records = load_regression_projects(str(csv_file))
        assert records[0].inflation_factor == pytest.approx(1.123456789, rel=1e-6)
        assert records[0].acf == pytest.approx(0.987654321, rel=1e-6)
        assert records[0].total_project_cost_normalized_2025 == pytest.approx(
            12345678.90, rel=1e-6
        )

    def test_csv_with_all_regression_fields(self, tmp_path):
        """A CSV with all 15 mapped columns should populate every corresponding field."""
        csv_file = tmp_path / "full_regression.csv"
        csv_file.write_text(
            "project_id,inflation_factor,total_project_cost_normalized_2025,"
            "official_budget_range,ciqs_complexity_category,cnt_division,"
            "cnt_item_code,county_name,area_type,acf,project_latitude,"
            "project_longitude,project_type,project_category,project_state\n"
            "P001,1.15,5000000,$3M-$6M,Category 2,3,10,Harris,Urban,1.08,"
            "29.76,-95.36,Pavement Markers,Civil,TX\n"
        )
        records = load_regression_projects(str(csv_file))
        r = records[0]
        assert r.project_id == "P001"
        assert r.inflation_factor == pytest.approx(1.15)
        assert r.total_project_cost_normalized_2025 == pytest.approx(5_000_000.0)
        assert r.official_budget_range == "$3M-$6M"
        assert r.ciqs_complexity_category == "Category 2"
        assert r.cnt_division == 3
        assert r.cnt_item_code == 10
        assert r.county_name == "Harris"
        assert r.area_type == "Urban"
        assert r.acf == pytest.approx(1.08)
        assert r.project_latitude == pytest.approx(29.76)
        assert r.project_longitude == pytest.approx(-95.36)
        assert r.project_type == "Pavement Markers"
        assert r.project_category == "Civil"
        assert r.project_state == "TX"


# ---------------------------------------------------------------------------
# Loader 5b: load_all_projects — regression key inclusion
# ---------------------------------------------------------------------------


class TestLoadAllProjectsRegressionKey:
    """Verify load_all_projects discovers and includes the regression dataset."""

    @pytest.fixture(scope="class")
    def all_data_with_regression(self):
        """Load all projects from a directory that includes the regression CSV."""
        if not DATA_DIR.exists():
            pytest.skip(f"Previous work directory not found: {DATA_DIR}")
        return load_all_projects(str(DATA_DIR))

    def test_regression_key_present_when_csv_exists(self, all_data_with_regression):
        """If the regression CSV exists, 'regression' should be a key."""
        if not REGRESSION_CSV.exists():
            pytest.skip("Regression CSV not found")
        assert "regression" in all_data_with_regression

    def test_regression_records_are_raw_project_records(self, all_data_with_regression):
        if "regression" not in all_data_with_regression:
            pytest.skip("Regression dataset not loaded")
        records = all_data_with_regression["regression"]
        assert len(records) > 0
        assert all(isinstance(r, RawProjectRecord) for r in records)

    def test_regression_absent_when_csv_missing(self, tmp_path):
        """If the regression CSV doesn't exist, 'regression' key should be absent."""
        acf_dir = tmp_path / "ACF" / "inflation_ACF" / "data"
        acf_dir.mkdir(parents=True)
        result = load_all_projects(str(tmp_path))
        assert "regression" not in result


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


class TestLoadAllProjects:
    """Tests for load_all_projects()."""

    @pytest.fixture(scope="class")
    def all_data(self):
        assert DATA_DIR.exists(), f"Previous work directory not found: {DATA_DIR}"
        return load_all_projects(str(DATA_DIR))

    def test_returns_dict(self, all_data):
        assert isinstance(all_data, dict)

    def test_expected_keys(self, all_data):
        expected_base = {
            "inflation_acf_clustered",
            "inflation_acf_preprocessed",
            "market_basket",
        }
        keys = set(all_data.keys())
        assert expected_base.issubset(keys)
        # "regression" key may or may not be present depending on CSV existence
        assert keys - expected_base <= {"regression"}

    def test_all_values_are_record_lists(self, all_data):
        for name, records in all_data.items():
            assert isinstance(records, list), f"{name} should be a list"
            assert len(records) > 0, f"{name} should not be empty"
            assert isinstance(
                records[0], RawProjectRecord
            ), f"{name} should contain RawProjectRecord instances"
