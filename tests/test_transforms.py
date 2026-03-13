"""
Tests for etl/transforms.py

Verifies that each transform function:
  1. Correctly maps RawProjectRecord fields to the target dataclass
  2. Raises ValueError for records missing required fields
  3. Handles optional/external fields (acf, region, scope_cluster, etc.)
  4. Market-basket row extraction populates all nested feature groups

Run:
    cd /Users/wangzimeiyi/Desktop/Practicum
    python3 -m pytest tests/test_transforms.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from etl.loaders import load_inflation_acf_projects, load_market_basket_projects
from etl.transforms import (
    batch_raw_to_acf_inflation,
    batch_raw_to_regression_advanced,
    load_market_basket_full,
    raw_to_acf_inflation_input,
    raw_to_regression_advanced,
    raw_to_regression_simple,
    row_to_market_basket_input,
)
from schema import (
    ACFInflationInput,
    ACFMarketBasketInput,
    RawProjectRecord,
    RegressionAdvancedInput,
    RegressionSimpleInput,
)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "Previous work"
INFLATION_CSV = (
    DATA_DIR / "ACF" / "inflation_ACF" / "data" / "projects_clusters_log_outliers.csv"
)
MARKET_BASKET_CSV = (
    DATA_DIR / "ACF" / "market_basket_acf" / "data" / "final_dataset_on_year.csv"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def inflation_records():
    assert INFLATION_CSV.exists(), f"CSV not found: {INFLATION_CSV}"
    return load_inflation_acf_projects(str(INFLATION_CSV))


@pytest.fixture(scope="module")
def market_basket_records():
    assert MARKET_BASKET_CSV.exists(), f"CSV not found: {MARKET_BASKET_CSV}"
    return load_market_basket_projects(str(MARKET_BASKET_CSV))


@pytest.fixture(scope="module")
def market_basket_df():
    assert MARKET_BASKET_CSV.exists(), f"CSV not found: {MARKET_BASKET_CSV}"
    df = pd.read_csv(MARKET_BASKET_CSV)
    if "YEAR" in df.columns and "project_year" in df.columns:
        df["project_year"] = df["project_year"].fillna(df["YEAR"])
    return df


@pytest.fixture
def complete_raw_record():
    """A fully-populated RawProjectRecord for unit testing."""
    return RawProjectRecord(
        project_id="TEST-001",
        project_type="Communication Devices",
        project_category="Offices & Warehouses",
        construction_category="Commercial",
        project_state="PA",
        project_city="PITTSBURGH",
        county_name="Allegheny",
        project_latitude=40.44,
        project_longitude=-79.99,
        project_sq_ft=50000.0,
        project_cost=5000000.0,
        price_per_sq_ft=100.0,
        project_year=2020,
        ciqs_complexity_category="Category 2",
        official_budget_range="$3M-$6M",
        area_type="Urban",
        cnt_division=3,
        cnt_item_code=10,
        median_cost_per_unit=150.0,
        median_quantity_most_common_unit=500.0,
        most_common_unit="SY",
        project_description="Commercial office building renovation",
        wpuip2300001=230.5,
        adjusted_total_mat_lab_equip=4800000.0,
        dod_acf_2024=1.06,
    )


@pytest.fixture
def minimal_raw_record():
    """A RawProjectRecord with only the bare minimum fields."""
    return RawProjectRecord(
        project_id="TEST-MINIMAL",
        project_year=2020,
    )


# ---------------------------------------------------------------------------
# 1. RegressionSimpleInput
# ---------------------------------------------------------------------------


class TestRawToRegressionSimple:
    def test_returns_correct_type(self, complete_raw_record):
        result = raw_to_regression_simple(complete_raw_record)
        assert isinstance(result, RegressionSimpleInput)

    def test_maps_fields_correctly(self, complete_raw_record):
        result = raw_to_regression_simple(
            complete_raw_record, inflation_factor=1.15, acf=1.06, region="Region_2"
        )
        assert result.inflation_factor == 1.15
        assert result.acf == 1.06
        assert result.region == "Region_2"
        assert result.project_type == "Communication Devices"
        assert result.project_category == "Offices & Warehouses"
        assert result.project_state == "PA"
        assert result.county_name == "Allegheny"
        assert result.area_type == "Urban"
        assert result.ciqs_complexity_category == "Category 2"
        assert result.official_budget_range == "$3M-$6M"
        assert result.cnt_division == 3
        assert result.cnt_item_code == 10

    def test_defaults_for_external_fields(self, complete_raw_record):
        result = raw_to_regression_simple(complete_raw_record)
        # With no explicit arguments, falls back to record values or defaults
        # complete_raw_record doesn't have inflation_factor/acf set, so → 1.0
        assert result.inflation_factor == 1.0
        assert result.acf == 1.0
        assert result.region == "Region_0"

    def test_uses_record_inflation_factor_as_fallback(self):
        """When inflation_factor not passed, use the value from the record."""
        rec = RawProjectRecord(
            project_id="INF-TEST",
            project_type="Pavement Markers",
            project_category="Civil",
            project_state="TX",
            inflation_factor=1.15,
            acf=1.08,
        )
        result = raw_to_regression_simple(rec)
        assert result.inflation_factor == 1.15
        assert result.acf == 1.08

    def test_explicit_args_override_record_values(self):
        """Explicit arguments should override record values."""
        rec = RawProjectRecord(
            project_id="OVERRIDE-TEST",
            project_type="Pavement Markers",
            project_category="Civil",
            project_state="TX",
            inflation_factor=1.15,
            acf=1.08,
        )
        result = raw_to_regression_simple(rec, inflation_factor=1.25, acf=0.95)
        assert result.inflation_factor == 1.25
        assert result.acf == 0.95

    def test_fallback_none_arg_none_record_defaults_to_one(self):
        """When both explicit arg and record are None, should default to 1.0."""
        rec = RawProjectRecord(
            project_id="NONE-BOTH",
            project_type="Pavement Markers",
            project_category="Civil",
            project_state="TX",
        )
        result = raw_to_regression_simple(rec)
        assert result.inflation_factor == 1.0
        assert result.acf == 1.0

    def test_fallback_none_arg_with_record_inflation_only(self):
        """When arg is None but record has inflation_factor, use record value."""
        rec = RawProjectRecord(
            project_id="PARTIAL-INF",
            project_type="Pavement Markers",
            project_category="Civil",
            project_state="TX",
            inflation_factor=1.20,
        )
        result = raw_to_regression_simple(rec)
        assert result.inflation_factor == 1.20
        assert result.acf == 1.0  # acf not on record → default

    def test_fallback_none_arg_with_record_acf_only(self):
        """When arg is None but record has acf, use record value."""
        rec = RawProjectRecord(
            project_id="PARTIAL-ACF",
            project_type="Pavement Markers",
            project_category="Civil",
            project_state="TX",
            acf=0.92,
        )
        result = raw_to_regression_simple(rec)
        assert result.inflation_factor == 1.0  # inflation not on record → default
        assert result.acf == 0.92

    def test_explicit_zero_inflation_is_not_treated_as_none(self):
        """Passing inflation_factor=0.0 explicitly should use 0.0, not fall back."""
        rec = RawProjectRecord(
            project_id="ZERO-INF",
            project_type="Pavement Markers",
            project_category="Civil",
            project_state="TX",
            inflation_factor=1.15,
        )
        result = raw_to_regression_simple(rec, inflation_factor=0.0)
        assert result.inflation_factor == 0.0

    def test_explicit_zero_acf_is_not_treated_as_none(self):
        """Passing acf=0.0 explicitly should use 0.0, not fall back."""
        rec = RawProjectRecord(
            project_id="ZERO-ACF",
            project_type="Pavement Markers",
            project_category="Civil",
            project_state="TX",
            acf=1.08,
        )
        result = raw_to_regression_simple(rec, acf=0.0)
        assert result.acf == 0.0

    def test_record_zero_inflation_used_as_fallback(self):
        """Record inflation_factor=0.0 is a valid value, should be used as fallback."""
        rec = RawProjectRecord(
            project_id="REC-ZERO-INF",
            project_type="Pavement Markers",
            project_category="Civil",
            project_state="TX",
            inflation_factor=0.0,
        )
        result = raw_to_regression_simple(rec)
        assert result.inflation_factor == 0.0

    def test_record_zero_acf_used_as_fallback(self):
        """Record acf=0.0 is a valid value, should be used as fallback."""
        rec = RawProjectRecord(
            project_id="REC-ZERO-ACF",
            project_type="Pavement Markers",
            project_category="Civil",
            project_state="TX",
            acf=0.0,
        )
        result = raw_to_regression_simple(rec)
        assert result.acf == 0.0

    def test_explicit_negative_values_accepted(self):
        """The function should accept negative values without error."""
        rec = RawProjectRecord(
            project_id="NEG",
            project_type="Pavement Markers",
            project_category="Civil",
            project_state="TX",
        )
        result = raw_to_regression_simple(rec, inflation_factor=-0.5, acf=-1.0)
        assert result.inflation_factor == -0.5
        assert result.acf == -1.0

    def test_fallback_chain_priority_arg_over_record_over_default(self):
        """Full priority chain: explicit arg > record value > 1.0 default."""
        base_rec = RawProjectRecord(
            project_id="CHAIN",
            project_type="Pavement Markers",
            project_category="Civil",
            project_state="TX",
            inflation_factor=1.15,
            acf=1.08,
        )

        # Level 1: explicit args win
        r1 = raw_to_regression_simple(base_rec, inflation_factor=2.0, acf=3.0)
        assert r1.inflation_factor == 2.0
        assert r1.acf == 3.0

        # Level 2: no args → record values used
        r2 = raw_to_regression_simple(base_rec)
        assert r2.inflation_factor == 1.15
        assert r2.acf == 1.08

        # Level 3: no args, no record values → 1.0 default
        empty_rec = RawProjectRecord(
            project_id="CHAIN-EMPTY",
            project_type="Pavement Markers",
            project_category="Civil",
            project_state="TX",
        )
        r3 = raw_to_regression_simple(empty_rec)
        assert r3.inflation_factor == 1.0
        assert r3.acf == 1.0

    def test_raises_on_missing_project_type(self, minimal_raw_record):
        with pytest.raises(ValueError, match="project_type"):
            raw_to_regression_simple(minimal_raw_record)

    def test_raises_on_missing_project_state(self):
        rec = RawProjectRecord(
            project_id="X",
            project_type="Foo",
            project_category="Bar",
        )
        with pytest.raises(ValueError, match="project_state"):
            raw_to_regression_simple(rec)

    def test_fills_defaults_for_optional_flask_fields(self):
        rec = RawProjectRecord(
            project_id="X",
            project_type="Foo",
            project_category="Bar",
            project_state="TX",
        )
        result = raw_to_regression_simple(rec)
        assert result.ciqs_complexity_category == "Category 1"
        assert result.official_budget_range == "$0-$1M"
        assert result.area_type == "Urban"
        assert result.cnt_division == 0
        assert result.cnt_item_code == 0


# ---------------------------------------------------------------------------
# 2. RegressionAdvancedInput
# ---------------------------------------------------------------------------


class TestRawToRegressionAdvanced:
    def test_returns_correct_type(self, complete_raw_record):
        result = raw_to_regression_advanced(complete_raw_record)
        assert isinstance(result, RegressionAdvancedInput)

    def test_maps_fields_correctly(self, complete_raw_record):
        result = raw_to_regression_advanced(
            complete_raw_record,
            acf=1.06,
            acf_state_norm=0.95,
            quantity_bin="100-1000",
            scope_cluster=5,
        )
        assert result.acf == 1.06
        assert result.acf_state_norm == 0.95
        assert result.quantity_bin == "100-1000"
        assert result.scope_cluster == 5
        assert result.project_year == 2020
        assert result.median_cost_per_unit == 150.0
        assert result.median_quantity_most_common_unit == 500.0
        assert result.project_city == "PITTSBURGH"
        assert result.project_state == "PA"
        assert result.project_type == "Communication Devices"
        assert result.construction_category == "Commercial"
        assert result.most_common_unit == "SY"
        assert result.project_description == "Commercial office building renovation"

    def test_raises_on_missing_project_year(self):
        rec = RawProjectRecord(project_id="X")
        with pytest.raises(ValueError, match="project_year"):
            raw_to_regression_advanced(rec)

    def test_defaults_for_external_fields(self, complete_raw_record):
        result = raw_to_regression_advanced(complete_raw_record)
        assert result.acf == 1.0
        assert result.acf_state_norm is None
        assert result.quantity_bin == ""
        assert result.scope_cluster == -1


# ---------------------------------------------------------------------------
# 3. ACFInflationInput
# ---------------------------------------------------------------------------


class TestRawToACFInflationInput:
    def test_returns_correct_type(self, complete_raw_record):
        result = raw_to_acf_inflation_input(complete_raw_record)
        assert isinstance(result, ACFInflationInput)

    def test_maps_fields_correctly(self, complete_raw_record):
        result = raw_to_acf_inflation_input(complete_raw_record)
        assert result.project_state == "PA"
        assert result.project_city == "PITTSBURGH"
        assert result.project_latitude == pytest.approx(40.44)
        assert result.project_longitude == pytest.approx(-79.99)
        assert result.cost_per_sqft == pytest.approx(100.0)
        assert result.project_sq_ft == pytest.approx(50000.0)
        assert result.project_type == "Communication Devices"
        assert result.wpuip2300001 == pytest.approx(230.5)
        assert result.adjusted_total_mat_lab_equip == pytest.approx(4800000.0)
        assert result.dod_acf_2024 == pytest.approx(1.06)

    def test_raises_on_missing_location(self):
        rec = RawProjectRecord(
            project_id="X",
            project_state="PA",
            project_city="PITTSBURGH",
        )
        with pytest.raises(ValueError, match="project_latitude"):
            raw_to_acf_inflation_input(rec)

    def test_raises_on_missing_cost(self):
        rec = RawProjectRecord(
            project_id="X",
            project_state="PA",
            project_city="PITTSBURGH",
            project_latitude=40.0,
            project_longitude=-80.0,
            project_sq_ft=1000.0,
        )
        with pytest.raises(ValueError, match="price_per_sq_ft"):
            raw_to_acf_inflation_input(rec)

    def test_integration_with_loaded_data(self, inflation_records):
        """Test transform on real loaded data."""
        first = inflation_records[0]
        result = raw_to_acf_inflation_input(first)
        assert isinstance(result, ACFInflationInput)
        assert result.project_state == "PA"
        assert result.project_city == "PITTSBURGH"
        assert result.cost_per_sqft > 0


# ---------------------------------------------------------------------------
# 4. ACFMarketBasketInput (from raw row)
# ---------------------------------------------------------------------------


class TestRowToMarketBasketInput:
    def test_returns_correct_type(self, market_basket_df):
        first_row = market_basket_df.iloc[0]
        result = row_to_market_basket_input(first_row)
        assert isinstance(result, ACFMarketBasketInput)

    def test_project_level_fields(self, market_basket_df):
        row = market_basket_df.iloc[1]  # second row (GREENVILLE, SC)
        result = row_to_market_basket_input(row)
        assert result.project_year == 2006
        assert result.construction_category == "Commercial"
        assert result.matched_metro_area is not None

    def test_material_prices_populated(self, market_basket_df):
        """Row 2 (GREENVILLE) has material prices populated."""
        row = market_basket_df.iloc[1]
        result = row_to_market_basket_input(row)
        assert result.materials.concrete_mix is not None
        assert result.materials.concrete_mix > 0
        assert result.materials.electrical_wire is not None

    def test_labor_wages_populated(self, market_basket_df):
        row = market_basket_df.iloc[1]
        result = row_to_market_basket_input(row)
        assert result.labor_wages.carpenters is not None
        assert result.labor_wages.carpenters > 0
        assert result.labor_wages.average is not None

    def test_transport_populated(self, market_basket_df):
        """Row 22 (AUSTIN, 2022) has transport logistics data populated."""
        row = market_basket_df.iloc[22]
        result = row_to_market_basket_input(row)
        assert result.transport_logistics.state_sales_tax_rate is not None

    def test_noaa_weather_populated(self, market_basket_df):
        row = market_basket_df.iloc[1]
        result = row_to_market_basket_input(row)
        # NOAA fields are counts — some may be 0.0 (valid) or None
        assert isinstance(result.noaa_weather.coastal_flood, (float, type(None)))

    def test_economic_indicators_populated(self, market_basket_df):
        row = market_basket_df.iloc[3]
        result = row_to_market_basket_input(row)
        assert result.economic_indicators.unemployment_rate is not None

    def test_empty_row_produces_none_features(self, market_basket_df):
        """First row (LOS ANGELES, 1994) has mostly empty metro features."""
        row = market_basket_df.iloc[0]
        result = row_to_market_basket_input(row)
        assert result.materials.concrete_mix is None
        assert result.labor_wages.carpenters is None

    def test_natural_hazard_risk(self, market_basket_df):
        row = market_basket_df.iloc[1]
        result = row_to_market_basket_input(row)
        # NRI fields exist — may be populated or None depending on data
        assert isinstance(
            result.natural_hazard_risk.composite_eals, (float, type(None))
        )


# ---------------------------------------------------------------------------
# 5. load_market_basket_full (convenience)
# ---------------------------------------------------------------------------


class TestLoadMarketBasketFull:
    @pytest.fixture(scope="class")
    def full_records(self):
        assert MARKET_BASKET_CSV.exists()
        return load_market_basket_full(str(MARKET_BASKET_CSV))

    def test_returns_list(self, full_records):
        assert isinstance(full_records, list)

    def test_correct_count(self, full_records):
        assert len(full_records) == 932

    def test_all_are_market_basket_input(self, full_records):
        assert all(isinstance(r, ACFMarketBasketInput) for r in full_records)

    def test_project_year_populated(self, full_records):
        first = full_records[0]
        assert first.project_year == 1994


# ---------------------------------------------------------------------------
# 6. Batch transforms
# ---------------------------------------------------------------------------


class TestBatchTransforms:
    def test_batch_acf_inflation(self, inflation_records):
        results = batch_raw_to_acf_inflation(inflation_records, skip_invalid=True)
        assert len(results) > 0
        assert all(isinstance(r, ACFInflationInput) for r in results)

    def test_batch_acf_inflation_skip_count(self, inflation_records):
        results = batch_raw_to_acf_inflation(inflation_records, skip_invalid=True)
        # Most records should have lat/lon/cost_per_sqft
        assert len(results) / len(inflation_records) > 0.9

    def test_batch_regression_advanced(self, inflation_records):
        results = batch_raw_to_regression_advanced(inflation_records, skip_invalid=True)
        assert len(results) > 0
        assert all(isinstance(r, RegressionAdvancedInput) for r in results)

    def test_batch_regression_advanced_with_lookups(self, inflation_records):
        first_id = inflation_records[0].project_id
        first_state = inflation_records[0].project_state
        acf_lookup = {first_id: 1.12}
        state_norm = {first_state: 0.95}
        results = batch_raw_to_regression_advanced(
            inflation_records[:5],
            acf_lookup=acf_lookup,
            acf_state_norm_lookup=state_norm,
            skip_invalid=True,
        )
        assert results[0].acf == pytest.approx(1.12)
        assert results[0].acf_state_norm == pytest.approx(0.95)

    def test_batch_raises_when_skip_disabled(self):
        bad_records = [RawProjectRecord(project_id="BAD")]
        with pytest.raises(ValueError):
            batch_raw_to_regression_advanced(bad_records, skip_invalid=False)
