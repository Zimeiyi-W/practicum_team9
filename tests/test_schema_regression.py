"""
Tests for schema.py — Regression-related additions.

Verifies that:
  1. REGRESSION_CSV_TO_SCHEMA mapping has the correct structure and entries
  2. RawProjectRecord new fields (inflation_factor, total_project_cost_normalized_2025, acf)
     initialise correctly and accept values
  3. The mapping aligns with RawProjectRecord field names

Run:
    cd /Users/wangzimeiyi/Desktop/Practicum
    python -m pytest tests/test_schema_regression.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schema import (
    REGRESSION_CSV_TO_SCHEMA,
    REGRESSION_FLASK_FEATURE_NAMES,
    RawProjectRecord,
)


# ---------------------------------------------------------------------------
# 1. REGRESSION_CSV_TO_SCHEMA mapping
# ---------------------------------------------------------------------------


class TestRegressionCsvToSchema:
    """Validate the column-name mapping dictionary."""

    def test_is_dict(self):
        assert isinstance(REGRESSION_CSV_TO_SCHEMA, dict)

    def test_all_keys_are_strings(self):
        for key in REGRESSION_CSV_TO_SCHEMA:
            assert isinstance(key, str), f"Key {key!r} is not a string"

    def test_all_values_are_strings(self):
        for value in REGRESSION_CSV_TO_SCHEMA.values():
            assert isinstance(value, str), f"Value {value!r} is not a string"

    def test_contains_project_id(self):
        assert "project_id" in REGRESSION_CSV_TO_SCHEMA

    def test_contains_inflation_factor(self):
        assert "inflation_factor" in REGRESSION_CSV_TO_SCHEMA
        assert REGRESSION_CSV_TO_SCHEMA["inflation_factor"] == "inflation_factor"

    def test_contains_target_column(self):
        assert "total_project_cost_normalized_2025" in REGRESSION_CSV_TO_SCHEMA
        assert (
            REGRESSION_CSV_TO_SCHEMA["total_project_cost_normalized_2025"]
            == "total_project_cost_normalized_2025"
        )

    def test_contains_acf(self):
        assert "acf" in REGRESSION_CSV_TO_SCHEMA
        assert REGRESSION_CSV_TO_SCHEMA["acf"] == "acf"

    def test_contains_expected_feature_keys(self):
        expected_keys = {
            "project_id",
            "inflation_factor",
            "total_project_cost_normalized_2025",
            "official_budget_range",
            "ciqs_complexity_category",
            "cnt_division",
            "cnt_item_code",
            "county_name",
            "area_type",
            "acf",
            "project_latitude",
            "project_longitude",
            "project_type",
            "project_category",
            "project_state",
        }
        assert set(REGRESSION_CSV_TO_SCHEMA.keys()) == expected_keys

    def test_has_15_entries(self):
        assert len(REGRESSION_CSV_TO_SCHEMA) == 15

    def test_values_are_valid_raw_record_fields(self):
        """Every mapped value must be a field on RawProjectRecord."""
        raw_fields = {f.name for f in RawProjectRecord.__dataclass_fields__.values()}
        for csv_col, schema_field in REGRESSION_CSV_TO_SCHEMA.items():
            assert schema_field in raw_fields, (
                f"Mapping value '{schema_field}' (from CSV column '{csv_col}') "
                f"is not a field on RawProjectRecord"
            )

    def test_flask_feature_names_are_subset_of_mapping(self):
        """Every feature the Flask model expects should appear in the mapping
        or be a known derived feature computed at runtime."""
        # Features computed at runtime by feature_engineering.py, not from CSV
        DERIVED_FEATURES = {"region"}
        mapping_values = set(REGRESSION_CSV_TO_SCHEMA.values())
        for feature in REGRESSION_FLASK_FEATURE_NAMES:
            assert (
                feature in mapping_values
                or feature in REGRESSION_CSV_TO_SCHEMA
                or feature in DERIVED_FEATURES
            ), (
                f"Flask feature '{feature}' not found in REGRESSION_CSV_TO_SCHEMA "
                f"or in known derived features {DERIVED_FEATURES}"
            )


# ---------------------------------------------------------------------------
# 2. RawProjectRecord — new regression fields
# ---------------------------------------------------------------------------


class TestRawProjectRecordRegressionFields:
    """Verify the three new Optional fields on RawProjectRecord."""

    def test_inflation_factor_default_none(self):
        rec = RawProjectRecord(project_id="X")
        assert rec.inflation_factor is None

    def test_inflation_factor_accepts_float(self):
        rec = RawProjectRecord(project_id="X", inflation_factor=1.15)
        assert rec.inflation_factor == 1.15

    def test_inflation_factor_accepts_zero(self):
        rec = RawProjectRecord(project_id="X", inflation_factor=0.0)
        assert rec.inflation_factor == 0.0

    def test_total_project_cost_normalized_default_none(self):
        rec = RawProjectRecord(project_id="X")
        assert rec.total_project_cost_normalized_2025 is None

    def test_total_project_cost_normalized_accepts_float(self):
        rec = RawProjectRecord(
            project_id="X", total_project_cost_normalized_2025=5_500_000.0
        )
        assert rec.total_project_cost_normalized_2025 == 5_500_000.0

    def test_total_project_cost_normalized_accepts_large_value(self):
        rec = RawProjectRecord(
            project_id="X", total_project_cost_normalized_2025=1_000_000_000.0
        )
        assert rec.total_project_cost_normalized_2025 == 1_000_000_000.0

    def test_acf_default_none(self):
        rec = RawProjectRecord(project_id="X")
        assert rec.acf is None

    def test_acf_accepts_float(self):
        rec = RawProjectRecord(project_id="X", acf=1.08)
        assert rec.acf == 1.08

    def test_acf_accepts_zero(self):
        rec = RawProjectRecord(project_id="X", acf=0.0)
        assert rec.acf == 0.0

    def test_all_three_fields_together(self):
        rec = RawProjectRecord(
            project_id="FULL",
            inflation_factor=1.15,
            total_project_cost_normalized_2025=2_500_000.0,
            acf=1.08,
        )
        assert rec.inflation_factor == 1.15
        assert rec.total_project_cost_normalized_2025 == 2_500_000.0
        assert rec.acf == 1.08

    def test_new_fields_coexist_with_existing_fields(self):
        """New fields should not interfere with existing RawProjectRecord fields."""
        rec = RawProjectRecord(
            project_id="COEXIST",
            project_state="TX",
            project_type="Pavement Markers",
            dod_acf_2024=1.06,
            inflation_factor=1.15,
            acf=1.08,
            total_project_cost_normalized_2025=3_000_000.0,
        )
        assert rec.project_state == "TX"
        assert rec.project_type == "Pavement Markers"
        assert rec.dod_acf_2024 == 1.06
        assert rec.inflation_factor == 1.15
        assert rec.acf == 1.08
        assert rec.total_project_cost_normalized_2025 == 3_000_000.0

    def test_new_fields_independent_of_dod_acf(self):
        """acf and dod_acf_2024 are independent fields — setting one doesn't affect the other."""
        rec = RawProjectRecord(
            project_id="IND",
            acf=1.10,
            dod_acf_2024=1.06,
        )
        assert rec.acf == 1.10
        assert rec.dod_acf_2024 == 1.06
        assert rec.acf != rec.dod_acf_2024
