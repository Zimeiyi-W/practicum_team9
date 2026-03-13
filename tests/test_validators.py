"""
Tests for etl/validators.py

Verifies that:
  1. Individual record validation catches errors and warnings correctly
  2. Dataset-level validation produces accurate summary statistics
  3. Edge cases (empty records, all-valid, all-invalid) are handled
  4. filter_valid() correctly separates valid from invalid records
  5. Integration with real loaded data

Run:
    cd /Users/wangzimeiyi/Desktop/Practicum
    /usr/bin/python3 -m pytest tests/test_validators.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from etl.validators import (
    DatasetValidationReport,
    ValidationResult,
    filter_valid,
    validate_dataset,
    validate_record,
)
from schema import RawProjectRecord

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "Previous work"
INFLATION_CSV = (
    DATA_DIR / "ACF" / "inflation_ACF" / "data" / "projects_clusters_log_outliers.csv"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_record():
    """A fully valid RawProjectRecord."""
    return RawProjectRecord(
        project_id="VALID-001",
        project_state="PA",
        project_city="PITTSBURGH",
        project_latitude=40.44,
        project_longitude=-79.99,
        project_sq_ft=50000.0,
        project_cost=5_000_000.0,
        price_per_sq_ft=100.0,
        project_year=2020,
        total_mat_lab_equip=4_500_000.0,
        construction_category="Commercial",
        project_type="Communication Devices",
        project_category="Offices & Warehouses",
    )


@pytest.fixture
def invalid_record_missing_id():
    """Record with missing project_id."""
    return RawProjectRecord(project_id="UNKNOWN")


@pytest.fixture
def invalid_record_negative_cost():
    """Record with negative cost fields."""
    return RawProjectRecord(
        project_id="NEG-001",
        labor_total=-100_000.0,
        material_total=-50_000.0,
        project_cost=-200_000.0,
    )


@pytest.fixture
def warning_record_outlier():
    """Record with suspicious but not blocking values."""
    return RawProjectRecord(
        project_id="WARN-001",
        project_state="CA",
        project_latitude=50.0,  # high but within US bounds (AK range)
        project_longitude=-75.0,
        price_per_sq_ft=60_000.0,  # unusually high
        project_year=1940,  # before MIN_YEAR (1950)
        project_sq_ft=200_000_000.0,  # > 100M SF
    )


# ---------------------------------------------------------------------------
# 1. ValidationResult
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_empty_is_valid(self):
        vr = ValidationResult(record_id="test")
        assert vr.is_valid

    def test_with_error_is_invalid(self):
        vr = ValidationResult(record_id="test", errors=["bad"])
        assert not vr.is_valid

    def test_warnings_only_is_valid(self):
        vr = ValidationResult(record_id="test", warnings=["suspicious"])
        assert vr.is_valid

    def test_repr(self):
        vr = ValidationResult(record_id="X", errors=["e1", "e2"], warnings=["w1"])
        r = repr(vr)
        assert "INVALID" in r
        assert "2 errors" in r


# ---------------------------------------------------------------------------
# 2. Record-level validation
# ---------------------------------------------------------------------------


class TestValidateRecord:
    def test_valid_record_passes(self, valid_record):
        vr = validate_record(valid_record)
        assert vr.is_valid
        assert len(vr.errors) == 0

    def test_missing_id_error(self, invalid_record_missing_id):
        vr = validate_record(invalid_record_missing_id)
        assert not vr.is_valid
        assert any("project_id" in e for e in vr.errors)

    def test_negative_costs_error(self, invalid_record_negative_cost):
        vr = validate_record(invalid_record_negative_cost)
        assert not vr.is_valid
        assert any("labor_total" in e for e in vr.errors)
        assert any("material_total" in e for e in vr.errors)
        assert any("project_cost" in e for e in vr.errors)

    def test_negative_sq_ft_error(self):
        rec = RawProjectRecord(project_id="X", project_sq_ft=-100.0)
        vr = validate_record(rec)
        assert any("project_sq_ft" in e for e in vr.errors)

    def test_zero_sq_ft_error(self):
        rec = RawProjectRecord(project_id="X", project_sq_ft=0.0)
        vr = validate_record(rec)
        assert any("project_sq_ft" in e for e in vr.errors)

    def test_negative_price_per_sf_error(self):
        rec = RawProjectRecord(project_id="X", price_per_sq_ft=-10.0)
        vr = validate_record(rec)
        assert any("price_per_sq_ft" in e for e in vr.errors)

    def test_state_code_three_letters_error(self):
        rec = RawProjectRecord(project_id="X", project_state="CAL")
        vr = validate_record(rec)
        assert any("2-letter" in e for e in vr.errors)

    def test_invalid_state_code_warning(self):
        rec = RawProjectRecord(project_id="X", project_state="ZZ")
        vr = validate_record(rec)
        assert vr.is_valid  # warning, not error
        assert any("recognised" in w for w in vr.warnings)

    def test_latitude_out_of_range_warning(self):
        rec = RawProjectRecord(project_id="X", project_latitude=5.0)
        vr = validate_record(rec)
        assert vr.is_valid
        assert any("project_latitude" in w for w in vr.warnings)

    def test_longitude_out_of_range_warning(self):
        rec = RawProjectRecord(project_id="X", project_longitude=10.0)
        vr = validate_record(rec)
        assert vr.is_valid
        assert any("project_longitude" in w for w in vr.warnings)

    def test_year_out_of_range_warning(self):
        rec = RawProjectRecord(project_id="X", project_year=1940)
        vr = validate_record(rec)
        assert vr.is_valid
        assert any("project_year" in w for w in vr.warnings)

    def test_high_price_per_sf_warning(self, warning_record_outlier):
        vr = validate_record(warning_record_outlier)
        assert any("price_per_sq_ft" in w for w in vr.warnings)

    def test_large_sq_ft_warning(self, warning_record_outlier):
        vr = validate_record(warning_record_outlier)
        assert any("project_sq_ft" in w and "100M" in w for w in vr.warnings)

    def test_cost_sf_consistency_warning(self):
        rec = RawProjectRecord(
            project_id="X",
            project_cost=1_000_000.0,
            project_sq_ft=10_000.0,
            price_per_sq_ft=200.0,  # should be 100 — 100% off
        )
        vr = validate_record(rec)
        assert any("discrepancy" in w for w in vr.warnings)

    def test_cost_sf_consistent_no_warning(self):
        rec = RawProjectRecord(
            project_id="X",
            project_cost=1_000_000.0,
            project_sq_ft=10_000.0,
            price_per_sq_ft=100.0,  # exact match
        )
        vr = validate_record(rec)
        assert not any("discrepancy" in w for w in vr.warnings)

    def test_none_fields_no_errors(self):
        """A minimal record with only project_id should pass."""
        rec = RawProjectRecord(project_id="MINIMAL")
        vr = validate_record(rec)
        assert vr.is_valid

    def test_valid_state_codes(self):
        for state in ["CA", "TX", "NY", "PA", "FL", "DC", "PR"]:
            rec = RawProjectRecord(project_id="X", project_state=state)
            vr = validate_record(rec)
            assert not any("recognised" in w for w in vr.warnings), f"{state} flagged"


# ---------------------------------------------------------------------------
# 3. Dataset-level validation
# ---------------------------------------------------------------------------


class TestValidateDataset:
    def test_all_valid(self, valid_record):
        records = [valid_record] * 5
        report = validate_dataset(records)
        assert report.total_records == 5
        assert report.valid_records == 5
        assert report.invalid_records == 0
        assert report.valid_pct == pytest.approx(100.0)

    def test_mixed_valid_invalid(self, valid_record, invalid_record_negative_cost):
        records = [valid_record, invalid_record_negative_cost]
        report = validate_dataset(records)
        assert report.total_records == 2
        assert report.valid_records == 1
        assert report.invalid_records == 1

    def test_empty_dataset(self):
        report = validate_dataset([])
        assert report.total_records == 0
        assert report.valid_records == 0
        assert report.valid_pct == 0.0

    def test_duplicate_ids_counted(self, valid_record):
        records = [valid_record, valid_record]
        report = validate_dataset(records)
        assert report.duplicate_ids == 1  # one ID appears more than once

    def test_top_errors_populated(self, invalid_record_negative_cost):
        records = [invalid_record_negative_cost] * 3
        report = validate_dataset(records)
        assert len(report.top_errors) > 0
        # Each error message should appear 3 times
        for msg, cnt in report.top_errors.items():
            assert cnt == 3

    def test_missing_field_counts(self):
        rec = RawProjectRecord(project_id="X")
        report = validate_dataset([rec])
        assert report.missing_field_counts["project_state"] == 1
        assert report.missing_field_counts["project_city"] == 1
        assert report.missing_field_counts["project_type"] == 1

    def test_report_has_results_list(self, valid_record):
        report = validate_dataset([valid_record])
        assert len(report.results) == 1
        assert isinstance(report.results[0], ValidationResult)


class TestDatasetValidationReport:
    def test_repr(self):
        report = DatasetValidationReport(
            total_records=100, valid_records=95, invalid_records=5
        )
        r = repr(report)
        assert "100" in r
        assert "95" in r

    def test_summary(self):
        report = DatasetValidationReport(
            total_records=100,
            valid_records=90,
            invalid_records=10,
            total_errors=15,
            total_warnings=20,
            duplicate_ids=3,
            top_errors={"bad field": 10, "missing data": 5},
            top_warnings={"suspicious value": 20},
            missing_field_counts={"project_state": 30, "project_year": 10},
        )
        s = report.summary()
        assert "Dataset Validation Report" in s
        assert "90" in s
        assert "bad field" in s
        assert "project_state" in s


# ---------------------------------------------------------------------------
# 4. filter_valid
# ---------------------------------------------------------------------------


class TestFilterValid:
    def test_keeps_valid_records(self, valid_record):
        result = filter_valid([valid_record, valid_record])
        assert len(result) == 2

    def test_removes_invalid_records(self, valid_record, invalid_record_negative_cost):
        result = filter_valid([valid_record, invalid_record_negative_cost])
        assert len(result) == 1
        assert result[0].project_id == "VALID-001"

    def test_empty_input(self):
        result = filter_valid([])
        assert result == []

    def test_all_invalid(self, invalid_record_negative_cost, invalid_record_missing_id):
        result = filter_valid([invalid_record_negative_cost, invalid_record_missing_id])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# 5. Integration with real data
# ---------------------------------------------------------------------------


class TestIntegrationWithRealData:
    @pytest.fixture(scope="class")
    def real_records(self):
        if not INFLATION_CSV.exists():
            pytest.skip(f"CSV not found: {INFLATION_CSV}")
        from etl.loaders import load_inflation_acf_projects

        return load_inflation_acf_projects(str(INFLATION_CSV))

    def test_most_records_valid(self, real_records):
        report = validate_dataset(real_records)
        assert report.valid_pct > 80.0, (
            f"Only {report.valid_pct:.1f}% valid — expected >80%"
        )

    def test_report_summary_runs(self, real_records):
        report = validate_dataset(real_records)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "Dataset Validation Report" in summary

    def test_filter_valid_returns_subset(self, real_records):
        valid = filter_valid(real_records)
        assert len(valid) > 0
        assert len(valid) <= len(real_records)
