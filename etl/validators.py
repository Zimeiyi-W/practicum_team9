"""
Data Validators — quality checks on RawProjectRecord instances.

Catches bad data before it reaches the models.  Each validation
function returns a ``ValidationResult`` with categorised errors
(blockers) and warnings (informational).

Usage:
    from etl.validators import validate_record, validate_dataset

    records = load_inflation_acf_projects("path/to/csv")
    report  = validate_dataset(records)
    print(report)
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from schema import RawProjectRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# US geographic bounds (continental + AK/HI/territories)
# ---------------------------------------------------------------------------
US_LAT_MIN, US_LAT_MAX = 17.5, 72.0   # American Samoa → northern AK
US_LON_MIN, US_LON_MAX = -180.0, -64.0  # western AK (Aleutians) → US Virgin Is.

# Two-letter state / territory codes
VALID_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "VI", "GU", "AS", "MP",
}

# Reasonable cost bounds (in USD)
MAX_COST_PER_SF = 50_000.0       # > $50k/SF is almost certainly an error
MAX_PROJECT_COST = 50_000_000_000.0  # $50 billion — largest-ever US projects
MIN_YEAR = 1950
MAX_YEAR = 2030


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """
    Holds errors and warnings for a single record validation.

    Errors are blockers — the record should not be used for training
    or inference without fixing them.  Warnings are informational
    (suspicious values that may still be valid).
    """

    record_id: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no blocking errors."""
        return len(self.errors) == 0

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else f"INVALID ({len(self.errors)} errors)"
        return (
            f"ValidationResult({self.record_id}: {status}, "
            f"{len(self.warnings)} warnings)"
        )


# ---------------------------------------------------------------------------
# Record-level validation
# ---------------------------------------------------------------------------


def validate_record(record: RawProjectRecord) -> ValidationResult:
    """
    Run all quality checks on a single RawProjectRecord.

    Checks performed:
      - Required fields present (project_id)
      - Cost fields non-negative
      - Square-footage positive (if present)
      - Cost-per-SF sanity range
      - Valid state code
      - Lat/lon within US bounds
      - Year within reasonable range
      - Coordinate / cost consistency

    Args:
        record: A loaded RawProjectRecord.

    Returns:
        ValidationResult with errors (blockers) and warnings.
    """
    vr = ValidationResult(record_id=record.project_id or "MISSING_ID")

    # -- Required: project_id --
    if not record.project_id or record.project_id == "UNKNOWN":
        vr.errors.append("project_id is missing or UNKNOWN")

    # -- Cost fields: must be non-negative when present --
    _check_non_negative(vr, "labor_total", record.labor_total)
    _check_non_negative(vr, "material_total", record.material_total)
    _check_non_negative(vr, "equipment_total", record.equipment_total)
    _check_non_negative(vr, "subcontractor_total", record.subcontractor_total)
    _check_non_negative(vr, "total_mat_lab_equip", record.total_mat_lab_equip)
    _check_non_negative(vr, "unit_cost_total", record.unit_cost_total)
    _check_non_negative(vr, "project_cost", record.project_cost)

    # -- Square footage: must be positive --
    if record.project_sq_ft is not None:
        if record.project_sq_ft <= 0:
            vr.errors.append(
                f"project_sq_ft must be positive, got {record.project_sq_ft}"
            )
        elif record.project_sq_ft > 100_000_000:
            vr.warnings.append(
                f"project_sq_ft={record.project_sq_ft:,.0f} seems unusually large (>100M)"
            )

    # -- Cost per SF --
    if record.price_per_sq_ft is not None:
        if record.price_per_sq_ft < 0:
            vr.errors.append(
                f"price_per_sq_ft must be non-negative, got {record.price_per_sq_ft}"
            )
        elif record.price_per_sq_ft > MAX_COST_PER_SF:
            vr.warnings.append(
                f"price_per_sq_ft={record.price_per_sq_ft:,.2f} exceeds {MAX_COST_PER_SF:,.0f}"
            )

    # -- Project cost ceiling --
    if record.project_cost is not None and record.project_cost > MAX_PROJECT_COST:
        vr.warnings.append(
            f"project_cost={record.project_cost:,.0f} exceeds ${MAX_PROJECT_COST:,.0f}"
        )

    # -- State code --
    if record.project_state is not None:
        if len(record.project_state) != 2:
            vr.errors.append(
                f"project_state should be 2-letter code, got '{record.project_state}'"
            )
        elif record.project_state.upper() not in VALID_STATE_CODES:
            vr.warnings.append(
                f"project_state='{record.project_state}' is not a recognised US state/territory"
            )

    # -- Latitude --
    if record.project_latitude is not None:
        if not (US_LAT_MIN <= record.project_latitude <= US_LAT_MAX):
            vr.warnings.append(
                f"project_latitude={record.project_latitude} outside US range "
                f"[{US_LAT_MIN}, {US_LAT_MAX}]"
            )

    # -- Longitude --
    if record.project_longitude is not None:
        if not (US_LON_MIN <= record.project_longitude <= US_LON_MAX):
            vr.warnings.append(
                f"project_longitude={record.project_longitude} outside US range "
                f"[{US_LON_MIN}, {US_LON_MAX}]"
            )

    # -- Year --
    if record.project_year is not None:
        if not (MIN_YEAR <= record.project_year <= MAX_YEAR):
            vr.warnings.append(
                f"project_year={record.project_year} outside expected range "
                f"[{MIN_YEAR}, {MAX_YEAR}]"
            )

    # -- Cross-field: cost vs SF consistency --
    if (
        record.project_cost is not None
        and record.project_sq_ft is not None
        and record.project_sq_ft > 0
        and record.price_per_sq_ft is not None
    ):
        computed = record.project_cost / record.project_sq_ft
        if abs(computed - record.price_per_sq_ft) / max(computed, 1) > 0.1:
            vr.warnings.append(
                f"price_per_sq_ft={record.price_per_sq_ft:.2f} does not match "
                f"project_cost/project_sq_ft={computed:.2f} (>10% discrepancy)"
            )

    # -- Normalized fields (populated by etl/normalizers.py) --
    if record.complexity_score is not None:
        if not (1 <= record.complexity_score <= 5):
            vr.errors.append(
                f"complexity_score must be 1–5, got {record.complexity_score}"
            )

    if record.project_gsf is not None:
        if record.project_gsf <= 0:
            vr.errors.append(
                f"project_gsf must be positive, got {record.project_gsf}"
            )
        elif (
            record.project_sq_ft is not None
            and record.project_sq_ft > 0
            and record.project_gsf < record.project_sq_ft
        ):
            vr.warnings.append(
                f"project_gsf={record.project_gsf:,.0f} is less than "
                f"project_sq_ft={record.project_sq_ft:,.0f} "
                "(GSF should be >= SF)"
            )

    if record.systems is not None and len(record.systems) > 0:
        from schema import SystemType

        valid_system_values = {s.value for s in SystemType}
        for sys_val in record.systems:
            if sys_val not in valid_system_values:
                vr.warnings.append(
                    f"systems contains unrecognised value '{sys_val}'"
                )

    if record.zip_code is not None:
        import re

        if not re.match(r"^\d{5}(-\d{4})?$", record.zip_code):
            vr.errors.append(
                f"zip_code format invalid: '{record.zip_code}' "
                "(expected 5-digit or ZIP+4)"
            )

    return vr


def _check_non_negative(
    vr: ValidationResult, field_name: str, value: Optional[float]
) -> None:
    """Append an error if value is negative."""
    if value is not None and value < 0:
        vr.errors.append(f"{field_name} must be non-negative, got {value}")


# ---------------------------------------------------------------------------
# Dataset-level validation
# ---------------------------------------------------------------------------


@dataclass
class DatasetValidationReport:
    """Summary statistics from validating an entire dataset."""

    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    total_errors: int = 0
    total_warnings: int = 0
    duplicate_ids: int = 0
    missing_field_counts: dict[str, int] = field(default_factory=dict)
    top_errors: dict[str, int] = field(default_factory=dict)
    top_warnings: dict[str, int] = field(default_factory=dict)
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def valid_pct(self) -> float:
        if self.total_records == 0:
            return 0.0
        return 100.0 * self.valid_records / self.total_records

    def __repr__(self) -> str:
        return (
            f"DatasetValidationReport(\n"
            f"  total={self.total_records}, "
            f"valid={self.valid_records} ({self.valid_pct:.1f}%), "
            f"invalid={self.invalid_records},\n"
            f"  errors={self.total_errors}, warnings={self.total_warnings}, "
            f"duplicate_ids={self.duplicate_ids}\n"
            f")"
        )

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines = [
            "=== Dataset Validation Report ===",
            f"Total records:   {self.total_records}",
            f"Valid:           {self.valid_records} ({self.valid_pct:.1f}%)",
            f"Invalid:         {self.invalid_records}",
            f"Total errors:    {self.total_errors}",
            f"Total warnings:  {self.total_warnings}",
            f"Duplicate IDs:   {self.duplicate_ids}",
        ]
        if self.top_errors:
            lines.append("\nTop errors:")
            for msg, cnt in sorted(
                self.top_errors.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                lines.append(f"  [{cnt:>4d}] {msg}")
        if self.top_warnings:
            lines.append("\nTop warnings:")
            for msg, cnt in sorted(
                self.top_warnings.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                lines.append(f"  [{cnt:>4d}] {msg}")
        if self.missing_field_counts:
            lines.append("\nMissing field counts:")
            for fld, cnt in sorted(
                self.missing_field_counts.items(), key=lambda x: x[1], reverse=True
            )[:15]:
                pct = 100.0 * cnt / max(self.total_records, 1)
                lines.append(f"  {fld:<40s}  {cnt:>5d}  ({pct:.1f}%)")
        return "\n".join(lines)


# Key fields to check for missingness
_KEY_FIELDS = [
    "project_state",
    "project_city",
    "project_type",
    "project_category",
    "construction_category",
    "project_latitude",
    "project_longitude",
    "project_sq_ft",
    "project_cost",
    "price_per_sq_ft",
    "total_mat_lab_equip",
    "project_year",
    "project_date",
    "most_common_unit",
    "median_cost_per_unit",
    "median_quantity_most_common_unit",
    "project_description",
    "county_name",
    "dod_acf_2024",
    "wpuip2300001",
    "normalized_unit",
    "normalized_project_type",
    "normalized_city",
    "complexity_score",
    "systems",
    "project_gsf",
]


def validate_dataset(
    records: list[RawProjectRecord],
    *,
    check_duplicates: bool = True,
) -> DatasetValidationReport:
    """
    Run validation on all records and return an aggregate report.

    Args:
        records: List of RawProjectRecord instances.
        check_duplicates: Whether to count duplicate project_ids.

    Returns:
        DatasetValidationReport with per-record results and summary statistics.
    """
    report = DatasetValidationReport(total_records=len(records))

    error_counter: Counter[str] = Counter()
    warning_counter: Counter[str] = Counter()
    missing_counter: Counter[str] = Counter()

    for rec in records:
        vr = validate_record(rec)
        report.results.append(vr)

        if vr.is_valid:
            report.valid_records += 1
        else:
            report.invalid_records += 1

        report.total_errors += len(vr.errors)
        report.total_warnings += len(vr.warnings)

        for e in vr.errors:
            error_counter[e] += 1
        for w in vr.warnings:
            warning_counter[w] += 1

        # Count missing key fields
        for fld in _KEY_FIELDS:
            val = getattr(rec, fld, None)
            if val is None:
                missing_counter[fld] += 1

    report.top_errors = dict(error_counter.most_common(20))
    report.top_warnings = dict(warning_counter.most_common(20))
    report.missing_field_counts = dict(missing_counter.most_common())

    # Duplicate project_id check
    if check_duplicates:
        id_counts = Counter(r.project_id for r in records)
        report.duplicate_ids = sum(1 for cnt in id_counts.values() if cnt > 1)

    logger.info(
        "Validated %d records: %d valid (%.1f%%), %d invalid, %d duplicates",
        report.total_records,
        report.valid_records,
        report.valid_pct,
        report.invalid_records,
        report.duplicate_ids,
    )
    return report


# ---------------------------------------------------------------------------
# Convenience: filter to valid records only
# ---------------------------------------------------------------------------


def filter_valid(records: list[RawProjectRecord]) -> list[RawProjectRecord]:
    """
    Return only records that pass validation (no errors).

    Args:
        records: List of RawProjectRecord instances.

    Returns:
        Filtered list containing only valid records.
    """
    valid = []
    skipped = 0
    for rec in records:
        vr = validate_record(rec)
        if vr.is_valid:
            valid.append(rec)
        else:
            skipped += 1

    if skipped:
        logger.info("filter_valid: kept %d, skipped %d invalid", len(valid), skipped)
    return valid
