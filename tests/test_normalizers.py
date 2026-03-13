"""
Tests for etl/normalizers.py

Verifies:
  1. Unit normalization — maps variant spellings to canonical MeasurementUnit
  2. SF ↔ GSF conversion — correct ratios and edge cases
  3. Project type normalization — exact, synonym, substring, and fallback
  4. City name canonicalization — abbreviations, aliases, whitespace
  5. ZIP code validation — 5-digit, ZIP+4, leading zeros, invalid formats
  6. Complexity scoring — CIQS, budget, category combinations
  7. Building systems extraction — CSI divisions and keyword matching
  8. Record-level normalization — all fields populated correctly
  9. Dataset-level normalization — batch processing

Test Data Sources
=================
The test values in this file are drawn from or inspired by real data in the
Construction Cost Database (CCheck3) system.  Each test section below notes
its provenance.

**Unit abbreviation variants** (Section 1)
  Observed in the ``most_common_unit`` column of the CCheck3 bid-item data,
  accessed via the cmay regression notebook (``final_model_cmay.ipynb``).
  Common values include "SY", "LF", "EA", "CY", "TON" — see also the
  ``UNITS`` list in ``tests/test_parametric.py:47``.  Extended variants
  ("sq ft", "S.F.", "Sq. Ft.", "ft²", etc.) were compiled from the
  original CCheck3 SQL export where unit fields are entered by contractors
  in free-text form.

**GSF/SF conversion ratios** (Section 2)
  The default 1.20 GSF-to-SF ratio follows the BOMA (Building Owners and
  Managers Association) measurement standard.  Typical range 1.15–1.25
  depending on building type; 1.20 is the midpoint used by industry
  estimators.  Area conversion factors (1 SY = 9 SF, 1 AC = 43,560 SF)
  are standard US customary unit definitions.

**Project type strings** (Section 3)
  Canonical types and synonyms were derived from the ``type`` column in
  the CCheck3 project tables.  The inflation ACF CSV
  (``projects_clusters_log_outliers.csv``) contains 19 distinct type
  values including "Communication Devices", "Cafeterias", "Critical Care
  Facility", "Oil Refineries", "Site Work", etc.  The taxonomy extends
  these with common construction industry terminology (paving, bridge,
  HVAC, etc.) drawn from CSI MasterFormat division descriptions and the
  RS Means cost data classification system.

**City names** (Section 4)
  Drawn from the ``project_city`` column in both the inflation ACF CSV
  and the market-basket CSV (``final_dataset_on_year.csv``).  Real
  examples include: "PITTSBURGH", "LOS ANGELES", "NEW YORK CITY",
  "GREENVILLE", "FARGO", "KANSAS CITY", "PHOENIX", "BOSTON",
  "ALLENTOWN", "CHICAGO", "DALLAS", "CLEVELAND".  The city data enters
  the system in ALL-CAPS from the CCheck3 database.  Abbreviation tests
  (St. → Saint, Ft. → Fort, Mt. → Mount) reflect common US city naming
  conventions (e.g. "St. Louis" in MO, "Ft. Worth" in TX).

**ZIP codes** (Section 5)
  The ``zip`` / ``ZCTA5CE20`` column in the CCheck3 data uses US Census
  ZIP Code Tabulation Areas (2020 vintage).  Tests cover 5-digit and
  ZIP+4 formats, as well as edge cases where leading zeros are stripped
  by numeric CSV parsing (e.g. New England ZIPs like 02134).

**Complexity categories** (Section 6)
  The ``ciqs_complexity_category`` field uses "Category 1" through
  "Category 4" as defined by the CIQS (Construction Industry Quality
  Standard) rating system.  Budget ranges ("Less than 1M", "$1M-$3M",
  ..., "$20M+") match the ``official_budget_range`` values in the
  regression CSV (``base_data_for_model.csv``, 17,025 projects).
  Construction categories ("Commercial", "Civil", "Water & Sewer",
  "Environmental", "Transportation") match the ``construction_category``
  enum in ``schema.py``, derived from the CCheck3 classification.

**CSI divisions and system keywords** (Section 7)
  CSI MasterFormat division codes (``cnt_division`` field, range 1–33)
  come from the regression CSV.  The division-to-system mapping follows
  the MasterFormat 2018 standard (e.g. Div 03 = Concrete → Structural,
  Div 23 = HVAC → Mechanical, Div 26 = Electrical).  Keyword patterns
  for description-based extraction were compiled from the
  ``project_description`` free-text field in the CCheck3 bid-tab data,
  which the advanced regression model converts to TF-IDF features.

**Record-level fixtures** (Sections 8–9)
  The test records (project IDs "P-001", "P-002", "P-003") use
  representative field values modeled after the CCheck3 dataset:
  state codes from ``VALID_STATE_CODES`` (``etl/validators.py``),
  coordinates within US geographic bounds (lat 17.5–72.0, lon -180 to
  -64), and cost/size ranges consistent with the validator thresholds
  (max $50k/SF, max $50B project cost).

Run:
    cd /Users/wangzimeiyi/Desktop/Practicum
    /usr/bin/python3 -m pytest tests/test_normalizers.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from etl.normalizers import (
    compute_complexity_score,
    convert_gsf_to_sf,
    convert_sf_to_gsf,
    convert_to_sf,
    extract_systems,
    normalize_city,
    normalize_dataset,
    normalize_project_type,
    normalize_record,
    normalize_unit,
    normalize_zip_code,
)
from schema import MeasurementUnit, RawProjectRecord


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Unit Normalization
# ═══════════════════════════════════════════════════════════════════════════════
# Source: most_common_unit column from CCheck3 bid-item data.
# Canonical abbreviations (SY, LF, EA, CY, TON) from tests/test_parametric.py.
# Free-text variants (sq ft, S.F., ft², etc.) from contractor-entered SQL exports.
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalizeUnit:
    """Tests for normalize_unit()."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("SF", MeasurementUnit.SF),
            ("sf", MeasurementUnit.SF),
            ("SqFt", MeasurementUnit.SF),
            ("sq ft", MeasurementUnit.SF),
            ("Sq. Ft.", MeasurementUnit.SF),
            ("S.F.", MeasurementUnit.SF),
            ("square feet", MeasurementUnit.SF),
            ("ft²", MeasurementUnit.SF),
        ],
    )
    def test_square_feet_variants(self, raw, expected):
        assert normalize_unit(raw) == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("GSF", MeasurementUnit.GSF),
            ("gross sf", MeasurementUnit.GSF),
            ("Gross Square Feet", MeasurementUnit.GSF),
        ],
    )
    def test_gross_square_feet_variants(self, raw, expected):
        assert normalize_unit(raw) == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("SY", MeasurementUnit.SY),
            ("sq yd", MeasurementUnit.SY),
            ("S.Y.", MeasurementUnit.SY),
            ("square yards", MeasurementUnit.SY),
        ],
    )
    def test_square_yards_variants(self, raw, expected):
        assert normalize_unit(raw) == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("LF", MeasurementUnit.LF),
            ("lin ft", MeasurementUnit.LF),
            ("L.F.", MeasurementUnit.LF),
            ("linear feet", MeasurementUnit.LF),
        ],
    )
    def test_linear_feet_variants(self, raw, expected):
        assert normalize_unit(raw) == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("CY", MeasurementUnit.CY),
            ("EA", MeasurementUnit.EA),
            ("LS", MeasurementUnit.LS),
            ("lump sum", MeasurementUnit.LS),
            ("TON", MeasurementUnit.TON),
            ("GAL", MeasurementUnit.GAL),
            ("HR", MeasurementUnit.HR),
            ("hours", MeasurementUnit.HR),
            ("DAY", MeasurementUnit.DAY),
            ("AC", MeasurementUnit.AC),
        ],
    )
    def test_other_units(self, raw, expected):
        assert normalize_unit(raw) == expected

    def test_none_input(self):
        assert normalize_unit(None) is None

    def test_empty_string(self):
        assert normalize_unit("") is None

    def test_whitespace_only(self):
        assert normalize_unit("   ") is None

    def test_unrecognised_unit(self):
        assert normalize_unit("WIDGETS") == MeasurementUnit.OTHER

    def test_leading_trailing_whitespace(self):
        assert normalize_unit("  SF  ") == MeasurementUnit.SF


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SF ↔ GSF Conversion
# ═══════════════════════════════════════════════════════════════════════════════
# Default ratio 1.20 from BOMA measurement standard.
# Area conversion factors: 1 SY = 9 SF, 1 AC = 43,560 SF (US customary).
# ═══════════════════════════════════════════════════════════════════════════════


class TestSfGsfConversion:
    """Tests for convert_sf_to_gsf() and convert_gsf_to_sf()."""

    def test_sf_to_gsf_default_ratio(self):
        assert convert_sf_to_gsf(10_000) == pytest.approx(12_000.0)

    def test_gsf_to_sf_default_ratio(self):
        assert convert_gsf_to_sf(12_000) == pytest.approx(10_000.0)

    def test_custom_ratio(self):
        assert convert_sf_to_gsf(10_000, ratio=1.25) == pytest.approx(12_500.0)
        assert convert_gsf_to_sf(12_500, ratio=1.25) == pytest.approx(10_000.0)

    def test_roundtrip(self):
        sf = 50_000.0
        gsf = convert_sf_to_gsf(sf)
        sf_back = convert_gsf_to_sf(gsf)
        assert sf_back == pytest.approx(sf)

    def test_zero_sf(self):
        assert convert_sf_to_gsf(0) == 0.0

    def test_zero_ratio_gsf_to_sf(self):
        assert convert_gsf_to_sf(12_000, ratio=0) == 12_000.0

    def test_convert_to_sf_from_sy(self):
        result = convert_to_sf(100, MeasurementUnit.SY)
        assert result == pytest.approx(900.0)

    def test_convert_to_sf_from_ac(self):
        result = convert_to_sf(1, MeasurementUnit.AC)
        assert result == pytest.approx(43_560.0)

    def test_convert_to_sf_from_sf(self):
        result = convert_to_sf(500, MeasurementUnit.SF)
        assert result == pytest.approx(500.0)

    def test_convert_to_sf_non_area_unit(self):
        assert convert_to_sf(100, MeasurementUnit.LF) is None
        assert convert_to_sf(100, MeasurementUnit.EA) is None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Project Type Normalization
# ═══════════════════════════════════════════════════════════════════════════════
# Source: 'type' column from projects_clusters_log_outliers.csv (19 distinct
# values: "Communication Devices", "Cafeterias", "Oil Refineries", etc.).
# Taxonomy extended with CSI MasterFormat divisions and RS Means categories.
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalizeProjectType:
    """Tests for normalize_project_type()."""

    def test_exact_canonical_match(self):
        assert normalize_project_type("Paving") == "Paving"

    def test_case_insensitive_canonical(self):
        assert normalize_project_type("paving") == "Paving"
        assert normalize_project_type("PAVING") == "Paving"

    def test_synonym_match(self):
        assert normalize_project_type("asphalt") == "Paving"
        assert normalize_project_type("overlay") == "Paving"
        assert normalize_project_type("resurfacing") == "Paving"

    def test_synonym_match_case_insensitive(self):
        assert normalize_project_type("Asphalt") == "Paving"
        assert normalize_project_type("BRIDGE REHABILITATION") == "Bridge"

    def test_substring_match(self):
        assert normalize_project_type("Road paving project") == "Paving"
        assert normalize_project_type("New bridge construction over river") == "Bridge"

    def test_building_types(self):
        assert normalize_project_type("renovation") == "Renovation"
        assert normalize_project_type("tenant improvement") == "Renovation"
        assert normalize_project_type("new building") == "New Building"
        assert normalize_project_type("demolition") == "Demolition"

    def test_water_sewer_types(self):
        assert normalize_project_type("water main") == "Water Line"
        assert normalize_project_type("sanitary sewer") == "Sewer Line"
        assert normalize_project_type("pump station") == "Pump Station"

    def test_hvac_electrical_plumbing(self):
        assert normalize_project_type("HVAC") == "HVAC"
        assert normalize_project_type("electrical") == "Electrical"
        assert normalize_project_type("plumbing") == "Plumbing"

    def test_no_match_returns_title_case(self):
        result = normalize_project_type("custom unique type xyz")
        assert result == "Custom Unique Type Xyz"

    def test_none_input(self):
        assert normalize_project_type(None) is None

    def test_empty_string(self):
        assert normalize_project_type("") is None

    def test_whitespace_stripped(self):
        assert normalize_project_type("  paving  ") == "Paving"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. City Name Canonicalization
# ═══════════════════════════════════════════════════════════════════════════════
# Source: project_city column from CCheck3 database.
# Inflation ACF CSV cities: PITTSBURGH, LOS ANGELES, NEW YORK CITY, GREENVILLE,
# FARGO, KANSAS CITY, PHOENIX, BOSTON, ALLENTOWN (ALL-CAPS from SQL export).
# Market basket CSV cities: same + Atlanta, CHICAGO, DALLAS, CLEVELAND, ANN ARBOR.
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalizeCity:
    """Tests for normalize_city()."""

    def test_saint_abbreviation(self):
        result = normalize_city("St. Louis")
        assert result == "Saint Louis"

    def test_fort_abbreviation(self):
        result = normalize_city("Ft. Worth")
        assert result == "Fort Worth"

    def test_mount_abbreviation(self):
        result = normalize_city("Mt. Vernon")
        assert result == "Mount Vernon"

    def test_known_alias_nyc(self):
        assert normalize_city("NYC") == "New York"
        assert normalize_city("New York City") == "New York"

    def test_known_alias_dc(self):
        assert normalize_city("Washington DC") == "Washington"
        assert normalize_city("Washington D.C.") == "Washington"

    def test_known_alias_la(self):
        assert normalize_city("LA") == "Los Angeles"

    def test_title_case(self):
        result = normalize_city("san francisco")
        assert result == "San Francisco"

    def test_alias_sf(self):
        assert normalize_city("SF") == "San Francisco"

    def test_whitespace_collapse(self):
        result = normalize_city("  los   angeles  ")
        assert result == "Los Angeles"

    def test_none_input(self):
        assert normalize_city(None) is None

    def test_empty_string(self):
        assert normalize_city("") is None

    def test_already_canonical(self):
        result = normalize_city("Chicago")
        assert result == "Chicago"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ZIP Code Validation
# ═══════════════════════════════════════════════════════════════════════════════
# Source: 'zip' / 'ZCTA5CE20' column from CCheck3 data, using US Census 2020
# ZIP Code Tabulation Areas.  Edge case: New England ZIPs lose leading zeros
# when parsed as numeric (e.g. 02134 → 2134).
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalizeZipCode:
    """Tests for normalize_zip_code()."""

    def test_valid_5_digit(self):
        assert normalize_zip_code("02134") == "02134"

    def test_valid_zip_plus_4(self):
        assert normalize_zip_code("02134-1234") == "02134-1234"

    def test_numeric_leading_zero_restoration(self):
        assert normalize_zip_code("2134") == "02134"

    def test_single_digit(self):
        assert normalize_zip_code("5") == "00005"

    def test_none_input(self):
        assert normalize_zip_code(None) is None

    def test_empty_string(self):
        assert normalize_zip_code("") is None

    def test_invalid_format(self):
        assert normalize_zip_code("ABCDE") is None

    def test_too_many_digits(self):
        assert normalize_zip_code("123456") is None

    def test_numeric_input(self):
        assert normalize_zip_code(2134) == "02134"

    def test_standard_zip(self):
        assert normalize_zip_code("90210") == "90210"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Complexity Scoring
# ═══════════════════════════════════════════════════════════════════════════════
# Source: ciqs_complexity_category from base_data_for_model.csv (Category 1–4).
# Budget ranges from official_budget_range column (same CSV, 17,025 projects).
# Construction categories from the ConstructionCategory enum in schema.py,
# derived from CCheck3 project classification.
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeComplexityScore:
    """Tests for compute_complexity_score()."""

    def test_ciqs_only(self):
        assert compute_complexity_score("Category 1") == 1
        assert compute_complexity_score("Category 2") == 2
        assert compute_complexity_score("Category 3") == 3
        assert compute_complexity_score("Category 4") == 4

    def test_budget_boost(self):
        score = compute_complexity_score("Category 3", "$20M+")
        assert score == 4  # 3 + 1

    def test_category_boost(self):
        score = compute_complexity_score("Category 3", None, "Water & Sewer")
        assert score == 4  # 3 + 1

    def test_combined_boost_capped_at_5(self):
        score = compute_complexity_score("Category 4", "$20M+", "Environmental")
        assert score == 5  # 4 + 1 + 1 = 6 → capped at 5

    def test_all_none_returns_none(self):
        assert compute_complexity_score(None, None, None) is None

    def test_minimum_floor(self):
        score = compute_complexity_score("Category 1", "Less than 1M", "Commercial")
        assert score >= 1

    def test_unknown_ciqs_defaults_to_2(self):
        score = compute_complexity_score("Unknown", None, None)
        assert score == 2

    def test_budget_only(self):
        score = compute_complexity_score(None, "$20M+", None)
        assert score == 3  # default base 2 + 1 boost


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Building Systems Extraction
# ═══════════════════════════════════════════════════════════════════════════════
# Source: cnt_division (CSI MasterFormat divisions 1–33) from regression CSV.
# Keyword patterns from project_description free-text in CCheck3 bid-tab data.
# Division-to-system mapping follows MasterFormat 2018 standard.
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractSystems:
    """Tests for extract_systems()."""

    def test_csi_division_concrete(self):
        systems = extract_systems(cnt_division=3)
        assert "Structural" in systems

    def test_csi_division_hvac(self):
        systems = extract_systems(cnt_division=23)
        assert "Mechanical" in systems

    def test_csi_division_electrical(self):
        systems = extract_systems(cnt_division=26)
        assert "Electrical" in systems

    def test_csi_division_plumbing(self):
        systems = extract_systems(cnt_division=22)
        assert "Plumbing" in systems

    def test_csi_division_fire(self):
        systems = extract_systems(cnt_division=21)
        assert "Fire Protection" in systems

    def test_csi_division_sitework(self):
        systems = extract_systems(cnt_division=31)
        assert "Sitework" in systems

    def test_description_keywords_hvac(self):
        systems = extract_systems(
            project_description="Replace existing HVAC system and install new boiler"
        )
        assert "Mechanical" in systems

    def test_description_keywords_electrical(self):
        systems = extract_systems(
            project_description="Upgrade electrical switchgear and panel"
        )
        assert "Electrical" in systems

    def test_description_keywords_multiple(self):
        systems = extract_systems(
            project_description="HVAC renovation with new electrical and plumbing"
        )
        assert "Mechanical" in systems
        assert "Electrical" in systems
        assert "Plumbing" in systems

    def test_project_type_keyword(self):
        systems = extract_systems(project_type="Demolition")
        assert "Demolition" in systems

    def test_combined_division_and_description(self):
        systems = extract_systems(
            cnt_division=3,
            project_description="Structural steel and elevator installation",
        )
        assert "Structural" in systems
        assert "Conveying" in systems

    def test_no_inputs_returns_empty(self):
        assert extract_systems() == []

    def test_returns_sorted_unique(self):
        systems = extract_systems(
            project_description="HVAC and heating and ventilation and HVAC again"
        )
        assert systems == sorted(set(systems))

    def test_unknown_division(self):
        systems = extract_systems(cnt_division=99)
        assert systems == []


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Record-Level Normalization
# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures use representative values from CCheck3: state codes from
# VALID_STATE_CODES (etl/validators.py), coordinates within US bounds
# (lat 17.5–72.0, lon −180 to −64), cost/size ranges within validator
# thresholds (max $50k/SF, max $50B).  Field combinations modeled after
# the valid_record fixture in tests/test_validators.py:48.
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalizeRecord:
    """Tests for normalize_record()."""

    def test_all_fields_populated(self):
        rec = RawProjectRecord(
            project_id="P-001",
            most_common_unit="sq ft",
            project_type="asphalt",
            project_city="St. Louis",
            project_state="MO",
            zip_code="63101",
            ciqs_complexity_category="Category 3",
            official_budget_range="$10M-$20M",
            construction_category="Civil",
            cnt_division=3,
            project_description="Concrete bridge rehabilitation",
            project_sq_ft=50_000.0,
        )
        result = normalize_record(rec)

        assert result is rec  # mutated in place
        assert rec.normalized_unit == "SF"
        assert rec.normalized_project_type == "Paving"
        assert rec.normalized_city == "Saint Louis"
        assert rec.zip_code == "63101"
        assert rec.complexity_score == 4  # Cat 3 + budget boost
        assert "Structural" in rec.systems
        assert rec.project_gsf == pytest.approx(60_000.0)

    def test_minimal_record(self):
        rec = RawProjectRecord(project_id="P-002")
        normalize_record(rec)
        assert rec.normalized_unit is None
        assert rec.normalized_project_type is None
        assert rec.normalized_city is None
        assert rec.complexity_score is None
        assert rec.systems == []
        assert rec.project_gsf is None

    def test_gsf_not_overwritten_if_already_set(self):
        rec = RawProjectRecord(
            project_id="P-003",
            project_sq_ft=10_000.0,
            project_gsf=15_000.0,  # already set (custom ratio)
        )
        normalize_record(rec)
        assert rec.project_gsf == 15_000.0  # preserved, not overwritten


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Dataset-Level Normalization
# ═══════════════════════════════════════════════════════════════════════════════
# Batch tests simulate a small mixed-quality dataset similar to the 1,819-row
# inflation ACF CSV (projects_clusters_log_outliers.csv) — some records fully
# populated, others minimal (only project_id).  Mirrors the pattern in
# tests/test_transforms.py::TestBatchTransforms.
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalizeDataset:
    """Tests for normalize_dataset()."""

    def test_batch_normalization(self):
        records = [
            RawProjectRecord(project_id="A", most_common_unit="LF", project_type="Bridge"),
            RawProjectRecord(project_id="B", most_common_unit="SY", project_type="paving"),
            RawProjectRecord(project_id="C"),
        ]
        result = normalize_dataset(records)

        assert result is records
        assert records[0].normalized_unit == "LF"
        assert records[0].normalized_project_type == "Bridge"
        assert records[1].normalized_unit == "SY"
        assert records[1].normalized_project_type == "Paving"
        assert records[2].normalized_unit is None

    def test_empty_dataset(self):
        result = normalize_dataset([])
        assert result == []
