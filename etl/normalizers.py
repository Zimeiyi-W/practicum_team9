"""
Data Normalizers — canonicalize raw field values into standard forms.

This module bridges the gaps identified in the schema's normalization
capabilities:

  1. **Unit standardization** — map variant abbreviations ("SqFt", "sq ft",
     "S.F.") to canonical MeasurementUnit enum values; convert SF ↔ GSF.
  2. **Project type normalization** — map free-text project_type strings
     to a controlled taxonomy using a synonym table + fuzzy fallback.
  3. **Key field normalization** — canonicalize city names, validate ZIP
     codes, score complexity on a 1–5 scale, and extract building systems
     from CSI division codes and text descriptions.

Usage:
    from etl.normalizers import normalize_record, normalize_dataset

    # Single record
    record = RawProjectRecord(project_id="P1", most_common_unit="sq ft", ...)
    normalize_record(record)
    assert record.normalized_unit == "SF"

    # Batch
    normalize_dataset(records)

All normalization functions mutate the record **in place** and also return
it for chaining convenience.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from schema import MeasurementUnit, RawProjectRecord, SystemType

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Unit-of-Measure Normalization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Maps raw unit strings (lowercased, stripped) to canonical MeasurementUnit.
# Covers every variant observed across the CCheck3 CSV exports.
_UNIT_ALIAS_MAP: dict[str, MeasurementUnit] = {
    # Square feet
    "sf": MeasurementUnit.SF,
    "sqft": MeasurementUnit.SF,
    "sq ft": MeasurementUnit.SF,
    "sq. ft.": MeasurementUnit.SF,
    "sq.ft.": MeasurementUnit.SF,
    "s.f.": MeasurementUnit.SF,
    "square feet": MeasurementUnit.SF,
    "square foot": MeasurementUnit.SF,
    "ft2": MeasurementUnit.SF,
    "ft²": MeasurementUnit.SF,
    # Gross square feet
    "gsf": MeasurementUnit.GSF,
    "gross sf": MeasurementUnit.GSF,
    "gross sq ft": MeasurementUnit.GSF,
    "gross square feet": MeasurementUnit.GSF,
    # Square yards
    "sy": MeasurementUnit.SY,
    "sq yd": MeasurementUnit.SY,
    "sq.yd.": MeasurementUnit.SY,
    "s.y.": MeasurementUnit.SY,
    "square yard": MeasurementUnit.SY,
    "square yards": MeasurementUnit.SY,
    "yd2": MeasurementUnit.SY,
    "yd²": MeasurementUnit.SY,
    # Linear feet
    "lf": MeasurementUnit.LF,
    "lin ft": MeasurementUnit.LF,
    "lin. ft.": MeasurementUnit.LF,
    "l.f.": MeasurementUnit.LF,
    "linear feet": MeasurementUnit.LF,
    "linear foot": MeasurementUnit.LF,
    "feet": MeasurementUnit.LF,
    "foot": MeasurementUnit.LF,
    "ft": MeasurementUnit.LF,
    # Miles
    "mi": MeasurementUnit.MI,
    "mile": MeasurementUnit.MI,
    "miles": MeasurementUnit.MI,
    # Cubic yards
    "cy": MeasurementUnit.CY,
    "cu yd": MeasurementUnit.CY,
    "cu.yd.": MeasurementUnit.CY,
    "c.y.": MeasurementUnit.CY,
    "cubic yard": MeasurementUnit.CY,
    "cubic yards": MeasurementUnit.CY,
    "yd3": MeasurementUnit.CY,
    "yd³": MeasurementUnit.CY,
    # Gallons
    "gal": MeasurementUnit.GAL,
    "gallon": MeasurementUnit.GAL,
    "gallons": MeasurementUnit.GAL,
    # Tons
    "ton": MeasurementUnit.TON,
    "tons": MeasurementUnit.TON,
    "tn": MeasurementUnit.TON,
    # Pounds
    "lb": MeasurementUnit.LB,
    "lbs": MeasurementUnit.LB,
    "pound": MeasurementUnit.LB,
    "pounds": MeasurementUnit.LB,
    # Each
    "ea": MeasurementUnit.EA,
    "each": MeasurementUnit.EA,
    # Lump sum
    "ls": MeasurementUnit.LS,
    "lump sum": MeasurementUnit.LS,
    "lump": MeasurementUnit.LS,
    "l.s.": MeasurementUnit.LS,
    # Hours
    "hr": MeasurementUnit.HR,
    "hrs": MeasurementUnit.HR,
    "hour": MeasurementUnit.HR,
    "hours": MeasurementUnit.HR,
    # Days
    "day": MeasurementUnit.DAY,
    "days": MeasurementUnit.DAY,
    "dy": MeasurementUnit.DAY,
    # Months
    "mo": MeasurementUnit.MO,
    "month": MeasurementUnit.MO,
    "months": MeasurementUnit.MO,
    # Acres
    "ac": MeasurementUnit.AC,
    "acre": MeasurementUnit.AC,
    "acres": MeasurementUnit.AC,
    # Cubic feet
    "cf": MeasurementUnit.CF,
    "cu ft": MeasurementUnit.CF,
    "cubic feet": MeasurementUnit.CF,
    "cubic foot": MeasurementUnit.CF,
    "ft3": MeasurementUnit.CF,
    "ft³": MeasurementUnit.CF,
}

# Unit conversion factors (all relative to SF for area, base unit for others)
_AREA_TO_SF: dict[MeasurementUnit, float] = {
    MeasurementUnit.SF: 1.0,
    MeasurementUnit.SY: 9.0,      # 1 SY = 9 SF
    MeasurementUnit.AC: 43_560.0,  # 1 AC = 43,560 SF
}

# Default GSF-to-SF ratio (gross → net).
# Source: BOMA Standard (Building Owners and Managers Association).
# Typical range 1.15–1.25; we use 1.20 as a reasonable midpoint.
DEFAULT_GSF_TO_SF_RATIO = 1.20


def normalize_unit(raw_unit: Optional[str]) -> Optional[MeasurementUnit]:
    """
    Map a raw unit-of-measure string to a canonical MeasurementUnit.

    The lookup is case-insensitive and strips whitespace and periods.
    Returns None if the input is None/empty, or MeasurementUnit.OTHER
    if no match is found.

    Args:
        raw_unit: Raw unit string from CSV data (e.g. "sq ft", "SY", "L.F.").

    Returns:
        Canonical MeasurementUnit, or None if input is None/empty.
    """
    if not raw_unit or not raw_unit.strip():
        return None

    cleaned = raw_unit.strip().lower()
    # Try exact match first
    result = _UNIT_ALIAS_MAP.get(cleaned)
    if result is not None:
        return result

    # Try with periods stripped
    no_dots = cleaned.replace(".", "").strip()
    result = _UNIT_ALIAS_MAP.get(no_dots)
    if result is not None:
        return result

    logger.debug("Unrecognised unit '%s' → mapped to OTHER", raw_unit)
    return MeasurementUnit.OTHER


def convert_sf_to_gsf(
    sf: float,
    ratio: float = DEFAULT_GSF_TO_SF_RATIO,
) -> float:
    """
    Convert net square footage (SF) to gross square footage (GSF).

    GSF includes common areas, stairwells, elevator shafts, and mechanical
    rooms that net/rentable SF excludes.

    Args:
        sf: Net square footage.
        ratio: GSF/SF ratio (default 1.20 per BOMA standard).

    Returns:
        Gross square footage.
    """
    return sf * ratio


def convert_gsf_to_sf(
    gsf: float,
    ratio: float = DEFAULT_GSF_TO_SF_RATIO,
) -> float:
    """
    Convert gross square footage (GSF) to net square footage (SF).

    Args:
        gsf: Gross square footage.
        ratio: GSF/SF ratio (default 1.20 per BOMA standard).

    Returns:
        Net square footage.
    """
    if ratio == 0:
        return gsf
    return gsf / ratio


def convert_to_sf(
    value: float,
    from_unit: MeasurementUnit,
) -> Optional[float]:
    """
    Convert an area measurement from the given unit to square feet.

    Only works for area units (SF, SY, AC). Returns None for non-area units.

    Args:
        value: The numeric measurement.
        from_unit: The unit of the measurement.

    Returns:
        Value in square feet, or None if the unit is not an area unit.
    """
    factor = _AREA_TO_SF.get(from_unit)
    if factor is None:
        return None
    return value * factor


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Project Type Normalization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Canonical project type taxonomy.  Keys are the canonical names; values are
# lists of known synonyms / variant spellings observed in the CCheck3 data.
# The normalizer tries exact match first, then synonym lookup, then
# substring matching on the canonical names.
PROJECT_TYPE_TAXONOMY: dict[str, list[str]] = {
    # ── Civil / Transportation ──
    "Paving": [
        "pavement", "paving", "asphalt", "concrete pavement", "road surface",
        "resurfacing", "overlay", "mill and overlay",
    ],
    "Pavement Markers": [
        "pavement markers", "road markers", "lane markers", "striping",
        "pavement marking", "thermoplastic marking",
    ],
    "Bridge": [
        "bridge", "overpass", "underpass", "viaduct", "bridge rehabilitation",
        "bridge deck", "bridge repair",
    ],
    "Road Construction": [
        "road construction", "highway", "roadway", "road widening",
        "road improvement", "road reconstruction", "intersection",
    ],
    "Grading & Earthwork": [
        "grading", "earthwork", "excavation", "fill", "embankment",
        "clearing and grubbing", "site grading",
    ],
    "Drainage": [
        "drainage", "storm drain", "culvert", "storm sewer", "stormwater",
        "catch basin", "inlet",
    ],
    "Signage & Signals": [
        "signage", "traffic signal", "sign", "traffic sign", "signal",
        "traffic control", "signing",
    ],
    "Guardrail & Barriers": [
        "guardrail", "barrier", "guard rail", "median barrier",
        "crash cushion", "attenuator",
    ],
    "Landscaping": [
        "landscaping", "landscape", "seeding", "sodding", "planting",
        "irrigation", "turf", "erosion control",
    ],
    # ── Building / Commercial ──
    "New Building": [
        "new building", "new construction", "new facility", "building construction",
        "new build",
    ],
    "Renovation": [
        "renovation", "remodel", "remodeling", "rehabilitation", "rehab",
        "modernization", "tenant improvement", "interior renovation",
    ],
    "Addition": [
        "addition", "building addition", "expansion",
    ],
    "Demolition": [
        "demolition", "demo", "abatement", "deconstruction", "removal",
    ],
    "Roofing": [
        "roofing", "roof", "roof replacement", "re-roofing", "reroofing",
    ],
    "HVAC": [
        "hvac", "heating", "ventilation", "air conditioning", "mechanical",
        "boiler", "chiller", "ahu", "air handler",
    ],
    "Electrical": [
        "electrical", "electric", "power distribution", "lighting",
        "switchgear", "panel", "generator",
    ],
    "Plumbing": [
        "plumbing", "piping", "fixtures", "water heater", "domestic water",
    ],
    "Fire Protection": [
        "fire protection", "fire suppression", "sprinkler", "fire alarm",
        "fire sprinkler",
    ],
    # ── Water & Sewer ──
    "Water Line": [
        "water line", "water main", "waterline", "water pipe",
        "water distribution", "water supply",
    ],
    "Sewer Line": [
        "sewer line", "sewer main", "sanitary sewer", "sewer pipe",
        "sewer rehabilitation", "sewer lining",
    ],
    "Water Treatment": [
        "water treatment", "treatment plant", "wtp", "water filtration",
        "water purification",
    ],
    "Pump Station": [
        "pump station", "pumping station", "lift station",
    ],
    # ── Environmental ──
    "Environmental Remediation": [
        "remediation", "environmental cleanup", "contamination",
        "hazardous waste", "soil remediation", "groundwater remediation",
    ],
    "Wetland Restoration": [
        "wetland", "wetland restoration", "wetlands mitigation",
    ],
    # ── General ──
    "Site Utilities": [
        "site utilities", "utilities", "utility", "underground utilities",
    ],
    "Concrete Work": [
        "concrete", "concrete work", "cast-in-place", "precast",
        "concrete repair", "flatwork",
    ],
    "Steel & Metals": [
        "steel", "structural steel", "metals", "metal fabrication",
        "miscellaneous metals",
    ],
    "Painting & Coatings": [
        "painting", "coating", "paint", "coatings", "surface preparation",
    ],
    "Fencing": [
        "fencing", "fence", "chain link", "security fence",
    ],
}

# Build reverse lookup: synonym → canonical name (case-insensitive)
_TYPE_SYNONYM_MAP: dict[str, str] = {}
for canonical, synonyms in PROJECT_TYPE_TAXONOMY.items():
    _TYPE_SYNONYM_MAP[canonical.lower()] = canonical
    for syn in synonyms:
        _TYPE_SYNONYM_MAP[syn.lower()] = canonical


def normalize_project_type(raw_type: Optional[str]) -> Optional[str]:
    """
    Map a raw project_type string to a canonical taxonomy name.

    Strategy (in order):
      1. Exact match against canonical names (case-insensitive).
      2. Exact match against synonym table.
      3. Substring match — check if any synonym appears inside the raw string.
      4. If no match, return the original string (title-cased) so data is
         preserved and can be manually reviewed.

    Args:
        raw_type: Raw project type string from the database.

    Returns:
        Canonical project type name, or title-cased original if no match.
        None if input is None/empty.
    """
    if not raw_type or not raw_type.strip():
        return None

    cleaned = raw_type.strip()
    key = cleaned.lower()

    # 1. Exact match
    if key in _TYPE_SYNONYM_MAP:
        return _TYPE_SYNONYM_MAP[key]

    # 2. Substring match — check if any synonym appears in the raw string
    for synonym, canonical in _TYPE_SYNONYM_MAP.items():
        if synonym in key:
            return canonical

    # 3. No match — preserve original, title-cased
    logger.debug("No taxonomy match for project_type '%s'", raw_type)
    return cleaned.title()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. City Name Canonicalization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Common abbreviation expansions for US city names.
_CITY_ABBREVIATIONS: dict[str, str] = {
    "st.": "Saint",
    "st ": "Saint ",
    "ft.": "Fort",
    "ft ": "Fort ",
    "mt.": "Mount",
    "mt ": "Mount ",
    "pt.": "Point",
    "pt ": "Point ",
    "n.": "North",
    "n ": "North ",
    "s.": "South",
    "s ": "South ",
    "e.": "East",
    "e ": "East ",
    "w.": "West",
    "w ": "West ",
    "jct": "Junction",
    "hts": "Heights",
    "spgs": "Springs",
    "vlg": "Village",
    "twp": "Township",
    "boro": "Borough",
}

# Known city name aliases (lowercased → canonical).
_CITY_ALIASES: dict[str, str] = {
    "nyc": "New York",
    "new york city": "New York",
    "la": "Los Angeles",
    "sf": "San Francisco",
    "dc": "Washington",
    "washington dc": "Washington",
    "washington d.c.": "Washington",
    "philly": "Philadelphia",
    "vegas": "Las Vegas",
    "nola": "New Orleans",
}


def normalize_city(raw_city: Optional[str]) -> Optional[str]:
    """
    Canonicalize a city name.

    Steps:
      1. Strip whitespace and check for known aliases.
      2. Expand common abbreviations (St. → Saint, Ft. → Fort, etc.).
      3. Title-case the result for consistent formatting.
      4. Collapse multiple spaces.

    Args:
        raw_city: Raw city name string.

    Returns:
        Canonicalized city name, or None if input is None/empty.
    """
    if not raw_city or not raw_city.strip():
        return None

    cleaned = raw_city.strip()
    key = cleaned.lower()

    # Check direct aliases
    if key in _CITY_ALIASES:
        return _CITY_ALIASES[key]

    # Expand abbreviations (work on lowercase, then title-case)
    result = key
    for abbr, expansion in _CITY_ABBREVIATIONS.items():
        if result.startswith(abbr):
            result = expansion.lower() + result[len(abbr):]
        elif f" {abbr}" in result:
            result = result.replace(f" {abbr}", f" {expansion.lower()}")

    # Title-case and collapse whitespace
    result = " ".join(result.split())
    result = result.title()

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. ZIP Code Validation & Normalization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_ZIP_PATTERN = re.compile(r"^\d{5}(-\d{4})?$")


def normalize_zip_code(raw_zip: Optional[str]) -> Optional[str]:
    """
    Validate and normalize a US ZIP code.

    Accepts:
      - 5-digit: "02134"
      - ZIP+4: "02134-1234"
      - Numeric without leading zero: "2134" → "02134"

    Returns None for invalid formats.

    Args:
        raw_zip: Raw ZIP code string.

    Returns:
        Normalized 5-digit or ZIP+4 string, or None if invalid.
    """
    if not raw_zip:
        return None

    cleaned = str(raw_zip).strip()

    # Handle numeric input that lost leading zeros (e.g. 2134 → "02134")
    if cleaned.isdigit():
        cleaned = cleaned.zfill(5)
    elif re.match(r"^\d{1,4}-\d{4}$", cleaned):
        parts = cleaned.split("-")
        cleaned = f"{parts[0].zfill(5)}-{parts[1]}"

    if _ZIP_PATTERN.match(cleaned):
        return cleaned

    logger.debug("Invalid ZIP code format: '%s'", raw_zip)
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Complexity Scoring (1–5 scale)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# The CIQS complexity category (1–4) is expanded to a 1–5 scale that also
# considers project budget and construction category.  This provides a
# richer, more granular signal for cost estimation.
#
# Scoring rubric:
#   1 = Minimal complexity  — simple paving, striping, fencing
#   2 = Low complexity      — standard road work, drainage, earthwork
#   3 = Moderate complexity — typical commercial renovation, water/sewer
#   4 = High complexity     — new building construction, treatment plants
#   5 = Very high complexity — hospitals, data centres, high-rise, large civil

# CIQS category → base score
_CIQS_TO_BASE_SCORE: dict[str, int] = {
    "Category 1": 1,
    "category 1": 1,
    "Category 2": 2,
    "category 2": 2,
    "Category 3": 3,
    "category 3": 3,
    "Category 4": 4,
    "category 4": 4,
}

# Budget range modifiers (adds 0 or 1 to the base score)
_BUDGET_COMPLEXITY_BOOST: dict[str, int] = {
    "Less than 1M": 0,
    "$0-$1M": 0,
    "$1M-$3M": 0,
    "$3M-$6M": 0,
    "$6M-$10M": 1,
    "$10M-$20M": 1,
    "$20M+": 1,
}

# Construction categories that inherently involve more complexity
_COMPLEX_CATEGORIES: set[str] = {
    "Water & Sewer",
    "Environmental",
}


def compute_complexity_score(
    ciqs_category: Optional[str] = None,
    budget_range: Optional[str] = None,
    construction_category: Optional[str] = None,
) -> Optional[int]:
    """
    Compute a normalized complexity score on a 1–5 scale.

    Combines the CIQS complexity category with budget range and
    construction category for a richer signal than CIQS alone.

    Args:
        ciqs_category: CIQS complexity category string (e.g. "Category 3").
        budget_range: Official budget range (e.g. "$10M-$20M").
        construction_category: Top-level sector (e.g. "Commercial").

    Returns:
        Integer score 1–5, or None if no inputs are available.
    """
    if not ciqs_category and not budget_range and not construction_category:
        return None

    # Start with CIQS base score or default to 2 (low)
    base = _CIQS_TO_BASE_SCORE.get(ciqs_category or "", 2)

    # Budget boost
    boost = _BUDGET_COMPLEXITY_BOOST.get(budget_range or "", 0)

    # Category boost
    cat_boost = 1 if construction_category in _COMPLEX_CATEGORIES else 0

    score = base + boost + cat_boost
    return max(1, min(5, score))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Building Systems Extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Infers which building systems are involved from two sources:
#   a) CSI MasterFormat division codes (cnt_division field)
#   b) Keywords in the project description

# CSI division → SystemType mapping
_CSI_DIVISION_TO_SYSTEM: dict[int, SystemType] = {
    2: SystemType.DEMOLITION,
    3: SystemType.STRUCTURAL,    # Concrete
    4: SystemType.STRUCTURAL,    # Masonry
    5: SystemType.STRUCTURAL,    # Metals
    6: SystemType.STRUCTURAL,    # Wood, Plastics, Composites
    7: SystemType.ENVELOPE,      # Thermal & Moisture Protection
    8: SystemType.ENVELOPE,      # Openings (doors, windows)
    9: SystemType.INTERIOR,      # Finishes
    10: SystemType.OTHER,        # Specialties
    11: SystemType.OTHER,        # Equipment
    12: SystemType.INTERIOR,     # Furnishings
    13: SystemType.OTHER,        # Special Construction
    14: SystemType.CONVEYING,    # Conveying Equipment
    21: SystemType.FIRE_PROTECTION,
    22: SystemType.PLUMBING,
    23: SystemType.MECHANICAL,   # HVAC
    25: SystemType.MECHANICAL,   # Integrated Automation
    26: SystemType.ELECTRICAL,
    27: SystemType.ELECTRICAL,   # Communications
    28: SystemType.ELECTRICAL,   # Electronic Safety & Security
    31: SystemType.SITEWORK,     # Earthwork
    32: SystemType.SITEWORK,     # Exterior Improvements
    33: SystemType.SITEWORK,     # Utilities
}

# Keyword patterns in project descriptions → SystemType
_DESCRIPTION_SYSTEM_PATTERNS: list[tuple[re.Pattern, SystemType]] = [
    (re.compile(r"\b(hvac|heating|cooling|ventilation|air\s+condition|boiler|chiller|ahu)\b", re.I), SystemType.MECHANICAL),
    (re.compile(r"\b(electrical|wiring|switchgear|panel|generator|lighting|power\s+distribution)\b", re.I), SystemType.ELECTRICAL),
    (re.compile(r"\b(plumbing|piping|water\s+heater|fixture|domestic\s+water)\b", re.I), SystemType.PLUMBING),
    (re.compile(r"\b(fire\s+(?:protection|suppression|alarm|sprinkler)|sprinkler\s+system)\b", re.I), SystemType.FIRE_PROTECTION),
    (re.compile(r"\b(structural|foundation|steel\s+frame|concrete\s+frame|structural\s+steel|beam|column|footing)\b", re.I), SystemType.STRUCTURAL),
    (re.compile(r"\b(roof|envelope|exterior\s+wall|curtain\s+wall|window|glazing|waterproofing|insulation)\b", re.I), SystemType.ENVELOPE),
    (re.compile(r"\b(finish|flooring|ceiling|drywall|paint|tile|carpet|interior)\b", re.I), SystemType.INTERIOR),
    (re.compile(r"\b(sitework|grading|earthwork|paving|parking\s+lot|landscape|retaining\s+wall)\b", re.I), SystemType.SITEWORK),
    (re.compile(r"\b(elevator|escalator|conveying|lift)\b", re.I), SystemType.CONVEYING),
    (re.compile(r"\b(demolition|demo|abatement|selective\s+demo)\b", re.I), SystemType.DEMOLITION),
]


def extract_systems(
    cnt_division: Optional[int] = None,
    project_description: Optional[str] = None,
    project_type: Optional[str] = None,
) -> list[str]:
    """
    Infer building systems involved in a project.

    Uses CSI division code (if available) and keyword matching against
    the project description and type to identify relevant systems.

    Args:
        cnt_division: CSI MasterFormat division code (1–33).
        project_description: Free-text project description.
        project_type: Project type string.

    Returns:
        Sorted list of unique SystemType values (as strings).
        Empty list if no systems can be inferred.
    """
    systems: set[str] = set()

    # From CSI division
    if cnt_division is not None:
        sys_type = _CSI_DIVISION_TO_SYSTEM.get(cnt_division)
        if sys_type is not None:
            systems.add(sys_type.value)

    # From description keywords
    text = " ".join(
        part for part in [project_description, project_type] if part
    )
    if text:
        for pattern, sys_type in _DESCRIPTION_SYSTEM_PATTERNS:
            if pattern.search(text):
                systems.add(sys_type.value)

    return sorted(systems)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Record-Level & Dataset-Level Normalization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def normalize_record(record: RawProjectRecord) -> RawProjectRecord:
    """
    Apply all normalizations to a single RawProjectRecord **in place**.

    Populates the following fields:
      - normalized_unit       (from most_common_unit)
      - normalized_project_type (from project_type)
      - normalized_city       (from project_city)
      - complexity_score      (from ciqs_complexity_category + budget + category)
      - systems               (from cnt_division + description + type)
      - project_gsf           (from project_sq_ft, if available)
      - zip_code              (validated/normalized)

    Args:
        record: The record to normalize.

    Returns:
        The same record (mutated) for chaining convenience.
    """
    # Unit normalization
    unit = normalize_unit(record.most_common_unit)
    record.normalized_unit = unit.value if unit else None

    # Project type normalization
    record.normalized_project_type = normalize_project_type(record.project_type)

    # City canonicalization
    record.normalized_city = normalize_city(record.project_city)

    # ZIP code validation
    record.zip_code = normalize_zip_code(record.zip_code)

    # Complexity scoring
    record.complexity_score = compute_complexity_score(
        ciqs_category=record.ciqs_complexity_category,
        budget_range=record.official_budget_range,
        construction_category=record.construction_category,
    )

    # Systems extraction
    record.systems = extract_systems(
        cnt_division=record.cnt_division,
        project_description=record.project_description,
        project_type=record.project_type,
    )

    # GSF estimation (if SF is available and GSF is not already set)
    if record.project_sq_ft is not None and record.project_gsf is None:
        record.project_gsf = convert_sf_to_gsf(record.project_sq_ft)

    return record


def normalize_dataset(
    records: list[RawProjectRecord],
) -> list[RawProjectRecord]:
    """
    Apply all normalizations to a list of RawProjectRecords.

    Args:
        records: List of records to normalize.

    Returns:
        The same list (each record mutated in place).
    """
    for rec in records:
        normalize_record(rec)

    # Log summary statistics
    n_unit = sum(1 for r in records if r.normalized_unit is not None)
    n_type = sum(1 for r in records if r.normalized_project_type is not None)
    n_city = sum(1 for r in records if r.normalized_city is not None)
    n_score = sum(1 for r in records if r.complexity_score is not None)
    n_sys = sum(1 for r in records if r.systems)

    logger.info(
        "Normalized %d records — units: %d, types: %d, cities: %d, "
        "complexity: %d, systems: %d",
        len(records), n_unit, n_type, n_city, n_score, n_sys,
    )
    return records
