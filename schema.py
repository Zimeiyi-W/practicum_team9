"""
Unified data schema for the Construction Cost Estimation System.

=== WHAT THIS FILE DOES ===
This file defines the "shape" of every piece of data that flows through the
construction-cost estimation pipeline.  Think of each @dataclass below as a
blueprint (like a form with labelled blanks) that tells the rest of the code
exactly what fields to expect, what type each field is, and whether it is
required or optional.

By keeping all definitions in one place we get:
  • A single source of truth — any developer or data scientist can look here
    to understand what the models consume and produce.
  • Type safety — Python type-checkers (mypy, Pyright) can catch mismatches
    before the code even runs.
  • Self-documenting column names — the CSV-to-schema mapping dictionaries
    at the bottom translate raw CSV headers into the clean names used here.

=== CONTEXT: THE THREE RESEARCH STREAMS ===
The system was built across multiple university practicum cohorts:

  1. **Regression / Parametric Cost Model** (Fall 2025 — Aryal, Sawan, Kafwimi)
     A Random Forest model that predicts total project cost from ~13 features
     (project type, location, budget bracket, complexity, etc.).

  2. **ACF Inflation-Adjusted Approach** (Summer 2025 — CCheck3 team)
     Computes an Area Cost Factor (ACF) per geographic cluster by comparing
     local cost-per-sq-ft to the national median, then adjusting for labor
     wages and PPI inflation.

  3. **ACF Market-Basket Approach** (Summer + Fall 2025 cohorts)
     Uses a "market basket" of real material prices (Home Depot), labor wages
     (BLS), weather/hazard risk, and economic indicators to train a LightGBM
     model that predicts a location-based ACF.

=== DATA FLOW (LAYERS) ===
  Raw record  → what comes straight out of the database / CSV files
  Model input → cleaned & engineered features that each model actually uses
  Model output → the predictions returned by each model
  Unified estimate → the combined final output of the full pipeline

=== ABOUT Optional[] FIELDS ===
All Optional[] fields may be absent (None) in early-phase data.  The ETL
pipeline (Phase 1) is responsible for filling them from external sources
or flagging rows where critical fields are missing.
"""

# ── Python compatibility ──
# This import makes *all* type annotations in the file behave as strings at
# runtime, which avoids circular-import issues and lets us reference classes
# before they are defined.  It is standard practice in typed Python ≥ 3.7.
from __future__ import annotations

# dataclass  — decorator that auto-generates __init__, __repr__, __eq__, etc.
# field      — helper to set per-field defaults (needed for mutable defaults
#              like nested dataclasses or lists).
from dataclasses import dataclass, field

# Enum — base class for creating a fixed set of named constants.
from enum import Enum

# Optional[X] is shorthand for "X or None".  It signals that the value may be
# missing and the code must handle that possibility.
from typing import Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enums / Taxonomies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enums define a closed list of allowed values.  By inheriting from both `str`
# and `Enum`, each member can be compared with plain strings AND used in
# JSON / CSV serialisation without extra conversion.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ConstructionCategory(str, Enum):
    """
    Top-level construction sector.

    Every project belongs to exactly one of these five categories.  The
    market-basket ACF model currently trains on Commercial projects only;
    the regression model uses all five.
    """

    COMMERCIAL = "Commercial"
    CIVIL = "Civil"
    WATER_SEWER = "Water & Sewer"
    ENVIRONMENTAL = "Environmental"
    TRANSPORTATION = "Transportation"


class ComplexityCategory(str, Enum):
    """
    CIQS complexity rating used by the regression model.

    "CIQS" stands for the Construction Industry Quality Standard complexity
    rating.  Category 1 is the simplest (e.g. a basic paving job); Category 4
    is the most complex (e.g. a multi-story hospital with specialty systems).
    The regression model uses this as a categorical feature.
    """

    CATEGORY_1 = "Category 1"
    CATEGORY_2 = "Category 2"
    CATEGORY_3 = "Category 3"
    CATEGORY_4 = "Category 4"


class AreaType(str, Enum):
    """
    Whether the project site is in an urban or rural area.

    This affects labor availability, material transport costs, and typical
    cost-per-square-foot benchmarks.  The regression model encodes this as a
    binary categorical feature.
    """

    URBAN = "Urban"
    RURAL = "Rural"


class MeasurementUnit(str, Enum):
    """
    Canonical units of measure used across construction bid items.

    Raw data contains many variant spellings and abbreviations for the same
    unit (e.g. "SqFt", "SF", "sq ft", "S.F." all mean square feet).  This
    enum defines the canonical set; the normalizer maps raw strings to these.

    Groups:
      Area   — SF (square feet), GSF (gross square feet), SY (square yards)
      Length — LF (linear feet), MI (miles)
      Volume — CY (cubic yards), GAL (gallons)
      Weight — TON (short tons), LB (pounds)
      Count  — EA (each), LS (lump sum), HR (hours), DAY (days)
      Other  — OTHER (catch-all for unrecognised units)
    """

    SF = "SF"
    GSF = "GSF"
    SY = "SY"
    LF = "LF"
    MI = "MI"
    CY = "CY"
    GAL = "GAL"
    TON = "TON"
    LB = "LB"
    EA = "EA"
    LS = "LS"
    HR = "HR"
    DAY = "DAY"
    MO = "MO"
    AC = "AC"
    CF = "CF"
    OTHER = "OTHER"


class SystemType(str, Enum):
    """
    Building system taxonomy aligned with CSI MasterFormat divisions.

    Each value represents a major building system used in cost estimation
    and report generation.  Projects typically involve multiple systems;
    the ``systems`` field on RawProjectRecord stores a list of these.

    Mapping from CSI MasterFormat divisions:
      STRUCTURAL     → Div 03 (Concrete), 05 (Metals)
      MECHANICAL     → Div 23 (HVAC), 22 (Plumbing)
      ELECTRICAL     → Div 26 (Electrical), 27 (Communications)
      PLUMBING       → Div 22 (Plumbing)
      FIRE_PROTECTION → Div 21 (Fire Suppression)
      ENVELOPE       → Div 07 (Thermal & Moisture), 08 (Openings)
      INTERIOR       → Div 09 (Finishes), 12 (Furnishings)
      SITEWORK       → Div 31 (Earthwork), 32 (Exterior Improvements)
      CONVEYING      → Div 14 (Conveying Equipment — elevators, escalators)
      DEMOLITION     → Div 02 (Existing Conditions — selective demolition)
      OTHER          → catch-all for systems not in the above categories
    """

    STRUCTURAL = "Structural"
    MECHANICAL = "Mechanical"
    ELECTRICAL = "Electrical"
    PLUMBING = "Plumbing"
    FIRE_PROTECTION = "Fire Protection"
    ENVELOPE = "Building Envelope"
    INTERIOR = "Interior Finishes"
    SITEWORK = "Sitework"
    CONVEYING = "Conveying"
    DEMOLITION = "Demolition"
    OTHER = "Other"


class ProjectPhase(str, Enum):
    """
    Design phase at the time the cost estimate was prepared (plan §1.1).

    Estimates become more accurate as the design progresses:
      • schematic   — very early; rough order of magnitude (±50 %).
      • DD          — Design Development; more detail, tighter range.
      • CD          — Construction Documents; near-final drawings.
      • bid         — contractor pricing; most accurate pre-construction.

    Knowing the phase helps calibrate the confidence band around each estimate.
    """

    SCHEMATIC = "schematic"
    DESIGN_DEVELOPMENT = "DD"
    CONSTRUCTION_DOCUMENTS = "CD"
    BID = "bid"


class BudgetRange(str, Enum):
    """
    Official budget bracket recognised by the regression model.

    These buckets were defined by the CCheck3 database and correspond to the
    typical government contracting tiers.  The Random Forest uses them as an
    ordinal categorical feature — larger budgets tend to have different cost
    distributions (economies of scale, overhead allocation, etc.).
    """

    LESS_THAN_1M = "Less than 1M"
    ONE_TO_THREE_M = "$1M-$3M"
    THREE_TO_SIX_M = "$3M-$6M"
    SIX_TO_TEN_M = "$6M-$10M"
    TEN_TO_TWENTY_M = "$10M-$20M"
    TWENTY_PLUS_M = "$20M+"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Raw Project Record  (common to all models)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# This is the "union" of every column referenced by any notebook in any of
# the three research streams.  In practice, a single CSV or database row will
# only fill a subset of these fields — the rest stay None.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class RawProjectRecord:
    """
    One row coming out of the Construction Check database or CSV export.
    This is the union of all fields referenced across the Regression and ACF
    notebooks.

    **Why a single giant class?**  Each model only needs a subset, but by
    putting every column in one place the ETL pipeline can load any CSV
    into this structure, and each model simply reads the fields it cares about.

    Source tables / CSVs that map onto this record:
      - projects_clusters_log_outliers.csv   → inflation ACF clustering
      - projects_preprocessed.csv            → inflation ACF pre-processing
      - final_dataset_on_year.csv            → market-basket ACF features
      - base_data_for_model.csv              → regression Flask app
      - SQL: public.projects, public.city_area_cost_factors  (cmay notebook)
    """

    # ── Identifiers ──
    # Unique project ID from the CCheck3 database; used to join tables.
    project_id: str

    # ── Cost components (all in USD) ──
    # These break down the project's total cost into labor, material, etc.
    # They come from the CCheck3 bid-tab data.  "other_cost1/2/3" are
    # catch-all columns for line items that don't fit the main categories.
    labor_total: Optional[float] = None
    material_total: Optional[float] = None
    equipment_total: Optional[float] = None
    subcontractor_total: Optional[float] = None
    other_cost1: Optional[float] = None
    other_cost2: Optional[float] = None
    other_cost3: Optional[float] = None
    unit_cost_total: Optional[float] = None
    # total_mat_lab_equip = labor + material + equipment.  Pre-summed in some
    # CSVs; used as the base cost figure by the inflation ACF.
    total_mat_lab_equip: Optional[float] = None

    # ── Location ──
    # Geographic information about where the project is built.  Several
    # models use lat/lon for spatial clustering; the regression model uses
    # state + county for categorical encoding.
    project_city: Optional[str] = None
    project_state: Optional[str] = None  # 2-letter US state code (e.g. "CA")
    county_name: Optional[str] = None
    project_longitude: Optional[float] = None
    project_latitude: Optional[float] = None
    zip_code: Optional[str] = None  # ZCTA5CE20 (Census ZIP Code Tabulation Area)

    # ── Metro area (market-basket ACF) ──
    # The market-basket model operates at the metropolitan-area level.  Each
    # project is fuzzy-matched to the nearest CBSA (Core-Based Statistical Area)
    # using the `matched_metro_area` field; `match_score` records how confident
    # that match is (0–100).
    matched_metro_area: Optional[str] = (
        None  # CBSA title, e.g. "Atlanta-Sandy Springs-Roswell, GA"
    )
    metro_area: Optional[str] = None  # same concept, used in a different CSV
    match_score: Optional[float] = None  # fuzzy-match confidence score
    project_region: Optional[str] = None  # USACE region label (e.g. "South Atlantic")

    # ── Classification ──
    # High-level categorisation of the project.
    # `construction_category` → one of the 5 ConstructionCategory enum values.
    # `project_category` → a finer sub-category within that sector.
    # `project_type` → the most specific label (e.g. "Pavement Markers").
    construction_category: Optional[str] = None
    project_category: Optional[str] = None
    project_type: Optional[str] = None
    phase_description: Optional[str] = None  # free-text description of the design phase

    # ── Size / units ──
    # Physical size and total cost used to derive cost_per_sqft.
    project_sq_ft: Optional[float] = None
    project_cost: Optional[float] = None  # total project cost in USD
    price_per_sq_ft: Optional[float] = None  # = project_cost / project_sq_ft
    # Gross square footage — includes common areas, stairwells, mechanical rooms
    # that are excluded from net/rentable SF.  Typical GSF/SF ratio: 1.15–1.25.
    project_gsf: Optional[float] = None

    # ── Normalized classification ──
    # These fields are populated by etl/normalizers.py after loading raw data.
    # They map free-text values to canonical enum values for consistent analytics.
    normalized_project_type: Optional[str] = None
    normalized_city: Optional[str] = None
    complexity_score: Optional[int] = None  # 1–5 scale (1 = simplest, 5 = most complex)
    systems: Optional[list[str]] = field(default_factory=lambda: None)

    # ── Unit-level data (cmay regression notebook) ──
    # For projects broken into bid items, these fields capture the dominant
    # unit of measure and its median cost/quantity across all line items.
    # `project_description` is free text that gets converted to TF-IDF
    # features in the advanced regression model.
    most_common_unit: Optional[str] = None
    # Canonical unit after normalisation (maps raw abbreviations to MeasurementUnit enum)
    normalized_unit: Optional[str] = None
    median_cost_per_unit: Optional[float] = None
    median_quantity_most_common_unit: Optional[float] = None
    project_description: Optional[str] = None

    # ── Time ──
    # When the project was bid / completed.  Used to align with PPI indices
    # and CPI inflation adjustments.
    project_date: Optional[str] = None  # ISO date string, e.g. "2022-06-15"
    project_year: Optional[int] = None  # extracted year (e.g. 2022)
    year_month: Optional[str] = None  # e.g. "2022-06"

    # ── External indices (pre-joined) ──
    # The DoD (Department of Defense) ACF for 2024, joined from the USACE
    # benchmark table.  Serves as a ground-truth comparison for our computed ACFs.
    dod_acf_2024: Optional[float] = None

    # ── Census / geo enrichment (inflation ACF) ──
    # Population and density from Census data, joined by ZIP code.
    # Used in clustering to distinguish dense urban cores from sparse rural areas.
    population: Optional[int] = None
    density: Optional[float] = None  # people per square mile
    state_name: Optional[str] = None  # full state name (e.g. "California")

    # ── Complexity / budget (regression Flask app) ──
    # Features specifically consumed by the deployed regression Flask app.
    # `cnt_division` (1–29) maps to CSI MasterFormat divisions (e.g. 03 = Concrete).
    # `cnt_item_code` (1–61) is a finer item-level code within a division.
    ciqs_complexity_category: Optional[str] = None
    official_budget_range: Optional[str] = None
    cnt_division: Optional[int] = None
    cnt_item_code: Optional[int] = None
    area_type: Optional[str] = None  # "Urban" or "Rural"

    # ── PPI inflation index ──
    # WPUIP2300001 = Bureau of Labor Statistics Producer Price Index (PPI) for
    # construction materials and components.  Used to adjust historical costs
    # to a common base year so that cost comparisons are fair across time.
    # `adjusted_total_mat_lab_equip` = total_mat_lab_equip deflated by this PPI.
    wpuip2300001: Optional[float] = None
    adjusted_total_mat_lab_equip: Optional[float] = None

    # ── Regression Flask app (base_data_for_model.csv) ──
    # `inflation_factor` is the CPI ratio (base_year / project_year) pre-computed
    # in the regression CSV.  `total_project_cost_normalized_2025` is the
    # inflation-adjusted total project cost (target variable for the Flask model).
    # `acf` is the Area Cost Factor already present in the regression CSV.
    inflation_factor: Optional[float] = None
    total_project_cost_normalized_2025: Optional[float] = None
    acf: Optional[float] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Regression (Parametric) Model — Feature Sets & Output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# "Parametric" means the model learns statistical relationships between
# measurable project attributes (parameters) and cost.  Two variants exist:
#   • Simple  — deployed in the Flask web app, 13 hand-picked features.
#   • Advanced — from the cmay research notebook, richer features + TF-IDF text.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class RegressionSimpleInput:
    """
    The 13-feature input consumed by the deployed Flask-app Random Forest model.

    This is the "simple" variant — it was designed to use only features that
    a user can easily provide through the web form at estimate time.

    Source files:
      - regression_models_README.md   (feature list)
      - RegressionModelsFinal.ipynb   (training code)

    Performance on held-out test set:
      - R² = 0.95   (explains 95 % of variance — very strong)
      - MAPE = 21.97 %  (average prediction is ~22 % off the actual cost)

    Trained on 17,025 projects spanning 2010–2025.
    """

    # ── Economic factors ──
    # `inflation_factor`: ratio of current-year CPI to the project-year CPI.
    #   A value of 1.15 means prices have risen 15 % since the project was bid.
    inflation_factor: float  # typical range 1.00 – 1.34
    # `acf`: Area Cost Factor — the location multiplier from the ACF engine.
    #   Values > 1.0 mean the location is more expensive than the baseline.
    acf: float  # typical range 0.80 – 1.19

    # ── Project classification ──
    project_type: str  # most specific label (e.g. "Pavement Markers")
    project_category: str  # sector sub-category (e.g. "Civil")
    ciqs_complexity_category: str  # "Category 1" through "Category 4"
    official_budget_range: str  # e.g. "$3M-$6M"

    # ── Geographic location ──
    project_state: str  # 2-letter state code
    county_name: str
    area_type: str  # "Urban" or "Rural"
    # `region`: cluster label produced by KMeans on lat/lon (Region_0 … Region_3).
    #   Captures broad geographic cost patterns beyond state boundaries.
    region: str

    # ── Construction detail ──
    # CSI MasterFormat division (1–29) and item code (1–61).  These tell the
    # model *what kind of work* dominates the project.
    cnt_division: int
    cnt_item_code: int


@dataclass
class RegressionAdvancedInput:
    """
    The richer feature set from the "exp10" experiment in final_model_cmay.ipynb.

    This variant adds text features (project_description via TF-IDF), scope
    clusters, and more granular categorical data.  It performs slightly
    differently:
      - R² = 0.93  (a bit lower because it uses the full unsampled dataset)
      - MAPE = 16.73 %  (better average error due to richer features)

    Key concept — `scope_cluster`:
      During training, projects are grouped into 15 clusters (KMeans) based
      on log1p(cost_per_quantity).  Each cluster captures a "scope profile"
      (e.g. cluster 3 might be "high-quantity / low-unit-cost" items).
      At inference, a new project is assigned to the nearest cluster centroid.

    Key concept — `target_column`:
      The model predicts `inf_adj_total_mat_lab_equip`, which is the
      PPI-inflation-adjusted sum of labor + material + equipment costs.
      This removes the effect of price inflation over time so the model
      learns real cost relationships, not just that "prices went up."
    """

    # ── Numeric features ──
    acf: float  # nearest-neighbor ACF from city_area_cost_factors table
    project_year: int  # year of the project (e.g. 2023)
    median_cost_per_unit: float  # median $/unit across the project's bid items
    median_quantity_most_common_unit: float  # median quantity of the most-used unit
    acf_state_norm: Optional[float] = None  # state-average ACF normalised so mean = 1.0

    # ── Categorical features ──
    project_city: str = ""
    project_state: str = ""
    project_type: str = ""  # matches the "type" column in the CSV
    project_category: str = ""
    construction_category: str = ""
    most_common_unit: str = ""  # e.g. "SY" (square yards), "LF" (linear feet)
    quantity_bin: str = ""  # binned quantity range, e.g. "100–1,000"
    scope_cluster: int = -1  # KMeans(k=15) cluster on log1p(cost_per_quantity)

    # ── Text feature ──
    # Free-text project description.  The training pipeline converts this to a
    # TF-IDF sparse matrix (bag-of-words weighted by importance) so the model
    # can learn from keywords like "bridge", "HVAC", "demolition", etc.
    project_description: Optional[str] = None

    # ── Target column reference (used only during training, not at inference) ──
    target_column: str = "inf_adj_total_mat_lab_equip"


@dataclass
class RegressionOutput:
    """
    Output returned by the parametric (regression) cost model.

    The model produces a point estimate and a confidence range:
      - `cost_estimate`: the model's single best-guess cost (in USD).
      - `cost_low` / `cost_high`: lower and upper bounds.  In the simple
        model these are ±25 % of the point estimate; the advanced model
        uses quantile regression for tighter, data-driven bounds.
      - `confidence_level`: 0–1 score reflecting how much training data
        supports this estimate (more similar projects → higher confidence).
      - `similar_projects_*`: stats on the most similar training projects,
        useful for "comparable project" reports.

    Aligns with the project plan §2.2 output specification.
    """

    cost_estimate: float  # point estimate in USD
    cost_low: float  # lower bound of the range
    cost_high: float  # upper bound of the range
    confidence_level: float  # 0.0 to 1.0
    model_version: str = ""  # e.g. "simple_rf_v2" or "advanced_rf_exp10"
    similar_projects_count: int = 0  # number of similar training projects found
    similar_projects_avg_cost: Optional[float] = None
    similar_projects_median_cost: Optional[float] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. ACF — Inflation-Adjusted Approach (clustering-based)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# "ACF" = Area Cost Factor.  It answers: "How much more (or less) expensive
# is it to build in Location X compared to the national average?"
#
# The inflation-adjusted approach works as follows:
#   1. Cluster projects geographically using DBSCAN and HDBSCAN on lat/lon.
#   2. Within each cluster, compute the median cost_per_sqft.
#   3. Divide the cluster median by the national median → that ratio is the ACF.
#   4. Optionally adjust by labor-wage ratios (BLS data) and PPI inflation.
#
# The result is a set of ACF variants (median vs mean, with/without labor
# adjustment, DBSCAN vs HDBSCAN, with/without project-type stratification).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ACFInflationInput:
    """
    Per-project data used by ACF_inflation_adjusted.ipynb.

    Each project needs its cost_per_sqft, geographic coordinates (for clustering),
    and optionally cluster IDs if they were pre-computed.

    Source: projects_clusters_log_outliers.csv
    Method: (cluster-median cost_per_sqft / national median) × labor adjustment
    """

    project_state: str
    project_city: str
    project_latitude: float
    project_longitude: float
    cost_per_sqft: float  # project_cost / project_sq_ft
    project_sq_ft: float
    project_type: str  # "type" column — allows per-type ACFs

    # ── Cluster assignments ──
    # Pre-computed from clustering_density_based.ipynb.  Parameters in the
    # name encode the clustering settings:
    #   41km  → epsilon (max neighbour distance)
    #   2ms / 3ms → min_samples (minimum cluster size)
    db_cluster_41km_2ms: Optional[int] = None  # DBSCAN cluster id (-1 = noise)
    hdb_cluster_41km_3ms: Optional[int] = None  # HDBSCAN cluster id (-1 = noise)

    # ── BLS labour wage normalisation ──
    # State-level median/mean hourly wage for construction occupations,
    # normalised so national median/mean = 1.0.  Multiplied into the ACF
    # to reflect that high-wage states have inherently higher construction costs.
    hour_median_norm: Optional[float] = None
    hour_mean_norm: Optional[float] = None

    # ── PPI (Producer Price Index) ──
    # Used to deflate costs to a common base year.
    wpuip2300001: Optional[float] = None
    adjusted_total_mat_lab_equip: Optional[float] = None

    # ── Benchmark ──
    # Official DoD ACF for 2024, used to validate our computed ACFs.
    dod_acf_2024: Optional[float] = None


@dataclass
class ACFInflationOutput:
    """
    One of several ACF variants produced by the inflation-adjusted approach.

    **All values are ratios where national average = 1.0.**
      • ACF = 1.15 means 15 % more expensive than the national average.
      • ACF = 0.88 means 12 % cheaper.

    Naming convention for fields:
      acf_{clustering}_{statistic}          — base ACF
      acf_type_{clustering}_{statistic}     — stratified by project type
      acf_{clustering}_labor_adj_{statistic} — multiplied by labor-wage ratio

    Where:
      clustering = "db" (DBSCAN) or "hdb" (HDBSCAN)
      statistic  = "median" or "mean" (of cost_per_sqft within the cluster)

    Multiple variants are kept so downstream analysis can compare them against
    the USACE benchmark and pick the most accurate one per region.
    """

    # ── Median variants (generally more robust to outliers) ──
    acf_db_median: Optional[float] = None  # DBSCAN cluster, median cost
    acf_hdb_median: Optional[float] = None  # HDBSCAN cluster, median cost
    acf_type_db_median: Optional[float] = None  # DBSCAN + stratified by project type
    acf_type_hdb_median: Optional[float] = None  # HDBSCAN + stratified by project type
    acf_db_labor_adj_median: Optional[float] = (
        None  # DBSCAN + BLS labor wage adjustment
    )
    acf_hdb_labor_adj_median: Optional[float] = (
        None  # HDBSCAN + BLS labor wage adjustment
    )
    acf_type_db_labor_adj_median: Optional[float] = None  # DBSCAN + type + labor
    acf_type_hdb_labor_adj_median: Optional[float] = None  # HDBSCAN + type + labor

    # ── Mean variants (more sensitive to extreme values) ──
    acf_db_mean: Optional[float] = None
    acf_hdb_mean: Optional[float] = None
    acf_type_db_mean: Optional[float] = None
    acf_type_hdb_mean: Optional[float] = None
    acf_db_labor_adj_mean: Optional[float] = None
    acf_hdb_labor_adj_mean: Optional[float] = None
    acf_type_db_labor_adj_mean: Optional[float] = None
    acf_type_hdb_labor_adj_mean: Optional[float] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. ACF — Market-Basket ML Approach
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Instead of using cluster-level statistics (Section 3), this approach
# *predicts* the ACF using an ML model trained on external economic data:
#
#   Material prices   — real retail prices scraped from Home Depot.
#   Labor wages       — BLS OEWS occupational wage statistics.
#   Transport costs   — USACE local-area logistics factors.
#   Productivity      — BLS state-level labor productivity indices.
#   Natural hazards   — FEMA National Risk Index scores.
#   Weather events    — NOAA Storm Events Database counts.
#   Economic health   — BEA GDP + BLS unemployment by state.
#
# The target variable is `acf_atl_baseline`:
#   = project's price_per_sq_ft / CPI-scaled Atlanta median price_per_sq_ft
# Atlanta was chosen as the baseline because it has the most data points.
#
# Best-performing model: LightGBM on log(acf_atl_baseline).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class MaterialPrices:
    """
    Home Depot market-basket material prices (10 categories).

    Scraped via Home Depot's internal API for 3 representative products per
    category, then aggregated to the metro-area × year level.  These capture
    real-world material cost differences across locations — e.g. plywood is
    cheaper in timber-producing regions, concrete is cheaper near quarries.
    """

    concrete_mix: Optional[float] = None
    electrical_wire: Optional[float] = None
    exterior_door_full: Optional[float] = None
    hvac_split_system: Optional[float] = None
    lighting_fixture: Optional[float] = None
    moisture_protection: Optional[float] = None
    plywood_panel: Optional[float] = None
    rebar: Optional[float] = None
    roll_insulation: Optional[float] = None
    white_paint: Optional[float] = None


@dataclass
class LaborWages:
    """
    BLS OEWS (Occupational Employment and Wage Statistics) median hourly wages
    for 8 key construction crafts at the metropolitan-area level.

    These are direct indicators of local labor cost.  The `average` field is
    the mean across all 8 crafts (H_MEDIAN_Avg in the CSV) and serves as a
    single summary of an area's labor cost position.
    """

    carpenters: Optional[float] = None
    cement_masons: Optional[float] = None
    construction_laborers: Optional[float] = None
    electricians: Optional[float] = None
    painters: Optional[float] = None
    plumbers_pipefitters: Optional[float] = None
    roofers: Optional[float] = None
    structural_iron_steel: Optional[float] = None
    average: Optional[float] = None  # mean of the 8 wages above


@dataclass
class TransportLogistics:
    """
    USACE EP 1110-1-8 local area factors (16 features, 2009-present).

    Published by the US Army Corps of Engineers, these capture region-specific
    logistics costs: fuel prices (gasoline, diesel, marine), electricity,
    freight rates by weight class, labor adjustment factors, working-hour
    norms, and sales tax.  Higher fuel and freight costs inflate material
    delivery expenses and are reflected in higher ACFs.
    """

    state_sales_tax_rate: Optional[float] = None
    working_hours_per_year: Optional[float] = None
    labor_adjustment_factor: Optional[float] = None
    electricity_cost_kwh: Optional[float] = None
    gasoline_cost_gal: Optional[float] = None
    diesel_cost_gal_offroad: Optional[float] = None
    diesel_cost_gal_onroad: Optional[float] = None
    marine_cost_gal_gasoline: Optional[float] = None
    marine_cost_gal_diesel: Optional[float] = None
    # Freight rates in USD per hundredweight (cwt), stratified by shipment
    # weight class.  Heavier shipments get discounted rates.
    freight_rate_0_240cwt: Optional[float] = None
    freight_rate_240_300cwt: Optional[float] = None
    freight_rate_300_400cwt: Optional[float] = None
    freight_rate_400_500cwt: Optional[float] = None
    freight_rate_500_700cwt: Optional[float] = None
    freight_rate_700_800cwt: Optional[float] = None
    freight_rate_800_plus_cwt: Optional[float] = None


@dataclass
class LaborProductivity:
    """
    BLS state-level labor productivity indices (11 features, 2007-present).

    These are *index* values (base year = 100).  Higher values mean the
    construction sector in that state has become more productive (or more
    expensive, depending on the metric) relative to the base year.

    Key indices:
      • labor_productivity_idx — output per hour worked; higher = more efficient.
      • unit_labor_costs_idx — labor cost per unit of output; higher = less
        efficient (workers cost more per unit of work produced).
      • real_hourly_compensation_idx — inflation-adjusted pay; captures
        whether worker pay has kept up with / exceeded inflation.
    """

    employment_idx: Optional[float] = None
    hourly_compensation_idx: Optional[float] = None
    hours_worked_idx: Optional[float] = None
    labor_compensation_idx: Optional[float] = None
    labor_productivity_idx: Optional[float] = None
    output_per_worker_idx: Optional[float] = None
    real_hourly_compensation_idx: Optional[float] = None
    real_labor_compensation_idx: Optional[float] = None
    real_value_added_output_idx: Optional[float] = None
    unit_labor_costs_idx: Optional[float] = None
    value_added_output_price_deflator_idx: Optional[float] = None


@dataclass
class NaturalHazardRisk:
    """
    FEMA National Risk Index data at the county level.

    Two metrics per hazard type:
      • EALS — Expected Annual Loss Score (composite dollar-weighted risk).
      • ALRB — Annualized Loss Ratio for Buildings (fraction of building
        value expected to be lost per year due to this hazard).

    Higher values increase construction insurance premiums, require
    more resilient (expensive) design, and can disrupt schedules — all of
    which drive up project costs.

    `composite_eals` is the all-hazards summary score.
    """

    composite_eals: Optional[float] = None
    avalanche_eals: Optional[float] = None
    avalanche_alrb: Optional[float] = None
    coastal_flooding_eals: Optional[float] = None
    coastal_flooding_alrb: Optional[float] = None
    cold_wave_eals: Optional[float] = None
    cold_wave_alrb: Optional[float] = None
    drought_eals: Optional[float] = None
    earthquake_eals: Optional[float] = None
    earthquake_alrb: Optional[float] = None
    hail_eals: Optional[float] = None
    hail_alrb: Optional[float] = None
    heat_wave_eals: Optional[float] = None
    heat_wave_alrb: Optional[float] = None
    hurricane_eals: Optional[float] = None
    hurricane_alrb: Optional[float] = None
    ice_storm_eals: Optional[float] = None
    ice_storm_alrb: Optional[float] = None
    landslide_eals: Optional[float] = None
    landslide_alrb: Optional[float] = None
    lightning_eals: Optional[float] = None
    lightning_alrb: Optional[float] = None
    riverine_flooding_eals: Optional[float] = None
    riverine_flooding_alrb: Optional[float] = None
    strong_wind_eals: Optional[float] = None
    strong_wind_alrb: Optional[float] = None
    tornado_eals: Optional[float] = None
    tornado_alrb: Optional[float] = None
    tsunami_eals: Optional[float] = None
    tsunami_alrb: Optional[float] = None
    volcanic_activity_eals: Optional[float] = None
    volcanic_activity_alrb: Optional[float] = None
    wildfire_eals: Optional[float] = None
    wildfire_alrb: Optional[float] = None
    winter_weather_eals: Optional[float] = None
    winter_weather_alrb: Optional[float] = None


@dataclass
class NOAAWeatherEvents:
    """
    NOAA Storm Events Database: annual episode counts at the city level.

    Each field is the number of times that weather event was recorded in the
    project's city during its project year (31 event categories).  Frequent
    severe weather events can delay construction, damage work in progress,
    and require more robust (expensive) structural design — all of which are
    correlated with higher area cost factors.
    """

    coastal_flood: Optional[float] = None
    cold_wind_chill: Optional[float] = None
    debris_flow: Optional[float] = None
    dense_fog: Optional[float] = None
    drought: Optional[float] = None
    dust_devil: Optional[float] = None
    flash_flood: Optional[float] = None
    flood: Optional[float] = None
    frost_freeze: Optional[float] = None
    funnel_cloud: Optional[float] = None
    hail: Optional[float] = None
    heat: Optional[float] = None
    heavy_rain: Optional[float] = None
    heavy_snow: Optional[float] = None
    high_surf: Optional[float] = None
    high_wind: Optional[float] = None
    hurricane_typhoon: Optional[float] = None
    lightning: Optional[float] = None
    marine_hail: Optional[float] = None
    marine_high_wind: Optional[float] = None
    marine_thunderstorm_wind: Optional[float] = None
    rip_current: Optional[float] = None
    seiche: Optional[float] = None
    storm_surge_tide: Optional[float] = None
    strong_wind: Optional[float] = None
    thunderstorm_wind: Optional[float] = None
    tornado: Optional[float] = None
    volcanic_ash: Optional[float] = None
    waterspout: Optional[float] = None
    wildfire: Optional[float] = None
    winter_weather: Optional[float] = None


@dataclass
class EconomicIndicators:
    """
    BEA (Bureau of Economic Analysis) GDP + BLS unemployment at state level.
    Added in the Fall 2025 cohort.

    GDP fields come from the BEA's SAGDP tables:
      • SAGDP2 — nominal GDP (in current dollars).
      • SAGDP9 — real GDP (adjusted for inflation to a base year).
      Both are provided for all industries and for the construction sector alone.

    `unemployment_rate` and `employment_count` come from BLS LAUS data.
    High unemployment can depress wages (and thus construction costs), while
    a booming economy with low unemployment drives costs up.
    """

    gdp_all_industry_nominal: Optional[float] = None
    gdp_construction_nominal: Optional[float] = None
    gdp_all_industry_real: Optional[float] = None
    gdp_construction_real: Optional[float] = None
    unemployment_rate: Optional[float] = None
    employment_count: Optional[float] = None


@dataclass
class ACFMarketBasketInput:
    """
    Full feature vector for the market-basket ACF model.

    This is the "mega" input that bundles project-level identifiers with all
    the metro-area-level external feature groups (materials, wages, transport,
    productivity, hazards, weather, economics).

    Source CSVs:
      - final_dataset_on_year.csv  (project-level + features joined by metro × year)
      - metro_area_features_pivot_by_year.csv  (metro-level feature matrix)

    Model details:
      - Best model: LightGBM
      - Predicts: log(acf_atl_baseline)
        where acf_atl_baseline = price_per_sq_ft / CPI-scaled Atlanta median ppsf
      - Filtered to: Commercial construction only
      - Outlier removal: IQR + Z-score + IsolationForest (triple filter)
      - Time range: 2009–2024

    At **training** time, the project-level fields (project_id, project_year,
    construction_category, matched_metro_area) are used for joining and
    filtering.  At **prediction** time, only the metro-area features matter —
    you just need to know *where* and *when* to look up the feature values.
    """

    # ── Project-level fields (for join/filter during training) ──
    project_id: Optional[str] = None
    project_year: int = 0
    construction_category: str = "Commercial"
    matched_metro_area: Optional[str] = None

    # ── Metro-area feature groups ──
    # Each is a nested dataclass.  `field(default_factory=...)` is required
    # because Python dataclasses don't allow mutable default values directly.
    materials: MaterialPrices = field(default_factory=MaterialPrices)
    labor_wages: LaborWages = field(default_factory=LaborWages)
    transport_logistics: TransportLogistics = field(default_factory=TransportLogistics)
    labor_productivity: LaborProductivity = field(default_factory=LaborProductivity)
    natural_hazard_risk: NaturalHazardRisk = field(default_factory=NaturalHazardRisk)
    noaa_weather: NOAAWeatherEvents = field(default_factory=NOAAWeatherEvents)
    economic_indicators: EconomicIndicators = field(default_factory=EconomicIndicators)


@dataclass
class ACFMarketBasketOutput:
    """
    ACF predictions from the market-basket ML models.

    Three models are trained and their predictions are all stored:
      • Random Forest  (`acf_rf_norm`)
      • XGBoost        (`acf_xgb_norm`)
      • LightGBM       (`acf_lgbm_norm`)  ← best performer

    All values are **normalised so that Atlanta = 1.0**.  For example,
    acf_lgbm_norm = 1.12 means that metro area is predicted to be 12 % more
    expensive than Atlanta for commercial construction.

    Source: output/normalized_acfs_by_year.csv
    """

    metro_area: str = ""
    project_year: int = 0
    acf_rf_norm: Optional[float] = None  # Random Forest prediction
    acf_xgb_norm: Optional[float] = None  # XGBoost prediction
    acf_lgbm_norm: Optional[float] = None  # LightGBM prediction (best)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. CPI / Inflation Reference
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CPI (Consumer Price Index) data is used to adjust dollar amounts across
# years.  For example, $1 M in 2015 ≠ $1 M in 2024 — CPI lets us convert
# between "nominal" (as-spent) dollars and "real" (constant-purchasing-power)
# dollars.
#
# The USACE benchmark ACFs are published reference values that we compare
# our computed ACFs against to assess accuracy.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class CPIRecord:
    """
    Consumer Price Index record (base year 1982-84 = 100).

    Source: cpi_1982.csv
    Used to compute inflation_factor = avg_cpi(current_year) / avg_cpi(project_year).
    """

    project_year: int
    avg_cpi: float


@dataclass
class USACEBenchmark:
    """
    USACE-published Area Cost Factor for benchmarking our predictions.

    Source: USACE_May2025_ACFs_and_S25_ACFs.csv
    `usace_acf`  — the official USACE value (our ground truth for validation).
    `s25_acf`    — the Summer 2025 cohort's predicted ACF for the same metro
                   area, stored here for easy comparison.
    """

    metro_area: str
    usace_acf: float
    s25_acf: Optional[float] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Unified Pipeline Output  (plan §2.2 + §3.3 combined)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# These two classes represent the FINAL output of the entire system.
#
# The pipeline works in two stages:
#   Stage A — Parametric model produces a "base" cost estimate (national avg).
#   Stage B — ACF engine produces a LocationFactor.
#   Final   — base estimate × location_factor = adjusted estimate.
#
# CostEstimate is the top-level object that the Foundry tool (plan §4)
# returns to the LLM pipeline (plan §5) for natural-language report generation.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class LocationFactor:
    """
    Output of the ACF engine (plan §3.3).

    Merges the best-performing variant from each ACF approach (inflation-
    adjusted cluster vs. market-basket ML) into a single location multiplier.

    Example:
      location_factor = 1.15, base_location = "national_average",
      target_location = "Boston-Cambridge-Newton, MA-NH"
      → Building in the Boston metro is ~15 % more expensive than the
        national average.

    Component breakdown (if available) shows what drives the factor:
      • labor_component    — share attributable to higher/lower wages.
      • materials_component — share from material price differences.
      • equipment_component — share from equipment cost differences.
    """

    location_factor: float  # final multiplier (e.g. 1.15)
    confidence: float  # 0–1; based on data density in the region
    base_location: str = "national_average"
    target_location: str = ""
    labor_component: Optional[float] = None
    materials_component: Optional[float] = None
    equipment_component: Optional[float] = None
    method: str = ""  # "inflation_cluster" or "market_basket_ml"
    model_version: str = ""


@dataclass
class CostEstimate:
    """
    Final unified output combining parametric cost estimate + location adjustment.

    This is the **top-level result** that the Foundry tool (plan §4) returns
    to the LLM pipeline (plan §5) for natural-language report generation.

    The estimate has two layers:
      1. **Base estimate** (before location adjustment) — from the parametric
         regression model, representing a "national average" cost.
      2. **Adjusted estimate** — base × location_factor, reflecting the
         actual expected cost at the project's specific location.

    Each layer has low / mid / high values to convey uncertainty:
      • low  — optimistic (everything goes well, competitive bids).
      • mid  — most-likely point estimate.
      • high — pessimistic (scope growth, supply-chain issues, etc.).

    `within_class5_tolerance` indicates whether the estimate falls within
    AACE Class 5 tolerance (±50 % of actual), which is the standard for
    early-phase conceptual estimates.
    """

    # ── Parametric base estimate (before location adjustment) ──
    base_cost_low: float
    base_cost_mid: float
    base_cost_high: float

    # ── Location-adjusted estimate (base × location_factor) ──
    adjusted_cost_low: float
    adjusted_cost_mid: float
    adjusted_cost_high: float

    # ── Per-square-foot metrics (useful for benchmarking) ──
    cost_per_sf_low: Optional[float] = None
    cost_per_sf_mid: Optional[float] = None
    cost_per_sf_high: Optional[float] = None

    # ── Confidence ──
    confidence_level: float = 0.0  # 0–1 overall confidence
    within_class5_tolerance: bool = True  # True if within ±50 % of actuals

    # ── Location factor that was applied ──
    location_factor: Optional[LocationFactor] = None

    # ── Model metadata ──
    parametric_model_version: str = ""  # e.g. "simple_rf_v2"
    acf_model_version: str = ""  # e.g. "lgbm_market_basket_v3"

    # ── Input echo (included in output for report generation) ──
    # These repeat key inputs so the LLM can reference them in the narrative
    # without needing to look them up separately.
    project_type: str = ""
    project_state: str = ""
    project_city: str = ""
    size_sf: Optional[float] = None  # project size in square feet


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Column-name mappings  (notebook CSV ↔ schema field)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Each research notebook used its own column names in CSV files.  These
# dictionaries translate those raw CSV headers into the standardised schema
# field names defined above.
#
# Usage pattern in the ETL pipeline:
#   df.rename(columns=INFLATION_ACF_CSV_TO_SCHEMA, inplace=True)
#
# When a CSV header maps directly (e.g. "project_id" → "project_id") the
# entry is still listed for completeness.  Non-obvious translations include:
#   "type"           → "project_type"   (avoiding Python keyword conflict)
#   "WPUIP2300001"   → "wpuip2300001"   (lowercase normalisation)
#   "cost_per_sqft"  → "price_per_sq_ft" (naming standardisation)
#   "ZCTA5CE20"      → "zip_code"        (human-readable alias)
#   "DoD_ACF2024"    → "dod_acf_2024"    (snake_case normalisation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Mapping for CSVs used by the inflation-adjusted ACF notebooks.
INFLATION_ACF_CSV_TO_SCHEMA = {
    "project_id": "project_id",
    "labor_total": "labor_total",
    "material_total": "material_total",
    "equipment_total": "equipment_total",
    "subcontractor_total": "subcontractor_total",
    "other_cost1": "other_cost1",
    "other_cost2": "other_cost2",
    "other_cost3": "other_cost3",
    "unit_cost_total": "unit_cost_total",
    "total_mat_lab_equip": "total_mat_lab_equip",
    "project_longitude": "project_longitude",
    "project_latitude": "project_latitude",
    "project_city": "project_city",
    "project_state": "project_state",
    "project_date": "project_date",
    "project_sq_ft": "project_sq_ft",
    "type": "project_type",  # "type" → standardised name
    "project_category": "project_category",
    "construction_category": "construction_category",
    "year_month": "year_month",
    "WPUIP2300001": "wpuip2300001",  # uppercase BLS code → lowercase
    "adjusted_total_mat_lab_equip": "adjusted_total_mat_lab_equip",
    "cost_per_sqft": "price_per_sq_ft",  # naming normalisation
    "ZCTA5CE20": "zip_code",  # Census code → friendly name
    "city": "project_city",  # alternate column name in some CSVs
    "state_name": "state_name",
    "population": "population",
    "density": "density",
    "DoD_ACF2024": "dod_acf_2024",  # CamelCase → snake_case
    "Hour_median_norm": "hour_median_norm",
    "Hour_mean_norm": "hour_mean_norm",
    "db_cluster_41km_2ms": "db_cluster_41km_2ms",
    "hdb_cluster_41km_3ms": "hdb_cluster_41km_3ms",
}

# Mapping for CSVs used by the market-basket ACF notebooks.
# Dot-notation values (e.g. "materials.concrete_mix") indicate fields inside
# nested dataclasses; the ETL code is expected to split on "." and assign
# to the sub-object accordingly.
MARKET_BASKET_CSV_TO_SCHEMA = {
    "project_id": "project_id",
    "construction_category": "construction_category",
    "project_category": "project_category",
    "type": "project_type",
    "project_city": "project_city",
    "project_state": "project_state",
    "project_year": "project_year",
    "project_sq_ft": "project_sq_ft",
    "project_cost": "project_cost",
    "matched_metro_area": "matched_metro_area",
    "match_score": "match_score",
    "metro_area": "metro_area",
    "state": "state_name",  # short CSV header → full name
    "YEAR": "project_year",  # alternate header in some CSVs
    "price_per_sq_ft": "price_per_sq_ft",
    "project_region": "project_region",
    # ── Material prices (nested under MaterialPrices dataclass) ──
    "materials_concrete_mix": "materials.concrete_mix",
    "materials_electrical_wire": "materials.electrical_wire",
    "materials_exterior_door_full": "materials.exterior_door_full",
    "materials_hvac_split_system": "materials.hvac_split_system",
    "materials_lighting_fixture": "materials.lighting_fixture",
    "materials_moisture_protection": "materials.moisture_protection",
    "materials_plywood_panel": "materials.plywood_panel",
    "materials_rebar": "materials.rebar",
    "materials_roll_insulation": "materials.roll_insulation",
    "materials_white_paint": "materials.white_paint",
    # ── Labor wages (nested under LaborWages dataclass) ──
    "H_MEDIAN_carpenters": "labor_wages.carpenters",
    "H_MEDIAN_cement masons and concrete finishers": "labor_wages.cement_masons",
    "H_MEDIAN_construction laborers": "labor_wages.construction_laborers",
    "H_MEDIAN_electricians": "labor_wages.electricians",
    "H_MEDIAN_painters, construction and maintenance": "labor_wages.painters",
    "H_MEDIAN_plumbers, pipefitters, and steamfitters": "labor_wages.plumbers_pipefitters",
    "H_MEDIAN_roofers": "labor_wages.roofers",
    "H_MEDIAN_structural iron and steel workers": "labor_wages.structural_iron_steel",
    "H_MEDIAN_Avg": "labor_wages.average",
    # ── Economic indicators (nested under EconomicIndicators dataclass) ──
    "real_gdp_all_industry_total": "economic_indicators.gdp_all_industry_real",
    "real_gdp_construction": "economic_indicators.gdp_construction_real",
    "unemployment_rate": "economic_indicators.unemployment_rate",
}

# Mapping for the Regression model's base_data_for_model.csv (17,025 projects).
# This CSV comes from the deployed Flask app (Fall 2025 — Aryal, Sawan, Kafwimi).
# Most column names already match RawProjectRecord fields; we only list the
# columns that are actually used or need renaming.
REGRESSION_CSV_TO_SCHEMA: dict[str, str] = {
    "project_id": "project_id",
    "inflation_factor": "inflation_factor",
    "total_project_cost_normalized_2025": "total_project_cost_normalized_2025",
    "official_budget_range": "official_budget_range",
    "ciqs_complexity_category": "ciqs_complexity_category",
    "cnt_division": "cnt_division",
    "cnt_item_code": "cnt_item_code",
    "county_name": "county_name",
    "area_type": "area_type",
    "acf": "acf",
    "project_latitude": "project_latitude",
    "project_longitude": "project_longitude",
    "project_type": "project_type",
    "project_category": "project_category",
    "project_state": "project_state",
}

# Ordered list of feature names expected by the deployed Flask regression app.
# The model was trained with these exact column names in this exact order —
# changing the order or names will break prediction.
REGRESSION_FLASK_FEATURE_NAMES = [
    "inflation_factor",
    "official_budget_range",
    "ciqs_complexity_category",
    "cnt_division",
    "cnt_item_code",
    "county_name",
    "area_type",
    "acf",
    "project_type",
    "project_category",
    "project_state",
    "region",
]

# Feature grouping used by the advanced regression experiment (exp10).
# The training pipeline processes each group differently:
#   • "numeric"     → scaled / normalised as-is.
#   • "categorical" → label-encoded or one-hot-encoded.
#   • "text"        → TF-IDF vectorised (bag-of-words + importance weighting).
#   • "target"      → the column being predicted (not a feature).
REGRESSION_ADVANCED_FEATURE_SETS = {
    "numeric": [
        "ACF",
        "project_year",
        "median_cost_per_unit",
        "median_quantity_most_common_unit",
        "acf_state_norm",
    ],
    "categorical": [
        "project_city",
        "project_state",
        "type",
        "project_category",
        "construction_category",
        "most_common_unit",
        "quantity_bin",
        "scope_cluster",
    ],
    "text": "project_description",
    "target": "inf_adj_total_mat_lab_equip",
}
