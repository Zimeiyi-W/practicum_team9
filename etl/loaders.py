"""
ETL Loaders — read CSV files or the cc_gold database into RawProjectRecord instances.

Each loader:
  1. Reads a CSV with pandas  **or**  queries the cc_gold PostgreSQL database
  2. Renames columns using the mapping dictionaries from schema.py
  3. Converts each row into a RawProjectRecord dataclass

Usage:
    from etl.loaders import load_inflation_acf_projects, load_projects_from_db

    # From CSV
    records = load_inflation_acf_projects("Previous work/ACF/inflation_ACF/data/projects_clusters_log_outliers.csv")

    # From the cc_gold database
    records = load_projects_from_db()
    print(f"Loaded {len(records)} records")
    print(records[0].project_city, records[0].project_state)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
from dotenv import find_dotenv, load_dotenv

# Add the project root to sys.path so we can import schema
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from schema import (
    INFLATION_ACF_CSV_TO_SCHEMA,
    MARKET_BASKET_CSV_TO_SCHEMA,
    REGRESSION_CSV_TO_SCHEMA,
    RawProjectRecord,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_str(value) -> Optional[str]:
    """Convert a value to string, returning None for NaN/None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(value).strip() if str(value).strip() else None


def _safe_float(value) -> Optional[float]:
    """Convert a value to float, returning None for NaN/None/non-numeric."""
    if value is None:
        return None
    try:
        result = float(value)
        return None if np.isnan(result) else result
    except (ValueError, TypeError):
        return None


def _safe_int(value) -> Optional[int]:
    """Convert a value to int, returning None for NaN/None/non-numeric."""
    f = _safe_float(value)
    if f is None:
        return None
    return int(f)


def _row_to_raw_record(row: pd.Series) -> RawProjectRecord:
    """
    Convert a single pandas row (already column-renamed) into a RawProjectRecord.

    Uses the schema field names as keys. Fields not present in the row
    are left at their default (None).
    """
    return RawProjectRecord(
        # -- required --
        project_id=_safe_str(row.get("project_id")) or "UNKNOWN",
        # -- cost components --
        labor_total=_safe_float(row.get("labor_total")),
        material_total=_safe_float(row.get("material_total")),
        equipment_total=_safe_float(row.get("equipment_total")),
        subcontractor_total=_safe_float(row.get("subcontractor_total")),
        other_cost1=_safe_float(row.get("other_cost1")),
        other_cost2=_safe_float(row.get("other_cost2")),
        other_cost3=_safe_float(row.get("other_cost3")),
        unit_cost_total=_safe_float(row.get("unit_cost_total")),
        total_mat_lab_equip=_safe_float(row.get("total_mat_lab_equip")),
        # -- location --
        project_city=_safe_str(row.get("project_city")),
        project_state=_safe_str(row.get("project_state")),
        county_name=_safe_str(row.get("county_name")),
        project_longitude=_safe_float(row.get("project_longitude")),
        project_latitude=_safe_float(row.get("project_latitude")),
        zip_code=_safe_str(row.get("zip_code")),
        # -- metro area --
        matched_metro_area=_safe_str(row.get("matched_metro_area")),
        metro_area=_safe_str(row.get("metro_area")),
        match_score=_safe_float(row.get("match_score")),
        project_region=_safe_str(row.get("project_region")),
        # -- classification --
        construction_category=_safe_str(row.get("construction_category")),
        project_category=_safe_str(row.get("project_category")),
        project_type=_safe_str(row.get("project_type")),
        phase_description=_safe_str(row.get("phase_description")),
        # -- size / units --
        project_sq_ft=_safe_float(row.get("project_sq_ft")),
        project_cost=_safe_float(row.get("project_cost")),
        price_per_sq_ft=_safe_float(row.get("price_per_sq_ft")),
        # -- unit-level --
        most_common_unit=_safe_str(row.get("most_common_unit")),
        median_cost_per_unit=_safe_float(row.get("median_cost_per_unit")),
        median_quantity_most_common_unit=_safe_float(
            row.get("median_quantity_most_common_unit")
        ),
        project_description=_safe_str(row.get("project_description")),
        # -- time --
        project_date=_safe_str(row.get("project_date")),
        project_year=_safe_int(row.get("project_year")),
        year_month=_safe_str(row.get("year_month")),
        # -- external indices --
        dod_acf_2024=_safe_float(row.get("dod_acf_2024")),
        # -- census / geo enrichment --
        population=_safe_int(row.get("population")),
        density=_safe_float(row.get("density")),
        state_name=_safe_str(row.get("state_name")),
        # -- complexity / budget --
        ciqs_complexity_category=_safe_str(row.get("ciqs_complexity_category")),
        official_budget_range=_safe_str(row.get("official_budget_range")),
        cnt_division=_safe_int(row.get("cnt_division")),
        cnt_item_code=_safe_int(row.get("cnt_item_code")),
        area_type=_safe_str(row.get("area_type")),
        # -- PPI --
        wpuip2300001=_safe_float(row.get("wpuip2300001")),
        adjusted_total_mat_lab_equip=_safe_float(
            row.get("adjusted_total_mat_lab_equip")
        ),
        # -- Regression Flask app --
        inflation_factor=_safe_float(row.get("inflation_factor")),
        total_project_cost_normalized_2025=_safe_float(
            row.get("total_project_cost_normalized_2025")
        ),
        acf=_safe_float(row.get("acf")),
    )


# ---------------------------------------------------------------------------
# Database connection & query
# ---------------------------------------------------------------------------

# SQL query from the notebook: fetches projects joined with types, phases,
# and aggregated line-item summaries (total cost, man-hours, unit stats).
_PROJECTS_SQL = """\
WITH
projects AS (
    SELECT *
    FROM public.projects p
    LEFT JOIN public.project_types t
        ON p.project_type_id = t.project_type_id
    LEFT JOIN public.project_phases ph
        ON p.project_phase_id = ph.project_phase_id
),

unit_stats AS (
    SELECT
        project_id,
        unit,
        COUNT(*) AS unit_count,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY quantity)
            AS median_quantity_for_unit
    FROM public.project_line_items
    WHERE unit IS NOT NULL
    GROUP BY project_id, unit
),

most_common_unit AS (
    SELECT DISTINCT ON (project_id)
        project_id,
        unit AS most_common_unit,
        median_quantity_for_unit AS median_quantity_most_common_unit
    FROM unit_stats
    ORDER BY project_id, unit_count DESC
),

line_summary AS (
    SELECT
        li.project_id,
        SUM(li.total_mat_lab_equip) AS total_mat_lab_equip,
        SUM(li.total_man_hours)     AS total_man_hours,
        SUM(li.unit_cost_total)     AS unit_cost_total,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY li.unit_cost_total)
            AS median_cost_per_unit,
        mcu.most_common_unit,
        mcu.median_quantity_most_common_unit
    FROM public.project_line_items li
    LEFT JOIN most_common_unit mcu
        ON li.project_id = mcu.project_id
    WHERE li.unit_cost_total IS NOT NULL
      AND li.unit_cost_total > 0
    GROUP BY
        li.project_id,
        mcu.most_common_unit,
        mcu.median_quantity_most_common_unit
)

SELECT *
FROM projects p
LEFT JOIN line_summary l
    ON p.project_id = l.project_id
"""


def get_db_connection() -> "psycopg2.extensions.connection":
    """
    Open a connection to the cc_gold PostgreSQL database.

    Credentials are read from environment variables (loaded via ``.env``):
        - ``CC_GOLD_DB_HOST``  (default: ccdl-gold.postgres.database.azure.com)
        - ``CC_GOLD_DB_NAME``  (default: cc_gold)
        - ``CC_GOLD_DB_USER``  (default: agb)
        - ``CC_GOLD_DB_PASSWORD`` (no default — required)

    Returns:
        An open psycopg2 connection.
    """
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)
    load_dotenv()

    host = os.getenv("CC_GOLD_DB_HOST", "ccdl-gold.postgres.database.azure.com")
    dbname = os.getenv("CC_GOLD_DB_NAME", "cc_gold")
    user = os.getenv("CC_GOLD_DB_USER", "agb")
    password = os.getenv("CC_GOLD_DB_PASSWORD")
    if password is None:
        raise EnvironmentError(
            "CC_GOLD_DB_PASSWORD is not set. Add it to your .env file or "
            "export it as an environment variable."
        )

    return psycopg2.connect(
        host=host,
        dbname=dbname,
        user=user,
        password=password,
        sslmode="require",
    )


# ---------------------------------------------------------------------------
# Loader 0: Database → RawProjectRecord
# ---------------------------------------------------------------------------


def load_projects_from_db(
    conn: Optional["psycopg2.extensions.connection"] = None,
) -> list[RawProjectRecord]:
    """
    Fetch projects directly from the cc_gold database and return RawProjectRecords.

    This replicates the ``import_projects()`` query from the analysis notebooks.
    The query joins ``public.projects`` with project types, phases, and an
    aggregated line-item summary (total cost, man-hours, unit statistics).

    Note: the query may take ~10 minutes depending on database load.

    Args:
        conn: An open psycopg2 connection.  If *None*, one is created (and
              closed afterwards) via :func:`get_db_connection`.

    Returns:
        List of RawProjectRecord instances, one per project row.
    """
    close_when_done = conn is None
    if conn is None:
        conn = get_db_connection()

    try:
        logger.info("Fetching projects from cc_gold database (this may take ~10 min)…")
        df = pd.read_sql(_PROJECTS_SQL, conn)
        logger.info("  Raw shape from DB: %s", df.shape)
    finally:
        if close_when_done:
            conn.close()

    # The DB returns duplicate column names from the SELECT * joins.
    # pandas suffixes them with ".1" — drop those duplicates.
    dupe_cols = [c for c in df.columns if c.endswith(".1")]
    if dupe_cols:
        df.drop(columns=dupe_cols, inplace=True)

    # DB column names already match most RawProjectRecord fields.
    # Map the few that differ:
    db_to_schema = {
        "type": "project_type",
        "phase_description": "phase_description",
    }
    df.rename(columns=db_to_schema, inplace=True)

    records = [_row_to_raw_record(row) for _, row in df.iterrows()]
    logger.info("  Loaded %d records from database", len(records))
    return records


# ---------------------------------------------------------------------------
# Loader 1: Inflation-ACF projects (clustered)
# ---------------------------------------------------------------------------


def load_inflation_acf_projects(csv_path: str) -> list[RawProjectRecord]:
    """
    Load projects_clusters_log_outliers.csv → list of RawProjectRecord.

    This CSV contains 1,819 projects with DBSCAN/HDBSCAN cluster assignments,
    BLS labor wage normalization, PPI-adjusted costs, and Census enrichment.

    Columns unique to this source (not mapped to RawProjectRecord, kept for reference):
      - geometry          (WKT point, used in geo-joins)
      - index_right       (spatial join artifact)
      - zip               (raw zip, mapped via ZCTA5CE20 → zip_code)
      - PRIM_STATE        (duplicate of project_state)
      - db_cluster_41km_2ms  (DBSCAN cluster id)
      - hdb_cluster_41km_3ms (HDBSCAN cluster id)

    Args:
        csv_path: Path to projects_clusters_log_outliers.csv

    Returns:
        List of RawProjectRecord instances, one per CSV row.
    """
    logger.info("Loading inflation ACF projects from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("  Raw shape: %s", df.shape)

    # Merge duplicate-target columns *before* rename so pandas never
    # creates two columns with the same name.
    # Both "project_city" and "city" map to "project_city" in the schema;
    # prefer the original project_city (uppercase) and fall back to census "city".
    if "city" in df.columns and "project_city" in df.columns:
        df["project_city"] = df["project_city"].fillna(df["city"])
        df.drop(columns=["city"], inplace=True)

    # Build a rename dict that excludes the already-handled "city" key
    renames = {k: v for k, v in INFLATION_ACF_CSV_TO_SCHEMA.items() if k != "city"}
    df.rename(columns=renames, inplace=True)

    records = [_row_to_raw_record(row) for _, row in df.iterrows()]
    logger.info("  Loaded %d records", len(records))
    return records


# ---------------------------------------------------------------------------
# Loader 2: Inflation-ACF projects (preprocessed, no clusters)
# ---------------------------------------------------------------------------


def load_preprocessed_projects(csv_path: str) -> list[RawProjectRecord]:
    """
    Load projects_preprocessed.csv → list of RawProjectRecord.

    Same schema as the clustered file but without the DBSCAN/HDBSCAN columns.
    Contains 1,819 projects after preprocessing (outlier removal, geo-joins).

    Args:
        csv_path: Path to projects_preprocessed.csv

    Returns:
        List of RawProjectRecord instances.
    """
    logger.info("Loading preprocessed projects from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("  Raw shape: %s", df.shape)

    if "city" in df.columns and "project_city" in df.columns:
        df["project_city"] = df["project_city"].fillna(df["city"])
        df.drop(columns=["city"], inplace=True)

    renames = {k: v for k, v in INFLATION_ACF_CSV_TO_SCHEMA.items() if k != "city"}
    df.rename(columns=renames, inplace=True)

    records = [_row_to_raw_record(row) for _, row in df.iterrows()]
    logger.info("  Loaded %d records", len(records))
    return records


# ---------------------------------------------------------------------------
# Loader 3: Market-basket ACF dataset
# ---------------------------------------------------------------------------


def load_market_basket_projects(csv_path: str) -> list[RawProjectRecord]:
    """
    Load final_dataset_on_year.csv → list of RawProjectRecord.

    This CSV contains 932 projects with metro-area-level features:
      - Material prices (Home Depot), labor wages (BLS OEWS)
      - Transport/logistics (USACE EP), labor productivity (BLS)
      - Natural hazard risk (FEMA NRI), weather events (NOAA)
      - Economic indicators (BEA GDP, BLS unemployment)

    The metro-area features (100+ columns) are NOT mapped into
    RawProjectRecord — they belong in ACFMarketBasketInput and will be
    handled by the transform layer (etl/transforms.py). This loader
    extracts only the project-level fields that fit RawProjectRecord.

    Args:
        csv_path: Path to final_dataset_on_year.csv

    Returns:
        List of RawProjectRecord instances.
    """
    logger.info("Loading market-basket projects from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("  Raw shape: %s", df.shape)

    # Only rename the project-level columns (not the 100+ metro features)
    project_level_renames = {
        k: v
        for k, v in MARKET_BASKET_CSV_TO_SCHEMA.items()
        if "." not in v  # skip nested fields like "materials.concrete_mix"
    }

    # Merge duplicate-target columns *before* rename:
    # Both "project_year" and "YEAR" map to "project_year" in the schema.
    # Prefer the existing project_year column; fall back to YEAR for NaN rows.
    if "YEAR" in df.columns and "project_year" in df.columns:
        df["project_year"] = df["project_year"].fillna(df["YEAR"])
        df.drop(columns=["YEAR"], inplace=True)
        project_level_renames = {
            k: v for k, v in project_level_renames.items() if k != "YEAR"
        }

    df.rename(columns=project_level_renames, inplace=True)

    records = [_row_to_raw_record(row) for _, row in df.iterrows()]
    logger.info("  Loaded %d records", len(records))
    return records


# ---------------------------------------------------------------------------
# Loader 4: Regression Flask app dataset
# ---------------------------------------------------------------------------


def load_regression_projects(csv_path: str) -> list[RawProjectRecord]:
    """
    Load base_data_for_model.csv → list of RawProjectRecord.

    This CSV contains 17,025 projects from the deployed Flask regression app
    (Fall 2025 — Aryal, Sawan, Kafwimi).  It includes 38 features covering
    project cost, location, complexity, budget range, and construction details.

    The target variable ``total_project_cost_normalized_2025`` is the
    inflation-adjusted total project cost in 2025 dollars.

    Args:
        csv_path: Path to base_data_for_model.csv

    Returns:
        List of RawProjectRecord instances, one per CSV row.
    """
    logger.info("Loading regression projects from %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("  Raw shape: %s", df.shape)

    # Rename columns to match schema
    renames = {k: v for k, v in REGRESSION_CSV_TO_SCHEMA.items() if k in df.columns and k != v}
    df.rename(columns=renames, inplace=True)

    records = [_row_to_raw_record(row) for _, row in df.iterrows()]
    logger.info("  Loaded %d records", len(records))
    return records


# ---------------------------------------------------------------------------
# Convenience: load all available data
# ---------------------------------------------------------------------------


def load_all_projects(
    data_dir: str,
    include_db: bool = False,
) -> dict[str, list[RawProjectRecord]]:
    """
    Discover and load all known CSV files from the Previous work directory,
    and optionally fetch fresh data from the cc_gold database.

    Args:
        data_dir: Path to the 'Previous work' directory.
        include_db: If *True*, also fetch projects from the cc_gold database
                    (requires ``CC_GOLD_DB_PASSWORD`` in the environment).

    Returns:
        Dict mapping source name → list of records:
          - "inflation_acf_clustered"   → from projects_clusters_log_outliers.csv
          - "inflation_acf_preprocessed" → from projects_preprocessed.csv
          - "market_basket"             → from final_dataset_on_year.csv
          - "database"                  → from cc_gold PostgreSQL (if *include_db*)
    """
    base = Path(data_dir)
    result: dict[str, list[RawProjectRecord]] = {}

    paths = {
        "inflation_acf_clustered": (
            base
            / "ACF"
            / "inflation_ACF"
            / "data"
            / "projects_clusters_log_outliers.csv"
        ),
        "inflation_acf_preprocessed": (
            base / "ACF" / "inflation_ACF" / "data" / "projects_preprocessed.csv"
        ),
        "market_basket": (
            base / "ACF" / "market_basket_acf" / "data" / "final_dataset_on_year.csv"
        ),
    }

    # Also check for the regression dataset in a sibling Regression/data/ dir
    regression_path = base / "Regression" / "data" / "base_data_for_model.csv"
    if regression_path.exists():
        paths["regression"] = regression_path

    loaders = {
        "inflation_acf_clustered": load_inflation_acf_projects,
        "inflation_acf_preprocessed": load_preprocessed_projects,
        "market_basket": load_market_basket_projects,
        "regression": load_regression_projects,
    }

    for name, path in paths.items():
        if path.exists():
            result[name] = loaders[name](str(path))
        else:
            logger.warning("  File not found, skipping: %s", path)

    if include_db:
        try:
            result["database"] = load_projects_from_db()
        except Exception:
            logger.exception("  Failed to load from database, skipping")

    return result
