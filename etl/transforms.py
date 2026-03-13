"""
ETL Transforms — convert RawProjectRecord into model-specific input dataclasses.

Each transform function:
  1. Takes a RawProjectRecord (the universal loader output)
  2. Extracts and validates the subset of fields needed by a specific model
  3. Returns the appropriate typed dataclass

Transform functions for market-basket features take a raw pandas row
(before the loader drops metro-area columns) so they can populate the
nested dataclasses (MaterialPrices, LaborWages, etc.).

Usage:
    from etl.loaders import load_inflation_acf_projects
    from etl.transforms import raw_to_acf_inflation_input

    records = load_inflation_acf_projects("path/to/csv")
    acf_inputs = [raw_to_acf_inflation_input(r) for r in records]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from schema import (
    ACFInflationInput,
    ACFMarketBasketInput,
    EconomicIndicators,
    LaborProductivity,
    LaborWages,
    MaterialPrices,
    NaturalHazardRisk,
    NOAAWeatherEvents,
    RawProjectRecord,
    RegressionAdvancedInput,
    RegressionSimpleInput,
    TransportLogistics,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
        return None if np.isnan(result) else result
    except (ValueError, TypeError):
        return None


def _extract_year(rec: RawProjectRecord) -> Optional[int]:
    """Extract a 4-digit year from the best available time field.

    Priority: project_year > year_month > project_date.
    The inflation ACF CSV has no project_year column, but does have
    year_month ("1993-06") and project_date ("1993-06-01").
    """
    if rec.project_year is not None:
        return rec.project_year
    if rec.year_month:
        try:
            return int(rec.year_month[:4])
        except (ValueError, IndexError):
            pass
    if rec.project_date:
        try:
            return int(rec.project_date[:4])
        except (ValueError, IndexError):
            pass
    return None


# ---------------------------------------------------------------------------
# 1. RawProjectRecord → RegressionSimpleInput
# ---------------------------------------------------------------------------


def raw_to_regression_simple(
    rec: RawProjectRecord,
    *,
    inflation_factor: Optional[float] = None,
    acf: Optional[float] = None,
    region: str = "Region_0",
) -> RegressionSimpleInput:
    """
    Convert a RawProjectRecord into a RegressionSimpleInput (13 features).

    The Flask regression model requires `inflation_factor`, `acf`, and `region`
    which are NOT stored in the raw record — they must be supplied externally
    (from the CPI table, ACF engine, and KMeans geo-clustering, respectively).

    When loading from the regression CSV (base_data_for_model.csv), the record
    already contains pre-computed ``inflation_factor`` and ``acf`` values.
    If ``inflation_factor`` or ``acf`` are not provided as arguments, the
    function falls back to the values on the record (if present), or to 1.0.

    Args:
        rec: A loaded RawProjectRecord.
        inflation_factor: CPI ratio (current_year / project_year).
            Falls back to rec.inflation_factor, then 1.0.
        acf: Area Cost Factor from the ACF engine.
            Falls back to rec.acf, then 1.0.
        region: Geographic cluster label (Region_0 … Region_3). Default "Region_0".

    Returns:
        RegressionSimpleInput ready for the Flask model.

    Raises:
        ValueError: If required fields are missing from the record.
    """
    missing = []
    if not rec.project_type:
        missing.append("project_type")
    if not rec.project_category:
        missing.append("project_category")
    if not rec.project_state:
        missing.append("project_state")
    if missing:
        raise ValueError(
            f"RawProjectRecord {rec.project_id} is missing required fields: {missing}"
        )

    # Resolve inflation_factor: argument > record value > default 1.0
    resolved_inflation = inflation_factor
    if resolved_inflation is None:
        resolved_inflation = rec.inflation_factor if rec.inflation_factor is not None else 1.0

    # Resolve acf: argument > record value > default 1.0
    resolved_acf = acf
    if resolved_acf is None:
        resolved_acf = rec.acf if rec.acf is not None else 1.0

    return RegressionSimpleInput(
        inflation_factor=resolved_inflation,
        acf=resolved_acf,
        project_type=rec.project_type or "",
        project_category=rec.project_category or "",
        ciqs_complexity_category=rec.ciqs_complexity_category or "Category 1",
        official_budget_range=rec.official_budget_range or "$0-$1M",
        project_state=rec.project_state or "",
        county_name=rec.county_name or "",
        area_type=rec.area_type or "Urban",
        region=region,
        cnt_division=rec.cnt_division or 0,
        cnt_item_code=rec.cnt_item_code or 0,
    )


# ---------------------------------------------------------------------------
# 2. RawProjectRecord → RegressionAdvancedInput
# ---------------------------------------------------------------------------


def raw_to_regression_advanced(
    rec: RawProjectRecord,
    *,
    acf: float = 1.0,
    acf_state_norm: Optional[float] = None,
    quantity_bin: str = "",
    scope_cluster: int = -1,
) -> RegressionAdvancedInput:
    """
    Convert a RawProjectRecord into a RegressionAdvancedInput (exp10 features).

    Like the simple variant, some derived features must be supplied externally:
      - `acf`: from the ACF engine or city_area_cost_factors table.
      - `acf_state_norm`: state-average ACF / overall mean ACF.
      - `quantity_bin`: binned median_quantity_most_common_unit (from pd.cut).
      - `scope_cluster`: KMeans(n=15) cluster id based on geo+cost+unit.

    Args:
        rec: A loaded RawProjectRecord.
        acf: Area Cost Factor. Default 1.0.
        acf_state_norm: State-normalised ACF. Default None.
        quantity_bin: Binned quantity range string. Default "".
        scope_cluster: KMeans cluster id. Default -1.

    Returns:
        RegressionAdvancedInput ready for the advanced model.

    Raises:
        ValueError: If required numeric fields are missing.
    """
    year = _extract_year(rec)
    if year is None:
        raise ValueError(
            f"RawProjectRecord {rec.project_id} is missing required field: project_year"
            " (also checked year_month and project_date)"
        )

    return RegressionAdvancedInput(
        acf=acf,
        project_year=year,
        median_cost_per_unit=rec.median_cost_per_unit or 0.0,
        median_quantity_most_common_unit=rec.median_quantity_most_common_unit or 0.0,
        acf_state_norm=acf_state_norm,
        project_city=rec.project_city or "",
        project_state=rec.project_state or "",
        project_type=rec.project_type or "",
        project_category=rec.project_category or "",
        construction_category=rec.construction_category or "",
        most_common_unit=rec.most_common_unit or "",
        quantity_bin=quantity_bin,
        scope_cluster=scope_cluster,
        project_description=rec.project_description,
    )


# ---------------------------------------------------------------------------
# 3. RawProjectRecord → ACFInflationInput
# ---------------------------------------------------------------------------


def raw_to_acf_inflation_input(rec: RawProjectRecord) -> ACFInflationInput:
    """
    Convert a RawProjectRecord into an ACFInflationInput.

    This transform is straightforward because all required fields live directly
    on the RawProjectRecord (loaded from projects_clusters_log_outliers.csv).

    Args:
        rec: A loaded RawProjectRecord (from the inflation ACF CSV).

    Returns:
        ACFInflationInput ready for the clustering-based ACF engine.

    Raises:
        ValueError: If required location/cost fields are missing.
    """
    missing = []
    if not rec.project_state:
        missing.append("project_state")
    if not rec.project_city:
        missing.append("project_city")
    if rec.project_latitude is None:
        missing.append("project_latitude")
    if rec.project_longitude is None:
        missing.append("project_longitude")
    if rec.price_per_sq_ft is None:
        missing.append("price_per_sq_ft")
    if rec.project_sq_ft is None:
        missing.append("project_sq_ft")
    if missing:
        raise ValueError(
            f"RawProjectRecord {rec.project_id} is missing required fields: {missing}"
        )

    return ACFInflationInput(
        project_state=rec.project_state or "",
        project_city=rec.project_city or "",
        project_latitude=rec.project_latitude or 0.0,
        project_longitude=rec.project_longitude or 0.0,
        cost_per_sqft=rec.price_per_sq_ft or 0.0,
        project_sq_ft=rec.project_sq_ft or 0.0,
        project_type=rec.project_type or "",
        db_cluster_41km_2ms=None,
        hdb_cluster_41km_3ms=None,
        hour_median_norm=None,
        hour_mean_norm=None,
        wpuip2300001=rec.wpuip2300001,
        adjusted_total_mat_lab_equip=rec.adjusted_total_mat_lab_equip,
        dod_acf_2024=rec.dod_acf_2024,
    )


# ---------------------------------------------------------------------------
# 4. Market-basket feature extraction from raw pandas row
# ---------------------------------------------------------------------------
#
# The market-basket CSV has 100+ metro-area feature columns that the loader
# intentionally drops (they don't fit in RawProjectRecord). To populate
# ACFMarketBasketInput, we need to read them directly from the DataFrame row.
#
# Column-name → dataclass-field mappings for each nested feature group:

_MATERIAL_COL_MAP = {
    "materials_concrete_mix": "concrete_mix",
    "materials_electrical_wire": "electrical_wire",
    "materials_exterior_door_full": "exterior_door_full",
    "materials_hvac_split_system": "hvac_split_system",
    "materials_lighting_fixture": "lighting_fixture",
    "materials_moisture_protection": "moisture_protection",
    "materials_plywood_panel": "plywood_panel",
    "materials_rebar": "rebar",
    "materials_roll_insulation": "roll_insulation",
    "materials_white_paint": "white_paint",
}

_LABOR_WAGES_COL_MAP = {
    "H_MEDIAN_carpenters": "carpenters",
    "H_MEDIAN_cement masons and concrete finishers": "cement_masons",
    "H_MEDIAN_construction laborers": "construction_laborers",
    "H_MEDIAN_electricians": "electricians",
    "H_MEDIAN_painters, construction and maintenance": "painters",
    "H_MEDIAN_plumbers, pipefitters, and steamfitters": "plumbers_pipefitters",
    "H_MEDIAN_roofers": "roofers",
    "H_MEDIAN_structural iron and steel workers": "structural_iron_steel",
    "H_MEDIAN_Avg": "average",
}

_TRANSPORT_COL_MAP = {
    "transport_logistics_total_state_sales_or_import_tax_rate": "state_sales_tax_rate",
    "transport_logistics_working_hours_per_year_whpy": "working_hours_per_year",
    "transport_logistics_labor_adjustment_factor_laf": "labor_adjustment_factor",
    "transport_logistics_electricity_cost_per_kilowatt_hour": "electricity_cost_kwh",
    "transport_logistics_gasoline_cost_per_gallon": "gasoline_cost_gal",
    "transport_logistics_diesel_cost_per_gallon_off_road_use": "diesel_cost_gal_offroad",
    "transport_logistics_diesel_cost_per_gallon_on_road_use": "diesel_cost_gal_onroad",
    "transport_logistics_marine_cost_per_gallon_gasoline": "marine_cost_gal_gasoline",
    "transport_logistics_marine_cost_per_gallon_diesel": "marine_cost_gal_diesel",
    "transport_logistics_freight_rates_0_cwt_240_cwt": "freight_rate_0_240cwt",
    "transport_logistics_freight_rates_240_cwt_300_cwt": "freight_rate_240_300cwt",
    "transport_logistics_freight_rates_300_cwt_400_cwt": "freight_rate_300_400cwt",
    "transport_logistics_freight_rates_400_cwt_500_cwt": "freight_rate_400_500cwt",
    "transport_logistics_freight_rates_500_cwt_700_cwt": "freight_rate_500_700cwt",
    "transport_logistics_freight_rates_700_cwt_800_cwt": "freight_rate_700_800cwt",
    "transport_logistics_freight_rates_800_cwt_and_over": "freight_rate_800_plus_cwt",
}

_LABOR_PROD_COL_MAP = {
    "labor_prod_employment_idx": "employment_idx",
    "labor_prod_hourly_compensation_idx": "hourly_compensation_idx",
    "labor_prod_hours_worked_idx": "hours_worked_idx",
    "labor_prod_labor_compensation_idx": "labor_compensation_idx",
    "labor_prod_labor_productivity_idx": "labor_productivity_idx",
    "labor_prod_output_per_worker_idx": "output_per_worker_idx",
    "labor_prod_real_hourly_compensation_idx": "real_hourly_compensation_idx",
    "labor_prod_real_labor_compensation_idx": "real_labor_compensation_idx",
    "labor_prod_real_value-added_output_idx": "real_value_added_output_idx",
    "labor_prod_unit_labor_costs_idx": "unit_labor_costs_idx",
    "labor_prod_value-added_output_price_deflator_idx": "value_added_output_price_deflator_idx",
}

_NRI_COL_MAP = {
    "nri_eals_composite": "composite_eals",
    "nri_avalanche_eals": "avalanche_eals",
    "nri_avalanche_alrb": "avalanche_alrb",
    "nri_coastal_flooding_eals": "coastal_flooding_eals",
    "nri_coastal_flooding_alrb": "coastal_flooding_alrb",
    "nri_cold_wave_eals": "cold_wave_eals",
    "nri_cold_wave_alrb": "cold_wave_alrb",
    "nri_drought_eals": "drought_eals",
    "nri_earthquake_eals": "earthquake_eals",
    "nri_earthquake_alrb": "earthquake_alrb",
    "nri_hail_eals": "hail_eals",
    "nri_hail_alrb": "hail_alrb",
    "nri_heat_wave__eals": "heat_wave_eals",
    "nri_heat_wave__alrb": "heat_wave_alrb",
    "nri_hurricane_eals": "hurricane_eals",
    "nri_hurricane_alrb": "hurricane_alrb",
    "nri_ice_storm__eals": "ice_storm_eals",
    "nri_ice_storm__alrb": "ice_storm_alrb",
    "nri_landslide_eals": "landslide_eals",
    "nri_landslide_alrb": "landslide_alrb",
    "nri_lightning_eals": "lightning_eals",
    "nri_lightning_alrb": "lightning_alrb",
    "nri_riverine_flooding_eals": "riverine_flooding_eals",
    "nri_riverine_flooding_alrb": "riverine_flooding_alrb",
    "nri_strong_wind_eals": "strong_wind_eals",
    "nri_strong_wind_alrb": "strong_wind_alrb",
    "nri_tornado_eals": "tornado_eals",
    "nri_tornado_alrb": "tornado_alrb",
    "nri_tsunami_eals": "tsunami_eals",
    "nri_tsunami_alrb": "tsunami_alrb",
    "nri_volcanic_activity_eals": "volcanic_activity_eals",
    "nri_volcanic_activity_alrb": "volcanic_activity_alrb",
    "nri_wildfire_eals": "wildfire_eals",
    "nri_wildfire_alrb": "wildfire_alrb",
    "nri_winter_weather_eals": "winter_weather_eals",
    "nri_winter_weather_alrb": "winter_weather_alrb",
}

_NOAA_COL_MAP = {
    "noaa_coastal_flood": "coastal_flood",
    "noaa_cold_wind_chill": "cold_wind_chill",
    "noaa_debris_flow": "debris_flow",
    "noaa_dense_fog": "dense_fog",
    "noaa_drought": "drought",
    "noaa_dust_devil": "dust_devil",
    "noaa_flash_flood": "flash_flood",
    "noaa_flood": "flood",
    "noaa_frost_freeze": "frost_freeze",
    "noaa_funnel_cloud": "funnel_cloud",
    "noaa_hail": "hail",
    "noaa_heat": "heat",
    "noaa_heavy_rain": "heavy_rain",
    "noaa_heavy_snow": "heavy_snow",
    "noaa_high_surf": "high_surf",
    "noaa_high_wind": "high_wind",
    "noaa_hurricane_typhoon": "hurricane_typhoon",
    "noaa_lightning": "lightning",
    "noaa_marine_hail": "marine_hail",
    "noaa_marine_high_wind": "marine_high_wind",
    "noaa_marine_thunderstorm_wind": "marine_thunderstorm_wind",
    "noaa_rip_current": "rip_current",
    "noaa_seiche": "seiche",
    "noaa_storm_surge_tide": "storm_surge_tide",
    "noaa_strong_wind": "strong_wind",
    "noaa_thunderstorm_wind": "thunderstorm_wind",
    "noaa_tornado": "tornado",
    "noaa_volcanic_ash": "volcanic_ash",
    "noaa_waterspout": "waterspout",
    "noaa_wildfire": "wildfire",
    "noaa_winter_weather": "winter_weather",
}

_ECONOMIC_COL_MAP = {
    "gdp_all_industry_total": "gdp_all_industry_nominal",
    "gdp_construction": "gdp_construction_nominal",
    "real_gdp_all_industry_total": "gdp_all_industry_real",
    "real_gdp_construction": "gdp_construction_real",
    "unemployment_rate": "unemployment_rate",
    "employment": "employment_count",
}


def _extract_nested(row: pd.Series, col_map: dict[str, str]) -> dict:
    """Extract fields from a pandas row using a CSV-column → field-name mapping."""
    result = {}
    for csv_col, field_name in col_map.items():
        result[field_name] = _safe_float(row.get(csv_col))
    return result


def row_to_market_basket_input(row: pd.Series) -> ACFMarketBasketInput:
    """
    Build an ACFMarketBasketInput directly from a raw pandas row.

    This function reads the 100+ metro-area feature columns that the
    RawProjectRecord loader deliberately skips. Call it on raw DataFrame rows
    *before* the loader drops those columns, or on a separately-read DataFrame.

    Args:
        row: A single row from final_dataset_on_year.csv (as pd.Series).

    Returns:
        ACFMarketBasketInput with all nested feature groups populated.
    """
    project_year_val = _safe_float(row.get("project_year"))
    if project_year_val is None:
        project_year_val = _safe_float(row.get("YEAR"))

    return ACFMarketBasketInput(
        project_id=str(row.get("project_id", "")) or None,
        project_year=int(project_year_val) if project_year_val else 0,
        construction_category=str(row.get("construction_category", "Commercial")),
        matched_metro_area=str(row.get("matched_metro_area", ""))
        if pd.notna(row.get("matched_metro_area"))
        else None,
        materials=MaterialPrices(**_extract_nested(row, _MATERIAL_COL_MAP)),
        labor_wages=LaborWages(**_extract_nested(row, _LABOR_WAGES_COL_MAP)),
        transport_logistics=TransportLogistics(
            **_extract_nested(row, _TRANSPORT_COL_MAP)
        ),
        labor_productivity=LaborProductivity(
            **_extract_nested(row, _LABOR_PROD_COL_MAP)
        ),
        natural_hazard_risk=NaturalHazardRisk(**_extract_nested(row, _NRI_COL_MAP)),
        noaa_weather=NOAAWeatherEvents(**_extract_nested(row, _NOAA_COL_MAP)),
        economic_indicators=EconomicIndicators(
            **_extract_nested(row, _ECONOMIC_COL_MAP)
        ),
    )


def load_market_basket_full(csv_path: str) -> list[ACFMarketBasketInput]:
    """
    Convenience: load final_dataset_on_year.csv directly into ACFMarketBasketInput.

    Unlike the loader in etl/loaders.py (which only extracts project-level fields
    into RawProjectRecord), this reads ALL 108 columns and populates the full
    nested feature structure.

    Args:
        csv_path: Path to final_dataset_on_year.csv.

    Returns:
        List of ACFMarketBasketInput instances.
    """
    logger.info("Loading full market-basket features from %s", csv_path)
    df = pd.read_csv(csv_path)

    if "YEAR" in df.columns and "project_year" in df.columns:
        df["project_year"] = df["project_year"].fillna(df["YEAR"])

    records = [row_to_market_basket_input(row) for _, row in df.iterrows()]
    logger.info("  Loaded %d market-basket records", len(records))
    return records


# ---------------------------------------------------------------------------
# 5. Batch transform helpers
# ---------------------------------------------------------------------------


def batch_raw_to_acf_inflation(
    records: list[RawProjectRecord],
    *,
    skip_invalid: bool = True,
) -> list[ACFInflationInput]:
    """
    Convert a batch of RawProjectRecords to ACFInflationInput, optionally
    skipping records that are missing required fields.

    Args:
        records: List of RawProjectRecords (from the inflation ACF loader).
        skip_invalid: If True, silently skip records with missing required
            fields. If False, raise on the first invalid record.

    Returns:
        List of ACFInflationInput instances.
    """
    results = []
    skipped = 0
    for rec in records:
        try:
            results.append(raw_to_acf_inflation_input(rec))
        except ValueError:
            if not skip_invalid:
                raise
            skipped += 1
    if skipped:
        logger.warning(
            "  Skipped %d / %d records with missing required fields",
            skipped,
            len(records),
        )
    return results


def batch_raw_to_regression_advanced(
    records: list[RawProjectRecord],
    *,
    acf_lookup: Optional[dict[str, float]] = None,
    acf_state_norm_lookup: Optional[dict[str, float]] = None,
    skip_invalid: bool = True,
) -> list[RegressionAdvancedInput]:
    """
    Convert a batch of RawProjectRecords to RegressionAdvancedInput.

    Args:
        records: List of RawProjectRecords.
        acf_lookup: Optional dict mapping project_id → ACF value.
        acf_state_norm_lookup: Optional dict mapping state code → acf_state_norm.
        skip_invalid: If True, silently skip invalid records.

    Returns:
        List of RegressionAdvancedInput instances.
    """
    results = []
    skipped = 0
    for rec in records:
        try:
            acf = (acf_lookup or {}).get(rec.project_id, 1.0)
            acf_sn = (acf_state_norm_lookup or {}).get(rec.project_state or "", None)
            results.append(
                raw_to_regression_advanced(rec, acf=acf, acf_state_norm=acf_sn)
            )
        except ValueError:
            if not skip_invalid:
                raise
            skipped += 1
    if skipped:
        logger.warning(
            "  Skipped %d / %d records with missing required fields",
            skipped,
            len(records),
        )
    return results
