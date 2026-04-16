"""
tools/estimate_tool.py — Unified cost estimation tool

Orchestrates:
  1. ACF lookup for location adjustment
  2. Construction of a canonical regression input
  3. Parametric baseline prediction
  4. Final structured estimate response

This is designed to be callable from:
  - Foundry tool calling
  - FastAPI /estimate endpoint
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from schema import RawProjectRecord
from etl.transforms import raw_to_regression_simple
from models.parametric import ParametricEngine
from tools.acf_tool import get_acf_factor

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "saved" / "parametric"

_parametric_engine: Optional[ParametricEngine] = None


def _load_parametric_engine() -> ParametricEngine:
    """Load the saved parametric engine once and cache it in memory."""
    global _parametric_engine
    if _parametric_engine is None:
        _parametric_engine = ParametricEngine.load(str(_DEFAULT_MODEL_DIR))
        if not _parametric_engine.simple_is_trained:
            raise RuntimeError(
                f"Simple parametric model not found or not trained in {_DEFAULT_MODEL_DIR}"
            )
    return _parametric_engine


def _infer_area_type(county_name: str, area_type: Optional[str]) -> str:
    """Return a safe area type value for the simple regression model."""
    if area_type in {"Urban", "Rural"}:
        return area_type
    return "Urban"


def _infer_region(state: str) -> str:
    """
    Temporary fallback geographic region label for the simple model.

    This should eventually be replaced by the persisted KMeans geo-cluster
    assignment used during training/inference. For now, use a stable mapping
    so the orchestration path works deterministically.
    """
    state = (state or "").upper()

    west = {"CA", "OR", "WA", "NV", "AZ", "UT", "CO", "ID", "MT", "WY", "AK", "HI", "NM"}
    south = {"TX", "OK", "LA", "AR", "MS", "AL", "GA", "FL", "SC", "NC", "TN", "KY", "VA", "WV"}
    midwest = {"ND", "SD", "NE", "KS", "MN", "IA", "MO", "WI", "IL", "IN", "MI", "OH"}
    northeast = {"PA", "NY", "NJ", "DE", "MD", "CT", "RI", "MA", "VT", "NH", "ME", "DC"}

    if state in west:
        return "Region_0"
    if state in south:
        return "Region_1"
    if state in midwest:
        return "Region_2"
    if state in northeast:
        return "Region_3"
    return "Region_0"


def _build_raw_record(
    *,
    project_description: str,
    city: str,
    state: str,
    county_name: str,
    project_type: str,
    project_category: str,
    ciqs_complexity_category: str,
    official_budget_range: str,
    cnt_division: int,
    cnt_item_code: int,
    inflation_factor: float,
    acf: float,
    area_type: Optional[str] = None,
) -> RawProjectRecord:
    """
    Build the minimal canonical record needed to transform into
    RegressionSimpleInput.
    """
    return RawProjectRecord(
        project_id="inference_request",
        project_description=project_description,
        project_city=city,
        project_state=state,
        county_name=county_name,
        project_type=project_type,
        project_category=project_category,
        ciqs_complexity_category=ciqs_complexity_category,
        official_budget_range=official_budget_range,
        cnt_division=cnt_division,
        cnt_item_code=cnt_item_code,
        area_type=_infer_area_type(county_name, area_type),
        inflation_factor=inflation_factor,
        acf=acf,
    )


def get_project_estimate(
    *,
    project_description: str,
    city: str,
    state: str,
    county_name: str = "",
    project_type: str = "unknown",
    project_category: str = "unknown",
    ciqs_complexity_category: str = "Category 1",
    official_budget_range: str = "$0-$1M",
    cnt_division: int = 0,
    cnt_item_code: int = 0,
    inflation_factor: float = 1.0,
    area_type: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> dict:
    """
    Generate a full project estimate by combining the parametric model
    with the Area Cost Factor engine.

    Returns:
        {
            "baseline_cost": ...,
            "baseline_cost_low": ...,
            "baseline_cost_high": ...,
            "acf": ...,
            "adjusted_cost": ...,
            "adjusted_cost_low": ...,
            "adjusted_cost_high": ...,
            "confidence": ...,
            "location": "...",
            "parametric_model_version": "...",
            "acf_model_version": "...",
            "acf_method": "...",
        }
    """
    try:
        acf_result = get_acf_factor(city=city, state=state, lat=lat, lon=lon)
        acf = float(acf_result.get("acf", 1.0))
        acf_confidence = float(acf_result.get("confidence", 0.0))
    except Exception as e:
        logger.exception("ACF lookup failed")
        return {
            "error": "acf_lookup_failed",
            "details": str(e),
            "location": f"{city}, {state}",
        }

    try:
        region = _infer_region(state)

        raw_record = _build_raw_record(
            project_description=project_description,
            city=city,
            state=state,
            county_name=county_name,
            project_type=project_type,
            project_category=project_category,
            ciqs_complexity_category=ciqs_complexity_category,
            official_budget_range=official_budget_range,
            cnt_division=cnt_division,
            cnt_item_code=cnt_item_code,
            inflation_factor=inflation_factor,
            acf=acf,
            area_type=area_type,
        )

        simple_input = raw_to_regression_simple(
            raw_record,
            inflation_factor=inflation_factor,
            acf=acf,
            region=region,
        )

        engine = _load_parametric_engine()
        baseline = engine.predict_simple(simple_input)

    except Exception as e:
        logger.exception("Parametric prediction failed")
        return {
            "error": "parametric_prediction_failed",
            "details": str(e),
            "location": f"{city}, {state}",
        }

    return {
        "baseline_cost": round(baseline.cost_estimate, 2),
        "baseline_cost_low": round(baseline.cost_low, 2),
        "baseline_cost_high": round(baseline.cost_high, 2),
        "acf": round(acf, 4),
        "adjusted_cost": round(baseline.cost_estimate, 2),
        "adjusted_cost_low": round(baseline.cost_low, 2),
        "adjusted_cost_high": round(baseline.cost_high, 2),
        "confidence": round((baseline.confidence_level + acf_confidence) / 2, 3),
        "location": f"{city}, {state}",
        "target_location": acf_result.get("target_location", f"{city}, {state}"),
        "acf_method": acf_result.get("method", ""),
        "parametric_model_version": baseline.model_version,
        "acf_model_version": acf_result.get("model_version", ""),
        "similar_projects_count": baseline.similar_projects_count,
    }
