"""
tools/acf_tool.py — Foundry tool schema for get_acf_factor

This is the function the LLM calls as a tool.
Input:  city (str), state (str), lat (float), lon (float)
Output: dict with acf, confidence, method, model_version
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Path where the trained engine was saved
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "saved" / "acf"


def get_acf_factor(
    city: str,
    state: str,
    lat: float = None,
    lon: float = None,
) -> dict:
    """
    Return the Area Cost Factor (ACF) for a given US location.

    The LLM calls this tool when it needs to adjust a baseline
    construction cost estimate for a specific location.

    Args:
        city:  City name (e.g. "Houston")
        state: 2-letter state code (e.g. "TX")
        lat:   Latitude (optional — geocoded from city/state if omitted)
        lon:   Longitude (optional)

    Returns:
        {
            "acf":           1.12,           # multiply baseline cost by this
            "confidence":    0.85,           # 0-1, higher = more reliable
            "target_location": "Houston, TX",
            "method":        "inflation_cluster_knn",
            "model_version": "acf_inflation_v1"
        }
    """
    from models.acf import ACFEngine
    from etl.feature_engineering import assign_nearest_acf
    import numpy as np

    # Load pre-trained engine
    try:
        engine = ACFEngine.load(str(_DEFAULT_MODEL_DIR))
    except Exception as e:
        logger.error("Failed to load ACFEngine: %s", e)
        return {
            "acf": 1.0,
            "confidence": 0.0,
            "target_location": f"{city}, {state}",
            "method": "fallback_national_average",
            "model_version": "none",
            "error": str(e),
        }

    # If lat/lon not provided, use a simple state-centroid lookup
    if lat is None or lon is None:
        lat, lon = _state_centroid(state)

    result = engine.predict(lat=lat, lon=lon, state=state, city=city)

    return {
        "acf":              result.location_factor,
        "confidence":       result.confidence,
        "target_location":  result.target_location,
        "method":           result.method,
        "model_version":    result.model_version,
    }


# Rough state centroids for when lat/lon is not provided
_STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "AL": (32.8, -86.8), "AK": (64.2, -153.4), "AZ": (34.3, -111.1),
    "AR": (34.8, -92.2), "CA": (36.8, -119.4), "CO": (39.0, -105.5),
    "CT": (41.6, -72.7), "DE": (39.0, -75.5),  "FL": (27.8, -81.7),
    "GA": (32.2, -83.4), "HI": (20.8, -156.3), "ID": (44.4, -114.6),
    "IL": (40.0, -89.2), "IN": (39.8, -86.1),  "IA": (42.0, -93.2),
    "KS": (38.5, -98.4), "KY": (37.6, -85.3),  "LA": (31.2, -91.8),
    "ME": (45.4, -69.0), "MD": (39.0, -76.8),  "MA": (42.3, -71.8),
    "MI": (44.3, -85.4), "MN": (46.4, -93.1),  "MS": (32.7, -89.7),
    "MO": (38.5, -92.5), "MT": (47.0, -110.4), "NE": (41.5, -99.9),
    "NV": (39.3, -116.6),"NH": (43.7, -71.6),  "NJ": (40.1, -74.5),
    "NM": (34.8, -106.2),"NY": (42.2, -74.9),  "NC": (35.6, -79.8),
    "ND": (47.5, -100.5),"OH": (40.4, -82.8),  "OK": (35.6, -96.9),
    "OR": (44.6, -122.1),"PA": (40.6, -77.2),  "RI": (41.7, -71.5),
    "SC": (33.9, -80.9), "SD": (44.4, -100.2), "TN": (35.9, -86.7),
    "TX": (31.5, -99.3), "UT": (39.3, -111.1), "VT": (44.1, -72.7),
    "VA": (37.8, -78.2), "WA": (47.4, -120.4), "WV": (38.6, -80.6),
    "WI": (44.3, -89.6), "WY": (43.0, -107.6), "DC": (38.9, -77.0),
}


def _state_centroid(state: str) -> tuple[float, float]:
    """Return a rough centroid lat/lon for a US state code."""
    return _STATE_CENTROIDS.get(state.upper(), (39.5, -98.4))