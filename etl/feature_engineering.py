"""
Feature Engineering — derived features needed by the parametric and ACF models.

These functions extract the feature-engineering logic that was originally
embedded in the research notebooks (``final_model_cmay.ipynb``,
``ACF_inflation_adjusted.ipynb``) into reusable, testable functions.

Feature progression (from the notebook experiments):
    1. Unit dynamics       → median_cost_per_unit, median_quantity_most_common_unit
    2. Text embeddings     → handled inside the sklearn pipeline (TF-IDF)
    3. Inflation / PPI     → compute_inflation_factor, adjust_cost_by_ppi
    4. ACF normalization   → compute_acf_state_norm
    5. Scope clusters      → compute_scope_clusters (biggest improvement after text)
    6. Geographic clusters  → compute_geo_clusters (Region_0 … Region_N)
    7. Quantity binning     → compute_quantity_bins

Usage:
    from etl.feature_engineering import (
        compute_inflation_factor,
        compute_acf_state_norm,
        compute_scope_clusters,
        compute_geo_clusters,
        compute_quantity_bins,
    )
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from schema import RawProjectRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default PPI reference table (WPUIP2300001 — BLS construction materials)
# Source: FRED series PCU236400236400 / BLS WPU data
# Keys are years; values are the annual average PPI index.
# This is a representative subset — extend as needed.
# ---------------------------------------------------------------------------

DEFAULT_PPI_TABLE: dict[int, float] = {
    1993: 123.3,
    1994: 126.6,
    1995: 129.5,
    1996: 131.1,
    1997: 131.7,
    1998: 131.0,
    1999: 133.3,
    2000: 136.6,
    2001: 137.3,
    2002: 137.0,
    2003: 140.5,
    2004: 152.0,
    2005: 163.5,
    2006: 174.2,
    2007: 178.2,
    2008: 191.3,
    2009: 176.7,
    2010: 180.5,
    2011: 192.4,
    2012: 194.2,
    2013: 196.0,
    2014: 199.4,
    2015: 192.5,
    2016: 187.0,
    2017: 195.3,
    2018: 208.2,
    2019: 207.2,
    2020: 205.0,
    2021: 248.1,
    2022: 278.6,
    2023: 272.3,
    2024: 275.0,
    2025: 278.0,
}

# Default CPI table (CPI-U, all items, annual average)
DEFAULT_CPI_TABLE: dict[int, float] = {
    1993: 144.5,
    1994: 148.2,
    1995: 152.4,
    1996: 156.9,
    1997: 160.5,
    1998: 163.0,
    1999: 166.6,
    2000: 172.2,
    2001: 177.1,
    2002: 179.9,
    2003: 184.0,
    2004: 188.9,
    2005: 195.3,
    2006: 201.6,
    2007: 207.3,
    2008: 215.3,
    2009: 214.5,
    2010: 218.1,
    2011: 224.9,
    2012: 229.6,
    2013: 233.0,
    2014: 236.7,
    2015: 237.0,
    2016: 240.0,
    2017: 245.1,
    2018: 251.1,
    2019: 255.7,
    2020: 258.8,
    2021: 270.9,
    2022: 292.7,
    2023: 304.7,
    2024: 314.2,
    2025: 320.0,
}

# Quantity bin edges and labels (from final_model_cmay.ipynb)
QUANTITY_BIN_EDGES = [0, 10, 100, 1_000, 10_000, 100_000, float("inf")]
QUANTITY_BIN_LABELS = [
    "0-10",
    "10-100",
    "100-1,000",
    "1,000-10,000",
    "10,000-100,000",
    "100,000+",
]


# ---------------------------------------------------------------------------
# 1. Inflation / PPI adjustment
# ---------------------------------------------------------------------------


def compute_inflation_factor(
    project_year: int,
    base_year: int = 2025,
    cpi_table: Optional[dict[int, float]] = None,
) -> float:
    """
    Compute the CPI-based inflation factor to adjust a project-year cost
    to a common base year.

    ``inflation_factor = CPI(base_year) / CPI(project_year)``

    A value > 1.0 means base_year is more expensive than project_year
    (i.e. costs need to be inflated upward).

    Args:
        project_year: The year the project was bid/completed.
        base_year: The target year to normalize to (default 2025).
        cpi_table: Dict mapping year → annual average CPI.
            Falls back to DEFAULT_CPI_TABLE if not provided.

    Returns:
        Inflation multiplier (float).  Returns 1.0 if either year is
        missing from the table.
    """
    table = cpi_table or DEFAULT_CPI_TABLE

    cpi_base = table.get(base_year)
    cpi_project = table.get(project_year)

    if cpi_base is None or cpi_project is None:
        logger.warning(
            "CPI not found for year(s) base=%d and/or project=%d — returning 1.0",
            base_year,
            project_year,
        )
        return 1.0

    if cpi_project == 0:
        return 1.0

    return cpi_base / cpi_project


def adjust_cost_by_ppi(
    cost: float,
    project_year: int,
    base_year: int = 2025,
    ppi_table: Optional[dict[int, float]] = None,
) -> float:
    """
    Adjust a historical cost to a common base year using PPI.

    ``adjusted = cost × (PPI(base_year) / PPI(project_year))``

    This removes the effect of material-price inflation so that
    projects from different years can be compared on a level playing field.

    Args:
        cost: The original cost in USD.
        project_year: The year the cost was incurred.
        base_year: The target year to normalize to (default 2025).
        ppi_table: Dict mapping year → PPI index value.

    Returns:
        PPI-adjusted cost in USD.  Returns the original cost unchanged
        if either year is missing from the table.
    """
    table = ppi_table or DEFAULT_PPI_TABLE

    ppi_base = table.get(base_year)
    ppi_project = table.get(project_year)

    if ppi_base is None or ppi_project is None:
        logger.warning(
            "PPI not found for year(s) base=%d and/or project=%d — returning original cost",
            base_year,
            project_year,
        )
        return cost

    if ppi_project == 0:
        return cost

    return cost * (ppi_base / ppi_project)


# ---------------------------------------------------------------------------
# 2. ACF state normalization
# ---------------------------------------------------------------------------


def compute_acf_state_norm(
    df: pd.DataFrame,
    acf_col: str = "acf",
    state_col: str = "project_state",
) -> pd.Series:
    """
    Normalize ACF values so the average across all states equals 1.0.

    For each row: ``acf_state_norm = state_mean_acf / overall_mean_acf``

    This removes absolute-level differences and isolates the *relative*
    cost positioning of each state, which the advanced regression model
    uses as a feature.

    Args:
        df: DataFrame with at least ``acf_col`` and ``state_col``.
        acf_col: Name of the ACF column.
        state_col: Name of the state column.

    Returns:
        A Series (same index as df) with the state-normalized ACF.
    """
    if acf_col not in df.columns or state_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain columns '{acf_col}' and '{state_col}'"
        )

    state_mean = df.groupby(state_col)[acf_col].transform("mean")
    overall_mean = df[acf_col].mean()

    if overall_mean == 0 or pd.isna(overall_mean):
        logger.warning("Overall ACF mean is zero or NaN — returning 1.0 for all rows")
        return pd.Series(1.0, index=df.index)

    return state_mean / overall_mean


def build_acf_state_norm_lookup(
    df: pd.DataFrame,
    acf_col: str = "acf",
    state_col: str = "project_state",
) -> dict[str, float]:
    """
    Build a state → acf_state_norm lookup dict.

    Useful for passing to ``batch_raw_to_regression_advanced()`` in
    ``etl/transforms.py``.

    Args:
        df: DataFrame with ACF and state columns.
        acf_col: Name of the ACF column.
        state_col: Name of the state column.

    Returns:
        Dict mapping state code → normalized ACF (float).
    """
    if acf_col not in df.columns or state_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain columns '{acf_col}' and '{state_col}'"
        )

    state_means = df.groupby(state_col)[acf_col].mean()
    overall_mean = df[acf_col].mean()

    if overall_mean == 0 or pd.isna(overall_mean):
        return {s: 1.0 for s in state_means.index}

    return (state_means / overall_mean).to_dict()


# ---------------------------------------------------------------------------
# 3. Quantity binning
# ---------------------------------------------------------------------------


def compute_quantity_bins(
    quantities: pd.Series,
    bins: Optional[list[float]] = None,
    labels: Optional[list[str]] = None,
) -> pd.Series:
    """
    Bin median quantities into categorical buckets.

    The regression model uses these bins as a categorical feature instead
    of the raw numeric value, which reduces sensitivity to outliers and
    lets the model learn non-linear cost steps.

    Default bins: [0, 10, 100, 1k, 10k, 100k, ∞]

    Args:
        quantities: Series of numeric quantities (e.g. median_quantity_most_common_unit).
        bins: Custom bin edges. Defaults to QUANTITY_BIN_EDGES.
        labels: Custom bin labels. Defaults to QUANTITY_BIN_LABELS.

    Returns:
        Categorical Series with string bin labels.  NaN inputs produce NaN.
    """
    bin_edges = bins or QUANTITY_BIN_EDGES
    bin_labels = labels or QUANTITY_BIN_LABELS

    if len(bin_edges) != len(bin_labels) + 1:
        raise ValueError(
            f"len(bins) must equal len(labels) + 1, "
            f"got {len(bin_edges)} edges and {len(bin_labels)} labels"
        )

    return pd.cut(quantities, bins=bin_edges, labels=bin_labels, right=True)


# ---------------------------------------------------------------------------
# 4. Scope clustering
# ---------------------------------------------------------------------------


def compute_scope_clusters(
    df: pd.DataFrame,
    cost_col: str = "total_mat_lab_equip",
    quantity_col: str = "median_quantity_most_common_unit",
    n_clusters: int = 15,
    random_state: int = 42,
) -> tuple[pd.Series, KMeans]:
    """
    KMeans clustering on log1p(cost_per_quantity) to create scope clusters.

    Scope clusters embed semantic patterns by grouping projects with similar
    cost-per-quantity ratios.  This was the biggest single improvement
    after text features in the notebook experiments (R² jumped to ~0.93).

    Args:
        df: DataFrame with cost and quantity columns.
        cost_col: Column containing total cost (e.g. total_mat_lab_equip).
        quantity_col: Column containing quantity (e.g. median_quantity_most_common_unit).
        n_clusters: Number of KMeans clusters (default 15, matching notebook).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (cluster_labels Series, fitted KMeans model).
        The KMeans model is returned so it can be reused at inference time.
    """
    quantity_safe = df[quantity_col].fillna(0).replace(0, 1e-6)
    cost_safe = df[cost_col].fillna(0)

    cost_per_qty = cost_safe / quantity_safe
    features = np.log1p(cost_per_qty.values).reshape(-1, 1)

    # Replace any remaining inf/nan with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(features)

    return pd.Series(labels, index=df.index, name="scope_cluster"), km


def assign_scope_cluster(
    cost: float,
    quantity: float,
    km_model: KMeans,
) -> int:
    """
    Assign a single project to a scope cluster using a pre-fitted KMeans model.

    Args:
        cost: Project total cost (total_mat_lab_equip).
        quantity: Median quantity of most common unit.
        km_model: A fitted KMeans model from ``compute_scope_clusters()``.

    Returns:
        Cluster label (int).
    """
    if quantity == 0 or quantity is None:
        quantity = 1e-6
    cost_per_qty = cost / quantity
    feature = np.log1p(cost_per_qty).reshape(1, -1)
    feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
    return int(km_model.predict(feature)[0])


# ---------------------------------------------------------------------------
# 5. Geographic clustering
# ---------------------------------------------------------------------------


def compute_geo_clusters(
    lat: pd.Series,
    lon: pd.Series,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[pd.Series, KMeans]:
    """
    KMeans clustering on latitude/longitude to create geographic regions.

    The regression model uses these as Region_0 … Region_N categorical
    features to capture broad geographic cost patterns beyond state boundaries.

    Args:
        lat: Series of latitudes.
        lon: Series of longitudes.
        n_clusters: Number of geographic clusters (default 4).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (region_labels Series, fitted KMeans model).
        Labels are formatted as "Region_0", "Region_1", etc.
    """
    coords = pd.DataFrame({"lat": lat, "lon": lon}).copy()

    # Fill missing coordinates with medians
    coords["lat"] = coords["lat"].fillna(coords["lat"].median())
    coords["lon"] = coords["lon"].fillna(coords["lon"].median())

    features = coords[["lat", "lon"]].values

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(features)

    region_labels = pd.Series(
        [f"Region_{lbl}" for lbl in labels],
        index=lat.index,
        name="region",
    )
    return region_labels, km


def assign_geo_cluster(
    lat: float,
    lon: float,
    km_model: KMeans,
) -> str:
    """
    Assign a single project to a geographic region using a pre-fitted KMeans.

    Args:
        lat: Project latitude.
        lon: Project longitude.
        km_model: A fitted KMeans model from ``compute_geo_clusters()``.

    Returns:
        Region label string (e.g. "Region_2").
    """
    feature = np.array([[lat, lon]])
    label = int(km_model.predict(feature)[0])
    return f"Region_{label}"


# ---------------------------------------------------------------------------
# 6. Nearest-neighbor ACF assignment
# ---------------------------------------------------------------------------


def assign_nearest_acf(
    project_coords: np.ndarray,
    reference_coords: np.ndarray,
    reference_acf_values: np.ndarray,
    n_neighbors: int = 1,
) -> np.ndarray:
    """
    Assign each project the ACF of its nearest reference city.

    Uses scikit-learn's NearestNeighbors with a KD-tree for efficient
    spatial lookup.

    Args:
        project_coords: Array of shape (n_projects, 2) with [lat, lon].
        reference_coords: Array of shape (n_references, 2) with [lat, lon].
        reference_acf_values: Array of shape (n_references,) with ACF values.
        n_neighbors: Number of neighbors to average (default 1 = closest only).

    Returns:
        Array of shape (n_projects,) with the assigned ACF values.
        If n_neighbors > 1, returns the mean ACF of the k nearest references.
    """
    if len(reference_coords) == 0:
        raise ValueError("reference_coords must not be empty")
    if len(reference_coords) != len(reference_acf_values):
        raise ValueError("reference_coords and reference_acf_values must have same length")

    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree")
    nn.fit(reference_coords)
    _, indices = nn.kneighbors(project_coords)

    if n_neighbors == 1:
        return reference_acf_values[indices.ravel()]

    # Average ACF of k nearest neighbors
    return np.mean(reference_acf_values[indices], axis=1)


# ---------------------------------------------------------------------------
# 7. Stratified sampling (for balanced training sets)
# ---------------------------------------------------------------------------


def stratified_sample(
    df: pd.DataFrame,
    strata_cols: list[str],
    max_per_stratum: int = 50,
    min_frequency: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Stratified downsampling that preserves rare categories.

    From the notebook: "Rare categories are fully preserved.  Common
    categories are downsampled using caps per stratum."  This reduces
    training volume while maintaining balanced representation.

    Args:
        df: Input DataFrame.
        strata_cols: Columns to stratify on (e.g. ["region", "quantity_bin"]).
        max_per_stratum: Maximum samples per stratum.  Strata with fewer
            samples are kept entirely.
        min_frequency: Category levels with fewer than this many rows are
            always fully kept (never downsampled).
        random_state: Random seed for reproducibility.

    Returns:
        Downsampled DataFrame.
    """
    rng = np.random.RandomState(random_state)

    # Identify rare rows — any row where at least one stratum column
    # has a level with fewer than min_frequency occurrences.
    rare_mask = pd.Series(False, index=df.index)
    for col in strata_cols:
        counts = df[col].value_counts()
        rare_levels = counts[counts < min_frequency].index
        rare_mask = rare_mask | df[col].isin(rare_levels)

    rare_rows = df[rare_mask]
    common_rows = df[~rare_mask]

    # Downsample common rows per stratum
    sampled_parts = [rare_rows]

    valid_strata_cols = [c for c in strata_cols if c in common_rows.columns]
    if valid_strata_cols and len(common_rows) > 0:
        for _, group in common_rows.groupby(valid_strata_cols, observed=True):
            if len(group) <= max_per_stratum:
                sampled_parts.append(group)
            else:
                idx = rng.choice(group.index, size=max_per_stratum, replace=False)
                sampled_parts.append(df.loc[idx])
    else:
        sampled_parts.append(common_rows)

    result = pd.concat(sampled_parts, ignore_index=False).drop_duplicates()
    logger.info(
        "Stratified sample: %d → %d rows (%.1f%% retained)",
        len(df),
        len(result),
        100 * len(result) / max(len(df), 1),
    )
    return result


# ---------------------------------------------------------------------------
# 8. Convenience: engineer all features for a DataFrame
# ---------------------------------------------------------------------------


def engineer_advanced_features(
    df: pd.DataFrame,
    acf_col: str = "acf",
    state_col: str = "project_state",
    cost_col: str = "total_mat_lab_equip",
    quantity_col: str = "median_quantity_most_common_unit",
    lat_col: str = "project_latitude",
    lon_col: str = "project_longitude",
    year_col: str = "project_year",
    base_year: int = 2025,
    n_scope_clusters: int = 15,
    n_geo_clusters: int = 4,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply all feature-engineering steps to a project DataFrame.

    Adds the following columns:
      - ``inflation_factor``   (CPI ratio)
      - ``acf_state_norm``     (state-normalized ACF)
      - ``quantity_bin``        (binned median quantity)
      - ``scope_cluster``       (KMeans on cost/quantity)
      - ``region``              (KMeans on lat/lon)
      - ``ppi_adjusted_cost``   (PPI-deflated total_mat_lab_equip)

    Args:
        df: DataFrame with project-level fields (already loaded via loaders.py).
        acf_col: ACF column name.
        state_col: State column name.
        cost_col: Total cost column name.
        quantity_col: Quantity column name.
        lat_col: Latitude column name.
        lon_col: Longitude column name.
        year_col: Year column name.
        base_year: Target year for inflation/PPI adjustment.
        n_scope_clusters: Number of scope clusters.
        n_geo_clusters: Number of geographic clusters.

    Returns:
        Tuple of (enriched DataFrame, artifacts dict).
        The artifacts dict contains the fitted KMeans models for reuse
        at inference time:
          - "scope_km": fitted KMeans for scope clusters
          - "geo_km": fitted KMeans for geographic clusters
          - "acf_state_norm_lookup": dict[str, float]
    """
    result = df.copy()
    artifacts = {}

    # 1. Inflation factor
    if year_col in result.columns:
        result["inflation_factor"] = result[year_col].apply(
            lambda y: compute_inflation_factor(int(y), base_year)
            if pd.notna(y)
            else 1.0
        )
    else:
        result["inflation_factor"] = 1.0

    # 2. PPI-adjusted cost
    if cost_col in result.columns and year_col in result.columns:
        result["ppi_adjusted_cost"] = result.apply(
            lambda row: adjust_cost_by_ppi(
                row[cost_col], int(row[year_col]), base_year
            )
            if pd.notna(row.get(cost_col)) and pd.notna(row.get(year_col))
            else row.get(cost_col),
            axis=1,
        )
    elif cost_col in result.columns:
        result["ppi_adjusted_cost"] = result[cost_col]

    # 3. ACF state normalization
    if acf_col in result.columns and state_col in result.columns:
        result["acf_state_norm"] = compute_acf_state_norm(
            result, acf_col=acf_col, state_col=state_col
        )
        artifacts["acf_state_norm_lookup"] = build_acf_state_norm_lookup(
            result, acf_col=acf_col, state_col=state_col
        )

    # 4. Quantity binning
    if quantity_col in result.columns:
        result["quantity_bin"] = compute_quantity_bins(
            result[quantity_col].fillna(0)
        )
    else:
        result["quantity_bin"] = "0-10"

    # 5. Scope clusters
    if cost_col in result.columns and quantity_col in result.columns:
        result["scope_cluster"], scope_km = compute_scope_clusters(
            result,
            cost_col=cost_col,
            quantity_col=quantity_col,
            n_clusters=n_scope_clusters,
        )
        artifacts["scope_km"] = scope_km

    # 6. Geographic clusters
    if lat_col in result.columns and lon_col in result.columns:
        result["region"], geo_km = compute_geo_clusters(
            result[lat_col],
            result[lon_col],
            n_clusters=n_geo_clusters,
        )
        artifacts["geo_km"] = geo_km

    logger.info(
        "Feature engineering complete — added columns: %s",
        [c for c in result.columns if c not in df.columns],
    )
    return result, artifacts
