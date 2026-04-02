"""
models/acf.py — ACFEngine

Productionizes the ACF logic from:
  Previous work/ACF/inflation_ACF/ACF_inflation_adjusted.ipynb

Best-performing variant (lowest mean_pct_diff vs DoD benchmark):
  cluster_acf_mean  →  10.76% error  (HDBSCAN clusters, mean cost_per_sqft)

Formula:
  ACF = cluster_mean_cost_per_sqft / national_mean_cost_per_sqft
  ACF_labor_adj = ACF x (0.63 x Hour_mean_norm + 0.35 + 0.02)

At inference:
  1. Load the trained cluster lookup table (cluster_id → ACF value)
  2. Find the nearest cluster centroid to the query lat/lon
  3. Return that cluster's ACF as the LocationFactor
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from schema import LocationFactor

logger = logging.getLogger(__name__)

_VERSION = "cluser_acf_mean_v1"


class ACFEngine:
    """
    Area Cost Factor engine — productionizes ACF_inflation_adjusted.ipynb.

    Trains on projects_clusters_log_outliers.csv.
    Best variant: cluster_acf_mean (HDBSCAN + mean cost_per_sqft, 10.76% vs DoD).
    Optional labor adjustment: ACF x (0.63 x Hour_mean_norm + 0.35 + 0.02)

    Usage:
        engine = ACFEngine()
        engine.train("path/to/projects_clusters_log_outliers.csv")
        factor = engine.predict(lat=29.76, lon=-95.37, state="TX", city="Houston")
        print(factor.location_factor)   # e.g. 0.82
        engine.save("models/saved/acf/")

        engine2 = ACFEngine.load("models/saved/acf/")
    """

    _NN_FILE   = "acf_nn_model.joblib"
    _DATA_FILE = "acf_reference_data.joblib"
    _META_FILE = "acf_metadata.joblib"

    def __init__(self) -> None:
        self.is_trained: bool = False
        self._nn_model: Optional[NearestNeighbors] = None
        # Each row: [lat, lon, cluster_acf_mean, cluster_acf_labor_adj_mean, dod_acf]
        self._reference_data: Optional[pd.DataFrame] = None
        self._national_mean: float = 1.0
        self._model_version: str = _VERSION
        self._training_stats: dict = {}

    # Training

    def train(
        self,
        csv_path: str,
        use_labor_adjustment: bool = True,
    ) -> dict:
        """
        Train from the CSV. Computes ACF using HDBSCAN cluster means,
        same logic as ACF_inflation_adjusted.ipynb.

        Reads projects_clusters_log_outliers.csv directly (same as notebook).
        Computes cluster_acf_mean and cluster_acf_labor_adj_mean per the notebook formula.
        Builds a KD-tree over cluster centroids for fast inference.

        Args:
            csv_path: Path to projects_clusters_log_outliers.csv
            use_labor_adjustment: If True (default), use labor-adjusted ACF
                                  (ACF × (0.63×Hour_mean_norm + 0.35 + 0.02))

        Returns:
            dict with training stats (n_clusters, national_mean, best_acf_col, etc.)
        """
        logger.info("Loading ACF training data from %s", csv_path)
        gdf = pd.read_csv(csv_path)
        logger.info("  Loaded %d rows, columns: %s", len(gdf), list(gdf.columns))

        # Replicate notebook ACF computation

        national_mean = gdf["cost_per_sqft"].mean()
        self._national_mean = national_mean

        # HDBSCAN cluster mean cost_per_sqft
        cluster_means = (
            gdf.groupby("hdb_cluster_41km_3ms")["cost_per_sqft"]
            .mean()
            .reset_index()
        )
        cluster_acf = cluster_means.set_index("hdb_cluster_41km_3ms")["cost_per_sqft"]

        # ACF = cluster mean / national mean
        gdf["cluster_acf_mean"] = gdf["hdb_cluster_41km_3ms"].map(cluster_acf) / national_mean

        # Labor adjustment: ACF × (0.63 × Hour_mean_norm + 0.35 + 0.02)
        gdf["cluster_acf_labor_adj_mean"] = (
            gdf["cluster_acf_mean"]
            * (0.63 * gdf["Hour_mean_norm"] + 0.35 + 0.02)
        )

        # Choose which ACF column to use for inference
        acf_col = "cluster_acf_labor_adj_mean" if use_labor_adjustment else "cluster_acf_mean"
        logger.info("  Using ACF column: %s", acf_col)

        # Build reference lookup: one row per unique city
        # Drop rows with missing lat/lon or ACF
        valid = gdf.dropna(
            subset=["project_latitude", "project_longitude", acf_col]
        ).copy()

        # Aggregate to city level (median lat/lon, mean ACF)
        city_lookup = (
            valid.groupby(["project_state", "city"])
            .agg(
                lat=(       "project_latitude",  "median"),
                lon=(       "project_longitude", "median"),
                acf=(        acf_col,             "mean"),
                acf_raw=(    "cluster_acf_mean",      "mean"),
                dod_acf=(    "DoD_ACF2024",        "first"),
                hour_norm=( "Hour_mean_norm",     "mean"),
            )
            .reset_index()
        )
        city_lookup = city_lookup.dropna(subset=["lat", "lon", "acf"])

        self._reference_data = city_lookup
        self._acf_col_used   = acf_col

        # Fit KD-tree on city centroids 
        coords = city_lookup[["lat", "lon"]].values
        self._nn_model = NearestNeighbors(n_neighbors=5, algorithm="kd_tree")
        self._nn_model.fit(coords)
        self.is_trained = True

        # Compute benchmark comparison
        state_acf = (
            valid.groupby("project_state")
            .agg(
                dod_acf=("DoD_ACF2024",         "first"),
                our_acf=(acf_col,                "mean"),
            )
            .dropna()
        )
        abs_diff = (state_acf["our_acf"] - state_acf["dod_acf"]).abs().mean()
        pct_diff = (
            (state_acf["our_acf"] - state_acf["dod_acf"]).abs()
            / state_acf["dod_acf"]
        ).mean() * 100

        self._training_stats = {
            "n_cities":           len(city_lookup),
            "n_states":           city_lookup["project_state"].nunique(),
            "national_mean_cost": round(national_mean, 2),
            "acf_col_used":       acf_col,
            "mean_abs_diff_vs_dod": round(abs_diff, 4),
            "mean_pct_diff_vs_dod": round(pct_diff, 2),
            "model_version":      self._model_version,
        }

        logger.info(
            "  Trained on %d cities | %.1f%% error vs DoD benchmark",
            len(city_lookup), pct_diff
        )
        return self._training_stats

    #  Prediction

    def predict(
        self,
        lat: float,
        lon: float,
        state: str = "",
        city: str = "",
        n_neighbors: int = 3,
    ) -> LocationFactor:
        """
        Return the ACF for a given lat/lon using nearest reference cities.

        Args:
            lat:         Project latitude.
            lon:         Project longitude.
            state:       2-letter state code (for output label only).
            city:        City name (for output label only).
            n_neighbors: Number of nearby cities to average (default 3).

        Returns:
            LocationFactor with location_factor = ACF multiplier.
            Values below 1.0  → cheaper than national average.
            Values above 1.0  → more expensive than national average.
        """
        if not self.is_trained:
            raise RuntimeError(
                "ACFEngine is not trained. Call train() or load() first."
            )

        query = np.array([[lat, lon]])
        k = min(n_neighbors, len(self._reference_data))
        distances, indices = self._nn_model.kneighbors(query, n_neighbors=k)

        distances = distances[0]
        indices   = indices[0]

        weights  = 1.0 / (distances + 1e-6)
        weights /= weights.sum()

        acf_values = self._reference_data.iloc[indices]["acf"].values
        acf_value  = float(np.dot(weights, acf_values))

        # Confidence: based on distance to nearest city
        nearest_km  = distances[0] * 111
        confidence  = float(np.clip(1.0 - nearest_km / 500.0, 0.15, 1.0))

        nearest_row  = self._reference_data.iloc[indices[0]]
        nearest_city = f"{nearest_row['city']}, {nearest_row['project_state']}"

        return LocationFactor(
            location_factor=round(acf_value, 4),
            confidence=round(confidence, 3),
            base_location="national_mean_cost_per_sqft",
            target_location=f"{city}, {state}".strip(", ") or nearest_city,
            method=self._acf_col_used,
            model_version=self._model_version,
        )

    def predict_batch(
        self,
        lat_lon_pairs: list[tuple[float, float]],
        states: list[str] = None,
        cities: list[str] = None,
    ) -> list[LocationFactor]:
        """
        Batch prediction for a list of (lat, lon) tuples.

        Args:
            lat_lon_pairs: List of (lat, lon) tuples.
            states:        Optional list of state codes (same length).
            cities:        Optional list of city names (same length).

        Returns:
            List of LocationFactor instances.
        """
        states = states or [""] * len(lat_lon_pairs)
        cities = cities or [""] * len(lat_lon_pairs)
        results = []
        for (lat, lon), state, city in zip(lat_lon_pairs, states, cities):
            try:
                results.append(self.predict(lat, lon, state, city))
            except Exception as e:
                logger.warning("Prediction failed for (%s, %s): %s", lat, lon, e)
                results.append(LocationFactor(
                    location_factor=1.0,
                    confidence=0.0,
                    method="fallback_national_mean",
                    model_version=self._model_version,
                ))
        return results

    def get_state_acf_table(self) -> pd.DataFrame:
        """
        Return a state-level ACF summary table — mirrors notebook output.

        Useful for validation against DoD_ACF2024 benchmark.

        Returns:
            DataFrame with columns: project_state, our_acf, dod_acf, pct_diff
        """
        if not self.is_trained:
            raise RuntimeError("Not trained yet.")

        state_table = (
            self._reference_data
            .groupby("project_state")
            .agg(
                our_acf=("acf",     "mean"),
                dod_acf=("dod_acf", "first"),
            )
            .reset_index()
            .dropna()
        )
        state_table["pct_diff"] = (
            (state_table["our_acf"] - state_table["dod_acf"]).abs()
            / state_table["dod_acf"] * 100
        ).round(2)
        return state_table.sort_values("project_state")

    # Save/Load

    def save(self, directory: str) -> None:
        """Save trained engine to directory (3 joblib files)."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained engine.")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._nn_model, path / self._NN_FILE)
        joblib.dump(
            {
                "reference_data":  self._reference_data,
                "national_mean":   self._national_mean,
                "acf_col_used":    self._acf_col_used,
            },
            path / self._DATA_FILE,
        )
        joblib.dump(
            {
                "is_trained":      self.is_trained,
                "model_version":   self._model_version,
                "training_stats":  self._training_stats,
            },
            path / self._META_FILE,
        )
        logger.info("ACFEngine saved to %s", directory)

    @classmethod
    def load(cls, directory: str) -> "ACFEngine":
        """Load a previously saved ACFEngine from directory."""
        path   = Path(directory)
        engine = cls()

        meta_path = path / cls._META_FILE
        if meta_path.exists():
            meta = joblib.load(meta_path)
            engine._model_version  = meta.get("model_version", _VERSION)
            engine._training_stats = meta.get("training_stats", {})

        data_path = path / cls._DATA_FILE
        if data_path.exists():
            data = joblib.load(data_path)
            engine._reference_data = data["reference_data"]
            engine._national_mean  = data["national_mean"]
            engine._acf_col_used   = data["acf_col_used"]

        nn_path = path / cls._NN_FILE
        if nn_path.exists():
            engine._nn_model  = joblib.load(nn_path)
            engine.is_trained = True

        return engine