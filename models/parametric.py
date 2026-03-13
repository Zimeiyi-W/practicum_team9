"""
models/parametric.py — ParametricEngine

Productionizes the Random Forest parametric cost model from:
  Previous work/RegressionModels/RegressionModelsFinal.ipynb  (simple variant)
  Previous work/RegressionModels/final_model_cmay.ipynb        (advanced variant)

Two variants
------------
Simple  (13 features):
    Numeric:     inflation_factor, acf, cnt_division, cnt_item_code
    Categorical: project_type, project_category, ciqs_complexity_category,
                 official_budget_range, project_state, county_name, area_type, region
    Pipeline:    StandardScaler (numeric) + OneHotEncoder (categorical)
                 → RandomForestRegressor(n_estimators=600, max_depth=17, min_samples_leaf=2)
    Performance: R²≈0.95, MAPE≈22% on held-out data (17,025 projects)

Advanced  (exp10 features):
    Numeric:     acf, project_year, median_cost_per_unit,
                 median_quantity_most_common_unit, acf_state_norm, scope_cluster
    Categorical: project_city, project_state, project_type, project_category,
                 construction_category, most_common_unit, quantity_bin
    Text:        project_description via TF-IDF (max_features=500)
    Pipeline:    SimpleImputer (numeric) + OneHotEncoder (categorical)
                 + TfidfVectorizer (text)
                 → RandomForestRegressor(n_estimators=100)
    Performance: R²≈0.93, MAPE≈17% on held-out data

Persistence
-----------
``save(directory)`` writes three files:
  simple_pipeline.joblib   — sklearn Pipeline for the simple variant
  advanced_pipeline.joblib — sklearn Pipeline for the advanced variant
  model_metadata.joblib    — training metrics and flags

``load(directory)`` is a classmethod that reconstructs the engine from a
previously saved directory.  Missing files are silently skipped
(untrained state for that variant).

apply() — one-step class method
--------------------------------
``ParametricEngine.apply(data, predict=None, variant="simple", ...)``
  1. Loads data from a DataFrame or CSV path.
  2. Extracts features and target.
  3. Trains the requested variant.
  4. Optionally predicts on ``predict`` (single input or list).
  5. Returns ``(predictions, engine, metrics)``.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schema import RegressionAdvancedInput, RegressionOutput, RegressionSimpleInput


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_SIMPLE_NUMERIC_COLS = ["inflation_factor", "acf", "cnt_division", "cnt_item_code"]
_SIMPLE_CAT_COLS = [
    "project_type",
    "project_category",
    "ciqs_complexity_category",
    "official_budget_range",
    "project_state",
    "county_name",
    "area_type",
    "region",
]

_ADVANCED_NUMERIC_COLS = [
    "acf",
    "project_year",
    "median_cost_per_unit",
    "median_quantity_most_common_unit",
    "acf_state_norm",
    "scope_cluster",
]
_ADVANCED_CAT_COLS = [
    "project_city",
    "project_state",
    "project_type",
    "project_category",
    "construction_category",
    "most_common_unit",
    "quantity_bin",
]
_ADVANCED_TEXT_COL = "project_description"

_SIMPLE_VERSION = "simple_rf_v1"
_ADVANCED_VERSION = "advanced_rf_exp10"

# Default target columns when using apply()
_SIMPLE_DEFAULT_TARGET = "total_mat_lab_equip"
_ADVANCED_DEFAULT_TARGET = "inf_adj_total_mat_lab_equip"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build sklearn pipelines
# ─────────────────────────────────────────────────────────────────────────────

def _build_simple_pipeline(rf_params: dict | None = None) -> Pipeline:
    """Return an unfitted simple-variant sklearn Pipeline."""
    params = {
        "n_estimators": 600,
        "max_depth": 17,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": 42,
    }
    if rf_params:
        params.update(rf_params)

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, _SIMPLE_NUMERIC_COLS),
            ("cat", categorical_transformer, _SIMPLE_CAT_COLS),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("rf", RandomForestRegressor(**params)),
    ])


def _build_advanced_pipeline(rf_params: dict | None = None) -> Pipeline:
    """Return an unfitted advanced-variant sklearn Pipeline."""
    params = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
    }
    if rf_params:
        params.update(rf_params)

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, _ADVANCED_NUMERIC_COLS),
            ("cat", categorical_transformer, _ADVANCED_CAT_COLS),
        ],
        remainder="drop",
    )

    # Advanced pipeline combines dense features with TF-IDF
    # We use a custom wrapper to combine them
    from sklearn.pipeline import FeatureUnion
    # We'll handle text separately inside the pipeline via a custom step
    return Pipeline([
        ("preprocessor", preprocessor),
        ("rf", RandomForestRegressor(**params)),
    ])


class _TextExtractor:
    """Helper that extracts a text column from a DataFrame for TfidfVectorizer."""

    def __init__(self, col: str):
        self.col = col

    def transform(self, X):
        return X[self.col].fillna("").tolist()

    def fit(self, X, y=None):
        return self


def _build_advanced_pipeline_with_text(rf_params: dict | None = None) -> "AdvancedPipelineWrapper":
    """
    Build the advanced pipeline that combines numeric/categorical features
    with a TF-IDF text feature.
    """
    params = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
    }
    if rf_params:
        params.update(rf_params)

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, _ADVANCED_NUMERIC_COLS),
            ("cat", categorical_transformer, _ADVANCED_CAT_COLS),
        ],
        remainder="drop",
    )

    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), lowercase=True)
    rf = RandomForestRegressor(**params)

    return AdvancedPipelineWrapper(col_transformer, tfidf, rf)


class AdvancedPipelineWrapper:
    """
    Custom pipeline that concatenates ColumnTransformer output with
    TF-IDF text features before feeding them to a RandomForest.
    """

    def __init__(
        self,
        col_transformer: ColumnTransformer,
        tfidf: TfidfVectorizer,
        rf: RandomForestRegressor,
    ):
        self._col = col_transformer
        self._tfidf = tfidf
        self._rf = rf

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "AdvancedPipelineWrapper":
        X_dense = self._col.fit_transform(X, y)
        texts = X[_ADVANCED_TEXT_COL].fillna("").tolist()
        X_text = self._tfidf.fit_transform(texts).toarray()
        X_full = np.hstack([X_dense, X_text])
        self._rf.fit(X_full, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_dense = self._col.transform(X)
        texts = X[_ADVANCED_TEXT_COL].fillna("").tolist()
        X_text = self._tfidf.transform(texts).toarray()
        X_full = np.hstack([X_dense, X_text])
        return self._rf.predict(X_full)

    @property
    def named_steps(self):
        """Shim so get_feature_importances() can access the RF."""
        class _Steps:
            def __init__(self_, rf):
                self_.rf = rf
        return _Steps(self._rf)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: input → DataFrame row
# ─────────────────────────────────────────────────────────────────────────────

def _simple_input_to_row(inp: RegressionSimpleInput) -> dict:
    return {
        "inflation_factor": inp.inflation_factor,
        "acf": inp.acf,
        "cnt_division": inp.cnt_division,
        "cnt_item_code": inp.cnt_item_code,
        "project_type": inp.project_type,
        "project_category": inp.project_category,
        "ciqs_complexity_category": inp.ciqs_complexity_category,
        "official_budget_range": inp.official_budget_range,
        "project_state": inp.project_state,
        "county_name": inp.county_name,
        "area_type": inp.area_type,
        "region": inp.region,
    }


def _advanced_input_to_row(inp: RegressionAdvancedInput) -> dict:
    return {
        "acf": inp.acf,
        "project_year": inp.project_year,
        "median_cost_per_unit": inp.median_cost_per_unit,
        "median_quantity_most_common_unit": inp.median_quantity_most_common_unit,
        "acf_state_norm": inp.acf_state_norm if inp.acf_state_norm is not None else 1.0,
        "scope_cluster": inp.scope_cluster if inp.scope_cluster >= 0 else 0,
        "project_city": inp.project_city or "unknown",
        "project_state": inp.project_state,
        "project_type": inp.project_type,
        "project_category": inp.project_category,
        "construction_category": inp.construction_category,
        "most_common_unit": inp.most_common_unit,
        "quantity_bin": inp.quantity_bin,
        _ADVANCED_TEXT_COL: inp.project_description or "",
    }


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_train: int, n_test: int) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = float(r2_score(y_true, y_pred))
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"r2": r2, "mape": mape, "rmse": rmse, "mae": mae, "n_train": n_train, "n_test": n_test}


# ─────────────────────────────────────────────────────────────────────────────
# ParametricEngine
# ─────────────────────────────────────────────────────────────────────────────

class ParametricEngine:
    """
    Random Forest parametric cost estimation engine.

    Supports two variants:
      * ``simple``   — 13-feature model matching the deployed Flask app.
      * ``advanced`` — 16-feature model with TF-IDF description features.

    Typical usage::

        engine = ParametricEngine()
        metrics = engine.train_simple(inputs, targets)
        output  = engine.predict_simple(new_input)
        engine.save("/models/parametric/")
        engine2 = ParametricEngine.load("/models/parametric/")
    """

    # Persistence file names
    _SIMPLE_FILE = "simple_pipeline.joblib"
    _ADVANCED_FILE = "advanced_pipeline.joblib"
    _META_FILE = "model_metadata.joblib"

    def __init__(self) -> None:
        self.simple_is_trained: bool = False
        self.advanced_is_trained: bool = False

        self._simple_pipeline: Optional[Pipeline] = None
        self._advanced_pipeline: Optional[AdvancedPipelineWrapper] = None
        self._simple_metrics: dict = {}
        self._advanced_metrics: dict = {}
        self._simple_n_train: int = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        """True if at least one variant has been trained."""
        return self.simple_is_trained or self.advanced_is_trained

    # ── Training ──────────────────────────────────────────────────────────────

    def train_simple(
        self,
        inputs: list[RegressionSimpleInput],
        targets: list[float],
        test_size: float = 0.2,
        rf_params: dict | None = None,
    ) -> dict:
        """
        Train the simple variant on a list of RegressionSimpleInput objects.

        Parameters
        ----------
        inputs  : list[RegressionSimpleInput]
        targets : list[float]  — total project cost in USD
        test_size : float      — fraction held out for evaluation
        rf_params : dict or None — override RandomForest hyperparameters

        Returns
        -------
        dict with keys: r2, mape, rmse, mae, n_train, n_test
        """
        if len(inputs) != len(targets):
            raise ValueError(
                f"inputs and targets must have the same length, "
                f"got {len(inputs)} and {len(targets)}"
            )
        if len(inputs) == 0:
            raise ValueError("inputs must be non-empty")

        df = pd.DataFrame([_simple_input_to_row(inp) for inp in inputs])
        y = np.array(targets, dtype=float)

        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=test_size, random_state=42
        )

        pipe = _build_simple_pipeline(rf_params)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        metrics = _compute_metrics(y_test, y_pred, len(X_train), len(X_test))

        self._simple_pipeline = pipe
        self._simple_metrics = metrics
        self._simple_n_train = len(X_train)
        self.simple_is_trained = True
        return metrics

    def train_advanced(
        self,
        inputs: list[RegressionAdvancedInput],
        targets: list[float],
        test_size: float = 0.2,
        rf_params: dict | None = None,
    ) -> dict:
        """
        Train the advanced variant on a list of RegressionAdvancedInput objects.

        Parameters
        ----------
        inputs  : list[RegressionAdvancedInput]
        targets : list[float]  — total project cost in USD
        test_size : float
        rf_params : dict or None

        Returns
        -------
        dict with keys: r2, mape, rmse, mae, n_train, n_test
        """
        if len(inputs) != len(targets):
            raise ValueError(
                f"inputs and targets must have the same length, "
                f"got {len(inputs)} and {len(targets)}"
            )
        if len(inputs) == 0:
            raise ValueError("inputs must be non-empty")

        df = pd.DataFrame([_advanced_input_to_row(inp) for inp in inputs])
        y = np.array(targets, dtype=float)

        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=test_size, random_state=42
        )

        pipe = _build_advanced_pipeline_with_text(rf_params)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        metrics = _compute_metrics(y_test, y_pred, len(X_train), len(X_test))

        self._advanced_pipeline = pipe
        self._advanced_metrics = metrics
        self.advanced_is_trained = True
        return metrics

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_simple(self, inp: RegressionSimpleInput) -> RegressionOutput:
        """Predict project cost for a single RegressionSimpleInput."""
        if not self.simple_is_trained:
            raise RuntimeError(
                "Simple model is not trained. Call train_simple() first."
            )
        row = pd.DataFrame([_simple_input_to_row(inp)])
        mid = float(self._simple_pipeline.predict(row)[0])
        mid = max(mid, 0.0)
        low = mid * 0.75
        high = mid * 1.25
        # Confidence: high when training data was large, diminishes for edge cases
        confidence = min(1.0, self._simple_n_train / 1000.0)
        return RegressionOutput(
            cost_estimate=mid,
            cost_low=low,
            cost_high=high,
            confidence_level=confidence,
            model_version=_SIMPLE_VERSION,
            similar_projects_count=self._simple_n_train,
        )

    def predict_advanced(self, inp: RegressionAdvancedInput) -> RegressionOutput:
        """Predict project cost for a single RegressionAdvancedInput."""
        if not self.advanced_is_trained:
            raise RuntimeError(
                "Advanced model is not trained. Call train_advanced() first."
            )
        row = pd.DataFrame([_advanced_input_to_row(inp)])
        mid = float(self._advanced_pipeline.predict(row)[0])
        mid = max(mid, 0.0)
        low = mid * 0.75
        high = mid * 1.25
        return RegressionOutput(
            cost_estimate=mid,
            cost_low=low,
            cost_high=high,
            confidence_level=0.70,
            model_version=_ADVANCED_VERSION,
            similar_projects_count=10,
        )

    def predict_simple_batch(
        self, inputs: list[RegressionSimpleInput]
    ) -> list[RegressionOutput]:
        """Batch prediction for a list of RegressionSimpleInput objects."""
        if not self.simple_is_trained:
            raise RuntimeError(
                "Simple model is not trained. Call train_simple() first."
            )
        df = pd.DataFrame([_simple_input_to_row(inp) for inp in inputs])
        preds = self._simple_pipeline.predict(df)
        confidence = min(1.0, self._simple_n_train / 1000.0)
        results = []
        for mid in preds:
            mid = max(float(mid), 0.0)
            results.append(
                RegressionOutput(
                    cost_estimate=mid,
                    cost_low=mid * 0.75,
                    cost_high=mid * 1.25,
                    confidence_level=confidence,
                    model_version=_SIMPLE_VERSION,
                    similar_projects_count=self._simple_n_train,
                )
            )
        return results

    def predict_advanced_batch(
        self, inputs: list[RegressionAdvancedInput]
    ) -> list[RegressionOutput]:
        """Batch prediction for a list of RegressionAdvancedInput objects."""
        if not self.advanced_is_trained:
            raise RuntimeError(
                "Advanced model is not trained. Call train_advanced() first."
            )
        df = pd.DataFrame([_advanced_input_to_row(inp) for inp in inputs])
        preds = self._advanced_pipeline.predict(df)
        results = []
        for mid in preds:
            mid = max(float(mid), 0.0)
            results.append(
                RegressionOutput(
                    cost_estimate=mid,
                    cost_low=mid * 0.75,
                    cost_high=mid * 1.25,
                    confidence_level=0.70,
                    model_version=_ADVANCED_VERSION,
                    similar_projects_count=10,
                )
            )
        return results

    # ── Feature importances ───────────────────────────────────────────────────

    def get_feature_importances(self, variant: str = "simple") -> dict:
        """
        Return a dict of feature_name → importance, summing to ~1.0.

        Parameters
        ----------
        variant : "simple" or "advanced"
        """
        if variant == "simple":
            if not self.simple_is_trained:
                raise RuntimeError(
                    "Simple model is not trained. Call train_simple() first."
                )
            pipe = self._simple_pipeline
            rf = pipe.named_steps["rf"]
            preprocessor = pipe.named_steps["preprocessor"]
            feature_names = preprocessor.get_feature_names_out()
            importances = rf.feature_importances_
            total = importances.sum()
            if total == 0:
                return {name: 0.0 for name in feature_names}
            return {str(name): float(imp / total) for name, imp in zip(feature_names, importances)}

        elif variant == "advanced":
            if not self.advanced_is_trained:
                raise RuntimeError(
                    "Advanced model is not trained. Call train_advanced() first."
                )
            rf = self._advanced_pipeline.named_steps.rf
            importances = rf.feature_importances_
            total = importances.sum()
            names = [f"feature_{i}" for i in range(len(importances))]
            if total == 0:
                return {name: 0.0 for name in names}
            return {name: float(imp / total) for name, imp in zip(names, importances)}

        else:
            raise ValueError(
                f"Unknown variant: {variant!r}. Choose 'simple' or 'advanced'."
            )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """
        Save trained pipelines and metadata to ``directory``.

        Creates three files:
          - ``simple_pipeline.joblib``
          - ``advanced_pipeline.joblib``
          - ``model_metadata.joblib``

        Missing (untrained) pipelines are not written.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        if self.simple_is_trained:
            joblib.dump(self._simple_pipeline, path / self._SIMPLE_FILE)

        if self.advanced_is_trained:
            joblib.dump(self._advanced_pipeline, path / self._ADVANCED_FILE)

        metadata = {
            "simple_is_trained": self.simple_is_trained,
            "advanced_is_trained": self.advanced_is_trained,
            "simple_metrics": self._simple_metrics,
            "advanced_metrics": self._advanced_metrics,
            "simple_n_train": self._simple_n_train,
        }
        joblib.dump(metadata, path / self._META_FILE)

    @classmethod
    def load(cls, directory: str) -> "ParametricEngine":
        """
        Load a ParametricEngine from a previously saved directory.

        If the directory is empty or pipelines are missing, returns an
        untrained engine (no error).

        Parameters
        ----------
        directory : str or Path
            Directory where ``save()`` wrote the files.

        Returns
        -------
        ParametricEngine
        """
        path = Path(directory)
        engine = cls()

        meta_path = path / cls._META_FILE
        if meta_path.exists():
            metadata = joblib.load(meta_path)
            engine._simple_metrics = metadata.get("simple_metrics", {})
            engine._advanced_metrics = metadata.get("advanced_metrics", {})
            engine._simple_n_train = metadata.get("simple_n_train", 0)

        simple_path = path / cls._SIMPLE_FILE
        if simple_path.exists():
            engine._simple_pipeline = joblib.load(simple_path)
            engine.simple_is_trained = True

        advanced_path = path / cls._ADVANCED_FILE
        if advanced_path.exists():
            engine._advanced_pipeline = joblib.load(advanced_path)
            engine.advanced_is_trained = True

        return engine

    # ── apply() — one-step class method ──────────────────────────────────────

    @classmethod
    def apply(
        cls,
        data: Union[pd.DataFrame, str, Path],
        predict: Union[
            None,
            RegressionSimpleInput,
            RegressionAdvancedInput,
            list[RegressionSimpleInput],
            list[RegressionAdvancedInput],
        ] = None,
        variant: str = "simple",
        target_col: Optional[str] = None,
        save_dir: Optional[str] = None,
        rf_params: Optional[dict] = None,
        test_size: float = 0.2,
    ) -> tuple:
        """
        One-step train (and optionally predict) pipeline.

        Parameters
        ----------
        data : DataFrame or str/Path (CSV file path)
            Training data.  Must contain the feature columns for the chosen
            ``variant`` and the ``target_col`` (default: ``total_mat_lab_equip``
            for simple, ``inf_adj_total_mat_lab_equip`` for advanced).
        predict : RegressionSimpleInput / RegressionAdvancedInput / list / None
            Optional input(s) to predict after training.
        variant : "simple" or "advanced"
        target_col : str or None
            Name of the target column in ``data``.
        save_dir : str or None
            If provided, saves the trained engine to this directory.
        rf_params : dict or None
            RandomForest hyperparameter overrides.
        test_size : float

        Returns
        -------
        (predictions, engine, metrics)
            predictions : RegressionOutput / list[RegressionOutput] / []
            engine      : trained ParametricEngine
            metrics     : dict with training metrics (r2, mape, ...)
        """
        if variant not in ("simple", "advanced"):
            raise ValueError(
                f"variant must be 'simple' or 'advanced', got {variant!r}"
            )

        # ── Load data ──────────────────────────────────────────────────────────
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(
                f"data must be a DataFrame, str, or Path, got {type(data).__name__}"
            )

        # ── Determine target column ────────────────────────────────────────────
        if target_col is None:
            if variant == "simple":
                target_col = _SIMPLE_DEFAULT_TARGET
            else:
                # Advanced: prefer inf_adj_total_mat_lab_equip, fall back to
                # total_mat_lab_equip if the inflation-adjusted column is absent.
                target_col = (
                    _ADVANCED_DEFAULT_TARGET
                    if _ADVANCED_DEFAULT_TARGET in df.columns
                    else _SIMPLE_DEFAULT_TARGET
                )

        if target_col not in df.columns:
            raise ValueError(
                f"Target column {target_col!r} not found in data. "
                f"Available columns: {list(df.columns)}"
            )

        targets = df[target_col].dropna().to_numpy(dtype=float)
        df_clean = df.loc[df[target_col].notna()].copy()

        # ── Build inputs ───────────────────────────────────────────────────────
        engine = cls()

        if variant == "simple":
            inputs = _df_to_simple_inputs(df_clean)
            metrics = engine.train_simple(inputs, targets, test_size=test_size, rf_params=rf_params)
        else:
            inputs = _df_to_advanced_inputs(df_clean)
            metrics = engine.train_advanced(inputs, targets, test_size=test_size, rf_params=rf_params)

        # ── Save if requested ─────────────────────────────────────────────────
        if save_dir is not None:
            engine.save(save_dir)

        # ── Predict if requested ───────────────────────────────────────────────
        if predict is None:
            return [], engine, metrics

        if isinstance(predict, list):
            if variant == "simple":
                predictions = engine.predict_simple_batch(predict)
            else:
                predictions = engine.predict_advanced_batch(predict)
        else:
            if variant == "simple":
                predictions = engine.predict_simple(predict)
            else:
                predictions = engine.predict_advanced(predict)

        return predictions, engine, metrics


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame → model input converters (for apply())
# ─────────────────────────────────────────────────────────────────────────────

def _df_to_simple_inputs(df: pd.DataFrame) -> list[RegressionSimpleInput]:
    """Convert a DataFrame to a list of RegressionSimpleInput objects."""
    inputs = []
    for _, row in df.iterrows():
        inputs.append(RegressionSimpleInput(
            inflation_factor=float(row.get("inflation_factor", 1.0)),
            acf=float(row.get("acf", 1.0)),
            project_type=str(row.get("project_type", "unknown")),
            project_category=str(row.get("project_category", "unknown")),
            ciqs_complexity_category=str(row.get("ciqs_complexity_category", "Category 1")),
            official_budget_range=str(row.get("official_budget_range", "$0-$1M")),
            project_state=str(row.get("project_state", "CA")),
            county_name=str(row.get("county_name", "unknown")),
            area_type=str(row.get("area_type", "Urban")),
            region=str(row.get("region", "Region_0")),
            cnt_division=int(row.get("cnt_division", 3)),
            cnt_item_code=int(row.get("cnt_item_code", 5)),
        ))
    return inputs


def _df_to_advanced_inputs(df: pd.DataFrame) -> list[RegressionAdvancedInput]:
    """Convert a DataFrame to a list of RegressionAdvancedInput objects."""
    inputs = []
    for _, row in df.iterrows():
        inputs.append(RegressionAdvancedInput(
            acf=float(row.get("acf", 1.0)),
            project_year=int(row.get("project_year", 2020)),
            median_cost_per_unit=float(row.get("median_cost_per_unit", 100.0)),
            median_quantity_most_common_unit=float(
                row.get("median_quantity_most_common_unit", 1000.0)
            ),
            acf_state_norm=float(row.get("acf_state_norm", 1.0))
            if pd.notna(row.get("acf_state_norm"))
            else 1.0,
            project_city=str(row.get("project_city", "unknown")),
            project_state=str(row.get("project_state", "CA")),
            project_type=str(row.get("project_type", "unknown")),
            project_category=str(row.get("project_category", "unknown")),
            construction_category=str(row.get("construction_category", "Commercial")),
            most_common_unit=str(row.get("most_common_unit", "SY")),
            quantity_bin=str(row.get("quantity_bin", "100-1,000")),
            scope_cluster=int(row.get("scope_cluster", 0)),
            project_description=str(row.get("project_description", "")),
        ))
    return inputs
