"""
Tests for models/parametric.py

Verifies that the ParametricEngine:
  1. Trains correctly on synthetic and real data
  2. Produces valid RegressionOutput with point estimate and range
  3. Saves and loads models correctly
  4. Reports feature importances
  5. Handles error cases (untrained model, mismatched inputs/targets)

Run:
    cd /Users/wangzimeiyi/Desktop/Practicum
    python3 -m pytest tests/test_parametric.py -v
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.parametric import ParametricEngine
from schema import (
    RegressionAdvancedInput,
    RegressionOutput,
    RegressionSimpleInput,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic data
# ---------------------------------------------------------------------------

_RNG = np.random.RandomState(42)

STATES = ["CA", "TX", "NY", "PA", "FL"]
TYPES = ["Communication Devices", "Pavement Markers", "Offices & Warehouses"]
CATEGORIES = ["Civil", "Commercial", "Residential"]
COMPLEXITY = ["Category 1", "Category 2", "Category 3"]
BUDGETS = ["$0-$1M", "$1M-$3M", "$3M-$6M", "$6M-$15M"]
AREAS = ["Urban", "Rural"]
REGIONS = ["Region_0", "Region_1", "Region_2", "Region_3"]
UNITS = ["SY", "LF", "EA", "CY", "TON"]


def _make_simple_inputs(n: int) -> tuple[list[RegressionSimpleInput], list[float]]:
    """Generate n synthetic RegressionSimpleInput instances and target costs."""
    inputs = []
    targets = []
    for i in range(n):
        state = STATES[i % len(STATES)]
        inf = 1.0 + _RNG.uniform(0, 0.3)
        acf = 0.8 + _RNG.uniform(0, 0.4)
        div = _RNG.randint(1, 30)
        item = _RNG.randint(1, 62)

        inp = RegressionSimpleInput(
            inflation_factor=inf,
            acf=acf,
            project_type=TYPES[i % len(TYPES)],
            project_category=CATEGORIES[i % len(CATEGORIES)],
            ciqs_complexity_category=COMPLEXITY[i % len(COMPLEXITY)],
            official_budget_range=BUDGETS[i % len(BUDGETS)],
            project_state=state,
            county_name=f"County_{i}",
            area_type=AREAS[i % len(AREAS)],
            region=REGIONS[i % len(REGIONS)],
            cnt_division=div,
            cnt_item_code=item,
        )
        inputs.append(inp)

        # Target: a function of inflation, acf, and division
        target = 1_000_000 * inf * acf * (1 + div / 30) + _RNG.normal(0, 100000)
        targets.append(max(target, 10000))

    return inputs, targets


def _make_advanced_inputs(
    n: int,
) -> tuple[list[RegressionAdvancedInput], list[float]]:
    """Generate n synthetic RegressionAdvancedInput instances and target costs."""
    inputs = []
    targets = []
    for i in range(n):
        inp = RegressionAdvancedInput(
            acf=0.8 + _RNG.uniform(0, 0.4),
            project_year=2010 + _RNG.randint(0, 15),
            median_cost_per_unit=50 + _RNG.uniform(0, 500),
            median_quantity_most_common_unit=100 + _RNG.uniform(0, 5000),
            acf_state_norm=0.9 + _RNG.uniform(0, 0.2),
            project_city=f"City_{i % 20}",
            project_state=STATES[i % len(STATES)],
            project_type=TYPES[i % len(TYPES)],
            project_category=CATEGORIES[i % len(CATEGORIES)],
            construction_category="Commercial",
            most_common_unit=UNITS[i % len(UNITS)],
            quantity_bin=f"{(i % 5) * 100}-{(i % 5 + 1) * 100}",
            scope_cluster=i % 15,
            project_description=f"Project {i} involving construction work on {TYPES[i % len(TYPES)]}",
        )
        inputs.append(inp)

        target = inp.acf * inp.median_cost_per_unit * 1000 + _RNG.normal(0, 50000)
        targets.append(max(target, 10000))

    return inputs, targets


@pytest.fixture(scope="module")
def simple_data():
    return _make_simple_inputs(200)


@pytest.fixture(scope="module")
def advanced_data():
    return _make_advanced_inputs(200)


@pytest.fixture(scope="module")
def trained_engine(simple_data, advanced_data):
    """A ParametricEngine trained on both synthetic datasets."""
    engine = ParametricEngine()
    simple_inputs, simple_targets = simple_data
    engine.train_simple(simple_inputs, simple_targets)
    adv_inputs, adv_targets = advanced_data
    engine.train_advanced(adv_inputs, adv_targets)
    return engine


# ---------------------------------------------------------------------------
# 1. Constructor and state
# ---------------------------------------------------------------------------


class TestEngineInit:
    def test_new_engine_not_trained(self):
        engine = ParametricEngine()
        assert not engine.simple_is_trained
        assert not engine.advanced_is_trained


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------


class TestTraining:
    def test_train_simple_returns_metrics(self, simple_data):
        engine = ParametricEngine()
        inputs, targets = simple_data
        metrics = engine.train_simple(inputs, targets)
        assert "r2" in metrics
        assert "mape" in metrics
        assert "n_train" in metrics
        assert "n_test" in metrics
        assert metrics["n_train"] + metrics["n_test"] == len(inputs)
        assert engine.simple_is_trained

    def test_train_simple_reasonable_metrics(self, simple_data):
        engine = ParametricEngine()
        inputs, targets = simple_data
        metrics = engine.train_simple(inputs, targets)
        # Synthetic data should give decent R² (but not perfect)
        assert metrics["r2"] > 0.3, f"R² too low: {metrics['r2']}"

    def test_train_advanced_returns_metrics(self, advanced_data):
        engine = ParametricEngine()
        inputs, targets = advanced_data
        metrics = engine.train_advanced(inputs, targets)
        assert "r2" in metrics
        assert "mape" in metrics
        assert engine.advanced_is_trained

    def test_train_mismatched_lengths_raises(self):
        engine = ParametricEngine()
        inputs, _ = _make_simple_inputs(10)
        with pytest.raises(ValueError, match="same length"):
            engine.train_simple(inputs, [1.0, 2.0])


# ---------------------------------------------------------------------------
# 3. Prediction
# ---------------------------------------------------------------------------


class TestPrediction:
    def test_predict_simple_returns_output(self, trained_engine, simple_data):
        inputs, _ = simple_data
        result = trained_engine.predict_simple(inputs[0])
        assert isinstance(result, RegressionOutput)

    def test_predict_simple_has_valid_range(self, trained_engine, simple_data):
        inputs, _ = simple_data
        result = trained_engine.predict_simple(inputs[0])
        assert result.cost_low < result.cost_estimate < result.cost_high
        assert result.cost_estimate > 0

    def test_predict_simple_confidence(self, trained_engine, simple_data):
        inputs, _ = simple_data
        result = trained_engine.predict_simple(inputs[0])
        assert 0.0 <= result.confidence_level <= 1.0

    def test_predict_simple_model_version(self, trained_engine, simple_data):
        inputs, _ = simple_data
        result = trained_engine.predict_simple(inputs[0])
        assert "simple" in result.model_version

    def test_predict_simple_batch(self, trained_engine, simple_data):
        inputs, _ = simple_data
        results = trained_engine.predict_simple_batch(inputs[:10])
        assert len(results) == 10
        assert all(isinstance(r, RegressionOutput) for r in results)

    def test_predict_advanced_returns_output(self, trained_engine, advanced_data):
        inputs, _ = advanced_data
        result = trained_engine.predict_advanced(inputs[0])
        assert isinstance(result, RegressionOutput)

    def test_predict_advanced_has_valid_range(self, trained_engine, advanced_data):
        inputs, _ = advanced_data
        result = trained_engine.predict_advanced(inputs[0])
        assert result.cost_low < result.cost_estimate < result.cost_high
        assert result.cost_estimate > 0

    def test_predict_advanced_batch(self, trained_engine, advanced_data):
        inputs, _ = advanced_data
        results = trained_engine.predict_advanced_batch(inputs[:10])
        assert len(results) == 10
        assert all(isinstance(r, RegressionOutput) for r in results)

    def test_predict_untrained_simple_raises(self):
        engine = ParametricEngine()
        inp = _make_simple_inputs(1)[0][0]
        with pytest.raises(RuntimeError, match="not trained"):
            engine.predict_simple(inp)

    def test_predict_untrained_advanced_raises(self):
        engine = ParametricEngine()
        inp = _make_advanced_inputs(1)[0][0]
        with pytest.raises(RuntimeError, match="not trained"):
            engine.predict_advanced(inp)


# ---------------------------------------------------------------------------
# 4. Save / Load
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_simple(self, simple_data):
        engine = ParametricEngine()
        inputs, targets = simple_data
        engine.train_simple(inputs, targets)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.save(tmpdir)
            assert (Path(tmpdir) / "simple_pipeline.joblib").exists()
            assert (Path(tmpdir) / "model_metadata.joblib").exists()

            loaded = ParametricEngine.load(tmpdir)
            assert loaded.simple_is_trained
            assert not loaded.advanced_is_trained

            # Predictions should match
            original = engine.predict_simple(inputs[0])
            restored = loaded.predict_simple(inputs[0])
            assert original.cost_estimate == pytest.approx(
                restored.cost_estimate, rel=1e-6
            )

    def test_save_and_load_both(self, trained_engine, simple_data, advanced_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            trained_engine.save(tmpdir)
            assert (Path(tmpdir) / "simple_pipeline.joblib").exists()
            assert (Path(tmpdir) / "advanced_pipeline.joblib").exists()

            loaded = ParametricEngine.load(tmpdir)
            assert loaded.simple_is_trained
            assert loaded.advanced_is_trained

            s_inp = simple_data[0][0]
            a_inp = advanced_data[0][0]
            assert trained_engine.predict_simple(s_inp).cost_estimate == pytest.approx(
                loaded.predict_simple(s_inp).cost_estimate, rel=1e-6
            )
            assert trained_engine.predict_advanced(
                a_inp
            ).cost_estimate == pytest.approx(
                loaded.predict_advanced(a_inp).cost_estimate, rel=1e-6
            )

    def test_load_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ParametricEngine.load(tmpdir)
            assert not engine.simple_is_trained
            assert not engine.advanced_is_trained


# ---------------------------------------------------------------------------
# 5. Feature importances
# ---------------------------------------------------------------------------


class TestFeatureImportances:
    def test_simple_importances(self, trained_engine):
        importances = trained_engine.get_feature_importances("simple")
        assert isinstance(importances, dict)
        assert len(importances) > 0
        total = sum(importances.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_advanced_importances(self, trained_engine):
        importances = trained_engine.get_feature_importances("advanced")
        assert isinstance(importances, dict)
        assert len(importances) > 0

    def test_importances_untrained_raises(self):
        engine = ParametricEngine()
        with pytest.raises(RuntimeError, match="not trained"):
            engine.get_feature_importances("simple")

    def test_importances_unknown_variant_raises(self, trained_engine):
        with pytest.raises(ValueError, match="Unknown variant"):
            trained_engine.get_feature_importances("foo")


# ---------------------------------------------------------------------------
# 6. apply() — one-step pipeline
# ---------------------------------------------------------------------------


def _make_synthetic_df(n: int = 200) -> pd.DataFrame:
    """Build a DataFrame that mimics loader output for apply() tests."""
    rng = np.random.RandomState(99)
    rows = []
    for i in range(n):
        rows.append(
            {
                "project_id": f"P{i:04d}",
                "project_type": TYPES[i % len(TYPES)],
                "project_category": CATEGORIES[i % len(CATEGORIES)],
                "project_state": STATES[i % len(STATES)],
                "project_city": f"City_{i % 20}",
                "county_name": f"County_{i % 10}",
                "ciqs_complexity_category": COMPLEXITY[i % len(COMPLEXITY)],
                "official_budget_range": BUDGETS[i % len(BUDGETS)],
                "area_type": AREAS[i % len(AREAS)],
                "construction_category": "Commercial",
                "most_common_unit": UNITS[i % len(UNITS)],
                "cnt_division": rng.randint(1, 30),
                "cnt_item_code": rng.randint(1, 62),
                "dod_acf_2024": 0.8 + rng.uniform(0, 0.4),
                "project_year": 2010 + rng.randint(0, 15),
                "project_latitude": 30.0 + rng.uniform(0, 15),
                "project_longitude": -120.0 + rng.uniform(0, 40),
                "acf": 0.8 + rng.uniform(0, 0.4),
                "median_cost_per_unit": 50 + rng.uniform(0, 500),
                "median_quantity_most_common_unit": 100 + rng.uniform(0, 5000),
                "project_description": f"Construction project {i}",
                "total_mat_lab_equip": 500_000 + rng.uniform(0, 2_000_000),
            }
        )
    return pd.DataFrame(rows)


class TestApply:
    """Tests for ParametricEngine.apply() one-step pipeline."""

    def test_apply_simple_with_dataframe(self):
        df = _make_synthetic_df(200)
        predictions, engine, metrics = ParametricEngine.apply(
            df, variant="simple"
        )
        assert engine.simple_is_trained
        assert "r2" in metrics
        assert "mape" in metrics
        assert isinstance(predictions, list)
        assert len(predictions) == 0

    def test_apply_simple_with_prediction(self):
        df = _make_synthetic_df(200)
        query = RegressionSimpleInput(
            inflation_factor=1.1,
            acf=1.0,
            project_type="Pavement Markers",
            project_category="Civil",
            ciqs_complexity_category="Category 1",
            official_budget_range="$0-$1M",
            project_state="CA",
            county_name="County_0",
            area_type="Urban",
            region="Region_0",
            cnt_division=5,
            cnt_item_code=10,
        )
        output, engine, metrics = ParametricEngine.apply(
            df, predict=query, variant="simple"
        )
        assert isinstance(output, RegressionOutput)
        assert output.cost_estimate > 0
        assert output.cost_low < output.cost_estimate < output.cost_high

    def test_apply_simple_batch_prediction(self):
        df = _make_synthetic_df(200)
        queries = [
            RegressionSimpleInput(
                inflation_factor=1.0 + i * 0.05,
                acf=1.0,
                project_type=TYPES[i % len(TYPES)],
                project_category=CATEGORIES[i % len(CATEGORIES)],
                ciqs_complexity_category="Category 1",
                official_budget_range="$0-$1M",
                project_state=STATES[i % len(STATES)],
                county_name=f"County_{i}",
                area_type="Urban",
                region="Region_0",
                cnt_division=5,
                cnt_item_code=10,
            )
            for i in range(5)
        ]
        preds, engine, metrics = ParametricEngine.apply(
            df, predict=queries, variant="simple"
        )
        assert isinstance(preds, list)
        assert len(preds) == 5
        assert all(isinstance(p, RegressionOutput) for p in preds)

    def test_apply_advanced_with_dataframe(self):
        df = _make_synthetic_df(200)
        predictions, engine, metrics = ParametricEngine.apply(
            df, variant="advanced"
        )
        assert engine.advanced_is_trained
        assert "r2" in metrics
        assert isinstance(predictions, list)
        assert len(predictions) == 0

    def test_apply_advanced_with_prediction(self):
        df = _make_synthetic_df(200)
        query = RegressionAdvancedInput(
            acf=1.0,
            project_year=2022,
            median_cost_per_unit=200.0,
            median_quantity_most_common_unit=500.0,
            acf_state_norm=1.0,
            project_city="City_0",
            project_state="CA",
            project_type="Pavement Markers",
            project_category="Civil",
            construction_category="Commercial",
            most_common_unit="SY",
            quantity_bin="100-1,000",
            scope_cluster=3,
            project_description="Road construction project",
        )
        output, engine, metrics = ParametricEngine.apply(
            df, predict=query, variant="advanced"
        )
        assert isinstance(output, RegressionOutput)
        assert output.cost_estimate > 0

    def test_apply_saves_artifacts(self):
        df = _make_synthetic_df(200)
        with tempfile.TemporaryDirectory() as tmpdir:
            _, engine, _ = ParametricEngine.apply(
                df, variant="simple", save_dir=tmpdir
            )
            assert (Path(tmpdir) / "simple_pipeline.joblib").exists()
            assert (Path(tmpdir) / "model_metadata.joblib").exists()

            loaded = ParametricEngine.load(tmpdir)
            assert loaded.simple_is_trained

    def test_apply_with_csv_path(self):
        df = _make_synthetic_df(200)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            df.to_csv(csv_path, index=False)

            _, engine, metrics = ParametricEngine.apply(
                str(csv_path), variant="simple"
            )
            assert engine.simple_is_trained
            assert "r2" in metrics

    def test_apply_invalid_variant_raises(self):
        df = _make_synthetic_df(50)
        with pytest.raises(ValueError, match="variant must be"):
            ParametricEngine.apply(df, variant="unknown")

    def test_apply_missing_target_col_raises(self):
        df = _make_synthetic_df(50)
        with pytest.raises(ValueError, match="Target column"):
            ParametricEngine.apply(df, target_col="nonexistent_col")

    def test_apply_invalid_data_type_raises(self):
        with pytest.raises(TypeError, match="data must be"):
            ParametricEngine.apply(12345, variant="simple")  # type: ignore[arg-type]

    def test_apply_simple_uses_record_inflation_factor(self):
        """apply() should use rec.inflation_factor from the DataFrame when available."""
        df = _make_synthetic_df(200)
        # Set a known inflation_factor on every row
        df["inflation_factor"] = 1.42
        _, engine, metrics = ParametricEngine.apply(df, variant="simple")
        assert engine.simple_is_trained
        assert metrics["n_train"] > 0

    def test_apply_simple_falls_back_to_computed_inflation(self):
        """When inflation_factor column is missing, apply() computes from project_year."""
        df = _make_synthetic_df(200)
        if "inflation_factor" in df.columns:
            df.drop(columns=["inflation_factor"], inplace=True)
        _, engine, metrics = ParametricEngine.apply(df, variant="simple")
        assert engine.simple_is_trained
        assert metrics["n_train"] > 0

    def test_apply_simple_prefers_acf_over_dod_acf(self):
        """apply() should use rec.acf when present, falling back to dod_acf_2024."""
        df = _make_synthetic_df(200)
        df["acf"] = 1.25
        df["dod_acf_2024"] = 0.80
        _, engine, metrics = ParametricEngine.apply(df, variant="simple")
        assert engine.simple_is_trained

    def test_apply_simple_uses_dod_acf_when_acf_missing(self):
        """When acf column is absent, apply() should fall back to dod_acf_2024."""
        df = _make_synthetic_df(200)
        if "acf" in df.columns:
            df.drop(columns=["acf"], inplace=True)
        # Ensure dod_acf_2024 exists
        df["dod_acf_2024"] = 1.05
        _, engine, metrics = ParametricEngine.apply(df, variant="simple")
        assert engine.simple_is_trained
        assert metrics["n_train"] > 0

    def test_apply_simple_defaults_acf_to_one_when_both_missing(self):
        """When both acf and dod_acf_2024 are NaN/missing, apply() uses 1.0."""
        df = _make_synthetic_df(200)
        if "acf" in df.columns:
            df.drop(columns=["acf"], inplace=True)
        if "dod_acf_2024" in df.columns:
            df.drop(columns=["dod_acf_2024"], inplace=True)
        _, engine, metrics = ParametricEngine.apply(df, variant="simple")
        assert engine.simple_is_trained
        assert metrics["n_train"] > 0

    def test_apply_simple_with_regression_target_col(self):
        """apply() should work with the regression CSV target column name."""
        df = _make_synthetic_df(200)
        df["total_project_cost_normalized_2025"] = df["total_mat_lab_equip"] * 1.1
        _, engine, metrics = ParametricEngine.apply(
            df,
            variant="simple",
            target_col="total_project_cost_normalized_2025",
        )
        assert engine.simple_is_trained
        assert metrics["n_train"] > 0

    def test_apply_simple_prediction_with_precomputed_fields(self):
        """Predictions should work after training on data with pre-computed fields."""
        df = _make_synthetic_df(200)
        df["inflation_factor"] = 1.15
        query = RegressionSimpleInput(
            inflation_factor=1.15,
            acf=1.08,
            project_type="Pavement Markers",
            project_category="Civil",
            ciqs_complexity_category="Category 1",
            official_budget_range="$0-$1M",
            project_state="CA",
            county_name="County_0",
            area_type="Urban",
            region="Region_0",
            cnt_division=5,
            cnt_item_code=10,
        )
        output, engine, metrics = ParametricEngine.apply(
            df, predict=query, variant="simple"
        )
        assert isinstance(output, RegressionOutput)
        assert output.cost_estimate > 0
        assert output.cost_low < output.cost_estimate < output.cost_high

    def test_apply_custom_rf_params(self):
        df = _make_synthetic_df(200)
        custom_params = {
            "n_estimators": 50,
            "max_depth": 5,
            "n_jobs": 1,
            "random_state": 42,
        }
        _, engine, metrics = ParametricEngine.apply(
            df, variant="simple", rf_params=custom_params
        )
        assert engine.simple_is_trained
        assert metrics["n_train"] > 0

    def test_apply_returns_engine_usable_for_further_predictions(self):
        df = _make_synthetic_df(200)
        _, engine, _ = ParametricEngine.apply(df, variant="simple")
        query = RegressionSimpleInput(
            inflation_factor=1.1,
            acf=1.0,
            project_type="Pavement Markers",
            project_category="Civil",
            ciqs_complexity_category="Category 1",
            official_budget_range="$0-$1M",
            project_state="CA",
            county_name="County_0",
            area_type="Urban",
            region="Region_0",
            cnt_division=5,
            cnt_item_code=10,
        )
        result = engine.predict_simple(query)
        assert isinstance(result, RegressionOutput)
        assert result.cost_estimate > 0
