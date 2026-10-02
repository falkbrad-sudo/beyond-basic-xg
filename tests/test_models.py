"""Tests for src/models/enhanced_xg_model.py and src/models/evaluate.py.

The most important test in this file is
TestSummaryStatistics::test_withholds_performance_metrics_on_few_goals. The
project's central reporting rule is implemented in code there, so it is
checked in code here.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.basic_xg_model import BASIC_FEATURES, fit_basic_model
from src.models.enhanced_xg_model import (
    ENHANCED_FEATURES,
    ENHANCED_FEATURES_FULL,
    enhanced_model_diagnostics,
    fit_enhanced_model,
)
from src.models.evaluate import (
    MIN_GOALS_FOR_PERFORMANCE_METRICS,
    compare_predictions,
    find_biggest_disagreements,
    summary_statistics,
)


def shots(n: int = 80, n_goals: int = 30, seed: int = 0) -> pd.DataFrame:
    """A synthetic shot table with every feature the models need.

    Synthetic on purpose: these tests are about model plumbing and the
    reporting rules, not about the real sample, and a fixture lets the goal
    count be set precisely to probe the reporting thresholds.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "distance_to_goal": rng.uniform(4, 30, n),
            "angle_to_goal": rng.uniform(0.1, 1.5, n),
            "pitch_control": rng.uniform(0, 0.7, n),
            "defenders_in_cone": rng.integers(0, 4, n),
            "goalkeeper_distance": rng.uniform(0.5, 12, n),
        }
    )
    goals = np.zeros(n, dtype=bool)
    goals[rng.choice(n, size=n_goals, replace=False)] = True
    frame["is_goal"] = goals
    return frame


class TestFitEnhancedModel:
    def test_returns_a_pipeline_with_scaling(self):
        model = fit_enhanced_model(shots())
        assert isinstance(model, Pipeline)
        # Scaling must travel with the model so predictions cannot be made
        # on unscaled inputs by accident.
        assert any("scaler" in name for name in model.named_steps)

    def test_defaults_to_the_three_feature_set(self):
        assert ENHANCED_FEATURES == [
            "distance_to_goal",
            "angle_to_goal",
            "pitch_control",
        ]

    def test_predicts_probabilities_in_range(self):
        data = shots()
        model = fit_enhanced_model(data)
        probabilities = model.predict_proba(
            data[ENHANCED_FEATURES].to_numpy(float)
        )[:, 1]
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

    def test_full_feature_variant_fits(self):
        model = fit_enhanced_model(shots(), feature_cols=ENHANCED_FEATURES_FULL)
        assert len(model[-1].coef_[0]) == len(ENHANCED_FEATURES_FULL)

    def test_missing_feature_raises(self):
        with pytest.raises(ValueError, match="missing required columns"):
            fit_enhanced_model(shots().drop(columns=["pitch_control"]))

    def test_single_class_sample_raises(self):
        with pytest.raises(ValueError, match="single outcome class"):
            fit_enhanced_model(shots(n=20, n_goals=0))

    def test_stronger_regularization_shrinks_coefficients(self):
        data = shots()
        loose = fit_enhanced_model(data, C=100.0)
        tight = fit_enhanced_model(data, C=0.01)
        assert np.abs(tight[-1].coef_).sum() < np.abs(loose[-1].coef_).sum()


class TestEnhancedModelDiagnostics:
    def test_reports_one_coefficient_per_feature(self):
        model = fit_enhanced_model(shots())
        diagnostics = enhanced_model_diagnostics(model)
        assert set(diagnostics["coefficients_per_sd"]) == set(ENHANCED_FEATURES)

    def test_carries_the_sample_size_caveat_with_the_numbers(self):
        model = fit_enhanced_model(shots())
        note = enhanced_model_diagnostics(model)["interpretation_note"]
        assert "directional" in note.lower()


class TestComparePredictions:
    def _models(self, data):
        basic, _ = fit_basic_model(data, holdout=False, scale=True)
        enhanced = fit_enhanced_model(data)
        return basic, enhanced

    def test_adds_the_three_prediction_columns(self):
        data = shots()
        basic, enhanced = self._models(data)
        out = compare_predictions(
            data, basic, enhanced, BASIC_FEATURES, ENHANCED_FEATURES
        )
        for column in ("basic_xg_pred", "enhanced_xg_pred", "prediction_diff"):
            assert column in out.columns

    def test_prediction_diff_is_enhanced_minus_basic(self):
        data = shots()
        basic, enhanced = self._models(data)
        out = compare_predictions(
            data, basic, enhanced, BASIC_FEATURES, ENHANCED_FEATURES
        )
        np.testing.assert_allclose(
            out["prediction_diff"],
            out["enhanced_xg_pred"] - out["basic_xg_pred"],
        )

    def test_missing_feature_raises(self):
        data = shots()
        basic, enhanced = self._models(data)
        with pytest.raises(ValueError, match="missing required columns"):
            compare_predictions(
                data.drop(columns=["pitch_control"]),
                basic, enhanced, BASIC_FEATURES, ENHANCED_FEATURES,
            )

    def test_does_not_mutate_input(self):
        data = shots()
        basic, enhanced = self._models(data)
        compare_predictions(data, basic, enhanced, BASIC_FEATURES, ENHANCED_FEATURES)
        assert "basic_xg_pred" not in data.columns


class TestFindBiggestDisagreements:
    def _compared(self, n=40):
        data = shots(n=n, n_goals=15)
        basic, _ = fit_basic_model(data, holdout=False, scale=True)
        enhanced = fit_enhanced_model(data)
        return compare_predictions(
            data, basic, enhanced, BASIC_FEATURES, ENHANCED_FEATURES
        )

    def test_returns_n_rows_sorted_by_absolute_difference(self):
        top = find_biggest_disagreements(self._compared(), n=5)
        assert len(top) == 5
        magnitudes = top["prediction_diff"].abs().tolist()
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_ranks_by_magnitude_not_sign(self):
        # A large negative disagreement must outrank a small positive one.
        compared = self._compared()
        top = find_biggest_disagreements(compared, n=3)
        assert top["prediction_diff"].abs().min() >= (
            compared["prediction_diff"].abs().nsmallest(1).iloc[0]
        )

    def test_carries_context_columns_for_lookup(self):
        top = find_biggest_disagreements(self._compared(), n=3)
        assert "distance_to_goal" in top.columns
        assert "is_goal" in top.columns

    def test_requires_compare_predictions_first(self):
        with pytest.raises(ValueError, match="prediction_diff"):
            find_biggest_disagreements(shots(), n=3)


class TestSummaryStatistics:
    def _compared(self, n, n_goals):
        data = shots(n=n, n_goals=n_goals)
        basic, _ = fit_basic_model(data, holdout=False, scale=True)
        enhanced = fit_enhanced_model(data)
        return compare_predictions(
            data, basic, enhanced, BASIC_FEATURES, ENHANCED_FEATURES
        )

    def test_withholds_performance_metrics_on_few_goals(self):
        """METHODOLOGY.md principle #2, enforced in code.

        On a sample like the project's real one (9 goals), log-loss and AUC
        must NOT be computed, because a number there would invite a conclusion the
        data cannot support.
        """
        results = summary_statistics(self._compared(n=66, n_goals=9))
        assert "performance_metrics_withheld" in results
        assert "basic_auc" not in results
        assert "enhanced_auc" not in results
        assert "basic_log_loss" not in results
        assert "9 goals" in results["performance_metrics_withheld"]

    def test_reports_performance_metrics_when_goals_suffice(self):
        results = summary_statistics(
            self._compared(n=200, n_goals=MIN_GOALS_FOR_PERFORMANCE_METRICS + 20)
        )
        assert "performance_metrics_withheld" not in results
        assert 0 <= results["basic_auc"] <= 1
        assert 0 <= results["enhanced_auc"] <= 1

    def test_threshold_is_exactly_at_the_documented_boundary(self):
        just_under = summary_statistics(
            self._compared(n=200, n_goals=MIN_GOALS_FOR_PERFORMANCE_METRICS - 1)
        )
        at_threshold = summary_statistics(
            self._compared(n=200, n_goals=MIN_GOALS_FOR_PERFORMANCE_METRICS)
        )
        assert "performance_metrics_withheld" in just_under
        assert "performance_metrics_withheld" not in at_threshold

    def test_always_reports_the_descriptive_comparison(self):
        results = summary_statistics(self._compared(n=66, n_goals=9))
        for key in (
            "prediction_correlation",
            "mean_abs_difference",
            "max_abs_difference",
            "n_goals",
        ):
            assert key in results

    def test_missing_prediction_columns_raise(self):
        with pytest.raises(ValueError, match="missing columns"):
            summary_statistics(shots())
