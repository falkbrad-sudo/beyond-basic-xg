"""Tests for src/models/uncertainty.py.

The bootstrap is the machinery that turns this project's central hedge
("the intervals include zero") into a measurement, so its correctness
matters more than most things here.
"""
import numpy as np
import pandas as pd

from src.models.basic_xg_model import BASIC_FEATURES, fit_basic_model
from src.models.enhanced_xg_model import ENHANCED_FEATURES, fit_enhanced_model
from src.models.uncertainty import (
    bootstrap_coefficients,
    bootstrap_prediction_difference,
    stratified_bootstrap_indices,
)


def shots(n=200, n_goals=40, seed=0, signal=True):
    """Synthetic shots; `signal` controls whether pitch_control truly matters."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "distance_to_goal": rng.uniform(4, 30, n),
            "angle_to_goal": rng.uniform(0.1, 1.5, n),
            "pitch_control": rng.uniform(0, 0.7, n),
        }
    )
    if signal:
        # Outcome depends strongly on pitch_control.
        logit = -1.0 + 6.0 * frame["pitch_control"] - 0.1 * frame["distance_to_goal"]
        frame["is_goal"] = rng.random(n) < 1 / (1 + np.exp(-logit))
    else:
        goals = np.zeros(n, dtype=bool)
        goals[rng.choice(n, size=n_goals, replace=False)] = True
        frame["is_goal"] = goals
    return frame


class TestStratifiedBootstrapIndices:
    def test_preserves_the_positive_count(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        rng = np.random.default_rng(0)
        for _ in range(20):
            index = stratified_bootstrap_indices(y, rng)
            assert y[index].sum() == y.sum()

    def test_returns_the_same_number_of_rows(self):
        y = np.array([1, 0, 0, 0, 1])
        index = stratified_bootstrap_indices(y, np.random.default_rng(0))
        assert len(index) == len(y)

    def test_never_yields_a_single_class_sample(self):
        """The reason stratification exists: with 2 goals, ordinary
        resampling frequently produces a draw with none, which cannot be
        fit, and silently dropping those biases the interval."""
        y = np.array([1, 1] + [0] * 40)
        rng = np.random.default_rng(1)
        for _ in range(200):
            index = stratified_bootstrap_indices(y, rng)
            assert len(np.unique(y[index])) == 2

    def test_actually_resamples(self):
        y = np.array([1] * 10 + [0] * 10)
        rng = np.random.default_rng(0)
        draws = {tuple(stratified_bootstrap_indices(y, rng)) for _ in range(5)}
        assert len(draws) > 1


class TestBootstrapCoefficients:
    def test_returns_one_row_per_feature(self):
        result = bootstrap_coefficients(
            lambda f: fit_enhanced_model(f, feature_cols=ENHANCED_FEATURES),
            shots(), ENHANCED_FEATURES, n_bootstrap=40,
        )
        assert list(result["feature"]) == ENHANCED_FEATURES

    def test_interval_brackets_the_point_estimate(self):
        result = bootstrap_coefficients(
            lambda f: fit_enhanced_model(f, feature_cols=ENHANCED_FEATURES),
            shots(), ENHANCED_FEATURES, n_bootstrap=60,
        )
        assert (result["ci_low"] <= result["coefficient"]).all()
        assert (result["coefficient"] <= result["ci_high"]).all()

    def test_detects_a_real_effect(self):
        # pitch_control drives the outcome here, so its interval
        # should exclude zero.
        result = bootstrap_coefficients(
            lambda f: fit_enhanced_model(f, feature_cols=ENHANCED_FEATURES),
            shots(n=400, signal=True), ENHANCED_FEATURES, n_bootstrap=120,
        )
        row = result.set_index("feature").loc["pitch_control"]
        assert row["coefficient"] > 0
        assert not row["crosses_zero"]

    def test_reports_no_effect_when_there_is_none(self):
        # Outcome independent of every feature: intervals must cross zero.
        result = bootstrap_coefficients(
            lambda f: fit_enhanced_model(f, feature_cols=ENHANCED_FEATURES),
            shots(n=200, n_goals=40, signal=False),
            ENHANCED_FEATURES, n_bootstrap=120,
        )
        assert result.set_index("feature").loc["pitch_control", "crosses_zero"]

    def test_accepts_a_fit_fn_returning_a_tuple(self):
        # fit_basic_model returns (model, diagnostics); that must just work.
        result = bootstrap_coefficients(
            lambda f: fit_basic_model(f, holdout=False, scale=True),
            shots(), BASIC_FEATURES, n_bootstrap=30,
        )
        assert len(result) == len(BASIC_FEATURES)

    def test_wider_intervals_on_a_smaller_sample(self):
        def widths(n):
            result = bootstrap_coefficients(
                lambda f: fit_enhanced_model(f, feature_cols=ENHANCED_FEATURES),
                shots(n=n, signal=True), ENHANCED_FEATURES,
                n_bootstrap=100, seed=3,
            )
            return (result["ci_high"] - result["ci_low"]).mean()

        assert widths(60) > widths(600)

    def test_records_draw_counts(self):
        result = bootstrap_coefficients(
            lambda f: fit_enhanced_model(f, feature_cols=ENHANCED_FEATURES),
            shots(), ENHANCED_FEATURES, n_bootstrap=25,
        )
        assert result.attrs["n_bootstrap"] + result.attrs["n_failed_draws"] == 25

    def test_is_reproducible_for_a_fixed_seed(self):
        def run():
            return bootstrap_coefficients(
                lambda f: fit_enhanced_model(f, feature_cols=ENHANCED_FEATURES),
                shots(), ENHANCED_FEATURES, n_bootstrap=30, seed=7,
            )["ci_low"].to_numpy()

        np.testing.assert_allclose(run(), run())


class TestBootstrapPredictionDifference:
    def test_interval_brackets_the_point_estimate(self):
        result = bootstrap_prediction_difference(
            shots(),
            lambda f: fit_basic_model(f, holdout=False, scale=True),
            lambda f: fit_enhanced_model(f, feature_cols=ENHANCED_FEATURES),
            BASIC_FEATURES, ENHANCED_FEATURES, n_bootstrap=40,
        )
        assert result["ci_low"] <= result["mean_abs_difference"] <= result["ci_high"]

    def test_difference_is_non_negative(self):
        result = bootstrap_prediction_difference(
            shots(),
            lambda f: fit_basic_model(f, holdout=False, scale=True),
            lambda f: fit_enhanced_model(f, feature_cols=ENHANCED_FEATURES),
            BASIC_FEATURES, ENHANCED_FEATURES, n_bootstrap=30,
        )
        assert result["ci_low"] >= 0

    def test_bigger_when_the_extra_feature_matters(self):
        def gap(signal):
            return bootstrap_prediction_difference(
                shots(n=300, signal=signal),
                lambda f: fit_basic_model(f, holdout=False, scale=True),
                lambda f: fit_enhanced_model(f, feature_cols=ENHANCED_FEATURES),
                BASIC_FEATURES, ENHANCED_FEATURES, n_bootstrap=40, seed=5,
            )["mean_abs_difference"]

        assert gap(signal=True) > gap(signal=False)
