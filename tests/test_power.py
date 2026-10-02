"""Tests for src/models/power.py.

Two things need checking: that the Wald standard errors from the
information matrix are right (validated against a bootstrap, which makes no
distributional assumption), and that the power curve behaves the way a
power curve must.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.basic_xg_model import BASIC_FEATURES, fit_basic_model
from src.models.power import (
    power_analysis,
    power_at_sample_size,
    required_sample_size,
    wald_standard_errors,
)
from src.models.uncertainty import bootstrap_coefficients


def shots(n=3000, seed=0):
    """Synthetic shots with a realistic xG-like goal rate."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "distance_to_goal": rng.uniform(4, 35, n),
            "angle_to_goal": rng.uniform(0.05, 1.5, n),
        }
    )
    logit = -1.2 - 0.09 * frame["distance_to_goal"] + 0.9 * frame["angle_to_goal"]
    frame["is_goal"] = rng.random(n) < 1 / (1 + np.exp(-logit))
    return frame


class TestWaldStandardErrors:
    def test_one_per_feature_and_all_positive(self):
        data = shots()
        model, _ = fit_basic_model(data, holdout=False, scale=True)
        se = wald_standard_errors(model, data, BASIC_FEATURES)
        assert list(se.index) == BASIC_FEATURES
        assert (se > 0).all()

    def test_shrinks_with_sample_size_like_one_over_sqrt_n(self):
        small, large = shots(n=1000, seed=1), shots(n=4000, seed=1)
        se_small = wald_standard_errors(
            fit_basic_model(small, holdout=False, scale=True)[0], small, BASIC_FEATURES
        )
        se_large = wald_standard_errors(
            fit_basic_model(large, holdout=False, scale=True)[0], large, BASIC_FEATURES
        )
        ratio = (se_small / se_large).mean()
        # 4x the data -> about half the standard error.
        assert 1.6 < ratio < 2.4

    def test_agrees_with_the_bootstrap(self):
        """The analytic SE is only trustworthy if it matches a method that
        assumes nothing. On a large sample the two should be close."""
        data = shots(n=3000, seed=2)
        model, _ = fit_basic_model(data, holdout=False, scale=True)
        analytic = wald_standard_errors(model, data, BASIC_FEATURES)

        intervals = bootstrap_coefficients(
            lambda f: fit_basic_model(f, holdout=False, scale=True),
            data, BASIC_FEATURES, n_bootstrap=250, seed=4,
        ).set_index("feature")
        # A 95% percentile interval spans about 2 * 1.96 standard errors.
        empirical = (intervals["ci_high"] - intervals["ci_low"]) / (2 * 1.96)

        for feature in BASIC_FEATURES:
            assert empirical[feature] == pytest.approx(analytic[feature], rel=0.35)


class TestPowerAtSampleSize:
    def test_power_rises_with_sample_size(self):
        values = power_at_sample_size(0.2, 0.05, 1000, np.array([100, 1000, 10000]))
        assert values[0] < values[1] < values[2]

    def test_power_rises_with_effect_size(self):
        small = power_at_sample_size(0.05, 0.05, 1000, 1000)
        large = power_at_sample_size(0.50, 0.05, 1000, 1000)
        assert small < large

    def test_bounded_in_zero_one(self):
        values = power_at_sample_size(0.3, 0.05, 1000, np.array([10, 10**7]))
        assert ((values >= 0) & (values <= 1)).all()

    def test_zero_effect_gives_power_equal_to_alpha(self):
        # With no true effect, rejection happens exactly at the test's size.
        assert power_at_sample_size(0.0, 0.05, 1000, 1000, alpha=0.05) == pytest.approx(
            0.05, abs=1e-6
        )


class TestRequiredSampleSize:
    def test_hits_the_requested_power(self):
        n = required_sample_size(0.2, 0.05, 1000, target_power=0.8)
        assert power_at_sample_size(0.2, 0.05, 1000, n) == pytest.approx(0.8, abs=0.02)

    def test_smaller_effects_need_more_data(self):
        assert required_sample_size(0.1, 0.05, 1000) > required_sample_size(
            0.3, 0.05, 1000
        )

    def test_quartering_the_effect_roughly_sixteen_times_the_data(self):
        # n scales as 1/effect^2.
        ratio = required_sample_size(0.05, 0.05, 1000) / required_sample_size(
            0.2, 0.05, 1000
        )
        assert 14 < ratio < 18


class TestPowerAnalysis:
    def test_reports_shots_and_goals_per_effect_size(self):
        data = shots()
        model, _ = fit_basic_model(data, holdout=False, scale=True)
        table = power_analysis(model, data, BASIC_FEATURES, "angle_to_goal")
        assert {"effect_size_per_sd", "required_shots", "required_goals"} <= set(
            table.columns
        )
        assert (table["required_goals"] <= table["required_shots"]).all()

    def test_required_sample_decreases_with_effect_size(self):
        data = shots()
        model, _ = fit_basic_model(data, holdout=False, scale=True)
        table = power_analysis(model, data, BASIC_FEATURES, "angle_to_goal")
        assert table["required_shots"].is_monotonic_decreasing

    def test_a_tiny_sample_has_low_power(self):
        data = shots()
        model, _ = fit_basic_model(data, holdout=False, scale=True)
        table = power_analysis(model, data, BASIC_FEATURES, "angle_to_goal")
        # The project's own 66-shot Metrica sample should look underpowered
        # for every modest effect size.
        assert (table[table["effect_size_per_sd"] <= 0.2]["power_with_66_shots"]
                < 0.3).all()

    def test_unknown_feature_raises(self):
        data = shots()
        model, _ = fit_basic_model(data, holdout=False, scale=True)
        with pytest.raises(ValueError, match="not in feature_cols"):
            power_analysis(model, data, BASIC_FEATURES, "nonsense")
