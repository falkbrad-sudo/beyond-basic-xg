"""Tests for src/viz/style.py.

The point of these is that the colour choices are CHECKED, not eyeballed.
Contrast is a computable property, so it gets an assertion like any other
number in this project.
"""
import pytest

from src.viz.style import (
    BLUE_RAMP,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ORDINAL_LIGHTEST,
    SERIES_1,
    SURFACE,
    contrast_ratio,
    relative_luminance,
)


class TestContrastRatio:
    def test_black_on_white_is_21_to_1(self):
        assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0, abs=0.05)

    def test_identical_colors_are_1_to_1(self):
        assert contrast_ratio("#2a78d6", "#2a78d6") == pytest.approx(1.0, abs=1e-9)

    def test_is_symmetric(self):
        a = contrast_ratio("#2a78d6", SURFACE)
        b = contrast_ratio(SURFACE, "#2a78d6")
        assert a == pytest.approx(b)

    def test_luminance_is_ordered_as_expected(self):
        assert relative_luminance("#000000") < relative_luminance("#2a78d6")
        assert relative_luminance("#2a78d6") < relative_luminance("#ffffff")


class TestRampGates:
    """The gates the ramp has to clear to be legible on this surface."""

    def test_ramp_is_monotonically_darkening(self):
        # A sequential ramp that is not monotonic in lightness misencodes
        # magnitude, which is the whole job of the ramp.
        steps = sorted(BLUE_RAMP)
        luminances = [relative_luminance(BLUE_RAMP[s]) for s in steps]
        assert luminances == sorted(luminances, reverse=True)

    def test_lightest_discrete_step_clears_the_2_to_1_floor(self):
        # Discrete marks need to be visible against the surface; lighter
        # steps than this are reserved for continuous fills where "nearly
        # invisible" correctly means "nearly zero".
        assert contrast_ratio(ORDINAL_LIGHTEST, SURFACE) >= 2.0

    def test_steps_lighter_than_the_ordinal_floor_would_fail_it(self):
        # Documents why ORDINAL_LIGHTEST is where it is, rather than lighter.
        assert contrast_ratio(BLUE_RAMP[100], SURFACE) < 2.0

    def test_series_accent_is_visible_on_the_surface(self):
        assert contrast_ratio(SERIES_1, SURFACE) >= 3.0

    def test_darkest_step_is_strongly_visible(self):
        assert contrast_ratio(BLUE_RAMP[700], SURFACE) >= 7.0


class TestInkGates:
    def test_primary_ink_meets_body_text_contrast(self):
        assert contrast_ratio(INK_PRIMARY, SURFACE) >= 7.0

    def test_secondary_ink_meets_body_text_contrast(self):
        assert contrast_ratio(INK_SECONDARY, SURFACE) >= 4.5

    def test_muted_ink_is_recessive_but_still_legible(self):
        ratio = contrast_ratio(INK_MUTED, SURFACE)
        assert 2.5 <= ratio < 4.5


class TestDivergingControlRamp:
    """Pitch control is a polarity, so its ramp needs opposite poles and a
    midpoint that reads as 'neither'."""

    def test_both_poles_are_visible_on_the_surface(self):
        from src.viz.style import ATTACK_COLOR, DEFEND_COLOR

        assert contrast_ratio(ATTACK_COLOR, SURFACE) >= 4.5
        assert contrast_ratio(DEFEND_COLOR, SURFACE) >= 4.5

    def test_midpoint_is_near_neutral(self):
        # A hue at the diverging midpoint would imply the boundary means
        # something; it must read as nothing.
        from src.viz.style import NEUTRAL_MID

        value = NEUTRAL_MID.lstrip("#")
        r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
        assert max(r, g, b) - min(r, g, b) <= 8

    def test_poles_are_far_apart_for_normal_vision(self):
        from src.viz.style import ATTACK_COLOR, DEFEND_COLOR

        # Crude but computed: the two poles must differ a lot in hue, not
        # only in lightness, or the scale reads as sequential.
        def rgb(h):
            h = h.lstrip("#")
            return [int(h[i : i + 2], 16) for i in (0, 2, 4)]

        a, d = rgb(ATTACK_COLOR), rgb(DEFEND_COLOR)
        # red channel up for the defence pole, blue channel up for the attack
        assert d[0] > a[0] and a[2] > d[2]

    def test_red_ramp_is_monotonically_darkening(self):
        from src.viz.style import RED_RAMP

        steps = sorted(RED_RAMP)
        luminances = [relative_luminance(RED_RAMP[s]) for s in steps]
        assert luminances == sorted(luminances, reverse=True)
