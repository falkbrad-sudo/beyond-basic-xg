"""Tests for src/data/cleaning.py."""
import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import (
    metrica_to_meters,
    normalize_playing_direction,
    statsbomb_to_meters,
    validate_coordinates,
)


class TestMetricaToMeters:
    def test_kickoff_point_maps_to_origin(self):
        df = pd.DataFrame({"x": [0.5], "y": [0.5]})
        out = metrica_to_meters(df, "x", "y")
        assert out["x"].iloc[0] == pytest.approx(0.0)
        assert out["y"].iloc[0] == pytest.approx(0.0)

    def test_corner_maps_to_pitch_edge(self):
        df = pd.DataFrame({"x": [1.0], "y": [1.0]})
        out = metrica_to_meters(df, "x", "y")
        assert out["x"].iloc[0] == pytest.approx(52.5)
        assert out["y"].iloc[0] == pytest.approx(34.0)


class TestStatsbombToMeters:
    def test_center_of_statsbomb_pitch_maps_to_origin(self):
        df = pd.DataFrame({"x": [60.0], "y": [40.0]})
        out = statsbomb_to_meters(df, "x", "y")
        assert out["x"].iloc[0] == pytest.approx(0.0)
        assert out["y"].iloc[0] == pytest.approx(0.0)


class TestValidateCoordinates:
    def test_valid_coordinates_do_not_raise(self):
        df = pd.DataFrame({"x": [0.0, 50.0], "y": [0.0, 30.0]})
        validate_coordinates(df, "x", "y")

    def test_out_of_bounds_raises(self):
        df = pd.DataFrame({"x": [1000.0], "y": [0.0]})
        with pytest.raises(ValueError):
            validate_coordinates(df, "x", "y")


class TestNormalizePlayingDirection:
    """The pitch frame -> attacking frame flip.

    geometry.py hardcodes the attacking goal at x = +52.5, so a shot taken
    toward x = -52.5 must be rotated before any distance/angle calculation.
    Without this, distance_to_goal silently measures to the wrong goal.
    """

    def test_attacking_left_is_rotated_180_degrees(self):
        df = pd.DataFrame({"x": [-40.0], "y": [10.0]})
        out = normalize_playing_direction(df, attacking_right=False)
        assert out["x"].iloc[0] == pytest.approx(40.0)
        # y is negated too: a 180-degree rotation, not an x-only mirror, so
        # left/right relationships between players survive the transform.
        assert out["y"].iloc[0] == pytest.approx(-10.0)

    def test_attacking_right_is_a_no_op(self):
        df = pd.DataFrame({"x": [40.0], "y": [10.0]})
        out = normalize_playing_direction(df, attacking_right=True)
        assert out["x"].iloc[0] == pytest.approx(40.0)
        assert out["y"].iloc[0] == pytest.approx(10.0)

    def test_row_wise_series_flips_only_the_rows_it_should(self):
        df = pd.DataFrame({"x": [-40.0, 40.0, -30.0], "y": [10.0, 10.0, -5.0]})
        mask = pd.Series([False, True, False])
        out = normalize_playing_direction(df, attacking_right=mask)
        np.testing.assert_allclose(out["x"], [40.0, 40.0, 30.0])
        np.testing.assert_allclose(out["y"], [-10.0, 10.0, 5.0])

    def test_flip_is_its_own_inverse(self):
        df = pd.DataFrame({"x": [-40.0, 12.5], "y": [10.0, -3.0]})
        once = normalize_playing_direction(df, attacking_right=False)
        twice = normalize_playing_direction(once, attacking_right=False)
        np.testing.assert_allclose(twice["x"], df["x"])
        np.testing.assert_allclose(twice["y"], df["y"])

    def test_center_spot_is_unchanged_by_the_flip(self):
        # The rotation is about the center spot, so (0, 0) is its fixed point.
        df = pd.DataFrame({"x": [0.0], "y": [0.0]})
        out = normalize_playing_direction(df, attacking_right=False)
        assert out["x"].iloc[0] == pytest.approx(0.0)
        assert out["y"].iloc[0] == pytest.approx(0.0)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"x": [-40.0], "y": [10.0]})
        normalize_playing_direction(df, attacking_right=False)
        assert df["x"].iloc[0] == pytest.approx(-40.0)

    def test_stays_within_pitch_bounds_after_flip(self):
        # Flipping must not push coordinates out of bounds -- the pitch is
        # symmetric about the center spot, so validate_coordinates still passes.
        df = pd.DataFrame({"x": [-52.5, 52.5], "y": [-34.0, 34.0]})
        out = normalize_playing_direction(df, attacking_right=False)
        validate_coordinates(out, "x", "y")
