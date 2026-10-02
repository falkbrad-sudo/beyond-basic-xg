"""Tests for src/features/geometry.py.

Every test here has a hand-computable expected value. This is pure geometry,
so there is no excuse for an untested formula.
"""
import math

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import normalize_playing_direction
from src.features.geometry import (
    GOAL_POST_Y,
    GOAL_X,
    add_geometry_features,
    angle_to_goal,
    distance_to_goal,
)


class TestDistanceToGoal:
    def test_on_goal_line_centered_is_zero(self):
        assert distance_to_goal(GOAL_X, 0) == pytest.approx(0.0)

    def test_center_circle_is_52_5m(self):
        # Center of the pitch (0, 0) to the goal line (52.5, 0): pure
        # x-distance since y=0, so this is just GOAL_X itself.
        assert distance_to_goal(0, 0) == pytest.approx(52.5)

    def test_penalty_spot_is_11m(self):
        assert distance_to_goal(GOAL_X - 11, 0) == pytest.approx(11.0)

    def test_pythagorean_off_center_shot(self):
        # 3-4-5 triangle: 4m short of goal line, 3m wide -> distance 5m.
        assert distance_to_goal(GOAL_X - 4, 3) == pytest.approx(5.0)

    def test_vectorized_input_returns_array(self):
        result = distance_to_goal(np.array([GOAL_X, 0]), np.array([0, 0]))
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [0.0, 52.5])


class TestAngleToGoal:
    def test_standing_on_goal_line_centered_gives_pi(self):
        # At (GOAL_X, 0), exactly between the posts on the line itself:
        # the angle should be pi (180 degrees) -- looking left and right
        # directly at each post along the goal line.
        assert angle_to_goal(GOAL_X, 0) == pytest.approx(math.pi, rel=1e-4)

    def test_penalty_spot_matches_hand_calculation(self):
        # Half-angle = arctan(GOAL_POST_Y / 11), full angle = 2x that.
        expected = 2 * math.atan(GOAL_POST_Y / 11)
        assert angle_to_goal(GOAL_X - 11, 0) == pytest.approx(expected, rel=1e-6)

    def test_angle_shrinks_further_from_goal_on_same_line(self):
        near = angle_to_goal(GOAL_X - 11, 0)
        far = angle_to_goal(GOAL_X - 30, 0)
        assert far < near

    def test_angle_shrinks_toward_tight_byline_angle(self):
        # Same distance-ish from goal line, but far out wide -> tighter angle
        # than dead center at a similar distance.
        center = angle_to_goal(GOAL_X - 11, 0)
        wide = angle_to_goal(GOAL_X - 2, 20)
        assert wide < center

    def test_symmetric_about_the_centerline(self):
        left = angle_to_goal(GOAL_X - 11, 5)
        right = angle_to_goal(GOAL_X - 11, -5)
        assert left == pytest.approx(right, rel=1e-9)

    def test_scalar_input_returns_python_float(self):
        result = angle_to_goal(GOAL_X - 11, 0)
        assert isinstance(result, float)

    def test_vectorized_input_returns_array(self):
        result = angle_to_goal(np.array([GOAL_X - 11, GOAL_X - 30]), np.array([0, 0]))
        assert isinstance(result, np.ndarray)
        assert result[0] > result[1]  # closer shot -> wider angle


class TestAddGeometryFeatures:
    def test_adds_expected_columns(self):
        shots = pd.DataFrame({"x": [GOAL_X - 11, GOAL_X - 20], "y": [0, 5]})
        out = add_geometry_features(shots)
        assert "distance_to_goal" in out.columns
        assert "angle_to_goal" in out.columns
        assert len(out) == len(shots)

    def test_does_not_mutate_input(self):
        shots = pd.DataFrame({"x": [GOAL_X - 11], "y": [0]})
        add_geometry_features(shots)
        assert "distance_to_goal" not in shots.columns


class TestGeometryIsDirectionNormalized:
    """Regression tests for the pitch-frame / attacking-frame distinction.

    geometry.py hardcodes GOAL_X = +52.5. These tests pin down that shots
    taken toward the other goal must be passed through
    cleaning.normalize_playing_direction() first, and that doing so makes
    them equivalent to their mirror image.
    """

    def test_normalized_left_attacking_shot_matches_its_mirror(self):
        # Same chance, taken toward each goal: 11m out, dead center.
        attacking_right = pd.DataFrame({"x": [GOAL_X - 11], "y": [0.0]})
        attacking_left = pd.DataFrame({"x": [-(GOAL_X - 11)], "y": [0.0]})

        right_feats = add_geometry_features(attacking_right)
        left_feats = add_geometry_features(
            normalize_playing_direction(attacking_left, attacking_right=False)
        )

        assert left_feats["distance_to_goal"].iloc[0] == pytest.approx(
            right_feats["distance_to_goal"].iloc[0]
        )
        assert left_feats["angle_to_goal"].iloc[0] == pytest.approx(
            right_feats["angle_to_goal"].iloc[0]
        )
        # And it is the hand-computable value, not just self-consistent.
        assert left_feats["distance_to_goal"].iloc[0] == pytest.approx(11.0)

    def test_off_center_shot_survives_the_flip(self):
        # 4m short of the goal line, 3m wide -> 5m by the 3-4-5 triangle,
        # whichever goal it was taken toward.
        attacking_left = pd.DataFrame({"x": [-(GOAL_X - 4)], "y": [-3.0]})
        feats = add_geometry_features(
            normalize_playing_direction(attacking_left, attacking_right=False)
        )
        assert feats["distance_to_goal"].iloc[0] == pytest.approx(5.0)

    def test_unnormalized_left_attacking_shot_is_wildly_wrong(self):
        # The failure this guards against: without normalization, an 11m
        # chance reads as ~94m -- plausible-looking garbage, since the
        # formula happily measures to the goal at the far end of the pitch.
        attacking_left = pd.DataFrame({"x": [-(GOAL_X - 11)], "y": [0.0]})
        unnormalized = add_geometry_features(attacking_left)
        assert unnormalized["distance_to_goal"].iloc[0] == pytest.approx(94.0)
