"""Tests for src/features/shot_context.py.

Every case is hand-placed so the expected count or distance is obvious by
inspection. These features are simple on purpose, and the tests pin them down.
"""
import pandas as pd
import pytest

from src.features.geometry import GOAL_X
from src.features.shot_context import (
    add_context_features,
    count_defenders_in_cone,
    goalkeeper_distance_from_goal_center,
    identify_goalkeeper,
)


def frame(players: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """Build a frame from (player, team, x, y) tuples."""
    return pd.DataFrame(
        [
            {"player": name, "team": team, "x": x, "y": y}
            for name, team, x, y in players
        ]
    )


class TestIdentifyGoalkeeper:
    def test_picks_the_player_nearest_the_defended_goal(self):
        f = frame(
            [
                ("keeper", "Away", 51.5, 0.3),
                ("defender", "Away", 40.0, 5.0),
                ("midfielder", "Away", 10.0, 0.0),
            ]
        )
        assert identify_goalkeeper(f, "Away")["player"] == "keeper"

    def test_ignores_the_other_team(self):
        f = frame([("striker", "Home", 52.0, 0.0), ("keeper", "Away", 50.0, 0.0)])
        assert identify_goalkeeper(f, "Away")["player"] == "keeper"

    def test_empty_team_raises(self):
        with pytest.raises(ValueError, match="no tracked players"):
            identify_goalkeeper(frame([("a", "Home", 0.0, 0.0)]), "Away")


class TestGoalkeeperDistance:
    def test_keeper_on_the_goal_line_centre_is_zero(self):
        f = frame([("keeper", "Away", GOAL_X, 0.0), ("d", "Away", 30.0, 0.0)])
        assert goalkeeper_distance_from_goal_center(f, "Away") == pytest.approx(0.0)

    def test_pythagorean_offset(self):
        # 4m off the line, 3m to the side -> 5m by the 3-4-5 triangle.
        f = frame([("keeper", "Away", GOAL_X - 4, 3.0), ("d", "Away", 20.0, 0.0)])
        assert goalkeeper_distance_from_goal_center(f, "Away") == pytest.approx(5.0)

    def test_stranded_keeper_reads_as_a_large_distance(self):
        f = frame([("keeper", "Away", GOAL_X - 12, 0.0), ("d", "Away", 10.0, 0.0)])
        assert goalkeeper_distance_from_goal_center(f, "Away") == pytest.approx(12.0)


class TestCountDefendersInCone:
    def test_defender_directly_in_front_of_goal_counts(self):
        f = frame([("d", "Away", 45.0, 0.0)])
        assert count_defenders_in_cone(f, shot_x=30.0, shot_y=0.0,
                                      defending_team="Away") == 1

    def test_defender_behind_the_shooter_does_not_count(self):
        f = frame([("d", "Away", 20.0, 0.0)])
        assert count_defenders_in_cone(f, shot_x=30.0, shot_y=0.0,
                                      defending_team="Away") == 0

    def test_defender_out_wide_is_outside_the_cone(self):
        f = frame([("d", "Away", 45.0, 25.0)])
        assert count_defenders_in_cone(f, shot_x=30.0, shot_y=0.0,
                                      defending_team="Away") == 0

    def test_counts_several_defenders(self):
        f = frame(
            [
                ("d1", "Away", 40.0, 0.0),
                ("d2", "Away", 45.0, 1.0),
                ("d3", "Away", 50.0, -1.0),
                ("wide", "Away", 45.0, 30.0),
                ("behind", "Away", 10.0, 0.0),
            ]
        )
        assert count_defenders_in_cone(f, shot_x=30.0, shot_y=0.0,
                                      defending_team="Away") == 3

    def test_attackers_are_not_counted(self):
        f = frame([("a", "Home", 45.0, 0.0)])
        assert count_defenders_in_cone(f, shot_x=30.0, shot_y=0.0,
                                      defending_team="Away") == 0

    def test_the_cone_follows_the_shooters_angle(self):
        # The same defender is in the lane for one shot and not the other,
        # because the cone is anchored at the shooter, not at the goal.
        # Defender at (50, 3.0): for a central shot at (20, 0) the cone
        # spans y in [-3.4, 3.4] at x=50, so they are inside it; for a wide
        # shot at (20, -20) it spans [-4.9, 1.9] there, so they are not.
        f = frame([("d", "Away", 50.0, 3.0)])
        central = count_defenders_in_cone(f, shot_x=20.0, shot_y=0.0,
                                          defending_team="Away")
        wide = count_defenders_in_cone(f, shot_x=20.0, shot_y=-20.0,
                                       defending_team="Away")
        assert central == 1
        assert wide == 0

    def test_no_defenders_gives_zero(self):
        assert count_defenders_in_cone(frame([("a", "Home", 1.0, 1.0)]),
                                      30.0, 0.0, "Away") == 0

    def test_goalkeeper_in_the_lane_is_counted(self):
        # A keeper on their line blocks the shot; excluding them
        # would make this feature disagree with the pitch-control value.
        f = frame([("keeper", "Away", GOAL_X - 0.5, 0.0)])
        assert count_defenders_in_cone(f, shot_x=30.0, shot_y=0.0,
                                      defending_team="Away") == 1


class TestAddContextFeatures:
    def _shots(self):
        f = frame(
            [
                ("a", "Home", 40.0, 0.0),
                ("keeper", "Away", GOAL_X - 1, 0.0),
                ("d", "Away", 45.0, 0.0),
            ]
        )
        return pd.DataFrame(
            {"x": [40.0], "y": [0.0], "team": ["Home"], "frame_positions": [f]}
        )

    def test_adds_both_columns_with_sensible_values(self):
        out = add_context_features(self._shots())
        assert out["defenders_in_cone"].iloc[0] == 2
        assert out["goalkeeper_distance"].iloc[0] == pytest.approx(1.0)

    def test_missing_columns_raise_clearly(self):
        with pytest.raises(ValueError, match="missing columns"):
            add_context_features(pd.DataFrame({"x": [1.0]}))

    def test_does_not_mutate_input(self):
        shots = self._shots()
        add_context_features(shots)
        assert "defenders_in_cone" not in shots.columns
