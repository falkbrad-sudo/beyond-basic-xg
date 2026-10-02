"""Tests for src/features/pitch_control_at_shot.py.

Built around hand-constructed frames where the right answer is obvious by
symmetry or by who is standing closest, so the geometry is pinned down
without depending on downloaded tracking data.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.geometry import GOAL_POST_Y, GOAL_X
from src.features.pitch_control_at_shot import (
    SPEARMAN,
    VORONOI,
    _lane_grid,
    add_pitch_control_features,
    pitch_control_at_point,
    pitch_control_surface,
    shooting_lane_control,
    spearman_control_at_points,
    voronoi_control_at_points,
)


def frame(players: list[tuple[str, float, float]], with_velocity: bool = False):
    """Build a tracking frame from (team, x, y) triples."""
    rows = [
        {"player": f"p{i}", "team": team, "x": x, "y": y}
        for i, (team, x, y) in enumerate(players)
    ]
    if with_velocity:
        for row in rows:
            row["vx"], row["vy"] = 0.0, 0.0
    return pd.DataFrame(rows)


class TestVoronoiControl:
    def test_point_next_to_an_attacker_is_attacking_controlled(self):
        f = frame([("Home", 40.0, 0.0), ("Away", 0.0, 0.0)])
        value = voronoi_control_at_points(f, np.array([[41.0, 0.0]]), "Home")
        assert value[0] == 1.0

    def test_point_next_to_a_defender_is_not(self):
        f = frame([("Home", 40.0, 0.0), ("Away", 0.0, 0.0)])
        value = voronoi_control_at_points(f, np.array([[1.0, 0.0]]), "Home")
        assert value[0] == 0.0

    def test_exact_midpoint_goes_to_the_defence(self):
        # Ties are resolved conservatively for a shot-quality feature; this
        # pins that choice down so it cannot drift silently.
        f = frame([("Home", 10.0, 0.0), ("Away", -10.0, 0.0)])
        value = voronoi_control_at_points(f, np.array([[0.0, 0.0]]), "Home")
        assert value[0] == 0.0

    def test_is_symmetric_between_the_teams(self):
        f = frame([("Home", 10.0, 0.0), ("Away", -10.0, 0.0)])
        point = np.array([[5.0, 0.0]])
        assert voronoi_control_at_points(f, point, "Home")[0] == 1.0
        assert voronoi_control_at_points(f, point, "Away")[0] == 0.0

    def test_ignores_the_ball_row(self):
        f = pd.DataFrame(
            [
                {"player": "a", "team": "Home", "x": 40.0, "y": 0.0},
                {"player": "d", "team": "Away", "x": 0.0, "y": 0.0},
                {"player": "ball", "team": "ball", "x": 41.0, "y": 0.0},
            ]
        )
        # The ball sitting on the query point must not decide its ownership.
        assert voronoi_control_at_points(f, np.array([[41.0, 0.0]]), "Home")[0] == 1.0

    def test_one_sided_frame_raises(self):
        f = frame([("Home", 40.0, 0.0)])
        with pytest.raises(ValueError, match="both teams"):
            voronoi_control_at_points(f, np.array([[0.0, 0.0]]), "Home")


class TestSpearmanControl:
    def test_is_bounded_between_zero_and_one(self):
        f = frame([("Home", 40.0, 0.0), ("Away", -40.0, 5.0)], with_velocity=True)
        points = np.array([[0.0, 0.0], [50.0, 0.0], [-50.0, 20.0]])
        values = spearman_control_at_points(f, points, "Home")
        assert ((values >= 0) & (values <= 1)).all()

    def test_mirror_symmetric_frame_gives_a_contested_midpoint(self):
        f = frame([("Home", 10.0, 0.0), ("Away", -10.0, 0.0)], with_velocity=True)
        value = spearman_control_at_points(f, np.array([[0.0, 0.0]]), "Home")
        assert value[0] == pytest.approx(0.5, abs=1e-9)

    def test_closer_team_gets_more_control(self):
        f = frame([("Home", 10.0, 0.0), ("Away", -10.0, 0.0)], with_velocity=True)
        near_home = spearman_control_at_points(f, np.array([[8.0, 0.0]]), "Home")[0]
        near_away = spearman_control_at_points(f, np.array([[-8.0, 0.0]]), "Home")[0]
        assert near_home > 0.5 > near_away

    def test_extra_defenders_reduce_attacking_control(self):
        # Summing over players rather than taking each team's fastest means a
        # double-up counts for more.
        one = frame([("Home", 10.0, 0.0), ("Away", -10.0, 0.0)], with_velocity=True)
        two = frame(
            [("Home", 10.0, 0.0), ("Away", -10.0, 0.0), ("Away", -10.0, 0.1)],
            with_velocity=True,
        )
        point = np.array([[0.0, 0.0]])
        assert (
            spearman_control_at_points(two, point, "Home")[0]
            < spearman_control_at_points(one, point, "Home")[0]
        )

    def test_running_toward_a_point_beats_standing_still(self):
        standing = frame([("Home", 20.0, 0.0), ("Away", -20.0, 0.0)], with_velocity=True)
        running = standing.copy()
        running.loc[running["team"] == "Home", "vx"] = -5.0  # sprinting at the point
        point = np.array([[0.0, 0.0]])
        assert (
            spearman_control_at_points(running, point, "Home")[0]
            > spearman_control_at_points(standing, point, "Home")[0]
        )

    def test_works_without_velocity_columns(self):
        f = frame([("Home", 10.0, 0.0), ("Away", -10.0, 0.0)])
        value = spearman_control_at_points(f, np.array([[0.0, 0.0]]), "Home")
        assert value[0] == pytest.approx(0.5, abs=1e-9)


class TestLaneGrid:
    def test_all_points_fall_inside_the_shot_to_posts_triangle(self):
        grid = _lane_grid(30.0, 0.0, n=15)
        # Inside the cone means between the goal line and the shot, and
        # within the lines to each post.
        assert (grid[:, 0] >= 30.0 - 1e-9).all()
        assert (grid[:, 0] <= GOAL_X + 1e-9).all()
        depth = (grid[:, 0] - 30.0) / (GOAL_X - 30.0)
        half_width = GOAL_POST_Y * depth + 1e-6
        assert (np.abs(grid[:, 1]) <= half_width).all()

    def test_degenerate_cone_falls_back_to_the_shot_point(self):
        # A shot taken from the goal line has a zero-area cone.
        grid = _lane_grid(GOAL_X, 0.0, n=10)
        assert len(grid) >= 1


class TestShootingLaneControl:
    def test_fully_uncontested_lane_is_one(self):
        # Attacker on the lane, defence parked in the other half.
        f = frame([("Home", 40.0, 0.0), ("Away", -40.0, 0.0)])
        assert shooting_lane_control(f, 40.0, 0.0, "Home", VORONOI) == pytest.approx(1.0)

    def test_fully_covered_lane_is_zero(self):
        # Defenders blanket the goal mouth; attacker is far away.
        f = pd.DataFrame(
            [
                {"player": "a", "team": "Home", "x": -40.0, "y": 0.0},
                {"player": "d1", "team": "Away", "x": 48.0, "y": 0.0},
                {"player": "d2", "team": "Away", "x": 44.0, "y": 0.0},
            ]
        )
        assert shooting_lane_control(f, 40.0, 0.0, "Home", VORONOI) < 0.2

    def test_is_bounded(self):
        f = frame([("Home", 40.0, 2.0), ("Away", 45.0, -1.0)])
        for method in (VORONOI, SPEARMAN):
            value = shooting_lane_control(f, 40.0, 2.0, "Home", method)
            assert 0.0 <= value <= 1.0

    def test_a_defender_in_the_lane_lowers_control(self):
        clear = frame([("Home", 35.0, 0.0), ("Away", -40.0, 25.0)])
        blocked = frame([("Home", 35.0, 0.0), ("Away", 45.0, 0.0)])
        assert (
            shooting_lane_control(blocked, 35.0, 0.0, "Home", VORONOI)
            < shooting_lane_control(clear, 35.0, 0.0, "Home", VORONOI)
        )

    def test_unknown_method_raises(self):
        f = frame([("Home", 40.0, 0.0), ("Away", 0.0, 0.0)])
        with pytest.raises(ValueError, match="method must be one of"):
            shooting_lane_control(f, 40.0, 0.0, "Home", "nonsense")


class TestPitchControlAtPoint:
    def test_shooters_own_position_is_controlled_by_their_team(self):
        # The documented degeneracy: this is why lane control is the feature.
        f = frame([("Home", 40.0, 0.0), ("Away", 30.0, 10.0)])
        assert pitch_control_at_point(f, 40.0, 0.0, "Home", VORONOI) == 1.0

    def test_returns_a_plain_float(self):
        f = frame([("Home", 40.0, 0.0), ("Away", 0.0, 0.0)])
        assert isinstance(pitch_control_at_point(f, 40.0, 0.0, "Home"), float)


class TestPitchControlSurface:
    def test_shapes_line_up_and_values_are_bounded(self):
        f = frame([("Home", 20.0, 0.0), ("Away", -20.0, 0.0)], with_velocity=True)
        x, y, z = pitch_control_surface(f, "Home", method=SPEARMAN, n_x=12, n_y=9)
        assert x.shape == y.shape == z.shape == (9, 12)
        assert ((z >= 0) & (z <= 1)).all()

    def test_attacking_half_is_more_attacking_controlled(self):
        f = frame([("Home", 20.0, 0.0), ("Away", -20.0, 0.0)], with_velocity=True)
        x, _, z = pitch_control_surface(f, "Home", method=SPEARMAN, n_x=20, n_y=11)
        assert z[:, x[0] > 10].mean() > z[:, x[0] < -10].mean()


class TestAddPitchControlFeatures:
    def _shots(self):
        f = frame([("Home", 40.0, 0.0), ("Away", 45.0, 1.0)], with_velocity=True)
        return pd.DataFrame(
            {"x": [40.0], "y": [0.0], "team": ["Home"], "frame_positions": [f]}
        )

    def test_adds_all_four_columns(self):
        out = add_pitch_control_features(self._shots())
        for column in (
            "pitch_control",
            "pitch_control_spearman",
            "pitch_control_at_shot",
            "pitch_control_spearman_at_shot",
        ):
            assert column in out.columns
            assert 0.0 <= out[column].iloc[0] <= 1.0

    def test_missing_columns_raise_clearly(self):
        with pytest.raises(ValueError, match="missing columns"):
            add_pitch_control_features(pd.DataFrame({"x": [1.0]}))

    def test_does_not_mutate_input(self):
        shots = self._shots()
        add_pitch_control_features(shots)
        assert "pitch_control" not in shots.columns
