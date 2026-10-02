"""Tests for src/viz/shot_maps.py and src/viz/validation_plots.py.

Figures are hard to assert on pixel-for-pixel, so these check the things that
actually break in practice: the contract (required columns, clear errors),
the encoding decisions that carry meaning (shared colour scale across
panels, shape not colour for outcome), and that nothing raises on realistic
input. Visual correctness is checked by rendering and looking, with
scripts/render_figures.py.
"""
import os

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp") + "/mpl")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.viz.shot_maps import plot_comparison_shot_maps, plot_shot_map  # noqa: E402
from src.viz.style import GOAL_MARKER, MISS_MARKER  # noqa: E402
from src.viz.validation_plots import plot_calibration, plot_xg_agreement  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def shots():
    """A few shots in the project's attacking frame (meters, centre origin)."""
    return pd.DataFrame(
        {
            "x": [44.0, 38.0, 48.0, 30.0],
            "y": [-2.0, 10.0, 1.0, -8.0],
            "is_goal": [True, False, True, False],
            "basic_xg_pred": [0.20, 0.04, 0.30, 0.02],
            "enhanced_xg_pred": [0.35, 0.03, 0.26, 0.05],
        }
    )


@pytest.fixture
def calibration():
    return pd.DataFrame(
        {
            "n_shots": [100, 100, 100],
            "mean_predicted": [0.02, 0.09, 0.25],
            "observed_goal_rate": [0.03, 0.08, 0.26],
        }
    )


class TestPlotShotMap:
    def test_returns_a_figure(self, shots):
        assert isinstance(plot_shot_map(shots, color_by="basic_xg_pred"), plt.Figure)

    def test_missing_color_column_raises_clearly(self, shots):
        with pytest.raises(ValueError, match="no column"):
            plot_shot_map(shots, color_by="not_a_column")

    def test_outcome_uses_two_marker_shapes_not_two_colors(self, shots):
        # The accessibility decision: goal vs miss must survive greyscale and
        # colour-vision deficiency, so it rides on shape.
        fig = plot_shot_map(shots, color_by="basic_xg_pred")
        legend_markers = {
            handle.get_marker()
            for ax in fig.axes
            for legend in ([ax.get_legend()] if ax.get_legend() else [])
            for handle in legend.legend_handles
        }
        assert legend_markers == {GOAL_MARKER, MISS_MARKER}

    def test_works_without_an_is_goal_column(self, shots):
        fig = plot_shot_map(shots.drop(columns=["is_goal"]), color_by="basic_xg_pred")
        assert isinstance(fig, plt.Figure)

    def test_does_not_mutate_input(self, shots):
        before = shots.copy()
        plot_shot_map(shots, color_by="basic_xg_pred")
        pd.testing.assert_frame_equal(shots, before)


class TestPlotComparisonShotMaps:
    def test_returns_a_figure(self, shots):
        assert isinstance(plot_comparison_shot_maps(shots), plt.Figure)

    def test_missing_columns_raise_clearly(self, shots):
        with pytest.raises(ValueError, match="missing columns"):
            plot_comparison_shot_maps(shots, right_col="not_a_column")

    def test_both_panels_share_one_color_scale(self, shots):
        # Without a shared scale the same hue would mean different
        # probabilities in each panel and the comparison would be a lie.
        fig = plot_comparison_shot_maps(shots)
        collections = [c for ax in fig.axes for c in ax.collections if c.get_array() is not None]
        limits = {c.get_clim() for c in collections}
        assert len(limits) == 1
        expected = (
            min(shots.basic_xg_pred.min(), shots.enhanced_xg_pred.min()),
            max(shots.basic_xg_pred.max(), shots.enhanced_xg_pred.max()),
        )
        assert limits.pop() == pytest.approx(expected)


class TestValidationPlots:
    def test_agreement_returns_a_figure(self):
        shots = pd.DataFrame(
            {"own_xg": [0.1, 0.2, 0.05, 0.4], "statsbomb_xg": [0.12, 0.3, 0.04, 0.7]}
        )
        assert isinstance(plot_xg_agreement(shots), plt.Figure)

    def test_agreement_missing_columns_raise_clearly(self):
        with pytest.raises(ValueError, match="missing columns"):
            plot_xg_agreement(pd.DataFrame({"own_xg": [0.1]}))

    def test_calibration_returns_a_figure(self, calibration):
        assert isinstance(plot_calibration(calibration), plt.Figure)

    def test_calibration_missing_columns_raise_clearly(self):
        with pytest.raises(ValueError, match="missing columns"):
            plot_calibration(pd.DataFrame({"mean_predicted": [0.1]}))


class TestPitchControlPlot:
    """The figure that does the explaining the numbers cannot."""

    @pytest.fixture
    def frame_positions(self):
        rows = []
        for i in range(11):
            rows.append(
                {"player": f"h{i}", "team": "Home", "x": 20.0 + i, "y": -10.0 + i,
                 "vx": 1.0, "vy": 0.0}
            )
        for i in range(11):
            rows.append(
                {"player": f"a{i}", "team": "Away", "x": 40.0 + i * 0.5,
                 "y": 10.0 - i, "vx": -1.0, "vy": 0.0}
            )
        rows.append({"player": "ball", "team": "ball", "x": 30.0, "y": 0.0,
                     "vx": 0.0, "vy": 0.0})
        return pd.DataFrame(rows)

    def test_returns_a_figure(self, frame_positions):
        from src.viz.pitch_control_plots import plot_pitch_control_at_shot

        fig = plot_pitch_control_at_shot(
            frame_positions, shot_x=30.0, shot_y=0.0, shooting_team="Home"
        )
        assert isinstance(fig, plt.Figure)

    def test_infers_the_shooting_team_from_the_nearest_player(self, frame_positions):
        from src.viz.pitch_control_plots import plot_pitch_control_at_shot

        # Nearest player to (20, -10) is h0, so Home must be treated as the
        # attacking side without being told.
        fig = plot_pitch_control_at_shot(
            frame_positions, shot_x=20.0, shot_y=-10.0
        )
        assert isinstance(fig, plt.Figure)

    def test_both_control_methods_render(self, frame_positions):
        from src.features.pitch_control_at_shot import SPEARMAN, VORONOI
        from src.viz.pitch_control_plots import plot_pitch_control_at_shot

        for method in (SPEARMAN, VORONOI):
            fig = plot_pitch_control_at_shot(
                frame_positions, 30.0, 0.0, shooting_team="Home", method=method
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_surface_spans_the_full_control_range(self, frame_positions):
        from src.viz.pitch_control_plots import plot_pitch_control_at_shot

        fig = plot_pitch_control_at_shot(
            frame_positions, 30.0, 0.0, shooting_team="Home"
        )
        meshes = [c for ax in fig.axes for c in ax.collections if hasattr(c, "get_clim")]
        # Fixed 0-1 limits, so the diverging midpoint always means "contested"
        # rather than floating with whatever this frame happened to contain.
        assert (0.0, 1.0) in [m.get_clim() for m in meshes]
