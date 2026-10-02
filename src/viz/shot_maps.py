"""Shot maps: visualize shots on a pitch, colored/sized by model probability.

Uses mplsoccer for pitch drawing.

Encoding
--------
Shot quality is a magnitude, so it gets a single-hue light-to-dark ramp
(`style.XG_CMAP`). Outcome is an identity, so it gets marker *shape* (a star
for a goal, a circle for a miss), never colour alone. That keeps the two
readable at once, survives greyscale printing, and does not depend on the
viewer distinguishing two hues.

Coordinates are this project's standard attacking frame: meters, origin at the
pitch centre, attacking goal at x = +52.5 (see src/data/cleaning.py). mplsoccer
wants a 0-105 / 0-68 box, so the shift happens here and nowhere else.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch

from src.data.cleaning import PITCH_LENGTH_M, PITCH_WIDTH_M
from src.viz.style import (
    GOAL_MARKER,
    INK_MUTED,
    INK_SECONDARY,
    MISS_MARKER,
    SURFACE,
    XG_CMAP,
    add_titles,
)


def _new_pitch() -> Pitch:
    """A half pitch in the project's standard dimensions and visual style."""
    return Pitch(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH_M,
        pitch_width=PITCH_WIDTH_M,
        half=True,
        pitch_color=SURFACE,
        line_color=INK_MUTED,
        linewidth=1.0,
        goal_type="box",
        pad_left=2,
        pad_right=2,
        pad_bottom=2,
        pad_top=2,
    )


def _to_pitch_coords(shots: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Shift centre-origin meters into mplsoccer's 0-105 / 0-68 box."""
    return shots["x"] + PITCH_LENGTH_M / 2, shots["y"] + PITCH_WIDTH_M / 2


def _draw_shots(
    pitch: Pitch,
    ax: plt.Axes,
    shots: pd.DataFrame,
    color_by: str,
    vmin: float,
    vmax: float,
):
    """Scatter one set of shots, goals and misses as separate mark shapes."""
    x, y = _to_pitch_coords(shots)
    is_goal = shots["is_goal"].astype(bool) if "is_goal" in shots.columns else None

    groups = (
        [(slice(None), MISS_MARKER, 110, "Shot")]
        if is_goal is None
        else [
            (~is_goal, MISS_MARKER, 110, "No goal"),
            (is_goal, GOAL_MARKER, 320, "Goal"),
        ]
    )

    handle = None
    for mask, marker, size, _label in groups:
        subset_x, subset_y = x[mask], y[mask]
        if not len(subset_x):
            continue
        drawn = pitch.scatter(
            subset_x,
            subset_y,
            c=shots.loc[mask, color_by],
            cmap=XG_CMAP,
            vmin=vmin,
            vmax=vmax,
            marker=marker,
            s=size,
            # A 2px surface ring separates overlapping marks without drawing
            # a border around every one of them.
            edgecolors=SURFACE,
            linewidths=1.6,
            zorder=3,
            ax=ax,
        )
        handle = handle or drawn
    return handle


def _add_colorbar(fig: plt.Figure, handle, cax: plt.Axes) -> None:
    """Attach the xG colour scale to a colorbar axes from the gridspec."""
    bar = fig.colorbar(handle, cax=cax)
    bar.set_label("Predicted goal probability", color=INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    bar.outline.set_visible(False)


def _outcome_legend(ax: plt.Axes) -> None:
    """Legend for the shape channel, in ink rather than series colour."""
    proxies = [
        plt.Line2D(
            [], [], marker=GOAL_MARKER, linestyle="none", markersize=13,
            markerfacecolor=INK_MUTED, markeredgecolor=SURFACE, label="Goal",
        ),
        plt.Line2D(
            [], [], marker=MISS_MARKER, linestyle="none", markersize=8,
            markerfacecolor=INK_MUTED, markeredgecolor=SURFACE, label="No goal",
        ),
    ]
    legend = ax.legend(
        handles=proxies,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        handletextpad=0.4,
        columnspacing=1.6,
        bbox_to_anchor=(0.5, -0.08),
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)


def plot_shot_map(
    shots: pd.DataFrame,
    color_by: str = "enhanced_xg_pred",
    title: str | None = None,
    subtitle: str | None = None,
) -> plt.Figure:
    """Scatter shots on a pitch, colored by a chosen probability column.

    Parameters
    ----------
    shots : pd.DataFrame
        Must contain 'x', 'y' (standard meter coords, attacking frame) and the
        column named in color_by. An 'is_goal' column, if present, selects the
        marker shape.
    color_by : str
        Typically 'basic_xg_pred' or 'enhanced_xg_pred'. For a side-by-side
        view use plot_comparison_shot_maps, which puts two columns on a shared
        colour scale so the panels are actually comparable.
    title, subtitle : str, optional

    Returns
    -------
    plt.Figure
    """
    if color_by not in shots.columns:
        raise ValueError(f"shots has no column {color_by!r}")

    pitch = _new_pitch()
    # Gridspec rather than a colorbar stealing space from the pitch axes: a
    # pitch has a locked aspect ratio, and letting matplotlib fit a colorbar
    # beside it either overlaps the plot or clips the goal. Giving the bar its
    # own column leaves each axes free to honour its own aspect.
    fig = plt.figure(figsize=(7.2, 6.4))
    fig.set_facecolor(SURFACE)
    grid = fig.add_gridspec(
        1, 2, width_ratios=[1, 0.045], wspace=0.10,
        left=0.03, right=0.90, top=0.80, bottom=0.10,
    )
    ax = fig.add_subplot(grid[0])
    pitch.draw(ax=ax)

    vmin, vmax = float(shots[color_by].min()), float(shots[color_by].max())
    handle = _draw_shots(pitch, ax, shots, color_by, vmin, vmax)

    if handle is not None:
        _add_colorbar(fig, handle, fig.add_subplot(grid[1]))

    if "is_goal" in shots.columns:
        _outcome_legend(ax)

    add_titles(fig, title, subtitle)
    return fig


def plot_comparison_shot_maps(
    shots: pd.DataFrame,
    left_col: str = "basic_xg_pred",
    right_col: str = "enhanced_xg_pred",
    left_label: str = "Basic (geometry only)",
    right_label: str = "Enhanced (+ pitch control)",
    title: str | None = None,
) -> plt.Figure:
    """Side-by-side shot maps: basic_xg_pred vs. enhanced_xg_pred.

    The most useful single figure for the README and app. It lets a viewer see at a
    glance which shots moved the most between the two models.

    Both panels share one colour scale, spanning the combined range of the two
    columns. Without that they would be individually normalised and the
    comparison would be meaningless: the same hue would stand for different
    probabilities in each panel.

    Parameters
    ----------
    shots : pd.DataFrame
        Must contain 'x', 'y' and both named columns.
    left_col, right_col : str
        The two probability columns to compare. Defaults name the basic and
        enhanced models, but any two comparable columns work.
    left_label, right_label : str
        Panel headings.
    title : str, optional

    Returns
    -------
    plt.Figure
    """
    missing = [c for c in (left_col, right_col) if c not in shots.columns]
    if missing:
        raise ValueError(f"shots is missing columns: {missing}")

    # One scale across both panels, so a colour means the same thing in each.
    vmin = float(min(shots[left_col].min(), shots[right_col].min()))
    vmax = float(max(shots[left_col].max(), shots[right_col].max()))

    pitch = _new_pitch()
    # Three gridspec columns -- panel, panel, colorbar -- so the bar never
    # lands on top of the second pitch. See plot_shot_map for why this is not
    # left to matplotlib.
    fig = plt.figure(figsize=(11.6, 6.4))
    fig.set_facecolor(SURFACE)
    grid = fig.add_gridspec(
        1, 3, width_ratios=[1, 1, 0.045], wspace=0.12,
        left=0.03, right=0.92, top=0.78, bottom=0.11,
    )
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    for ax in axes:
        pitch.draw(ax=ax)

    handle = None
    for ax, column, label in zip(axes, (left_col, right_col), (left_label, right_label)):
        handle = _draw_shots(pitch, ax, shots, column, vmin, vmax) or handle
        ax.set_title(label, color=INK_SECONDARY, fontsize=10.5, pad=10)

    if handle is not None:
        _add_colorbar(fig, handle, fig.add_subplot(grid[2]))

    if "is_goal" in shots.columns:
        _outcome_legend(axes[0])

    add_titles(fig, title, subtitle=None, width=86)
    return fig
