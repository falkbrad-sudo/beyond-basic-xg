"""Figures for the StatsBomb validation gate (src/models/basic_xg_model.py).

This is the panel that establishes the baseline: before any
pitch-control claim, it shows whether a distance/angle-only baseline tracks
StatsBomb's own published xG across thousands of real shots, and where it
does not.

Encoding
--------
Both figures compare a model against a reference, so both carry a y = x
reference line in muted ink: chrome, not a series. The agreement plot bins
~3.5k shots into hexagons because at that density individual dots overplot
into a solid mass that hides the distribution; bin counts are a magnitude, so
they get the single-hue sequential ramp. The calibration plot has ten points
and one series, so it needs no legend: the title names it.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.viz.style import (
    DENSITY_CMAP,
    FIGURE_DPI,
    INK_MUTED,
    INK_SECONDARY,
    SERIES_1,
    SURFACE,
    add_titles,
    apply_base_style,
)


def plot_xg_agreement(
    shots: pd.DataFrame,
    own_col: str = "own_xg",
    reference_col: str = "statsbomb_xg",
    title: str | None = None,
    subtitle: str | None = None,
) -> plt.Figure:
    """Own basic xG against StatsBomb's published xG, one hexagon per bin.

    Parameters
    ----------
    shots : pd.DataFrame
        Must contain own_col and reference_col.
    own_col, reference_col : str
    title, subtitle : str, optional

    Returns
    -------
    plt.Figure

    Notes
    -----
    The y = x line is the thing to read against. Mass sitting below it on the
    right-hand side means this model rates those chances lower than StatsBomb
    does. That is the main result of the validation step, so the line is
    labelled directly rather than left to a legend.
    """
    missing = [c for c in (own_col, reference_col) if c not in shots.columns]
    if missing:
        raise ValueError(f"shots is missing columns: {missing}")

    data = shots[[own_col, reference_col]].dropna()
    upper = float(max(data[own_col].max(), data[reference_col].max())) * 1.04

    # Sized so a square (equal-aspect) plot plus its colorbar fills the
    # figure; tight_layout cannot do this itself once the aspect is locked,
    # and leaves a band of dead space instead.
    fig, ax = plt.subplots(figsize=(7.4, 6.9), dpi=FIGURE_DPI)
    fig.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.10, right=0.86, top=0.80, bottom=0.09)
    apply_base_style(ax)

    hexes = ax.hexbin(
        data[reference_col],
        data[own_col],
        gridsize=38,
        extent=(0, upper, 0, upper),
        mincnt=1,
        cmap=DENSITY_CMAP,
        bins="log",
        linewidths=0.2,
        edgecolors=SURFACE,
        zorder=2,
    )

    ax.plot(
        [0, upper], [0, upper],
        color=INK_MUTED, linewidth=1.5, linestyle="-", zorder=3,
    )
    ax.annotate(
        "equal xG",
        xy=(upper * 0.82, upper * 0.82),
        xytext=(upper * 0.60, upper * 0.90),
        color=INK_SECONDARY,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 1.0},
    )

    bar = fig.colorbar(hexes, ax=ax, shrink=0.72, pad=0.02)
    bar.set_label("Shots per bin", color=INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    bar.outline.set_visible(False)

    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal")
    ax.set_xlabel("StatsBomb published xG", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("This project's basic xG", color=INK_SECONDARY, fontsize=10)

    add_titles(fig, title, subtitle)
    return fig


def plot_calibration(
    calibration: pd.DataFrame,
    title: str | None = None,
    subtitle: str | None = None,
) -> plt.Figure:
    """Mean predicted probability against observed goal rate, by decile.

    Parameters
    ----------
    calibration : pd.DataFrame
        Output of basic_xg_model._calibration_table: needs 'mean_predicted',
        'observed_goal_rate' and 'n_shots'.
    title, subtitle : str, optional

    Returns
    -------
    plt.Figure

    Notes
    -----
    A correlation says a model ranks shots correctly; this says whether its
    probabilities are on the right scale. Points on the line mean a shot the
    model calls 20% is scored about 20% of the time.
    """
    required = ["mean_predicted", "observed_goal_rate"]
    missing = [c for c in required if c not in calibration.columns]
    if missing:
        raise ValueError(f"calibration table is missing columns: {missing}")

    upper = float(
        max(calibration["mean_predicted"].max(), calibration["observed_goal_rate"].max())
    ) * 1.12

    fig, ax = plt.subplots(figsize=(6.6, 6.6), dpi=FIGURE_DPI)
    fig.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.79, bottom=0.10)
    apply_base_style(ax)

    ax.plot(
        [0, upper], [0, upper],
        color=INK_MUTED, linewidth=1.5, zorder=2,
    )
    ax.annotate(
        "perfect calibration",
        xy=(upper * 0.70, upper * 0.70),
        xytext=(upper * 0.38, upper * 0.86),
        color=INK_SECONDARY,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 1.0},
    )

    ax.plot(
        calibration["mean_predicted"],
        calibration["observed_goal_rate"],
        color=SERIES_1,
        linewidth=2.0,
        marker="o",
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=1.6,
        zorder=4,
    )

    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted xG in decile", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Observed goal rate in decile", color=INK_SECONDARY, fontsize=10)

    add_titles(fig, title, subtitle)
    return fig
