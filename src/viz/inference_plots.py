"""Figures for uncertainty and statistical power.

These carry the part of the argument that point estimates cannot: how much
the data actually pins down, and how much data the question needs.

Encoding
--------
The coefficient figure compares two fits of the same features, so the two
samples are an identity and get two categorical hues, plus direct labels so
identity never rests on colour alone. Zero is chrome, not a series: it is
drawn in muted ink.

The power figure uses EMPHASIS rather than a full categorical palette: one
highlighted curve for the effect actually observed, the rest in de-emphasis
grey. The story is "this one line", and giving every line its own hue would
bury it.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.power import power_at_sample_size
from src.viz.style import (
    BLUE_RAMP,
    FIGURE_DPI,
    INK_MUTED,
    INK_SECONDARY,
    SERIES_1,
    SURFACE,
    add_titles,
    apply_base_style,
)

# Categorical slots 1 and 2, the validated opening pair.
SMALL_SAMPLE_COLOR = "#eb6834"
LARGE_SAMPLE_COLOR = SERIES_1

# De-emphasis grey for context curves. Deliberately darker than the
# gridlines, so a context line never reads as chart furniture.
CONTEXT_LINE = "#c3c2b7"


def plot_coefficient_intervals(
    intervals: dict[str, pd.DataFrame],
    title: str | None = None,
    subtitle: str | None = None,
) -> plt.Figure:
    """Forest plot of bootstrap coefficient intervals, one colour per sample.

    Parameters
    ----------
    intervals : dict[str, pd.DataFrame]
        Label -> table from uncertainty.bootstrap_coefficients. Each needs
        'feature', 'coefficient', 'ci_low', 'ci_high'.
    title, subtitle : str, optional

    Returns
    -------
    plt.Figure
    """
    labels = list(intervals)
    colors = [SMALL_SAMPLE_COLOR, LARGE_SAMPLE_COLOR][: len(labels)]

    features: list[str] = []
    for table in intervals.values():
        for feature in table["feature"]:
            if feature not in features:
                features.append(feature)
    features = features[::-1]

    fig, ax = plt.subplots(figsize=(9.6, 0.95 * len(features) + 3.8), dpi=FIGURE_DPI)
    fig.set_facecolor(SURFACE)
    # Generous bottom margin: the legend sits below the axes and identifying
    # which sample each colour belongs to is not optional.
    fig.subplots_adjust(left=0.25, right=0.97, top=0.74, bottom=0.21)
    apply_base_style(ax, grid_axis="x")

    offsets = np.linspace(0.18, -0.18, len(labels)) if len(labels) > 1 else [0.0]

    for (label, table), color, offset in zip(intervals.items(), colors, offsets):
        indexed = table.set_index("feature")
        for row_index, feature in enumerate(features):
            if feature not in indexed.index:
                continue
            row = indexed.loc[feature]
            y = row_index + offset
            ax.plot(
                [row["ci_low"], row["ci_high"]], [y, y],
                color=color, linewidth=2.4, solid_capstyle="round", zorder=3,
            )
            ax.plot(
                [row["coefficient"]], [y], marker="o", markersize=9,
                color=color, markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4,
            )
        ax.plot([], [], color=color, linewidth=2.4, marker="o", markersize=8,
                label=label)

    # Zero is the reference the whole figure is read against.
    ax.axvline(0.0, color=INK_MUTED, linewidth=1.5, zorder=2)

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([f.replace("_", " ") for f in features], fontsize=10)
    ax.tick_params(axis="y", colors=INK_SECONDARY, length=0)
    ax.set_ylim(-0.6, len(features) - 0.4)
    ax.set_xlabel(
        "Coefficient (log-odds per standard deviation), 95% bootstrap interval",
        color=INK_SECONDARY, fontsize=10,
    )
    ax.grid(False, axis="y")

    legend = fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=1,
        frameon=False, fontsize=9.5, handletextpad=0.7,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    add_titles(fig, title, subtitle, width=76, subtitle_width=104)
    return fig


def plot_power_curve(
    se_reference: float,
    n_reference: int,
    observed_effect: float,
    context_effects: tuple[float, ...] = (0.05, 0.10, 0.20),
    goal_rate: float = 0.086,
    sample_markers: dict[str, tuple[int, int]] | None = None,
    target_power: float = 0.80,
    title: str | None = None,
    subtitle: str | None = None,
) -> plt.Figure:
    """Power against sample size, highlighting the effect actually observed.

    Parameters
    ----------
    se_reference : float
        Standard error measured at n_reference.
    n_reference : int
    observed_effect : float
        The effect size measured on the large sample, drawn as the emphasised curve.
    context_effects : tuple[float, ...]
        Smaller effects drawn in grey for scale.
    goal_rate : float
        Used for the secondary "goals" axis.
    sample_markers : dict[str, tuple[int, int]], optional
        Label -> (shots, goals), drawn as vertical reference lines. Goals
        are passed explicitly rather than derived from goal_rate, because
        the two samples have different conversion rates and deriving them
        would mislabel one of them.
    target_power : float
    title, subtitle : str, optional

    Returns
    -------
    plt.Figure
    """
    sizes = np.logspace(1.3, 4.6, 400)

    fig, ax = plt.subplots(figsize=(8.6, 6.0), dpi=FIGURE_DPI)
    fig.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.74, bottom=0.14)
    apply_base_style(ax)

    for effect in context_effects:
        ax.plot(
            sizes, power_at_sample_size(effect, se_reference, n_reference, sizes),
            color=CONTEXT_LINE, linewidth=1.8, zorder=2,
        )
        ax.annotate(
            f"{effect:.2f}",
            xy=(sizes[-1], power_at_sample_size(
                effect, se_reference, n_reference, sizes[-1])),
            xytext=(4, 0), textcoords="offset points",
            color=INK_MUTED, fontsize=8.5, va="center",
        )

    ax.plot(
        sizes, power_at_sample_size(observed_effect, se_reference, n_reference, sizes),
        color=SERIES_1, linewidth=2.6, zorder=4,
    )
    ax.annotate(
        f"observed effect  {observed_effect:.2f}",
        xy=(sizes[-1], 1.0), xytext=(-6, -14), textcoords="offset points",
        color=SERIES_1, fontsize=9.5, ha="right", fontweight="bold",
    )

    ax.axhline(target_power, color=INK_MUTED, linewidth=1.3, linestyle="-", zorder=3)
    ax.annotate(
        f"{target_power:.0%} power", xy=(sizes[0], target_power),
        xytext=(2, 5), textcoords="offset points",
        color=INK_SECONDARY, fontsize=9,
    )

    for label, (n, goals) in (sample_markers or {}).items():
        ax.axvline(n, color=BLUE_RAMP[250], linewidth=1.3, zorder=2)
        ax.annotate(
            f"{label}\n{n:,} shots\n{goals:,} goals",
            xy=(n, 0.03), xytext=(5, 0), textcoords="offset points",
            color=INK_SECONDARY, fontsize=8.5, va="bottom",
        )

    ax.set_xscale("log")
    ax.set_xlim(sizes[0], sizes[-1])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Shots in the sample (log scale)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Power to detect the effect", color=INK_SECONDARY, fontsize=10)

    add_titles(fig, title, subtitle, width=72, subtitle_width=100)
    return fig
