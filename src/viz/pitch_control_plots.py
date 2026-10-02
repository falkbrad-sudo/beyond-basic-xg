"""Visualize the pitch-control surface at the moment of a specific shot.

The most illustrative single figure for explaining WHY the enhanced model
scored a given shot differently than the basic one. Pairs well with
find_biggest_disagreements() output in the evaluate module.

Encoding
--------
Pitch control is a polarity, not a magnitude: 0.5 is contested, and the two
directions mean opposite things. So the surface uses a diverging scale
(style.CONTROL_CMAP) with a neutral midpoint, and the player markers take the
same two poles so the players and the surface they generate read as one
encoding. Teams are additionally separated by marker shape, so which side is
which never depends on colour alone.

Credit for the control model itself (Spearman 2018, adapted from the
"Friends of Tracking" reference implementation) is in
src/features/pitch_control_at_shot.py, which computes the surface this module
only draws.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch

from src.data.cleaning import PITCH_LENGTH_M, PITCH_WIDTH_M
from src.features.geometry import GOAL_POST_Y, GOAL_X
from src.features.pitch_control_at_shot import SPEARMAN, pitch_control_surface
from src.viz.style import (
    ATTACK_COLOR,
    BALL_COLOR,
    CONTROL_CMAP,
    DEFEND_COLOR,
    INK_MUTED,
    INK_SECONDARY,
    SURFACE,
    add_titles,
)

BALL = "ball"
ATTACK_MARKER = "o"
DEFEND_MARKER = "s"


def _pitch() -> Pitch:
    return Pitch(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH_M,
        pitch_width=PITCH_WIDTH_M,
        # Cropped to the attacking half: at full length the shot is a speck
        # and the half that decides it is unreadable. Players behind the
        # halfway line barely affect control near the goal anyway.
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


def plot_pitch_control_at_shot(
    frame_positions: pd.DataFrame,
    shot_x: float,
    shot_y: float,
    title: str | None = None,
    subtitle: str | None = None,
    shooting_team: str | None = None,
    method: str = SPEARMAN,
    show_lane: bool = True,
) -> plt.Figure:
    """Draw all players' positions plus a pitch-control overlay at
    the moment of one specific shot, with the shot location marked.

    Parameters
    ----------
    frame_positions : pd.DataFrame
        Output of metrica_loader.get_normalized_tracking_frame_at_shot():
        meters, attacking frame, with 'player', 'team', 'x', 'y' and
        optionally 'vx', 'vy'.
    shot_x, shot_y : float
        Shot location in the same frame.
    title, subtitle : str, optional
    shooting_team : str, optional
        'Home' or 'Away'. Inferred from whichever team's player is nearest
        the shot location when omitted, since that player is the shooter.
    method : str
        'spearman' (default) renders a smooth surface; 'voronoi' renders hard
        territory boundaries. The default is the soft model because a
        heatmap of flat polygons carries less information, even though
        Voronoi is what the primary model feature uses.
    show_lane : bool
        Outline the triangle from the shot to the two goalposts, the region
        whose control is the model's actual pitch-control feature.

    Returns
    -------
    plt.Figure
    """
    players = frame_positions[frame_positions["team"] != BALL]
    if shooting_team is None:
        distances = np.hypot(players["x"] - shot_x, players["y"] - shot_y)
        shooting_team = players.loc[distances.idxmin(), "team"]

    grid_x, grid_y, control = pitch_control_surface(
        frame_positions, shooting_team, method=method, x_range=(0.0, GOAL_X)
    )

    pitch = _pitch()
    fig = plt.figure(figsize=(9.6, 6.6))
    fig.set_facecolor(SURFACE)
    grid = fig.add_gridspec(
        1, 2, width_ratios=[1, 0.035], wspace=0.08,
        left=0.03, right=0.90, top=0.78, bottom=0.08,
    )
    ax = fig.add_subplot(grid[0])
    pitch.draw(ax=ax)

    mesh = ax.pcolormesh(
        grid_x + PITCH_LENGTH_M / 2,
        grid_y + PITCH_WIDTH_M / 2,
        control,
        cmap=CONTROL_CMAP,
        vmin=0.0,
        vmax=1.0,
        shading="auto",
        alpha=0.75,
        zorder=1,
    )

    if show_lane:
        lane = np.array(
            [
                [shot_x + PITCH_LENGTH_M / 2, shot_y + PITCH_WIDTH_M / 2],
                [GOAL_X + PITCH_LENGTH_M / 2, GOAL_POST_Y + PITCH_WIDTH_M / 2],
                [GOAL_X + PITCH_LENGTH_M / 2, -GOAL_POST_Y + PITCH_WIDTH_M / 2],
            ]
        )
        ax.add_patch(
            plt.Polygon(
                lane, closed=True, fill=False,
                edgecolor=INK_SECONDARY, linewidth=1.4, linestyle="--", zorder=4,
            )
        )

    for is_attacking, marker, color in (
        (True, ATTACK_MARKER, ATTACK_COLOR),
        (False, DEFEND_MARKER, DEFEND_COLOR),
    ):
        mask = players["team"] == shooting_team
        subset = players[mask if is_attacking else ~mask]
        if subset.empty:
            continue
        ax.scatter(
            subset["x"] + PITCH_LENGTH_M / 2,
            subset["y"] + PITCH_WIDTH_M / 2,
            s=150, marker=marker, c=color,
            edgecolors=SURFACE, linewidths=1.8, zorder=5,
        )

    ball = frame_positions[frame_positions["team"] == BALL]
    if not ball.empty:
        ax.scatter(
            ball["x"] + PITCH_LENGTH_M / 2,
            ball["y"] + PITCH_WIDTH_M / 2,
            s=55, marker="o", c=BALL_COLOR,
            edgecolors=SURFACE, linewidths=1.4, zorder=6,
        )

    ax.scatter(
        [shot_x + PITCH_LENGTH_M / 2], [shot_y + PITCH_WIDTH_M / 2],
        s=460, marker="*", c="#ffffff",
        edgecolors=BALL_COLOR, linewidths=1.6, zorder=7,
    )

    legend = ax.legend(
        handles=[
            plt.Line2D([], [], marker=ATTACK_MARKER, linestyle="none", markersize=10,
                       markerfacecolor=ATTACK_COLOR, markeredgecolor=SURFACE,
                       label="Attacking"),
            plt.Line2D([], [], marker=DEFEND_MARKER, linestyle="none", markersize=10,
                       markerfacecolor=DEFEND_COLOR, markeredgecolor=SURFACE,
                       label="Defending"),
            plt.Line2D([], [], marker="*", linestyle="none", markersize=15,
                       markerfacecolor="#ffffff", markeredgecolor=BALL_COLOR,
                       label="Shot"),
            plt.Line2D([], [], linestyle="--", color=INK_SECONDARY, linewidth=1.4,
                       label="Shooting lane"),
        ],
        loc="lower center", ncol=4, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, -0.07), handletextpad=0.4, columnspacing=1.4,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    cax = fig.add_subplot(grid[1])
    bar = fig.colorbar(mesh, cax=cax, ticks=[0.0, 0.5, 1.0])
    bar.ax.set_yticklabels(["Defence", "Contested", "Attack"])
    bar.set_label("Who controls the space", color=INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8.5)
    bar.outline.set_visible(False)

    add_titles(fig, title, subtitle)
    return fig
