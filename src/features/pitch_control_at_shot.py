"""Pitch control value at the exact location and moment of a shot.

Requires Metrica tracking data. This is the feature that makes the
"enhanced" model different from the basic one. It is never computed from
StatsBomb freeze frames, which are single instants with no velocities and
only the players in camera view (METHODOLOGY.md principle #3).

Credit
------
The time-to-intercept control model in `spearman_control_at_points` follows
the structure of Spearman's pitch-control model:

    W. Spearman, "Beyond Expected Goals", MIT Sloan Sports Analytics
    Conference 2018.
    https://www.sloansportsconference.com/wp-content/uploads/2018/02/2002.pdf

as implemented in Laurie Shaw's "Friends of Tracking" reference code:

    https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking

This is an **adaptation and a simplification of** that work, not a
from-scratch derivation and not a faithful reimplementation. It keeps the
constant-velocity-plus-reaction-time estimate of when each player could
reach a point, then weights every player by how soon they could arrive and
takes the attacking team's share of the total weight. It drops the ball
travel time and the full PPCF differential-equation integration of the
paper. Read it as "who would get to this point first, softened", not as
Spearman's model proper.

Why the model feature is lane control, not control at the shot point
-------------------------------------------------------------------
The obvious feature, pitch control evaluated at the shot's own coordinates,
is degenerate here, and it is worth being explicit about why rather than
shipping a feature that looks fine and carries no information. The shot
location IS the shooter's position (measured: a median of 0.24m apart across
the sample), so the nearest player to it is virtually always the shooter,
and the attacking team "controls" essentially every shot location. Measured
on the 66 shots the model is fit on, Voronoi control at the shot point is
1.0 for 65 of them. That is effectively a constant, and a constant is not
a feature. The value is still computed and kept as
'pitch_control_at_shot' so that claim stays auditable rather than asserted.

What does vary is how much of the path to goal the attacking team controls.
So the main feature is `shooting_lane_control`: the share of the
triangle between the shot location and the two goalposts that belongs to the
shooting team. That is the quantity the project's premise is actually about
(a covered lane with defenders closing versus a clear sight of goal), and it
is not degenerate: it spans 0.00 to 0.66 across the sample.

A caveat that turned out to matter: because that triangle IS the shot angle,
the feature is correlated with `angle_to_goal` at r = 0.35, so it partly
re-encodes information the basic model already has. That is the main reason
the enhanced model barely moves on this data; see the README's findings
section. A future version might define the lane relative to the goal mouth
only, or weight the far end of the cone more heavily, to reduce the overlap.

Voronoi is the primary method, per the tradeoff in this project's notes: on
a sample this small, a feature that is fully understood beats a more
sophisticated one that adds noise. The Spearman-style variant is computed
alongside it as a cross-check and is what the visualisations render.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.features.geometry import GOAL_POST_Y, GOAL_X

BALL = "ball"
VORONOI = "voronoi"
SPEARMAN = "spearman"
METHODS = (VORONOI, SPEARMAN)

# Spread of the logistic on arrival-time difference, in seconds. Spearman's
# sigma; 0.45s is the value used in the Friends of Tracking implementation.
ARRIVAL_TIME_SIGMA_S = 0.45

# Resolution of the grid laid over the shooting-lane triangle. 26x26 over a
# cone that is at most ~35m deep gives better than 1.5m spacing, which is
# finer than the tracking data's own positional accuracy.
LANE_GRID_N = 26


def _players_only(frame_positions: pd.DataFrame) -> pd.DataFrame:
    """Drop the ball row: pitch control is about players."""
    return frame_positions[frame_positions["team"] != BALL]


def _split_teams(
    frame_positions: pd.DataFrame, shooting_team: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition a frame into (attacking, defending) players."""
    players = _players_only(frame_positions)
    attacking = players[players["team"] == shooting_team]
    defending = players[players["team"] != shooting_team]
    if attacking.empty or defending.empty:
        raise ValueError(
            f"frame must contain players from both teams; got "
            f"{len(attacking)} attacking and {len(defending)} defending for "
            f"shooting_team={shooting_team!r}"
        )
    return attacking, defending


def _min_distance(players: pd.DataFrame, points: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest of these players, in meters."""
    positions = players[["x", "y"]].to_numpy(dtype=float)
    deltas = points[:, None, :] - positions[None, :, :]
    return np.sqrt((deltas**2).sum(axis=2)).min(axis=1)


def voronoi_control_at_points(
    frame_positions: pd.DataFrame, points: np.ndarray, shooting_team: str
) -> np.ndarray:
    """Hard Voronoi control: 1 where the nearest player is an attacker.

    Parameters
    ----------
    frame_positions : pd.DataFrame
        Players (and optionally the ball) with 'team', 'x', 'y' in meters in
        the attacking frame.
    points : np.ndarray
        Shape (n, 2) array of (x, y) query points in meters.
    shooting_team : str
        'Home' or 'Away'.

    Returns
    -------
    np.ndarray
        Shape (n,) of 0.0/1.0. Ties go to the defence, which is the
        conservative choice for a shot-quality feature.
    """
    attacking, defending = _split_teams(frame_positions, shooting_team)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return (
        _min_distance(attacking, points) < _min_distance(defending, points)
    ).astype(float)


def _time_to_intercept(
    players: pd.DataFrame,
    points: np.ndarray,
    reaction_time_s: float,
    max_speed_mps: float,
) -> np.ndarray:
    """Seconds for each player to reach each point, per Spearman's model.

    A player keeps their current velocity for `reaction_time_s`, then runs at
    `max_speed_mps` straight to the target. Shape (n_points, n_players).
    """
    positions = players[["x", "y"]].to_numpy(dtype=float)
    if {"vx", "vy"}.issubset(players.columns):
        velocities = players[["vx", "vy"]].to_numpy(dtype=float)
        velocities = np.nan_to_num(velocities)
    else:
        velocities = np.zeros_like(positions)

    reaction_positions = positions + velocities * reaction_time_s
    deltas = points[:, None, :] - reaction_positions[None, :, :]
    distances = np.sqrt((deltas**2).sum(axis=2))
    return reaction_time_s + distances / max_speed_mps


def spearman_control_at_points(
    frame_positions: pd.DataFrame,
    points: np.ndarray,
    shooting_team: str,
    reaction_time_s: float | None = None,
    max_speed_mps: float | None = None,
    sigma_s: float = ARRIVAL_TIME_SIGMA_S,
) -> np.ndarray:
    """Soft, velocity-aware control in [0, 1] from arrival-time difference.

    Each player is weighted by how soon they could reach the point, and the
    value is the attacking team's share of the total weight. A point the
    attack would reach well before the defence approaches 1, an even race
    gives 0.5, and extra players covering a space increase their team's
    share.

    Parameters
    ----------
    frame_positions : pd.DataFrame
        Needs 'team', 'x', 'y'; uses 'vx', 'vy' if present (and treats
        players as stationary if not).
    points : np.ndarray
        Shape (n, 2) in meters.
    shooting_team : str
    reaction_time_s, max_speed_mps : float, optional
        Default to config.yaml's pitch_control section.
    sigma_s : float
        Time constant in seconds for the arrival-time weighting: how much
        sooner a player must arrive to dominate the result.

    Returns
    -------
    np.ndarray
        Shape (n,) in [0, 1]: the shooting team's control.

    Notes
    -----
    A simplification of Spearman (2018). See this module's docstring for
    exactly what is and is not reproduced, and for the credit this is due.
    """
    if reaction_time_s is None or max_speed_mps is None:
        cfg = load_config()["pitch_control"]
        reaction_time_s = (
            cfg["reaction_time_s"] if reaction_time_s is None else reaction_time_s
        )
        max_speed_mps = (
            cfg["max_player_speed_mps"] if max_speed_mps is None else max_speed_mps
        )

    attacking, defending = _split_teams(frame_positions, shooting_team)
    points = np.atleast_2d(np.asarray(points, dtype=float))

    t_attack = _time_to_intercept(attacking, points, reaction_time_s, max_speed_mps)
    t_defend = _time_to_intercept(defending, points, reaction_time_s, max_speed_mps)

    # Weight every player by how soon they could arrive, then take the
    # attacking team's share. Summing over players rather than taking each
    # team's single fastest arrival matters twice over: two defenders
    # covering a space control it more firmly than one, and the
    # hard minimum produces spiky artefacts where the nearest player
    # changes, which a sum smooths out. Closer in spirit to Spearman's
    # formulation, which also aggregates across players.
    stabilizer = np.minimum(t_attack.min(axis=1), t_defend.min(axis=1))[:, None]
    weight_attack = np.exp(-(t_attack - stabilizer) / sigma_s).sum(axis=1)
    weight_defend = np.exp(-(t_defend - stabilizer) / sigma_s).sum(axis=1)
    return weight_attack / (weight_attack + weight_defend)


def control_at_points(
    frame_positions: pd.DataFrame,
    points: np.ndarray,
    shooting_team: str,
    method: str = VORONOI,
) -> np.ndarray:
    """Dispatch to the chosen control model. See METHODS."""
    if method == VORONOI:
        return voronoi_control_at_points(frame_positions, points, shooting_team)
    if method == SPEARMAN:
        return spearman_control_at_points(frame_positions, points, shooting_team)
    raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def pitch_control_at_point(
    frame_positions: pd.DataFrame,
    shot_x: float,
    shot_y: float,
    shooting_team: str,
    method: str = VORONOI,
) -> float:
    """Compute the shooting team's pitch control value at the shot location.

    Parameters
    ----------
    frame_positions : pd.DataFrame
        All players' positions (both teams) at the exact frame of the shot:
        the output of metrica_loader.get_normalized_tracking_frame_at_shot(),
        already converted to standard meter coordinates in the attacking
        frame.
    shot_x, shot_y : float
        The shot's location in the same frame.
    shooting_team : str
        'Home' or 'Away', whichever team took the shot.
    method : str
        'voronoi' (hard, 0/1) or 'spearman' (soft, continuous).

    Returns
    -------
    float
        Value in [0, 1] representing how much control the shooting team had
        at that exact point.

    Notes
    -----
    Retained for completeness and for the visualisations, but NOT the model
    feature: at the shot's own coordinates the shooter is the nearest player
    by construction, so this is ~1.0 for almost every shot. Use
    `shooting_lane_control` for modelling; see this module's docstring.
    """
    value = control_at_points(
        frame_positions, np.array([[shot_x, shot_y]]), shooting_team, method
    )
    return float(value[0])


def _lane_grid(shot_x: float, shot_y: float, n: int = LANE_GRID_N) -> np.ndarray:
    """Points inside the triangle from the shot to the two goalposts.

    Reuses the goalpost geometry that geometry.angle_to_goal is built on, so
    the "shooting lane" here is exactly the angle that model measures.
    """
    posts = np.array([[GOAL_X, GOAL_POST_Y], [GOAL_X, -GOAL_POST_Y]])
    apex = np.array([shot_x, shot_y])
    triangle = np.vstack([apex, posts])

    xs = np.linspace(triangle[:, 0].min(), triangle[:, 0].max(), n)
    ys = np.linspace(triangle[:, 1].min(), triangle[:, 1].max(), n)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)

    # Barycentric point-in-triangle test.
    a, b, c = triangle
    v0, v1, v2 = b - a, c - a, grid - a
    denominator = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(denominator) < 1e-12:
        return apex.reshape(1, 2)
    u = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / denominator
    v = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / denominator
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)
    return grid[inside] if inside.any() else apex.reshape(1, 2)


def shooting_lane_control(
    frame_positions: pd.DataFrame,
    shot_x: float,
    shot_y: float,
    shooting_team: str,
    method: str = VORONOI,
) -> float:
    """Share of the shot's path to goal that the shooting team controls.

    The triangle between the shot location and the two goalposts is sampled
    on a grid and the control model evaluated at each point; the result is
    the mean. 1.0 is a completely uncontested sight of goal, 0.0 a lane owned
    entirely by the defence.

    Parameters
    ----------
    frame_positions : pd.DataFrame
        As for pitch_control_at_point.
    shot_x, shot_y : float
    shooting_team : str
    method : str
        'voronoi' or 'spearman'.

    Returns
    -------
    float
        Value in [0, 1].

    Notes
    -----
    This is the project's actual pitch-control feature, for the reason set
    out in the module docstring: control at the shot point alone is a
    constant on this data, while lane control varies and means something.

    The mean is unweighted, which is a deliberate simplification: points near
    the goal mouth arguably matter more than points at the shooter's feet,
    but any weighting scheme is a free parameter and this project has 9 goals
    to fit, so the version with no tunable knobs is the defensible one.
    """
    grid = _lane_grid(shot_x, shot_y)
    return float(control_at_points(frame_positions, grid, shooting_team, method).mean())


def pitch_control_surface(
    frame_positions: pd.DataFrame,
    shooting_team: str,
    method: str = SPEARMAN,
    n_x: int = 120,
    n_y: int = 80,
    x_range: tuple[float, float] = (-GOAL_X, GOAL_X),
    y_range: tuple[float, float] = (-34.0, 34.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the control model over the whole pitch, for plotting.

    Parameters
    ----------
    frame_positions : pd.DataFrame
    shooting_team : str
    method : str
        Defaults to 'spearman', because a hard Voronoi surface renders as flat
        polygons, which is less informative as a heatmap.
    n_x, n_y : int
        Grid resolution.
    x_range, y_range : tuple[float, float]

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (X, Y, Z) meshgrid arrays suitable for matplotlib's pcolormesh, with
        Z the shooting team's control in [0, 1].
    """
    xs = np.linspace(*x_range, n_x)
    ys = np.linspace(*y_range, n_y)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    values = control_at_points(frame_positions, points, shooting_team, method)
    return grid_x, grid_y, values.reshape(grid_x.shape)


def add_pitch_control_features(shots_with_frames: pd.DataFrame) -> pd.DataFrame:
    """Apply the control models across a DataFrame of shots.

    Parameters
    ----------
    shots_with_frames : pd.DataFrame
        One row per shot, with 'x', 'y' (meters, attacking frame), 'team',
        and a 'frame_positions' column holding that shot's normalized
        tracking frame (see src/pipeline.py, where the join happens).

    Returns
    -------
    pd.DataFrame
        Copy with four new columns:

        - 'pitch_control': Voronoi shooting-lane control. The model feature.
        - 'pitch_control_spearman': the same quantity, Spearman-style model.
          Carried as a cross-check, not fed to the primary model.
        - 'pitch_control_at_shot': Voronoi control at the shot point.
          Near-constant by construction; kept so that claim stays auditable.
        - 'pitch_control_spearman_at_shot': the soft equivalent.

    Notes
    -----
    A plain row-wise loop: the sample is 67 shots, so there is nothing to
    gain from vectorising across shots and a loop is easier to verify.
    """
    required = {"x", "y", "team", "frame_positions"}
    missing = required - set(shots_with_frames.columns)
    if missing:
        raise ValueError(f"shots_with_frames is missing columns: {sorted(missing)}")

    out = shots_with_frames.copy()
    records = []
    for row in out.itertuples():
        frame = row.frame_positions
        records.append(
            {
                "pitch_control": shooting_lane_control(
                    frame, row.x, row.y, row.team, VORONOI
                ),
                "pitch_control_spearman": shooting_lane_control(
                    frame, row.x, row.y, row.team, SPEARMAN
                ),
                "pitch_control_at_shot": pitch_control_at_point(
                    frame, row.x, row.y, row.team, VORONOI
                ),
                "pitch_control_spearman_at_shot": pitch_control_at_point(
                    frame, row.x, row.y, row.team, SPEARMAN
                ),
            }
        )
    computed = pd.DataFrame(records, index=out.index)
    for column in computed.columns:
        out[column] = computed[column]
    return out
