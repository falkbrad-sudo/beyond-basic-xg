"""Shot geometry: distance and angle to goal. The basic xG feature set.

Pure deterministic geometry with known-correct answers for simple cases,
which makes it a good smoke test that an environment is set up correctly.

Standard coordinate convention (see cleaning.py): meters, origin at pitch
center, attacking goal at x = +52.5, goal centered at y = 0, goal width
7.32m (so goalposts at y = +3.66 and y = -3.66).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.cleaning import GOAL_WIDTH_M, PITCH_LENGTH_M

GOAL_X = PITCH_LENGTH_M / 2  # 52.5, the attacking goal line
GOAL_POST_Y = GOAL_WIDTH_M / 2  # 3.66


def distance_to_goal(x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    """Euclidean distance from a shot location to the center of the goal.

    Parameters
    ----------
    x, y : float or array-like
        Shot location(s) in standard meter coordinates.

    Returns
    -------
    float or np.ndarray
        Distance in meters.

    Examples
    --------
    >>> round(distance_to_goal(GOAL_X, 0), 6)  # standing on the goal line, centered
    0.0
    >>> round(distance_to_goal(0, 0), 2)  # center of the pitch
    52.5
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dist = np.sqrt((GOAL_X - x) ** 2 + y**2)
    return float(dist) if dist.ndim == 0 else dist


def angle_to_goal(x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    """Angle (in radians) subtended by the goal mouth from a shot location.

    This is the standard "shot angle" used in most public xG models: the
    angle between the lines from the shot location to each goalpost. A
    shot dead-center, close to goal, has a wide angle (near pi); a shot from
    a tight angle near the byline has a narrow one (near 0).

    Parameters
    ----------
    x, y : float or array-like
        Shot location(s) in standard meter coordinates. Should have x < GOAL_X
        (i.e., taken from before the goal line) for the geometry to make
        sense as a real shot.

    Returns
    -------
    float or np.ndarray
        Angle in radians, always non-negative. Returns a scalar float if the
        input was scalar, otherwise an array matching the input shape.

    Examples
    --------
    >>> round(angle_to_goal(GOAL_X - 11, 0), 4)  # penalty spot, dead center
    0.6424
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    scalar_input = x_arr.ndim == 0
    x_flat = np.atleast_1d(x_arr)
    y_flat = np.atleast_1d(y_arr)

    dx = GOAL_X - x_flat
    y_near = GOAL_POST_Y - y_flat
    y_far = -GOAL_POST_Y - y_flat

    dot = dx * dx + y_near * y_far
    norm_near = np.sqrt(dx**2 + y_near**2)
    norm_far = np.sqrt(dx**2 + y_far**2)
    cos_angle = np.clip(dot / (norm_near * norm_far), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return float(angle[0]) if scalar_input else angle


def add_geometry_features(shots: pd.DataFrame, x_col: str = "x", y_col: str = "y") -> pd.DataFrame:
    """Add distance_to_goal and angle_to_goal columns to a shots DataFrame.

    Parameters
    ----------
    shots : pd.DataFrame
        Must contain x_col, y_col in standard meter coordinates.
    x_col, y_col : str

    Returns
    -------
    pd.DataFrame
        Copy of shots with two new columns: 'distance_to_goal',
        'angle_to_goal'. These are the two features the basic xG model
        (src/models/basic_xg_model.py) is fit on.
    """
    out = shots.copy()
    out["distance_to_goal"] = distance_to_goal(out[x_col].values, out[y_col].values)
    out["angle_to_goal"] = angle_to_goal(out[x_col].values, out[y_col].values)
    return out
