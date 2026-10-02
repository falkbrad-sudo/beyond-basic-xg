"""Shared cleaning and coordinate-conversion utilities.

There are TWO frames in play here, and the distinction matters:

1. **Pitch frame**: what `metrica_to_meters` / `statsbomb_to_meters` return.
   Units meters, origin at the center of the pitch, x in [-52.5, +52.5] for a
   105m pitch, y in [-34, +34] for a 68m pitch. This frame is fixed to the
   stadium: it says nothing about which goal anyone is attacking.

2. **Attacking frame**: what `normalize_playing_direction` returns, and what
   every module past this one assumes. Same units and origin, but x is
   flipped where needed so the goal being shot at is ALWAYS at x = +52.5.

The distinction exists because `src/features/geometry.py` hardcodes
`GOAL_X = +52.5`. In Metrica tracking data the home and away teams attack
opposite goals and both swap ends at halftime, so roughly three quarters of
shots are taken toward x = -52.5 in the pitch frame. Feeding those straight
into `distance_to_goal` measures the distance to the *wrong goal*, and the
answer looks perfectly plausible, which is what makes it dangerous. Call
`normalize_playing_direction` after converting units and before computing any
geometry feature.

StatsBomb event data needs no such flip: its x axis already points at the
goal the shooting team is attacking, so the pitch and attacking frames
coincide. See `statsbomb_to_meters`.
"""
from __future__ import annotations

import pandas as pd

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GOAL_WIDTH_M = 7.32  # standard goal width, needed for shot-angle calculations

STATSBOMB_LENGTH_UNITS = 120.0
STATSBOMB_WIDTH_UNITS = 80.0


def metrica_to_meters(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Convert Metrica's normalized (0-1) coordinates to standard meters."""
    out = df.copy()
    out[x_col] = (out[x_col] - 0.5) * PITCH_LENGTH_M
    out[y_col] = (out[y_col] - 0.5) * PITCH_WIDTH_M
    return out


def statsbomb_to_meters(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Convert StatsBomb's 120x80 pitch-unit coordinates to standard meters."""
    out = df.copy()
    out[x_col] = (out[x_col] / STATSBOMB_LENGTH_UNITS - 0.5) * PITCH_LENGTH_M
    out[y_col] = (out[y_col] / STATSBOMB_WIDTH_UNITS - 0.5) * PITCH_WIDTH_M
    return out


def validate_coordinates(df: pd.DataFrame, x_col: str, y_col: str) -> None:
    """Raise if any (x, y) pair falls meaningfully outside pitch bounds."""
    margin = 1.0
    bad_x = (df[x_col].abs() > PITCH_LENGTH_M / 2 + margin).any()
    bad_y = (df[y_col].abs() > PITCH_WIDTH_M / 2 + margin).any()
    if bad_x or bad_y:
        raise ValueError(
            f"Coordinates in columns '{x_col}'/'{y_col}' fall outside "
            f"expected pitch bounds after conversion. Check units."
        )


def normalize_playing_direction(
    df: pd.DataFrame,
    attacking_right: pd.Series | bool,
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """Flip coordinates so the attacking goal is always at x = +52.5.

    Converts from the pitch frame to the attacking frame (see this module's
    docstring). Rows already attacking toward +x are left untouched; rows
    attacking toward -x have both x and y negated.

    Both axes are negated rather than x alone, because negating x by itself is
    a mirror reflection: it reverses handedness, so a defender on the
    shooter's left would come out on the right. Negating both is a 180-degree
    rotation about the center spot, which preserves the left/right
    relationships that the pitch-control and visualization work depends on.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain x_col and y_col, already in meters (pitch frame).
    attacking_right : pd.Series or bool
        True where the row is already attacking toward x = +52.5. A bool
        applies to every row; a Series is aligned per row, which is the usual
        case (direction depends on both team and period).
    x_col, y_col : str
        Columns to flip. Defaults to 'x'/'y'.

    Returns
    -------
    pd.DataFrame
        Copy of df with x_col, y_col in the attacking frame. Does not mutate
        the input.

    Notes
    -----
    Same idea as `to_single_playing_direction` in the "Friends of Tracking"
    reference implementation
    (https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking),
    generalized to take the direction per row rather than assuming it from
    the period alone.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [-40.0], "y": [10.0]})
    >>> float(normalize_playing_direction(df, attacking_right=False)["x"].iloc[0])
    40.0
    """
    out = df.copy()
    if isinstance(attacking_right, bool):
        flip = pd.Series(not attacking_right, index=out.index)
    else:
        flip = ~attacking_right.astype(bool)
        flip = flip.reindex(out.index, fill_value=False)
    out.loc[flip, x_col] = -out.loc[flip, x_col]
    out.loc[flip, y_col] = -out.loc[flip, y_col]
    return out
