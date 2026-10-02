"""Additional shot-context features beyond raw pitch control: defenders
between the ball and goal, and goalkeeper position.

REQUIRES Metrica tracking data, same as pitch_control_at_shot.py. These are
simpler, more interpretable complements to the pitch-control value, useful
both as standalone features and as a sanity check on pitch control (if a
shot has zero defenders between ball and goal but a low computed pitch-
control value, that's worth investigating as a possible bug before trusting
the number).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.geometry import GOAL_POST_Y, GOAL_X

BALL = "ball"

# StatsBomb's freeze frames name the playing position outright, so the
# keeper never has to be guessed on that data.
GOALKEEPER_POSITION = "Goalkeeper"


def _outfield(frame_positions: pd.DataFrame, team: str) -> pd.DataFrame:
    """One team's tracked players, ball row excluded."""
    players = frame_positions[frame_positions["team"] == team]
    return players[players["player"] != BALL]


def identify_goalkeeper(
    frame_positions: pd.DataFrame, team: str, goal_x: float = GOAL_X
) -> pd.Series:
    """Return the row of `team` most likely to be the goalkeeper.

    Uses an explicit role when the data has one and falls back to geometry
    when it does not:

    - If the frame carries a 'position' column (StatsBomb freeze frames do),
      the player whose position is 'Goalkeeper' is used. Exact, no guessing.
    - Otherwise (Metrica's sample data has no role labels at all), the
      keeper is inferred as the player closest to the goal their team is
      defending, the same heuristic metrica_loader.playing_direction uses.

    Parameters
    ----------
    frame_positions : pd.DataFrame
        A single tracking frame with 'player', 'team', 'x', 'y' in meters in
        the attacking frame.
    team : str
        The DEFENDING team, the one whose keeper is wanted.
    goal_x : float
        x of the goal that team is defending. In the attacking frame this is
        +52.5 for the defending side, which is the default.

    Returns
    -------
    pd.Series
        The goalkeeper's row.

    Raises
    ------
    ValueError
        If the team has no tracked players in this frame.

    Notes
    -----
    The geometric fallback is a heuristic and is verifiable: across the
    Metrica sample the inferred keeper averages ~0.12 or ~0.88 in normalized
    pitch units, i.e. right on a goal line and far from any outfield player,
    and the same inference cross-checks successfully against Metrica's own
    stated attack directions for game 3. It would misfire in the rare frame
    where a keeper is well upfield (a late corner, say), so treat a keeper
    distance of 20m+ as a flag to look at the frame rather than as a fact.
    On StatsBomb data the labelled path is used and this does not apply.
    """
    players = _outfield(frame_positions, team)
    if players.empty:
        raise ValueError(f"no tracked players for team {team!r} in this frame")

    if "position" in players.columns:
        keepers = players[players["position"] == GOALKEEPER_POSITION]
        if not keepers.empty:
            return keepers.iloc[0]

    return players.loc[(players["x"] - goal_x).abs().idxmin()]


def count_defenders_in_cone(
    frame_positions: pd.DataFrame,
    shot_x: float,
    shot_y: float,
    defending_team: str,
    goal_x: float = GOAL_X,
) -> int:
    """Count defenders positioned between the shot location and the goal.

    A simple, interpretable proxy for "was the shot blocked/contested" that
    doesn't require the full pitch-control machinery. Defined as: defenders
    whose position falls within the triangular cone from the shot location
    to both goalposts, and who are closer to goal than the shooter.

    Parameters
    ----------
    frame_positions : pd.DataFrame
        Same input as pitch_control_at_shot.pitch_control_at_point.
    shot_x, shot_y : float
    defending_team : str
        Whichever team did NOT take the shot.
    goal_x : float
        x-coordinate of the goal line being shot at. In this project's
        attacking frame that is always +52.5 (the default), but it stays an
        explicit argument because which goal is "attacking" depends on team
        and half in the raw data.

    Returns
    -------
    int
        Count of defenders in the cone. Likely single digits (0-3) for
        almost every shot. Small numbers are realistic.

    Notes
    -----
    The cone edges are the lines from the shot to each goalpost, which is the
    same geometry geometry.angle_to_goal measures, so this feature and the
    shot angle are describing the same region of the pitch. The goalkeeper is
    included in the count: a keeper in the lane does block the
    shot, and excluding them would make the feature disagree with what the
    pitch-control value sees.
    """
    defenders = _outfield(frame_positions, defending_team)
    if defenders.empty:
        return 0

    apex = np.array([shot_x, shot_y], dtype=float)
    post_a = np.array([goal_x, GOAL_POST_Y], dtype=float)
    post_b = np.array([goal_x, -GOAL_POST_Y], dtype=float)
    points = defenders[["x", "y"]].to_numpy(dtype=float)

    # Barycentric coordinates of each defender in the shot/post triangle.
    v0, v1 = post_a - apex, post_b - apex
    v2 = points - apex
    denominator = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(denominator) < 1e-12:
        return 0
    u = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / denominator
    v = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / denominator
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)

    # "Closer to goal than the shooter", with the goal at +x.
    beyond_shooter = (
        points[:, 0] > shot_x if goal_x > shot_x else points[:, 0] < shot_x
    )
    return int((inside & beyond_shooter).sum())


def goalkeeper_distance_from_goal_center(
    frame_positions: pd.DataFrame,
    defending_team: str,
    goal_x: float = GOAL_X,
) -> float:
    """Distance of the defending goalkeeper from the center of their goal.

    A stranded/off-position goalkeeper is a classic reason a "low basic xG"
    shot is actually a great chance, so it is worth having as its own explicit
    feature rather than hoping pitch control alone captures it.

    Parameters
    ----------
    frame_positions : pd.DataFrame
        A single tracking frame in meters, attacking frame. The goalkeeper is
        identified by `identify_goalkeeper`; see its docstring for the
        heuristic and its limits, since Metrica's sample data carries no role
        labels.
    defending_team : str
    goal_x : float

    Returns
    -------
    float
        Distance in meters from (goal_x, 0).
    """
    keeper = identify_goalkeeper(frame_positions, defending_team, goal_x)
    return float(np.hypot(keeper["x"] - goal_x, keeper["y"]))


def add_context_features(shots_with_frames: pd.DataFrame) -> pd.DataFrame:
    """Apply both context functions across a DataFrame of shots.

    Parameters
    ----------
    shots_with_frames : pd.DataFrame
        One row per shot with 'x', 'y' (meters, attacking frame), 'team', and
        a 'frame_positions' column holding that shot's normalized tracking
        frame. This is the same input add_pitch_control_features takes.

    Returns
    -------
    pd.DataFrame
        Copy with 'defenders_in_cone' (int) and 'goalkeeper_distance'
        (meters) added.
    """
    required = {"x", "y", "team", "frame_positions"}
    missing = required - set(shots_with_frames.columns)
    if missing:
        raise ValueError(f"shots_with_frames is missing columns: {sorted(missing)}")

    out = shots_with_frames.copy()
    records = []
    for row in out.itertuples():
        frame = row.frame_positions
        defending = {"Home": "Away", "Away": "Home"}[row.team]
        records.append(
            {
                "defenders_in_cone": count_defenders_in_cone(
                    frame, row.x, row.y, defending
                ),
                "goalkeeper_distance": goalkeeper_distance_from_goal_center(
                    frame, defending
                ),
            }
        )
    computed = pd.DataFrame(records, index=out.index)
    for column in computed.columns:
        out[column] = computed[column]
    return out
