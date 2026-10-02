"""Loads Metrica Sports open sample tracking + event data, with a focus on
extracting shot events and their corresponding tracking frame.

This is the only data source in this project with continuous tracking of
every player, which pitch control needs: positions frame by frame, from
which velocities can be estimated. StatsBomb freeze frames give positions at
a single instant only (see METHODOLOGY.md principle #3), so the two data
sources stay in separate pipelines.

Source: https://github.com/metrica-sports/sample-data
Games 1-2: Metrica's own CSV format. Game 3: EPTS/JSON format (read via
kloppy). Raw coordinates are normalized (0, 1); convert with
src/data/cleaning.py:metrica_to_meters() before any distance/angle
calculation.

Playing direction
-----------------
Every function here returns coordinates in Metrica's raw frame, which is
fixed to the pitch: it says nothing about which goal a team is attacking.
The two teams attack opposite goals and swap ends at halftime, and the
direction is NOT consistent between sample games (in game 1 the home team
attacks toward x=1 in the first period; in game 2 it attacks toward x=0).
So direction is derived from each goalkeeper's average position rather
than assumed (see `playing_direction()`), and `extract_shot_events()`
carries an `attacking_right` column so callers can hand it straight to
`cleaning.normalize_playing_direction()`.
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path
from src.data.cleaning import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    metrica_to_meters,
    normalize_playing_direction,
)

HOME = "Home"
AWAY = "Away"
BALL = "ball"

CSV_GAMES = (1, 2)
EPTS_GAMES = (3,)

# Metrica's event taxonomy marks goals via the Subtype field: 'ON TARGET-GOAL'
# and 'HEAD-ON TARGET-GOAL' in the CSV games, a 'GOAL' subtype in game 3.
# Verified against the Subtype values actually present in the downloaded
# sample data; the taxonomy has no OWN GOAL subtype, so a substring test is
# unambiguous here.
# Half-width, in frames, of the central difference used for player velocity.
# At 25fps, +/-3 frames is a 0.24s window: long enough to damp tracking jitter,
# short enough that a player's velocity at the shot is still their velocity at
# the shot rather than an average over a run.
_VELOCITY_HALF_WINDOW = 3
FRAME_RATE_HZ = 25.0

_GOAL_TOKEN = "GOAL"
_PENALTY_TOKEN = "PENALTY"

EVENT_COLUMNS = [
    "game_id",
    "team",
    "type",
    "subtype",
    "period",
    "start_frame",
    "start_time_s",
    "end_frame",
    "end_time_s",
    "from_player",
    "from_player_id",
    "to_player",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
]

SHOT_COLUMNS = [
    "game_id",
    "team",
    "period",
    "frame",
    "time_s",
    "player",
    "player_id",
    "subtype",
    "x",
    "y",
    "is_goal",
    "is_penalty",
    "attacking_right",
]


def _metrica_root() -> Path:
    """Absolute path to the cloned metrica-sports/sample-data repo."""
    cfg = load_config()
    return resolve_path(cfg["paths"]["metrica_dir"])


def _game_dir(game_id: int) -> Path:
    """Directory holding one sample game's files."""
    return _metrica_root() / "data" / f"Sample_Game_{game_id}"


def list_available_games() -> list[int]:
    """Return the game_ids actually present under data/external/metrica.

    Useful as a sanity check after running scripts/download_data.sh. Don't
    assume all 3 games downloaded successfully without checking.

    Returns
    -------
    list[int]
        Sorted game_ids from config.yaml's metrica.game_ids whose required
        files are all present on disk.
    """
    cfg = load_config()
    available = []
    for game_id in cfg["metrica"]["game_ids"]:
        game_dir = _game_dir(game_id)
        if game_id in CSV_GAMES:
            required = [
                game_dir / f"Sample_Game_{game_id}_RawEventsData.csv",
                game_dir / f"Sample_Game_{game_id}_RawTrackingData_Home_Team.csv",
                game_dir / f"Sample_Game_{game_id}_RawTrackingData_Away_Team.csv",
            ]
        else:
            required = [
                game_dir / f"Sample_Game_{game_id}_events.json",
                game_dir / f"Sample_Game_{game_id}_metadata.xml",
                game_dir / f"Sample_Game_{game_id}_tracking.txt",
            ]
        if all(path.exists() for path in required):
            available.append(game_id)
    return sorted(available)


def _load_tracking_csv(game_id: int, team: str) -> pd.DataFrame:
    """Read a games-1-2 raw tracking CSV into a tidy wide frame.

    Metrica's CSV has three header rows, not the two its own documentation
    implies: team name, jersey number, then the row carrying 'Period',
    'Frame', 'Time [s]' and one label per player occupying the FIRST of that
    player's two columns (x then y), with the second column unnamed.
    """
    path = _game_dir(game_id) / f"Sample_Game_{game_id}_RawTrackingData_{team}_Team.csv"
    raw = pd.read_csv(path, skiprows=2)

    out = pd.DataFrame(
        {
            "period": raw["Period"].astype("Int64"),
            "frame": raw["Frame"].astype("Int64"),
            "time_s": raw["Time [s]"].astype(float),
        }
    )

    columns = list(raw.columns)
    for i, name in enumerate(columns):
        if i < 3 or name.startswith("Unnamed"):
            continue
        # The y column is the immediately following (unnamed) one.
        label = BALL if name == "Ball" else name
        out[f"{label}_x"] = raw[columns[i]].astype(float)
        out[f"{label}_y"] = raw[columns[i + 1]].astype(float)
    return out


@cache
def _epts_dataset(game_id: int):
    """Load game 3's EPTS tracking via kloppy, in Metrica's native 0-1 frame."""
    from kloppy import metrica

    game_dir = _game_dir(game_id)
    return metrica.load_tracking_epts(
        meta_data=str(game_dir / f"Sample_Game_{game_id}_metadata.xml"),
        raw_data=str(game_dir / f"Sample_Game_{game_id}_tracking.txt"),
        # Keep kloppy from rescaling: the rest of this project expects raw
        # normalized Metrica coordinates out of this module.
        coordinates="metrica",
    )


@cache
def _epts_team_players(game_id: int) -> dict[str, list[str]]:
    """Map 'Home'/'Away' to the kloppy player-id column prefixes for game 3."""
    dataset = _epts_dataset(game_id)
    mapping: dict[str, list[str]] = {HOME: [], AWAY: []}
    for team in dataset.metadata.teams:
        side = HOME if str(team.ground) == "home" else AWAY
        mapping[side] = [str(player.player_id) for player in team.players]
    return mapping


def load_tracking(game_id: int, team: str) -> pd.DataFrame:
    """Load raw tracking data for one team in one Metrica sample game.

    Parameters
    ----------
    game_id : int
        1, 2, or 3 (see config.yaml: metrica.game_ids).
    team : str
        'Home' or 'Away'.

    Returns
    -------
    pd.DataFrame
        One row per frame. Columns: 'period', 'frame', 'time_s', then
        '<player>_x'/'<player>_y' for each of that team's players, plus
        'ball_x'/'ball_y'. Coordinates are Metrica's raw normalized (0, 1)
        values; convert with cleaning.metrica_to_meters() before computing
        any distance-based feature.

    Notes
    -----
    Games 1-2 ship as `Sample_Game_{game_id}_RawTrackingData_{team}_Team.csv`
    with a non-standard multi-row header (Metrica's own quirk), handled here
    so nothing downstream has to know about it. Game 3 uses the FIFA EPTS
    format and is read via kloppy. Either path returns the same schema, so
    callers never branch on game_id.
    """
    if team not in (HOME, AWAY):
        raise ValueError(f"team must be '{HOME}' or '{AWAY}', got {team!r}")
    # Copy on the way out: the cache hands back the same object every time,
    # and callers are free to mutate what they receive.
    return _load_tracking_cached(game_id, team).copy()


@cache
def _load_tracking_cached(game_id: int, team: str) -> pd.DataFrame:
    """Parse one team's tracking file once per process.

    Each sample game is ~145k frames per team, and playing_direction(),
    extract_shot_events() and get_tracking_frame_at_shot() all need the same
    data; re-parsing per call made a three-game run take minutes.
    """
    if game_id in CSV_GAMES:
        return _load_tracking_csv(game_id, team)

    dataset = _epts_dataset(game_id)
    raw = dataset.to_df()
    player_ids = _epts_team_players(game_id)[team]

    out = pd.DataFrame(
        {
            "period": raw["period_id"].astype("Int64"),
            "frame": raw["frame_id"].astype("Int64"),
            "time_s": raw["timestamp"].dt.total_seconds().astype(float),
        }
    )
    for player_id in player_ids:
        if f"{player_id}_x" in raw.columns:
            out[f"{player_id}_x"] = raw[f"{player_id}_x"].astype(float)
            out[f"{player_id}_y"] = raw[f"{player_id}_y"].astype(float)
    out[f"{BALL}_x"] = raw["ball_x"].astype(float)
    out[f"{BALL}_y"] = raw["ball_y"].astype(float)
    return out


def _player_x_columns(tracking: pd.DataFrame) -> list[str]:
    """The '<player>_x' columns of a tracking frame, excluding the ball."""
    return [
        c
        for c in tracking.columns
        if c.endswith("_x") and not c.startswith(BALL)
    ]


@cache
def playing_direction(game_id: int) -> dict[tuple[str, int], bool]:
    """Infer, per (team, period), whether that team attacks toward x = 1.

    Derived from the tracking data rather than assumed, because the sample
    games are not consistent with each other: in game 1 the home team attacks
    toward x = 1 in period 1, in game 2 it attacks toward x = 0.

    The signal is the goalkeeper, identified as the player whose mean x over
    the period is furthest from the halfway line, the same heuristic
    src/features/shot_context.py relies on. A keeper sits in front of the goal
    his team is DEFENDING, so a keeper averaging x < 0.5 means his team is
    attacking toward x = 1.

    Parameters
    ----------
    game_id : int

    Returns
    -------
    dict[tuple[str, int], bool]
        Keyed by (team, period); True where that team attacks toward x = 1,
        which is the +x direction after cleaning.metrica_to_meters(). Feed
        the matching value to cleaning.normalize_playing_direction().

    Raises
    ------
    ValueError
        If both teams come out attacking the same goal in some period, which
        is impossible and means the heuristic failed.

    Notes
    -----
    An earlier version averaged x over all of a team's players in the opening
    seconds of the period. That was too weak a signal: in game 1 the away
    team's first-period mean was 0.483, marginally on the wrong side of the
    halfway line, and the direction came out backwards. The keeper's mean x
    lands at roughly 0.12 or 0.88, so the margin is wide.

    For game 3 the result is cross-checked against the
    `attack_direction_first_half` field in Metrica's own EPTS metadata, and
    raises if the two disagree, because the heuristic should never silently diverge
    from the provider's own statement where that statement exists.
    """
    directions: dict[tuple[str, int], bool] = {}
    for team in (HOME, AWAY):
        tracking = load_tracking(game_id, team)
        x_columns = _player_x_columns(tracking)
        for period in sorted(p for p in tracking["period"].dropna().unique()):
            window = tracking[tracking["period"] == period]
            mean_x = window[x_columns].mean()
            keeper = (mean_x - 0.5).abs().idxmax()
            directions[(team, int(period))] = bool(mean_x[keeper] < 0.5)

    _assert_teams_attack_opposite_goals(game_id, directions)
    if game_id in EPTS_GAMES:
        _assert_matches_epts_metadata(game_id, directions)
    return directions


def _assert_teams_attack_opposite_goals(
    game_id: int, directions: dict[tuple[str, int], bool]
) -> None:
    """Raise if both teams appear to attack the same goal in a period."""
    periods = {period for _, period in directions}
    for period in sorted(periods):
        home = directions.get((HOME, period))
        away = directions.get((AWAY, period))
        if home is None or away is None:
            continue
        if home == away:
            raise ValueError(
                f"Game {game_id} period {period}: both teams inferred to be "
                f"attacking the same goal (attacking_right={home}). The "
                f"playing-direction heuristic has failed; do not use these "
                f"coordinates for geometry features."
            )


def _assert_matches_epts_metadata(
    game_id: int, directions: dict[tuple[str, int], bool]
) -> None:
    """Raise if the inferred direction contradicts Metrica's own metadata."""
    import xml.etree.ElementTree as ET

    path = _game_dir(game_id) / f"Sample_Game_{game_id}_metadata.xml"
    root = ET.parse(path).getroot()
    ground_by_id = {
        str(team.team_id): (HOME if str(team.ground) == "home" else AWAY)
        for team in _epts_dataset(game_id).metadata.teams
    }

    for team_node in root.iter("Team"):
        side = ground_by_id.get(team_node.get("id"))
        if side is None:
            continue
        for param in team_node.iter("ProviderParameter"):
            name = param.findtext("Name")
            if name != "attack_direction_first_half":
                continue
            stated_right = param.findtext("Value") == "left_to_right"
            inferred = directions.get((side, 1))
            if inferred is not None and inferred != stated_right:
                raise ValueError(
                    f"Game {game_id}: inferred first-half direction for {side} "
                    f"(attacking_right={inferred}) contradicts Metrica's own "
                    f"metadata (attacking_right={stated_right}). Do not trust "
                    f"either until this is resolved."
                )


def _load_events_csv(game_id: int) -> pd.DataFrame:
    """Read a games-1-2 raw events CSV into the unified event schema."""
    path = _game_dir(game_id) / f"Sample_Game_{game_id}_RawEventsData.csv"
    raw = pd.read_csv(path)
    out = raw.rename(
        columns={
            "Team": "team",
            "Type": "type",
            "Subtype": "subtype",
            "Period": "period",
            "Start Frame": "start_frame",
            "Start Time [s]": "start_time_s",
            "End Frame": "end_frame",
            "End Time [s]": "end_time_s",
            "From": "from_player",
            "To": "to_player",
            "Start X": "start_x",
            "Start Y": "start_y",
            "End X": "end_x",
            "End Y": "end_y",
        }
    )
    out["game_id"] = game_id
    # In the CSV games the tracking columns are labelled with exactly these
    # names ('Player9'), so the name doubles as the join key.
    out["from_player_id"] = out["from_player"]
    return out[EVENT_COLUMNS]


def _load_events_json(game_id: int) -> pd.DataFrame:
    """Read game 3's events JSON into the unified event schema.

    Team A / Team B are mapped to Home / Away using kloppy's reading of the
    EPTS metadata, so the label matches load_tracking()'s team argument.
    """
    path = _game_dir(game_id) / f"Sample_Game_{game_id}_events.json"
    with open(path) as f:
        payload = json.load(f)

    ground_by_id = {
        str(team.team_id): (HOME if str(team.ground) == "home" else AWAY)
        for team in _epts_dataset(game_id).metadata.teams
    }

    rows = []
    for event in payload["data"]:
        subtypes = event.get("subtypes") or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        # Joined with '-' to match the CSV games' Subtype spelling (e.g.
        # 'ON TARGET-SAVED'), so downstream filters work on either format.
        subtype = "-".join(s["name"] for s in subtypes) or None
        start, end = event.get("start") or {}, event.get("end") or {}
        rows.append(
            {
                "game_id": game_id,
                "team": ground_by_id.get(str((event.get("team") or {}).get("id"))),
                "type": (event.get("type") or {}).get("name"),
                "subtype": subtype,
                "period": event.get("period"),
                "start_frame": start.get("frame"),
                "start_time_s": start.get("time"),
                "end_frame": end.get("frame"),
                "end_time_s": end.get("time"),
                "from_player": (event.get("from") or {}).get("name"),
                # kloppy names game 3's tracking columns by player id
                # ('P3574'), not by display name ('Player 7'), so the id is
                # the only usable join key here.
                "from_player_id": (event.get("from") or {}).get("id"),
                "to_player": (event.get("to") or {}).get("name"),
                "start_x": start.get("x"),
                "start_y": start.get("y"),
                "end_x": end.get("x"),
                "end_y": end.get("y"),
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


@cache
def _load_events_cached(game_id: int) -> pd.DataFrame:
    """Parse one game's event file once per process."""
    if game_id in CSV_GAMES:
        return _load_events_csv(game_id)
    return _load_events_json(game_id)


def load_events(game_id: int) -> pd.DataFrame:
    """Load Metrica's event data for one sample game (synced with tracking).

    Parameters
    ----------
    game_id : int

    Returns
    -------
    pd.DataFrame
        One row per event in the unified schema EVENT_COLUMNS: event type,
        subtype, team ('Home'/'Away'), period, start/end frame and time, the
        players involved, and start/end (x, y) in Metrica's raw normalized
        coordinates. Identical columns for games 1-2 (CSV) and game 3 (JSON).
    """
    if game_id in CSV_GAMES:
        return _load_events_csv(game_id)
    return _load_events_json(game_id)


def is_goal_subtype(subtype: pd.Series) -> pd.Series:
    """Flag which shot Subtype strings represent a goal.

    Metrica spells goals into the Subtype field rather than giving them their
    own column: 'ON TARGET-GOAL' and 'HEAD-ON TARGET-GOAL' in the CSV games,
    and a 'GOAL' entry in game 3's subtype list (joined to the same spelling
    by _load_events_json). The taxonomy in game 3's metadata lists no OWN
    GOAL subtype and none appears in the sample data, so a substring test is
    unambiguous.

    Parameters
    ----------
    subtype : pd.Series
        The 'subtype' column of shot events. Nulls count as not-a-goal.

    Returns
    -------
    pd.Series
        Boolean, same index as the input.

    Examples
    --------
    >>> import pandas as pd
    >>> is_goal_subtype(pd.Series(["ON TARGET-GOAL", "BLOCKED", None])).tolist()
    [True, False, False]
    """
    return subtype.fillna("").str.contains(_GOAL_TOKEN)


def _penalty_keys(events: pd.DataFrame) -> set[tuple]:
    """(team, period, frame) keys of penalties, from SET PIECE events.

    Metrica records a penalty twice: once as a SET PIECE with subtype
    'PENALTY' (and null coordinates) and once as the SHOT itself at the same
    frame. The SHOT row carries the coordinates, so the SET PIECE row is only
    useful as a flag, matched back onto the shot by frame.
    """
    set_pieces = events[
        (events["type"] == "SET PIECE")
        & events["subtype"].fillna("").str.contains(_PENALTY_TOKEN)
    ]
    return {
        (row.team, row.period, row.start_frame)
        for row in set_pieces.itertuples()
    }


def extract_shot_events(game_id: int) -> pd.DataFrame:
    """Return only the shot events from one game's event data.

    Parameters
    ----------
    game_id : int

    Returns
    -------
    pd.DataFrame
        One row per shot in the schema SHOT_COLUMNS: team, 'frame' (for
        joining to tracking data via get_tracking_frame_at_shot), shot
        location 'x'/'y' in Metrica's raw normalized coordinates, and
        'is_goal'. Also carries 'is_penalty' and 'attacking_right'.

        'player' is the display name; 'player_id' is the key that matches
        the 'player' column of get_tracking_frame_at_shot(). The two differ
        for game 3 (see _load_events_json), which is why both are kept.

    Notes
    -----
    'is_penalty' is provided rather than filtered out, so the caller decides.
    Penalties should normally be excluded before fitting: their geometry is
    fixed and their conversion rate is far above open play, which distorts a
    sample this small. There is exactly one penalty across the three sample
    games.

    'attacking_right' comes from playing_direction() and is what makes the
    output usable: pass it to cleaning.normalize_playing_direction() after
    metrica_to_meters(), or every shot taken toward x = 0 will have its
    distance measured to the wrong goal.
    """
    events = load_events(game_id)
    shots = events[events["type"] == "SHOT"].copy()

    penalties = _penalty_keys(events)
    directions = playing_direction(game_id)

    out = pd.DataFrame(
        {
            "game_id": shots["game_id"],
            "team": shots["team"],
            "period": shots["period"],
            "frame": shots["start_frame"],
            "time_s": shots["start_time_s"],
            "player": shots["from_player"],
            "player_id": shots["from_player_id"],
            "subtype": shots["subtype"],
            "x": shots["start_x"],
            "y": shots["start_y"],
            "is_goal": is_goal_subtype(shots["subtype"]),
            "is_penalty": [
                (row.team, row.period, row.start_frame) in penalties
                for row in shots.itertuples()
            ],
            "attacking_right": [
                directions[(row.team, int(row.period))]
                for row in shots.itertuples()
            ],
        }
    )
    return out[SHOT_COLUMNS].reset_index(drop=True)


def get_tracking_frame_at_shot(
    game_id: int, shot_frame: int, with_velocity: bool = False
) -> pd.DataFrame:
    """Return all 22 players' + ball positions at the exact frame of a shot.

    This is the join point between events and tracking that makes the whole
    project possible. Every downstream pitch-control-at-shot feature depends
    on this returning the correct, single frame.

    Parameters
    ----------
    game_id : int
    shot_frame : int
        From extract_shot_events() output.
    with_velocity : bool
        Add 'vx'/'vy' columns in m/s, central-differenced from neighbouring
        frames. Needed by the Spearman-style pitch-control model, which
        accounts for where players are heading, not only where they are.

    Returns
    -------
    pd.DataFrame
        One row per player (both teams) plus one row for the ball, with
        'player', 'team' ('Home', 'Away', or 'ball'), and 'x', 'y' in
        Metrica's raw coordinates (convert downstream). Players not on the
        pitch at that frame are dropped rather than returned as NaN, so the
        row count is the number of tracked entities, not a fixed 23.

        Prefer get_normalized_tracking_frame_at_shot() for feature work. It
        pairs the unit conversion with the playing-direction rotation so the
        two cannot be applied inconsistently.

    Raises
    ------
    ValueError
        If the frame is absent from either team's tracking data. A silent
        empty frame here would quietly corrupt every pitch-control value.
    """
    rows = []
    for team in (HOME, AWAY):
        tracking = load_tracking(game_id, team)
        match = tracking[tracking["frame"] == shot_frame]
        if match.empty:
            raise ValueError(
                f"Frame {shot_frame} not found in game {game_id} {team} tracking "
                f"data (frames run {tracking['frame'].min()}-"
                f"{tracking['frame'].max()})."
            )
        row = match.iloc[0]
        labels = [c[:-2] for c in _player_x_columns(tracking)]
        velocities = (
            _velocity_at_frame(tracking, shot_frame, labels)
            if with_velocity
            else {}
        )
        for label in labels:
            entry = {
                "player": label,
                "team": team,
                "x": row[f"{label}_x"],
                "y": row[f"{label}_y"],
            }
            if with_velocity:
                entry["vx"], entry["vy"] = velocities.get(label, (0.0, 0.0))
            rows.append(entry)
        if team == HOME:
            # Both teams' files carry identical ball columns; take it once.
            ball = {
                "player": BALL,
                "team": BALL,
                "x": row[f"{BALL}_x"],
                "y": row[f"{BALL}_y"],
            }
            if with_velocity:
                ball["vx"], ball["vy"] = velocities.get(BALL, (0.0, 0.0))
            rows.append(ball)

    columns = ["player", "team", "x", "y"] + (["vx", "vy"] if with_velocity else [])
    frame = pd.DataFrame(rows, columns=columns)
    return frame.dropna(subset=["x", "y"]).reset_index(drop=True)


def _velocity_at_frame(
    tracking: pd.DataFrame, frame: int, labels: list[str]
) -> dict[str, tuple[float, float]]:
    """Central-difference velocity (m/s) for each player at one frame.

    Metrica ships positions, not velocities, so they are differenced here.
    Returned in meters per second in the pitch frame, which means they must
    be rotated along with positions if the coordinates are (see
    get_normalized_tracking_frame_at_shot).
    """
    frames = tracking["frame"]
    before = frames.searchsorted(frame - _VELOCITY_HALF_WINDOW)
    after = frames.searchsorted(frame + _VELOCITY_HALF_WINDOW)
    before = max(int(before), 0)
    after = min(int(after), len(tracking) - 1)
    if after <= before:
        return {label: (0.0, 0.0) for label in labels}

    row_before, row_after = tracking.iloc[before], tracking.iloc[after]
    dt = (row_after["frame"] - row_before["frame"]) / FRAME_RATE_HZ
    if not dt:
        return {label: (0.0, 0.0) for label in labels}

    velocities = {}
    for label in labels:
        dx = (row_after[f"{label}_x"] - row_before[f"{label}_x"]) * PITCH_LENGTH_M
        dy = (row_after[f"{label}_y"] - row_before[f"{label}_y"]) * PITCH_WIDTH_M
        if pd.isna(dx) or pd.isna(dy):
            # A substitute who is untracked at either end of the window has
            # no defined velocity. Zero is the safest placeholder: it says
            # "not moving as far as we know" rather than inventing a vector,
            # and such players are dropped from the frame anyway if their
            # position at the shot is itself missing.
            velocities[label] = (0.0, 0.0)
            continue
        velocities[label] = (float(dx / dt), float(dy / dt))
    return velocities


def get_normalized_tracking_frame_at_shot(
    game_id: int, shot_frame: int, attacking_right: bool
) -> pd.DataFrame:
    """Tracking frame at a shot, in meters and in the shooter's attacking frame.

    Convenience wrapper that does the three things that must always happen
    together, in the right order: pull the raw frame, convert to meters, and
    rotate so the goal being shot at is at x = +52.5.

    It exists because doing them separately is the single easiest way to
    reintroduce the bug this project already had once: shot coordinates
    normalized but tracking coordinates left in the pitch frame puts every
    defender at the wrong end of the pitch, and every pitch-control value
    downstream is then quietly wrong while looking entirely plausible.

    Parameters
    ----------
    game_id : int
    shot_frame : int
        From extract_shot_events()['frame'].
    attacking_right : bool
        From extract_shot_events()['attacking_right'] for the SAME shot.

    Returns
    -------
    pd.DataFrame
        One row per tracked player plus the ball, with 'player', 'team'
        ('Home'/'Away'/'ball'), 'x', 'y' in meters in the attacking frame,
        and 'vx', 'vy' in m/s in the same frame.
    """
    frame = get_tracking_frame_at_shot(game_id, shot_frame, with_velocity=True)
    frame = metrica_to_meters(frame, "x", "y")
    frame = normalize_playing_direction(frame, attacking_right=attacking_right)
    if not attacking_right:
        # Velocities are vectors in the same frame, so the 180-degree
        # rotation applies to them too.
        frame["vx"] = -frame["vx"]
        frame["vy"] = -frame["vy"]
    return frame
