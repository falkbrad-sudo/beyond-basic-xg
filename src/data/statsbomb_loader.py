"""Loads StatsBomb open event data for the 2023 NWSL season.

This module has two jobs. It supplies a large, real sample of shots, with
StatsBomb's own published xG, for validating the basic distance and angle
model. It also supplies each shot's freeze frame, which supports the
defender-context features on far more goals than the Metrica sample has.
It is never used for pitch control, which needs continuous tracking (see
METHODOLOGY.md principle #3).

Source: https://github.com/statsbomb/open-data
competition_id=49, season_id=107, the 2023 NWSL season, as configured in
config.yaml. That the season is in the free release is CHECKED, not
trusted: see `verify_season_available()`, which raises rather than silently
falling back to a different season. See METHODOLOGY.md principle #4.

Reading strategy
----------------
`scripts/download_data.sh` clones the whole open-data repo, so this module
reads the cloned JSON directly instead of going through `statsbombpy`'s
network calls. That keeps a full-season pull reproducible and offline, and
avoids several hundred HTTP requests per run. `statsbombpy` is used as a
fallback when the clone is absent.

Freeze frames
-------------
99% of these shots carry defender positions: `shot.freeze_frame` lists every
player in camera view at the instant of the shot, flagged teammate or
opponent, with a playing position. No NWSL match has a separate
`three-sixty/` file, so the positions are easy to miss.

That supports the two context features that looked most promising on the
Metrica sample (defenders in the shooting cone, and goalkeeper distance) on a
sample with 299 open-play goals rather than 9. It does not support Spearman
pitch control or whole-pitch Voronoi: a freeze frame is a single instant,
carries no velocities, and holds only the players in shot (a median of 9
opponents, not 11). Those remain Metrica-only.

Coordinates stay in StatsBomb's native 120x80 pitch units here; convert with
cleaning.statsbomb_to_meters() downstream. Unlike the Metrica data, no
playing-direction normalization is needed: StatsBomb's x axis already points
at the goal the shooting team is attacking, so x = 120 is always the
attacking goal.
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path

SHOT_COLUMNS = [
    "match_id",
    "shot_id",
    "team",
    "player",
    "period",
    "minute",
    "second",
    "x",
    "y",
    "body_part",
    "shot_type",
    "outcome",
    "is_goal",
    "is_penalty",
    "statsbomb_xg",
    "n_opponents_in_frame",
    "has_freeze_frame",
]

# Long-format freeze-frame table: one row per visible player per shot.
FREEZE_FRAME_COLUMNS = [
    "shot_id",
    "match_id",
    "player",
    "team",
    "position",
    "x",
    "y",
]

# Team labels used in the freeze-frame table. Deliberately not Home/Away:
# StatsBomb frames are relative to the shooter, and the downstream feature
# functions only need to know which side is defending.
ATTACKING = "attack"
DEFENDING = "defence"

_SHOTS_CACHE_NAME = "statsbomb_shots.parquet"
_FRAMES_CACHE_NAME = "statsbomb_shot_frames.parquet"

def _statsbomb_root() -> Path:
    """Absolute path to the cloned statsbomb/open-data repo."""
    cfg = load_config()
    return resolve_path(cfg["paths"]["statsbomb_dir"])


def _open_data_dir() -> Path | None:
    """The clone's `data/` directory, or None if the clone isn't usable.

    Requires both competitions.json and the matches/events trees, because
    `git clone` writes files progressively: a clone still being checked out
    has competitions.json long before it has the match lists, and treating
    that as ready produces a confusing FileNotFoundError deep in a later
    call rather than a clean fall back to statsbombpy.
    """
    data_dir = _statsbomb_root() / "data"
    required = [
        data_dir / "competitions.json",
        data_dir / "matches",
        data_dir / "events",
    ]
    return data_dir if all(path.exists() for path in required) else None


def _season_ids() -> tuple[int, int]:
    """The (competition_id, season_id) pair this project is scoped to."""
    cfg = load_config()
    return int(cfg["statsbomb"]["competition_id"]), int(cfg["statsbomb"]["season_id"])


@cache
def verify_season_available() -> dict:
    """Confirm the configured competition/season actually exists in the data.

    config.yaml names competition_id=49, season_id=107. This checks that
    the pair is really in the free release against StatsBomb's own
    competitions listing instead of assuming it, and raises if it does not
    hold. Substituting a different season would quietly change what the
    whole validation step means, and an unverified claim about which data is
    public is exactly what METHODOLOGY.md principle #4 exists to prevent.

    Returns
    -------
    dict
        The matching competitions.json record (competition and season names,
        data update timestamps).

    Raises
    ------
    FileNotFoundError
        If neither the local clone nor statsbombpy can supply the listing.
    ValueError
        If the configured (competition_id, season_id) is absent.
    """
    comp_id, season_id = _season_ids()
    data_dir = _open_data_dir()

    if data_dir is not None:
        with open(data_dir / "competitions.json") as f:
            competitions = pd.DataFrame(json.load(f))
    else:
        from statsbombpy import sb

        competitions = sb.competitions()

    match = competitions[
        (competitions["competition_id"] == comp_id)
        & (competitions["season_id"] == season_id)
    ]
    if match.empty:
        available = (
            competitions[competitions["competition_id"] == comp_id][
                ["season_id", "season_name"]
            ]
            .to_dict("records")
        )
        raise ValueError(
            f"config.yaml asks for competition_id={comp_id}, "
            f"season_id={season_id}, which is not in StatsBomb's open data. "
            f"Seasons actually available for competition {comp_id}: "
            f"{available or 'none'}. Stop and resolve this rather than "
            f"substituting another season. See this module's docstring."
        )
    return match.iloc[0].to_dict()


def get_2023_nwsl_matches() -> pd.DataFrame:
    """Return the match list for the 2023 NWSL season.

    Returns
    -------
    pd.DataFrame
        One row per match, as published by StatsBomb (match_id, date, teams,
        score, and competition/season metadata).
    """
    verify_season_available()
    comp_id, season_id = _season_ids()
    data_dir = _open_data_dir()

    if data_dir is None:
        from statsbombpy import sb

        return sb.matches(competition_id=comp_id, season_id=season_id)

    with open(data_dir / "matches" / str(comp_id) / f"{season_id}.json") as f:
        matches = json.load(f)
    return pd.json_normalize(matches, sep="_")


def _shot_rows_from_events(events: list[dict], match_id: int) -> list[dict]:
    """Pull the shot events out of one match's raw event list.

    Pure apart from its inputs. The parsing shape StatsBomb uses (nested
    `shot` dict, `location` as a two-element list) is what the unit tests in
    tests/test_statsbomb_loader.py exercise.
    """
    rows = []
    for event in events:
        if (event.get("type") or {}).get("name") != "Shot":
            continue
        shot = event.get("shot") or {}
        location = event.get("location") or [None, None]
        outcome = (shot.get("outcome") or {}).get("name")
        shot_type = (shot.get("type") or {}).get("name")
        freeze = shot.get("freeze_frame") or []
        rows.append(
            {
                "match_id": match_id,
                "shot_id": event.get("id"),
                "team": (event.get("team") or {}).get("name"),
                "player": (event.get("player") or {}).get("name"),
                "period": event.get("period"),
                "minute": event.get("minute"),
                "second": event.get("second"),
                "x": location[0],
                "y": location[1] if len(location) > 1 else None,
                "body_part": (shot.get("body_part") or {}).get("name"),
                "shot_type": shot_type,
                "outcome": outcome,
                "is_goal": outcome == "Goal",
                "is_penalty": shot_type == "Penalty",
                "statsbomb_xg": shot.get("statsbomb_xg"),
                # Coverage is recorded per shot so an undercount caused by
                # players being out of camera view is visible downstream
                # rather than silently depressing the defender counts.
                "n_opponents_in_frame": sum(
                    1 for p in freeze if not p.get("teammate")
                ),
                "has_freeze_frame": bool(freeze),
            }
        )
    return rows


def _freeze_frame_rows(event: dict, match_id: int) -> list[dict]:
    """Flatten one shot's freeze frame into long-format rows.

    Teams are labelled relative to the shooter (ATTACKING / DEFENDING)
    rather than Home/Away, because that is what the shot-context features
    need and it keeps the table usable without a separate lineup join.
    """
    shot = event.get("shot") or {}
    rows = []
    for player in shot.get("freeze_frame") or []:
        location = player.get("location") or [None, None]
        rows.append(
            {
                "shot_id": event.get("id"),
                "match_id": match_id,
                "player": (player.get("player") or {}).get("name"),
                "team": ATTACKING if player.get("teammate") else DEFENDING,
                "position": (player.get("position") or {}).get("name"),
                "x": location[0],
                "y": location[1] if len(location) > 1 else None,
            }
        )
    return rows


def get_shot_freeze_frames(
    match_ids: list[int] | None = None, use_cache: bool = True
) -> pd.DataFrame:
    """Defender and teammate positions at the instant of each shot.

    Parameters
    ----------
    match_ids : list[int], optional
        If None, every 2023 NWSL match.
    use_cache : bool
        Reuse a previously assembled full-season pull from
        paths.data_processed.

    Returns
    -------
    pd.DataFrame
        Long format, schema FREEZE_FRAME_COLUMNS: one row per visible player
        per shot, joinable to get_all_shots() on 'shot_id'. Coordinates are
        in StatsBomb's 120x80 units; convert with
        cleaning.statsbomb_to_meters().

        'team' is 'attack' or 'defence' relative to the shooter, and
        'position' is StatsBomb's playing position, which makes the
        goalkeeper identifiable exactly rather than by the positional
        heuristic the Metrica path has to use.

    Notes
    -----
    Only players in camera view are included (the median frame holds 9
    opponents, not 11), so a defender count from this data is a count of
    VISIBLE defenders. get_all_shots() carries 'n_opponents_in_frame' per
    shot so that limit stays checkable.
    """
    verify_season_available()
    cfg = load_config()
    cache_path = resolve_path(cfg["paths"]["data_processed"]) / _FRAMES_CACHE_NAME
    full_season = match_ids is None

    if full_season and use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    if full_season:
        match_ids = get_2023_nwsl_matches()["match_id"].astype(int).tolist()

    data_dir = _open_data_dir()
    if data_dir is None:
        raise FileNotFoundError(
            "Freeze frames are read from the local statsbomb/open-data clone, "
            "which is absent. Run scripts/download_data.sh. (statsbombpy's "
            "flattened events drop the nested freeze_frame, so there is no "
            "fallback path for this one.)"
        )

    rows: list[dict] = []
    for match_id in match_ids:
        event_path = data_dir / "events" / f"{match_id}.json"
        if not event_path.exists():
            continue
        with open(event_path) as f:
            events = json.load(f)
        for event in events:
            if (event.get("type") or {}).get("name") == "Shot":
                rows.extend(_freeze_frame_rows(event, int(match_id)))

    frames = pd.DataFrame(rows, columns=FREEZE_FRAME_COLUMNS)

    if full_season and not frames.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frames.to_parquet(cache_path, index=False)
    return frames


def get_all_shots(
    match_ids: list[int] | None = None, use_cache: bool = True
) -> pd.DataFrame:
    """Return shot events (with StatsBomb's own xG) across given matches.

    Parameters
    ----------
    match_ids : list[int], optional
        If None, pulls shots from every 2023 NWSL match.
    use_cache : bool
        When True (default) and match_ids is None, a previously assembled
        full-season pull is reused from paths.data_processed instead of
        re-parsing every match file. Pass False to force a rebuild.

    Returns
    -------
    pd.DataFrame
        One row per shot in the schema SHOT_COLUMNS: match_id, team, player,
        location (x, y) in StatsBomb's 120x80 units, body part, shot type,
        outcome, and StatsBomb's own xG in the 'statsbomb_xg' column.

        'statsbomb_xg' is useful as a sanity-check target for your own basic
        model. If a distance/angle-only model tracks StatsBomb's
        considerably more sophisticated xG reasonably well on this large
        sample, that's evidence the basic model is sound before extending it
        on the much smaller Metrica sample. It is not evidence the two models
        are equivalent: StatsBomb's uses freeze-frame defender positions this
        project has no access to.

        'is_penalty' is flagged rather than filtered, so the caller decides.
        Penalties should normally be excluded before fitting, because of their
        fixed geometry and a conversion rate far above open play.
    """
    verify_season_available()
    cfg = load_config()
    cache_path = resolve_path(cfg["paths"]["data_processed"]) / _SHOTS_CACHE_NAME
    full_season = match_ids is None

    if full_season and use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    if full_season:
        match_ids = get_2023_nwsl_matches()["match_id"].astype(int).tolist()

    data_dir = _open_data_dir()
    rows: list[dict] = []
    for match_id in match_ids:
        if data_dir is not None:
            event_path = data_dir / "events" / f"{match_id}.json"
            if not event_path.exists():
                continue
            with open(event_path) as f:
                events = json.load(f)
            rows.extend(_shot_rows_from_events(events, int(match_id)))
        else:
            from statsbombpy import sb

            events = sb.events(match_id=match_id)
            shots = events[events["type"] == "Shot"].copy()
            shots["match_id"] = int(match_id)
            rows.extend(_rows_from_statsbombpy(shots))

    shots_df = pd.DataFrame(rows, columns=SHOT_COLUMNS)

    if full_season and not shots_df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shots_df.to_parquet(cache_path, index=False)

    return shots_df


def _rows_from_statsbombpy(shots: pd.DataFrame) -> list[dict]:
    """Adapt statsbombpy's flattened shot frame to SHOT_COLUMNS.

    Only used on the no-local-clone fallback path.
    """
    rows = []
    for row in shots.itertuples():
        location = getattr(row, "location", None) or [None, None]
        outcome = getattr(row, "shot_outcome", None)
        shot_type = getattr(row, "shot_type", None)
        rows.append(
            {
                "match_id": row.match_id,
                "shot_id": getattr(row, "id", None),
                "team": getattr(row, "team", None),
                "player": getattr(row, "player", None),
                "period": getattr(row, "period", None),
                "minute": getattr(row, "minute", None),
                "second": getattr(row, "second", None),
                "x": location[0],
                "y": location[1] if len(location) > 1 else None,
                "body_part": getattr(row, "shot_body_part", None),
                "shot_type": shot_type,
                "outcome": outcome,
                "is_goal": outcome == "Goal",
                "is_penalty": shot_type == "Penalty",
                "statsbomb_xg": getattr(row, "shot_statsbomb_xg", None),
            }
        )
    return rows
