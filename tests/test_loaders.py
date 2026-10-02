"""Integration tests for src/data/*_loader.py. They require downloaded data or
network access. Run explicitly with `pytest -m integration`.
"""
import pytest

from src.data import metrica_loader, statsbomb_loader
from src.data.cleaning import (
    metrica_to_meters,
    normalize_playing_direction,
    validate_coordinates,
)
from src.features.geometry import GOAL_X, add_geometry_features


@pytest.mark.integration
def test_metrica_shot_extraction_returns_reasonable_count():
    shots = metrica_loader.extract_shot_events(game_id=1)
    # Sanity bound, not a precise expectation -- a single match should have
    # somewhere on the order of 5-30 shots total, between both teams. If
    # this returns 0 or 500, something's wrong with the event filtering.
    assert 0 < len(shots) < 50


@pytest.mark.integration
def test_statsbomb_shots_include_statsbomb_xg_column():
    shots = statsbomb_loader.get_all_shots(match_ids=None)
    assert "statsbomb_xg" in shots.columns or any(
        "xg" in c.lower() for c in shots.columns
    )


@pytest.mark.integration
def test_statsbomb_shots_all_come_from_the_configured_season():
    """METHODOLOGY.md principle #4, made executable: every shot must belong
    to a match in the verified 2023 NWSL match list, so no other season or
    competition can leak into the sample.
    """
    shots = statsbomb_loader.get_all_shots(match_ids=None)
    season_matches = set(statsbomb_loader.get_2023_nwsl_matches()["match_id"])
    assert set(shots["match_id"]) <= season_matches


@pytest.mark.integration
def test_metrica_teams_attack_opposite_goals_in_every_period():
    """The invariant that caught the first (broken) direction heuristic."""
    for game_id in metrica_loader.list_available_games():
        directions = metrica_loader.playing_direction(game_id)
        periods = {period for _, period in directions}
        for period in periods:
            home = directions[(metrica_loader.HOME, period)]
            away = directions[(metrica_loader.AWAY, period)]
            assert home != away, f"game {game_id} period {period}"


@pytest.mark.integration
def test_metrica_shots_land_in_the_attacking_half_after_normalization():
    """Guards the pitch-frame / attacking-frame bug.

    Without normalize_playing_direction, shots taken toward x = 0 come out
    with distances near 90m instead of near 10m -- plausible-looking numbers
    that would silently corrupt every geometry feature.
    """
    shots = metrica_loader.extract_shot_events(game_id=1)
    shots = metrica_to_meters(shots, "x", "y")
    shots = normalize_playing_direction(
        shots, attacking_right=shots["attacking_right"]
    )
    validate_coordinates(shots, "x", "y")
    feats = add_geometry_features(shots)

    # Every shot should be in the attacking half, and no further from goal
    # than the halfway line is.
    assert (feats["x"] > 0).all()
    assert feats["distance_to_goal"].max() < GOAL_X
    # Real shots are not taken from 1m or 60m out.
    assert feats["distance_to_goal"].between(1.0, 45.0).all()


@pytest.mark.integration
def test_metrica_tracking_frame_at_shot_has_both_teams_and_the_ball():
    shots = metrica_loader.extract_shot_events(game_id=1)
    frame = metrica_loader.get_tracking_frame_at_shot(
        game_id=1, shot_frame=int(shots["frame"].iloc[0])
    )
    teams = set(frame["team"])
    assert metrica_loader.HOME in teams
    assert metrica_loader.AWAY in teams
    assert metrica_loader.BALL in teams
    # 22 players + ball, allowing for substitutes untracked at that instant.
    assert 15 <= len(frame) <= 30


@pytest.mark.integration
def test_metrica_unknown_frame_raises_rather_than_returning_empty():
    with pytest.raises(ValueError, match="not found"):
        metrica_loader.get_tracking_frame_at_shot(game_id=1, shot_frame=99_999_999)


@pytest.mark.integration
def test_statsbomb_configured_season_actually_exists():
    """config.yaml claims competition 49 / season 107 is a free NWSL season.

    This is the check, not the claim -- it raises with the list of seasons
    actually available if the configured pair is absent.
    """
    season = statsbomb_loader.verify_season_available()
    assert season["competition_id"] == 49
    assert season["season_id"] == 107


@pytest.mark.integration
def test_statsbomb_sample_is_large_enough_to_validate_against():
    """The whole point of this dataset is being much bigger than Metrica."""
    shots = statsbomb_loader.get_all_shots()
    assert len(shots) > 300
    assert shots["statsbomb_xg"].notna().sum() > 300
