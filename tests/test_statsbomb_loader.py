"""Unit tests for src/data/statsbomb_loader.py.

Covers the event-JSON shot parser with hand-built fixtures shaped like real
StatsBomb open-data events, so these run without the ~5GB clone. The
network/clone-dependent paths are covered by tests/test_loaders.py.
"""
import pandas as pd

from src.data.statsbomb_loader import SHOT_COLUMNS, _shot_rows_from_events


def _shot_event(**overrides) -> dict:
    """An event shaped like a real StatsBomb open-data shot."""
    event = {
        "id": "abc-123",
        "type": {"id": 16, "name": "Shot"},
        "team": {"id": 1, "name": "Team A"},
        "player": {"id": 99, "name": "A Player"},
        "period": 1,
        "minute": 23,
        "second": 45,
        "location": [102.5, 38.0],
        "shot": {
            "statsbomb_xg": 0.1234,
            "body_part": {"id": 40, "name": "Right Foot"},
            "type": {"id": 87, "name": "Open Play"},
            "outcome": {"id": 98, "name": "Off T"},
        },
    }
    event.update(overrides)
    return event


class TestShotRowsFromEvents:
    def test_ignores_non_shot_events(self):
        events = [
            {"type": {"name": "Pass"}, "location": [50, 40]},
            {"type": {"name": "Carry"}},
            _shot_event(),
        ]
        rows = _shot_rows_from_events(events, match_id=7)
        assert len(rows) == 1

    def test_splits_location_list_into_x_and_y(self):
        rows = _shot_rows_from_events([_shot_event()], match_id=7)
        assert rows[0]["x"] == 102.5
        assert rows[0]["y"] == 38.0

    def test_lifts_statsbomb_xg_to_a_flat_column(self):
        # tests/test_loaders.py asserts on this exact column name.
        rows = _shot_rows_from_events([_shot_event()], match_id=7)
        assert rows[0]["statsbomb_xg"] == 0.1234

    def test_match_id_is_attached(self):
        rows = _shot_rows_from_events([_shot_event()], match_id=3895302)
        assert rows[0]["match_id"] == 3895302

    def test_goal_outcome_sets_is_goal(self):
        event = _shot_event(
            shot={
                "statsbomb_xg": 0.9,
                "outcome": {"name": "Goal"},
                "type": {"name": "Open Play"},
                "body_part": {"name": "Left Foot"},
            }
        )
        rows = _shot_rows_from_events([event], match_id=7)
        assert rows[0]["is_goal"] is True
        assert rows[0]["outcome"] == "Goal"

    def test_saved_outcome_is_not_a_goal(self):
        event = _shot_event(
            shot={"statsbomb_xg": 0.2, "outcome": {"name": "Saved"},
                  "type": {"name": "Open Play"}}
        )
        assert _shot_rows_from_events([event], match_id=7)[0]["is_goal"] is False

    def test_penalty_is_flagged_not_dropped(self):
        event = _shot_event(
            shot={"statsbomb_xg": 0.78, "outcome": {"name": "Goal"},
                  "type": {"name": "Penalty"}}
        )
        row = _shot_rows_from_events([event], match_id=7)[0]
        assert row["is_penalty"] is True
        assert row["shot_type"] == "Penalty"

    def test_open_play_is_not_a_penalty(self):
        row = _shot_rows_from_events([_shot_event()], match_id=7)[0]
        assert row["is_penalty"] is False

    def test_missing_nested_fields_do_not_raise(self):
        # Real open data omits optional keys rather than nulling them.
        event = {"id": "x", "type": {"name": "Shot"}, "location": [100.0, 40.0]}
        row = _shot_rows_from_events([event], match_id=7)[0]
        assert row["statsbomb_xg"] is None
        assert row["team"] is None
        assert row["is_goal"] is False

    def test_missing_location_gives_null_coordinates(self):
        event = _shot_event(location=None)
        row = _shot_rows_from_events([event], match_id=7)[0]
        assert row["x"] is None
        assert row["y"] is None

    def test_rows_match_the_declared_schema(self):
        rows = _shot_rows_from_events([_shot_event()], match_id=7)
        assert set(rows[0]) == set(SHOT_COLUMNS)
        # And survive the DataFrame construction get_all_shots does.
        df = pd.DataFrame(rows, columns=SHOT_COLUMNS)
        assert list(df.columns) == SHOT_COLUMNS
        assert len(df) == 1

    def test_empty_event_list_gives_no_rows(self):
        assert _shot_rows_from_events([], match_id=7) == []
