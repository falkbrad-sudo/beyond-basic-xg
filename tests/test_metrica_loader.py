"""Unit tests for src/data/metrica_loader.py.

These cover the loader's pure logic (goal flagging, penalty matching, and
the playing-direction invariant) with small hand-built fixtures, so they
run without any downloaded data. The file-reading paths are covered by the
integration tests in tests/test_loaders.py.
"""
import pandas as pd
import pytest

from src.data.metrica_loader import (
    _assert_teams_attack_opposite_goals,
    _penalty_keys,
    is_goal_subtype,
)


class TestIsGoalSubtype:
    """Metrica spells the outcome into Subtype; these are the real values
    present in the downloaded sample data, not invented ones."""

    @pytest.mark.parametrize(
        "subtype",
        ["ON TARGET-GOAL", "HEAD-ON TARGET-GOAL", "GOAL"],
    )
    def test_real_goal_subtypes_are_goals(self, subtype):
        assert is_goal_subtype(pd.Series([subtype])).iloc[0]

    @pytest.mark.parametrize(
        "subtype",
        [
            "ON TARGET-SAVED",
            "OFF TARGET-OUT",
            "HEAD-OFF TARGET-OUT",
            "BLOCKED",
            "HEAD-WOODWORK-OUT",
            "OFF TARGET",
        ],
    )
    def test_real_non_goal_subtypes_are_not_goals(self, subtype):
        assert not is_goal_subtype(pd.Series([subtype])).iloc[0]

    def test_null_subtype_is_not_a_goal(self):
        assert not is_goal_subtype(pd.Series([None])).iloc[0]

    def test_preserves_index(self):
        subtype = pd.Series(["BLOCKED", "ON TARGET-GOAL"], index=[7, 9])
        result = is_goal_subtype(subtype)
        assert list(result.index) == [7, 9]
        assert result.tolist() == [False, True]


class TestPenaltyKeys:
    """Metrica records a penalty twice: a SET PIECE row with null
    coordinates, and the SHOT itself at the same frame."""

    def _events(self):
        return pd.DataFrame(
            {
                "team": ["Away", "Away", "Home", "Home"],
                "type": ["SET PIECE", "SHOT", "SET PIECE", "SHOT"],
                "subtype": ["PENALTY", "ON TARGET-GOAL", "CORNER KICK", "BLOCKED"],
                "period": [2, 2, 1, 1],
                "start_frame": [115009, 115009, 500, 505],
            }
        )

    def test_finds_the_penalty_frame(self):
        assert _penalty_keys(self._events()) == {("Away", 2, 115009)}

    def test_ignores_other_set_pieces(self):
        keys = _penalty_keys(self._events())
        assert ("Home", 1, 500) not in keys

    def test_no_penalties_gives_empty_set(self):
        events = self._events()
        events = events[events["subtype"] != "PENALTY"]
        assert _penalty_keys(events) == set()

    def test_tolerates_null_subtypes(self):
        events = self._events()
        events.loc[0, "subtype"] = None
        assert _penalty_keys(events) == set()


class TestTeamsAttackOppositeGoals:
    """The invariant that catches a failed direction heuristic loudly. An
    earlier mean-position heuristic put both of game 1's teams attacking the
    same goal in period 1; this is the check that would have caught it."""

    def test_opposite_directions_pass(self):
        directions = {
            ("Home", 1): True,
            ("Away", 1): False,
            ("Home", 2): False,
            ("Away", 2): True,
        }
        _assert_teams_attack_opposite_goals(1, directions)

    def test_same_direction_raises(self):
        directions = {("Home", 1): True, ("Away", 1): True}
        with pytest.raises(ValueError, match="attacking the same goal"):
            _assert_teams_attack_opposite_goals(1, directions)

    def test_reports_the_offending_period(self):
        directions = {
            ("Home", 1): True,
            ("Away", 1): False,
            ("Home", 2): False,
            ("Away", 2): False,
        }
        with pytest.raises(ValueError, match="period 2"):
            _assert_teams_attack_opposite_goals(3, directions)

    def test_incomplete_data_is_skipped_not_flagged(self):
        # One team missing from a period is a data gap, not a contradiction.
        _assert_teams_attack_opposite_goals(1, {("Home", 1): True})
