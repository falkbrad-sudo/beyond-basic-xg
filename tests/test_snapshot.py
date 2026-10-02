"""Tests for the committed data snapshot the Streamlit app runs on.

The app reads a handful of small pipeline outputs from data/processed/.
They are committed so the app works on a fresh clone and on Streamlit
Community Cloud, which never runs the pipeline. These tests catch the two
ways that can break silently: the app starting to read a file that is not
committed (the deploy then fails while everything works locally), and names
creeping back into a committed file.
"""
import re
import subprocess

import pandas as pd
import pytest

from src.config import PROJECT_ROOT

APP = PROJECT_ROOT / "app" / "streamlit_app.py"
PROCESSED = PROJECT_ROOT / "data" / "processed"


def _files_the_app_reads() -> set[str]:
    return set(re.findall(r'"([a-z_]+\.(?:parquet|csv|json))"', APP.read_text()))


def _committed_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "data/processed"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    ).stdout.split()
    return {name.rsplit("/", 1)[-1] for name in out}


def test_the_app_reads_at_least_the_known_files():
    # Guards the regex below against silently matching nothing.
    assert len(_files_the_app_reads()) >= 10


def test_every_file_the_app_reads_exists():
    missing = [f for f in _files_the_app_reads() if not (PROCESSED / f).exists()]
    assert not missing, f"app reads files that are not present: {missing}"


def test_every_file_the_app_reads_is_committed():
    """If this fails, the app works locally but breaks on a cloud deploy.

    Fix by adding a matching `!data/processed/<file>` line to .gitignore.
    """
    committed = _committed_files()
    if not committed:
        pytest.skip("not running inside a git checkout")
    missing = sorted(_files_the_app_reads() - committed)
    assert not missing, f"app reads uncommitted files: {missing}"


def test_published_statsbomb_shots_carry_no_names():
    shots = pd.read_parquet(PROCESSED / "statsbomb_shots_with_geometry.parquet")
    assert "player" not in shots.columns
    assert "team" not in shots.columns


def test_published_metrica_data_uses_anonymized_labels_only():
    for name in ("metrica_shots_compared.parquet", "metrica_shot_frames.parquet"):
        frame = pd.read_parquet(PROCESSED / name)
        assert set(frame["team"].dropna()) <= {"Home", "Away", "ball"}
