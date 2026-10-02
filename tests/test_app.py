"""Tests for app/streamlit_app.py.

Uses Streamlit's own AppTest harness, which executes the app script in
process and surfaces any exception it raises. An HTTP 200 from the server
proves only that the port is open. The script body does not run until a
session connects, so a broken app serves 200 happily. These run it.

Marked integration because the app reads the pipeline's outputs from
data/processed/.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402

# Absolute: AppTest resolves relative paths against the calling file, not cwd.
APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")
TIMEOUT_S = 120


@pytest.fixture(scope="module")
def app():
    runner = AppTest.from_file(APP, default_timeout=TIMEOUT_S).run()
    return runner


@pytest.mark.integration
def test_app_runs_without_exceptions(app):
    assert not app.exception, [str(e) for e in app.exception]


@pytest.mark.integration
def test_app_has_the_four_tabs(app):
    labels = [t.label for t in app.tabs]
    assert len(labels) == 4, labels
    assert any("Summary" in label for label in labels)
    assert any("Defender context" in label for label in labels)


@pytest.mark.integration
def test_app_shows_the_public_data_disclaimer(app):
    # METHODOLOGY.md principles #1 and #2: the app must not imply club data or a
    # validated result. If this warning ever disappears, that is a problem.
    warnings = " ".join(w.value for w in app.warning)
    assert "public sample data" in warnings
    assert "not a" in warnings.lower()


@pytest.mark.integration
def test_shot_selector_is_populated(app):
    assert app.selectbox, "no shot selector rendered"
    assert len(app.selectbox[0].options) > 10


@pytest.mark.integration
def test_selecting_a_different_shot_does_not_break(app):
    runner = AppTest.from_file(APP, default_timeout=TIMEOUT_S).run()
    runner.selectbox[0].select_index(3).run()
    assert not runner.exception, [str(e) for e in runner.exception]


@pytest.mark.integration
def test_switching_the_control_model_does_not_break():
    runner = AppTest.from_file(APP, default_timeout=TIMEOUT_S).run()
    assert runner.radio, "no control-model toggle rendered"
    runner.radio[0].set_value("voronoi").run()
    assert not runner.exception, [str(e) for e in runner.exception]


@pytest.mark.integration
def test_summary_tab_reports_both_sample_sizes():
    """Both claims must be visible and distinguishable in the app.

    The project makes two different claims on two different samples (an
    underpowered null on 9 goals and a real effect on 299), and
    conflating them would be the exact overclaim METHODOLOGY.md principle #2
    rules out.
    """
    runner = AppTest.from_file(APP, default_timeout=TIMEOUT_S).run()
    body = " ".join(m.value for m in runner.markdown)
    assert "9 goals" in body
    assert "299 goals" in body


@pytest.mark.integration
def test_app_does_not_claim_parity_with_statsbomb():
    """METHODOLOGY.md principle #2: no overclaiming. The app must keep saying what it is not."""
    runner = AppTest.from_file(APP, default_timeout=TIMEOUT_S).run()
    body = " ".join(m.value for m in runner.markdown).lower()
    assert "do not match statsbomb" in body or "not a validated" in body


@pytest.mark.integration
def test_app_script_runs_standalone_from_an_unrelated_directory():
    """Catches the two bugs AppTest cannot see.

    AppTest imports the app in-process, under pytest, where the repo root is
    already on sys.path and dataframes are never serialized for a browser.
    A real `streamlit run` does neither: it puts the SCRIPT's directory on
    sys.path (so a bare `import src` raises ModuleNotFoundError) and it ships
    every dataframe through Arrow (which rejects a column of mixed types).
    Both shipped, and both were invisible to every other test in this file.

    Running the script with a plain interpreter from an unrelated working
    directory reproduces that environment closely enough to catch them.
    """
    import os
    import subprocess
    import sys

    environment = {
        **os.environ,
        "MPLCONFIGDIR": os.environ.get("TMPDIR", "/tmp") + "/mpl",
    }
    completed = subprocess.run(
        [sys.executable, APP],
        cwd=str(PROJECT_ROOT.parent),
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
    )
    output = completed.stdout + completed.stderr
    assert "ModuleNotFoundError" not in output, output[-2000:]
    assert "Traceback" not in output, output[-2000:]


@pytest.mark.integration
def test_app_uses_no_removed_streamlit_apis():
    """`use_container_width` was slated for removal after 2025-12-31."""
    source = Path(APP).read_text()
    assert "use_container_width" not in source
