"""Interactive demo: Beyond Basic xG.

Run with: streamlit run app/streamlit_app.py

Goal: let a viewer pick a shot and see, side by side, why the basic model
and the pitch-control-enhanced model rated it differently. The pitch diagram
at the moment of the shot explains more here than any table of numbers.

Everything is read from data/processed/, written by `python -m src.pipeline`.
Nothing is refit or re-parsed on a page load: Metrica's tracking files are
~145k frames per team per game, and re-reading them interactively would make
the app unusable. The one thing computed live is the pitch-control surface
for the selected shot, which is a grid evaluation over 22 players and takes
milliseconds.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# `streamlit run app/streamlit_app.py` puts THIS file's directory (app/) on
# sys.path, not the repo root, so a bare `import src` fails with
# ModuleNotFoundError. Putting the root on the path here keeps the documented
# one-command launch working from any shell and any working directory,
# without requiring the project to be pip-installed first.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp") + "/mpl")

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import load_config, resolve_path  # noqa: E402
from src.features.pitch_control_at_shot import SPEARMAN, VORONOI  # noqa: E402
from src.viz.inference_plots import (  # noqa: E402
    plot_coefficient_intervals,
    plot_power_curve,
)
from src.viz.pitch_control_plots import plot_pitch_control_at_shot  # noqa: E402
from src.viz.validation_plots import (  # noqa: E402
    plot_calibration,
    plot_xg_agreement,
)

st.set_page_config(page_title="Beyond Basic xG", layout="wide")

PIPELINE_HINT = (
    "Run `python -m src.pipeline` first. It writes everything this app "
    "reads into `data/processed/`."
)


def _processed_dir() -> Path:
    return resolve_path(load_config()["paths"]["data_processed"])


@st.cache_data(show_spinner=False)
def load_metrica_comparison() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Shots with both models' predictions, their tracking frames, and summary."""
    processed = _processed_dir()
    shots = pd.read_parquet(processed / "metrica_shots_compared.parquet")
    frames = pd.read_parquet(processed / "metrica_shot_frames.parquet")
    with open(processed / "metrica_comparison_summary.json") as f:
        summary = json.load(f)
    return shots, frames, summary


@st.cache_data(show_spinner=False)
def load_statsbomb_validation() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The large-sample validation shots, calibration table and summary."""
    processed = _processed_dir()
    shots = pd.read_parquet(processed / "statsbomb_shots_with_geometry.parquet")
    calibration = pd.read_csv(processed / "basic_model_calibration.csv")
    with open(processed / "basic_model_validation.json") as f:
        summary = json.load(f)
    return shots, calibration, summary


@st.cache_data(show_spinner=False)
def load_context_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Large-sample freeze-frame results: intervals, power, summary."""
    processed = _processed_dir()
    metrica_ci = pd.read_csv(processed / "metrica_coefficient_intervals.csv")
    context_ci = pd.read_csv(processed / "statsbomb_context_intervals.csv")
    power = pd.read_csv(processed / "statsbomb_power_analysis.csv")
    with open(processed / "statsbomb_context_summary.json") as f:
        summary = json.load(f)
    return metrica_ci, context_ci, power, summary


def render_context_tab() -> None:
    """The large-sample test: the same question, 299 goals instead of 9."""
    try:
        metrica_ci, context_ci, power, summary = load_context_results()
    except FileNotFoundError:
        st.error(f"Context-model output not found. {PIPELINE_HINT}")
        return

    comparison = summary["comparison"]
    st.subheader("Does defender position matter? Asked of 299 goals")
    st.caption(
        "StatsBomb's shot events carry freeze frames listing every player in "
        "camera view at the instant of the shot. That allows the same two "
        "context features to be fit on 3,490 shots instead of 66."
    )

    metrics = st.columns(4)
    metrics[0].metric("Shots", f"{comparison['n_shots']:,}")
    metrics[1].metric("Goals", comparison["n_goals"])
    metrics[2].metric(
        "AUC, geometry only", f"{comparison['basic_auc']:.4f}"
    )
    metrics[3].metric(
        "AUC, + defender context",
        f"{comparison['enhanced_auc']:.4f}",
        delta=f"{comparison['enhanced_auc'] - comparison['basic_auc']:+.4f}",
    )

    st.pyplot(
        plot_coefficient_intervals(
            {
                "Metrica tracking (66 shots, 9 goals)": metrica_ci,
                "StatsBomb freeze frames (3,490 shots, 299 goals)": context_ci,
            },
        ),
        width="stretch",
    )

    st.markdown(
        "On the large sample both context features point the way the "
        "hypothesis predicted: **more defenders in the shooting lane lowers a "
        "chance, and a goalkeeper further from the centre of goal raises it.** "
        "Neither interval touches zero. The 66-shot tracking sample was too "
        "small to answer the same question."
    )

    st.divider()
    st.markdown("#### How much data the question needed")
    reference = summary["power_reference"]
    st.pyplot(
        plot_power_curve(
            se_reference=float(reference["se_reference"]),
            n_reference=int(reference["n_reference"]),
            observed_effect=abs(
                float(
                    context_ci.set_index("feature").loc[
                        reference["feature"], "coefficient"
                    ]
                )
            ),
            goal_rate=float(reference["goal_rate"]),
            sample_markers={
                "Metrica": (66, 9),
                "NWSL 2023": (int(reference["n_reference"]), 299),
            },
        ),
        width="stretch",
    )
    st.dataframe(power, width="stretch", hide_index=True)


def _shot_label(row: pd.Series) -> str:
    """A one-line description that identifies a shot to a human reader."""
    outcome = "GOAL" if row["is_goal"] else "no goal"
    return (
        f"Game {int(row['game_id'])} · {row['team']} · "
        f"{int(row['time_s']) // 60}' · {row['player']} · "
        f"{row['distance_to_goal']:.1f}m · {outcome} "
        f"(models differ by {row['prediction_diff']:+.3f})"
    )


def _format_value(value) -> str:
    """Render a feature value for display, trimming float noise."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _explain(row: pd.Series) -> str:
    """Plain-language account of why the two models differ on this shot."""
    diff = row["prediction_diff"]
    direction = "higher" if diff > 0 else "lower"
    lane = row["pitch_control"]
    defenders = int(row["defenders_in_cone"])
    keeper = row["goalkeeper_distance"]

    defender_phrase = {
        0: "no defender inside the shooting lane",
        1: "one defender inside the shooting lane",
    }.get(defenders, f"{defenders} defenders inside the shooting lane")

    keeper_phrase = (
        f"the goalkeeper {keeper:.1f}m off their goal centre"
        if keeper >= 4
        else f"the goalkeeper close to their line ({keeper:.1f}m)"
    )

    return (
        f"The basic model sees only **{row['distance_to_goal']:.1f}m** and a "
        f"shot angle of **{row['angle_to_goal']:.2f} rad**, and rates this "
        f"**{row['basic_xg_pred']:.3f}**.\n\n"
        f"The enhanced model additionally sees that the shooting team "
        f"controlled **{lane:.0%}** of the triangle between the ball and the "
        f"goalposts, with {defender_phrase} and {keeper_phrase}. It rates the "
        f"shot **{row['enhanced_xg_pred']:.3f}**, which is "
        f"**{abs(diff):.3f} {direction}**.\n\n"
        f"That difference is small, and on this sample it is small for every "
        f"shot. The *Summary* tab explains why, so the direction of this one "
        f"gap should not be read as a finding."
    )


def render_comparison_tab() -> None:
    try:
        shots, frames, summary = load_metrica_comparison()
    except FileNotFoundError:
        st.error(f"Metrica comparison output not found. {PIPELINE_HINT}")
        return

    st.subheader("Pick a shot")
    st.caption(
        "Ordered by how much the two models disagree, so the most "
        "interesting cases are at the top."
    )

    ordered = shots.reindex(
        shots["prediction_diff"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    controls = st.columns([3, 1])
    with controls[0]:
        choice = st.selectbox(
            "Shot",
            options=range(len(ordered)),
            format_func=lambda i: _shot_label(ordered.iloc[i]),
            label_visibility="collapsed",
        )
    with controls[1]:
        method = st.radio(
            "Control model",
            options=[SPEARMAN, VORONOI],
            format_func=lambda m: {
                SPEARMAN: "Spearman (soft)",
                VORONOI: "Voronoi (hard)",
            }[m],
            horizontal=False,
        )

    row = ordered.iloc[choice]

    metrics = st.columns(4)
    metrics[0].metric("Basic xG", f"{row['basic_xg_pred']:.3f}")
    metrics[1].metric(
        "Enhanced xG",
        f"{row['enhanced_xg_pred']:.3f}",
        delta=f"{row['prediction_diff']:+.3f}",
    )
    metrics[2].metric("Outcome", "Goal" if row["is_goal"] else "No goal")
    metrics[3].metric("Lane control", f"{row['pitch_control']:.0%}")

    frame = frames[
        (frames["game_id"] == row["game_id"]) & (frames["frame"] == row["frame"])
    ]
    if frame.empty:
        st.warning("No tracking frame stored for this shot.")
        return

    left, right = st.columns([3, 2])
    with left:
        figure = plot_pitch_control_at_shot(
            frame,
            shot_x=float(row["x"]),
            shot_y=float(row["y"]),
            shooting_team=row["team"],
            method=method,
            title=None,
            subtitle=None,
        )
        st.pyplot(figure, width="stretch")
    with right:
        st.markdown("#### Why the models differ here")
        st.markdown(_explain(row))
        with st.expander("All features for this shot"):
            fields = [
                "game_id", "team", "period", "frame", "player", "subtype",
                "distance_to_goal", "angle_to_goal", "pitch_control",
                "pitch_control_spearman", "pitch_control_at_shot",
                "defenders_in_cone", "goalkeeper_distance",
                "basic_xg_pred", "enhanced_xg_pred", "prediction_diff",
            ]
            # Values are forced to strings because this column mixes ints,
            # floats and text, and Arrow -- which Streamlit uses to ship
            # dataframes to the browser -- infers one type per column and
            # fails on the first value that disagrees.
            st.dataframe(
                pd.DataFrame(
                    {
                        "feature": fields,
                        "value": [_format_value(row[f]) for f in fields],
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    st.divider()
    st.markdown("#### The five biggest disagreements, in full")
    st.caption(
        "The project's main output on this sample: specific, nameable shots "
        "rather than an aggregate score."
    )
    st.dataframe(
        ordered.head(5)[
            [
                "game_id", "team", "player", "subtype", "distance_to_goal",
                "angle_to_goal", "pitch_control", "defenders_in_cone",
                "goalkeeper_distance", "is_goal", "basic_xg_pred",
                "enhanced_xg_pred", "prediction_diff",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def render_validation_tab() -> None:
    try:
        shots, calibration, summary = load_statsbomb_validation()
    except FileNotFoundError:
        st.error(f"StatsBomb validation output not found. {PIPELINE_HINT}")
        return

    validation = summary["validation"]
    st.subheader("Basic model vs. StatsBomb's published xG")
    st.caption(
        "Why you can trust the baseline before any pitch-control claim: a "
        "distance-and-angle-only model, checked against StatsBomb's own xG "
        "across a full NWSL season."
    )

    metrics = st.columns(4)
    metrics[0].metric("Shots", f"{validation['n_shots']:,}")
    metrics[1].metric("Spearman rho", f"{validation['spearman_rho']:.3f}")
    metrics[2].metric("Pearson r", f"{validation['pearson_r']:.3f}")
    metrics[3].metric("Mean |difference|", f"{validation['mean_abs_difference']:.3f}")

    left, right = st.columns(2)
    with left:
        st.pyplot(
            plot_xg_agreement(shots.assign(own_xg=_own_xg(shots))),
            width="stretch",
        )
    with right:
        st.pyplot(plot_calibration(calibration), width="stretch")

    st.markdown(
        "A correlation this high means distance and angle carry most of the "
        "signal. It does **not** mean this model is equivalent to StatsBomb's, "
        "which uses defender and goalkeeper positions that this baseline does "
        "not see. The *Defender context* tab tests that directly."
    )


@st.cache_data(show_spinner=False)
def _own_xg(shots: pd.DataFrame) -> pd.Series:
    """Refit the baseline to score the validation panel's shots."""
    from src.models.basic_xg_model import BASIC_FEATURES, fit_basic_model

    model, _ = fit_basic_model(shots)
    return pd.Series(
        model.predict_proba(shots[BASIC_FEATURES].to_numpy(float))[:, 1],
        index=shots.index,
    )


def render_summary_tab() -> None:
    try:
        _, _, summary = load_metrica_comparison()
        metrica_ci, _, power, context_summary = load_context_results()
    except FileNotFoundError:
        st.error(f"Pipeline output not found. {PIPELINE_HINT}")
        return

    comparison = summary["comparison"]
    context = context_summary["comparison"]
    pitch_control = metrica_ci.set_index("feature").loc["pitch_control"]
    moderate = power.set_index("effect_size_per_sd").loc[0.20]
    st.subheader("What this project shows, and what it does not")

    metrics = st.columns(4)
    metrics[0].metric("Tracking sample", f"{comparison['n_goals']} goals")
    metrics[1].metric("Freeze-frame sample", f"{context['n_goals']} goals")
    metrics[2].metric(
        "AUC gain from defender context",
        f"+{context['enhanced_auc'] - context['basic_auc']:.4f}",
    )
    metrics[3].metric(
        "Power of the tracking sample",
        f"{100 * moderate['power_with_66_shots']:.0f}%",
    )

    st.markdown(
        f"""
**Defensive context does change shot quality, but only a sample with enough
goals can show it.**

On the **66-shot Metrica tracking sample (9 goals)**, adding pitch control
changed almost nothing. The two models' predictions correlate at
{comparison['prediction_correlation']:.4f}, and the 95% bootstrap interval
for the pitch-control coefficient is
[{pitch_control['ci_low']:.2f}, {pitch_control['ci_high']:.2f}], which
crosses zero. That does not show the idea is wrong. It shows this sample
cannot answer the question.

On the **3,490-shot StatsBomb sample (299 goals)**, the same kind of
defensive information has a clear effect. Defenders in the shooting lane and
goalkeeper position both have intervals that exclude zero, both with the
predicted sign, and together they raise AUC from {context['basic_auc']:.4f}
to {context['enhanced_auc']:.4f}.

**Why the small sample could not answer it.** Detecting an effect of the
size actually observed takes several hundred shots. The Metrica sample had
about **{100 * moderate['power_with_66_shots']:.0f}% power** for a moderate
effect, so its null result says little either way, and reading it as "pitch
control doesn't matter" would be wrong.

**What these results do not show.**

- They do not match StatsBomb's own xG model, which reaches an AUC of 0.788
  on the same shots. Adding defender context closes about half the gap
  between a geometry-only model and theirs.
- They do not validate the pitch-control model. Spearman control and Voronoi
  need continuous tracking, so they are still fit on 66 shots. A freeze frame
  has no velocities and shows only the players in camera view.
- They are not causal. These are associations in observational data.
- The lane-control feature partly repeats shot angle (r = 0.35) by
  construction, a design flaw to fix before drawing conclusions about pitch
  control specifically.
"""
    )

    st.info(comparison.get("performance_metrics_withheld", ""), icon="ℹ️")


def main() -> None:
    st.title("Beyond Basic xG")
    st.caption(
        "Does defensive pressure and space change how good a shot really was? "
        "A pitch-control-informed take on Expected Goals."
    )

    st.warning(
        "Built on public sample data: Metrica Sports open tracking data for "
        "the pitch-control model, and StatsBomb open 2023 NWSL data for "
        "validation and the defender-context model. No club's proprietary "
        "data is used. The pitch-control model is fit on 66 shots containing "
        "9 goals and is a methodology demonstration, not a validated result. "
        "The defender-context model is fit on 3,490 shots with 299 goals and "
        "is a real result. These are different claims; the Summary tab "
        "explains how.",
        icon="ℹ️",
    )

    tab_compare, tab_validation, tab_context, tab_summary = st.tabs(
        [
            "Shot-by-shot comparison (Metrica)",
            "Basic model validation (StatsBomb)",
            "Defender context (299 goals)",
            "Summary",
        ]
    )
    with tab_compare:
        render_comparison_tab()
    with tab_validation:
        render_validation_tab()
    with tab_context:
        render_context_tab()
    with tab_summary:
        render_summary_tab()


# Streamlit executes this file top to bottom on every interaction, so the
# entry point is called unconditionally rather than guarded by __main__.
main()
