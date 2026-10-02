"""End-to-end orchestration: load -> clean -> features -> models -> compare.

Run as `python -m src.pipeline`.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data import metrica_loader, statsbomb_loader
from src.data.cleaning import (
    metrica_to_meters,
    normalize_playing_direction,
    statsbomb_to_meters,
    validate_coordinates,
)
from src.features.geometry import add_geometry_features
from src.features.pitch_control_at_shot import add_pitch_control_features
from src.features.shot_context import (
    add_context_features,
    count_defenders_in_cone,
    goalkeeper_distance_from_goal_center,
)
from src.models.basic_xg_model import (
    BASIC_FEATURES,
    fit_basic_model,
    validate_against_statsbomb_xg,
)
from src.models.enhanced_xg_model import (
    ENHANCED_FEATURES,
    ENHANCED_FEATURES_FULL,
    enhanced_model_diagnostics,
    fit_enhanced_model,
)
from src.models.evaluate import (
    compare_predictions,
    find_biggest_disagreements,
    summary_statistics,
)
from src.models.power import power_analysis
from src.models.uncertainty import (
    bootstrap_coefficients,
    bootstrap_prediction_difference,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Maximum tolerated gap between a shot's event coordinates and the tracked
# position of the player credited with it. The sample's median is 0.25m, so
# 3m is loose; it exists to catch frames where the event/tracking sync has
# actually failed rather than to filter noise. Exactly one shot across the
# three games exceeds it.
MAX_SHOOTER_POSITION_ERROR_M = 3.0

# Columns kept out of every output file that is committed to the repo.
PUBLISHED_NAME_COLUMNS = ["player", "team"]


def load_metrica_shots(cfg: dict) -> pd.DataFrame:
    """All Metrica shots, in the attacking frame, with geometry features.

    Shared by the shot-count report and the (not yet implemented) Metrica
    comparison. Converts units, normalizes playing direction (without which
    roughly three quarters of shots would have their distance measured to the
    wrong goal), then validates and adds geometry.
    """
    games = metrica_loader.list_available_games()
    if not games:
        raise FileNotFoundError(
            "No Metrica games found. Run scripts/download_data.sh first."
        )

    frames = [metrica_loader.extract_shot_events(game_id) for game_id in games]
    shots = pd.concat(frames, ignore_index=True)

    shots = metrica_to_meters(shots, "x", "y")
    shots = normalize_playing_direction(shots, attacking_right=shots["attacking_right"])
    validate_coordinates(shots, "x", "y")
    return add_geometry_features(shots)


def report_metrica_shot_counts(cfg: dict) -> pd.DataFrame:
    """Log how many shots the Metrica sample actually contains, per game.

    This number governs what the whole project may claim (METHODOLOGY.md
    principle #2), so it is surfaced explicitly rather than left for someone
    to infer from a model summary.
    """
    shots = load_metrica_shots(cfg)
    per_game = (
        shots.groupby("game_id")
        .agg(
            shots=("is_goal", "size"),
            goals=("is_goal", "sum"),
            penalties=("is_penalty", "sum"),
        )
        .reset_index()
    )
    logger.info("Metrica shot counts per game:\n%s", per_game.to_string(index=False))
    logger.info(
        "Metrica TOTAL: %d shots, %d goals, %d penalties across %d games.",
        len(shots),
        int(shots["is_goal"].sum()),
        int(shots["is_penalty"].sum()),
        shots["game_id"].nunique(),
    )
    return per_game


def run_statsbomb_validation(cfg: dict) -> dict:
    """Fit and validate the basic model on the large StatsBomb NWSL sample.

    This is the gate the README puts before the Metrica work: the basic model
    has to look reasonable against StatsBomb's own published xG on hundreds
    of shots before it is worth extending with pitch control on a few dozen.
    """
    logger.info("Running StatsBomb basic-model validation...")

    season = statsbomb_loader.verify_season_available()
    logger.info(
        "Verified StatsBomb season: %s %s (competition_id=%s, season_id=%s)",
        season.get("competition_name"),
        season.get("season_name"),
        season.get("competition_id"),
        season.get("season_id"),
    )

    shots = statsbomb_loader.get_all_shots()
    logger.info("Pulled %d shots from the 2023 NWSL season.", len(shots))

    # Penalties have fixed geometry and a conversion rate far above open
    # play; leaving them in would flatter a distance/angle model.
    open_play = shots[~shots["is_penalty"]].copy()
    logger.info(
        "Excluded %d penalties, leaving %d shots.",
        len(shots) - len(open_play),
        len(open_play),
    )

    open_play = statsbomb_to_meters(open_play, "x", "y")
    validate_coordinates(open_play, "x", "y")
    open_play = add_geometry_features(open_play)

    model, diagnostics = fit_basic_model(open_play, feature_cols=BASIC_FEATURES)
    logger.info("Basic model fit diagnostics: %s", diagnostics)

    results = validate_against_statsbomb_xg(open_play, model, BASIC_FEATURES)
    logger.info(
        "Own basic xG vs StatsBomb xG on %d shots: pearson r=%.3f, "
        "spearman rho=%.3f, mean|diff|=%.3f",
        results["n_shots"],
        results["pearson_r"],
        results["spearman_rho"],
        results["mean_abs_difference"],
    )
    logger.info(
        "Calibration (own model, by predicted decile):\n%s",
        results["calibration"].to_string(index=False),
    )
    logger.info(
        "Largest disagreements with StatsBomb:\n%s",
        results["largest_disagreements"].to_string(index=False),
    )
    logger.info(
        "Read the correlation as evidence that distance and angle carry most "
        "of the signal, not as parity with StatsBomb's model, which uses "
        "defender positions the geometry-only baseline does not see."
    )

    open_play = build_statsbomb_context_features(cfg, open_play)
    _save_validation_results(cfg, open_play, diagnostics, results)
    results["shots"] = open_play
    return results


def _save_validation_results(
    cfg: dict, shots: pd.DataFrame, diagnostics: dict, results: dict
) -> None:
    """Write the validation outputs to paths.data_processed."""
    out_dir = resolve_path(cfg["paths"]["data_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Player and team names are dropped from this file because it is
    # committed as part of the app snapshot, and nothing downstream reads
    # them. Dropping them here, rather than only in the published copy,
    # means a re-run of the pipeline cannot put them back.
    shots.drop(columns=PUBLISHED_NAME_COLUMNS, errors="ignore").to_parquet(
        out_dir / "statsbomb_shots_with_geometry.parquet", index=False
    )
    results["calibration"].to_csv(out_dir / "basic_model_calibration.csv", index=False)
    results["largest_disagreements"].to_csv(
        out_dir / "basic_model_largest_disagreements.csv", index=False
    )

    summary = {
        "fit_diagnostics": diagnostics,
        "validation": {
            k: v for k, v in results.items()
            if not isinstance(v, pd.DataFrame)
        },
    }
    with open(out_dir / "basic_model_validation.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved validation outputs to %s", out_dir)


def build_metrica_feature_table(cfg: dict) -> pd.DataFrame:
    """Metrica shots with geometry, pitch-control and context features.

    Each shot is joined to its own tracking frame via
    metrica_loader.get_normalized_tracking_frame_at_shot(), which converts
    units and rotates playing direction together; see that function for why
    those two steps must not be separated.

    Also computes 'shooter_position_error_m': how far the event's shot
    coordinates sit from the tracked position of the player credited with the
    shot. It is a check on the events-to-tracking sync, which is the join the
    whole project rests on, and it is kept as a column so the exclusion it
    drives stays visible rather than silent.
    """
    shots = load_metrica_shots(cfg)

    shots["frame_positions"] = [
        metrica_loader.get_normalized_tracking_frame_at_shot(
            int(row.game_id), int(row.frame), bool(row.attacking_right)
        )
        for row in shots.itertuples()
    ]

    errors = []
    for row in shots.itertuples():
        shooter = row.frame_positions[
            row.frame_positions["player"] == str(row.player_id)
        ]
        errors.append(
            float(np.hypot(shooter["x"].iloc[0] - row.x, shooter["y"].iloc[0] - row.y))
            if not shooter.empty
            else np.nan
        )
    shots["shooter_position_error_m"] = errors

    shots = add_pitch_control_features(shots)
    shots = add_context_features(shots)
    return shots


def _usable_metrica_shots(shots: pd.DataFrame) -> pd.DataFrame:
    """Drop penalties and shots whose event/tracking sync cannot be trusted."""
    penalties = int(shots["is_penalty"].sum())
    bad_sync = (
        shots["shooter_position_error_m"].isna()
        | (shots["shooter_position_error_m"] > MAX_SHOOTER_POSITION_ERROR_M)
    )
    logger.info(
        "Excluding %d penalty and %d shot(s) whose shooter is more than %.0fm "
        "from the event location (event/tracking sync failure).",
        penalties,
        int(bad_sync.sum()),
        MAX_SHOOTER_POSITION_ERROR_M,
    )
    for row in shots[bad_sync].itertuples():
        logger.info(
            "  excluded for sync: game %s frame %s player %s (%.1fm off)",
            row.game_id, row.frame, row.player_id, row.shooter_position_error_m,
        )
    return shots[~shots["is_penalty"] & ~bad_sync].reset_index(drop=True)


# Features computable from a StatsBomb freeze frame. Pitch control is
# deliberately absent: a freeze frame has no velocities and only the players
# in camera view, so time-to-intercept is undefined (METHODOLOGY.md principle #3).
STATSBOMB_CONTEXT_FEATURES = [
    "distance_to_goal",
    "angle_to_goal",
    "defenders_in_cone",
    "goalkeeper_distance",
]


def build_statsbomb_context_features(cfg: dict, shots: pd.DataFrame) -> pd.DataFrame:
    """Add defender-position features to StatsBomb shots from freeze frames.

    This is the step that lifts the shot-context question off a 9-goal
    sample and onto a 299-goal one. The per-shot frames are converted to
    meters and handed to exactly the same `shot_context` functions the
    Metrica path uses, with no StatsBomb-specific feature code. That is the
    standardized-schema claim `club_loader.py` makes, demonstrated.

    No playing-direction rotation is applied or needed: StatsBomb's x axis
    already points at the goal the shooting team is attacking.
    """
    frames = statsbomb_loader.get_shot_freeze_frames()
    frames = statsbomb_to_meters(frames, "x", "y")
    logger.info(
        "Loaded %d freeze-frame player rows across %d shots.",
        len(frames),
        frames["shot_id"].nunique(),
    )

    by_shot = dict(tuple(frames.groupby("shot_id")))
    records = []
    for row in shots.itertuples():
        frame = by_shot.get(row.shot_id)
        if frame is None or frame.empty:
            records.append({"defenders_in_cone": np.nan, "goalkeeper_distance": np.nan})
            continue
        try:
            records.append(
                {
                    "defenders_in_cone": count_defenders_in_cone(
                        frame, row.x, row.y, statsbomb_loader.DEFENDING
                    ),
                    "goalkeeper_distance": goalkeeper_distance_from_goal_center(
                        frame, statsbomb_loader.DEFENDING
                    ),
                }
            )
        except ValueError:
            # No defender visible at all, so neither feature is defined.
            records.append({"defenders_in_cone": np.nan, "goalkeeper_distance": np.nan})

    out = shots.copy()
    computed = pd.DataFrame(records, index=out.index)
    for column in computed.columns:
        out[column] = computed[column]

    complete = out[STATSBOMB_CONTEXT_FEATURES].notna().all(axis=1)
    logger.info(
        "Context features computed for %d of %d shots (%.1f%%); %d lack a "
        "usable freeze frame and are dropped from the context model.",
        int(complete.sum()), len(out), 100 * complete.mean(),
        int((~complete).sum()),
    )
    return out


def run_statsbomb_context_model(cfg: dict, shots: pd.DataFrame) -> dict:
    """Fit geometry + freeze-frame context on the large sample, with intervals.

    The Metrica comparison cannot distinguish two models on 9 goals. This
    can: same context features, 299 goals. Reported with bootstrap
    confidence intervals rather than point estimates alone.
    """
    logger.info("Fitting the freeze-frame context model on the StatsBomb sample...")

    usable = shots.dropna(subset=STATSBOMB_CONTEXT_FEATURES).copy()
    logger.info(
        "Context model sample: %d shots, %d goals.",
        len(usable), int(usable["is_goal"].sum()),
    )

    context_model, diagnostics = fit_basic_model(
        usable, feature_cols=STATSBOMB_CONTEXT_FEATURES, scale=True
    )
    logger.info("Context model diagnostics: %s", diagnostics)

    geometry_model, geometry_diagnostics = fit_basic_model(usable, scale=True)
    logger.info("Geometry-only (same sample, scaled): %s", geometry_diagnostics)

    intervals = bootstrap_coefficients(
        lambda frame: fit_basic_model(
            frame, feature_cols=STATSBOMB_CONTEXT_FEATURES, holdout=False, scale=True
        ),
        usable,
        STATSBOMB_CONTEXT_FEATURES,
        n_bootstrap=400,
    )
    logger.info(
        "Bootstrap 95%% intervals (%d draws, %d failed):\n%s",
        intervals.attrs["n_bootstrap"], intervals.attrs["n_failed_draws"],
        intervals.to_string(index=False),
    )

    compared = compare_predictions(
        usable, geometry_model, context_model,
        BASIC_FEATURES, STATSBOMB_CONTEXT_FEATURES,
    )
    summary = summary_statistics(compared)
    logger.info("Geometry vs. context on the large sample: %s", summary)

    power = power_analysis(
        context_model, usable, STATSBOMB_CONTEXT_FEATURES, "goalkeeper_distance"
    )
    logger.info(
        "Power analysis for goalkeeper_distance (se=%.4f at n=%d, goal rate %.3f):"
        "\n%s",
        power.attrs["se_reference"], power.attrs["n_reference"],
        power.attrs["goal_rate"], power.to_string(index=False),
    )

    _save_context_results(cfg, usable, compared, intervals, power, summary, diagnostics)
    return {
        "intervals": intervals,
        "power": power,
        "summary": summary,
        "diagnostics": diagnostics,
    }


def _save_context_results(cfg, shots, compared, intervals, power, summary, diagnostics):
    """Persist the large-sample context results for the app and figures."""
    out_dir = resolve_path(cfg["paths"]["data_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    compared.to_parquet(out_dir / "statsbomb_context_compared.parquet", index=False)
    intervals.to_csv(out_dir / "statsbomb_context_intervals.csv", index=False)
    power.to_csv(out_dir / "statsbomb_power_analysis.csv", index=False)
    with open(out_dir / "statsbomb_context_summary.json", "w") as f:
        json.dump(
            {
                "fit_diagnostics": diagnostics,
                "comparison": summary,
                "power_reference": {
                    "feature": power.attrs["feature"],
                    "se_reference": power.attrs["se_reference"],
                    "n_reference": power.attrs["n_reference"],
                    "goal_rate": power.attrs["goal_rate"],
                },
            },
            f, indent=2, default=str,
        )
    logger.info("Saved context-model outputs to %s", out_dir)


def run_metrica_comparison(cfg: dict) -> dict:
    """Run the full pitch-control-enhanced comparison on the Metrica sample.

    Fits the basic and enhanced models on the SAME small sample (both
    scaled, so the comparison is between feature sets rather than between
    regularization regimes) and reports the shots where they disagree most.

    Deliberately does NOT report a predictive-performance comparison: see
    evaluate.summary_statistics, which withholds log-loss and AUC when the
    sample carries too few goals to support them.
    """
    logger.info("Running Metrica pitch-control comparison...")

    shots = build_metrica_feature_table(cfg)
    usable = _usable_metrica_shots(shots)
    n_goals = int(usable["is_goal"].sum())
    logger.info(
        "Fitting on %d shots containing %d goals.", len(usable), n_goals
    )

    # Both scaled: an unscaled basic model and a scaled enhanced model would
    # differ in how L2 penalizes them, and the comparison would partly
    # measure that rather than the pitch-control feature.
    basic_model, basic_diagnostics = fit_basic_model(
        usable, holdout=False, scale=True
    )
    enhanced_model = fit_enhanced_model(usable, feature_cols=ENHANCED_FEATURES)
    exploratory_model = fit_enhanced_model(
        usable, feature_cols=ENHANCED_FEATURES_FULL
    )

    logger.info("Basic (Metrica fit) diagnostics: %s", basic_diagnostics)
    logger.info(
        "Enhanced (3-feature) diagnostics: %s",
        enhanced_model_diagnostics(enhanced_model, ENHANCED_FEATURES),
    )
    logger.info(
        "Exploratory (5-feature) diagnostics: %s",
        enhanced_model_diagnostics(exploratory_model, ENHANCED_FEATURES_FULL),
    )

    compared = compare_predictions(
        usable, basic_model, enhanced_model, BASIC_FEATURES, ENHANCED_FEATURES
    )
    disagreements = find_biggest_disagreements(compared, n=5)
    summary = summary_statistics(compared)

    # The project has been asserting that these coefficients' intervals
    # include zero. Measure it instead.
    intervals = bootstrap_coefficients(
        lambda frame: fit_enhanced_model(frame, feature_cols=ENHANCED_FEATURES),
        usable,
        ENHANCED_FEATURES,
        n_bootstrap=1000,
    )
    logger.info(
        "Bootstrap 95%% intervals on the Metrica fit (%d draws, %d failed):\n%s",
        intervals.attrs["n_bootstrap"], intervals.attrs["n_failed_draws"],
        intervals.to_string(index=False),
    )

    # And how far apart the two models could plausibly be at all.
    difference = bootstrap_prediction_difference(
        usable,
        lambda frame: fit_basic_model(frame, holdout=False, scale=True),
        lambda frame: fit_enhanced_model(frame, feature_cols=ENHANCED_FEATURES),
        BASIC_FEATURES,
        ENHANCED_FEATURES,
        n_bootstrap=400,
    )
    logger.info("Bootstrap interval on mean |enhanced - basic|: %s", difference)
    summary["bootstrap_mean_abs_difference"] = difference

    logger.info("Comparison summary: %s", summary)
    logger.info(
        "Five shots where the two models disagree most:\n%s",
        disagreements.to_string(index=False),
    )
    if "performance_metrics_withheld" in summary:
        logger.info("NOTE: %s", summary["performance_metrics_withheld"])

    _save_metrica_results(
        cfg, compared, disagreements, summary, basic_diagnostics, intervals
    )
    return {
        "compared": compared,
        "disagreements": disagreements,
        "summary": summary,
        "intervals": intervals,
    }


def _save_metrica_results(
    cfg: dict,
    compared: pd.DataFrame,
    disagreements: pd.DataFrame,
    summary: dict,
    basic_diagnostics: dict,
    intervals: pd.DataFrame,
) -> None:
    """Persist the comparison so the Streamlit app never recomputes it.

    Tracking frames are flattened into their own long table: the app needs
    them to draw a pitch-control surface, and re-reading Metrica's 145k-frame
    CSVs on a page load would make the app unusable.
    """
    out_dir = resolve_path(cfg["paths"]["data_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for row in compared.itertuples():
        frame = row.frame_positions.copy()
        frame["game_id"] = row.game_id
        frame["frame"] = row.frame
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(
        out_dir / "metrica_shot_frames.parquet", index=False
    )

    compared.drop(columns=["frame_positions"]).to_parquet(
        out_dir / "metrica_shots_compared.parquet", index=False
    )
    disagreements.to_csv(out_dir / "metrica_biggest_disagreements.csv", index=False)
    intervals.to_csv(out_dir / "metrica_coefficient_intervals.csv", index=False)

    with open(out_dir / "metrica_comparison_summary.json", "w") as f:
        json.dump(
            {"basic_fit_diagnostics": basic_diagnostics, "comparison": summary},
            f,
            indent=2,
            default=str,
        )
    logger.info("Saved Metrica comparison outputs to %s", out_dir)


def main() -> None:
    cfg = load_config()
    report_metrica_shot_counts(cfg)
    validation = run_statsbomb_validation(cfg)
    run_statsbomb_context_model(cfg, validation["shots"])
    run_metrica_comparison(cfg)
    logger.info("Pipeline complete. See data/processed/ for output.")
    logger.info(
        "Render the figures with `python -m scripts.render_figures`, or "
        "explore interactively with `streamlit run app/streamlit_app.py`."
    )


if __name__ == "__main__":
    main()
