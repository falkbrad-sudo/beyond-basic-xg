"""Render the project's eight figures to reports/figures/.

Run with: python -m scripts.render_figures

Requires `python -m src.pipeline` to have run first, since every figure is
drawn from the outputs it writes to data/processed/. The figures cover the
StatsBomb baseline validation, the Metrica shot maps and pitch-control view,
and the bootstrap intervals and power curve.
"""
from __future__ import annotations

import logging
import os

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp") + "/mpl")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from src.config import load_config, resolve_path  # noqa: E402
from src.models.basic_xg_model import BASIC_FEATURES, fit_basic_model  # noqa: E402
from src.pipeline import load_metrica_shots  # noqa: E402
from src.viz.inference_plots import (  # noqa: E402
    plot_coefficient_intervals,
    plot_power_curve,
)
from src.viz.pitch_control_plots import plot_pitch_control_at_shot  # noqa: E402
from src.viz.shot_maps import plot_comparison_shot_maps, plot_shot_map  # noqa: E402
from src.viz.validation_plots import plot_calibration, plot_xg_agreement  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = "reports/figures"


def _describe_subtype(subtype: str) -> str:
    """Turn Metrica's 'HEAD-ON TARGET-GOAL' into 'header, on target, goal'."""
    parts = [part.strip().lower() for part in str(subtype).split("-")]
    return ", ".join("header" if part == "head" else part for part in parts)


def _count(n: int, noun: str) -> str:
    """'no defenders', '1 defender', '3 defenders'."""
    if n == 0:
        return f"no {noun}s"
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def main() -> None:
    cfg = load_config()
    processed = resolve_path(cfg["paths"]["data_processed"])
    out_dir = resolve_path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    shots_path = processed / "statsbomb_shots_with_geometry.parquet"
    if not shots_path.exists():
        raise FileNotFoundError(
            f"{shots_path} not found. Run `python -m src.pipeline` first."
        )

    # --- 1 & 2: the StatsBomb validation panels ---------------------------
    sb_shots = pd.read_parquet(shots_path)
    model, diagnostics = fit_basic_model(sb_shots)
    sb_shots["own_xg"] = model.predict_proba(
        sb_shots[BASIC_FEATURES].to_numpy(float)
    )[:, 1]

    n_shots = len(sb_shots)
    n_goals = int(sb_shots["is_goal"].sum())
    above_half_sb = int((sb_shots["statsbomb_xg"] > 0.5).sum())
    above_half_own = int((sb_shots["own_xg"] > 0.5).sum())

    fig = plot_xg_agreement(
        sb_shots,
        title="A geometry-only baseline tracks StatsBomb's xG but compresses the top end",
        subtitle=(
            f"{n_shots:,} open-play shots, 2023 NWSL. Spearman rho 0.78. "
            f"StatsBomb rates {above_half_sb} shots above 0.5 xG; this model rates "
            f"{above_half_own}."
        ),
    )
    fig.savefig(out_dir / "01_xg_agreement.png", facecolor=fig.get_facecolor())
    logger.info("wrote %s", out_dir / "01_xg_agreement.png")

    calibration = pd.read_csv(processed / "basic_model_calibration.csv")
    fig = plot_calibration(
        calibration,
        title="The baseline is well calibrated across its whole range",
        subtitle=(
            f"{n_shots:,} open-play shots, {n_goals} goals. Deciles of predicted xG; "
            "mean prediction 0.085 vs 0.086 observed."
        ),
    )
    fig.savefig(out_dir / "02_calibration.png", facecolor=fig.get_facecolor())
    logger.info("wrote %s", out_dir / "02_calibration.png")

    # --- 3 & 4: the Metrica sample, scored by the validated baseline ------
    metrica = load_metrica_shots(cfg)
    open_play = metrica[~metrica["is_penalty"]].copy()
    open_play["basic_xg_statsbomb_fit"] = model.predict_proba(
        open_play[BASIC_FEATURES].to_numpy(float)
    )[:, 1]

    fig = plot_shot_map(
        open_play,
        color_by="basic_xg_statsbomb_fit",
        title="Every shot in the Metrica sample, scored by the validated baseline",
        subtitle=(
            f"{len(open_play)} open-play shots, {int(open_play['is_goal'].sum())} goals, "
            "3 matches. Pitch control is not in this model yet."
        ),
    )
    fig.savefig(out_dir / "03_metrica_shot_map.png", facecolor=fig.get_facecolor())
    logger.info("wrote %s", out_dir / "03_metrica_shot_map.png")

    # Exercises the side-by-side function on two fits of the SAME model. This
    # is a sample-size diagnostic, not the pitch-control comparison: it shows
    # how much the geometry model moves when refit on 67 shots instead of
    # 3,490. The real use of this figure arrives with the enhanced model.
    small_fit, small_diagnostics = fit_basic_model(open_play, holdout=False)
    open_play["basic_xg_metrica_fit"] = small_fit.predict_proba(
        open_play[BASIC_FEATURES].to_numpy(float)
    )[:, 1]

    fig = plot_comparison_shot_maps(
        open_play,
        left_col="basic_xg_statsbomb_fit",
        right_col="basic_xg_metrica_fit",
        left_label=f"Fit on 3,490 NWSL shots (AUC {diagnostics['test_auc']:.2f})",
        right_label=f"Same model refit on {small_diagnostics['n_shots']} Metrica shots",
        title="Sample-size diagnostic: the same geometry model, fit on two samples",
    )
    fig.savefig(out_dir / "04_sample_size_diagnostic.png", facecolor=fig.get_facecolor())
    logger.info("wrote %s", out_dir / "04_sample_size_diagnostic.png")

    # --- 5 & 6: the pitch-control comparison -----------------------------
    compared_path = processed / "metrica_shots_compared.parquet"
    frames_path = processed / "metrica_shot_frames.parquet"
    if not compared_path.exists():
        logger.warning(
            "Metrica comparison output missing; skipping the pitch-control "
            "figures. Run `python -m src.pipeline`."
        )
        return

    compared = pd.read_parquet(compared_path)
    frames = pd.read_parquet(frames_path)

    fig = plot_comparison_shot_maps(
        compared,
        left_col="basic_xg_pred",
        right_col="enhanced_xg_pred",
        left_label="Basic (distance + angle)",
        right_label="Enhanced (+ pitch control)",
        title=(
            "The pitch-control term barely moves anything: the two models "
            "agree at r = 0.999"
        ),
    )
    fig.savefig(out_dir / "05_basic_vs_enhanced.png", facecolor=fig.get_facecolor())
    logger.info("wrote %s", out_dir / "05_basic_vs_enhanced.png")

    # The single shot the two models disagree on most -- the figure that does
    # the explaining the numbers cannot.
    top = compared.reindex(
        compared["prediction_diff"].abs().sort_values(ascending=False).index
    ).iloc[0]
    frame = frames[
        (frames["game_id"] == top["game_id"]) & (frames["frame"] == top["frame"])
    ]
    fig = plot_pitch_control_at_shot(
        frame,
        shot_x=float(top["x"]),
        shot_y=float(top["y"]),
        shooting_team=top["team"],
        title="The shot the two models disagree on most",
        subtitle=(
            f"Game {int(top['game_id'])}, {top['team']}, minute "
            f"{int(top['time_s']) // 60} ({_describe_subtype(top['subtype'])}). "
            f"Taken {top['distance_to_goal']:.1f}m out with "
            f"{top['pitch_control']:.0%} of the lane controlled, "
            f"{_count(int(top['defenders_in_cone']), 'defender')} in the lane and "
            f"the keeper {top['goalkeeper_distance']:.1f}m from the centre of goal. "
            f"Basic xG {top['basic_xg_pred']:.3f}, enhanced "
            f"{top['enhanced_xg_pred']:.3f}."
        ),
    )
    fig.savefig(out_dir / "06_pitch_control_at_shot.png", facecolor=fig.get_facecolor())
    logger.info("wrote %s", out_dir / "06_pitch_control_at_shot.png")

    # --- 7 & 8: uncertainty and power ------------------------------------
    import json

    metrica_ci_path = processed / "metrica_coefficient_intervals.csv"
    context_ci_path = processed / "statsbomb_context_intervals.csv"
    power_path = processed / "statsbomb_power_analysis.csv"
    if not (metrica_ci_path.exists() and context_ci_path.exists()):
        logger.warning("Interval outputs missing; skipping figures 7-8.")
        return

    fig = plot_coefficient_intervals(
        {
            "Metrica tracking sample (66 shots, 9 goals)":
                pd.read_csv(metrica_ci_path),
            "StatsBomb freeze frames (3,490 shots, 299 goals)":
                pd.read_csv(context_ci_path),
        },
        title=(
            "The same questions, asked of 9 goals and of 299: only one sample "
            "can answer them"
        ),
        subtitle=(
            "95% bootstrap intervals, stratified by outcome. An interval "
            "crossing the zero line means the data cannot sign the effect. "
            "Features differ between the samples because only tracking data "
            "supports pitch control."
        ),
    )
    fig.savefig(out_dir / "07_coefficient_intervals.png", facecolor=fig.get_facecolor())
    logger.info("wrote %s", out_dir / "07_coefficient_intervals.png")

    with open(processed / "statsbomb_context_summary.json") as f:
        context_summary = json.load(f)
    reference = context_summary["power_reference"]
    power_table = pd.read_csv(power_path)
    observed = float(
        pd.read_csv(context_ci_path)
        .set_index("feature")
        .loc[reference["feature"], "coefficient"]
    )

    fig = plot_power_curve(
        se_reference=float(reference["se_reference"]),
        n_reference=int(reference["n_reference"]),
        observed_effect=abs(observed),
        goal_rate=float(reference["goal_rate"]),
        sample_markers={
            "Metrica": (66, 9),
            "NWSL 2023": (int(reference["n_reference"]), 299),
        },
        title="How much data the question actually needs",
        subtitle=(
            f"Power to detect an effect on {reference['feature'].replace('_', ' ')}, "
            f"by sample size. The Metrica sample had roughly "
            f"{100 * power_table.iloc[3]['power_with_66_shots']:.0f}% power for a "
            f"moderate effect, so the null result there reflects sample size."
        ),
    )
    fig.savefig(out_dir / "08_power_curve.png", facecolor=fig.get_facecolor())
    logger.info("wrote %s", out_dir / "08_power_curve.png")

    logger.info("Done. 8 figures in %s.", out_dir)


if __name__ == "__main__":
    main()
