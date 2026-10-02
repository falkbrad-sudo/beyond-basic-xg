"""Compare basic vs. enhanced model predictions, shot by shot.

The main output of this comparison is NOT an aggregate metric (AUC,
log-loss) given the tiny sample. It's a small number of specific, nameable
shots where the two models disagree most, discussed individually. See
METHODOLOGY.md principle #2: don't lead with or overstate aggregate performance
claims this sample can't support.

`summary_statistics` therefore checks the goal count before it computes
anything that assumes a reasonable number of positive cases, and returns a
refusal in place of a number when there aren't enough.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

# Below this many goals, a difference in AUC or log-loss between two models
# is noise. 20 is already generous for a two-model comparison; the point of
# the constant is that the threshold is explicit and checkable rather than
# left to whoever reads the output.
MIN_GOALS_FOR_PERFORMANCE_METRICS = 20

# Context columns worth carrying into a disagreement table, so each shot can
# actually be looked up and understood rather than just ranked.
_CONTEXT_COLUMNS = [
    "game_id",
    "team",
    "period",
    "frame",
    "time_s",
    "player",
    "subtype",
    "x",
    "y",
    "distance_to_goal",
    "angle_to_goal",
    "pitch_control",
    "pitch_control_spearman",
    "defenders_in_cone",
    "goalkeeper_distance",
    "is_goal",
    "basic_xg_pred",
    "enhanced_xg_pred",
    "prediction_diff",
]


def compare_predictions(
    shots: pd.DataFrame,
    basic_model: LogisticRegression | Pipeline,
    enhanced_model: LogisticRegression | Pipeline,
    basic_features: list[str],
    enhanced_features: list[str],
) -> pd.DataFrame:
    """Add both models' predicted probabilities to the shots DataFrame.

    Parameters
    ----------
    shots : pd.DataFrame
        Must contain every column named in basic_features and
        enhanced_features.
    basic_model, enhanced_model : LogisticRegression or Pipeline
        Both must expose predict_proba. Fit them the same way (both scaled,
        or both not) or the comparison measures the fitting choice rather
        than the features (see basic_xg_model.fit_basic_model's `scale`).
    basic_features, enhanced_features : list[str]

    Returns
    -------
    pd.DataFrame
        Copy of shots with 'basic_xg_pred' and 'enhanced_xg_pred' columns
        added, plus 'prediction_diff' (enhanced - basic) to make sorting for
        the biggest disagreements trivial.
    """
    needed = sorted(set(basic_features) | set(enhanced_features))
    missing = [c for c in needed if c not in shots.columns]
    if missing:
        raise ValueError(f"shots is missing required columns: {missing}")

    out = shots.copy()
    out["basic_xg_pred"] = basic_model.predict_proba(
        out[basic_features].to_numpy(dtype=float)
    )[:, 1]
    out["enhanced_xg_pred"] = enhanced_model.predict_proba(
        out[enhanced_features].to_numpy(dtype=float)
    )[:, 1]
    out["prediction_diff"] = out["enhanced_xg_pred"] - out["basic_xg_pred"]
    return out


def find_biggest_disagreements(
    compared_shots: pd.DataFrame, n: int = 5
) -> pd.DataFrame:
    """Return the N shots where the two models disagree most.

    This is the most useful output of the whole project: a short, concrete
    list of specific shots (with enough context columns to look each one up
    and understand why) to discuss individually, rather than a single
    aggregate performance number from a ~67-shot sample.

    Parameters
    ----------
    compared_shots : pd.DataFrame
        Output of compare_predictions().
    n : int

    Returns
    -------
    pd.DataFrame
        Top N rows by absolute prediction_diff, sorted descending, carrying
        whichever context columns are present.
    """
    if "prediction_diff" not in compared_shots.columns:
        raise ValueError(
            "compared_shots has no 'prediction_diff' column. Run "
            "compare_predictions() first."
        )
    columns = [c for c in _CONTEXT_COLUMNS if c in compared_shots.columns]
    order = compared_shots["prediction_diff"].abs().sort_values(ascending=False).index
    return compared_shots.reindex(order).head(n)[columns].reset_index(drop=True)


def summary_statistics(compared_shots: pd.DataFrame) -> dict:
    """Basic aggregate comparison stats. Report these, but don't lead with
    them or treat them as more conclusive than a small sample supports.

    Parameters
    ----------
    compared_shots : pd.DataFrame
        Output of compare_predictions(); needs 'is_goal' for the
        outcome-dependent entries.

    Returns
    -------
    dict
        Correlation between the two models' predictions, mean absolute
        difference, the goal count, and (only if the sample carries at least
        MIN_GOALS_FOR_PERFORMANCE_METRICS goals) log-loss and AUC for each
        model against actual outcomes.

        When there are too few goals, the metrics are deliberately NOT
        computed, and 'performance_metrics_withheld' explains why in words
        that can be quoted directly. Silently computing a log-loss on 9 goals
        and letting a reader treat it as evidence is the specific failure
        this function is built to avoid.
    """
    required = ["basic_xg_pred", "enhanced_xg_pred", "prediction_diff"]
    missing = [c for c in required if c not in compared_shots.columns]
    if missing:
        raise ValueError(f"compared_shots is missing columns: {missing}")

    basic = compared_shots["basic_xg_pred"].to_numpy(dtype=float)
    enhanced = compared_shots["enhanced_xg_pred"].to_numpy(dtype=float)

    results: dict = {
        "n_shots": int(len(compared_shots)),
        "prediction_correlation": float(stats.pearsonr(basic, enhanced).statistic),
        "spearman_correlation": float(stats.spearmanr(basic, enhanced).statistic),
        "mean_abs_difference": float(np.abs(enhanced - basic).mean()),
        "max_abs_difference": float(np.abs(enhanced - basic).max()),
        "mean_basic_xg": float(basic.mean()),
        "mean_enhanced_xg": float(enhanced.mean()),
    }

    if "is_goal" not in compared_shots.columns:
        return results

    actual = compared_shots["is_goal"].to_numpy(dtype=int)
    n_goals = int(actual.sum())
    results["n_goals"] = n_goals
    results["observed_goal_rate"] = float(actual.mean())

    if n_goals < MIN_GOALS_FOR_PERFORMANCE_METRICS or len(np.unique(actual)) < 2:
        results["performance_metrics_withheld"] = (
            f"Log-loss and AUC were not computed: the sample contains "
            f"{n_goals} goals, below the {MIN_GOALS_FOR_PERFORMANCE_METRICS} "
            f"this project requires before reporting a predictive-performance "
            f"comparison. With this few positive cases the difference between "
            f"two models' scores is dominated by which individual shots "
            f"happened to go in, so any such number would invite a conclusion "
            f"the data cannot support. The shot-by-shot disagreements from "
            f"find_biggest_disagreements() are the usable output here."
        )
        return results

    results["basic_log_loss"] = float(log_loss(actual, basic))
    results["enhanced_log_loss"] = float(log_loss(actual, enhanced))
    results["basic_auc"] = float(roc_auc_score(actual, basic))
    results["enhanced_auc"] = float(roc_auc_score(actual, enhanced))
    return results
