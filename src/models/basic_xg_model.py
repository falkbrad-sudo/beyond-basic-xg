"""The basic xG model: logistic regression on distance and angle alone.

This should be validated on the LARGER StatsBomb sample before ever being
trusted on the small Metrica sample; see METHODOLOGY.md principle #2 and the
README. Don't skip that validation step even though it's tempting
to jump straight to the "interesting" pitch-control comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config

BASIC_FEATURES = ["distance_to_goal", "angle_to_goal"]


def fit_basic_model(
    shots: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "is_goal",
    holdout: bool = True,
    scale: bool = False,
) -> tuple[LogisticRegression | Pipeline, dict]:
    """Fit a logistic regression xG model on geometry features only.

    Parameters
    ----------
    shots : pd.DataFrame
        Must contain feature_cols (from geometry.add_geometry_features) and
        target_col (boolean/int: did this shot result in a goal).
    feature_cols : list[str], optional
        Defaults to BASIC_FEATURES (distance and angle to goal).
    target_col : str
    holdout : bool
        When True (default) a train/test split is used and test metrics are
        reported. Pass False to fit on everything, which is the right choice
        on the Metrica sample (see Notes).
    scale : bool
        Standardize features before fitting. Off by default, which keeps the
        large-sample StatsBomb result reproducible. Turn it ON when this
        model is being compared against the enhanced model, so that both are
        regularized on the same footing. L2 penalizes a coefficient on a
        0-1 feature far more than one on a 3-32m feature, and an unscaled
        comparison would be a comparison of penalties, not of models.

    Returns
    -------
    tuple[LogisticRegression, dict]
        The fitted model, and a dict of basic fit diagnostics (train/test
        split sizes, coefficients, intercept, and, only when holdout is
        True and the test set contains both classes, test AUC and log-loss).
        Not full evaluation, which lives in evaluate.py.

    Notes
    -----
    On the small Metrica sample, a train/test split may leave almost no test
    shots at all, so call this with holdout=False there and fit on the full
    sample; the StatsBomb sample is large enough for the split. Whichever is
    used is recorded under 'evaluation' in the returned diagnostics, so the
    write-up can state it accurately rather than from memory.

    Test metrics are omitted rather than computed when the held-out set has
    only one class. AUC is undefined there, and returning a number anyway
    would be worse than returning nothing.
    """
    cfg = load_config()
    feature_cols = list(feature_cols or BASIC_FEATURES)

    missing = [c for c in [*feature_cols, target_col] if c not in shots.columns]
    if missing:
        raise ValueError(f"shots is missing required columns: {missing}")

    usable = shots.dropna(subset=[*feature_cols, target_col])
    X = usable[feature_cols].to_numpy(dtype=float)
    y = usable[target_col].to_numpy(dtype=int)

    diagnostics: dict = {
        "n_shots": int(len(usable)),
        "n_goals": int(y.sum()),
        "n_dropped_missing": int(len(shots) - len(usable)),
        "feature_cols": feature_cols,
    }

    if holdout:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg["model"]["test_size"],
            random_state=cfg["model"]["random_state"],
            stratify=y if y.sum() >= 2 and (len(y) - y.sum()) >= 2 else None,
        )
        diagnostics["evaluation"] = "train/test split"
    else:
        X_train, y_train = X, y
        X_test, y_test = np.empty((0, X.shape[1])), np.empty(0, dtype=int)
        diagnostics["evaluation"] = "fit on full sample (no holdout)"

    model = (
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        if scale
        else LogisticRegression(max_iter=1000)
    )
    model.fit(X_train, y_train)

    estimator = model[-1] if isinstance(model, Pipeline) else model
    diagnostics["n_train"] = int(len(y_train))
    diagnostics["n_test"] = int(len(y_test))
    diagnostics["scaled"] = bool(scale)
    diagnostics["coefficients"] = dict(
        zip(feature_cols, estimator.coef_[0].round(6).tolist())
    )
    diagnostics["intercept"] = float(estimator.intercept_[0])

    if len(y_test) and len(np.unique(y_test)) == 2:
        probabilities = model.predict_proba(X_test)[:, 1]
        diagnostics["test_auc"] = float(roc_auc_score(y_test, probabilities))
        diagnostics["test_log_loss"] = float(log_loss(y_test, probabilities))
    elif len(y_test):
        diagnostics["test_metrics_note"] = (
            "Held-out set contains a single outcome class; AUC and log-loss "
            "are undefined and were not computed."
        )

    return model, diagnostics


def _calibration_table(
    predicted: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Mean predicted probability vs. observed goal rate, by predicted decile.

    A correlation says the two models rank shots alike; this says whether the
    predicted probabilities are on the right scale. Both are needed before
    calling a baseline sound.
    """
    frame = pd.DataFrame({"predicted": predicted, "actual": actual})
    # Ranked bins rather than fixed-width: xG is heavily skewed toward zero,
    # so fixed-width bins leave the top deciles nearly empty.
    frame["bin"] = pd.qcut(
        frame["predicted"].rank(method="first"), n_bins, labels=False
    )
    table = (
        frame.groupby("bin")
        .agg(
            n_shots=("actual", "size"),
            mean_predicted=("predicted", "mean"),
            observed_goal_rate=("actual", "mean"),
        )
        .reset_index(drop=True)
    )
    return table.round(4)


def validate_against_statsbomb_xg(
    shots_with_statsbomb_xg: pd.DataFrame,
    fitted_model: LogisticRegression | Pipeline,
    feature_cols: list[str],
    n_disagreements: int = 10,
) -> dict:
    """Compare the basic model's predictions against StatsBomb's own xG.

    This is the validation step described in the README. Run this
    on the large StatsBomb sample BEFORE trusting the basic model enough to
    extend it (with pitch control) on the small Metrica sample.

    Parameters
    ----------
    shots_with_statsbomb_xg : pd.DataFrame
        Must contain feature_cols and a 'statsbomb_xg' column.
    fitted_model : LogisticRegression or Pipeline
    feature_cols : list[str]
    n_disagreements : int
        How many of the largest disagreements to return for inspection.

    Returns
    -------
    dict
        Pearson and Spearman correlation between this model's predictions and
        StatsBomb's xG, mean absolute difference, a calibration table against
        actual outcomes, and the largest individual disagreements as rows.

        The correlation speaks to whether the geometry-only baseline is
        SOUND, not whether it is equivalent to StatsBomb's model: theirs uses
        freeze-frame defender positions and goalkeeper location that this
        project has no access to. A high correlation means distance and angle
        carry most of the signal, which is the premise the Metrica step
        builds on. It is not a claim of parity. Read the largest
        disagreements individually; they are where the missing information
        shows up.

    Raises
    ------
    ValueError
        If 'statsbomb_xg' or any feature column is absent.
    """
    required = [*feature_cols, "statsbomb_xg"]
    missing = [c for c in required if c not in shots_with_statsbomb_xg.columns]
    if missing:
        raise ValueError(f"shots frame is missing required columns: {missing}")

    usable = shots_with_statsbomb_xg.dropna(subset=required).copy()
    if usable.empty:
        raise ValueError("No shots with both features and statsbomb_xg available.")

    X = usable[feature_cols].to_numpy(dtype=float)
    usable["own_xg"] = fitted_model.predict_proba(X)[:, 1]
    usable["xg_diff"] = usable["own_xg"] - usable["statsbomb_xg"]

    pearson = stats.pearsonr(usable["own_xg"], usable["statsbomb_xg"])
    spearman = stats.spearmanr(usable["own_xg"], usable["statsbomb_xg"])

    results: dict = {
        "n_shots": int(len(usable)),
        "pearson_r": float(pearson.statistic),
        "spearman_rho": float(spearman.statistic),
        "mean_abs_difference": float(usable["xg_diff"].abs().mean()),
        "mean_own_xg": float(usable["own_xg"].mean()),
        "mean_statsbomb_xg": float(usable["statsbomb_xg"].mean()),
    }

    if "is_goal" in usable.columns:
        actual = usable["is_goal"].to_numpy(dtype=int)
        results["observed_goal_rate"] = float(actual.mean())
        results["calibration"] = _calibration_table(
            usable["own_xg"].to_numpy(), actual
        )
        if len(np.unique(actual)) == 2:
            results["own_auc"] = float(roc_auc_score(actual, usable["own_xg"]))
            results["statsbomb_auc"] = float(
                roc_auc_score(actual, usable["statsbomb_xg"])
            )

    context_cols = [
        c
        for c in [
            "match_id",
            "team",
            "player",
            "minute",
            "x",
            "y",
            "body_part",
            "shot_type",
            "outcome",
            *feature_cols,
            "statsbomb_xg",
            "own_xg",
            "xg_diff",
        ]
        if c in usable.columns
    ]
    results["largest_disagreements"] = (
        usable.reindex(usable["xg_diff"].abs().sort_values(ascending=False).index)
        .head(n_disagreements)[context_cols]
        .reset_index(drop=True)
    )
    return results
