"""The enhanced xG model: geometry + pitch control + shot context.

Fit ONLY on the Metrica sample (the only data with pitch control/context
features available). Expect a very small n; see METHODOLOGY.md principle #2
before writing any evaluation language that implies statistical confidence
this sample size can't support.

On this sample that means: 67 open-play shots containing 9 goals. For a
binary outcome it is the 9 positive cases that set the ceiling, not the 67
rows, so the primary model here takes THREE features and the five-feature
version is exploratory only. Even three is roughly three goals per
parameter; nothing fit here can separate two models on predictive
performance, and the comparison in evaluate.py is framed accordingly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

# The primary comparison against the basic model: geometry plus the one
# pitch-control term the project is actually about.
ENHANCED_FEATURES = [
    "distance_to_goal",
    "angle_to_goal",
    "pitch_control",
]

# Exploratory variant. Reported as a secondary result, never as the main result:
# five parameters on nine goals is a demonstration that the plumbing works,
# not an estimate anyone should trust.
ENHANCED_FEATURES_FULL = [
    "distance_to_goal",
    "angle_to_goal",
    "pitch_control",
    "defenders_in_cone",
    "goalkeeper_distance",
]

# Stronger regularization than sklearn's default C=1.0. With 9 positive
# cases, a logistic regression will happily drive coefficients large to chase
# individual shots; C=0.3 keeps them near zero unless the data really insists.
# Paired with StandardScaler so the penalty applies evenly across features on
# very different scales (meters vs. a 0-1 control share).
DEFAULT_C = 0.3


def fit_enhanced_model(
    shots: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "is_goal",
    C: float = DEFAULT_C,
) -> Pipeline:
    """Fit a logistic regression xG model on geometry + space/context features.

    Parameters
    ----------
    shots : pd.DataFrame
        Must contain all feature_cols, i.e. the output of running shots
        through geometry.add_geometry_features,
        pitch_control_at_shot.add_pitch_control_features, and
        shot_context.add_context_features, joined together.
    feature_cols : list[str], optional
        Defaults to ENHANCED_FEATURES (geometry + pitch control). Pass
        ENHANCED_FEATURES_FULL for the exploratory five-feature variant.
    target_col : str
    C : float
        Inverse regularization strength; see DEFAULT_C for why the default is
        well below sklearn's.

    Returns
    -------
    Pipeline
        StandardScaler + LogisticRegression. A pipeline rather than a bare
        estimator so the scaling travels with the model and predictions
        cannot be made on unscaled inputs by accident; it exposes the same
        `predict_proba` interface the basic model does, which is what
        evaluate.compare_predictions needs.

    Raises
    ------
    ValueError
        If any feature or the target is missing, or if the sample contains
        fewer than two outcome classes.
    """
    feature_cols = list(feature_cols or ENHANCED_FEATURES)
    missing = [c for c in [*feature_cols, target_col] if c not in shots.columns]
    if missing:
        raise ValueError(f"shots is missing required columns: {missing}")

    usable = shots.dropna(subset=[*feature_cols, target_col])
    y = usable[target_col].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError(
            "shots contains a single outcome class; a logistic model cannot "
            "be fit on it."
        )

    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, C=C)
    )
    # No held-out split: with 9 goals a test set either contains almost no
    # goals or strips them from training. The sound choice is to fit on
    # everything and refuse to report a predictive-performance number. See
    # evaluate.summary_statistics, which checks the goal count before
    # computing anything that assumes one.
    model.fit(usable[feature_cols].to_numpy(dtype=float), y)
    return model


def enhanced_model_diagnostics(
    model: Pipeline, feature_cols: list[str] | None = None
) -> dict:
    """Standardized coefficients of a fitted enhanced model.

    Because the pipeline standardizes its inputs, the coefficients are
    directly comparable to each other: each is the log-odds change per one
    standard deviation of that feature. That makes the sign and rough
    magnitude readable, which is the most that should be taken from a fit
    this small.

    Parameters
    ----------
    model : Pipeline
        From fit_enhanced_model.
    feature_cols : list[str], optional

    Returns
    -------
    dict
        'coefficients' (per standard deviation), 'intercept', and a
        'interpretation_note' spelling out the sample-size caveat so it
        travels with the numbers.
    """
    feature_cols = list(feature_cols or ENHANCED_FEATURES)
    estimator = model[-1]
    return {
        "coefficients_per_sd": dict(
            zip(feature_cols, estimator.coef_[0].round(4).tolist())
        ),
        "intercept": float(estimator.intercept_[0]),
        "interpretation_note": (
            "Coefficients are per standard deviation of a standardized "
            "feature. Fit on 9 goals, so read the signs as directional at "
            "best; the confidence intervals on these would comfortably "
            "include zero and they are not reported as effects."
        ),
    }
