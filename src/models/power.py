"""How much data would it actually take to test the pitch-control question?

The project's central limitation is that the Metrica sample holds 9 goals.
Saying "that is too few" is easy; saying *how many would be enough* is the
useful version, and it is the natural next question. This module answers
it.

Method
------
For a logistic regression the covariance of the fitted coefficients is the
inverse Fisher information,

    Cov(beta) = (X' W X)^-1,    W = diag(p_i (1 - p_i)),

which needs no extra dependency beyond numpy and is exact for a fitted
model. Standard errors scale as 1/sqrt(n), so a standard error measured on
a reference sample of size n_ref implies

    se(n) = se_ref * sqrt(n_ref / n).

Power for a two-sided Wald test at level alpha then follows in closed form
from an assumed true effect size.

This is appropriate HERE because it is calibrated on the 3,490-shot
StatsBomb sample, where asymptotic normality is reasonable. It would not be
appropriate on the 66-shot Metrica sample; that is what the bootstrap in
`src/models/uncertainty.py` is for. The two modules deliberately use
different machinery for different regimes.

Results are reported in GOALS as well as shots, because for a binary
outcome it is the positive cases that bind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline

DEFAULT_ALPHA = 0.05
DEFAULT_TARGET_POWER = 0.80


def _design_and_probabilities(
    model, shots: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Model matrix (with intercept) and fitted probabilities.

    The matrix is taken AFTER any scaling in the pipeline, so the resulting
    standard errors are on the same scale as the coefficients they describe.
    """
    X = shots[feature_cols].to_numpy(dtype=float)
    if isinstance(model, Pipeline):
        for name, step in model.steps[:-1]:
            X = step.transform(X)
    probabilities = model.predict_proba(shots[feature_cols].to_numpy(dtype=float))[:, 1]
    return np.column_stack([np.ones(len(X)), X]), probabilities


def wald_standard_errors(
    model, shots: pd.DataFrame, feature_cols: list[str]
) -> pd.Series:
    """Standard errors of the fitted coefficients, from the information matrix.

    Parameters
    ----------
    model : LogisticRegression or Pipeline
    shots : pd.DataFrame
    feature_cols : list[str]

    Returns
    -------
    pd.Series
        Standard error per feature (the intercept is dropped).

    Notes
    -----
    Assumes an unpenalized fit. A regularized model's coefficients are
    biased toward zero and their true sampling distribution is narrower
    than this; using these SEs for a heavily penalized fit would overstate
    the uncertainty. Calibrate on a weakly-penalized large-sample fit.
    """
    design, probabilities = _design_and_probabilities(model, shots, feature_cols)
    weights = probabilities * (1.0 - probabilities)
    information = design.T @ (design * weights[:, None])
    covariance = np.linalg.pinv(information)
    return pd.Series(np.sqrt(np.diag(covariance))[1:], index=feature_cols)


def power_at_sample_size(
    effect_size: float,
    se_reference: float,
    n_reference: int,
    n: np.ndarray | int,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:
    """Two-sided Wald power for an effect of `effect_size` at sample size n.

    Parameters
    ----------
    effect_size : float
        Assumed true coefficient, on the same (standardized) scale as the
        reference fit.
    se_reference : float
        Standard error measured at n_reference.
    n_reference : int
    n : int or array-like
        Sample size(s) to evaluate.
    alpha : float

    Returns
    -------
    np.ndarray
        Power in [0, 1].
    """
    n = np.asarray(n, dtype=float)
    se = se_reference * np.sqrt(n_reference / n)
    critical = stats.norm.ppf(1 - alpha / 2)
    ratio = np.abs(effect_size) / se
    return stats.norm.sf(critical - ratio) + stats.norm.cdf(-critical - ratio)


def required_sample_size(
    effect_size: float,
    se_reference: float,
    n_reference: int,
    target_power: float = DEFAULT_TARGET_POWER,
    alpha: float = DEFAULT_ALPHA,
) -> int:
    """Smallest n reaching `target_power` for the given effect. Closed form.

    Returns
    -------
    int
        Required number of shots, rounded up.
    """
    critical = stats.norm.ppf(1 - alpha / 2)
    needed = stats.norm.ppf(target_power) + critical
    n = n_reference * (needed * se_reference / abs(effect_size)) ** 2
    return int(np.ceil(n))


def power_analysis(
    model,
    shots: pd.DataFrame,
    feature_cols: list[str],
    feature: str,
    effect_sizes: list[float] | None = None,
    goal_rate: float | None = None,
    target_power: float = DEFAULT_TARGET_POWER,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """How many shots (and goals) are needed to detect a given effect.

    Parameters
    ----------
    model : fitted model used only to calibrate the standard error
    shots : pd.DataFrame
        The reference sample the standard error is measured on.
    feature_cols : list[str]
    feature : str
        Which coefficient the analysis is about.
    effect_sizes : list[float], optional
        Candidate true coefficients, per standard deviation. Defaults to a
        range spanning "barely there" to "large" for an xG feature.
    goal_rate : float, optional
        Used to convert a shot count into a goal count. Defaults to the
        observed rate in `shots`.
    target_power : float
    alpha : float

    Returns
    -------
    pd.DataFrame
        One row per effect size: the required shots and goals, plus the
        power that the Metrica sample (66 shots) would have had.
    """
    if feature not in feature_cols:
        raise ValueError(f"{feature!r} is not in feature_cols")

    effect_sizes = effect_sizes or [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    standard_errors = wald_standard_errors(model, shots, feature_cols)
    se_reference = float(standard_errors[feature])
    n_reference = int(len(shots))
    goal_rate = (
        float(shots["is_goal"].mean()) if goal_rate is None else float(goal_rate)
    )

    rows = []
    for effect in effect_sizes:
        needed = required_sample_size(
            effect, se_reference, n_reference, target_power, alpha
        )
        rows.append(
            {
                "effect_size_per_sd": effect,
                "required_shots": needed,
                "required_goals": int(np.ceil(needed * goal_rate)),
                "power_with_66_shots": float(
                    power_at_sample_size(effect, se_reference, n_reference, 66, alpha)
                ),
                "power_with_reference_sample": float(
                    power_at_sample_size(
                        effect, se_reference, n_reference, n_reference, alpha
                    )
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.attrs["se_reference"] = se_reference
    result.attrs["n_reference"] = n_reference
    result.attrs["goal_rate"] = goal_rate
    result.attrs["feature"] = feature
    return result
