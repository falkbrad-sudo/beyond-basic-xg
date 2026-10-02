"""Bootstrap confidence intervals for model coefficients and comparisons.

This project repeatedly says that its coefficients "have confidence
intervals that comfortably include zero". That is a claim, and claims here
get measured rather than asserted, and this module is where it gets measured.

Why the bootstrap rather than Wald standard errors: on the Metrica sample
the model is fit on 9 goals with regularization, where the asymptotic
normality a Wald interval assumes does not hold. Resampling makes no such
assumption. (`src/models/power.py` does use the Wald information matrix,
but only on the large sample where it is appropriate.)

Resampling is STRATIFIED by outcome: goals and non-goals are resampled
separately, preserving the observed count of each. With 9 goals, ordinary
resampling would produce samples containing no goals at all, which cannot be
fit; discarding those would quietly bias the interval toward
goal-heavy draws. Stratifying conditions on the thing that is scarce.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_SEED = 42
DEFAULT_ALPHA = 0.05


def _coefficients(model) -> np.ndarray:
    """Coefficient vector of a bare estimator or a fitted Pipeline."""
    estimator = model[-1] if isinstance(model, Pipeline) else model
    return estimator.coef_[0]


def stratified_bootstrap_indices(
    y: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Resample row indices with replacement, preserving the class counts.

    Parameters
    ----------
    y : np.ndarray
        Binary outcome.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        Index array the same length as y, with the same number of positives.
    """
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    return np.concatenate(
        [
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ]
    )


def bootstrap_coefficients(
    fit_fn: Callable[[pd.DataFrame], object],
    shots: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "is_goal",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Percentile bootstrap intervals for each coefficient.

    Parameters
    ----------
    fit_fn : callable
        Takes a shots DataFrame and returns a fitted model (or a
        (model, diagnostics) tuple, which is unpacked). Must fit exactly the
        model whose uncertainty is wanted, including any scaling.
    shots : pd.DataFrame
    feature_cols : list[str]
        Names, in the order the model's coefficients are in.
    target_col : str
    n_bootstrap : int
    alpha : float
        0.05 gives a 95% interval.
    seed : int

    Returns
    -------
    pd.DataFrame
        One row per feature: 'coefficient' (the point estimate from the full
        sample), 'ci_low', 'ci_high', 'crosses_zero', and
        'p_sign_disagrees', the share of bootstrap draws whose sign differs
        from the point estimate, which is a readable stand-in for a p-value
        and does not pretend to be one.

    Notes
    -----
    Draws that cannot be fit (a resample with a degenerate feature, say) are
    skipped and counted; the count is attached to the result as
    `.attrs['n_failed_draws']` so a silently thin bootstrap is detectable.
    """
    usable = shots.dropna(subset=[*feature_cols, target_col])
    y = usable[target_col].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)

    point = _coefficients(_unwrap(fit_fn(usable)))

    draws, failed = [], 0
    for _ in range(n_bootstrap):
        index = stratified_bootstrap_indices(y, rng)
        sample = usable.iloc[index]
        try:
            draws.append(_coefficients(_unwrap(fit_fn(sample))))
        except Exception:
            failed += 1
    if not draws:
        raise RuntimeError("every bootstrap draw failed to fit")

    matrix = np.vstack(draws)
    low = np.percentile(matrix, 100 * alpha / 2, axis=0)
    high = np.percentile(matrix, 100 * (1 - alpha / 2), axis=0)
    sign_disagrees = (np.sign(matrix) != np.sign(point)).mean(axis=0)

    result = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": point,
            "ci_low": low,
            "ci_high": high,
            "crosses_zero": (low <= 0) & (high >= 0),
            "p_sign_disagrees": sign_disagrees,
        }
    )
    result.attrs["n_bootstrap"] = len(draws)
    result.attrs["n_failed_draws"] = failed
    return result


def _unwrap(fitted):
    """Accept either a model or a (model, diagnostics) tuple."""
    return fitted[0] if isinstance(fitted, tuple) else fitted


def bootstrap_prediction_difference(
    shots: pd.DataFrame,
    basic_fit_fn: Callable[[pd.DataFrame], object],
    enhanced_fit_fn: Callable[[pd.DataFrame], object],
    basic_features: list[str],
    enhanced_features: list[str],
    target_col: str = "is_goal",
    n_bootstrap: int = 500,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Interval for how far apart the two models' predictions actually are.

    Refits BOTH models inside every bootstrap draw and re-scores the
    original shots, so the interval covers the uncertainty in fitting, not
    just in sampling. Reports the mean absolute difference between the two
    models' predictions.

    Returns
    -------
    dict
        'mean_abs_difference' point estimate, 'ci_low', 'ci_high',
        'n_bootstrap', 'n_failed_draws'.

    Notes
    -----
    This answers "could the two models differ meaningfully and we just
    cannot see it?" more directly than a coefficient interval does. An
    interval that sits near zero says the models are substantively the same
    on this data, which is a stronger and more accurate statement than a
    non-significant coefficient.
    """
    usable = shots.dropna(subset=[*set(basic_features) | set(enhanced_features),
                                  target_col])
    y = usable[target_col].to_numpy(dtype=int)
    X_basic = usable[basic_features].to_numpy(dtype=float)
    X_enhanced = usable[enhanced_features].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    def difference(fit_sample: pd.DataFrame) -> float:
        basic = _unwrap(basic_fit_fn(fit_sample))
        enhanced = _unwrap(enhanced_fit_fn(fit_sample))
        return float(
            np.abs(
                enhanced.predict_proba(X_enhanced)[:, 1]
                - basic.predict_proba(X_basic)[:, 1]
            ).mean()
        )

    point = difference(usable)

    draws, failed = [], 0
    for _ in range(n_bootstrap):
        sample = usable.iloc[stratified_bootstrap_indices(y, rng)]
        try:
            draws.append(difference(sample))
        except Exception:
            failed += 1
    if not draws:
        raise RuntimeError("every bootstrap draw failed to fit")

    return {
        "mean_abs_difference": point,
        "ci_low": float(np.percentile(draws, 100 * alpha / 2)),
        "ci_high": float(np.percentile(draws, 100 * (1 - alpha / 2))),
        "n_bootstrap": len(draws),
        "n_failed_draws": failed,
    }
