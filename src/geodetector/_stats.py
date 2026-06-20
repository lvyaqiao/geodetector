"""Core statistical functions for geographical detectors.

This module provides pure functions — no state, no side effects, no dependencies
on pandas or sklearn. Only numpy and scipy are required.
"""

import numpy as np
from scipy.stats import ncf as _ncf


def q_statistic(y, x, sst=None):
    """Compute the q-statistic (spatial stratified heterogeneity).

    q = 1 - SSW / SST

    where SSW = within-stratum sum of squares and SST = total sum of squares.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Response variable.
    x : ndarray of shape (n_samples,), dtype int
        Stratum labels (must be integer-coded).
    sst : float, optional
        Pre-computed total sum of squares for efficiency.

    Returns
    -------
    q : float
        The q-statistic in [0, 1]. Returns ``np.nan`` if SST == 0.

    Notes
    -----
    Strata containing only a single observation are excluded because
    within-stratum variance is undefined for N=1.

    The q-statistic is algebraically identical to the R² of a
    stratum-mean predictor: q ≡ R² ≡ η² (ANOVA effect size).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=int).ravel()

    # Remove strata with fewer than 2 observations
    unique, counts = np.unique(x, return_counts=True)
    valid = unique[counts >= 2]

    if len(valid) < 2:
        return 0.0

    mask = np.isin(x, valid)
    y = y[mask]
    x = x[mask]

    if sst is None:
        sst = np.sum((y - np.mean(y)) ** 2)

    if sst == 0:
        return np.nan

    ssw = 0.0
    for stratum in valid:
        y_h = y[x == stratum]
        n_h = len(y_h)
        if n_h > 1:
            ssw += (n_h - 1) * np.var(y_h, ddof=1)

    q = 1.0 - ssw / sst
    return float(max(0.0, min(1.0, q)))


def q_significance_test(y, x):
    """Non-central F-test for q-statistic significance.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Response variable.
    x : ndarray of shape (n_samples,), dtype int
        Stratum labels (integer-coded).

    Returns
    -------
    p_value : float
        P-value from the non-central F distribution.
        p < 0.05 indicates the stratification is statistically significant.

    References
    ----------
    Wang JF, Zhang TL, Fu BJ. 2016. A measure of spatial stratified
    heterogeneity. Ecological Indicators 67: 250-256.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=int).ravel()

    unique, counts = np.unique(x, return_counts=True)
    valid = unique[counts >= 2]

    if len(valid) < 2:
        return 1.0

    mask = np.isin(x, valid)
    y = y[mask]
    x = x[mask]

    L = len(valid)
    N = len(y)
    var_y = np.var(y, ddof=1)
    sst = var_y * (N - 1)

    if sst == 0:
        return 1.0

    q = q_statistic(y, x, sst=sst)
    if np.isnan(q):
        return 1.0
    if q <= 0:
        return 1.0
    if q >= 1:
        return 0.0

    # Non-centrality parameter lambda — matches R GD / gdverse formula:
    #   m1 = Σ μ_h²  (unweighted sum of squared stratum means)
    #   m2 = (Σ √N_h · μ_h)² / N
    #   λ = (m1 - m2) / (var(y) · (N - 1) / N)
    m1 = 0.0
    m2_sum = 0.0
    for stratum in valid:
        y_h = y[x == stratum]
        n_h = len(y_h)
        mean_h = np.mean(y_h)
        m1 += mean_h * mean_h
        m2_sum += np.sqrt(n_h) * mean_h
    lam = (m1 - (m2_sum * m2_sum) / N) / (var_y * (N - 1) / N)

    v1 = L - 1
    v2 = N - L
    F = (v2 * q) / (v1 * (1 - q))

    p_value = 1.0 - _ncf.cdf(F, v1, v2, lam)
    return float(p_value)


def interaction_type(q1, q2, q12):
    """Classify the interaction between two factors.

    Parameters
    ----------
    q1 : float
        q-statistic for factor 1.
    q2 : float
        q-statistic for factor 2.
    q12 : float
        q-statistic for the intersection of factor 1 ∩ factor 2.

    Returns
    -------
    int
        0: Weaken, nonlinear      (q12 < min(q1, q2))
        1: Weaken, uni-variable   (min <= q12 <= max)
        2: Enhance, bi-variable   (max < q12 < q1+q2)
        3: Independent            (q12 ≈ q1+q2)
        4: Enhance, nonlinear     (q12 > q1+q2)
    """
    if q12 < min(q1, q2):
        return 0
    elif q12 <= max(q1, q2):
        return 1
    elif q12 < q1 + q2:
        return 2
    elif abs(q12 - q1 - q2) < 1e-10:
        return 3
    elif q12 > q1 + q2:
        return 4
    else:
        # Should never reach here
        raise ValueError(
            f"Unable to classify interaction: q1={q1}, q2={q2}, q12={q12}"
        )


# Pre-compute SST for a target column to avoid repeated computation
def compute_sst(y):
    """Compute total sum of squares (SST) for a response vector."""
    y = np.asarray(y, dtype=float)
    return np.sum((y - np.mean(y)) ** 2)
