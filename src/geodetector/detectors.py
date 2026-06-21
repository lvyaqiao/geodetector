"""Detector classes: Factor, Interaction, Risk, Ecological.

Each detector follows sklearn conventions (fit / predict / score) where
applicable, and stores results as trailing-underscore attributes.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import f as _f_dist
from scipy.stats import ttest_ind

from ._base import BaseEstimator, RegressorMixin
from ._stats import interaction_type, q_significance_test, q_statistic
from ._types import INTERACTION_TYPES
from .discretize import discretize, should_discretize


def _prepare_x(x_col, *, discretize_method="quantile", n_strata=5):
    """Convert a column to integer stratum labels.

    Discretizes if necessary; factorizes string/object columns.
    """
    if should_discretize(x_col):
        return discretize(x_col, discretize_method=discretize_method, n_strata=n_strata)
    try:
        x_arr = np.asarray(x_col, dtype=float)
    except (ValueError, TypeError):
        return pd.factorize(x_col)[0]
    if np.allclose(x_arr, np.rint(x_arr)):
        return np.rint(x_arr).astype(int)
    return pd.factorize(x_col)[0]


# ──────────────────────────────────────────────────────────────
# FactorDetector
# ──────────────────────────────────────────────────────────────

class FactorDetector(RegressorMixin, BaseEstimator):
    """Factor detector (q-statistic) for a single explanatory variable.

    Parameters
    ----------
    discretize_method : str, default="quantile"
        Discretization method for continuous X.
    n_strata : int, default=5
        Number of strata for discretization.
    random_state : int, default=42
        Random seed (reserved for future use).
    """

    def __init__(self, *, discretize_method="quantile", n_strata=5, random_state=42):
        self.discretize_method = discretize_method
        self.n_strata = n_strata
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the factor detector.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, 1)
            A single-factor column.
        y : array-like of shape (n_samples,)
            Response variable.

        Returns
        -------
        self
        """
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=float).ravel()

        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty input data")

        x_col = X.iloc[:, 0]

        # Handle missing values
        valid_mask = ~(pd.isna(x_col) | pd.isna(y))
        if valid_mask.sum() < 3:
            self.q_value_ = np.nan
            self.p_value_ = 1.0
            self.n_strata_ = 0
            self._group_means_ = {}
            self._was_discretized_ = False
            return self

        x_col = x_col[valid_mask]
        y = y[valid_mask]

        self._was_discretized_ = should_discretize(x_col)
        if self._was_discretized_:
            x_discrete = discretize(x_col, discretize_method=self.discretize_method,
                                    n_strata=self.n_strata)
        else:
            x_discrete = _prepare_x(x_col)

        self.q_value_ = q_statistic(y, x_discrete)
        self.p_value_ = q_significance_test(y, x_discrete)
        unique, counts = np.unique(x_discrete, return_counts=True)
        self.n_strata_ = len(unique[counts >= 2])

        # Store group means for predict()
        self._group_means_ = {}
        for g in np.unique(x_discrete):
            mask = x_discrete == g
            if mask.sum() > 0:
                self._group_means_[g] = y[mask].mean()

        return self

    def predict(self, X):
        """Predict Y as the mean of the stratum each sample belongs to.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, 1)
            Input factor column.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values (stratum means).
        """
        from ._base import NotFittedError

        if not hasattr(self, "_group_means_"):
            raise NotFittedError(
                "This FactorDetector instance is not fitted yet. "
                "Call 'fit' before 'predict'."
            )

        X = pd.DataFrame(X)
        x_col = X.iloc[:, 0]

        if self._was_discretized_:
            x_prep = discretize(x_col, discretize_method=self.discretize_method,
                                n_strata=self.n_strata)
        else:
            x_prep = _prepare_x(x_col)

        result = np.array([
            self._group_means_.get(int(g), np.nan) for g in x_prep
        ])
        return result

    def transform(self, X):
        """Return discretized stratum labels for X.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, 1)
            Input factor column.

        Returns
        -------
        ndarray of shape (n_samples,)
            Integer stratum labels.
        """
        from ._base import NotFittedError

        if not hasattr(self, "_was_discretized_"):
            raise NotFittedError(
                "This FactorDetector instance is not fitted yet. "
                "Call 'fit' before 'transform'."
            )

        X = pd.DataFrame(X)
        x_col = X.iloc[:, 0]

        if self._was_discretized_:
            return discretize(x_col, discretize_method=self.discretize_method,
                              n_strata=self.n_strata)
        return _prepare_x(x_col)


# ──────────────────────────────────────────────────────────────
# InteractionDetector
# ──────────────────────────────────────────────────────────────

class InteractionDetector(BaseEstimator):
    """Interaction detector: pair-wise factor interactions.

    Tests whether the combined stratification of two factors enhances
    or weakens their individual explanatory power.

    Parameters
    ----------
    discretize_method : str, default="quantile"
        Discretization method for continuous X.
    n_strata : int, default=5
        Number of strata for discretization.

    Attributes (after fit)
    ----------------------
    interaction_q_ : pd.DataFrame
        Symmetric matrix of q-values for each factor pair.
    interaction_type_ : pd.DataFrame
        Symmetric matrix of interaction types (0-4).
    pairs_ : list of dict
        Detailed per-pair results.
    """

    def __init__(self, *, discretize_method="quantile", n_strata=5):
        self.discretize_method = discretize_method
        self.n_strata = n_strata

    def fit(self, X, y):
        """Fit the interaction detector.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_factors)
            Factor columns.
        y : array-like of shape (n_samples,)
            Response variable.

        Returns
        -------
        self
        """
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=float).ravel()

        if X.shape[1] < 2:
            raise ValueError(
                "Interaction detection requires at least 2 factors. "
                f"Got {X.shape[1]}."
            )

        factors = list(X.columns)
        n = len(factors)

        q_matrix = pd.DataFrame(np.eye(n), index=factors, columns=factors, dtype=float)
        type_matrix = pd.DataFrame(
            np.full((n, n), -1, dtype=int), index=factors, columns=factors
        )

        # Pre-compute individual q-values and their discretized columns
        q_vals = {}
        x_disc = {}
        for f in factors:
            valid = ~pd.isna(X[f]) & ~pd.isna(y)
            if valid.sum() < 3:
                q_vals[f] = np.nan
                continue
            xd = _prepare_x(X[f][valid], discretize_method=self.discretize_method,
                            n_strata=self.n_strata)
            x_disc[f] = xd
            q_vals[f] = q_statistic(y[valid], xd)
            q_matrix.loc[f, f] = q_vals[f]

        # Pairwise interactions
        self.pairs_ = []
        for f1, f2 in combinations(factors, 2):
            # Use a *single* valid mask for q1, q2 and q12 to ensure
            # the interaction type is computed on consistent data.
            valid_pair = ~pd.isna(X[f1]) & ~pd.isna(X[f2]) & ~pd.isna(y)
            if valid_pair.sum() < 3:
                q1_p = q2_p = q12 = np.nan
                itype = -1
            else:
                y_v = y[valid_pair]
                x1d = _prepare_x(X[f1][valid_pair], discretize_method=self.discretize_method,
                                 n_strata=self.n_strata)
                x2d = _prepare_x(X[f2][valid_pair], discretize_method=self.discretize_method,
                                 n_strata=self.n_strata)
                # Recompute q1, q2 on the pairwise-valid subset
                q1_p = q_statistic(y_v, x1d)
                q2_p = q_statistic(y_v, x2d)

                combined = pd.Series(
                    ["_".join(row) for row in zip(x1d.astype(str), x2d.astype(str))]
                )
                combined_codes = pd.factorize(combined)[0]
                q12 = q_statistic(y_v, combined_codes)
                itype = (interaction_type(q1_p, q2_p, q12)
                         if not np.isnan(q1_p) and not np.isnan(q2_p) and not np.isnan(q12)
                         else -1)

            # Maintain per-factor q-values from the widest valid sample
            q1 = q_vals.get(f1, np.nan)
            q2 = q_vals.get(f2, np.nan)

            q_matrix.loc[f1, f2] = q12
            q_matrix.loc[f2, f1] = q12
            type_matrix.loc[f1, f2] = itype
            type_matrix.loc[f2, f1] = itype

            self.pairs_.append({
                "factor_1": f1,
                "factor_2": f2,
                "q_1": q1,
                "q_2": q2,
                "q_1_pair": q1_p,
                "q_2_pair": q2_p,
                "q_12": q12,
                "interaction_type": itype,
                "interaction_label": INTERACTION_TYPES.get(itype, "Unknown"),
            })

        self.interaction_q_ = q_matrix.astype(float)
        self.interaction_type_ = type_matrix.astype(int)
        return self


# ──────────────────────────────────────────────────────────────
# RiskDetector
# ──────────────────────────────────────────────────────────────

class RiskDetector(BaseEstimator):
    """Risk detector: pairwise t-test between strata.

    Tests whether the mean Y differs significantly between any two
    strata of a factor.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level.
    discretize_method : str, default="quantile"
        Discretization method for continuous X.
    n_strata : int, default=5
        Number of strata for discretization.

    Attributes (after fit)
    ----------------------
    risk_result_ : dict
        Keys are factor names, values are DataFrames with columns:
        ``stratum_1``, ``stratum_2``, ``t_stat``, ``p_value``,
        ``significant``.
    """

    def __init__(self, *, alpha=0.05, discretize_method="quantile", n_strata=5):
        self.alpha = alpha
        self.discretize_method = discretize_method
        self.n_strata = n_strata

    def fit(self, X, y):
        """Fit the risk detector.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_factors)
            Factor columns.
        y : array-like of shape (n_samples,)
            Response variable.

        Returns
        -------
        self
        """
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=float).ravel()

        results = {}
        for col in X.columns:
            valid = ~pd.isna(X[col]) & ~pd.isna(y)
            if valid.sum() < 3:
                results[col] = pd.DataFrame()
                continue

            x_prep = _prepare_x(X[col][valid], discretize_method=self.discretize_method,
                                n_strata=self.n_strata)
            y_valid = y[valid]

            strata = sorted(np.unique(x_prep))
            if len(strata) < 2:
                results[col] = pd.DataFrame()
                continue

            rows = []
            for i in range(len(strata)):
                for j in range(i + 1, len(strata)):
                    s1, s2 = int(strata[i]), int(strata[j])
                    y1 = y_valid[x_prep == s1]
                    y2 = y_valid[x_prep == s2]
                    if len(y1) < 2 or len(y2) < 2:
                        continue
                    try:
                        t_stat, p_val = ttest_ind(y1, y2, equal_var=False)
                    except Exception:
                        t_stat, p_val = 0.0, 1.0
                    rows.append({
                        "factor": col,
                        "stratum_1": s1,
                        "stratum_2": s2,
                        "t_stat": float(t_stat),
                        "p_value": float(p_val),
                        "significant": p_val < self.alpha,
                    })
            results[col] = pd.DataFrame(rows)

        self.risk_result_ = results
        return self


# ──────────────────────────────────────────────────────────────
# EcologicalDetector
# ──────────────────────────────────────────────────────────────

class EcologicalDetector(BaseEstimator):
    """Ecological detector: F-test comparing two factors' effects.

    Tests whether the stratified heterogeneity (q-value) of two factors
    differs significantly.

    Notes
    -----
    This implementation uses the ratio of *unexplained* variance
    ``F = (1 - q1) / (1 - q2)`` with a two-tailed F-test at
    ``df1 = df2 = N - 1``.

    Differences from reference implementations:

    - **GD-main (gdeco)**: uses ``F = q2 / q1`` (ratio of q-values)
      with hardcoded critical value ``qf(0.9, N-1, N-1)``.
    - **gdverse (geodetector)**: uses the same ``F = (1-q1)/(1-q2)``
      formula but applies a *one-tailed* test
      (``pf(F, N-1, N-1, lower.tail=FALSE)``).
    - This implementation uses a **two-tailed** test, which is more
      conservative and symmetric (invariant to factor order).

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level.
    discretize_method : str, default="quantile"
        Discretization method for continuous X.
    n_strata : int, default=5
        Number of strata for discretization.

    Attributes (after fit)
    ----------------------
    eco_result_ : pd.DataFrame
        Columns: ``factor_1``, ``factor_2``, ``f_stat``, ``p_value``,
        ``significant``.
    """

    def __init__(self, *, alpha=0.05, discretize_method="quantile", n_strata=5):
        self.alpha = alpha
        self.discretize_method = discretize_method
        self.n_strata = n_strata

    def fit(self, X, y):
        """Fit the ecological detector.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_factors)
            Factor columns.
        y : array-like of shape (n_samples,)
            Response variable.

        Returns
        -------
        self
        """
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=float).ravel()

        if X.shape[1] < 2:
            raise ValueError(
                "Ecological detection requires at least 2 factors. "
                f"Got {X.shape[1]}."
            )

        N = len(y)
        factors = list(X.columns)

        # Compute q for each factor
        q_vals = {}
        for col in factors:
            valid = ~pd.isna(X[col]) & ~pd.isna(y)
            if valid.sum() < 3:
                q_vals[col] = np.nan
                continue
            x_prep = _prepare_x(X[col][valid], discretize_method=self.discretize_method,
                                n_strata=self.n_strata)
            q_vals[col] = q_statistic(y[valid], x_prep)

        rows = []
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                f1, f2 = factors[i], factors[j]
                q1, q2 = q_vals.get(f1, np.nan), q_vals.get(f2, np.nan)

                if np.isnan(q1) or np.isnan(q2):
                    rows.append({
                        "factor_1": f1, "factor_2": f2,
                        "f_stat": np.nan, "p_value": np.nan,
                        "significant": False,
                    })
                    continue

                # F = (1 - q1) / (1 - q2)  — ratio of unexplained variance
                denom = 1.0 - q2
                numer = 1.0 - q1

                if abs(denom) < 1e-12 and abs(numer) < 1e-12:
                    f_stat, p_val = 1.0, 1.0
                elif abs(denom) < 1e-12:
                    f_stat, p_val = float("inf"), 0.0
                elif abs(numer) < 1e-12:
                    f_stat, p_val = 0.0, 0.0
                else:
                    f_stat = numer / denom
                    # Two-tailed F-test with df1 = df2 = N-1
                    if f_stat >= 1:
                        p_val = 2.0 * _f_dist.sf(f_stat, N - 1, N - 1)
                    else:
                        p_val = 2.0 * _f_dist.cdf(f_stat, N - 1, N - 1)
                    p_val = min(p_val, 1.0)

                rows.append({
                    "factor_1": f1,
                    "factor_2": f2,
                    "f_stat": float(f_stat),
                    "p_value": float(p_val),
                    "significant": p_val < self.alpha,
                })

        self.eco_result_ = pd.DataFrame(rows)
        return self
