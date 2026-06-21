"""Discretization module for converting continuous variables into strata.

Uses mapclassify for classification methods (Jenks, Quantile, Equal Interval,
etc.) and provides automatic heuristics for detecting which columns need
discretization.
"""

import numpy as np
import pandas as pd
from mapclassify import (
    EqualInterval,
    FisherJenks,
    MaximumBreaks,
    Quantiles,
    StdMean,
)

from ._base import BaseEstimator
from ._stats import q_statistic

# ── Method name resolution ───────────────────────────────────────

_METHOD_MAP = {
    "quantile":     Quantiles,
    "equal":        EqualInterval,
    "jenks":        FisherJenks,
    "natural":      FisherJenks,
    "nb":           FisherJenks,
    "fisher_jenks": FisherJenks,
    "maxbreaks":    MaximumBreaks,
    "maximum":      MaximumBreaks,
    "sd":           StdMean,
    "stdev":        StdMean,
    "geometric":    None,  # handled separately in discretize()
}


def discretize(x, *, discretize_method="quantile", n_strata=5):
    """Discretize a continuous array into integer-coded strata.

    Parameters
    ----------
    x : array-like
        Continuous values (1-D).
    discretize_method : str, default="quantile"
        Classification method. One of: "quantile", "equal", "jenks"
        ("natural", "nb"), "geometric", "maximum", "sd" ("stdev").
    n_strata : int, default=5
        Number of strata / classes.

    Returns
    -------
    ndarray of int
        Integer stratum labels in [0, n_strata-1].
    """
    x = np.asarray(x, dtype=float).ravel()

    # Route geometric to our own implementation
    if discretize_method.lower() == "geometric":
        from .extensions._geometric import discretize_geometric
        return discretize_geometric(x, n_strata=n_strata)

    x_clean = x[~np.isnan(x)]

    if len(x_clean) == 0:
        raise ValueError("discretize: all input values are NaN")

    n_unique = len(np.unique(x_clean))
    # Cap strata count at number of unique values
    n = min(n_strata, n_unique)
    if n < 2:
        return np.zeros(len(x), dtype=int)

    mc_class = _METHOD_MAP.get(discretize_method.lower())
    if mc_class is None:
        raise ValueError(
            f"Unknown discretization method: {discretize_method!r}. "
            f"Available: {list(_METHOD_MAP.keys())}"
        )

    try:
        mc_inst = mc_class(x_clean, k=n)
    except Exception as e:
        raise RuntimeError(
            f"Discretization failed for method={discretize_method}, k={n}: {e}"
        ) from e

    # digitize: bins are left-open, right-closed
    # Result labels start at 0
    labels = np.digitize(x, mc_inst.bins, right=True)
    # Fill NaN values with -1 (will be excluded downstream)
    labels[np.isnan(x)] = -1
    return labels.astype(int)


def should_discretize(x, max_levels=None):
    """Heuristic: should this column be discretized?

    A column is treated as already-stratified (categorical) if:
    - dtype is object / string / boolean / category
    - number of unique values <= max_levels

    Parameters
    ----------
    x : array-like or pd.Series
        Input column.
    max_levels : int, optional
        Maximum number of unique values to be considered categorical.
        Defaults to ``min(20, sqrt(N))``.

    Returns
    -------
    bool
        True if the column should be discretized.
    """
    if isinstance(x, pd.Series):
        dtype = x.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            return False
        if dtype == np.dtype(object) or dtype == "string":
            return False
        if dtype == np.dtype(bool):
            return False
        if dtype.kind in ("O", "U", "S"):
            return False

    x_arr = np.asarray(x)
    if x_arr.dtype.kind in ("O", "U", "S", "b"):
        return False

    valid = x_arr[~pd.isna(x_arr)]
    if len(valid) == 0:
        return False

    n_unique = len(np.unique(valid))
    N = len(valid)

    if max_levels is None:
        max_levels = min(20, int(np.sqrt(N)))

    return n_unique > max_levels


# ── Transformer classes ──────────────────────────────────────────

class Discretizer(BaseEstimator):
    """Discretize continuous columns into integer strata.

    Parameters
    ----------
    discretize_method : str, default="quantile"
        Discretization method.
    n_strata : int, default=5
        Number of strata.
    """

    def __init__(self, *, discretize_method="quantile", n_strata=5):
        self.discretize_method = discretize_method
        self.n_strata = n_strata

    def fit(self, X, y=None):
        """Compute bin edges for each column that needs discretization.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : ignored
            (Present for sklearn compatibility.)

        Returns
        -------
        self
        """
        X = pd.DataFrame(X)
        self.bins_ = {}
        self._columns_to_discretize_ = []

        for col in X.columns:
            if should_discretize(X[col]):
                x_clean = X[col].dropna().values
                n = min(self.n_strata, len(np.unique(x_clean)))
                mc_class = _METHOD_MAP.get(self.discretize_method.lower(), Quantiles)
                mc_inst = mc_class(x_clean, k=max(n, 2))
                self.bins_[col] = mc_inst.bins
                self._columns_to_discretize_.append(col)

        return self

    def transform(self, X):
        """Discretize columns using fitted bin edges.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        pd.DataFrame
            Discretized copy. Categorical columns are left unchanged
            but integer-coded.
        """
        X = pd.DataFrame(X)
        result = X.copy()

        for col, bins in self.bins_.items():
            col_vals = result[col].values
            labels = np.digitize(col_vals, bins, right=True)
            labels[np.isnan(col_vals)] = -1
            result[col] = labels.astype(int)

        # Integer-code any remaining categorical columns
        from .utils import all2int

        for col in X.columns:
            if col not in self.bins_:
                result[col] = all2int(X[col])

        return result

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class OptimalDiscretizer(BaseEstimator):
    """Find the optimal discretization method and number of strata.

    Exhaustively tries (method × k) combinations and selects the one
    that maximises the q-statistic for each column.

    Parameters
    ----------
    discretize_methods : list of str, optional
        Discretization methods to try.
        Default: ["quantile", "equal", "jenks"].
    k_range : tuple (min_k, max_k), default=(3, 8)
        Range of strata counts to try.
    random_state : int, default=42
        Random seed (unused, reserved for future use).
    """

    def __init__(self, *, discretize_methods=None, k_range=(3, 8), random_state=42):
        self.discretize_methods = discretize_methods or ["quantile", "equal", "jenks"]
        self.k_range = k_range
        self.random_state = random_state

    def fit(self, X, y):
        """Find optimal discretization parameters per column.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : array-like
            Response variable.

        Returns
        -------
        self
        """
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=float).ravel()

        self.best_method_ = {}
        self.best_k_ = {}
        self.best_q_ = {}
        self.bins_ = {}

        for col in X.columns:
            if not should_discretize(X[col]):
                continue

            valid_mask = ~X[col].isna()
            x_vals = X[col].loc[valid_mask].values
            y_valid = y[valid_mask]

            best_q = -float("inf")
            best_method = None
            best_k = None
            best_bins = None

            n_unique = len(np.unique(x_vals))
            k_min = max(2, min(self.k_range[0], n_unique))
            k_max = max(k_min, min(self.k_range[1], n_unique))

            for method in self.discretize_methods:
                mc_class = _METHOD_MAP.get(method.lower())
                if mc_class is None:
                    continue

                for k in range(k_min, k_max + 1):
                    try:
                        mc_inst = mc_class(x_vals, k=max(k, 2))
                        labels = np.digitize(x_vals, mc_inst.bins, right=True)
                        q = q_statistic(y_valid, labels)
                        if q > best_q:
                            best_q = q
                            best_method = method
                            best_k = k
                            best_bins = mc_inst.bins
                    except Exception:
                        continue

            if best_method is not None:
                self.best_method_[col] = best_method
                self.best_k_[col] = best_k
                self.best_q_[col] = best_q
                self.bins_[col] = best_bins

        return self

    def transform(self, X):
        """Discretize columns using the optimal parameters found during fit.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        pd.DataFrame
            Discretized copy.
        """
        X = pd.DataFrame(X)
        result = X.copy()

        for col, bins in self.bins_.items():
            col_vals = result[col].values
            labels = np.digitize(col_vals, bins, right=True)
            labels[np.isnan(col_vals)] = -1
            result[col] = labels.astype(int)

        from .utils import all2int

        for col in X.columns:
            if col not in self.bins_:
                result[col] = all2int(X[col])

        return result

    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
