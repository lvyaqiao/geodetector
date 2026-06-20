"""LESH — Locally Explained Stratified Heterogeneity.

Shapley-value decomposition of q-statistics, attributing the combined
explanatory power to individual factors.

Reference
---------
Li Y, Luo P, Song Y, et al. 2023. A locally explained heterogeneity
model for examining wetland disparity. Int. J. Digital Earth.
"""

import math
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from .._stats import q_statistic
from ..discretize import discretize, should_discretize
from ..utils import all2int


def shapley_decompose(data, factors, target, *,
                      discretize_method="quantile",
                      method="gozh",
                      n_strata=5,
                      max_depth=5, min_samples_leaf=5,
                      random_state=42, progress=False):
    """Compute Shapley values for each factor's contribution to q.

    The Shapley value θ_j for factor j is:

        θ_j = Σ_{S ⊆ M\\{j}}  w(|S|) · [v(S ∪ {j}) - v(S)]

    where  w(|S|) = |S|! · (|M| - |S| - 1)! / |M|!
    and    v(S) is the q-statistic of the combined stratification of S.

    Warning: complexity is O(2^M). For M > 10, this may be slow.

    Parameters
    ----------
    data : pd.DataFrame
    factors : list of str
    target : str
    discretize_method : str, default "quantile"
        Discretization method for "quantile" mode.
    method : str, default "gozh"
        Discretization strategy:
        - ``"quantile"`` : use mapclassify discretization (data-driven).
        - ``"gozh"`` : use decision-tree discretization (data-driven).
    n_strata : int, default 5
        Number of strata for "quantile" mode.
    max_depth : int, default 5
        Maximum tree depth for "gozh" mode.
    min_samples_leaf : int, default 5
        Minimum samples per leaf for "gozh" mode.
    random_state : int, default 42
        Random seed for "gozh" mode.
    progress : bool, default False
        If True, prints progress information.

    Returns
    -------
    pd.DataFrame
        Columns: variable, shapley_value, shapley_pct.
        Sorted by shapley_value descending.
    """
    M = len(factors)
    if M > 12:
        warnings.warn(
            f"shapley_decompose with {M} factors requires "
            f"2^{M} = {2**M} evaluations. This may take a long time."
        )

    y = np.asarray(data[target], dtype=float).ravel()

    if method == "gozh":
        v = _compute_gozh_subsets(
            data, factors, target, y,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            progress=progress,
        )
    else:
        v = _compute_quantile_subsets(
            data, factors, target, y,
            discretize_method=discretize_method,
            n_strata=n_strata,
            progress=progress,
        )

    # Compute Shapley values (same for both methods)
    indices = list(range(M))
    factorial_vals = [math.factorial(k) for k in range(M + 1)]

    if progress:
        print(f"[LESH] Computing Shapley values for {M} factors ...")

    shapley = {}
    for j in range(M):
        theta = 0.0
        for r in range(M):
            for subset in combinations(indices, r):
                if j in subset:
                    continue
                S = frozenset(subset)
                S_union_j = frozenset(subset + (j,))

                v_S = v.get(S, 0.0)
                v_Sj = v.get(S_union_j, v_S)

                weight = (factorial_vals[r] * factorial_vals[M - r - 1]) / factorial_vals[M]
                theta += weight * (v_Sj - v_S)

        shapley[factors[j]] = max(0.0, theta)

    # Build result DataFrame
    total = sum(shapley.values())
    rows = []
    for f, val in sorted(shapley.items(), key=lambda x: -x[1]):
        rows.append({
            "variable": f,
            "shapley_value": val,
            "shapley_pct": val / total if total > 0 else 0.0,
        })

    return pd.DataFrame(rows)


def _compute_quantile_subsets(data, factors, target, y,
                               discretize_method, n_strata, progress):
    """Compute q-values for all subsets using quantile/etc. discretization."""
    indices = list(range(len(factors)))
    M = len(factors)

    if progress:
        print(f"[LESH] Computing v(S) for {2**M - 1} subsets (quantile) ...")

    v = {}
    for r in range(1, M + 1):
        for subset in combinations(indices, r):
            subset_key = frozenset(subset)
            if len(subset) == 1:
                idx = subset[0]
                factor = factors[idx]
                x_col = _prepare_col(data[factor], discretize_method=discretize_method,
                                     n_strata=n_strata)
                v[subset_key] = q_statistic(y, x_col)
            else:
                xs = []
                for idx in subset:
                    x_col = _prepare_col(data[factors[idx]],
                                         discretize_method=discretize_method,
                                         n_strata=n_strata)
                    xs.append(x_col.astype(str))
                combined = pd.Series(
                    ["_".join(row) for row in zip(*xs)]
                )
                combined_codes = pd.factorize(combined)[0]
                v[subset_key] = q_statistic(y, combined_codes)
    return v


def _compute_gozh_subsets(data, factors, target, y,
                           max_depth, min_samples_leaf, random_state, progress):
    """Compute q-values for all subsets using decision tree discretization."""
    try:
        from sklearn.tree import DecisionTreeRegressor
    except ImportError:
        raise ImportError(
            "GOZH method requires scikit-learn. Install with: pip install scikit-learn"
        )

    indices = list(range(len(factors)))
    M = len(factors)
    v = {}

    if progress:
        print(f"[LESH] Computing v(S) for {2**M - 1} subsets (GOZH) ...")

    for r in range(1, M + 1):
        for subset in combinations(indices, r):
            subset_key = frozenset(subset)
            subset_factors = [factors[i] for i in subset]

            X_subset = data[subset_factors].copy()
            # Convert to numeric
            for col in X_subset.columns:
                if X_subset[col].dtype.kind in ("O", "U", "S"):
                    X_subset[col] = pd.factorize(X_subset[col])[0]

            valid = ~X_subset.isna().any(axis=1) & ~pd.isna(y)
            if valid.sum() < 3:
                v[subset_key] = 0.0
                continue

            X_valid = X_subset[valid].values.astype(float)
            y_valid = y[valid]

            tree = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )
            tree.fit(X_valid, y_valid)
            strata = tree.apply(X_valid).ravel()
            v[subset_key] = q_statistic(y_valid, strata)

    return v


class LESH:
    """Full LESH analysis: Shapley decomposition + interaction attribution.

    Parameters
    ----------
    factors : list of str
        Column names of explanatory variables.
    target : str
        Column name of the response variable.
    discretize_method : str, default "quantile"
        Discretization method when ``method="quantile"``.
    method : str, default "gozh"
        Discretization strategy:
        - ``"quantile"`` : use mapclassify discretization.
        - ``"gozh"`` : use decision-tree discretization.
    n_strata : int, default 5
        Number of strata for "quantile" mode.
    max_depth : int, default 5
        Max tree depth for "gozh" mode.
    min_samples_leaf : int, default 5
        Min samples per leaf for "gozh" mode.
    random_state : int, default 42
        Random seed for "gozh" mode.
    progress : bool, default False

    Attributes (after fit)
    ----------------------
    shapley_ : pd.DataFrame
        Shapley decomposition: variable, shapley_value, shapley_pct.
    interaction_ : pd.DataFrame
        Interaction-detector result with Shapley-attributed contributions.
    """

    def __init__(self, factors, target, *,
                 discretize_method="quantile",
                 method="gozh",
                 n_strata=5,
                 max_depth=5, min_samples_leaf=5,
                 random_state=42, progress=False):
        self.factors = factors
        self.target = target
        self.discretize_method = discretize_method
        self.method = method
        self.n_strata = n_strata
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.progress = progress

    def fit(self, data):
        """Fit LESH: Shapley decomposition + interaction attribution.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain all ``factors`` columns and the ``target`` column.

        Returns
        -------
        self
        """
        from ..detectors import InteractionDetector

        # 1. Shapley decomposition
        self.shapley_ = shapley_decompose(
            data,
            self.factors,
            self.target,
            discretize_method=self.discretize_method,
            method=self.method,
            n_strata=self.n_strata,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            progress=self.progress,
        )

        # 2. Interaction detector
        X = data[self.factors]
        y = data[self.target]
        id_ = InteractionDetector(
            discretize_method=self.discretize_method,
            n_strata=self.n_strata,
        )
        id_.fit(X, y)

        # 3. Attribute interaction q to individual factors by Shapley proportion
        shapley_map = dict(zip(self.shapley_["variable"], self.shapley_["shapley_value"]))
        rows = []
        for pair in id_.pairs_:
            f1, f2 = pair["factor_1"], pair["factor_2"]
            spd1 = abs(shapley_map.get(f1, 0))
            spd2 = abs(shapley_map.get(f2, 0))
            denom = spd1 + spd2 if (spd1 + spd2) > 0 else 1.0
            rows.append({
                "factor_1": f1,
                "factor_2": f2,
                "q_1": pair["q_1"],
                "q_2": pair["q_2"],
                "q_12": pair["q_12"],
                "interaction_type": pair["interaction_type"],
                "interaction_label": pair["interaction_label"],
                "spd_1": spd1 / denom * pair["q_12"],
                "spd_2": spd2 / denom * pair["q_12"],
            })

        self.interaction_ = pd.DataFrame(rows)
        return self

    def summary(self):
        """Return a multi-line summary string."""
        lines = ["=" * 60, "  LESH Summary", "=" * 60]
        lines.append("")
        lines.append("Shapley decomposition:")
        lines.append("-" * 45)
        for _, row in self.shapley_.iterrows():
            lines.append(
                f"  {row['variable']:<15s}  "
                f"theta = {row['shapley_value']:.4f}  "
                f"({row['shapley_pct'] * 100:.1f}%)"
            )
        return "\n".join(lines)


def _prepare_col(x_col, discretize_method="quantile", n_strata=5):
    """Prepare a single column: discretize if continuous, else integer-code."""
    if should_discretize(x_col):
        return discretize(x_col, discretize_method=discretize_method, n_strata=n_strata)
    return all2int(x_col)
