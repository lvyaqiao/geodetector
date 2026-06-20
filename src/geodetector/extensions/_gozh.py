"""GOZH — Geographically Optimal Zones-based Heterogeneity.

Uses a decision tree regressor to automatically find optimal spatial
zones (strata) from continuous variables.

Reference
---------
Luo P, Song Y, et al. 2022. Geographically optimal zones-based
heterogeneity (GOZH) model. ISPRS J. Photogramm. Remote Sens.
"""

import numpy as np
import pandas as pd

from ..geodetector import GeoDetector


def rpart_discretize(x, y, *, max_depth=5, min_samples_leaf=5,
                     random_state=42):
    """Discretize a continuous variable using a decision tree.

    Fits a DecisionTreeRegressor on (x) → y, then returns the
    terminal node index of each sample as its stratum label.

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        Single continuous variable.
    y : array-like of shape (n_samples,)
        Response variable.
    max_depth : int, default=5
        Maximum tree depth.
    min_samples_leaf : int, default=5
        Minimum samples per leaf node.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    ndarray of int
        Integer stratum labels.
    """
    try:
        from sklearn.tree import DecisionTreeRegressor
    except ImportError:
        raise ImportError(
            "GOZH requires scikit-learn. Install with: pip install scikit-learn"
        )

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 3:
        return np.zeros(len(x), dtype=int)

    x_valid = x[valid].reshape(-1, 1)
    y_valid = y[valid]

    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree.fit(x_valid, y_valid)

    # Terminal node for each sample
    labels = np.full(len(x), -1, dtype=int)
    labels[valid] = tree.apply(x_valid).ravel()
    return labels


class GOZH:
    """GOZH: decision-tree-based discretization + full geo-detection.

    Parameters
    ----------
    factors : list of str
        Column names of explanatory variables.
    target : str
        Column name of the response variable.
    max_depth : int, default=5
        Maximum decision tree depth for zoning.
    min_samples_leaf : int, default=5
        Minimum samples per leaf.
    alpha : float, default=0.05
        Significance level.
    random_state : int, default=42

    Attributes (after fit)
    ----------------------
    discretized_data_ : pd.DataFrame
        GOZH-discretized data.
    n_zones_ : dict
        Number of zones (strata) per factor.
    q_values_ / interaction_q_ / ... : standard GD results.
    """

    def __init__(self, factors, target, *,
                 max_depth=5, min_samples_leaf=5,
                 alpha=0.05, random_state=42):
        self.factors = factors
        self.target = target
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, data):
        """Fit GOZH.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain all ``factors`` columns and the ``target`` column.

        Returns
        -------
        self
        """
        y = data[self.target].values
        discretized = data[self.factors].copy()
        self.n_zones_ = {}

        for f in self.factors:
            x_col = data[f]
            x_d = rpart_discretize(
                x_col, y,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            discretized[f] = x_d
            n = len(np.unique(x_d[x_d >= 0]))
            self.n_zones_[f] = n

        self.discretized_data_ = discretized

        # Run standard GD
        df_disc = discretized.copy()
        df_disc[self.target] = data[self.target].values
        gd = GeoDetector(
            factors=self.factors,
            target=self.target,
            n_strata=5,  # not used, data already discretized
            alpha=self.alpha,
            random_state=self.random_state,
        )
        gd.fit(df_disc)

        self.q_values_ = gd.q_values_
        self.risk_result_ = gd.risk_result_
        self.ecological_result_ = gd.ecological_result_

        # Interaction: joint decision tree per pair (gdverse method)
        self._compute_gozh_interaction(data, y)

        return self

    def _compute_gozh_interaction(self, data, y):
        """Compute interaction q-values using joint decision trees per pair.

        Matches gdverse's approach: for each pair (f1, f2),
        fits rpart(y ~ f1 + f2) and uses terminal nodes as combined strata.
        q1 and q2 are recomputed on the pairwise-valid subset for consistency.
        """
        from itertools import combinations

        try:
            from sklearn.tree import DecisionTreeRegressor
        except ImportError:
            self.interaction_q_ = pd.DataFrame()
            self.interaction_type_ = pd.DataFrame()
            self.interaction_pairs_ = []
            return

        from .._stats import q_statistic, interaction_type
        from .._types import INTERACTION_TYPES

        factors_list = list(self.factors)
        n = len(factors_list)

        # Build q matrix and type matrix
        q_matrix = pd.DataFrame(
            np.eye(n), index=factors_list, columns=factors_list, dtype=float
        )
        type_matrix = pd.DataFrame(
            np.full((n, n), -1, dtype=int), index=factors_list, columns=factors_list
        )
        for i, f in enumerate(factors_list):
            q_full = self.q_values_.loc[self.q_values_["variable"] == f, "q_value"].values
            q_matrix.loc[f, f] = q_full[0] if len(q_full) > 0 else np.nan

        self.interaction_pairs_ = []

        if n < 2:
            self.interaction_q_ = q_matrix
            self.interaction_type_ = type_matrix
            return

        for f1, f2 in combinations(factors_list, 2):
            # Build joint feature matrix
            X_pair = data[[f1, f2]].copy()
            for col in X_pair.columns:
                if X_pair[col].dtype.kind in ("O", "U", "S"):
                    X_pair[col] = pd.factorize(X_pair[col])[0]

            valid = ~X_pair.isna().any(axis=1) & ~pd.isna(y)
            if valid.sum() < 3:
                q12 = np.nan
                q1_pair = np.nan
                q2_pair = np.nan
                itype = -1
            else:
                y_valid = y[valid]

                # Individual tree for f1 on pairwise subset
                x1_arr = np.asarray(X_pair[f1][valid], dtype=float).reshape(-1, 1)
                tree1 = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                )
                tree1.fit(x1_arr, y_valid)
                strata1 = tree1.apply(x1_arr).ravel()
                q1_pair = q_statistic(y_valid, strata1)

                # Individual tree for f2 on pairwise subset
                x2_arr = np.asarray(X_pair[f2][valid], dtype=float).reshape(-1, 1)
                tree2 = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                )
                tree2.fit(x2_arr, y_valid)
                strata2 = tree2.apply(x2_arr).ravel()
                q2_pair = q_statistic(y_valid, strata2)

                # Joint tree for interaction
                X_valid = np.asarray(X_pair[valid], dtype=float)
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                )
                tree.fit(X_valid, y_valid)
                joint_strata = tree.apply(X_valid).ravel()
                q12 = q_statistic(y_valid, joint_strata)

                itype = (
                    interaction_type(q1_pair, q2_pair, q12)
                    if not np.isnan(q1_pair) and not np.isnan(q2_pair) and not np.isnan(q12)
                    else -1
                )

            q_matrix.loc[f1, f2] = q12
            q_matrix.loc[f2, f1] = q12
            type_matrix.loc[f1, f2] = itype
            type_matrix.loc[f2, f1] = itype

            self.interaction_pairs_.append({
                "factor_1": f1,
                "factor_2": f2,
                "q_1": q1_pair,
                "q_2": q2_pair,
                "q_12": q12,
                "interaction_type": itype,
                "interaction_label": INTERACTION_TYPES.get(itype, "Unknown"),
            })

        self.interaction_q_ = q_matrix.astype(float)
        self.interaction_type_ = type_matrix.astype(int)

    def summary(self):
        """Return a multi-line summary string."""
        lines = ["=" * 60, "  GOZH Summary", "=" * 60]
        lines.append("")
        lines.append("Zones per variable:")
        lines.append("-" * 30)
        for f, n in self.n_zones_.items():
            lines.append(f"  {f:<15s}  {n} zones")
        lines.append("")
        lines.append("Factor Detector:")
        lines.append("-" * 30)
        for _, row in self.q_values_.iterrows():
            sig = "***" if row["p_value"] < 0.001 else (
                "**" if row["p_value"] < 0.01 else (
                    "*" if row["p_value"] < 0.05 else ""
                )
            )
            lines.append(
                f"  {row['variable']:<15s}  "
                f"q = {row['q_value']:.4f}  "
                f"p = {row['p_value']:.4f}{sig}"
            )
        return "\n".join(lines)
