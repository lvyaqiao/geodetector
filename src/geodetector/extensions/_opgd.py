"""OPGD — Optimal Parameter Geographical Detector.

Finds the best discretization method and number of strata for each
continuous factor by maximising the q-statistic, then runs all four
detectors on the optimally discretized data.

Reference
---------
Song Y, Wang J, Ge Y, Xu C. 2020. An optimal parameters-based
geographical detector model enhances geographic characteristics of
explanatory variables for spatial heterogeneity analysis.
GIScience & Remote Sensing 57(5): 593-610.
"""

import numpy as np
import pandas as pd

from ..discretize import (
    discretize,
    should_discretize,
    OptimalDiscretizer,
    _METHOD_MAP,
)
from .._stats import q_statistic
from ..utils import all2int


class OPGD:
    """Complete OPGD: optimal discretization + full geo-detection.

    Parameters
    ----------
    factors : list of str
        Column names of explanatory variables.
    target : str
        Column name of the response variable.
    discretize_methods : list of str, optional
        Discretization methods to search over.
        Default: ["sd", "equal", "geometric", "quantile", "natural"].
    k_range : tuple (min, max), default=(3, 8)
        Range of strata counts to try.
    alpha : float, default=0.05
        Significance level.
    random_state : int, default=42
        Random seed.

    Attributes (after fit)
    ----------------------
    opt_params_ : pd.DataFrame
        Optimal parameters per variable: columns ``variable``,
        ``method``, ``k``, ``q_value_before``.
    discretized_data_ : pd.DataFrame
        The optimally discretized copy of the data.
    q_values_ : pd.DataFrame
        Factor detector results on the discretized data.
    interaction_q_ : pd.DataFrame
        Interaction q-value matrix.
    interaction_type_ : pd.DataFrame
        Interaction type matrix (0-4).
    risk_result_ : dict
        Risk detector results.
    ecological_result_ : pd.DataFrame
        Ecological detector results.
    """

    def __init__(self, factors, target, *,
                 discretize_methods=None,
                 k_range=(3, 8),
                 alpha=0.05,
                 random_state=42):
        self.factors = factors
        self.target = target
        self.discretize_methods = discretize_methods if discretize_methods is not None else [
            "sd", "equal", "geometric", "quantile", "natural"
        ]
        self.k_range = k_range
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, data):
        """Fit OPGD.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain all ``factors`` columns and the ``target`` column.

        Returns
        -------
        self
        """
        y = np.asarray(data[self.target], dtype=float).ravel()
        disc_params = []
        discretized = data[self.factors].copy()

        for f in self.factors:
            x_col = data[f]
            if should_discretize(x_col):
                # Search optimal parameters
                od = OptimalDiscretizer(
                    discretize_methods=self.discretize_methods,
                    k_range=self.k_range,
                    random_state=self.random_state,
                )
                od.fit(data[[f]], y)
                if f in od.best_method_:
                    best_m = od.best_method_[f]
                    best_k = od.best_k_[f]
                    best_q = od.best_q_[f]
                    best_bins = od.bins_[f]
                    disc_params.append({
                        "variable": f,
                        "method": best_m,
                        "k": best_k,
                        "q_value_before": best_q,
                    })
                    # Discretize (NaN rows are excluded by GeoDetector.fit)
                    labels = np.digitize(
                        data[f].values,
                        best_bins,
                        right=True,
                    )
                    labels[np.isnan(data[f].values)] = -1
                    discretized[f] = labels.astype(int)
                else:
                    # Fallback: use default
                    discretized[f] = discretize(
                        data[f],
                        discretize_method="quantile",
                        n_strata=self.k_range[0],
                    )
                    disc_params.append({
                        "variable": f,
                        "method": "quantile",
                        "k": self.k_range[0],
                        "q_value_before": np.nan,
                    })
            else:
                discretized[f] = all2int(data[f])
                disc_params.append({
                    "variable": f,
                    "method": "categorical",
                    "k": data[f].nunique(),
                    "q_value_before": np.nan,
                })

        self.opt_params_ = pd.DataFrame(disc_params)
        self.discretized_data_ = discretized

        # Run standard GD on optimally discretized data
        from ..geodetector import GeoDetector

        gd = GeoDetector(
            factors=self.factors,
            target=self.target,
            discretize_method="quantile",  # data already discretized, not used
            n_strata=self.k_range[0],
            alpha=self.alpha,
            random_state=self.random_state,
        )
        df_disc = discretized.copy()
        df_disc[self.target] = data[self.target].values
        gd.fit(df_disc)

        self.q_values_ = gd.q_values_
        self.interaction_q_ = gd.interaction_q_
        self.interaction_type_ = gd.interaction_type_
        self.risk_result_ = gd.risk_result_
        self.ecological_result_ = gd.ecological_result_

        return self

    def summary(self):
        """Return a multi-line summary string."""
        lines = []
        lines.append("=" * 60)
        lines.append("  OPGD Summary — Optimal Parameters")
        lines.append("=" * 60)
        lines.append("")
        lines.append("Optimal discretization per variable:")
        lines.append("-" * 45)
        for _, row in self.opt_params_.iterrows():
            lines.append(
                f"  {row['variable']:<15s}  method={row['method']:<10s}  "
                f"k={int(row['k'])}  q={row['q_value_before']:.4f}"
            )
        lines.append("")

        # Factor detector on discretized data
        lines.append("Factor Detector (optimal):")
        lines.append("-" * 45)
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
