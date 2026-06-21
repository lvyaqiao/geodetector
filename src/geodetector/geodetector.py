"""GeoDetector — master orchestrator for all four detectors.

Provides a single-entry API that fits all detectors at once and stores
results as attributes, matching the R GD / gdverse workflow.
"""

from typing import Optional

import pandas as pd

from ._base import BaseEstimator
from .detectors import (
    EcologicalDetector,
    FactorDetector,
    InteractionDetector,
    RiskDetector,
)
from .utils import validate_data


class GeoDetector(BaseEstimator):
    """Master class for geographical detector analysis.

    Parameters
    ----------
    factors : list of str, optional
        Column names of explanatory variables. If None, all columns
        except ``target`` are used.
    target : str, optional
        Column name of the response variable. If None, the first
        column is used.
    discretize : str, default="quantile"
        Discretization method for continuous factors.
    n_strata : int, default=5
        Number of strata for discretization.
    alpha : float, default=0.05
        Significance level for statistical tests.
    random_state : int, default=42
        Random seed.

    Attributes (after fit)
    ----------------------
    q_values_ : pd.DataFrame
        Factor detector: columns ``variable``, ``q_value``,
        ``p_value``, ``significant``.
    interaction_q_ : pd.DataFrame
        Interaction q-value matrix.
    interaction_type_ : pd.DataFrame
        Interaction type matrix (0-4).
    risk_result_ : dict
        Risk detector results per factor.
    ecological_result_ : pd.DataFrame
        Ecological detector: columns ``factor_1``, ``factor_2``,
        ``f_stat``, ``p_value``, ``significant``.

    Examples
    --------
    >>> from geodetector import GeoDetector
    >>> from geodetector.dataset import load_disease
    >>> df = load_disease()
    >>> gd = GeoDetector(factors=["type", "region", "level"], target="incidence")
    >>> gd.fit(df)
    >>> gd.q_values_
    >>> gd.plot()
    """

    def __init__(self,
                 factors: Optional[list] = None,
                 target: Optional[str] = None,
                 *,
                 discretize_method: str = "quantile",
                 n_strata: int = 5,
                 alpha: float = 0.05,
                 random_state: int = 42):
        self.factors = factors
        self.target = target
        self.discretize_method = discretize_method
        self.n_strata = n_strata
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, data: pd.DataFrame):
        """Fit all detectors.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain all ``factors`` columns and the ``target`` column.

        Returns
        -------
        self
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        # Resolve factors and target
        if self.factors is None:
            if self.target is None:
                self.target = data.columns[0]
            self.factors = [c for c in data.columns if c != self.target]
        if self.target is None:
            self.target = data.columns[0]
            self.factors = [c for c in self.factors if c != self.target]

        validate_data(data, self.factors, self.target)

        X = data[self.factors]
        y = data[self.target]

        # Cache for later plotting (e.g., risk_mean, discretization)
        self._data_cache = data.copy()

        # ── Factor Detector ─────────────────────────────────
        q_rows = []
        for f in self.factors:
            fd = FactorDetector(
                discretize_method=self.discretize_method,
                n_strata=self.n_strata,
                random_state=self.random_state,
            )
            fd.fit(X[[f]], y)
            q_rows.append({
                "variable": f,
                "q_value": fd.q_value_,
                "p_value": fd.p_value_,
                "significant": fd.p_value_ < self.alpha,
            })
        self.q_values_ = pd.DataFrame(q_rows)

        # ── Interaction Detector ────────────────────────────
        if len(self.factors) >= 2:
            id_ = InteractionDetector(
                discretize_method=self.discretize_method,
                n_strata=self.n_strata,
            )
            id_.fit(X, y)
            self.interaction_q_ = id_.interaction_q_
            self.interaction_type_ = id_.interaction_type_
        else:
            self.interaction_q_ = None
            self.interaction_type_ = None

        # ── Risk Detector ───────────────────────────────────
        rd = RiskDetector(
            alpha=self.alpha,
            discretize_method=self.discretize_method,
            n_strata=self.n_strata,
        )
        rd.fit(X, y)
        # RiskDetector always returns dict now
        self.risk_result_ = rd.risk_result_

        # ── Ecological Detector ─────────────────────────────
        if len(self.factors) >= 2:
            ed = EcologicalDetector(
                alpha=self.alpha,
                discretize_method=self.discretize_method,
                n_strata=self.n_strata,
            )
            ed.fit(X, y)
            self.ecological_result_ = ed.eco_result_
        else:
            self.ecological_result_ = None

        return self

    # ── Plotting (thin wrappers over plotting.py) ───────────

    def plot(self, **kwargs):
        """Plot q-values as a horizontal bar chart.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from .plotting import plot_factor
        return plot_factor(self.q_values_, sig_level=self.alpha, **kwargs)

    def plot_interaction(self, style="heatmap", **kwargs):
        """Plot interaction matrix.

        Parameters
        ----------
        style : str, default="heatmap"
            "heatmap" or "bubble".

        Returns
        -------
        matplotlib.axes.Axes
        """
        from .plotting import plot_interaction
        return plot_interaction(
            self.interaction_q_,
            self.interaction_type_,
            style=style,
            **kwargs,
        )

    def plot_risk(self, **kwargs):
        """Plot risk detector matrices.

        Returns
        -------
        dict of matplotlib.axes.Axes or single Axes
        """
        from .plotting import plot_risk
        return plot_risk(self.risk_result_, **kwargs)

    def plot_ecological(self, **kwargs):
        """Plot ecological detector matrix.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from .plotting import plot_ecological
        return plot_ecological(self.ecological_result_, **kwargs)

    def plot_dashboard(self, **kwargs):
        """Plot all four detectors on one figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .plotting import plot_dashboard
        return plot_dashboard(self, **kwargs)

    def plot_risk_mean(self, **kwargs):
        """Plot mean response per stratum for each factor.

        Returns
        -------
        dict of matplotlib.axes.Axes or matplotlib.figure.Figure
        """
        from .plotting import compute_risk_means, plot_risk_mean
        X = pd.DataFrame({f: self._data_cache[f] for f in self.factors})
        y = self._data_cache[self.target]
        means = compute_risk_means(X, y, discretize_method=self.discretize_method,
                                   n_strata=self.n_strata)
        return plot_risk_mean(means, **kwargs)

    def plot_optimal_discretization(self, data=None, **kwargs):
        """Plot optimal discretization process: q-k curves + histogram.

        Parameters
        ----------
        data : pd.DataFrame, optional
            If None, uses the data passed to ``fit()``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .plotting import plot_optimal_discretization
        if data is None:
            if not hasattr(self, "_data_cache"):
                raise ValueError("No data available. Pass data= or call fit() first.")
            data = self._data_cache
        return plot_optimal_discretization(
            data, self.factors, self.target,
            n_strata=self.n_strata, **kwargs,
        )

    # ── Summary ─────────────────────────────────────────────

    def summary(self) -> str:
        """Return a formatted multi-line summary of all results.

        Returns
        -------
        str
        """
        lines = []
        lines.append("=" * 60)
        lines.append("  GeoDetector Summary")
        lines.append("=" * 60)

        # Factor detector
        lines.append("")
        lines.append("Factor Detector (q-values):")
        lines.append("-" * 45)
        col_w = max(len(v) for v in self.factors) if self.factors else 10
        for _, row in self.q_values_.iterrows():
            sig_mark = ""
            if row["p_value"] < 0.001:
                sig_mark = "  ***"
            elif row["p_value"] < 0.01:
                sig_mark = "  **"
            elif row["p_value"] < 0.05:
                sig_mark = "  *"
            lines.append(
                f"  {row['variable']:<{col_w}s}  "
                f"q = {row['q_value']:.4f}  "
                f"p = {row['p_value']:.4f}{sig_mark}"
            )

        # Interaction detector
        if self.interaction_q_ is not None:
            lines.append("")
            lines.append("Interaction Detector (q-values matrix):")
            lines.append("-" * 45)
            lines.append(self.interaction_q_.round(4).to_string())

        # Risk detector (summary by factor)
        lines.append("")
        lines.append("Risk Detector (significant stratum pairs):")
        lines.append("-" * 45)
        for factor, df in self.risk_result_.items():
            if df is not None and not df.empty:
                sig_pairs = df[df["significant"]]
                lines.append(
                    f"  {factor}: {len(sig_pairs)}/{len(df)} "
                    f"pairs significantly different"
                )

        # Ecological detector
        if self.ecological_result_ is not None:
            lines.append("")
            lines.append("Ecological Detector (pairwise F-test):")
            lines.append("-" * 45)
            for _, row in self.ecological_result_.iterrows():
                sig = " *" if row["significant"] else ""
                lines.append(
                    f"  {row['factor_1']} vs {row['factor_2']}: "
                    f"F = {row['f_stat']:.3f}, p = {row['p_value']:.4f}{sig}"
                )

        lines.append("")
        return "\n".join(lines)
