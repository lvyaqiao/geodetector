"""RGD — Robust Geographical Detector.

Uses variance-based change-point detection to find natural strata
boundaries, providing discretization that is robust to outliers.

If the ``ruptures`` library is available, it is used for offline
change-point detection. Otherwise, a simple variance-minimisation
heuristic is used as fallback.

Reference
---------
Zhang Z, Song Y, Wu P. 2022. Robust geographical detector.
Int. J. Appl. Earth Obs. Geoinf. 109, 102782.
"""

import numpy as np
import pandas as pd

from ..geodetector import GeoDetector
from .._stats import q_statistic


def robust_discretize(x, k, *, y=None, minsize=1, random_state=42):
    """Discretize using variance-based change point detection.

    Matches the gdverse RGD approach: sorts data by X, then uses the Y
    values (sorted in X order) as the change-point detection signal.
    This finds locations along the X gradient where Y changes abruptly,
    creating strata that maximise Y contrast between zones.

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        Continuous variable.
    k : int
        Target number of strata.
    y : array-like of shape (n_samples,), optional
        Response variable. Required for the gdverse-compatible path
        (uses Y sorted by X as change-point signal). When None,
        falls back to X-only quantile-based discretization.
    minsize : int, default=1
        Minimum number of observations per stratum.
    random_state : int, default=42

    Returns
    -------
    ndarray of int
        Integer stratum labels.
    """
    x = np.asarray(x, dtype=float).ravel()
    valid = ~np.isnan(x)
    if y is not None:
        y_arr = np.asarray(y, dtype=float).ravel()
        valid = valid & ~np.isnan(y_arr)
        y_clean = y_arr[valid]

    x_clean = x[valid]
    n = len(x_clean)
    n_unique = len(np.unique(x_clean))

    if n < 3 or n_unique < 2:
        lbl = np.full(len(x), -1, dtype=int)
        lbl[valid] = 0
        return lbl

    n_k = min(k, n_unique)
    sort_idx = np.argsort(x_clean)

    has_y = y is not None
    try:
        import ruptures as rpt
        _HAS_RUPTURES = True
    except ImportError:
        _HAS_RUPTURES = False

    if _HAS_RUPTURES and has_y:
        # gdverse‑compatible: sort Y by X, detect change points in Y
        y_sorted = y_clean[sort_idx]
        algo = rpt.Dynp(model="l2", min_size=max(2, minsize))
        try:
            bkps = algo.fit(y_sorted).predict(n_bkps=n_k - 1)
        except Exception:
            _HAS_RUPTURES = False
    elif _HAS_RUPTURES:
        # Fallback when Y is not available — detect change points in X itself
        x_sorted = x_clean[sort_idx]
        signal = (x_sorted - x_sorted.mean()) / (x_sorted.std() + 1e-10)
        algo = rpt.Dynp(model="l2", min_size=max(2, minsize))
        try:
            bkps = algo.fit(signal).predict(n_bkps=n_k - 1)
        except Exception:
            _HAS_RUPTURES = False

    if not _HAS_RUPTURES:
        # Fallback: quantile-based with variance refinement (on X)
        x_sorted = x_clean[sort_idx]
        q_breaks = np.percentile(x_sorted, np.linspace(0, 100, n_k + 1))[1:-1]
        refined = list(q_breaks)
        for _ in range(3):
            for i in range(len(refined)):
                b = refined[i]
                delta = max(abs(b) * 0.1, 0.01)
                nbhd = np.linspace(b - delta, b + delta, 20)
                best_b = b
                best_var = float("inf")
                for candidate in nbhd:
                    if i == 0:
                        left = x_sorted[x_sorted <= candidate]
                    else:
                        left = x_sorted[
                            (x_sorted > refined[i-1]) & (x_sorted <= candidate)
                        ]
                    if i == len(refined) - 1:
                        right = x_sorted[x_sorted > candidate]
                    else:
                        right = x_sorted[
                            (x_sorted > candidate) & (x_sorted <= refined[i+1])
                        ]
                    if len(left) >= minsize and len(right) >= minsize:
                        total_var = np.var(left) + np.var(right)
                        if total_var < best_var:
                            best_var = total_var
                            best_b = candidate
                refined[i] = best_b
        bkps = refined

    # Assign labels. When Y was used, bkps are *indices* into the sorted
    # array.  When X was used, bkps are actual X-value breakpoints.
    labels = np.full(len(x), -1, dtype=int)

    if has_y and _HAS_RUPTURES:
        # bkps are segment endpoint indices (e.g. [8, 15, 20])
        # Assign each valid point the label of the segment it falls into
        if len(bkps) > 0:
            sort_pos = np.empty(len(x_clean), dtype=int)
            sort_pos[sort_idx] = np.arange(len(x_clean))
            labels_sorted = np.searchsorted(bkps[:-1], sort_pos, side="right")
            labels[valid] = labels_sorted
        else:
            labels[valid] = 0
    else:
        # bkps are X-value breakpoints
        break_points = sorted(bkps) if isinstance(bkps, (list, np.ndarray)) else []
        if len(break_points) > 0:
            bins = np.array([-np.inf] + list(break_points) + [np.inf])
            labels[valid] = np.digitize(x_clean, bins[1:-1], right=True)
        else:
            labels[valid] = 0

    return labels.astype(int)


class RGD:
    """Robust Geographical Detector.

    Uses variance-based change point detection to find natural strata,
    then runs the standard detectors.

    Parameters
    ----------
    factors : list of str
        Column names of explanatory variables.
    target : str
        Column name of the response variable.
    k : int, default=5
        Target number of strata (fallback when discnum is not set).
    discnum : sequence of int, optional
        Range of strata numbers to search over (default: range(3, 9)).
        When provided, the detector searches multiple k values and selects
        the optimal one per factor.
    strategy : int, default=2
        Optimal k selection strategy:
        - 1: pick k that maximizes q-value.
        - 2: use LOESS elbow detection (increase_rate threshold).
    increase_rate : float, default=0.05
        Critical increase rate for LOESS elbow detection (strategy=2).
    alpha : float, default=0.05
    random_state : int, default=42

    Attributes (after fit)
    ----------------------
    discretized_data_ : pd.DataFrame
    n_strata_ : dict
        Number of strata per factor (optimal).
    opt_discnum_ : dict
        Optimal discnum per factor.
    all_q_values_ : pd.DataFrame
        q-values for all discnums (columns: variable, discnum, q_value).
    q_values_ / interaction_q_ / ... : standard GD results.
    """

    def __init__(self, factors, target, *,
                 k=5, discnum=None, strategy=2,
                 increase_rate=0.05, alpha=0.05, random_state=42):
        self.factors = factors
        self.target = target
        self.k = k
        self.discnum = discnum if discnum is not None else range(3, 9)
        self.strategy = strategy
        self.increase_rate = increase_rate
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, data):
        """Fit RGD.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain all ``factors`` columns and the ``target`` column.

        Returns
        -------
        self
        """
        from ..discretize import should_discretize
        from ..utils import all2int

        y = data[self.target].values
        discnums = list(self.discnum)

        # Determine which factors need discretization
        continuous_factors = [f for f in self.factors
                              if should_discretize(data[f])]
        categorical_factors = [f for f in self.factors
                               if f not in continuous_factors]

        if len(discnums) <= 1 or len(continuous_factors) == 0:
            # Single-k mode (backward compatible)
            discretized = data[self.factors].copy()
            self.n_strata_ = {}
            for f in self.factors:
                if f in continuous_factors:
                    x_d = robust_discretize(
                        data[f], self.k,
                        y=y,
                        random_state=self.random_state,
                    )
                    discretized[f] = x_d
                    self.n_strata_[f] = len(np.unique(x_d[x_d >= 0]))
                else:
                    discretized[f] = all2int(data[f])
                    self.n_strata_[f] = data[f].nunique()
            self._finalize_fit(discretized, data)
            self.opt_discnum_ = {}
            self.all_q_values_ = pd.DataFrame()
            return self

        # Multi-k mode: search over discnums
        from .._stats import q_statistic

        all_rows = []
        all_discretized = {}  # discnum -> discretized DataFrame

        for dk in discnums:
            discretized = data[self.factors].copy()
            for f in self.factors:
                if f in continuous_factors:
                    x_d = robust_discretize(
                        data[f], dk,
                        y=y,
                        random_state=self.random_state,
                    )
                    discretized[f] = x_d
                else:
                    discretized[f] = all2int(data[f])
            all_discretized[dk] = discretized

            # Compute q-values for this discnum
            for f in self.factors:
                xd = discretized[f].values
                valid = ~pd.isna(xd) & ~pd.isna(y)
                if valid.sum() < 3:
                    qv = np.nan
                else:
                    qv = q_statistic(y[valid], xd[valid])
                all_rows.append({
                    "variable": f,
                    "discnum": dk,
                    "q_value": qv,
                })

        self.all_q_values_ = pd.DataFrame(all_rows)

        # Select optimal discnum per factor
        self.opt_discnum_ = {}
        for f in self.factors:
            f_rows = self.all_q_values_[self.all_q_values_["variable"] == f]
            if len(f_rows) == 0:
                self.opt_discnum_[f] = self.k
                continue
            if f_rows["q_value"].isna().all():
                self.opt_discnum_[f] = discnums[0]
                continue
            opt = _select_optimal_discnum(
                f_rows["discnum"].values,
                f_rows["q_value"].values,
                self.strategy,
                self.increase_rate,
            )
            self.opt_discnum_[f] = opt

        # Build final discretized data using optimal discnums
        final_disc = data[self.factors].copy()
        self.n_strata_ = {}
        for f in self.factors:
            opt_k = self.opt_discnum_.get(f, self.k)
            if f in continuous_factors:
                x_d = robust_discretize(
                    data[f], opt_k,
                    y=y,
                    random_state=self.random_state,
                )
                final_disc[f] = x_d
                self.n_strata_[f] = len(np.unique(x_d[x_d >= 0]))
            else:
                final_disc[f] = all2int(data[f])
                self.n_strata_[f] = data[f].nunique()

        self._finalize_fit(final_disc, data)
        return self

    def _finalize_fit(self, discretized, data):
        """Run standard GD on discretized data."""
        self.discretized_data_ = discretized

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
        self.interaction_q_ = gd.interaction_q_
        self.interaction_type_ = gd.interaction_type_
        self.risk_result_ = gd.risk_result_
        self.ecological_result_ = gd.ecological_result_

    def summary(self):
        """Return a multi-line summary string."""
        lines = ["=" * 60, "  RGD Summary", "=" * 60]
        lines.append("")
        if hasattr(self, "opt_discnum_") and self.opt_discnum_:
            lines.append("Optimal discnum per variable:")
            lines.append("-" * 30)
            for f, n in self.opt_discnum_.items():
                lines.append(f"  {f:<15s}  discnum={n}")
        else:
            lines.append("Strata per variable:")
            lines.append("-" * 30)
            for f, n in self.n_strata_.items():
                lines.append(f"  {f:<15s}  {n} strata")
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

    def plot(self, **kwargs):
        from ..plotting import plot_factor
        return plot_factor(self.q_values_, **kwargs)

    def plot_interaction(self, style="heatmap", **kwargs):
        from ..plotting import plot_interaction
        return plot_interaction(self.interaction_q_, self.interaction_type_,
                                style=style, **kwargs)


def _loess_smooth(x, y, frac=0.6):
    """LOESS (locally estimated scatterplot smoothing) via local linear regression.

    Parameters
    ----------
    x : ndarray, sorted
        Independent variable.
    y : ndarray
        Response variable.
    frac : float
        Fraction of data used in each local fit (0 < frac <= 1).

    Returns
    -------
    ndarray
        Smoothed y values, same shape as x.
    """
    n = len(x)
    if n < 3:
        return y.copy()

    r = int(np.ceil(frac * n))
    y_smooth = np.empty(n)

    for i in range(n):
        # Find the r nearest neighbours in x
        dist = np.abs(x - x[i])
        idx = np.argpartition(dist, min(r, n - 1))[:r]
        d_max = dist[idx].max()
        if d_max == 0:
            y_smooth[i] = y[i]
            continue

        # Tricube weights
        u = dist[idx] / d_max
        w = np.where(u < 1, (1 - u ** 3) ** 3, 0)

        if np.sum(w) < 1e-10:
            y_smooth[i] = y[i]
            continue

        # Weighted linear regression
        X = np.column_stack([np.ones(r), x[idx]])
        W_sqrt = np.sqrt(w)
        Xw = X * W_sqrt[:, None]
        yw = y[idx] * W_sqrt
        try:
            beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
            y_smooth[i] = beta[0] + beta[1] * x[i]
        except np.linalg.LinAlgError:
            y_smooth[i] = y[i]

    return y_smooth


def _select_optimal_discnum(discnums, q_values, strategy, increase_rate):
    """Select optimal discnum per factor.

    Parameters
    ----------
    discnums : ndarray
    q_values : ndarray
    strategy : int
        1 = max q, 2 = LOESS elbow
    increase_rate : float

    Returns
    -------
    int
    """
    # Remove NaN
    valid = ~np.isnan(q_values)
    if not valid.any():
        return int(discnums[0])
    d = discnums[valid]
    q = q_values[valid]

    if len(d) == 1:
        return int(d[0])

    # Sort by discnum
    order = np.argsort(d)
    d_sorted = d[order].astype(float)
    q_sorted = q[order].astype(float)

    if strategy == 1:
        return int(d_sorted[np.argmax(q_sorted)])

    # Strategy 2: LOESS elbow detection
    if len(d_sorted) < 2:
        return int(d_sorted[0])

    q_range = q_sorted[-1] - q_sorted[0]
    if q_range <= 0:
        return int(d_sorted[-1])

    # Smooth q-values using LOESS
    q_smooth = _loess_smooth(d_sorted, q_sorted)

    # Find elbow on smoothed curve: first point where incremental gain
    # (normalized by total range) falls below the increase_rate threshold
    for i in range(1, len(d_sorted)):
        delta = q_smooth[i] - q_smooth[i - 1]
        if delta > 0 and delta / q_range < increase_rate:
            return int(d_sorted[i - 1])

    return int(d_sorted[-1])
