"""Plotting functions for geodetector results — gdverse / GD R compatible.

Style follows the gdverse R package (ggplot2 aesthetics) where possible.
All functions are pure: they accept data and return matplotlib Axes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from itertools import combinations
import warnings

# ── Global colour palette (matching GD / gdverse) ─────────────

INTERACTION_COLORS = {
    0: ("Weaken, nonlinear",    "#6EE9EF"),
    1: ("Weaken, uni-variable",  "#558DE8"),
    2: ("Enhance, bi-variable",  "#F2C55E"),
    3: ("Independent",           "#E08338"),
    4: ("Enhance, nonlinear",    "#EA4848"),
}

PRIMARY_COLOR   = "#DE3533"   # gdverse red for top q variable
SECONDARY_COLOR = "#808080"   # gdverse gray for other variables
RISK_YES        = "#FFA500"
RISK_NO         = "#7FDBFF"
MEAN_MAX_COLOR  = "#DE3533"
MEAN_MIN_COLOR  = "#2E86AB"
MEAN_MID_COLOR  = "#808080"


# ═══════════════════════════════════════════════════════════════
# 1. Factor detector plot
# ═══════════════════════════════════════════════════════════════

def plot_factor(q_values, *, ax=None, sig_level=0.05, slicenum=2,
                keep=True, qlabelsize=3.88, figsize=None, **kwargs):
    """Horizontal bar chart of q-values (gdverse style).

    Parameters
    ----------
    q_values : pd.DataFrame
        Columns ``variable``, ``q_value``, ``p_value``.
    ax : matplotlib Axes, optional
    sig_level : float, default=0.05
        Threshold for filtering when ``keep=False``.
    slicenum : int, default=2
        Number of top variables to show labels inside the bars.
    keep : bool, default=True
        If False, only show variables with p < (1 - sig_level).
    qlabelsize : float, default=3.88
        Font size multiplier for q-value labels.
    figsize : tuple, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (8, max(3, len(q_values) * 0.5)))

    df = q_values.dropna(subset=["q_value"]).copy()

    if not keep:
        df = df[df["p_value"] <= 1 - sig_level]

    if df.empty:
        ax.text(0.5, 0.5, "No significant variables", ha="center", va="center",
                transform=ax.transAxes)
        return ax

    df = df.sort_values("q_value", ascending=True).reset_index(drop=True)
    n = len(df)

    # Colours: first (highest q) red, others gray — gdverse convention
    colors = [PRIMARY_COLOR if i == n - 1 else SECONDARY_COLOR for i in range(n)]

    bars = ax.barh(range(n), df["q_value"], color=colors,
                   edgecolor="white", height=0.6, linewidth=0.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels(df["variable"], fontfamily="serif")
    ax.set_xlabel("Q value", fontfamily="serif")
    ax.set_xlim(0, min(1.0, np.nanmax(df["q_value"]) * 1.15 + 0.02))
    ax.invert_yaxis()
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    # Text: top `slicenum` inside bars (hjust right), rest outside (hjust left)
    q_max = np.nanmax(df["q_value"]) or 1.0
    for i in range(n):
        qv = df.iloc[i]["q_value"]
        q_str = f"{qv * 100:.2f}%"
        # Significance stars
        pv = df.iloc[i]["p_value"]
        if pv < 0.001:
            q_str += "  ***"
        elif pv < 0.01:
            q_str += "  **"
        elif pv < 0.05:
            q_str += "  *"
        if i >= n - slicenum:  # top variables: label inside
            ax.text(qv * 0.98, i, q_str, ha="right", va="center",
                    fontsize=qlabelsize + 6, fontweight="bold", color="white",
                    fontfamily="serif")
        else:  # others: label outside
            ax.text(qv + q_max * 0.01, i, q_str, ha="left", va="center",
                    fontsize=qlabelsize + 6, fontweight="bold",
                    fontfamily="serif")

    return ax


# ═══════════════════════════════════════════════════════════════
# 2. Interaction detector plot
# ═══════════════════════════════════════════════════════════════

def plot_interaction(interaction_q, interaction_type=None, *,
                     ax=None, style="heatmap", figsize=None,
                     alpha=0.95, **kwargs):
    """Plot interaction matrix.

    Parameters
    ----------
    interaction_q : pd.DataFrame
        Symmetric q-value matrix.
    interaction_type : pd.DataFrame, optional
        Symmetric type matrix (0-4). Used in ``"bubble"`` style.
    ax : matplotlib Axes, optional
    style : str, default="heatmap"
        ``"heatmap"`` or ``"bubble"``.
    figsize : tuple, optional
    alpha : float, default=0.95
        Opacity for bubble style.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if style == "heatmap":
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (5.5, 5))
        import seaborn as sns
        sns.heatmap(
            interaction_q.astype(float),
            annot=True, fmt=".3f", cmap="crest",
            vmin=0, vmax=1, linewidths=0.5, linecolor="white",
            ax=ax, cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Interaction q-values", fontfamily="serif")
        ax.tick_params(axis="both", labelsize=9)
        plt.tight_layout()
        return ax

    elif style == "bubble":
        factors = list(interaction_q.columns)
        n = len(factors)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (n * 1.4, n * 1.4))

        q_max = max(np.nanmax(interaction_q.values), 0.01)
        type_seen = set()

        for i, f1 in enumerate(factors):
            for j, f2 in enumerate(factors):
                qv = interaction_q.loc[f1, f2]
                y_pos = n - 1 - j
                if i == j:
                    ax.scatter(i, y_pos, s=qv / q_max * 500, c="gray",
                               edgecolors="black", linewidth=0.8, alpha=0.7, zorder=3)
                    ax.text(i, y_pos, f"{qv:.2f}", ha="center", va="center",
                            fontsize=8, fontweight="bold", color="black")
                elif i < j:
                    color = "gray"
                    label = ""
                    if interaction_type is not None:
                        t = int(interaction_type.loc[f1, f2])
                        if t in INTERACTION_COLORS:
                            label, color = INTERACTION_COLORS[t]
                            type_seen.add(t)
                    ax.scatter(i, y_pos, s=qv / q_max * 500, c=color,
                               edgecolors="black", linewidth=0.8, alpha=alpha, zorder=3)
                    ax.text(i, y_pos, f"{qv:.2f}", ha="center", va="center",
                            fontsize=7)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(factors, fontfamily="serif")
        ax.set_yticklabels([factors[n - 1 - k] for k in range(n)], fontfamily="serif")
        ax.set_xlim(-0.7, n - 0.3)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_aspect("equal")
        ax.set_title("Interaction q-values", fontfamily="serif")

        if type_seen:
            handles = []
            for t in sorted(type_seen):
                label, color = INTERACTION_COLORS[t]
                handles.append(Patch(facecolor=color, edgecolor="black", label=label))
            ax.legend(handles=handles, loc="lower left", fontsize=7,
                      title="Interaction", title_fontsize=8, ncol=1)

        plt.tight_layout()
        return ax

    else:
        raise ValueError(f"Unknown style: {style!r}. Use 'heatmap' or 'bubble'.")


# ═══════════════════════════════════════════════════════════════
# 3. Risk detector plot
# ═══════════════════════════════════════════════════════════════

def plot_risk(risk_result, *, factor=None, ax=None, figsize=None,
              show_labels=True, **kwargs):
    """Plot risk detector as a coloured grid matrix.

    Parameters
    ----------
    risk_result : dict or pd.DataFrame
        Either ``GeoDetector.risk_result_`` (dict of DataFrames)
        or a single DataFrame for one factor.
    factor : str, optional
        Which factor to plot from a dict.
    ax : matplotlib Axes, optional
    figsize : tuple, optional
    show_labels : bool, default=True

    Returns
    -------
    matplotlib.axes.Axes or dict of Axes
    """
    if isinstance(risk_result, pd.DataFrame):
        return _plot_risk_one(risk_result, ax=ax, show_labels=show_labels, **kwargs)

    if factor is not None:
        return _plot_risk_one(risk_result[factor], ax=ax, show_labels=show_labels, title=factor, **kwargs)

    factors = list(risk_result.keys())
    valid = [f for f in factors
             if risk_result[f] is not None and not risk_result[f].empty]
    if not valid:
        if ax is None:
            fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No significant comparisons", ha="center", va="center")
        return ax

    if len(valid) == 1:
        return _plot_risk_one(risk_result[valid[0]], ax=ax, show_labels=show_labels,
                              title=valid[0], **kwargs)

    ncols = min(3, len(valid))
    nrows = (len(valid) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize or (ncols * 3.8, nrows * 3.5))
    axes = np.atleast_1d(axes).ravel()
    for i, f in enumerate(valid):
        _plot_risk_one(risk_result[f], ax=axes[i], show_labels=show_labels,
                       title=f, **kwargs)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    return dict(zip(valid, axes[:len(valid)]))


def _plot_risk_one(df, *, ax=None, show_labels=True, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax
    strata = sorted(set(df["stratum_1"]) | set(df["stratum_2"]))
    n = len(strata)
    if n < 2:
        ax.text(0.5, 0.5, "Single stratum", ha="center", va="center")
        return ax
    idx_map = {s: i for i, s in enumerate(strata)}
    matrix = np.full((n, n), np.nan)
    for _, row in df.iterrows():
        i = idx_map[row["stratum_1"]]
        j = idx_map[row["stratum_2"]]
        val = 1 if row["significant"] else 0
        matrix[i, j] = val
        matrix[j, i] = val
    cmap = ListedColormap([RISK_NO, RISK_YES])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    if show_labels and not np.all(np.isnan(matrix)):
        for i in range(n):
            for j in range(n):
                if not np.isnan(matrix[i, j]) and i != j:
                    label = "Y" if matrix[i, j] == 1 else "N"
                    ax.text(j, i, label, ha="center", va="center",
                            fontsize=9, fontweight="bold", color="black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(strata)
    ax.set_yticklabels(strata)
    if title:
        ax.set_title(title, fontfamily="serif", color="#DE3533", fontsize=12)
    return ax


# ═══════════════════════════════════════════════════════════════
# 4. Ecological detector plot
# ═══════════════════════════════════════════════════════════════

def plot_ecological(eco_result, *, ax=None, figsize=None,
                    show_labels=True, **kwargs):
    """Plot ecological detector as a coloured grid matrix.

    Parameters
    ----------
    eco_result : pd.DataFrame
        Columns ``factor_1``, ``factor_2``, ``significant``.
    ax : matplotlib Axes, optional
    figsize : tuple, optional
    show_labels : bool, default=True

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (4, 4))
    if eco_result is None or eco_result.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax
    factors = sorted(set(eco_result["factor_1"]) | set(eco_result["factor_2"]))
    n = len(factors)
    if n < 2:
        ax.text(0.5, 0.5, "Single factor", ha="center", va="center")
        return ax
    idx_map = {f: i for i, f in enumerate(factors)}
    matrix = np.full((n, n), np.nan)
    for _, row in eco_result.iterrows():
        i = idx_map[row["factor_1"]]
        j = idx_map[row["factor_2"]]
        matrix[i, j] = 1 if row["significant"] else 0
        matrix[j, i] = matrix[i, j]
    cmap = ListedColormap([RISK_NO, RISK_YES])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    if show_labels and not np.all(np.isnan(matrix)):
        for i in range(n):
            for j in range(n):
                if not np.isnan(matrix[i, j]) and i != j:
                    label = "Y" if matrix[i, j] == 1 else "N"
                    ax.text(j, i, label, ha="center", va="center",
                            fontsize=10, fontweight="bold", color="black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(factors);
    ax.set_yticklabels(factors)
    ax.set_title("Ecological Detector", fontfamily="serif")
    return ax


# ═══════════════════════════════════════════════════════════════
# 5. LESH pie chart (NEW — gdverse scatterpie equivalent)
# ═══════════════════════════════════════════════════════════════

def plot_lesh(lesh, *, ax=None, figsize=None,
              pieradius_factor=16, pielegend_num=3, **kwargs):
    """Shapley pie chart on interaction grid (gdverse style).

    Each interaction pair is shown as a pie chart whose radius is
    proportional to the interaction q-value and whose two slices
    show the Shapley-attributed contributions of each factor.

    Parameters
    ----------
    lesh : LESH instance (after fit) or pd.DataFrame
        Must have columns ``factor_1``, ``factor_2``, ``q_12``,
        ``spd_1``, ``spd_2``.
    ax : matplotlib Axes, optional
    figsize : tuple, optional
    pieradius_factor : float, default=16
        Scale factor for pie radius.
    pielegend_num : int, default=3
        Number of legend entries for the size legend.

    Returns
    -------
    matplotlib.axes.Axes
    """
    # Accept both LESH class instance and raw DataFrame
    if hasattr(lesh, "interaction_"):
        df = lesh.interaction_.copy()
    else:
        df = lesh.copy()

    required = {"factor_1", "factor_2", "q_12", "spd_1", "spd_2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    factors = sorted(set(df["factor_1"]) | set(df["factor_2"]))
    n = len(factors)
    idx = {f: i for i, f in enumerate(factors)}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (n * 1.3 + 2, n * 1.3 + 1))

    q_max = max(df["q_12"].max(), 0.01)
    radius_scale = pieradius_factor * 0.04 / (n if n > 0 else 1)

    # Scatter points and pie wedges
    for _, row in df.iterrows():
        f1, f2 = row["factor_1"], row["factor_2"]
        i, j = idx[f1], idx[f2]
        x, y = float(i), float(j)
        r = radius_scale * row["q_12"]
        if r <= 0:
            continue
        # Two slices
        frac1 = row["spd_1"] / row["q_12"] if row["q_12"] > 0 else 0.5
        # Draw pie wedge for factor 1 (color #75c7af — gdverse green)
        theta1 = 0
        theta2 = frac1 * 360
        wedges = ax.add_patch(plt.matplotlib.patches.Wedge(
            (x, y), r, theta1, theta2,
            facecolor="#75c7af", edgecolor="white", linewidth=0.3, alpha=0.85,
        ))
        # Draw pie wedge for factor 2 (color #fb9872 — gdverse orange)
        wedges = ax.add_patch(plt.matplotlib.patches.Wedge(
            (x, y), r, theta2, 360,
            facecolor="#fb9872", edgecolor="white", linewidth=0.3, alpha=0.85,
        ))

    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_aspect("equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(factors, fontfamily="serif", color="#75c7af")
    ax.set_yticklabels(factors, fontfamily="serif", color="#fb9872")
    ax.tick_params(axis="x", colors="#75c7af")
    ax.tick_params(axis="y", colors="#fb9872")
    ax.set_title("LESH — Shapley Decomposition", fontfamily="serif")

    # Size legend
    legend_qs = np.linspace(q_max * 0.3, q_max, pielegend_num)
    leg_x = ax.get_xlim()[1] - 0.15
    leg_y = ax.get_ylim()[0] + 0.5
    for lq in legend_qs:
        lr = radius_scale * lq
        ax.add_patch(plt.matplotlib.patches.Circle(
            (leg_x, leg_y), lr, facecolor="gray", edgecolor="black",
            linewidth=0.5, alpha=0.5,
        ))
        ax.text(leg_x - lr - 0.08, leg_y, f"{lq:.2f}",
                ha="right", va="center", fontsize=7, fontfamily="serif")
        leg_y += max(lr * 3, 0.3)

    # Legend for factor colours
    legend_elements = [
        Patch(facecolor="#75c7af", label="Variable (X-axis)"),
        Patch(facecolor="#fb9872", label="Variable (Y-axis)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8,
              framealpha=0.9, title="Shapley", title_fontsize=9)

    plt.tight_layout()
    return ax


# ═══════════════════════════════════════════════════════════════
# 6. Optimal discretization process plot (NEW)
# ═══════════════════════════════════════════════════════════════

def plot_optimal_discretization(data, factors, target, *,
                                 methods=None, k_range=(3, 8),
                                 ax=None, figsize=None, **kwargs):
    """Plot optimal discretization process: q vs k curves + histogram.

    Grid-searches discretization methods × number-of-strata for each
    factor, then plots (A) q-value curves and (B) histogram with
    optimal breakpoints.

    Parameters
    ----------
    data : pd.DataFrame
    factors : list of str
    target : str
    methods : list of str, optional
        Default: ``["sd", "equal", "geometric", "quantile", "natural"]``.
    k_range : tuple (min, max), default=(3, 8)
    ax : unused (creates own subplot grid)
    figsize : tuple, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .discretize import should_discretize, _METHOD_MAP
    from ._stats import q_statistic

    if methods is None:
        methods = ["sd", "equal", "geometric", "quantile", "natural"]

    y_arr = np.asarray(data[target], dtype=float).ravel()

    # Collect search data for each continuous factor
    disc_factors = [f for f in factors if should_discretize(data[f])]
    if not disc_factors:
        fig, ax0 = plt.subplots()
        ax0.text(0.5, 0.5, "No continuous factors to discretize",
                 ha="center", va="center")
        return fig

    search_data = {}
    best_params = {}
    for f in disc_factors:
        x_arr = data[f].dropna().values
        valid = ~pd.isna(data[f]) & ~pd.isna(data[target])
        xv = data[f][valid].values
        yv = y_arr[valid]
        n_unique = len(np.unique(xv))
        k_min = max(2, min(k_range[0], n_unique))
        k_max = max(k_min, min(k_range[1], n_unique))

        rows = []
        best_q = -1.0
        best_combo = None
        for method in methods:
            mc_class = _METHOD_MAP.get(method.lower())
            if mc_class is None:
                continue
            for k in range(k_min, k_max + 1):
                try:
                    mc_inst = mc_class(xv, k=max(k, 2))
                    lbls = np.digitize(xv, mc_inst.bins, right=True)
                    q = q_statistic(yv, lbls)
                    if not np.isnan(q):
                        rows.append({"method": method, "k": k, "q": q})
                        if q > best_q:
                            best_q = q
                            best_combo = (method, k, mc_inst.bins)
                except Exception:
                    continue
        if rows:
            search_data[f] = pd.DataFrame(rows)
            best_params[f] = best_combo

    # Plot
    nd = len(disc_factors)
    ncols = min(2, nd)
    nrows = nd * 2 // ncols  # two rows per factor (curves + histogram)
    # Actually, one panel per factor with two sub-subplots
    # Simpler: one figure with nd rows x 2 cols
    fig, axes = plt.subplots(nd, 2, figsize=figsize or (12, 3 * nd))
    if nd == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for row_idx, f in enumerate(disc_factors):
        ax_curves = axes[row_idx, 0]
        ax_hist = axes[row_idx, 1]

        sd = search_data[f]
        if sd.empty:
            continue

        # (A) q vs k curves
        marker_map = dict(zip(methods, ["o", "s", "D", "^", "v", "p", "*", "X"][:len(methods)]))
        for method in methods:
            sub = sd[sd["method"] == method]
            if sub.empty:
                continue
            sub = sub.sort_values("k")
            ax_curves.plot(sub["k"], sub["q"], marker=marker_map.get(method, "o"),
                           label=method, linewidth=1.5, markersize=6)
        ax_curves.set_title(f"{f}", fontfamily="serif", fontweight="bold")
        ax_curves.set_xlabel("Number of intervals", fontfamily="serif")
        ax_curves.set_ylabel("Q value", fontfamily="serif")
        ax_curves.legend(fontsize=7, loc="lower right")
        ax_curves.set_xticks(sorted(sd["k"].unique()))
        ax_curves.grid(alpha=0.3)
        ax_curves.set_axisbelow(True)

        # (B) Histogram with optimal breakpoints
        xv = data[f].dropna().values.astype(float)
        ax_hist.hist(xv, bins=30, color="gray", edgecolor="gray", alpha=0.7)
        ax_hist.set_xlabel("Variable", fontfamily="serif")
        ax_hist.set_ylabel("Frequency", fontfamily="serif")
        if f in best_params and best_params[f] is not None:
            _, _, bins = best_params[f]
            for b in bins[1:-1]:
                ax_hist.axvline(b, color="red", linewidth=1.5, linestyle="-")
        ax_hist.set_title(f"{f} — optimal discretization", fontfamily="serif",
                          fontsize=10)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# 7. Risk mean plot (NEW)
# ═══════════════════════════════════════════════════════════════

def plot_risk_mean(mean_data, *, ax=None, figsize=None, **kwargs):
    """Horizontal bar chart of mean response per stratum.

    Parameters
    ----------
    mean_data : pd.DataFrame or dict
        If DataFrame: ``stratum`` and ``mean`` columns.
        If dict: mapping factor name → DataFrame with ``stratum`` and ``mean``.
    ax : matplotlib Axes, optional
    figsize : tuple, optional

    Returns
    -------
    matplotlib.axes.Axes or matplotlib.figure.Figure
    """
    if isinstance(mean_data, pd.DataFrame):
        return _plot_risk_mean_one(mean_data, ax=ax, **kwargs)

    # dict: multiple factors
    factors = list(mean_data.keys())
    valid = [f for f in factors if mean_data[f] is not None and not mean_data[f].empty]
    if not valid:
        if ax is None:
            fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    ncols = min(3, len(valid))
    nrows = (len(valid) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (ncols * 4, nrows * 3.5))
    axes = np.atleast_1d(axes).ravel()
    for i, f in enumerate(valid):
        _plot_risk_mean_one(mean_data[f], ax=axes[i], title=f)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    return fig


def _plot_risk_mean_one(df, *, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    # Support both "stratum"/"mean" and older column names
    str_col = "stratum" if "stratum" in df.columns else df.columns[0]
    mean_col = "mean" if "mean" in df.columns else df.columns[1]
    df = df.sort_values(mean_col, ascending=True).reset_index(drop=True)
    n = len(df)

    vals = df[mean_col].values
    vmin, vmax = vals.min(), vals.max()
    if vmax > vmin:
        colors = [MEAN_MIN_COLOR if v == vmin else
                  MEAN_MAX_COLOR if v == vmax else MEAN_MID_COLOR
                  for v in vals]
    else:
        colors = [MEAN_MID_COLOR] * n

    ax.barh(range(n), vals, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(df[str_col].astype(str))
    ax.set_xlabel("Mean Value", fontfamily="serif")
    if title:
        ax.set_title(title, fontfamily="serif", fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    # Text labels
    for i, v in enumerate(vals):
        ax.text(v + (vmax - vmin) * 0.01, i, f"{v:.3f}",
                va="center", fontsize=9, fontfamily="serif")
    return ax


# ═══════════════════════════════════════════════════════════════
# 8. Dashboard
# ═══════════════════════════════════════════════════════════════

def plot_dashboard(gd, *, figsize=None, **kwargs):
    """Four-panel dashboard: Factor, Interaction, Risk, Ecological.

    Parameters
    ----------
    gd : GeoDetector
        A fitted ``GeoDetector`` instance.
    figsize : tuple, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    has_interaction = (gd.interaction_q_ is not None)
    has_ecological = (gd.ecological_result_ is not None)

    n_panels = 1 + has_interaction + 1 + has_ecological
    ncols = min(2, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize or (ncols * 6, nrows * 5))
    axes = np.atleast_1d(axes).ravel()

    idx = 0
    plot_factor(gd.q_values_, ax=axes[idx], sig_level=gd.alpha)
    idx += 1

    if has_interaction:
        plot_interaction(gd.interaction_q_, gd.interaction_type_,
                         ax=axes[idx], style="heatmap")
        idx += 1

    if gd.risk_result_:
        first_factor = list(gd.risk_result_.keys())[0]
        plot_risk(gd.risk_result_, factor=first_factor, ax=axes[idx])
    idx += 1

    if has_ecological:
        plot_ecological(gd.ecological_result_, ax=axes[idx])

    for j in range(idx, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# Helper for computing risk means
# ═══════════════════════════════════════════════════════════════

def compute_risk_means(X, y, discretize_method="quantile", n_strata=5):
    """Compute mean Y per stratum for each factor.

    Parameters
    ----------
    X : pd.DataFrame
        Factor columns.
    y : array-like
        Response variable.
    discretize_method : str
    n_strata : int

    Returns
    -------
    dict
        factor name → pd.DataFrame with columns ``stratum``, ``mean``.
    """
    from .discretize import discretize, should_discretize
    from .utils import all2int

    y_arr = np.asarray(y, dtype=float).ravel()
    result = {}
    for col in X.columns:
        x_col = X[col]
        valid = ~pd.isna(x_col) & ~pd.isna(y_arr)
        if valid.sum() < 2:
            result[col] = pd.DataFrame()
            continue
        if should_discretize(x_col):
            x_prep = discretize(x_col[valid], discretize_method=discretize_method,
                                n_strata=n_strata)
        else:
            x_prep = all2int(x_col[valid])
        yv = y_arr[valid]
        means = {}
        for stratum in np.unique(x_prep):
            mask = x_prep == stratum
            if mask.sum() > 0:
                means[int(stratum)] = yv[mask].mean()
        df = pd.DataFrame({"stratum": list(means.keys()),
                           "mean": list(means.values())})
        result[col] = df
    return result
