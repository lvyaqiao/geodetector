"""Plotting functions for geodetector results.

All functions are pure: they accept data and return matplotlib Axes.
The GeoDetector class methods are thin wrappers around these.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


# ── Colour palette ──────────────────────────────────────────

# Interaction type colours (matching GD package convention)
INTERACTION_COLORS = {
    0: ("Weaken, nonlinear",     "#6EE9EF"),
    1: ("Weaken, uni-variable",  "#558DE8"),
    2: ("Enhance, bi-variable",  "#F2C55E"),
    3: ("Independent",           "#E08338"),
    4: ("Enhance, nonlinear",    "#EA4848"),
}

PRIMARY_COLOR = "#2E86AB"
INSIGNIFICANT_COLOR = "#D3D3D3"
RISK_YES = "#FFA500"
RISK_NO = "#7FDBFF"


# ── Factor detector plot ────────────────────────────────────

def plot_factor(q_values, *, ax=None, sig_level=0.05,
                figsize=None, **kwargs):
    """Horizontal bar chart of q-values.

    Parameters
    ----------
    q_values : pd.DataFrame
        Must contain columns ``variable``, ``q_value``, ``p_value``.
    ax : matplotlib Axes, optional
    sig_level : float, default=0.05
    figsize : tuple, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (8, max(3, len(q_values) * 0.5)))

    df = q_values.copy()
    df = df.sort_values("q_value", ascending=True)
    df = df.reset_index(drop=True)

    colors = [
        PRIMARY_COLOR if p < sig_level else INSIGNIFICANT_COLOR
        for p in df["p_value"]
    ]

    bars = ax.barh(range(len(df)), df["q_value"], color=colors,
                   edgecolor="black", height=0.55)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["variable"])
    ax.set_xlabel("q-value")
    ax.set_xlim(0, min(1.05, np.nanmax(df["q_value"]) * 1.3 + 0.05))

    # Annotate q-values and significance stars
    for i, (_, row) in enumerate(df.iterrows()):
        sig = ""
        if row["p_value"] < 0.001:
            sig = "  ***"
        elif row["p_value"] < 0.01:
            sig = "  **"
        elif row["p_value"] < 0.05:
            sig = "  *"
        ax.text(row["q_value"] + 0.01, i,
                f'{row["q_value"]:.3f}{sig}',
                va="center", fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PRIMARY_COLOR, label=f"p < {sig_level}"),
        Patch(facecolor=INSIGNIFICANT_COLOR, label=f"p ≥ {sig_level}"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    return ax


# ── Interaction detector plot ───────────────────────────────

def plot_interaction(interaction_q, interaction_type=None, *,
                     ax=None, style="heatmap", figsize=None, **kwargs):
    """Plot interaction matrix.

    Parameters
    ----------
    interaction_q : pd.DataFrame
        Symmetric q-value matrix.
    interaction_type : pd.DataFrame, optional
        Symmetric type matrix (0-4). Only used in "bubble" style.
    ax : matplotlib Axes, optional
    style : str, default="heatmap"
        ``"heatmap"`` for a seaborn heatmap, ``"bubble"`` for GD-style
        scatter bubbles.
    figsize : tuple, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if style == "heatmap":
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (5.5, 5))
        sns.heatmap(
            interaction_q.astype(float),
            annot=True,
            fmt=".3f",
            cmap="crest",
            vmin=0, vmax=1,
            linewidths=0.5,
            linecolor="black",
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Interaction q-values")
        plt.tight_layout()
        return ax

    elif style == "bubble":
        factors = list(interaction_q.columns)
        n = len(factors)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (n * 1.3, n * 1.3))

        q_max = max(np.nanmax(interaction_q.values), 0.01)
        for i, f1 in enumerate(factors):
            for j, f2 in enumerate(factors):
                qv = interaction_q.loc[f1, f2]
                y_pos = n - 1 - j
                if i == j:
                    ax.scatter(i, y_pos, s=qv / q_max * 600, c="gray",
                               edgecolors="black", linewidth=1, alpha=0.7)
                    ax.text(i, y_pos, f"{qv:.3f}", ha="center",
                            va="center", fontsize=8, fontweight="bold")
                elif i < j:
                    color = "gray"
                    label = ""
                    if interaction_type is not None:
                        t = int(interaction_type.loc[f1, f2])
                        if t in INTERACTION_COLORS:
                            label, color = INTERACTION_COLORS[t]
                    ax.scatter(i, y_pos, s=qv / q_max * 600, c=color,
                               edgecolors="black", linewidth=1, alpha=0.8)
                    ax.text(i, y_pos, f"{qv:.3f}", ha="center",
                            va="center", fontsize=8)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(factors)
        ax.set_yticklabels([factors[n - 1 - k] for k in range(n)])
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_title("Interaction q-values")
        plt.tight_layout()
        return ax

    else:
        raise ValueError(f"Unknown style: {style!r}. Use 'heatmap' or 'bubble'.")


# ── Risk detector plot ──────────────────────────────────────

def plot_risk(risk_result, *, factor=None, ax=None, figsize=None,
              show_labels=True, **kwargs):
    """Plot risk detector as a coloured grid matrix.

    Parameters
    ----------
    risk_result : dict
        Risk detector output (``GeoDetector.risk_result_``).
        Keys are factor names, values are DataFrames.
    factor : str, optional
        Which factor to plot. If None and only one factor exists,
        plots it. If multiple, plots all in subplots.
    ax : matplotlib Axes, optional
        Only used when ``factor`` is specified.
    figsize : tuple, optional
    show_labels : bool, default=True
        Whether to show "Y"/"N" labels on cells.

    Returns
    -------
    matplotlib Axes or dict of Axes
    """
    if factor is not None:
        return _plot_risk_one(risk_result[factor], ax=ax,
                              show_labels=show_labels, **kwargs)

    factors = list(risk_result.keys())
    valid = [f for f in factors
             if risk_result[f] is not None and not risk_result[f].empty]
    if not valid:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No significant comparisons", ha="center", va="center")
        return ax

    if len(valid) == 1:
        return _plot_risk_one(risk_result[valid[0]], ax=ax,
                              show_labels=show_labels, **kwargs)

    # Multi-factor: grid of subplots
    n = len(valid)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize or (ncols * 4, nrows * 3.5))
    axes = np.atleast_1d(axes).ravel()
    for i, f in enumerate(valid):
        _plot_risk_one(risk_result[f], ax=axes[i],
                       show_labels=show_labels, title=f, **kwargs)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    return dict(zip(valid, axes[:len(valid)]))


def _plot_risk_one(df, *, ax=None, show_labels=True, title=None):
    """Plot a single risk detector matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3.5))

    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    # Build matrix
    strata = sorted(set(df["stratum_1"]) | set(df["stratum_2"]))
    n = len(strata)
    if n < 2:
        ax.text(0.5, 0.5, "Single stratum", ha="center", va="center")
        return ax

    # Map stratum value → position
    idx_map = {s: i for i, s in enumerate(strata)}
    matrix = np.full((n, n), np.nan)

    for _, row in df.iterrows():
        i = idx_map[row["stratum_1"]]
        j = idx_map[row["stratum_2"]]
        val = 1 if row["significant"] else 0
        matrix[i, j] = val
        matrix[j, i] = val

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([RISK_NO, RISK_YES])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    if show_labels and not np.all(np.isnan(matrix)):
        for i in range(n):
            for j in range(n):
                if not np.isnan(matrix[i, j]) and i != j:
                    label = "Y" if matrix[i, j] == 1 else "N"
                    ax.text(j, i, label, ha="center", va="center",
                            fontsize=9, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(strata)
    ax.set_yticklabels(strata)
    if title:
        ax.set_title(title)
    return ax


# ── Ecological detector plot ────────────────────────────────

def plot_ecological(eco_result, *, ax=None, figsize=None,
                    show_labels=True, **kwargs):
    """Plot ecological detector as a coloured grid matrix.

    Parameters
    ----------
    eco_result : pd.DataFrame
        Ecological detector output (``GeoDetector.ecological_result_``).
    ax : matplotlib Axes, optional
    figsize : tuple, optional
    show_labels : bool, default=True

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (4, 3.5))

    if eco_result is None or eco_result.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    # Extract unique factor names
    factors = sorted(set(eco_result["factor_1"]) | set(eco_result["factor_2"]))
    n = len(factors)
    if n < 2:
        ax.text(0.5, 0.5, "Single factor", ha="center", va="center")
        return ax

    # Build matrix
    idx_map = {f: i for i, f in enumerate(factors)}
    matrix = np.full((n, n), np.nan)

    for _, row in eco_result.iterrows():
        i = idx_map[row["factor_1"]]
        j = idx_map[row["factor_2"]]
        val = 1 if row["significant"] else 0
        matrix[i, j] = val
        matrix[j, i] = val

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([RISK_NO, RISK_YES])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    if show_labels and not np.all(np.isnan(matrix)):
        for i in range(n):
            for j in range(n):
                if not np.isnan(matrix[i, j]) and i != j:
                    label = "Y" if matrix[i, j] == 1 else "N"
                    ax.text(j, i, label, ha="center", va="center",
                            fontsize=10, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(factors)
    ax.set_yticklabels(factors)
    ax.set_title("Ecological Detector")
    return ax


# ── Dashboard ───────────────────────────────────────────────

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
    # Panel 1: Factor detector
    plot_factor(gd.q_values_, ax=axes[idx], sig_level=gd.alpha)
    idx += 1

    # Panel 2: Interaction
    if has_interaction:
        plot_interaction(gd.interaction_q_, gd.interaction_type_,
                         ax=axes[idx], style="heatmap")
        idx += 1

    # Panel 3: Risk detector (first factor)
    if gd.risk_result_:
        first_factor = list(gd.risk_result_.keys())[0]
        plot_risk(gd.risk_result_, factor=first_factor, ax=axes[idx])
    idx += 1

    # Panel 4: Ecological
    if has_ecological:
        plot_ecological(gd.ecological_result_, ax=axes[idx])

    # Hide unused panels
    for j in range(idx, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig
