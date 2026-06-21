"""Generate example plots for all geodetector visualization functions.

Usage:
    python scripts/plot_demo.py

Outputs to tests/output/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geodetector import GeoDetector
from geodetector.dataset import load_disease
from geodetector import plotting
from geodetector.extensions import (
    LESH, OPGD, GOZH, RGD,
    shapley_decompose, robust_discretize,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Real data (disease dataset) ─────────────────────────────────────────────
df = load_disease()
print(f"Disease dataset: {df.shape[0]} rows x {df.shape[1]} cols")
print(df.head(), "\n")

gd = GeoDetector(factors=["type", "region", "level"], target="incidence",
                 discretize_method="quantile", n_strata=5, alpha=0.05)
gd.fit(df)

# ═══════════════════════════════════════════════════════════════
# 1. Factor plot (gdverse style)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 3))
plotting.plot_factor(gd.q_values_, ax=ax)
ax.set_title("Factor Detector (gdverse style)", fontfamily="serif", fontweight="bold")
fig.savefig(OUT_DIR / "demo_1_factor.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: demo_1_factor.png")

# ═══════════════════════════════════════════════════════════════
# 2. Interaction — heatmap + bubble
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5, 4.5))
plotting.plot_interaction(gd.interaction_q_, gd.interaction_type_,
                          style="heatmap", ax=ax)
fig.savefig(OUT_DIR / "demo_2_interaction_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: demo_2_interaction_heatmap.png")

fig, ax = plt.subplots(figsize=(6, 5.5))
plotting.plot_interaction(gd.interaction_q_, gd.interaction_type_,
                          style="bubble", ax=ax)
fig.savefig(OUT_DIR / "demo_2_interaction_bubble.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: demo_2_interaction_bubble.png")

# ═══════════════════════════════════════════════════════════════
# 3. Risk detector
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(4, 3.5))
plotting.plot_risk(gd.risk_result_, factor="type", ax=ax)
fig.savefig(OUT_DIR / "demo_3_risk.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: demo_3_risk.png")

# ═══════════════════════════════════════════════════════════════
# 4. Ecological detector
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(4, 3.5))
plotting.plot_ecological(gd.ecological_result_, ax=ax)
fig.savefig(OUT_DIR / "demo_4_ecological.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: demo_4_ecological.png")

# ═══════════════════════════════════════════════════════════════
# 5. Risk mean plot (NEW)
# ═══════════════════════════════════════════════════════════════
means = plotting.compute_risk_means(
    df[["type", "region", "level"]], df["incidence"],
    discretize_method="quantile", n_strata=5,
)
fig = plotting.plot_risk_mean(means, figsize=(12, 4))
fig.savefig(OUT_DIR / "demo_5_risk_mean.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: demo_5_risk_mean.png")

# ═══════════════════════════════════════════════════════════════
# 6. LESH — Shapley pie chart (NEW)
# ═══════════════════════════════════════════════════════════════
lesh = LESH(factors=["type", "region", "level"], target="incidence",
            method="quantile", discretize_method="quantile", n_strata=5,
            random_state=42, progress=False)
lesh.fit(df)
print(f"\nLESH fitted — Shapley values:")
for _, row in lesh.shapley_.iterrows():
    print(f"  {row['variable']:<10s}  theta={row['shapley_value']:.4f}  ({row['shapley_pct']*100:.1f}%)")

fig, ax = plt.subplots(figsize=(7, 6))
plotting.plot_lesh(lesh, ax=ax)
fig.savefig(OUT_DIR / "demo_6_lesh_pie.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: demo_6_lesh_pie.png")

# ═══════════════════════════════════════════════════════════════
# 7. Optimal discretization process (NEW)
#    disease data has only categorical factors → create a synthetic
#    continuous dataset to demonstrate the q-k curves + histogram
# ═══════════════════════════════════════════════════════════════
np.random.seed(42)
N_syn = 300
X1 = np.random.uniform(0, 100, N_syn)
X2 = np.random.gamma(2, 10, N_syn)
syn_y = np.where(X1 < 40, 3.0, np.where(X1 < 70, 7.0, 12.0)) + np.random.normal(0, 0.5, N_syn)
df_syn = pd.DataFrame({"x_linear": X1, "x_skewed": X2, "y": syn_y})

fig = plotting.plot_optimal_discretization(
    df_syn, ["x_linear", "x_skewed"], "y",
    methods=["sd", "equal", "geometric", "quantile", "natural"],
    k_range=(3, 8),
    figsize=(12, 5),
)
fig.savefig(OUT_DIR / "demo_7_discretization.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: demo_7_discretization.png")

# ═══════════════════════════════════════════════════════════════
# 8. Dashboard
# ═══════════════════════════════════════════════════════════════
fig = gd.plot_dashboard()
fig.savefig(OUT_DIR / "demo_8_dashboard.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: demo_8_dashboard.png")

# ═══════════════════════════════════════════════════════════════
# 9. OPGD plot
# ═══════════════════════════════════════════════════════════════
opgd = OPGD(factors=["type", "region", "level"], target="incidence",
            k_range=(3, 8), random_state=42)
opgd.fit(df)

fig, ax = plt.subplots(figsize=(6, 3))
opgd.plot(ax=ax)
ax.set_title("OPGD — Factor Detector", fontfamily="serif", fontweight="bold")
fig.savefig(OUT_DIR / "demo_9_opgd.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: demo_9_opgd.png")

# ═══════════════════════════════════════════════════════════════
# 10. RGD plot
# ═══════════════════════════════════════════════════════════════
rgd = RGD(factors=["type", "region", "level"], target="incidence",
          strategy=2, increase_rate=0.05, random_state=42)
rgd.fit(df)

fig, ax = plt.subplots(figsize=(6, 3))
rgd.plot(ax=ax)
ax.set_title("RGD — Factor Detector", fontfamily="serif", fontweight="bold")
fig.savefig(OUT_DIR / "demo_10_rgd.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: demo_10_rgd.png")

# ═══════════════════════════════════════════════════════════════
# 11. GOZH plot (with sklearn)
# ═══════════════════════════════════════════════════════════════
try:
    gozh = GOZH(factors=["type", "region", "level"], target="incidence",
                max_depth=4, min_samples_leaf=5, random_state=42)
    gozh.fit(df)

    fig, ax = plt.subplots(figsize=(6, 3))
    gozh.plot(ax=ax)
    ax.set_title("GOZH — Factor Detector", fontfamily="serif", fontweight="bold")
    fig.savefig(OUT_DIR / "demo_11_gozh.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: demo_11_gozh.png")
except ImportError:
    print("Skipped demo_11 (no sklearn)")

print("\n" + "=" * 60)
print("  All plots generated in tests/output/")
print("=" * 60)
