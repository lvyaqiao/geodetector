"""Verify RGD fix: compare old (X-signal) vs new (Y-sorted-by-X-signal)
discretization on synthetic data with known Y breakpoints.

Scenario 1: X has density gaps at DIFFERENT locations than Y breaks.
Scenario 2: X is uniform (no gaps) — Y-signal method clearly dominates.

Usage:
    python scripts/verify_rgd_fix.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib.pyplot as plt

from geodetector.extensions._rgd import robust_discretize

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def old_discretize(x, k):
    """Replicate pre-fix logic: normalised X as change-point signal."""
    x = np.asarray(x, dtype=float).ravel()
    valid = ~np.isnan(x)
    x_clean = x[valid]
    n = len(x_clean)
    n_k = min(k, len(np.unique(x_clean)))
    sort_idx = np.argsort(x_clean)
    x_sorted = x_clean[sort_idx]
    signal = (x_sorted - x_sorted.mean()) / (x_sorted.std() + 1e-10)
    import ruptures as rpt
    algo = rpt.Dynp(model="l2", min_size=max(2, 1))
    bkps = algo.fit(signal).predict(n_bkps=n_k - 1)
    bp_vals = x_sorted[np.array(bkps[:-1]) - 1]
    labels = np.full(len(x), -1, dtype=int)
    bins = np.array([-np.inf] + list(bp_vals) + [np.inf])
    labels[valid] = np.digitize(x_clean, bins[1:-1], right=True)
    return labels, bp_vals


def bp_from_labels(x, labels):
    bp = []
    for g in sorted(set(labels) - {-1}):
        bp.append(x[labels == g].max())
    return np.array(bp[:-1])


# ── Scenario 1: X clustered, Y breaks inside clusters ─────────────────────
np.random.seed(42)
N = 300
c1, c2, c3 = (np.random.normal(10, 5, N // 3),
              np.random.normal(50, 5, N // 3),
              np.random.normal(90, 5, N // 3))
X1 = np.sort(np.concatenate([c1, c2, c3]))
noise1 = np.random.normal(0, 0.3, N)
Y1 = np.where(X1 < 25, 2.0, np.where(X1 < 65, 7.0, 12.0)) + noise1
TRUE1, GAPS1 = [25, 65], [30, 70]

old1_l, old1_bp = old_discretize(X1, k=3)
new1_l = robust_discretize(X1, k=3, y=Y1, random_state=42)
new1_bp = bp_from_labels(X1, new1_l)
err_old1 = np.mean([min(abs(b - t) for b in old1_bp) for t in TRUE1])
err_new1 = np.mean([min(abs(b - t) for b in new1_bp) for t in TRUE1])

# ── Scenario 2: X uniform, single Y break — Y-signal dominates ────────────
np.random.seed(123)
N2 = 200
X2 = np.sort(np.random.uniform(0, 100, N2))
noise2 = np.random.normal(0, 0.2, N2)
Y2 = np.where(X2 < 40, 3.0, 9.0) + noise2
TRUE2 = [40]

old2_l, old2_bp = old_discretize(X2, k=2)
new2_l = robust_discretize(X2, k=2, y=Y2, random_state=42)
new2_bp = bp_from_labels(X2, new2_l)
err_old2 = abs(old2_bp[0] - TRUE2[0])
err_new2 = abs(new2_bp[0] - TRUE2[0])

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("RGD Fix Verification — OLD (X-signal) vs NEW (Y-signal) Discretization",
             fontsize=14, fontweight="bold")

# Row 1 — Scenario 1
# (A) raw data
ax = axes[0, 0]
ax.scatter(X1, Y1, c="0.4", s=6, alpha=0.5, zorder=2)
for b in TRUE1:
    ax.axvline(b, color="red", ls="-", lw=2, label="true break" if b == TRUE1[0] else "")
for g in GAPS1:
    ax.axvline(g, color="orange", ls=":", lw=1.5, label="X-density gap" if g == GAPS1[0] else "")
ax.set_title("(A) Scenario 1: tri-modal X, Y breaks at 25, 65")
ax.set_xlabel("X"); ax.set_ylabel("Y")
handles, labels = ax.get_legend_handles_labels()
ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), fontsize=7)

# (B) OLD
ax = axes[0, 1]
for g in sorted(set(old1_l) - {-1}):
    ax.scatter(X1[old1_l == g], Y1[old1_l == g], s=6, alpha=0.7, label=f"zone {g}")
for b in old1_bp:
    ax.axvline(b, color="k", lw=2)
ax.set_title(f"(B) OLD (X-signal)  |  bp = {np.round(old1_bp,1)}  |  err = {err_old1:.1f}")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.legend(markerscale=2, fontsize=7)

# (C) NEW
ax = axes[0, 2]
for g in sorted(set(new1_l) - {-1}):
    ax.scatter(X1[new1_l == g], Y1[new1_l == g], s=6, alpha=0.7, label=f"zone {g}")
for b in new1_bp:
    ax.axvline(b, color="k", lw=2)
ax.set_title(f"(C) NEW (Y-signal)  |  bp = {np.round(new1_bp,1)}  |  err = {err_new1:.1f}")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.legend(markerscale=2, fontsize=7)

# Row 2 — Scenario 2
ax = axes[1, 0]
ax.scatter(X2, Y2, c="0.4", s=6, alpha=0.5, zorder=2)
ax.axvline(TRUE2[0], color="red", ls="-", lw=2, label=f"true break X={TRUE2[0]}")
ax.set_title("(D) Scenario 2: uniform X, Y breaks at 40")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.legend(fontsize=7)

ax = axes[1, 1]
for g in sorted(set(old2_l) - {-1}):
    ax.scatter(X2[old2_l == g], Y2[old2_l == g], s=6, alpha=0.7, label=f"zone {g}")
ax.axvline(old2_bp[0], color="k", lw=2)
ax.set_title(f"(E) OLD (X-signal)  |  bp = {old2_bp[0]:.1f}  |  err = {err_old2:.1f}")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.legend(markerscale=2, fontsize=7)

ax = axes[1, 2]
for g in sorted(set(new2_l) - {-1}):
    ax.scatter(X2[new2_l == g], Y2[new2_l == g], s=6, alpha=0.7, label=f"zone {g}")
ax.axvline(new2_bp[0], color="k", lw=2)
ax.set_title(f"(F) NEW (Y-signal)  |  bp = {new2_bp[0]:.1f}  |  err = {err_new2:.1f}")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.legend(markerscale=2, fontsize=7)

plt.tight_layout()
out = OUTPUT_DIR / "rgd_verification.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Summary ────────────────────────────────────────────────────────────────
print("=" * 62)
print("  RGD Fix Verification — Final Results")
print("=" * 62)
print()
print(" Scenario 1 (tri-modal X, 3 strata):")
print(f"   True Y breaks:    {TRUE1}")
print(f"   OLD (X-signal):   {np.round(old1_bp, 1)}  |  mean error = {err_old1:.1f}")
print(f"   NEW (Y-signal):   {np.round(new1_bp, 1)}  |  mean error = {err_new1:.1f}")
print()
print(" Scenario 2 (uniform X, 2 strata):")
print(f"   True Y break:     {TRUE2}")
print(f"   OLD (X-signal):   {old2_bp[0]:.1f}  |  error = {err_old2:.1f}")
print(f"   NEW (Y-signal):   {new2_bp[0]:.1f}  |  error = {err_new2:.1f}")
print()
print(f" Saved plot -> {out}")
