"""Shared fixtures for geodetector tests."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Deterministic fixtures for reproducible tests
# ──────────────────────────────────────────────────────────────────────

SEED = 42
rng = np.random.default_rng(SEED)


@pytest.fixture(scope="function")
def df_simple():
    """Two clear groups with different means.

    Group 0: y ~ N(-2, 0.5), Group 1: y ~ N(+2, 0.5)
    Strong explanatory power expected (q >> 0.5).
    """
    n = 100
    x = np.array([0] * (n // 2) + [1] * (n // 2))
    y = np.concatenate([
        rng.normal(-2, 0.5, n // 2),
        rng.normal(2, 0.5, n // 2),
    ])
    return pd.DataFrame({"group": x, "y": y})


@pytest.fixture(scope="function")
def df_multifactor():
    """Three factors: two categorical + one continuous.

    type:   0, 1, 2        (categorical, 3 levels)
    region: 0, 1, 2, 3     (categorical, 4 levels)
    elev:   continuous      (needs discretization)

    y has a structured dependence on type and region.
    """
    n = 120
    df = pd.DataFrame({
        "type":   rng.choice([0, 1, 2], n),
        "region": rng.choice([0, 1, 2, 3], n),
        "elev":   rng.normal(100, 30, n),
    })
    # y depends on type (strong) and region (weak), noise added
    y = (df["type"] * 5.0
         + df["region"] * 1.5
         + rng.normal(0, 2.0, n))
    df["y"] = y
    return df


@pytest.fixture(scope="function")
def df_no_variance():
    """Y has zero variance (all identical)."""
    return pd.DataFrame({
        "group": [0, 0, 1, 1, 2, 2],
        "y":     [5.0] * 6,
    })


@pytest.fixture(scope="function")
def df_single_group():
    """All X values are identical (single stratum)."""
    return pd.DataFrame({
        "group": [1] * 50,
        "y":     rng.normal(0, 1, 50),
    })


@pytest.fixture(scope="function")
def df_continuous_only():
    """X are all continuous — must be discretized before use."""
    n = 100
    return pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(50, 15, n),
        "y":  rng.normal(0, 1, n),
    })


@pytest.fixture(scope="function")
def df_with_na():
    """Data with some NaN entries."""
    n = 50
    y = rng.normal(0, 1, n)
    x = rng.choice([0, 1, 2], n).astype(float)
    x[:3] = np.nan
    y[40:] = np.nan
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture(scope="function")
def df_disease():
    """Real built-in dataset (185 rows × 4 columns).

    Columns: incidence (float), type, region, level (all int-coded strata).
    """
    return pd.read_csv(
        Path(__file__).parent.parent / "src" / "geodetector" / "dataset" / "data" / "disease.csv"
    )


@pytest.fixture(scope="function")
def df_large():
    """Larger synthetic dataset for performance edge cases (2000 rows)."""
    n = 2000
    df = pd.DataFrame({
        "f1": rng.choice([0, 1, 2, 3, 4], n),
        "f2": rng.choice([0, 1, 2], n),
        "f3": rng.normal(10, 3, n),
    })
    df["y"] = (df["f1"] * 2.0 + df["f2"] * 1.0 + rng.normal(0, 1.5, n))
    return df


@pytest.fixture(scope="function")
def df_perfect_q():
    """Perfect stratification: each stratum has identical Y values.

    y = f(x) exactly, so q should be 1.0.
    """
    df = pd.DataFrame({
        "x": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
        "y": [10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 30.0],
    })
    return df
