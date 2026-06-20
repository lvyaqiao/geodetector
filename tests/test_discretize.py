"""Tests for discretization module."""

import numpy as np
import pandas as pd
import pytest

from mapclassify import classify  # noqa — prerequisite check


class TestDiscretize:
    """Tests for discretize() — the single-column discretization function."""

    def test_discretize_quantile_default(self):
        from geodetector.discretize import discretize

        x = np.random.default_rng(42).normal(0, 1, 100)
        result = discretize(x, discretize_method="quantile", n_strata=5)
        assert len(result) == 100
        assert result.dtype in (np.int_, np.int32, np.int64)
        assert len(set(result)) == 5  # exactly 5 strata

    def test_discretize_equal_interval(self):
        from geodetector.discretize import discretize

        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
        result = discretize(x, discretize_method="equal", n_strata=3)
        assert len(result) == 10
        assert result.dtype in (np.int_, np.int32, np.int64)
        assert 0 <= result.min() <= 2
        assert 2 <= result.max() <= 2

    def test_discretize_jenks_natural_breaks(self):
        from geodetector.discretize import discretize

        # Clustered data — Jenks should separate clusters
        x = np.concatenate([
            np.random.default_rng(42).normal(0, 1, 50),
            np.random.default_rng(42).normal(10, 1, 50),
        ])
        result = discretize(x, discretize_method="jenks", n_strata=2)
        assert len(result) == 100
        assert len(set(result)) == 2

    def test_discretize_returns_same_for_same_input(self):
        """Deterministic: same input → same output."""
        from geodetector.discretize import discretize

        x = np.random.default_rng(42).normal(0, 1, 50)
        r1 = discretize(x, discretize_method="quantile", n_strata=5)
        r2 = discretize(x, discretize_method="quantile", n_strata=5)
        assert np.array_equal(r1, r2)

    def test_discretize_fewer_unique_than_n_strata(self):
        """If n_strata > n_unique_values, returns all unique values as strata."""
        from geodetector.discretize import discretize

        x = np.array([1.0, 2.0, 3.0] * 10)
        result = discretize(x, discretize_method="quantile", n_strata=10)
        assert len(set(result)) <= 3

    def test_discretize_invalid_method_raises(self):
        from geodetector.discretize import discretize

        x = np.random.default_rng(42).normal(0, 1, 50)
        with pytest.raises(ValueError, match="method|unknown|unsupported"):
            discretize(x, discretize_method="nonexistent", n_strata=5)

    def test_discretize_pandas_series(self):
        from geodetector.discretize import discretize

        s = pd.Series(np.random.default_rng(42).normal(0, 1, 100))
        result = discretize(s, discretize_method="quantile", n_strata=4)
        assert len(result) == 100
        assert result.dtype in (np.int_, np.int32, np.int64)


class TestCategoricalDetection:
    """Tests for is_categorical() / should_discretize() heuristics."""

    def test_low_cardinality_integer_is_categorical(self):
        """Small number of unique ints → treated as already stratified."""
        from geodetector.discretize import should_discretize

        x = np.array([0, 1, 0, 1, 0, 1] * 10)
        assert not should_discretize(x)

    def test_many_unique_values_is_continuous(self):
        """Many unique values → needs discretization."""
        from geodetector.discretize import should_discretize

        x = np.random.default_rng(42).normal(0, 1, 200)
        assert should_discretize(x)

    def test_object_dtype_never_discretized(self):
        """String / categorical columns are not discretized."""
        from geodetector.discretize import should_discretize

        x = pd.Series(["a", "b", "c", "a", "b"] * 10)
        assert not should_discretize(x)
