"""Tests for core statistical functions."""

import numpy as np
import pytest


class TestQStatistic:
    """Tests for q_statistic() — the core q-value computation."""

    def test_q_between_0_and_1(self):
        """q should always fall in [0, 1]."""
        from geodetector._stats import q_statistic

        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        x = np.array([0, 0, 0, 1, 1, 1])
        q = q_statistic(y, x)
        assert 0.0 <= q <= 1.0

    def test_q_perfect_stratification_is_1(self):
        """When every stratum has identical Y values, q == 1."""
        from geodetector._stats import q_statistic

        y = np.array([10.0, 10.0, 10.0, 20.0, 20.0, 20.0])
        x = np.array([0, 0, 0, 1, 1, 1])
        q = q_statistic(y, x)
        assert q == pytest.approx(1.0)

    def test_q_random_stratification_is_near_0(self):
        """When Y and X are unrelated, q should be close to 0."""
        from geodetector._stats import q_statistic

        rng = np.random.default_rng(42)
        y = rng.normal(0, 10, 1000)
        x = rng.choice([0, 1, 2, 3, 4], 1000)
        q = q_statistic(y, x)
        assert q < 0.05

    def test_q_equals_r_squared(self):
        """q_statistic should be identical to R² of stratum-mean predictor."""
        from geodetector._stats import q_statistic

        y = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
        x = np.array([0, 0, 0, 1, 1, 1])

        q = q_statistic(y, x)
        # Manual R² computation
        y_pred = np.array([np.mean(y[x == 0])] * 3 + [np.mean(y[x == 1])] * 3)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - sse / sst

        assert q == pytest.approx(r2)

    def test_q_single_group(self):
        """Single stratum → q should be NaN or 0 since no stratification."""
        from geodetector._stats import q_statistic

        y = np.array([1.0, 2.0, 3.0])
        x = np.array([0, 0, 0])
        q = q_statistic(y, x)
        # Single group cannot stratify; q should be 0 (no explanatory power)
        assert q == 0.0 or np.isnan(q)

    def test_q_only_ones_in_any_group_handled(self):
        """Strata with 1 observation are excluded."""
        from geodetector._stats import q_statistic

        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = np.array([0, 0, 0, 1, 2])  # strata 1 and 2 have only 1 obs each
        q = q_statistic(y, x)
        # After removing single-obs strata, only one group remains → q == 0
        assert q == 0.0


class TestNoncentralFTest:
    """Tests for the non-central F-test that computes p-values for q."""

    def test_significant_when_strong_effect(self):
        """A strong genuine effect should yield p < 0.05."""
        from geodetector._stats import q_significance_test

        rng = np.random.default_rng(42)
        n = 200
        y0 = rng.normal(0, 1, n // 2)
        y1 = rng.normal(5, 1, n // 2)
        y = np.concatenate([y0, y1])
        x = np.array([0] * (n // 2) + [1] * (n // 2))

        p_val = q_significance_test(y, x)
        assert p_val < 0.05

    def test_not_significant_when_no_effect(self):
        """No genuine effect should yield p > 0.05."""
        from geodetector._stats import q_significance_test

        rng = np.random.default_rng(42)
        n = 200
        y = rng.normal(0, 1, n)
        x = rng.choice([0, 1, 2], n)

        p_val = q_significance_test(y, x)
        assert p_val > 0.05


class TestInteractionType:
    """Tests for the interaction type classification."""

    def test_five_interaction_types_cover_all_cases(self):
        """The 5 interaction types should be distinguishable for explicit data."""
        from geodetector._stats import interaction_type

        # Create distinct cases
        assert interaction_type(0.2, 0.3, 0.1) == 0  # weaken nonlinear
        assert interaction_type(0.2, 0.5, 0.3) == 1  # weaken uni-
        assert interaction_type(0.2, 0.3, 0.4) == 2  # enhance bi-
        assert interaction_type(0.2, 0.3, 0.5) == 3  # independent
        assert interaction_type(0.2, 0.3, 0.6) == 4  # enhance nonlinear

    def test_interaction_independent(self):
        """Independent factors should yield type 3."""
        from geodetector._stats import interaction_type

        # q12 exactly equals q1 + q2 → type 3
        result = interaction_type(0.2, 0.3, 0.5)
        assert result == 3

    def test_interaction_enhance_nonlinear(self):
        """Nonlinear enhancement should yield type 4."""
        from geodetector._stats import interaction_type

        # q12 > q1 + q2 → type 4
        result = interaction_type(0.1, 0.2, 0.5)
        assert result == 4

    def test_interaction_return_type_is_int(self):
        from geodetector._stats import interaction_type

        result = interaction_type(0.1, 0.2, 0.35)
        assert isinstance(result, int)
        assert 0 <= result <= 4
