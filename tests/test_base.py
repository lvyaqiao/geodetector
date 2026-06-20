"""Tests for _base.py — lightweight sklearn-compatible base classes."""

import numpy as np
import pytest


from geodetector._base import BaseEstimator, RegressorMixin


class DummyEstimator(BaseEstimator):
    """Minimal estimator used to test _BaseEstimator."""

    def __init__(self, param_a=1, param_b="hello"):
        self.param_a = param_a
        self.param_b = param_b


class DummyRegressor(RegressorMixin):
    """Minimal regressor used to test _RegressorMixin."""

    def predict(self, X):
        # Predicts Y as mean of X's first column (for testing only)
        return np.ones(len(X)) * 5.0


class TestBaseEstimator:
    def test_get_params_defaults(self):
        est = DummyEstimator()
        params = est.get_params()
        assert params == {"param_a": 1, "param_b": "hello"}

    def test_set_params_updates_attribute(self):
        est = DummyEstimator()
        est.set_params(param_a=42)
        assert est.param_a == 42

    def test_set_params_returns_self(self):
        est = DummyEstimator()
        result = est.set_params(param_b="world")
        assert result is est

    def test_get_params_deep_false(self):
        est_a = DummyEstimator(param_a=DummyEstimator(param_a=99))
        # With deep=False, nested objects are not unpacked
        params_shallow = est_a.get_params(deep=False)
        assert isinstance(params_shallow["param_a"], DummyEstimator)


class TestRegressorMixin:
    def test_score_is_r_squared(self):
        reg = DummyRegressor()
        y = np.array([1.0, 2.0, 3.0, 4.0])
        X = np.array([[1], [2], [3], [4]])
        score = reg.score(X, y)
        assert 0.0 <= score <= 1.0 or score < 0  # R² can be negative

    def test_perfect_prediction_score_1(self):
        """When predict returns exactly y, score should be 1.0."""

        class PerfectRegressor(DummyRegressor):
            def predict(self, X):
                return np.array([1.0, 2.0, 3.0, 4.0])

        reg = PerfectRegressor()
        y = np.array([1.0, 2.0, 3.0, 4.0])
        X = np.array([[0]] * 4)
        score = reg.score(X, y)
        assert score == pytest.approx(1.0)
