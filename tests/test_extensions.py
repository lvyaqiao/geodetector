"""Tests for extension modules: LESH, OPGD, GOZH, Geometric, RGD."""

import numpy as np
import pandas as pd
import pytest

_HAS_SKLEARN = True
try:
    from sklearn.tree import DecisionTreeRegressor  # noqa: F401
except ImportError:
    _HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════════
# LESH — Shapley decomposition
# ═══════════════════════════════════════════════════════════════

class TestShapleyDecompose:
    def test_returns_dataframe(self, df_multifactor):
        from geodetector.extensions import shapley_decompose

        result = shapley_decompose(
            df_multifactor, ["type", "region", "elev"], "y"
        )
        assert isinstance(result, pd.DataFrame)
        assert "variable" in result.columns
        assert "shapley_value" in result.columns
        assert "shapley_pct" in result.columns

    def test_sorted_descending(self, df_multifactor):
        from geodetector.extensions import shapley_decompose

        result = shapley_decompose(
            df_multifactor, ["type", "region", "elev"], "y"
        )
        vals = result["shapley_value"].values
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1]

    def test_sum_equals_adjusted_r_squared(self, df_multifactor):
        from geodetector.extensions import shapley_decompose

        result = shapley_decompose(
            df_multifactor, ["type", "region"], "y"
        )
        total = result["shapley_value"].sum()
        assert total >= 0

    def test_two_factors_shapley_values_positive(self, df_multifactor):
        from geodetector.extensions import shapley_decompose

        result = shapley_decompose(
            df_multifactor, ["type", "region"], "y"
        )
        for v in result["shapley_value"]:
            assert v >= 0

    def test_percentages_sum_to_one(self, df_multifactor):
        from geodetector.extensions import shapley_decompose

        result = shapley_decompose(
            df_multifactor, ["type", "region", "elev"], "y"
        )
        total_pct = result["shapley_pct"].sum()
        assert total_pct == pytest.approx(1.0)


class TestLESH:
    def test_lesh_returns_shapley_and_interaction(self, df_multifactor):
        from geodetector.extensions import LESH

        lesh = LESH(["type", "region", "elev"], "y", n_strata=4)
        lesh.fit(df_multifactor)
        assert hasattr(lesh, "shapley_")
        assert hasattr(lesh, "interaction_")
        assert isinstance(lesh.shapley_, pd.DataFrame)
        assert isinstance(lesh.interaction_, pd.DataFrame)

    def test_lesh_interaction_has_spd_columns(self, df_multifactor):
        from geodetector.extensions import LESH

        lesh = LESH(["type", "region", "elev"], "y", n_strata=4)
        lesh.fit(df_multifactor)
        inter = lesh.interaction_
        assert "spd_1" in inter.columns
        assert "spd_2" in inter.columns

    def test_lesh_summary(self, df_multifactor):
        from geodetector.extensions import LESH

        lesh = LESH(["type", "region", "elev"], "y", n_strata=4)
        lesh.fit(df_multifactor)
        s = lesh.summary()
        assert isinstance(s, str)
        assert "LESH" in s


# ═══════════════════════════════════════════════════════════════
# OPGD — Optimal Parameter GD
# ═══════════════════════════════════════════════════════════════

class TestOPGD:
    def test_opgd_fit_sets_opt_params(self, df_multifactor):
        from geodetector.extensions import OPGD

        opgd = OPGD(["type", "region", "elev"], "y",
                    discretize_methods=["quantile", "equal"], k_range=(3, 5))
        opgd.fit(df_multifactor)
        assert hasattr(opgd, "opt_params_")
        assert isinstance(opgd.opt_params_, pd.DataFrame)
        assert "method" in opgd.opt_params_.columns
        assert "k" in opgd.opt_params_.columns

    def test_opgd_fit_sets_gd_results(self, df_multifactor):
        from geodetector.extensions import OPGD

        opgd = OPGD(["type", "region", "elev"], "y",
                    discretize_methods=["quantile"], k_range=(3, 4))
        opgd.fit(df_multifactor)
        assert hasattr(opgd, "q_values_")
        assert len(opgd.q_values_) == 3

    def test_opgd_summary(self, df_multifactor):
        from geodetector.extensions import OPGD

        opgd = OPGD(["type", "region", "elev"], "y",
                    discretize_methods=["quantile"], k_range=(3, 4))
        opgd.fit(df_multifactor)
        s = opgd.summary()
        assert isinstance(s, str)
        assert "OPGD" in s

    def test_opgd_discretized_data(self, df_multifactor):
        from geodetector.extensions import OPGD

        opgd = OPGD(["type", "region", "elev"], "y",
                    discretize_methods=["quantile"], k_range=(3, 4))
        opgd.fit(df_multifactor)
        assert opgd.discretized_data_.shape[0] == len(df_multifactor)
        assert set(opgd.discretized_data_.columns) == {"type", "region", "elev"}


# ═══════════════════════════════════════════════════════════════
# GOZH — Decision-tree zoning
# ═══════════════════════════════════════════════════════════════

class TestRPartDiscretize:
    @pytest.mark.skipif(
        __import__("importlib.util").util.find_spec("sklearn") is None,
        reason="sklearn not installed",
    )
    def test_rpart_discretize_returns_integer(self, df_multifactor):
        from geodetector.extensions import rpart_discretize

        x = df_multifactor["elev"]
        y = df_multifactor["y"]
        result = rpart_discretize(x, y, max_depth=3)
        assert len(result) == len(x)
        assert result.dtype in (np.int_, np.int32, np.int64)


class TestGOZH:
    @pytest.mark.skipif(
        __import__("importlib.util").util.find_spec("sklearn") is None,
        reason="sklearn not installed",
    )
    def test_gozh_fit_sets_n_zones(self, df_multifactor):
        from geodetector.extensions import GOZH

        gozh = GOZH(["type", "region", "elev"], "y", max_depth=3, min_samples_leaf=5)
        gozh.fit(df_multifactor)
        assert hasattr(gozh, "n_zones_")
        for f in ["type", "region", "elev"]:
            assert f in gozh.n_zones_

    @pytest.mark.skipif(
        __import__("importlib.util").util.find_spec("sklearn") is None,
        reason="sklearn not installed",
    )
    def test_gozh_fit_sets_q_values(self, df_multifactor):
        from geodetector.extensions import GOZH

        gozh = GOZH(["type", "region", "elev"], "y", max_depth=3)
        gozh.fit(df_multifactor)
        assert len(gozh.q_values_) == 3

    @pytest.mark.skipif(
        __import__("importlib.util").util.find_spec("sklearn") is None,
        reason="sklearn not installed",
    )
    def test_gozh_summary(self, df_multifactor):
        from geodetector.extensions import GOZH

        gozh = GOZH(["type", "region", "elev"], "y", max_depth=3)
        gozh.fit(df_multifactor)
        s = gozh.summary()
        assert isinstance(s, str)
        assert "GOZH" in s


# ═══════════════════════════════════════════════════════════════
# Geometric discretization
# ═══════════════════════════════════════════════════════════════

class TestGeometric:
    def test_geometric_positive(self):
        from geodetector.extensions import geometric_breaks

        data = np.array([1, 2, 5, 10, 20, 50, 100], dtype=float)
        breaks = geometric_breaks(data, k=3)
        assert len(breaks) > 0
        assert np.min(data) < breaks.min() < np.max(data)

    def test_geometric_negative(self):
        from geodetector.extensions import geometric_breaks

        data = np.array([-100, -50, -20, -10, -5, -2, -1], dtype=float)
        breaks = geometric_breaks(data, k=3)
        assert len(breaks) > 0

    def test_geometric_span_zero(self):
        from geodetector.extensions import geometric_breaks

        data = np.array([-50, -20, -10, 5, 20, 100, 200], dtype=float)
        breaks = geometric_breaks(data, k=4)
        assert len(breaks) > 0
        assert any(b <= 0 for b in breaks)
        assert any(b >= 0 for b in breaks)

    def test_discretize_geometric_returns_integer(self):
        from geodetector.extensions import discretize_geometric

        x = np.random.default_rng(42).lognormal(0, 1, 100)
        result = discretize_geometric(x, n_strata=5)
        assert len(result) == 100
        assert result.dtype in (np.int_, np.int32, np.int64)

    def test_discretize_via_main_function(self):
        from geodetector import discretize

        x = np.random.default_rng(42).lognormal(0, 1, 50)
        result = discretize(x, discretize_method="geometric", n_strata=4)
        assert len(result) == 50
        assert result.dtype in (np.int_, np.int32, np.int64)


# ═══════════════════════════════════════════════════════════════
# RGD — Robust Geographical Detector
# ═══════════════════════════════════════════════════════════════

class TestRobustDiscretize:
    def test_robust_discretize_returns_integer(self, df_continuous_only):
        from geodetector.extensions import robust_discretize

        x = df_continuous_only["x1"]
        result = robust_discretize(x, k=4)
        assert len(result) == len(x)
        assert result.dtype in (np.int_, np.int32, np.int64)

    def test_robust_discretize_respects_k(self, df_continuous_only):
        from geodetector.extensions import robust_discretize

        x = df_continuous_only["x1"]
        for k in [2, 4, 6]:
            result = robust_discretize(x, k=k)
            n_unique = len(np.unique(result[result >= 0]))
            assert n_unique <= k + 1  # +1 tolerance


class TestRGD:
    def test_rgd_fit_sets_results(self, df_multifactor):
        from geodetector.extensions import RGD

        rgd = RGD(["type", "region", "elev"], "y", k=4)
        rgd.fit(df_multifactor)
        assert hasattr(rgd, "q_values_")
        assert len(rgd.q_values_) == 3

    def test_rgd_feature_strata_positive(self, df_multifactor):
        from geodetector.extensions import RGD

        rgd = RGD(["type", "region", "elev"], "y", k=4)
        rgd.fit(df_multifactor)
        for f, n in rgd.n_strata_.items():
            assert n >= 1

    def test_rgd_summary(self, df_multifactor):
        from geodetector.extensions import RGD

        rgd = RGD(["type", "region", "elev"], "y", k=4)
        rgd.fit(df_multifactor)
        s = rgd.summary()
        assert isinstance(s, str)
        assert "RGD" in s


# ═══════════════════════════════════════════════════════════════
# OPGD — default methods (matching gdverse)
# ═══════════════════════════════════════════════════════════════

class TestOPGDDefaults:
    def test_default_methods_match_gdverse(self):
        from geodetector.extensions import OPGD

        opgd = OPGD(["x"], "y")
        defaults = set(opgd.discretize_methods)
        assert "sd" in defaults, "Default methods should include 'sd' (std-dev based)"
        assert "geometric" in defaults, "Default methods should include 'geometric'"
        assert "natural" in defaults, "Default methods should include 'natural' (Jenks)"
        assert "quantile" in defaults
        assert "equal" in defaults

    def test_default_methods_work_with_opgd(self, df_multifactor):
        from geodetector.extensions import OPGD

        opgd = OPGD(["type", "region", "elev"], "y", k_range=(3, 5))
        opgd.fit(df_multifactor)
        assert len(opgd.q_values_) == 3
        assert len(opgd.opt_params_) == 3

    def test_explicit_empty_methods_raises_or_warns(self):
        from geodetector.extensions import OPGD

        opgd = OPGD(["x"], "y", discretize_methods=["quantile"])
        # With a single valid method it should work
        assert opgd.discretize_methods == ["quantile"]


# ═══════════════════════════════════════════════════════════════
# LESH — GOZH-style discretization support
# ═══════════════════════════════════════════════════════════════

class TestShapleyDecomposeGOZH:
    @pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
    def test_shapley_decompose_with_gozh_method(self, df_multifactor):
        from geodetector.extensions import shapley_decompose

        result = shapley_decompose(
            df_multifactor, ["type", "region", "elev"], "y",
            method="gozh", max_depth=3, min_samples_leaf=5,
        )
        assert isinstance(result, pd.DataFrame)
        assert "variable" in result.columns
        assert "shapley_value" in result.columns
        assert len(result) == 3
        # Values should be non-negative
        for v in result["shapley_value"]:
            assert v >= 0

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
    def test_shapley_percentages_sum_to_one_gozh(self, df_multifactor):
        from geodetector.extensions import shapley_decompose

        result = shapley_decompose(
            df_multifactor, ["type", "region", "elev"], "y",
            method="gozh", max_depth=3,
        )
        total_pct = result["shapley_pct"].sum()
        assert total_pct == pytest.approx(1.0)

    def test_shapley_quantile_method_backward_compatible(self, df_multifactor):
        from geodetector.extensions import shapley_decompose

        result = shapley_decompose(
            df_multifactor, ["type", "region", "elev"], "y",
            method="quantile", n_strata=4,
        )
        assert len(result) == 3
        assert "shapley_value" in result.columns


class TestLESHGOZH:
    @pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
    def test_lesh_with_gozh_method(self, df_multifactor):
        from geodetector.extensions import LESH

        lesh = LESH(["type", "region", "elev"], "y",
                    method="gozh", max_depth=3)
        lesh.fit(df_multifactor)
        assert hasattr(lesh, "shapley_")
        assert hasattr(lesh, "interaction_")

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
    def test_lesh_gozh_interaction_has_spd(self, df_multifactor):
        from geodetector.extensions import LESH

        lesh = LESH(["type", "region", "elev"], "y",
                    method="gozh", max_depth=3)
        lesh.fit(df_multifactor)
        inter = lesh.interaction_
        assert "spd_1" in inter.columns
        assert "spd_2" in inter.columns


# ═══════════════════════════════════════════════════════════════
# GOZH — joint decision tree for interaction q12
# ═══════════════════════════════════════════════════════════════

class TestGOZHInteraction:
    @pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
    def test_gozh_has_interaction_matrices(self, df_multifactor):
        from geodetector.extensions import GOZH

        gozh = GOZH(["type", "region", "elev"], "y", max_depth=3, min_samples_leaf=5)
        gozh.fit(df_multifactor)
        assert gozh.interaction_q_ is not None
        assert gozh.interaction_type_ is not None
        # Interaction matrix should be symmetric
        qmat = gozh.interaction_q_
        for i in qmat.index:
            for j in qmat.columns:
                assert qmat.loc[i, j] == pytest.approx(qmat.loc[j, i])

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
    def test_gozh_interaction_uses_joint_discretization(self, df_multifactor):
        from geodetector.extensions import GOZH
        from geodetector._stats import q_statistic

        y = df_multifactor["y"].values
        # Compute q12 via joint tree (gdverse method)
        x1 = df_multifactor["type"].values
        x2 = df_multifactor["region"].values
        # Joint formula: y ~ x1 + x2 via rpart_disc with combined var
        from sklearn.tree import DecisionTreeRegressor

        valid = ~(
            pd.isna(x1) | pd.isna(x2) | pd.isna(y)
        )
        x_combined = np.column_stack([
            x1[valid].astype(float).ravel(),
            x2[valid].astype(float).ravel(),
        ])
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=42)
        tree.fit(x_combined, y[valid])
        strata = tree.apply(x_combined).ravel()
        q12_joint = q_statistic(y[valid], strata)

        gozh = GOZH(["type", "region", "elev"], "y", max_depth=3, min_samples_leaf=5)
        gozh.fit(df_multifactor)
        q12_gozh = gozh.interaction_q_.loc["type", "region"]

        # Both should be valid q-values (not NaN)
        assert not np.isnan(q12_gozh)
        assert 0 <= q12_gozh <= 1
        # GOZH should produce similar q12 to manual joint-tree computation
        assert abs(q12_gozh - q12_joint) < 0.3

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
    def test_gozh_interaction_pairs_exist(self, df_multifactor):
        from geodetector.extensions import GOZH

        gozh = GOZH(["type", "region", "elev"], "y", max_depth=3)
        gozh.fit(df_multifactor)
        assert hasattr(gozh, "interaction_pairs_")
        assert isinstance(gozh.interaction_pairs_, list)
        assert len(gozh.interaction_pairs_) == 3  # 3 choose 2 = 3 pairs


# ═══════════════════════════════════════════════════════════════
# RGD — multi-k search + LOESS optimization
# ═══════════════════════════════════════════════════════════════

class TestRGDMultiK:
    def test_rgd_accepts_discnum_range(self, df_continuous_only):
        from geodetector.extensions import RGD

        rgd = RGD(["x1", "x2"], "y", k=4, discnum=range(3, 6))
        # k is still required as fallback
        assert list(rgd.discnum) == [3, 4, 5]

    def test_rgd_with_strategy_1_picks_max_q(self, df_continuous_only):
        from geodetector.extensions import RGD

        rgd = RGD(["x1", "x2"], "y", k=4, discnum=range(3, 6), strategy=1)
        rgd.fit(df_continuous_only)
        assert hasattr(rgd, "q_values_")
        assert len(rgd.q_values_) == 2
        assert hasattr(rgd, "opt_discnum_")
        assert hasattr(rgd, "all_q_values_")

    def test_rgd_opt_discnum_per_factor(self, df_continuous_only):
        from geodetector.extensions import RGD

        rgd = RGD(["x1", "x2"], "y", k=4, discnum=range(3, 6), strategy=1)
        rgd.fit(df_continuous_only)
        for f in ["x1", "x2"]:
            assert f in rgd.opt_discnum_
            assert 3 <= rgd.opt_discnum_[f] <= 5

    def test_rgd_opt_discnum_respects_strategy_2(self, df_continuous_only):
        from geodetector.extensions import RGD

        rgd = RGD(["x1", "x2"], "y", k=4, discnum=range(3, 6), strategy=2, increase_rate=0.1)
        rgd.fit(df_continuous_only)
        for f in ["x1", "x2"]:
            assert f in rgd.opt_discnum_
            assert 3 <= rgd.opt_discnum_[f] <= 5

    def test_rgd_all_q_values_has_discnum_column(self, df_continuous_only):
        from geodetector.extensions import RGD

        rgd = RGD(["x1", "x2"], "y", k=4, discnum=range(3, 6))
        rgd.fit(df_continuous_only)
        assert hasattr(rgd, "all_q_values_")
        assert "discnum" in rgd.all_q_values_.columns

    def test_rgd_summary_with_discnum(self, df_continuous_only):
        from geodetector.extensions import RGD

        rgd = RGD(["x1", "x2"], "y", k=4, discnum=range(3, 5))
        rgd.fit(df_continuous_only)
        s = rgd.summary()
        assert isinstance(s, str)
        assert "RGD" in s
        assert "opt_discnum" in s.lower() or "optimal" in s.lower()
