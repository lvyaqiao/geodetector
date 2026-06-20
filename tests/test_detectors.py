"""Tests for detector classes (FactorDetector, InteractionDetector,
RiskDetector, EcologicalDetector) and the master GeoDetector."""

import numpy as np
import pandas as pd
import pytest


# ────────────────────────────────────────────────────────────────
# FactorDetector
# ────────────────────────────────────────────────────────────────

class TestFactorDetector:
    def test_fit_returns_self(self, df_simple):
        from geodetector.detectors import FactorDetector

        fd = FactorDetector()
        result = fd.fit(df_simple[["group"]], df_simple["y"])
        assert result is fd

    def test_predict_returns_group_means(self, df_simple):
        from geodetector.detectors import FactorDetector

        fd = FactorDetector().fit(df_simple[["group"]], df_simple["y"])
        y_pred = fd.predict(df_simple[["group"]])
        assert len(y_pred) == len(df_simple)
        # Predictions should be the group means
        for grp in np.unique(df_simple["group"]):
            mask = df_simple["group"] == grp
            expected = df_simple.loc[mask, "y"].mean()
            assert np.allclose(y_pred[mask], expected)

    def test_score_equals_q_value(self, df_simple):
        from geodetector.detectors import FactorDetector

        fd = FactorDetector().fit(df_simple[["group"]], df_simple["y"])
        score = fd.score(df_simple[["group"]], df_simple["y"])
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(fd.q_value_)

    def test_q_value_for_strong_signal(self, df_simple):
        from geodetector.detectors import FactorDetector

        fd = FactorDetector().fit(df_simple[["group"]], df_simple["y"])
        # Groups -2 vs +2 with small noise → high q
        assert fd.q_value_ > 0.8

    def test_p_value_for_strong_signal(self, df_simple):
        from geodetector.detectors import FactorDetector

        fd = FactorDetector().fit(df_simple[["group"]], df_simple["y"])
        assert fd.p_value_ < 0.001

    def test_predict_before_fit_raises(self, df_simple):
        from geodetector.detectors import FactorDetector
        from geodetector._base import NotFittedError

        fd = FactorDetector()
        with pytest.raises(NotFittedError):
            fd.predict(df_simple[["group"]])

    def test_get_params_default(self):
        from geodetector.detectors import FactorDetector

        fd = FactorDetector()
        params = fd.get_params()
        assert "discretize_method" in params
        assert "n_strata" in params

    def test_set_params_updates(self):
        from geodetector.detectors import FactorDetector

        fd = FactorDetector(n_strata=10)
        fd.set_params(n_strata=7)
        assert fd.n_strata == 7

    def test_continuous_x_is_discretized(self, df_continuous_only):
        from geodetector.detectors import FactorDetector

        fd = FactorDetector(discretize_method="quantile", n_strata=5)
        fd.fit(df_continuous_only[["x1"]], df_continuous_only["y"])
        assert hasattr(fd, "q_value_")
        assert 0.0 <= fd.q_value_ <= 1.0

    def test_perfect_q(self, df_perfect_q):
        from geodetector.detectors import FactorDetector

        fd = FactorDetector().fit(df_perfect_q[["x"]], df_perfect_q["y"])
        assert fd.q_value_ == pytest.approx(1.0)

    def test_transform_returns_discretized_labels(self, df_simple):
        from geodetector.detectors import FactorDetector

        fd = FactorDetector(discretize_method="quantile", n_strata=3)
        fd.fit(df_simple[["group"]], df_simple["y"])
        labels = fd.transform(df_simple[["group"]])
        assert len(labels) == len(df_simple)
        assert labels.dtype in (np.int_, np.int32, np.int64)


# ────────────────────────────────────────────────────────────────
# InteractionDetector
# ────────────────────────────────────────────────────────────────

class TestInteractionDetector:
    def test_fit_on_multifactor(self, df_multifactor):
        from geodetector.detectors import InteractionDetector

        id_ = InteractionDetector()
        id_.fit(df_multifactor[["type", "region"]], df_multifactor["y"])
        assert hasattr(id_, "interaction_q_")
        assert hasattr(id_, "interaction_type_")
        assert id_.interaction_q_.shape == (2, 2)

    def test_pair_has_interaction_type_in_0_to_4(self, df_multifactor):
        from geodetector.detectors import InteractionDetector

        id_ = InteractionDetector()
        id_.fit(df_multifactor[["type", "region"]], df_multifactor["y"])
        t = id_.interaction_type_
        assert t.loc["type", "region"] in (0, 1, 2, 3, 4)

    def test_diagonal_is_q_values(self, df_multifactor):
        from geodetector.detectors import InteractionDetector
        from geodetector.detectors import FactorDetector

        id_ = InteractionDetector()
        id_.fit(df_multifactor[["type", "region"]], df_multifactor["y"])

        fd = FactorDetector().fit(df_multifactor[["type"]], df_multifactor["y"])
        expected_q_type = fd.q_value_

        assert id_.interaction_q_.loc["type", "type"] == pytest.approx(expected_q_type)

    def test_single_factor_raises(self, df_simple):
        from geodetector.detectors import InteractionDetector

        id_ = InteractionDetector()
        with pytest.raises(ValueError, match="at least 2|need.*2"):
            id_.fit(df_simple[["group"]], df_simple["y"])


# ────────────────────────────────────────────────────────────────
# RiskDetector
# ────────────────────────────────────────────────────────────────

class TestRiskDetector:
    def test_returns_dataframe_with_expected_columns(self, df_simple):
        from geodetector.detectors import RiskDetector

        rd = RiskDetector()
        result = rd.fit(df_simple[["group"]], df_simple["y"]).risk_result_
        assert isinstance(result, dict)
        assert "group" in result
        df = result["group"]
        assert isinstance(df, pd.DataFrame)
        assert "stratum_1" in df.columns
        assert "stratum_2" in df.columns
        assert "p_value" in df.columns

    def test_significant_difference_for_clear_groups(self, df_simple):
        from geodetector.detectors import RiskDetector

        rd = RiskDetector()
        result = rd.fit(df_simple[["group"]], df_simple["y"]).risk_result_
        df = result["group"]
        # Group 0 (mean≈-2) vs Group 1 (mean≈+2) should be significant
        pair = df[((df["stratum_1"] == 0) & (df["stratum_2"] == 1))
                  | ((df["stratum_1"] == 1) & (df["stratum_2"] == 0))]
        assert len(pair) > 0, "Expected to find group 0 vs group 1 comparison"
        assert pair["p_value"].iloc[0] < 0.05

    def test_no_comparison_for_single_stratum(self, df_single_group):
        from geodetector.detectors import RiskDetector

        rd = RiskDetector()
        result = rd.fit(df_single_group[["group"]], df_single_group["y"]).risk_result_
        # Only 1 stratum → no pairwise comparison possible
        df = result.get("group")
        assert df is None or len(df) == 0

    def test_multifactor_returns_dict_of_results(self, df_multifactor):
        """When multiple X columns are given, returns results per factor."""
        from geodetector.detectors import RiskDetector

        rd = RiskDetector()
        result = rd.fit(df_multifactor[["type", "region"]], df_multifactor["y"]).risk_result_
        # Should be a dict with one key per factor
        assert isinstance(result, dict)
        assert "type" in result
        assert "region" in result


# ────────────────────────────────────────────────────────────────
# EcologicalDetector
# ────────────────────────────────────────────────────────────────

class TestEcologicalDetector:
    def test_returns_f_stat_and_p_value(self, df_multifactor):
        from geodetector.detectors import EcologicalDetector

        ed = EcologicalDetector()
        result = ed.fit(df_multifactor[["type", "region"]], df_multifactor["y"]).eco_result_
        assert isinstance(result, pd.DataFrame)
        assert "factor_1" in result.columns
        assert "factor_2" in result.columns
        assert "f_stat" in result.columns
        assert "p_value" in result.columns

    def test_significant_for_different_q_strength(self, df_multifactor):
        """type (strong) vs region (weak) may be significantly different."""
        from geodetector.detectors import EcologicalDetector

        ed = EcologicalDetector()
        result = ed.fit(df_multifactor[["type", "region"]], df_multifactor["y"]).eco_result_
        pair = result[((result["factor_1"] == "type") & (result["factor_2"] == "region"))
                      | ((result["factor_1"] == "region") & (result["factor_2"] == "type"))]
        assert len(pair) > 0
        assert pair["f_stat"].iloc[0] > 0
        assert 0.0 <= pair["p_value"].iloc[0] <= 1.0

    def test_single_factor_raises(self, df_simple):
        from geodetector.detectors import EcologicalDetector

        ed = EcologicalDetector()
        with pytest.raises(ValueError, match="at least 2|need.*2"):
            ed.fit(df_simple[["group"]], df_simple["y"])


# ────────────────────────────────────────────────────────────────
# GeoDetector (master)
# ────────────────────────────────────────────────────────────────

class TestGeoDetector:
    def test_fit_sets_expected_properties(self, df_multifactor):
        from geodetector import GeoDetector

        gd = GeoDetector(factors=["type", "region"], target="y")
        gd.fit(df_multifactor)

        assert hasattr(gd, "q_values_")
        assert hasattr(gd, "interaction_q_")
        assert hasattr(gd, "interaction_type_")
        assert hasattr(gd, "risk_result_")
        assert hasattr(gd, "ecological_result_")

    def test_q_values_is_dataframe(self, df_multifactor):
        from geodetector import GeoDetector

        gd = GeoDetector(factors=["type", "region"], target="y").fit(df_multifactor)
        qv = gd.q_values_
        assert isinstance(qv, pd.DataFrame)
        assert "variable" in qv.columns
        assert "q_value" in qv.columns
        assert "p_value" in qv.columns
        assert len(qv) == 2  # type and region

    def test_auto_detect_columns(self, df_multifactor):
        """When factors not specified, all columns except target are used."""
        from geodetector import GeoDetector

        gd = GeoDetector(target="y")
        gd.fit(df_multifactor)
        assert len(gd.q_values_) > 0

    def test_interaction_q_matrix_symmetric(self, df_multifactor):
        from geodetector import GeoDetector

        gd = GeoDetector(factors=["type", "region", "elev"], target="y")
        gd.fit(df_multifactor)
        m = gd.interaction_q_
        # Symmetric matrix
        assert (m == m.T).all().all()

    def test_plot_does_not_crash(self, df_multifactor):
        from geodetector import GeoDetector
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        gd = GeoDetector(factors=["type", "region"], target="y").fit(df_multifactor)
        try:
            gd.plot()
        except Exception as e:
            pytest.fail(f"plot() raised: {e}")

    def test_plot_interaction_does_not_crash(self, df_multifactor):
        from geodetector import GeoDetector
        import matplotlib

        matplotlib.use("Agg")
        gd = GeoDetector(factors=["type", "region"], target="y").fit(df_multifactor)
        try:
            gd.plot_interaction()
        except Exception as e:
            pytest.fail(f"plot_interaction() raised: {e}")

    def test_summary_returns_string(self, df_multifactor):
        from geodetector import GeoDetector

        gd = GeoDetector(factors=["type", "region"], target="y").fit(df_multifactor)
        result = gd.summary()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_real_disease_data(self, df_disease):
        from geodetector import GeoDetector

        gd = GeoDetector(factors=["type", "region", "level"], target="incidence")
        gd.fit(df_disease)

        assert len(gd.q_values_) == 3
        for qv in gd.q_values_["q_value"]:
            assert 0.0 <= qv <= 1.0

        # Interaction matrix should be 3×3
        assert gd.interaction_q_.shape == (3, 3)

    def test_large_data_no_timeout(self, df_large):
        from geodetector import GeoDetector

        gd = GeoDetector(factors=["f1", "f2", "f3"], target="y")
        gd.fit(df_large)
        assert len(gd.q_values_) == 3


# ────────────────────────────────────────────────────────────────
# Discretizer (Transformer)
# ────────────────────────────────────────────────────────────────

class TestDiscretizerTransformer:
    def test_fit_transform_returns_integer_matrix(self, df_continuous_only):
        from geodetector.discretize import Discretizer

        dt = Discretizer(discretize_method="quantile", n_strata=5)
        result = dt.fit_transform(df_continuous_only[["x1", "x2"]])
        assert result.shape == df_continuous_only[["x1", "x2"]].shape
        assert result.dtypes.unique().tolist()[0] in (np.int_, np.int32, np.int64)

    def test_fit_then_transform_idempotent(self, df_continuous_only):
        from geodetector.discretize import Discretizer

        dt = Discretizer(discretize_method="quantile", n_strata=5)
        dt.fit(df_continuous_only[["x1", "x2"]])
        r1 = dt.transform(df_continuous_only[["x1", "x2"]])
        r2 = dt.transform(df_continuous_only[["x1", "x2"]])
        assert np.array_equal(r1, r2)


# ────────────────────────────────────────────────────────────────
# OptimalDiscretizer
# ────────────────────────────────────────────────────────────────

class TestOptimalDiscretizer:
    def test_fit_finds_best_method_and_k(self, df_multifactor):
        from geodetector.discretize import OptimalDiscretizer

        od = OptimalDiscretizer(discretize_methods=["quantile", "equal"], k_range=(3, 5))
        od.fit(df_multifactor[["elev"]], df_multifactor["y"])
        assert hasattr(od, "best_method_")
        assert hasattr(od, "best_k_")
        assert od.best_method_["elev"] in ("quantile", "equal")
        assert 3 <= od.best_k_["elev"] <= 5

    def test_transform_uses_best_params(self, df_multifactor):
        from geodetector.discretize import OptimalDiscretizer

        od = OptimalDiscretizer(discretize_methods=["quantile", "equal"], k_range=(3, 5))
        od.fit(df_multifactor[["elev"]], df_multifactor["y"])
        result = od.transform(df_multifactor[["elev"]])
        assert result.shape[0] == len(df_multifactor)
        assert result.dtypes.iloc[0] in (np.int_, np.int32, np.int64)
