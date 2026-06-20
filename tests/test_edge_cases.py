"""Tests for edge cases and error handling."""

import numpy as np
import pandas as pd
import pytest
import warnings


class TestZeroVarianceY:
    def test_factor_detector_handles_zero_variance(self, df_no_variance):
        """When SST=0 (all Y identical), q should be NaN."""
        from geodetector.detectors import FactorDetector

        fd = FactorDetector()
        fd.fit(df_no_variance[["group"]], df_no_variance["y"])
        # SST=0 → q is mathematically undefined
        assert np.isnan(fd.q_value_)

    def test_geo_detector_handles_zero_variance(self, df_no_variance):
        from geodetector import GeoDetector

        gd = GeoDetector(factors=["group"], target="y")
        gd.fit(df_no_variance)
        # Should not raise
        assert isinstance(gd.q_values_, pd.DataFrame)


class TestSingleStratum:
    def test_factor_detector_single_stratum(self, df_single_group):
        from geodetector.detectors import FactorDetector

        fd = FactorDetector()
        fd.fit(df_single_group[["group"]], df_single_group["y"])
        # Single group → no stratification possible
        assert fd.q_value_ == 0.0 or np.isnan(fd.q_value_)

    def test_interaction_single_factor_raises(self, df_single_group):
        from geodetector.detectors import InteractionDetector

        id_ = InteractionDetector()
        with pytest.raises(ValueError):
            id_.fit(df_single_group[["group"]], df_single_group["y"])


class TestMissingValues:
    def test_nan_rows_filtered_correctly(self, df_with_na):
        """NaN in x or y should be dropped; remaining data produces valid q."""
        from geodetector.detectors import FactorDetector

        fd = FactorDetector()
        fd.fit(df_with_na[["x"]], df_with_na["y"])
        # After NaN filtering, at least some data should remain
        assert not np.isnan(fd.q_value_)
        assert 0.0 <= fd.q_value_ <= 1.0
        assert 0.0 <= fd.p_value_ <= 1.0


class TestEmptyOrTinyData:
    def test_empty_dataframe(self):
        from geodetector.detectors import FactorDetector

        df = pd.DataFrame({"x": [], "y": []})
        fd = FactorDetector()
        with pytest.raises(ValueError):
            fd.fit(df[["x"]], df["y"])

    def test_single_row(self):
        from geodetector.detectors import FactorDetector

        df = pd.DataFrame({"x": [0], "y": [5.0]})
        fd = FactorDetector()
        fd.fit(df[["x"]], df["y"])
        assert np.isnan(fd.q_value_) or fd.q_value_ == 0.0


class TestDuplicateColumns:
    def test_duplicate_x_columns(self):
        from geodetector import GeoDetector

        df = pd.DataFrame({
            "f1": [1, 2, 3, 1, 2, 3],
            "f2": [1, 2, 3, 1, 2, 3],  # identical to f1
            "y":  [10, 20, 30, 11, 19, 29],
        })
        gd = GeoDetector(factors=["f1", "f2"], target="y")
        gd.fit(df)
        # f1 and f2 have identical q since they are the same column
        assert gd.q_values_.loc[gd.q_values_["variable"] == "f1", "q_value"].values[0] == pytest.approx(
            gd.q_values_.loc[gd.q_values_["variable"] == "f2", "q_value"].values[0]
        )


class TestLargeNFactors:
    def test_10_factors_no_crash(self):
        """10+ factors should not crash, only warn about interaction explosion."""
        from geodetector import GeoDetector

        n = 100
        df = pd.DataFrame({f"f{i}": np.random.default_rng(42).choice([0, 1, 2], n) for i in range(10)})
        df["y"] = np.random.default_rng(42).normal(0, 1, n)

        gd = GeoDetector(target="y")
        # For 10 factors, 10 q-values + C(10,2)=45 interaction pairs = reasonable
        gd.fit(df)
        assert len(gd.q_values_) == 10


class TestNonIntegerX:
    def test_float_x_is_cast_or_discretized(self):
        from geodetector.detectors import FactorDetector

        df = pd.DataFrame({"x": [0.0, 1.0, 0.0, 1.0, 2.0], "y": [1.0, 3.0, 1.1, 2.9, 5.0]})
        fd = FactorDetector()
        fd.fit(df[["x"]], df["y"])
        assert 0.0 <= fd.q_value_ <= 1.0


class TestReproducibility:
    def test_factor_detector_deterministic(self, df_multifactor):
        """Same data and seed → same results."""
        from geodetector.detectors import FactorDetector

        fd1 = FactorDetector(random_state=42).fit(df_multifactor[["type"]], df_multifactor["y"])
        fd2 = FactorDetector(random_state=42).fit(df_multifactor[["type"]], df_multifactor["y"])
        assert fd1.q_value_ == pytest.approx(fd2.q_value_)
        assert fd1.p_value_ == pytest.approx(fd2.p_value_)
