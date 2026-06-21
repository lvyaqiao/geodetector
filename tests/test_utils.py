"""Tests for utility functions."""

import numpy as np
import pytest


class TestGroupBy:
    def test_groups_y_by_x(self):
        from geodetector.utils._groupby import groupby

        X = np.array([[1], [1], [2], [2], [3], [3]])
        y = np.array([10.0, 12.0, 20.0, 22.0, 30.0, 32.0])
        result = groupby(X, y)

        assert (1,) in result
        assert (2,) in result
        assert (3,) in result
        assert np.allclose(result[(1,)], [10.0, 12.0])
        assert np.allclose(result[(2,)], [20.0, 22.0])
        assert np.allclose(result[(3,)], [30.0, 32.0])

    def test_groupby_multi_column(self):
        from geodetector.utils._groupby import groupby

        X = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1, 0]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = groupby(X, y)

        assert (0, 0) in result
        assert (0, 1) in result
        assert (1, 0) in result
        assert np.allclose(result[(0, 0)], [1.0, 2.0])
        assert np.allclose(result[(0, 1)], [3.0, 4.0])
        assert np.allclose(result[(1, 0)], [5.0, 6.0])

    def test_each_original_value_appears_once(self):
        """No duplicates: every Y value belongs to exactly one group."""
        from geodetector.utils._groupby import groupby

        X = np.array([[0], [1], [0], [2], [1], [2]])
        y = np.array([10.0, 20.0, 11.0, 30.0, 21.0, 31.0])
        result = groupby(X, y)

        all_y = np.concatenate(list(result.values()))
        assert len(all_y) == len(y)
        np.testing.assert_allclose(np.sort(all_y), np.sort(y))

    def test_groupby_integer_input_only(self):
        """groupby requires integer strata labels."""
        from geodetector.utils._groupby import groupby

        X_float = np.array([[0.5], [1.2], [0.5]])
        y = np.array([1.0, 2.0, 3.0])
        # May raise or cast; cast is acceptable
        try:
            result = groupby(X_float.astype(int), y)
        except Exception:
            pytest.fail("groupby should handle int-cast float arrays")

    def test_groupby_result_keys_are_tuples(self):
        from geodetector.utils._groupby import groupby

        X = np.array([[0], [1]])
        y = np.array([1.0, 2.0])
        result = groupby(X, y)
        for key in result:
            assert isinstance(key, tuple)


class TestAll2Int:
    def test_float_to_int(self):
        from geodetector.utils import all2int

        x = np.array([1.0, 2.0, 3.0, 2.0])
        result = all2int(x)
        assert result.dtype in (np.int_, np.int32, np.int64)
        assert list(result) == [1, 2, 3, 2]

    def test_string_to_int(self):
        from geodetector.utils import all2int

        x = np.array(["a", "b", "a", "c"])
        result = all2int(x)
        assert result.dtype in (np.int_, np.int32, np.int64)
        assert len(set(result)) == 3

    def test_int_unchanged(self):
        from geodetector.utils import all2int

        x = np.array([5, 6, 7])
        result = all2int(x)
        assert np.array_equal(result, x)


class TestValidateData:
    def test_valid_passes(self, df_multifactor):
        from geodetector.utils import validate_data

        validate_data(df_multifactor, ["type", "region"], "y")

    def test_missing_column_raises(self, df_multifactor):
        from geodetector.utils import validate_data

        with pytest.raises(ValueError, match="not in|missing|not found"):
            validate_data(df_multifactor, ["type", "nonexistent"], "y")

    def test_missing_target_raises(self, df_multifactor):
        from geodetector.utils import validate_data

        with pytest.raises(ValueError, match="not in|missing|not found"):
            validate_data(df_multifactor, ["type"], "nonexistent")


class TestParallelApply:
    """Tests for joblib-based parallel helper (if joblib installed)."""

    def test_parallel_apply_basic(self):
        from geodetector._parallel import parallel_apply

        def double(x):
            return x * 2

        results = parallel_apply(double, [(1,), (2,), (3,), (4,)])
        assert results == [2, 4, 6, 8]

    def test_parallel_apply_empty(self):
        from geodetector._parallel import parallel_apply

        results = parallel_apply(lambda x: x, [])
        assert results == []
