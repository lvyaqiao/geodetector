"""Lightweight sklearn-compatible base classes.

No dependency on scikit-learn. Provides just enough API surface
for get_params / set_params / score to work.

If scikit-learn is installed, the classes below are replaced by
their sklearn counterparts in __init__.py.
"""

import inspect

import numpy as np


class NotFittedError(Exception):
    """Exception raised when calling predict/transform before fit."""
    pass


class BaseEstimator:
    """Minimal sklearn-compatible estimator base."""

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, recursively get parameters of nested estimators.

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        out = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and isinstance(value, BaseEstimator):
                deep_items = value.get_params(deep=True).items()
                out.update({key + "__" + k: v for k, v in deep_items})
            else:
                out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Returns
        -------
        self
        """
        if not params:
            return self
        valid_params = self.get_params(deep=False)
        nested_params = {}
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if delim:
                nested_params.setdefault(key, {})[sub_key] = value
            else:
                if key not in valid_params:
                    raise ValueError(
                        f"Invalid parameter {key!r} for estimator {self.__class__.__name__}. "
                        f"Valid parameters are: {list(valid_params.keys())}"
                    )
                setattr(self, key, value)
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
        return self

    def _get_param_names(self):
        """Extract parameter names from __init__ signature."""
        init = getattr(self.__init__, "deprecated_original", self.__init__)
        if init is object.__init__:
            return []
        sig = inspect.signature(init)
        return [p.name for p in sig.parameters.values()
                if p.name != "self" and p.kind != p.VAR_KEYWORD]


class RegressorMixin:
    """Minimal sklearn-compatible regressor mixin."""

    def score(self, X, y):
        """Return the coefficient of determination R².

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.

        Returns
        -------
        score : float
            R² of self.predict(X) w.r.t. y.
        """
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        if sst == 0:
            return 0.0
        return 1.0 - sse / sst


# Classes exported at package level:
# Users get BaseEstimator / RegressorMixin from geodetector._base
# which may be swapped for sklearn versions at import time.
