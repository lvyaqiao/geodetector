"""Utility functions for geodetector."""

import numpy as np
import pandas as pd

from ._groupby import groupby


def all2int(x):
    """Convert any array-like to integer codes.

    Parameters
    ----------
    x : array-like
        Input array (can be float, string, categorical).

    Returns
    -------
    ndarray of int
        Integer-coded version of x.
    """
    x = np.asarray(x)
    if x.dtype.kind in ("i", "u"):
        return x.astype(int)
    if x.dtype.kind == "f":
        return np.rint(x).astype(int)
    codes, _ = pd.factorize(x)
    return codes.astype(int)


def validate_data(data, x_names, y_name):
    """Validate that column names exist in the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input data.
    x_names : list of str
        Column names of explanatory variables.
    y_name : str
        Column name of the response variable.

    Raises
    ------
    ValueError
        If any column name is not found in data.
    """
    for x_name in x_names:
        if x_name not in data.columns:
            raise ValueError(
                f"Factor column '{x_name}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )
    if y_name not in data.columns:
        raise ValueError(
            f"Target column '{y_name}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )


def remove_single_strata(y, x):
    """Remove observations belonging to strata with only 1 observation.

    Parameters
    ----------
    y : ndarray
        Response variable.
    x : ndarray of int
        Stratum labels.

    Returns
    -------
    y_clean, x_clean : tuple of ndarray
        Filtered arrays.
    """
    x = np.asarray(x, dtype=int).ravel()
    y = np.asarray(y, dtype=float)
    unique, counts = np.unique(x, return_counts=True)
    valid = unique[counts >= 2]
    if len(valid) < 2:
        return np.array([]), np.array([])
    mask = np.isin(x, valid)
    return y[mask], x[mask]
