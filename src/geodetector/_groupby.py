"""Group-by utility for stratified heterogeneity computation.

Groups a response vector y by stratum labels in X.  The input X must be
integer-typed (already discretized).  Returns a dictionary mapping each
stratum tuple to the array of y values within that stratum.
"""

import numpy as np


def groupby(X, y):
    """Group y values by stratum keys in X.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype int
        Integer-coded stratum labels (one column per factor).
    y : ndarray of shape (n_samples,), dtype float
        Response variable.

    Returns
    -------
    dict
        Keys are tuples of integer stratum labels, values are ndarrays
        of y values belonging to that stratum.

    Examples
    --------
    >>> X = np.array([[1, 2], [1, 3], [1, 2], [2, 3], [2, 4], [2, 5]])
    >>> y = np.array([1.7, 2.5, 3.3, 4.1, 5.6, 6.2])
    >>> groupby(X, y)
    {(1, 2): array([1.7, 3.3]), (1, 3): array([2.5]),
     (2, 3): array([4.1]), (2, 4): array([5.6]),
     (2, 5): array([6.2])}
    """
    col_num = X.shape[1]

    # Attach row index then stable-sort column by column (lexicographic)
    idx = np.arange(X.shape[0]).reshape(-1, 1)
    X_idx = np.hstack((X, idx))

    for i in range(col_num - 1, -1, -1):
        X_idx = X_idx[X_idx[:, i].argsort(kind="stable")]

    X_sorted = X_idx[:, :-1]
    idx_sorted = X_idx[:, -1].astype(int)
    y_sorted = y[idx_sorted]

    unique_val, unique_idx = np.unique(X_sorted, axis=0, return_index=True)
    layer_y = np.split(y_sorted, unique_idx)[1:]
    result = dict(zip(map(tuple, unique_val), layer_y))
    return result
