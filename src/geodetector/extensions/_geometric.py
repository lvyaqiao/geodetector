"""Geometric-interval discretization.

Implements geometric progression breaks, which mapclassify 2.x no longer
provides.  Used when method="geometric" is requested.
"""

import numpy as np


def geometric_breaks(data, k):
    """Compute geometric-progression break points.

    For strictly positive data:  breaks follow  a, a·r, a·r², ..., a·rᵏ
    where  a = min(data)  and  r = (max/min)^(1/(k-1)).

    For data spanning zero (negative and positive), the positive and
    negative sides are handled separately.

    Parameters
    ----------
    data : ndarray
        1-D array of values.
    k : int
        Number of classes.

    Returns
    -------
    ndarray
        (k-1) internal break points (excludes min/max endpoints,
        matching mapclassify convention where ``bins`` are the
        upper bounds of each class).
    """
    data = np.asarray(data, dtype=float)
    dmin, dmax = np.nanmin(data), np.nanmax(data)

    if k < 2:
        return np.array([])

    if dmin <= 0 and dmax >= 0:
        # Data spans zero — split into negative and positive portions
        neg = data[data < 0]
        pos = data[data > 0]

        n_neg = max(1, int(k * abs(dmin) / (abs(dmin) + abs(dmax))))
        n_pos = k - n_neg

        breaks = []
        if len(neg) > 0 and n_neg > 1:
            neg_min, neg_max = np.min(neg), np.max(neg)
            r_neg = (abs(neg_max) / abs(neg_min)) ** (1.0 / (n_neg - 1))
            for i in range(1, n_neg):
                breaks.append(-neg_min * (r_neg ** (n_neg - i)))
            breaks.append(0.0)
        elif len(neg) > 0:
            breaks.append(0.0)

        if len(pos) > 0 and n_pos > 1:
            pos_min, pos_max = np.min(pos), np.max(pos)
            r_pos = (pos_max / pos_min) ** (1.0 / (n_pos - 1))
            for i in range(1, n_pos):
                breaks.append(pos_min * (r_pos ** i))
        elif len(pos) > 0:
            breaks.append(pos_max)

        return np.array(sorted(set(breaks)))

    # Strictly positive or strictly negative
    if dmin >= 0:
        offset = 1.0 if dmin < 1 else 0.0
        adj = data + offset
        r = (np.max(adj) / np.min(adj)) ** (1.0 / (k - 1))
        breaks = [np.min(adj) * (r ** i) - offset for i in range(1, k)]
    else:  # all negative
        adj = -data
        r = (np.max(adj) / np.min(adj)) ** (1.0 / (k - 1))
        breaks = [-(np.min(adj) * (r ** i)) for i in range(1, k)]

    return np.array(breaks)


def discretize_geometric(x, *, n_strata=5):
    """Discretize using geometric-progression breaks.

    Parameters
    ----------
    x : array-like
        Continuous values.
    n_strata : int, default=5

    Returns
    -------
    ndarray of int
        Integer stratum labels.
    """
    x = np.asarray(x, dtype=float).ravel()
    x_clean = x[~np.isnan(x)]
    n = min(n_strata, len(np.unique(x_clean)))
    if n < 2:
        return np.zeros(len(x), dtype=int)

    bins = geometric_breaks(x_clean, n)
    if len(bins) == 0:
        return np.zeros(len(x), dtype=int)

    labels = np.digitize(x, bins, right=True)
    labels[np.isnan(x)] = -1
    return labels.astype(int)
