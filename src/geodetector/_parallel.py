"""Parallel processing helpers.

Uses joblib if available, falls back to serial execution otherwise.
"""

import os


def _n_jobs():
    """Detect number of usable CPU cores, minus 1 for breathing room."""
    try:
        n = os.cpu_count() or 1
        return max(1, n - 1)
    except Exception:
        return 1


def parallel_apply(func, items, *, n_jobs=-1, verbose=0):
    """Apply a function to a list of argument tuples in parallel.

    Parameters
    ----------
    func : callable
        Function to apply. Receives unpacked tuple arguments.
    items : list of tuple
        List of argument tuples, e.g. ``[(1,), (2, 3), (4,)]``.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means all available cores.
    verbose : int, default=0
        Verbosity level for joblib.

    Returns
    -------
    list
        Results in the same order as ``items``.
    """
    if len(items) == 0:
        return []

    try:
        from joblib import Parallel, delayed
    except ImportError:
        # Fallback: serial execution
        return [func(*args) for args in items]

    if n_jobs <= 0:
        n_jobs = _n_jobs() if n_jobs == -1 else 1

    return Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(func)(*args) for args in items
    )
