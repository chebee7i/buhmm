# encoding: utf-8
"""
Miscellaneous functions.

"""
from __future__ import division

import numpy as np

def logspace_int(limit, num=50):
    """
    Returns (approximately) logarithmically spaced integers from 0 to `limit`.

    This is often more appropriate than calling `np.logspace(...).astype(int)`,
    or something similar, as those approaches will contain duplicate integers.

    One common use case is to generate logarithmically spaced indices.

    Parameters
    ----------
    limit : int
        The maximum possible integer.
    num : int, optional
        Number of samples to generate. Default is 50.

    Returns
    -------
    samples : NumPy array
        The `num` logarithmically spaced integer samples.

    References
    ----------
    .. [1] http://stackoverflow.com/a/12421820

    """
    if limit <= 0:
        raise Exception('`limit` must be greater than zero.')

    if num == 0:
        return np.array([], dtype=np.uint64)
    elif num == 1:
        return np.array([0], dtype=np.uint64)

    if limit < num:
        msg = "Not enough integers between 0 and {0}".format(limit)
        raise Exception(msg)

    result = [1]
    if num > 1:
        # Only calculate ratio if we avoid a ZeroDivisionError.
        ratio = ( limit / result[-1] ) ** (1 / (num - len(result)))

    while len(result) < num:
        next_value = result[-1] * ratio
        if next_value - result[-1] >= 1:
            # Safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # Problem! Same integer. We need to find next_value by
            # artificially incrementing previous value
            result.append(result[-1] + 1)
            # Recalculate the ratio so that remaining values scale correctly.
            ratio = (limit / (result[-1])) ** (1 / (num - len(result)))

    # Round and return np.uint64 array
    result = np.round(result) - 1
    return result.astype(np.int64)
