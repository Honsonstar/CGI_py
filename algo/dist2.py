"""
DIST2 - Calculates squared distance between two sets of points.

D = DIST2(X, C) takes two matrices of vectors and calculates the
squared Euclidean distance between them. Both matrices must be of the
same column dimension. If X has M rows and N columns, and C has L rows
and N columns, then the result has M rows and L columns.

Copyright (c) Ian T Nabney (1996-2001)
"""

import numpy as np


def dist2(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Calculate squared Euclidean distance between two sets of points.

    Args:
        x: (ndata, dimx) array
        c: (ncentres, dimc) array

    Returns:
        n2: (ndata, ncentres) array of squared distances
    """
    ndata, dimx = x.shape
    ncentres, dimc = c.shape

    if dimx != dimc:
        raise ValueError('Data dimension does not match dimension of centres')

    # Compute squared distances using vectorized operations
    n2 = (np.ones((ncentres, 1)) * np.sum(x ** 2, axis=1, keepdims=True).T).T + \
         np.ones((ndata, 1)) * np.sum(c ** 2, axis=1, keepdims=True).T - \
         2 * np.dot(x, c.T)

    # Rounding errors occasionally cause negative entries in n2
    n2[n2 < 0] = 0

    return n2
