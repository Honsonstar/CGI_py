"""
MEDBW - Median distance heuristic for setting RBF kernel bandwidth

Uses the median distance between points for setting the bandwidth for RBF kernels.

Copyright (c) 2010 Robert Tillman, 2007 Arthur Gretton
"""

import numpy as np


def medbw(X: np.ndarray, maxpoints: int) -> float:
    """
    Compute bandwidth using median distance heuristic.

    Args:
        X: (n, p) matrix of n datapoints with dimensionality p
        maxpoints: maximum number of points to use

    Returns:
        sigma: bandwidth value
    """
    if maxpoints < 1 or maxpoints != int(maxpoints):
        raise ValueError('maxpoints must be a positive integer')

    n = X.shape[0]

    # Truncate data if more points than maxpoints
    if n > maxpoints:
        med = X[:maxpoints, :]
        n = maxpoints
    else:
        med = X

    # Find median distance between points
    G = np.sum(med * med, axis=1)
    Q = np.tile(G.reshape(-1, 1), (1, n))
    R = np.tile(G.reshape(1, -1), (n, 1))
    dists = Q + R - 2 * med @ med.T

    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dists_upper = dists[mask]
    dists_upper = dists_upper[dists_upper > 0]

    if len(dists_upper) == 0:
        return 1.0

    sigma = np.sqrt(0.5 * np.median(dists_upper))

    return sigma
