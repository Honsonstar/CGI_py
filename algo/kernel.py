"""
KERNEL - Compute the RBF (Gaussian) kernel

Computes the radial basis function (RBF) kernel between two sets of points.

Copyright (c) 2010-2011
"""

import numpy as np
from .dist2 import dist2


def kernel(x: np.ndarray, x_kern: np.ndarray, theta: np.ndarray) -> tuple:
    """
    Compute the RBF kernel matrix.

    Args:
        x: Input points (n1, d)
        x_kern: Input points (n2, d)
        theta: Hyperparameters [length_scale, variance]

    Returns:
        kx: Kernel matrix (n1, n2)
        bw_new: 1/length_scale^2
    """
    n2 = dist2(x, x_kern)

    if theta[0] == 0:
        # Automatic bandwidth selection using median heuristic
        n2_valid = n2[np.tril_indices_from(n2, k=-1)]
        n2_valid = n2_valid[n2_valid > 0]
        if len(n2_valid) > 0:
            theta[0] = 2 / np.median(n2_valid)
        else:
            theta[0] = 1.0

    wi2 = theta[0] / 2
    kx = theta[1] * np.exp(-n2 * wi2)
    bw_new = 1 / theta[0]

    return kx, bw_new


def kernel_matrix(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute the kernel matrix for a single set of points.

    Args:
        x: Input points (n, d)
        theta: Hyperparameters [length_scale, variance]

    Returns:
        K: Kernel matrix (n, n)
    """
    n2 = dist2(x, x)
    wi2 = theta[0] / 2
    K = theta[1] * np.exp(-n2 * wi2)
    return K
