"""
EIGDEC - Sorted eigendecomposition

EVALS = EIGDEC(X, N) computes the largest N eigenvalues of the
matrix X in descending order.

[EVALS, EVEC] = EIGDEC(X, N) also computes the corresponding eigenvectors.

Copyright (c) Ian T Nabney (1996-2001)
"""

import numpy as np
from scipy import linalg


def eigdec(x: np.ndarray, N: int) -> tuple:
    """
    Compute the largest N eigenvalues and optionally eigenvectors.

    Args:
        x: Input matrix
        N: Number of eigenvalues to compute

    Returns:
        evals: (N,) array of largest N eigenvalues in descending order
        evec: (n, N) array of corresponding eigenvectors (if requested)
    """
    n = x.shape[0]

    if N < 1 or N > n:
        raise ValueError('Number of PCs must be integer, >0, < dim')

    # Use eig function as it's generally more reliable
    temp_evals, temp_evec = linalg.eigh(x)
    # eigh returns in ascending order, so we need to reverse
    temp_evals = temp_evals[::-1]
    temp_evec = temp_evec[:, ::-1]

    evals = temp_evals[:N]

    if N == len(temp_evals):
        evec = temp_evec[:, :N]
    else:
        evec = temp_evec[:, :N]

    return evals, evec


def eigdec_evals_only(x: np.ndarray, N: int) -> np.ndarray:
    """
    Compute only the largest N eigenvalues.

    Args:
        x: Input matrix
        N: Number of eigenvalues to compute

    Returns:
        evals: (N,) array of largest N eigenvalues
    """
    temp_evals = np.linalg.eigvalsh(x)
    evals = np.sort(temp_evals)[::-1][:N]
    return evals
