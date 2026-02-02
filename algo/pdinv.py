"""
PDINV - Computes the inverse of a positive definite matrix

Copyright (c) 2010-2011
"""

import numpy as np
from scipy import linalg


def pdinv(A: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a positive definite matrix.

    Uses Cholesky decomposition for efficiency, with SVD fallback
    for non-positive definite matrices.

    Args:
        A: Positive definite matrix

    Returns:
        Ainv: Inverse of A
    """
    n = A.shape[0]

    try:
        U = linalg.cholesky(A, lower=False)
        invU = linalg.solve_triangular(U, np.eye(n), lower=False)
        Ainv = invU.T @ invU
    except linalg.LinAlgError:
        # Fall back to SVD if Cholesky fails
        U, S, V = linalg.svd(A)
        Ainv = V.T @ np.diag(1.0 / S) @ U.T

    return Ainv
