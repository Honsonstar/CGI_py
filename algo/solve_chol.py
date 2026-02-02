"""
SOLVE_CHOL - Solve linear system using Cholesky decomposition

Copyright (c) 2010-2011
"""

import numpy as np
from scipy import linalg


def solve_chol(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solve L @ L^T @ X = B for X, where L is lower triangular Cholesky factor.

    Args:
        L: Lower triangular Cholesky factor (from chol(A) where A = L @ L.T)
        B: Right-hand side matrix/vector

    Returns:
        X: Solution matrix/vector
    """
    return linalg.solve_triangular(L.T, linalg.solve_triangular(L, B, lower=True), lower=False)
