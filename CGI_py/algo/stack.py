"""
STACK - Stack the matrix M into the vector v

Copyright (c) 2010-2011
"""

import numpy as np


def stack(M: np.ndarray) -> np.ndarray:
    """
    Stack the matrix M into a vector.

    Args:
        M: Input matrix (n, t)

    Returns:
        v: Stacked vector of length n*t
    """
    n, t = M.shape
    v = np.zeros(n * t)

    for i in range(t):
        v[i * n:(i + 1) * n] = M[:, i]

    return v
