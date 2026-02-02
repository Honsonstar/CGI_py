"""
KCIT - Kernel Conditional Independence Test

A wrapper for unconditional and conditional independence tests using
kernel-based methods (HSIC).

Copyright (c) 2011 Kun Zhang, Jonas Peters
"""

import numpy as np
from .uind_test import uind_test
from .cind_test import cind_test_new_with_gp


def kcit(X: np.ndarray, Y: np.ndarray, Z: np.ndarray = None,
         pairwise: bool = False, bonferroni: bool = False,
         width: float = 0, alpha: float = 0.05) -> tuple:
    """
    Kernel Conditional Independence Test.

    Args:
        X: (n, d1) matrix of samples for X
        Y: (n, d2) matrix of samples for Y
        Z: (n, d3) matrix of samples for conditioning variables, or None
        pairwise: if True, perform pairwise tests for multi-dimensional variables
        bonferroni: if True, apply Bonferroni correction
        width: kernel width (0 for automatic selection)
        alpha: significance level

    Returns:
        ind: True if X and Y are independent, False otherwise
        stat: test statistic
        p_val: p-value
    """
    # width=0 is passed through to uind_test/cind_test_new_with_gp
    # for automatic selection (median heuristic or sample-size based)

    # Check dimensions
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # This test only works for one-dimensional X and Y
    if X.shape[1] > 1 or Y.shape[1] > 1:
        raise ValueError('This test only works for one-dimensional X and Y')

    if Z is None or Z.size == 0:
        # Unconditional HSIC
        stat, Cri, p_val, Cri_appr, p_appr = uind_test(X.flatten(), Y.flatten(), alpha, width)
    else:
        # Conditional HSIC with GP regression
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        stat, Cri, p_val, Cri_appr, p_appr, _ = cind_test_new_with_gp(
            X.flatten(), Y.flatten(), Z, alpha, width
        )

    # Determine independence based on p-value
    if p_val > alpha:
        ind = True
    else:
        ind = False

    return ind, stat, p_val
