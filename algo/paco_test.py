"""
PaCoTest - Partial Correlation Test

Implements partial correlation test for conditional independence.

Based on partial correlation computation from:
http://en.wikipedia.org/wiki/Partial_correlation
"""

import numpy as np
from scipy import stats


def paco_test(x: np.ndarray, y: np.ndarray, Z: np.ndarray = None, alpha: float = 0.05) -> bool:
    """
    Test conditional independence using partial correlation.

    Args:
        x: (n,) array of samples for variable X
        y: (n,) array of samples for variable Y
        Z: (n, d) array of conditioning variables, or None for unconditional test
        alpha: significance level

    Returns:
        cit: True if X and Y are conditionally independent, False otherwise
    """
    x = x.flatten()
    y = y.flatten()
    n = len(x)

    if Z is None or Z.size == 0:
        # Unconditional test: compute Pearson correlation
        pcc, _ = stats.pearsonr(x, y)
        ncit = 0
    else:
        Z = Z.reshape(n, -1)
        Z = np.column_stack([np.ones(n), Z])

        # Linear regression residuals
        wx, _, _, _ = np.linalg.lstsq(Z, x, rcond=None)
        rx = x - Z @ wx

        wy, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
        ry = y - Z @ wy

        pcc, _ = stats.pearsonr(rx, ry)
        ncit = Z.shape[1] - 1  # Number of conditioning variables (excluding intercept)

    # Fisher's z-transform
    zpcc = 0.5 * np.log((1 + pcc) / (1 - pcc)) if abs(pcc) < 1 else 0

    # Test statistic
    stat = np.sqrt(n - ncit - 3) * np.abs(zpcc)

    # Critical value
    crit = stats.norm.ppf(1 - alpha / 2)

    # Return True if independent (fail to reject null)
    cit = stat <= crit

    return cit


def paco_test_stat(x: np.ndarray, y: np.ndarray, Z: np.ndarray = None) -> tuple:
    """
    Compute partial correlation and p-value.

    Returns:
        pcc: partial correlation coefficient
        p_value: p-value for independence test
    """
    x = x.flatten()
    y = y.flatten()
    n = len(x)

    if Z is None or Z.size == 0:
        pcc, p_value = stats.pearsonr(x, y)
        return pcc, p_value

    Z = Z.reshape(n, -1)
    Z = np.column_stack([np.ones(n), Z])

    wx, _, _, _ = np.linalg.lstsq(Z, x, rcond=None)
    rx = x - Z @ wx

    wy, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
    ry = y - Z @ wy

    pcc, p_value = stats.pearsonr(rx, ry)
    return pcc, p_value
