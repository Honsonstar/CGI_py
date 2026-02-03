"""
UInd_test - Unconditional Independence Test using HSIC

Tests if x and y are unconditionally independent using the Hilbert-Schmidt
Independence Criterion (HSIC).

Copyright (c) 2010-2011 Kun Zhang, Jonas Peters
"""

import numpy as np
from scipy import stats
from scipy import linalg
from .kernel import kernel
from .eigdec import eigdec
from .stack import stack


def uind_test(x: np.ndarray, y: np.ndarray, alpha: float = 0.8, width: float = 0) -> tuple:
    """
    Test unconditional independence between x and y.

    Args:
        x: (n,) or (n, 1) array
        y: (n,) or (n, 1) array
        alpha: significance level (default 0.8 for 80%, but typically use 0.05)
        width: kernel width (0 for automatic selection)

    Returns:
        Sta: test statistic (HSIC)
        Cri: critical value from bootstrap
        p_val: p-value from bootstrap
        Cri_appr: critical value from Gamma approximation
        p_appr: p-value from Gamma approximation
    """
    x = x.flatten()
    y = y.flatten()
    T = len(y)  # sample size

    # Automatic width selection
    if width == 0:
        if T < 200:
            width = 0.8
        elif T < 1200:
            width = 0.5
        else:
            width = 0.3

    # Normalize data
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)

    theta = 1 / (width ** 2)

    # Centering matrix
    H = np.eye(T) - np.ones((T, T)) / T

    # Compute kernel matrices
    res_x = kernel(x.reshape(-1, 1), x.reshape(-1, 1), np.array([theta, 1]))
    Kx = res_x[0] if isinstance(res_x, tuple) else res_x

    res_y = kernel(y.reshape(-1, 1), y.reshape(-1, 1), np.array([theta, 1]))
    Ky = res_y[0] if isinstance(res_y, tuple) else res_y

    Kx = H @ Kx @ H
    Ky = H @ Ky @ H

    Sta = np.trace(Kx @ Ky)

    # Compute eigenvalues
    num_eig = min(T // 2, 100)
    res_eig_x = eigdec((Kx + Kx.T) / 2, num_eig)
    eig_Kx = res_eig_x[0] if isinstance(res_eig_x, tuple) else res_eig_x

    res_eig_y = eigdec((Ky + Ky.T) / 2, num_eig)
    eig_Ky = res_eig_y[0] if isinstance(res_eig_y, tuple) else res_eig_y

    # Stack eigenvalues for product computation
    eig_Kx_2d = eig_Kx.reshape(-1, 1)
    eig_Ky_2d = eig_Ky.reshape(1, -1)
    eig_prod = eig_Kx_2d @ eig_Ky_2d

    thresh = 1e-6
    max_prod = np.max(eig_prod)
    II = eig_prod > max_prod * thresh
    eig_prod = eig_prod[II]

    # Bootstrap for critical value
    T_BS = 5000
    if len(eig_prod) * T < 1e6:
        f_rand1 = np.random.chisquare(1, (len(eig_prod), T_BS))
        Null_dstr = (eig_prod.reshape(-1, 1) / T) * f_rand1
        Null_dstr = np.sum(Null_dstr, axis=0)

        sort_Null = np.sort(Null_dstr)
        Cri = sort_Null[int(np.ceil((1 - alpha) * T_BS))]
        p_val = np.mean(Null_dstr > Sta)
    else:
        # Fallback for large eigenvalue products
        Cri = -1
        p_val = -1

    # Gamma approximation
    mean_appr = np.trace(Kx) * np.trace(Ky) / T
    var_appr = 2 * np.trace(Kx @ Kx) * np.trace(Ky @ Ky) / (T ** 2)

    if mean_appr > 0 and var_appr > 0:
        k_appr = mean_appr ** 2 / var_appr
        theta_appr = var_appr / mean_appr
        Cri_appr = stats.gamma.ppf(1 - alpha, a=k_appr, scale=theta_appr)
        p_appr = 1 - stats.gamma.cdf(Sta, a=k_appr, scale=theta_appr)
    else:
        Cri_appr = -1
        p_appr = -1

    return Sta, Cri, p_val, Cri_appr, p_appr
