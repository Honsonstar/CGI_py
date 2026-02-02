"""
CInd_test_new_withGP - Conditional Independence Test with GP regression

Tests if x and y are conditionally independent given z using
GP regression for residual computation.

Copyright (c) 2010-2011
"""

import numpy as np
from scipy import stats
from scipy import linalg
from .kernel import kernel
from .eigdec import eigdec
from .pdinv import pdinv
from .medbw import medbw
from .gpr import gpr_multi
from .minimize import minimize
from .covariance import cov_sum


def cind_test_new_with_gp(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          alpha: float = 0.8, width: float = 0) -> tuple:
    """
    Test conditional independence using GP regression.

    Args:
        x: (n,) array of samples for X
        y: (n,) array of samples for Y
        z: (n, d) array of conditioning variables
        alpha: significance level
        width: kernel width (0 for automatic)

    Returns:
        Sta: test statistic
        Cri: critical value from bootstrap
        p_val: p-value from bootstrap
        Cri_appr: critical value from Gamma approximation
        p_appr: p-value from Gamma approximation
        ind: True if independent, False otherwise
    """
    x = x.flatten()
    y = y.flatten()
    T = len(y)  # sample size

    # Parameters
    IF_GP = True
    Approximate = True
    Bootstrap = True
    Thresh = 1e-5

    # Normalize data
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    z = (z - np.mean(z, axis=0)) / (np.std(z, axis=0) + 1e-10)

    D = z.shape[1]

    # Automatic width selection using median heuristic
    if width == 0:
        width = np.sqrt(2) * medbw(x.reshape(-1, 1), 1000)

    theta = 1 / (width ** 2)

    # Centering matrix
    H = np.eye(T) - np.ones((T, T)) / T

    # Compute kernel matrices
    z_half = z / 2
    Kx, _ = kernel(np.column_stack([x, z_half]),
                   np.column_stack([x, z_half]), np.array([theta, 1]))
    Ky, _ = kernel(y.reshape(-1, 1), y.reshape(-1, 1), np.array([theta, 1]))

    Kx = H @ Kx @ H
    Ky = H @ Ky @ H

    if IF_GP:
        # Compute eigenvalues for GP regression
        max_eig_x = min(400, T // 4)
        max_eig_y = min(200, T // 5)

        eig_Kx, eix = eigdec((Kx + Kx.T) / 2, max_eig_x)
        eig_Ky, eiy = eigdec((Ky + Ky.T) / 2, max_eig_y)

        # Filter small eigenvalues
        IIx = eig_Kx > max(eig_Kx) * Thresh
        eig_Kx = eig_Kx[IIx]
        eix = eix[:, IIx]

        IIy = eig_Ky > max(eig_Ky) * Thresh
        eig_Ky = eig_Ky[IIy]
        eiy = eiy[:, IIy]

        # Set up covariance function
        covfunc = ['covSum', ['covSEard', 'covNoise']]
        logtheta0 = np.concatenate([np.log(width) * np.ones(D), [0, np.log(np.sqrt(0.1))]])

        # Normalize eigenvalues
        scale_x = 2 * np.sqrt(T) * eix @ np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0])
        scale_y = 2 * np.sqrt(T) * eiy @ np.diag(np.sqrt(eig_Ky)) / np.sqrt(eig_Ky[0])

        # Optimize hyperparameters for X
        def nlml_x(theta):
            return gpr_multi(theta, covfunc, z, scale_x)

        logtheta_x, _, _ = minimize(logtheta0.copy(), nlml_x, -500)

        # Optimize hyperparameters for Y
        def nlml_y(theta):
            return gpr_multi(theta, covfunc, z, scale_y)

        logtheta_y, _, _ = minimize(logtheta0.copy(), nlml_y, -500)

        # Compute kernel matrices for z
        covfunc_z = ['covSEard']
        Kz_x = cov_sum(covfunc_z, logtheta_x, z)
        Kz_y = cov_sum(covfunc_z, logtheta_y, z)

        # Compute conditional kernel matrices (residuals)
        noise_x = np.exp(2 * logtheta_x[-1])
        noise_y = np.exp(2 * logtheta_y[-1])

        P1_x = np.eye(T) - Kz_x @ pdinv(Kz_x + noise_x * np.eye(T))
        Kxz = P1_x @ Kx @ P1_x.T

        P1_y = np.eye(T) - Kz_y @ pdinv(Kz_y + noise_y * np.eye(T))
        Kyz = P1_y @ Ky @ P1_y.T

        # Test statistic
        Sta = np.trace(Kxz @ Kyz)

        # Degrees of freedom
        df_x = np.trace(np.eye(T) - P1_x)
        df_y = np.trace(np.eye(T) - P1_y)
    else:
        # Non-GP version (simplified)
        Kz, _ = kernel(z, z, np.array([theta, 1]))
        Kz = H @ Kz @ H
        lam = 1e-3

        P1 = np.eye(T) - Kz @ pdinv(Kz + lam * np.eye(T))
        Kxz = P1 @ Kx @ P1.T
        Kyz = P1 @ Ky @ P1.T

        Sta = np.trace(Kxz @ Kyz)
        df_x = df_y = np.trace(np.eye(T) - P1)

    # Compute eigenvalues of the product
    num_eig = T
    eig_Kxz, eivx = eigdec((Kxz + Kxz.T) / 2, num_eig)
    eig_Kyz, eivy = eigdec((Kyz + Kyz.T) / 2, num_eig)

    IIx = eig_Kxz > max(eig_Kxz) * Thresh
    IIy = eig_Kyz > max(eig_Kyz) * Thresh
    eig_Kxz = eig_Kxz[IIx]
    eivx = eivx[:, IIx]
    eig_Kyz = eig_Kyz[IIy]
    eivy = eivy[:, IIy]

    # Compute products of eigenvector components
    eiv_prodx = eivx @ np.diag(np.sqrt(eig_Kxz))
    eiv_prody = eivy @ np.diag(np.sqrt(eig_Kyz))

    num_eigx = eiv_prodx.shape[1]
    num_eigy = eiv_prody.shape[1]
    Size_u = num_eigx * num_eigy

    uu = np.zeros((T, Size_u))
    for i in range(num_eigx):
        for j in range(num_eigy):
            uu[:, i * num_eigy + j] = eiv_prodx[:, i] * eiv_prody[:, j]

    if Size_u > T:
        uu_prod = uu @ uu.T
    else:
        uu_prod = uu.T @ uu

    # Bootstrap for critical value
    T_BS = 5000
    Cri, p_val = -1, -1

    if Bootstrap:
        eig_uu, _ = eigdec(uu_prod, min(T, Size_u))
        II_f = eig_uu > max(eig_uu) * Thresh
        eig_uu = eig_uu[II_f]

        if len(eig_uu) * T < 1e6:
            f_rand1 = np.random.chisquare(1, (len(eig_uu), T_BS))
            Null_dstr = eig_uu.reshape(-1, 1) / T * f_rand1
            Null_dstr = np.sum(Null_dstr, axis=0)

            sort_Null = np.sort(Null_dstr)
            Cri = sort_Null[int(np.ceil((1 - alpha) * T_BS))]
            p_val = np.mean(Null_dstr > Sta)

    # Gamma approximation
    Cri_appr, p_appr = -1, -1
    if Approximate:
        mean_appr = np.trace(uu_prod)
        var_appr = 2 * np.trace(uu_prod @ uu_prod)

        if mean_appr > 0 and var_appr > 0:
            k_appr = mean_appr ** 2 / var_appr
            theta_appr = var_appr / mean_appr
            Cri_appr = stats.gamma.ppf(1 - alpha, a=k_appr, scale=theta_appr)
            p_appr = 1 - stats.gamma.cdf(Sta, a=k_appr, scale=theta_appr)

    # Determine independence
    if Sta > Cri_appr:
        ind = False
    else:
        ind = True

    return Sta, Cri, p_val, Cri_appr, p_appr, ind
