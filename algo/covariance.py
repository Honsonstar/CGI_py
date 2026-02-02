"""
Covariance functions for Gaussian Process Regression

Part of GPML (Gaussian Process for Machine Learning) toolbox.
These functions are ported from the MATLAB GPML implementation.

(C) Copyright 2006 by Carl Edward Rasmussen
"""

import numpy as np
from .dist2 import dist2


def cov_sum(covfunc, logtheta, x, z=None):
    """
    covSum - Compose a covariance function as the sum of other functions.

    This function does bookkeeping and calls other covariance functions.
    For more help on design of covariance functions, try "help covFunctions".

    Usage:
        cov_sum(covfunc, logtheta, x) -> A      # compute covariance matrix
        cov_sum(covfunc, logtheta, x, z) -> A   # compute test set covariances
        cov_sum(covfunc, logtheta, x, None, j)  # compute derivative w.r.t. j-th hyperparameter
    """
    n = x.shape[0]

    # Parse covfunc to get number of hyperparameters for each component
    def get_n_hyp(cov_name):
        if 'covSEiso' in cov_name:
            return 2  # length scale + variance
        elif 'covSEard' in cov_name:
            return x.shape[1] + 1  # D length scales + variance
        elif 'covNoise' in cov_name:
            return 1  # noise variance
        elif 'covMatern' in cov_name:
            return 2
        else:
            return 2  # default

    if isinstance(covfunc, str):
        covfunc = [covfunc]

    # Count hyperparameters for each component
    n_hyp_list = []
    for cf in covfunc:
        if isinstance(cf, (list, tuple)):
            cf_name = cf[0]
            if cf_name == 'covSum':
                # Nested covSum
                n_hyp_list.extend([2] * (len(cf) - 1))  # simplified
            else:
                n_hyp_list.append(get_n_hyp(cf_name))
        else:
            n_hyp_list.append(get_n_hyp(cf))

    if z is None:
        # Compute covariance matrix
        A = np.zeros((n, n))
        start_idx = 0
        for i, cf in enumerate(covfunc):
            n_hyp = n_hyp_list[i]
            theta_i = logtheta[start_idx:start_idx + n_hyp]
            start_idx += n_hyp

            if isinstance(cf, (list, tuple)):
                cf_name = cf[0]
                sub_covfunc = cf[1] if len(cf) > 1 else None
            else:
                cf_name = cf
                sub_covfunc = None

            if 'covSEiso' in cf_name:
                A += cov_se_iso(theta_i, x)
            elif 'covSEard' in cf_name:
                A += cov_se_ard(theta_i, x)
            elif 'covNoise' in cf_name:
                A += cov_noise(theta_i, x)
            elif 'covMatern' in cf_name:
                A += cov_matern(theta_i, x, nu=3)
            else:
                A += cov_se_iso(theta_i, x)

        return A
    else:
        # Test set covariances
        m = z.shape[0]
        A = np.zeros((m,))
        B = np.zeros((n, m))
        start_idx = 0
        for i, cf in enumerate(covfunc):
            n_hyp = n_hyp_list[i]
            theta_i = logtheta[start_idx:start_idx + n_hyp]
            start_idx += n_hyp

            if isinstance(cf, (list, tuple)):
                cf_name = cf[0]
            else:
                cf_name = cf

            if 'covSEiso' in cf_name:
                aa, bb = cov_se_iso(theta_i, x, z)
            elif 'covSEard' in cf_name:
                aa, bb = cov_se_ard(theta_i, x, z)
            elif 'covNoise' in cf_name:
                aa, bb = cov_noise(theta_i, x, z)
            else:
                aa, bb = cov_se_iso(theta_i, x, z)

            A += aa
            B += bb

        return A, B


def cov_se_iso(logtheta, x, z=None):
    """
    Squared Exponential (RBF) covariance function with isotropic length scale.

    cov = exp(-0.5 * (r/length_scale)^2) * variance

    Args:
        logtheta: [log(length_scale), log(sqrt(variance))]
        x: Training inputs (n, d)
        z: Test inputs (m, d) or None

    Returns:
        K: Covariance matrix (n, n) or (n, m)
    """
    if z is None:
        z = x

    if np.isscalar(logtheta):
        logtheta = np.array([logtheta])

    length_scale = np.exp(logtheta[0])
    variance = np.exp(2 * logtheta[1])

    r2 = dist2(x, z)
    K = variance * np.exp(-0.5 * r2 / (length_scale ** 2))

    return K


def cov_se_ard(logtheta, x, z=None):
    """
    Squared Exponential (RBF) covariance function with ARD length scales.

    Args:
        logtheta: [log(length_scale_1), ..., log(length_scale_D), log(sqrt(variance))]
        x: Training inputs (n, d)
        z: Test inputs (m, d) or None

    Returns:
        K: Covariance matrix
    """
    if z is None:
        z = x

    D = x.shape[1]
    length_scales = np.exp(logtheta[:D])
    variance = np.exp(2 * logtheta[D])

    # Scale inputs by length scales
    x_scaled = x / length_scales
    z_scaled = z / length_scales

    r2 = dist2(x_scaled, z_scaled)
    K = variance * np.exp(-0.5 * r2)

    return K


def cov_noise(logtheta, x, z=None):
    """
    Noise covariance function (diagonal matrix with noise variance).

    Args:
        logtheta: [log(sqrt(noise_variance))]
        x: Inputs (n, d)
        z: Test inputs or None

    Returns:
        K: Noise covariance matrix
    """
    noise_var = np.exp(2 * logtheta[0])

    if z is None:
        return np.eye(x.shape[0]) * noise_var
    else:
        if x.shape[0] != z.shape[0]:
            raise ValueError("x and z must have same number of rows")
        return np.zeros((x.shape[0], z.shape[0]))


def cov_matern(logtheta, x, z=None, nu=3):
    """
    Matern covariance function.

    Args:
        logtheta: [log(length_scale), log(sqrt(variance))]
        x: Inputs
        z: Test inputs or None
        nu: Matern parameter (1, 3, or 5)
    """
    if z is None:
        z = x

    length_scale = np.exp(logtheta[0])
    variance = np.exp(2 * logtheta[1])

    r = np.sqrt(dist2(x, z)) * np.sqrt(2 * nu) / length_scale

    if nu == 1:
        K = variance * (1 + r) * np.exp(-r)
    elif nu == 3:
        K = variance * (1 + r + r ** 2 / 3) * np.exp(-r)
    elif nu == 5:
        K = variance * (1 + r + r ** 2 * 2 / 5 + r ** 3 / 15) * np.exp(-r)
    else:
        K = variance * np.exp(-0.5 * r ** 2)  # Fallback to SE

    return K


def cov_rq(logtheta, x, z=None):
    """
    Rational Quadratic covariance function.

    cov = (1 + r^2/(2*alpha*length_scale^2))^(-alpha) * variance
    """
    if z is None:
        z = x

    length_scale = np.exp(logtheta[0])
    variance = np.exp(2 * logtheta[1])
    alpha = np.exp(logtheta[2])

    r2 = dist2(x, z)
    K = variance * (1 + r2 / (2 * alpha * length_scale ** 2)) ** (-alpha)

    return K


def cov_linear(logtheta, x, z=None):
    """
    Linear covariance function.

    cov = variance * x @ x.T
    """
    variance = np.exp(2 * logtheta[0])

    if z is None:
        return variance * (x @ x.T)
    else:
        return variance * (x @ z.T)
