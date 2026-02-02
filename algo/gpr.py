"""
GPR - Gaussian Process Regression

Part of GPML (Gaussian Process for Machine Learning) toolbox.
Ported from MATLAB GPML implementation.

(C) Copyright 2006 by Carl Edward Rasmussen
"""

import numpy as np
from scipy import linalg
from .solve_chol import solve_chol


def gpr(logtheta, covfunc, x, y, xstar=None):
    """
    Gaussian process regression.

    Usage:
        nlml = gpr(logtheta, covfunc, x, y)          # negative log marginal likelihood
        [mu, S2] = gpr(logtheta, covfunc, x, y, xstar)  # predictions

    Args:
        logtheta: (D,) column vector of log hyperparameters
        covfunc: covariance function name or list
        x: (n, D) matrix of training inputs
        y: (n, 1) or (n,) column vector of targets
        xstar: (nn, D) matrix of test inputs

    Returns:
        If xstar is None:
            nlml: negative log marginal likelihood
        Else:
            mu: predicted means (nn,)
            S2: predicted variances (nn,)
    """
    from .covariance import cov_sum

    if isinstance(covfunc, str):
        covfunc = [covfunc]

    n = x.shape[0]

    # Handle multiple outputs
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    m = y.shape[1]

    # Compute covariance matrix
    if isinstance(covfunc, list) and covfunc[0] == 'covSum':
        K = cov_sum(covfunc[1], logtheta, x)
    else:
        K = cov_sum(covfunc, logtheta, x)

    # Cholesky decomposition
    L = linalg.cholesky(K, lower=True)

    # Solve for alpha = K^{-1} y
    alpha = solve_chol(L, y)

    if xstar is None:
        # Compute negative log marginal likelihood
        nlml = 0.5 * np.sum(y * alpha) + m * np.sum(np.log(np.diag(L))) + 0.5 * m * n * np.log(2 * np.pi)
        return nlml
    else:
        # Compute predictions
        if isinstance(covfunc, list) and covfunc[0] == 'covSum':
            Kss, Kstar = cov_sum(covfunc[1], logtheta, x, xstar)
        else:
            Kss, Kstar = cov_sum(covfunc, logtheta, x, xstar)

        mu = Kstar.T @ alpha

        # Compute variance
        v = linalg.solve_triangular(L, Kstar, lower=True)
        S2 = Kss - np.sum(v * v, axis=0)

        return mu, S2


def gpr_multi(logtheta, covfunc, x, y):
    """
    Gaussian process regression for multiple output vectors.

    Similar to gpr but y can have multiple columns.

    Usage: nlml = gpr_multi(logtheta, covfunc, x, y)

    Args:
        logtheta: log hyperparameters
        covfunc: covariance function
        x: (n, D) training inputs
        y: (n, m) targets (m output vectors)

    Returns:
        nlml: negative log marginal likelihood
    """
    from .covariance import cov_sum

    n, D = x.shape
    n, m = y.shape

    if isinstance(covfunc, str):
        covfunc = [covfunc]

    # Compute covariance matrix
    K = cov_sum(covfunc, logtheta, x)

    # Cholesky decomposition
    L = linalg.cholesky(K, lower=True)

    # Solve L @ L.T @ alpha = y
    alpha = solve_chol(L, y)

    # Negative log marginal likelihood
    nlml = 0.5 * np.sum(y * alpha) + m * np.sum(np.log(np.diag(L))) + 0.5 * m * n * np.log(2 * np.pi)

    return nlml
