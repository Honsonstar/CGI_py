"""
FIT_GPR - Fit Gaussian Process Regression model

Fits a GP with RBF kernel to the data pairs (X, Y).

Copyright (c) 2008-2010 Joris Mooij
"""

import numpy as np
from .gpr import gpr
from .minimize import minimize
from .covariance import cov_sum


def fit_gpr(X: np.ndarray, Y: np.ndarray, cov: str = 'covSEiso',
            hyp: np.ndarray = None, Ncg: int = 100) -> np.ndarray:
    """
    Fit a Gaussian Process Regression model.

    Args:
        X: (n, d) matrix of training inputs
        Y: (n, 1) or (n,) matrix of training targets
        cov: covariance function ('covSEiso' or 'covSEard')
        hyp: initial hyperparameters [length_scale; log_variance; log_noise]
        Ncg: number of conjugate gradient iterations

    Returns:
        Yfit: fitted Y values at X
    """
    X = np.atleast_2d(X)
    Y = Y.flatten()

    n = X.shape[0]

    if Y.shape[0] != n:
        raise ValueError('X should be Nxd and Y should be Nx1')

    if hyp is None:
        hyp = np.array([4.0, np.log(4.0), np.log(np.sqrt(0.01))])

    # Adjust hyperparameters based on covariance type
    if 'covSEard' in cov:
        # ARD: length scale for each dimension
        D = X.shape[1]
        hyp_adjusted = np.zeros(D + 2)
        hyp_adjusted[:D] = np.log(np.exp(hyp[0]) * np.ones(D))
        hyp_adjusted[D] = hyp[1]
        hyp_adjusted[D + 1] = hyp[2]
    else:
        # Isotropic: single length scale
        hyp_adjusted = np.array([np.log(np.exp(hyp[0])), hyp[1], hyp[2]])

    # Covariance function with noise
    covfunc = ['covSum', [cov, 'covNoise']]

    # Optimize hyperparameters
    def nlml(theta):
        return gpr(theta, covfunc, X, Y)

    hyp_opt, _, _ = minimize(hyp_adjusted.copy(), nlml, -Ncg)

    # Get predictions
    Yfit = gpr(hyp_opt, covfunc, X, Y, X)[0]

    return Yfit
