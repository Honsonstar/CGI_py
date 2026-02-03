"""
FIT_GPR - Fit Gaussian Process Regression model

Fits a GP with RBF kernel to the data pairs (X, Y).

Copyright (c) 2008-2010 Joris Mooij
"""

import numpy as np
from .gpr import gpr
from scipy.optimize import minimize

def fit_gpr(X: np.ndarray, Y: np.ndarray, cov: str = 'covSEiso',
            hyp: np.ndarray = None, Ncg: int = 100) -> np.ndarray:
    """
    Fit a Gaussian Process Regression model.
    """
    # 1. 维度修复
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X = np.atleast_2d(X)
    Y = Y.flatten()

    n = X.shape[0]
    if n != Y.shape[0] and X.shape[1] == Y.shape[0]:
        X = X.T
        n = X.shape[0]

    if Y.shape[0] != n:
        raise ValueError(f'X should be Nxd and Y should be Nx1. Got X:{X.shape}, Y:{Y.shape}')

    # 2. 超参数初始化 (Log化)
    if hyp is None:
        hyp = np.array([np.log(4.0), np.log(4.0), np.log(np.sqrt(0.01))])

    if 'covSEard' in cov:
        D = X.shape[1]
        hyp_adjusted = np.zeros(D + 2)
        hyp_adjusted[:D] = hyp[0]
        hyp_adjusted[D] = hyp[1]
        hyp_adjusted[D + 1] = hyp[2]
    else:
        hyp_adjusted = np.array([hyp[0], hyp[1], hyp[2]])

    covfunc = ['covSum', [cov, 'covNoise']]

    # 3. 目标函数 (只返回数值)
    def nlml(theta):
        val = gpr(theta, covfunc, X, Y)
        # 确保 val 是标量
        if isinstance(val, (tuple, list, np.ndarray)):
            return val[0]
        return val

    # 4. 使用 Scipy 优化器
    # 注意：这里不需要接收返回值，直接用 res.x
    res = minimize(nlml, hyp_adjusted.copy(), method='L-BFGS-B',
                   options={'maxiter': Ncg, 'disp': False})

    hyp_opt = res.x

    # 5. 获取预测值
    # gpr 返回 (mu, S2)，我们需要 mu (即索引 [0])
    prediction_result = gpr(hyp_opt, covfunc, X, Y, X)

    # 防御性编程：检查返回类型
    if isinstance(prediction_result, tuple):
        Yfit = prediction_result[0]
    else:
        Yfit = prediction_result

    return Yfit
