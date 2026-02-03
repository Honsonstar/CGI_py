"""
FIT_GPR - Fit Gaussian Process Regression model

Fits a GP with RBF kernel to the data pairs (X, Y).
"""

import numpy as np
from .gpr import gpr
# 关键修改：移除旧的 minimize，使用 scipy 标准优化器
from scipy.optimize import minimize

def fit_gpr(X: np.ndarray, Y: np.ndarray, cov: str = 'covSEiso',
            hyp: np.ndarray = None, Ncg: int = 100) -> np.ndarray:
    """
    Fit a Gaussian Process Regression model.
    """
    # --- 修复 1: 维度修正 (防止 X should be Nxd 报错) ---
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    X = np.atleast_2d(X)
    Y = Y.flatten()

    n = X.shape[0]
    # 双重保险：防止维度转置错误
    if n != Y.shape[0] and X.shape[1] == Y.shape[0]:
        X = X.T
        n = X.shape[0]

    if Y.shape[0] != n:
        raise ValueError(f'X should be Nxd and Y should be Nx1. Got X:{X.shape}, Y:{Y.shape}')

    # --- 修复 2: 超参数初始化 (配合 find_genes_gci 的修改) ---
    if hyp is None:
        # 默认值使用 log(4.0)，匹配 MATLAB 逻辑
        hyp = np.array([np.log(4.0), np.log(4.0), np.log(np.sqrt(0.01))])

    # 准备优化参数
    if 'covSEard' in cov:
        D = X.shape[1]
        hyp_adjusted = np.zeros(D + 2)
        hyp_adjusted[:D] = hyp[0] 
        hyp_adjusted[D] = hyp[1]
        hyp_adjusted[D + 1] = hyp[2]
    else:
        hyp_adjusted = np.array([hyp[0], hyp[1], hyp[2]])

    covfunc = ['covSum', [cov, 'covNoise']]

    # --- 修复 3: 定义目标函数 (处理返回值) ---
    def nlml(theta):
        val = gpr(theta, covfunc, X, Y)
        # 防御性编程：如果 gpr 返回了元组，只取第一个值 (Likelihood)
        if isinstance(val, (tuple, list, np.ndarray)) and np.size(val) > 1:
            return val[0]
        return val

    # --- 修复 4: 使用 Scipy 优化器 (解决 cannot unpack/too many values 报错) ---
    # L-BFGS-B 不需要手动提供梯度，会自动数值估算
    res = minimize(nlml, hyp_adjusted.copy(), method='L-BFGS-B', 
                   options={'maxiter': Ncg, 'disp': False})
    
    hyp_opt = res.x

    # --- 修复 5: 获取预测值 (处理返回值结构) ---
    # gpr 预测时返回 (mu, S2)，我们需要 mu ([0])
    prediction_result = gpr(hyp_opt, covfunc, X, Y, X)
    
    if isinstance(prediction_result, tuple):
        Yfit = prediction_result[0]
    else:
        Yfit = prediction_result

    return Yfit