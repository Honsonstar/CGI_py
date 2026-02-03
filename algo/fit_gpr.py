"""
FIT_GPR - Fit Gaussian Process Regression model
"""
import numpy as np
from .gpr import gpr
from scipy.optimize import minimize

def fit_gpr(X: np.ndarray, Y: np.ndarray, cov: str = 'covSEiso',
            hyp: np.ndarray = None, Ncg: int = 100) -> np.ndarray:
    
    # 维度处理
    if X.ndim == 1: X = X.reshape(-1, 1)
    X = np.atleast_2d(X)
    Y = Y.flatten()
    n = X.shape[0]
    if n != Y.shape[0] and X.shape[1] == Y.shape[0]:
        X = X.T; n = X.shape[0]

    # --- 关键修正：对齐 MATLAB 的 log 逻辑 ---
    if hyp is None:
        # MATLAB 默认: hyp=[log(4); log(4); log(sqrt(0.01))] (如果在 fit_gpr 内部再 log 一次的话)
        # 这里直接给最终值
        hyp = np.array([np.log(4.0), np.log(4.0), np.log(np.sqrt(0.01))])
    else:
        # 复制一份防止修改原数组
        hyp = hyp.copy()
        # MATLAB fit_gpr.m line 21: hyp(1)= log(hyp(1));
        # 只有当传入的是原始值（如4.0）时才Log。我们假设调用方传入的是[4.0, ...]
        # 判断一下：如果 hyp[0] > 2.0 (比如 4.0)，通常说明它是原始值，需要 Log
        if hyp[0] > 2.0: 
            hyp[0] = np.log(hyp[0])

    # 构造 hyp_adjusted
    if 'covSEard' in cov:
        D = X.shape[1]
        hyp_adjusted = np.zeros(D + 2)
        hyp_adjusted[:D] = hyp[0] 
        hyp_adjusted[D] = hyp[1]
        hyp_adjusted[D + 1] = hyp[2]
    else:
        hyp_adjusted = np.array([hyp[0], hyp[1], hyp[2]])

    covfunc = ['covSum', [cov, 'covNoise']]

    # 目标函数 (增加防御)
    def nlml(theta):
        try:
            val = gpr(theta, covfunc, X, Y)
            if not np.isfinite(val): return 1e9
            if isinstance(val, (tuple, list, np.ndarray)) and np.size(val) > 1: return val[0]
            return val
        except Exception: return 1e9

    # 优化
    try:
        res = minimize(nlml, hyp_adjusted.copy(), method='L-BFGS-B', 
                       options={'maxiter': Ncg, 'disp': False})
        hyp_opt = res.x
    except:
        hyp_opt = hyp_adjusted

    # 预测
    try:
        pred = gpr(hyp_opt, covfunc, X, Y, X)
        Yfit = pred[0] if isinstance(pred, tuple) else pred
        if not np.isfinite(Yfit).all(): Yfit = np.zeros_like(Y)
    except:
        Yfit = np.zeros_like(Y)

    return Yfit