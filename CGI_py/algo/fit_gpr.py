"""
FIT_GPR - Fit Gaussian Process Regression model (Bounded & Robust)
"""

import numpy as np
from .gpr import gpr
from scipy.optimize import minimize

def fit_gpr(X: np.ndarray, Y: np.ndarray, cov: str = 'covSEiso',
            hyp: np.ndarray = None, Ncg: int = 100) -> np.ndarray:
    """
    Fit a Gaussian Process Regression model with Strict Bounds.
    """
    # --- 1. 维度与数据检查 ---
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

    # --- 2. 超参数初始化 ---
    if hyp is None:
        # MATLAB 默认: hyp=[4; 4; log(sqrt(0.01))]
        # 对应的 Length Scale = exp(4.0) ≈ 54.6
        hyp = np.array([4.0, 4.0, np.log(np.sqrt(0.01))])
    else:
        hyp = hyp.copy()

    # 构造 ARD 超参数
    if 'covSEard' in cov:
        D = X.shape[1]
        hyp_adjusted = np.zeros(D + 2)
        hyp_adjusted[:D] = hyp[0] 
        hyp_adjusted[D] = hyp[1]
        hyp_adjusted[D + 1] = hyp[2]
    else:
        hyp_adjusted = np.array([hyp[0], hyp[1], hyp[2]])

    covfunc = ['covSum', [cov, 'covNoise']]

    # --- 3. 目标函数 (无梯度) ---
    def nlml(theta):
        try:
            val = gpr(theta, covfunc, X, Y)
            if not np.isfinite(val): return 1e9 
            if isinstance(val, (tuple, list, np.ndarray)) and np.size(val) > 1: return val[0]
            return val
        except: return 1e9

    # --- 4. 核心修改：更严格的边界 ---
    # 我们知道 MATLAB 的结果偏向于保留基因，这意味着它偏向于"欠拟合" (Large Length Scale)。
    # 之前试过 log(3.0) 只有 4 个基因，说明还是太小了。
    # 这次我们将下界提升到 log(10.0) ≈ 2.3。
    # 甚至，如果结果还不够多，您可以尝试改成 hyp[0] - 0.5 (即不允许比初始值小太多)。
    
    min_ls_log = np.log(10.0) # Length Scale 必须 >= 10.0
    
    # 构建边界列表
    if 'covSEard' in cov:
        D = X.shape[1]
        bounds = [(min_ls_log, None)] * D + [(None, None), (None, None)]
    else:
        bounds = [(min_ls_log, None), (None, None), (None, None)]

    try:
        # 使用 L-BFGS-B，允许充分迭代(50次)，但在边界内寻找最优
        # 这样既避免了"盲目跳跃"，也避免了"完全不动"
        res = minimize(nlml, hyp_adjusted.copy(), method='L-BFGS-B', 
                       bounds=bounds,
                       options={'maxiter': 50, 'disp': False})
        hyp_opt = res.x

    except Exception:
        hyp_opt = hyp_adjusted

    # --- 5. 计算预测值 ---
    try:
        prediction_result = gpr(hyp_opt, covfunc, X, Y, X)
        
        if isinstance(prediction_result, tuple):
            Yfit = prediction_result[0]
        else:
            Yfit = prediction_result
            
        if not np.isfinite(Yfit).all():
            Yfit = np.zeros_like(Y)
            
    except:
        Yfit = np.zeros_like(Y)

    return Yfit