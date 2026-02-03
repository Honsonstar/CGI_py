"""
UInd_test - Unconditional Independence Test using HSIC
"""

import numpy as np
from scipy import stats
from .kernel import kernel
from .eigdec import eigdec

def uind_test(x: np.ndarray, y: np.ndarray, alpha: float = 0.8, width: float = 0) -> tuple:
    x = x.flatten().reshape(-1, 1)
    y = y.flatten().reshape(-1, 1)
    T = len(y)

    # --- 关键修正 2: 恢复 MATLAB 的硬编码宽度逻辑 ---
    if width == 0:
        if T < 200:
            width = 0.8
        elif T < 1200:
            width = 0.5
        else:
            width = 0.3

    # theta = 1/width^2
    theta = 1.0 / (width ** 2)

    # 标准化 (复现 MATLAB 行为)
    x = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y = (y - np.mean(y)) / (np.std(y) + 1e-10)

    H = np.eye(T) - np.ones((T, T)) / T

    # 计算核矩阵
    res_x = kernel(x, x, np.array([theta, 1.0]))
    Kx = res_x[0] if isinstance(res_x, (tuple, list)) else res_x

    res_y = kernel(y, y, np.array([theta, 1.0]))
    Ky = res_y[0] if isinstance(res_y, (tuple, list)) else res_y

    Kx = H @ Kx @ H
    Ky = H @ Ky @ H

    Sta = np.trace(Kx @ Ky)

    num_eig = min(T // 2, 100)
    
    res_ex = eigdec((Kx + Kx.T) / 2, num_eig)
    eig_Kx = res_ex[0] if isinstance(res_ex, (tuple, list)) else res_ex

    res_ey = eigdec((Ky + Ky.T) / 2, num_eig)
    eig_Ky = res_ey[0] if isinstance(res_ey, (tuple, list)) else res_ey

    eig_prod = (eig_Kx.reshape(-1, 1) @ eig_Ky.reshape(1, -1)).flatten()
    eig_prod = eig_prod[eig_prod > np.max(eig_prod) * 1e-6]

    mean_appr = np.trace(Kx) * np.trace(Ky) / T
    var_appr = 2 * np.trace(Kx @ Kx) * np.trace(Ky @ Ky) / (T ** 2)

    if mean_appr > 0 and var_appr > 0:
        k_appr = mean_appr ** 2 / var_appr
        theta_appr = var_appr / mean_appr
        # 使用传入的 alpha 计算临界值
        Cri_appr = stats.gamma.ppf(1 - alpha, a=k_appr, scale=theta_appr)
        p_appr = 1 - stats.gamma.cdf(Sta, a=k_appr, scale=theta_appr)
    else:
        Cri_appr = 0
        p_appr = 1.0

    return Sta, Cri_appr, p_appr, Cri_appr, p_appr