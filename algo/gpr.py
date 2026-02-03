"""
Gaussian Process Regression

Part of GPML toolbox.
"""

import numpy as np
from scipy import linalg
from .covariance import cov_sum


def gpr(logtheta, covfunc, x, y, xstar=None):
    """
    Gaussian Process Regression
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    n, D = x.shape
    
    # 训练模式：计算负对数边缘似然 (Negative Log Marginal Likelihood)
    if xstar is None:
        K = cov_sum(covfunc, logtheta, x)
        
        # --- 修复 1: 增加数值稳定性 (Jitter) ---
        # 如果矩阵不是正定，Cholesky 会失败。我们在对角线加一点点噪声。
        jitter = 1e-6 * np.eye(n)
        K = K + jitter
        
        try:
            L = linalg.cholesky(K, lower=True)
        except linalg.LinAlgError:
            # 如果还是失败，尝试更大的 jitter
            jitter = 1e-4 * np.eye(n)
            K = K + jitter
            try:
                L = linalg.cholesky(K, lower=True)
            except linalg.LinAlgError:
                # 实在不行返回一个很大的 loss，让优化器跳过这个参数
                return 1e9

        alpha = linalg.cho_solve((L, True), y)
        
        nlml = 0.5 * np.dot(y.T, alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
        
        # 如果 y 是多维的 (虽然这里应该是向量)，nlml 可能是数组，取标量
        if isinstance(nlml, np.ndarray):
            return nlml.item()
        return nlml

    # 预测模式
    else:
        if xstar.ndim == 1:
            xstar = xstar.reshape(-1, 1)
            
        K = cov_sum(covfunc, logtheta, x)
        
        # 同样的 Jitter 逻辑
        jitter = 1e-6 * np.eye(n)
        K = K + jitter
        
        try:
            L = linalg.cholesky(K, lower=True)
        except linalg.LinAlgError:
            # 预测时如果失败，通常意味着训练失败了，这里简单处理
            jitter = 1e-4 * np.eye(n)
            K = K + jitter
            L = linalg.cholesky(K, lower=True)

        alpha = linalg.cho_solve((L, True), y)
        
        # 获取自协方差和交叉协方差 (利用我们之前修好的 covariance.py)
        # 注意：cov_sum 现在返回 (kss, kstar)
        kss, kstar = cov_sum(covfunc, logtheta, x, xstar)
        
        mu = np.dot(kstar.T, alpha)
        
        v = linalg.solve_triangular(L, kstar, lower=True)
        s2 = kss - np.sum(v**2, axis=0)
        
        return mu, s2