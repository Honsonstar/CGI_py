"""
Covariance functions for Gaussian Process Regression

Part of GPML (Gaussian Process for Machine Learning) toolbox.
"""

import numpy as np
from .dist2 import dist2


def cov_sum(covfunc, logtheta, x, z=None):
    """
    covSum - Compose a covariance function as the sum of other functions.
    """
    n = x.shape[0]

    # --- 修复核心：解包 covSum 结构 ---
    # 输入通常是 ['covSum', ['covSEiso', 'covNoise']]
    # 我们需要将其解包为 ['covSEiso', 'covNoise']
    if isinstance(covfunc, (list, tuple)) and len(covfunc) > 1 and covfunc[0] == 'covSum':
        covfunc = covfunc[1]
    
    # 如果只是单个字符串，转为列表
    if isinstance(covfunc, str):
        covfunc = [covfunc]

    # Helper to determine number of hyperparameters
    def get_n_hyp(cov_name):
        if 'covSEiso' in cov_name:
            return 2
        elif 'covSEard' in cov_name:
            return x.shape[1] + 1
        elif 'covNoise' in cov_name:
            return 1
        elif 'covMatern' in cov_name:
            return 2
        else:
            return 2

    # Count hyperparameters
    n_hyp_list = []
    for cf in covfunc:
        if isinstance(cf, (list, tuple)):
            cf_name = cf[0]
            # 如果里面还嵌套了 covSum (递归情况)，这里简单处理
            if cf_name == 'covSum': 
                # 这里简化处理，假设没有深层嵌套
                pass
            else:
                n_hyp_list.append(get_n_hyp(cf_name))
        else:
            n_hyp_list.append(get_n_hyp(cf))

    if z is None:
        # --- Training Mode (返回一个矩阵) ---
        A = np.zeros((n, n))
        start_idx = 0
        for i, cf in enumerate(covfunc):
            n_hyp = n_hyp_list[i]
            # 安全切片
            theta_i = logtheta[start_idx:start_idx + n_hyp]
            start_idx += n_hyp

            if isinstance(cf, (list, tuple)):
                cf_name = cf[0]
            else:
                cf_name = cf

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
        # --- Test Mode (返回两个值) ---
        m = z.shape[0]
        A = np.zeros((m,))   # 自协方差
        B = np.zeros((n, m)) # 交叉协方差
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
            elif 'covMatern' in cf_name:
                aa, bb = cov_matern(theta_i, x, z, nu=3)
            else:
                aa, bb = cov_se_iso(theta_i, x, z)

            A += aa
            B += bb

        return A, B


def cov_se_iso(logtheta, x, z=None):
    """ Squared Exponential (Isotropic) """
    if np.isscalar(logtheta):
        logtheta = np.array([logtheta])

    length_scale = np.exp(logtheta[0])
    # 确保 logtheta 长度足够，否则说明上层切片错了
    if len(logtheta) < 2:
        raise IndexError(f"cov_se_iso expected 2 params, got {len(logtheta)}. Check cov_sum logic.")
        
    variance = np.exp(2 * logtheta[1])

    if z is None:
        r2 = dist2(x, x)
        K = variance * np.exp(-0.5 * r2 / (length_scale ** 2))
        return K
    else:
        r2 = dist2(x, z)
        K_cross = variance * np.exp(-0.5 * r2 / (length_scale ** 2))
        K_self = np.full(z.shape[0], variance)
        return K_self, K_cross


def cov_se_ard(logtheta, x, z=None):
    """ Squared Exponential (ARD) """
    D = x.shape[1]
    length_scales = np.exp(logtheta[:D])
    variance = np.exp(2 * logtheta[D])

    if z is None:
        x_scaled = x / length_scales
        r2 = dist2(x_scaled, x_scaled)
        K = variance * np.exp(-0.5 * r2)
        return K
    else:
        x_scaled = x / length_scales
        z_scaled = z / length_scales
        r2 = dist2(x_scaled, z_scaled)
        K_cross = variance * np.exp(-0.5 * r2)
        K_self = np.full(z.shape[0], variance)
        return K_self, K_cross


def cov_noise(logtheta, x, z=None):
    """ Noise Covariance """
    if len(logtheta) < 1:
        # Default value if something goes wrong, but shouldn't happen
        noise_var = 1e-6 
    else:
        noise_var = np.exp(2 * logtheta[0])

    if z is None:
        return np.eye(x.shape[0]) * noise_var
    else:
        m = z.shape[0]
        K_cross = np.zeros((x.shape[0], m))
        K_self = np.full(m, noise_var)
        return K_self, K_cross


def cov_matern(logtheta, x, z=None, nu=3):
    """ Matern Covariance """
    length_scale = np.exp(logtheta[0])
    variance = np.exp(2 * logtheta[1])

    def calc_k(dist_sq):
        r = np.sqrt(dist_sq) * np.sqrt(2 * nu) / length_scale
        if nu == 1:
            return variance * (1 + r) * np.exp(-r)
        elif nu == 3:
            return variance * (1 + r + r ** 2 / 3) * np.exp(-r)
        elif nu == 5:
            return variance * (1 + r + r ** 2 * 2 / 5 + r ** 3 / 15) * np.exp(-r)
        else:
            return variance * np.exp(-0.5 * r ** 2)

    if z is None:
        r2 = dist2(x, x)
        return calc_k(r2)
    else:
        r2 = dist2(x, z)
        K_cross = calc_k(r2)
        K_self = np.full(z.shape[0], variance)
        return K_self, K_cross