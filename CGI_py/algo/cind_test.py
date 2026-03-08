"""
Conditional Independence Test using HSIC and GP
"""
import numpy as np
from .uind_test import uind_test
from .gpr import gpr
from scipy.optimize import minimize 

def cind_test_new_with_gp(x, y, z, alpha=0.8, width=0): # 默认改为 0.8
    # ... (前面的 GP 拟合逻辑保持不变，只需修改最后一行) ...
    
    n = x.shape[0]
    if width == 0:
        width = 0.8 if n < 200 else (0.5 if n < 1200 else 0.3)
    
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    z = (z - np.mean(z, axis=0)) / np.std(z, axis=0)

    # GP 1 (x|z)
    hyp_x = np.array([np.log(1.0), np.log(1.0), np.log(0.1)])
    covfunc = ['covSum', ['covSEiso', 'covNoise']]
    def nlml_x(theta):
        val = gpr(theta, covfunc, z, x)
        if isinstance(val, (tuple, list, np.ndarray)) and np.size(val) > 1: return val[0]
        return val
    res_x = minimize(nlml_x, hyp_x, method='L-BFGS-B', options={'disp':False})
    pred_x = gpr(res_x.x, covfunc, z, x, z)
    res_x_val = x - (pred_x[0].flatten() if isinstance(pred_x, tuple) else pred_x.flatten())

    # GP 2 (y|z)
    hyp_y = np.array([np.log(1.0), np.log(1.0), np.log(0.1)])
    def nlml_y(theta):
        val = gpr(theta, covfunc, z, y)
        if isinstance(val, (tuple, list, np.ndarray)) and np.size(val) > 1: return val[0]
        return val
    res_y = minimize(nlml_y, hyp_y, method='L-BFGS-B', options={'disp':False})
    pred_y = gpr(res_y.x, covfunc, z, y, z)
    res_y_val = y - (pred_y[0].flatten() if isinstance(pred_y, tuple) else pred_y.flatten())

    # --- 关键修正：传递 alpha (应该是 0.8) ---
    stat, cri, p_val, cri_appr, p_appr = uind_test(res_x_val, res_y_val, alpha, width)

    return p_appr, stat

# 别名
cind_test = cind_test_new_with_gp