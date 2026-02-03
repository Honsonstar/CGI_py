"""
KCIT - Kernel-based Conditional Independence Test
"""
import numpy as np
from .uind_test import uind_test
from .cind_test import cind_test_new_with_gp

def kcit(x, y, z=None, width=0, alpha=0.05):
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    
    # --- 关键修正 3: 强制对齐 MATLAB 的 hardcoded alpha ---
    # MATLAB KCIT.m 第 47 行使用的是 0.8，而非传入的 alpha
    # 这个 0.8 影响 Cri 的计算，虽然我们主要看 p_val，但为了绝对一致，必须对齐。
    force_alpha_for_calc = 0.8

    if z is None or (isinstance(z, np.ndarray) and z.size == 0):
        # Unconditional
        stat, cri, p_val, cri_appr, p_appr = uind_test(x, y, force_alpha_for_calc, width)
        # 判定逻辑：p_value > 0.05 (这个阈值是固定的，与 force_alpha 无关)
        ind = (p_appr > 0.05)
        return ind, stat, p_appr
    else:
        # Conditional
        if z.ndim == 1: z = z.reshape(-1, 1)
        p_appr, stat = cind_test_new_with_gp(x, y, z, force_alpha_for_calc, width)
        ind = (p_appr > 0.05)
        return ind, stat, p_appr