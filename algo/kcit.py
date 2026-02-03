"""
KCIT - Kernel-based Conditional Independence Test
"""
import numpy as np
from .uind_test import uind_test
from .cind_test import cind_test_new_with_gp

def kcit(x, y, z=None, width=0, alpha=0.05): # 虽然这里接收 0.05...
    
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    
    # --- 关键修正：强制使用 alpha = 0.8 以匹配 MATLAB KCIT.m ---
    # MATLAB KCIT.m line 47: UInd_test(..., 0.8, pars.width)
    force_alpha = 0.8

    if z is None or (isinstance(z, np.ndarray) and z.size == 0):
        # Unconditional
        stat, cri, p_val, cri_appr, p_appr = uind_test(x, y, force_alpha, width)
        # MATLAB KCIT.m: if pval > 0.05 ind=1 else ind=0 (Independence)
        # 注意：MATLAB 的 ind=1 表示独立(Independence)。
        # 但 find_Genes_GCI.m 的逻辑是：if KCIT(...) -> non=[non, i] (加入非因果列表)
        # 所以如果 KCIT 返回 1 (独立)，则它是非因果。
        
        # 我们的 kcit 需要返回什么？
        # MATLAB: return ind (1=Indep, 0=Dep)
        # Python find_genes_gci.py: if ind2: non.append(i)
        # 所以我们需要返回 True (独立) 或 False (相关)
        
        ind = (p_appr > 0.05) # 这里 0.05 是判断阈值，那个 0.8 是计算临界值的参数
        return ind, stat, p_appr
    else:
        # Conditional
        if z.ndim == 1: z = z.reshape(-1, 1)
        # 同样传递 force_alpha = 0.8
        p_appr, stat = cind_test_new_with_gp(x, y, z, force_alpha, width)
        ind = (p_appr > 0.05)
        return ind, stat, p_appr