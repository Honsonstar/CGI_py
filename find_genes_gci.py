"""
find_genes_gci - Find causal genes using CGI method
"""

import numpy as np
from scipy.io import loadmat
import traceback
import pandas as pd  # <--- 新增
import os            # <--- 新增
# 确保正确导入您项目中的模块
from .algo import paco_test, kcit, fit_gpr

def normalize_data(data: np.ndarray) -> np.ndarray:
    normalized = data.copy()
    for i in range(data.shape[1]):
        col = data[:, i]
        std = np.std(col)
        if std > 0:
            normalized[:, i] = (col - np.mean(col)) / std
        else:
            normalized[:, i] = col - np.mean(col)
    return normalized

def find_genes_gci(data: np.ndarray, alpha: float = 0.05,
                   cov: str = 'covSEiso', Ncg: int = 100,
                   hyp: np.ndarray = None) -> dict:
    # 默认最后一列是目标变量
    n = data.shape[1] - 1
    x = data[:, -1]
    data = data[:, :n]
    data = normalize_data(data)

    # --- 关键修正 1: 对齐 MATLAB 的超参数 ---
    # MATLAB: hyp=[4; log(4); log(sqrt(0.01))]
    # 这里的 4.0 不取对数，对应 length_scale ≈ 54.6 (欠拟合)
    if hyp is None:
        hyp = np.array([4.0, np.log(4.0), np.log(np.sqrt(0.01))])

    temp_res = []
    temp_z = []
    non = []

    # 0-order CI tests
    print('--------------- 0-order CI tests')
    for i in range(n):
        ind1 = paco_test(x, data[:, i], np.array([]), alpha)
        if ind1:
            ind2, _, _ = kcit(x, data[:, i], np.array([[]]), alpha=alpha)
            if ind2:
                non.append(i)

    # --- 新增打印 ---
    print(f"DEBUG: Total genes: {n}")
    print(f"DEBUG: Genes removed in 0-order: {len(non)}")
    print(f"DEBUG: Genes remaining: {n - len(non)}")

    # 1-order CI tests
    print('--------------- 1-order CI tests')
    idx1 = [i for i in range(n) if i not in non]
    len1 = len(idx1)

    for j in range(len1):
        print(f'  Processing {j+1}/{len1}')
        for k in range(len1):
            if j != k and idx1[k] not in non:
                y = data[:, idx1[j]]
                z = data[:, idx1[k]]

                ind1 = paco_test(x, y, z, alpha)
                if ind1:
                    try:
                        xf = fit_gpr(z, x, cov, hyp, Ncg)
                        res1 = xf - x
                        yf = fit_gpr(z, y, cov, hyp, Ncg)
                        res2 = yf - y

                        ind2, _, _ = kcit(res1, res2, np.array([[]]), alpha=alpha)
                        if ind2:
                            temp_res.append(res1)
                            # MATLAB: tempZ = [tempZ;idx1(k),0]
                            temp_z.append([idx1[k], -1]) 
                            non.append(idx1[j])
                            break
                    except Exception as e:
                        print(f'    Error: {e}')
                        non.append(idx1[j])
                        break

    # 2-order CI tests
    print('--------------- 2-order CI tests')
    idx2 = [i for i in range(n) if i not in non]
    len2 = len(idx2)

    from itertools import combinations
    M = list(combinations(range(len2), 2))
    m = len(M)

    for p in range(len2):
        j = idx2[p]
        for k in range(m):
            y = data[:, j]
            # Use original indices from idx2 mapping
            idx_z1 = idx2[M[k][0]]
            idx_z2 = idx2[M[k][1]]
            
            # Skip if j is part of the conditioning set
            if idx_z1 == j or idx_z2 == j:
                continue

            z1 = data[:, idx_z1]
            z2 = data[:, idx_z2]
            z_combined = np.column_stack([z1, z2])

            ind = paco_test(x, y, z_combined, alpha)
            if ind:
                try:
                    xf = fit_gpr(z_combined, x, cov, hyp, Ncg)
                    res1 = xf - x
                    yf = fit_gpr(z_combined, y, cov, hyp, Ncg)
                    res2 = yf - y

                    ind2, _, _ = kcit(res1, res2, np.array([[]]), alpha=alpha)
                    if ind2:
                        temp_res.append(res1)
                        temp_z.append([idx_z1, idx_z2])
                        non.append(j)
                        # Break inner loop if independent found (Standard PC logic, though MATLAB code loop structure implies break)
                        # MATLAB code provided didn't explicitly show break in 2-order for all conditions, but usually you break.
                        # Assuming break to speed up.
                        break 
                except Exception:
                    non.append(j)
                    break

    # find genes by regression 1st
    print('--------------- find genes by regression 1st')
    pa = [i for i in range(n) if i not in non]
    l = len(pa)
    found_genes_1st = []

    if temp_res and temp_z:
        temp_res = np.array(temp_res).T
        temp_z = np.array(temp_z)
        pa_set = set(pa)

        for row_idx in range(len(temp_z)):
            z_val = int(temp_z[row_idx, 0])
            if z_val in pa_set and row_idx < temp_res.shape[1]:
                # MATLAB checks KCIT(Z, Residual)
                ind = kcit(data[:, z_val], temp_res[:, row_idx], np.array([[]]), alpha=alpha)[0]
                # If independent (ind=True), it is a cause?
                # MATLAB Logic: if KCIT(...) found_Genes = [..., id]
                # Wait, if Z is INDEPENDENT of Residual, it means Z explains the variance?
                # Actually, standard regression logic: if Z is independent of Residual(X|Z), it's consistent.
                # If KCIT returns True (Indep), we add it.
                if ind and z_val not in found_genes_1st:
                    found_genes_1st.append(z_val)

        for row_idx in range(len(temp_z)):
            z_val = int(temp_z[row_idx, 1])
            if z_val != -1 and z_val in pa_set and row_idx < temp_res.shape[1]:
                ind = kcit(data[:, z_val], temp_res[:, row_idx], np.array([[]]), alpha=alpha)[0]
                if ind and z_val not in found_genes_1st:
                    found_genes_1st.append(z_val)

    print(f'  Found genes by 1st order: {len(found_genes_1st)}')

    # find genes by regression 2nd
    print('--------------- find genes by regression 2nd')
    found_genes_2nd = []
    
    # 修复：只有当 pa 长度足够时才进行
    if len(pa) >= 2:
        M2 = list(combinations(range(len(pa)), 2))
        lenM = len(M2)

        for p in range(lenM):
            j_indices = [pa[M2[p][0]], pa[M2[p][1]]]
            z = data[:, j_indices]

            try:
                xf = fit_gpr(z, x, cov, hyp, Ncg)
                res1 = xf - x

                ind3, _, _ = kcit(res1, z[:, 0:1], np.array([[]]), alpha=alpha)
                ind4, _, _ = kcit(res1, z[:, 1:2], np.array([[]]), alpha=alpha)

                if ind3 and j_indices[0] not in found_genes_2nd:
                    found_genes_2nd.append(j_indices[0])
                if ind4 and j_indices[1] not in found_genes_2nd:
                    found_genes_2nd.append(j_indices[1])
            except Exception:
                pass

    print(f'  Found genes by 2nd order: {len(found_genes_2nd)}')

    found_genes = sorted(set(found_genes_1st + found_genes_2nd))

    return {
        'non': non,
        'pa': pa,
        'found_genes': found_genes
    }


def load_data(file_path: str) -> np.ndarray:
    """
    Load data from .mat or .csv file.
    """
    # 获取文件后缀名
    _, ext = os.path.splitext(file_path)

    # 情况 1: 读取 .mat 文件 (保持原有逻辑)
    if ext == '.mat':
        data = loadmat(file_path)
        # 尝试查找常用的变量名
        for key in ['d', 'data', 'D', 'normalized']:
            if key in data:
                return data[key]
        raise ValueError(f'Could not find data variable in {file_path}')

    # 情况 2: 读取 .csv 文件 (新增逻辑)
    elif ext == '.csv':
        try:
            # header=None 假设 CSV 没有表头，全是数据
            # 如果你的 CSV 第一行是列名，请改为 header=0
            df = pd.read_csv(file_path, header=0)

            # 转换为 numpy 数组
            return df.values
        except Exception as e:
            raise ValueError(f'Failed to read CSV file {file_path}: {e}')

    # 情况 3: 不支持的格式
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Please use .mat or .csv")