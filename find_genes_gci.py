"""
find_genes_gci - Find causal genes using CGI method

Python port of find_Genes_GCI.m
"""

import numpy as np
from scipy.io import loadmat
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
    n = data.shape[1] - 1
    x = data[:, -1]
    data = data[:, :n]
    data = normalize_data(data)

    if hyp is None:
        hyp = np.array([np.log(4.0), np.log(4.0), np.log(np.sqrt(0.01))])

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
                            temp_z.append([idx1[k], -1])  # Use -1 as padding (invalid index in Python)
                            non.append(idx1[j])
                            break
                    except Exception as e:
                        print(f'    Error in 1st order test (j={idx1[j]}, k={idx1[k]}): {e}')
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
            z1 = data[:, idx2[M[k][0]]]
            z2 = data[:, idx2[M[k][1]]]
            z_combined = np.column_stack([z1, z2])

            if idx2[M[k][0]] != j and idx2[M[k][1]] != j:
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
                            temp_z.append([idx2[M[k][0]], idx2[M[k][1]]])
                            non.append(j)
                    except Exception as e:
                        print(f'  Error in 2nd order test (j={j}): {e}')
                        non.append(j)
                        break

    # find genes by regression 1st
    print('--------------- find genes by regression 1st')
    pa = [i for i in range(n) if i not in non]
    l = len(pa)
    found_genes_1st = []

    if temp_res and temp_z:
        # MATLAB: tempRes is N x K where each column is a residual vector (n_samples x n_tests)
        # Python: temp_res is list of arrays, each row is a residual
        # Transpose to match MATLAB format: (n_samples, n_tests)
        temp_res = np.array(temp_res).T  # Now shape is (72, N)
        temp_z = np.array(temp_z)

        # MATLAB: [~,idz1] = intersect(tempZ(:,1),pa)
        # This returns the VALUES in tempZ(:,1) that are also in pa
        pa_set = set(pa)

        # For each row in tempZ, check if the value in col 1 is in pa
        # Then use that row's index to get tempRes (now columns after transpose)
        for row_idx in range(len(temp_z)):
            z_val = int(temp_z[row_idx, 0])
            if z_val in pa_set and row_idx < temp_res.shape[1]:
                ind = kcit(data[:, z_val], temp_res[:, row_idx],
                          np.array([[]]), alpha=alpha)[0]
                if ind and z_val not in found_genes_1st:
                    found_genes_1st.append(z_val)

        # Same for column 2
        for row_idx in range(len(temp_z)):
            z_val = int(temp_z[row_idx, 1])
            if z_val != -1 and z_val in pa_set and row_idx < temp_res.shape[1]:
                ind = kcit(data[:, z_val], temp_res[:, row_idx],
                          np.array([[]]), alpha=alpha)[0]
                if ind and z_val not in found_genes_1st:
                    found_genes_1st.append(z_val)

    print(f'  Found genes by 1st order: {len(found_genes_1st)}')

    # find genes by regression 2nd
    print('--------------- find genes by regression 2nd')
    found_genes_2nd = []

    if l >= 2:
        M2 = list(combinations(range(l), 2))
        lenM = len(M2)

        for p in range(lenM):
            print(f'  Processing {p+1}/{lenM}')
            j = [pa[M2[p][0]], pa[M2[p][1]]]
            z = data[:, j]

            try:
                xf = fit_gpr(z, x, cov, hyp, Ncg)
                res1 = xf - x

                # Note: MATLAB has a bug - both use ind3 instead of ind3 and ind4
                ind3, _, _ = kcit(res1, z[:, 0:1], np.array([[]]), alpha=alpha)
                ind4, _, _ = kcit(res1, z[:, 1:2], np.array([[]]), alpha=alpha)

                if ind3 and j[0] not in found_genes_2nd:
                    found_genes_2nd.append(j[0])
                if ind4 and j[1] not in found_genes_2nd:
                    found_genes_2nd.append(j[1])
            except Exception as e:
                print(f'  Error in regression 2nd order (p={p+1}/{lenM}): {e}')

    print(f'  Found genes by 2nd order: {len(found_genes_2nd)}')

    # results - sort for deterministic output (MATLAB's unique returns sorted)
    found_genes = sorted(set(found_genes_1st + found_genes_2nd))

    return {
        'non': non,
        'pa': pa,
        'found_genes': found_genes,
        'found_genes_1st': found_genes_1st,
        'found_genes_2nd': found_genes_2nd
    }


def load_data(mat_file: str) -> np.ndarray:
    data = loadmat(mat_file)
    for key in ['d', 'data', 'D', 'normalized']:
        if key in data:
            return data[key]
    raise ValueError(f'Could not find data variable in {mat_file}')
