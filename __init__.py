"""
CGI (Causality Graphical Inference) - Python Implementation
A causal inference tool using Gaussian processes and kernel-based independence tests.
"""

from .algo import (
    kernel, kernel_matrix, dist2, eigdec, pdinv, stack, solve_chol,
    minimize, cov_sum, cov_se_iso, cov_se_ard, cov_noise, cov_matern,
    gpr, gpr_multi, medbw, paco_test, kcit, uind_test,
    cind_test_new_with_gp, fit_gpr
)

__version__ = '1.0.0'
