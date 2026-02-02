"""
CGI (Causality Graphical Inference) - Python Implementation
A causal inference tool using Gaussian processes and kernel-based independence tests.

Ported from MATLAB implementation by Kun Zhang and Jonas Peters.

References:
- Zhang, K., Peters, J., Janzing, D., & Schölkopf, B. (2011).
  Kernel-based conditional independence test and application in causal discovery.
  arXiv:1202.2775
"""

from .kernel import kernel, kernel_matrix
from .dist2 import dist2
from .eigdec import eigdec
from .pdinv import pdinv
from .stack import stack
from .solve_chol import solve_chol
from .minimize import minimize
from .covariance import cov_sum, cov_se_iso, cov_se_ard, cov_noise, cov_matern
from .gpr import gpr, gpr_multi
from .medbw import medbw
from .paco_test import paco_test
from .kcit import kcit
from .uind_test import uind_test
from .cind_test import cind_test_new_with_gp
from .fit_gpr import fit_gpr

__version__ = '1.0.0'
__author__ = 'CGI Authors'

__all__ = [
    'kernel', 'kernel_matrix', 'dist2', 'eigdec', 'pdinv', 'stack',
    'solve_chol', 'minimize', 'cov_sum', 'cov_se_iso', 'cov_se_ard',
    'cov_noise', 'cov_matern', 'gpr', 'gpr_multi', 'medbw', 'paco_test',
    'kcit', 'uind_test', 'cind_test_new_with_gp', 'fit_gpr'
]
