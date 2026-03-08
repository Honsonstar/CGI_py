"""
MINIMIZE - Minimize a differentiable multivariate function

Uses the Polack-Ribiere flavour of conjugate gradients with a line search
using quadratic and cubic polynomial approximations and the Wolfe-Powell
stopping criteria.

Copyright (C) 2001 - 2006 by Carl Edward Rasmussen (2006-09-08)
"""

import numpy as np


def minimize(X, f, length, *args, **kwargs):
    """
    Minimize a differentiable multivariate function.

    Usage: X, fX, i = minimize(X, f, length, P1, P2, P3, ...)

    Args:
        X: Starting point (D by 1)
        f: Function that returns (value, gradient)
        length: If positive, max number of line searches. If negative,
                max number of function evaluations.
        *args: Additional arguments passed to f

    Returns:
        X: Solution point
        fX: Vector of function values
        i: Number of iterations
    """
    # Constants
    INT = 0.1
    EXT = 3.0
    MAX = 20
    RATIO = 10
    SIG = 0.1
    RHO = SIG / 2

    # Parse length
    if np.isscalar(length):
        red = 1.0
        length = abs(length)
    else:
        red = length[1]
        length = abs(length[0])

    # Initialize
    i = 0
    ls_failed = False
    f0, df0 = f(X, *args, **kwargs)
    fX = np.array([f0])
    s = -df0
    d0 = -np.dot(s, s)
    x3 = red / (1 - d0) if (1 - d0) != 0 else red

    while i < length:
        i += 1 if length > 0 else 0

        X0, F0, dF0 = X.copy(), f0, df0.copy()

        if length > 0:
            M = MAX
        else:
            M = min(MAX, -length - i)

        # Extrapolation phase
        x2 = 0
        f2 = f0
        d2 = d0
        f3 = f0
        df3 = df0.copy()
        success = False

        while not success and M > 0:
            M -= 1
            i += 1 if length < 0 else 0
            try:
                f3, df3 = f(X + x3 * s, *args, **kwargs)
                if np.isnan(f3) or np.isinf(f3) or \
                   np.any(np.isnan(df3)) or np.any(np.isinf(df3)):
                    raise ValueError('NaN or Inf detected')
                success = True
            except:
                x3 = (x2 + x3) / 2

        if f3 < F0:
            X0 = X + x3 * s
            F0 = f3
            dF0 = df3.copy()

        d3 = np.dot(df3, s)

        if d3 > SIG * d0 or f3 > f0 + x3 * RHO * d0 or M == 0:
            break

        # Cubic extrapolation
        x1, f1, d1 = x2, f2, d2
        x2, f2, d2 = x3, f3, d3

        A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1)
        B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1)

        denom = B + np.sqrt(max(0, B * B - A * d1 * (x2 - x1)))
        if np.isreal(denom):
            x3 = x1 - d1 * (x2 - x1) ** 2 / denom
        else:
            x3 = x2 * EXT

        # Bounds on extrapolation
        if x3 < 0 or np.isnan(x3) or np.isinf(x3):
            x3 = x2 * EXT
        elif x3 > x2 * EXT:
            x3 = x2 * EXT
        elif x3 < x2 + INT * (x2 - x1):
            x3 = x2 + INT * (x2 - x1)

        # Interpolation phase
        while (abs(d3) > -SIG * d0 or f3 > f0 + x3 * RHO * d0) and M > 0:
            if d3 > 0 or f3 > f0 + x3 * RHO * d0:
                x4, f4, d4 = x3, f3, d3
            else:
                x2, f2, d2 = x3, f3, d3

            if f4 > f0:
                x3 = x2 - (0.5 * d2 * (x4 - x2) ** 2) / (f4 - f2 - d2 * (x4 - x2))
            else:
                A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2)
                B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2)
                x3 = x2 + (np.sqrt(B * B - A * d2 * (x4 - x2) ** 2) - B) / A

            x3 = max(min(x3, x4 - INT * (x4 - x2)), x2 + INT * (x4 - x2))

            try:
                f3, df3 = f(X + x3 * s, *args, **kwargs)
                if f3 < F0:
                    X0 = X + x3 * s
                    F0, dF0 = f3, df3.copy()
                M -= 1
                i += 1 if length < 0 else 0
                d3 = np.dot(df3, s)
            except:
                x3 = (x2 + x4) / 2

        # Line search succeeded
        if abs(d3) < -SIG * d0 and f3 < f0 + x3 * RHO * d0:
            X = X + x3 * s
            f0, fX = f3, np.append(fX, f3)

            # Polack-Ribiere CG direction
            s = (np.dot(df3, df3) - np.dot(df0, df3)) / max(np.dot(df0, df0), 1e-10) * s - df3
            df0 = df3.copy()
            d3 = d0
            d0 = np.dot(df0, s)

            if d0 > 0:
                s = -df0
                d0 = -np.dot(s, s)

            x3 = x3 * min(RATIO, d3 / max(d0, 1e-10))
            ls_failed = False
        else:
            X, f0, df0 = X0.copy(), F0, dF0.copy()
            if ls_failed or i > abs(length):
                break
            s = -df0
            d0 = -np.dot(s, s)
            x3 = 1 / (1 - d0) if (1 - d0) != 0 else 1.0
            ls_failed = True

    return X, fX, i
