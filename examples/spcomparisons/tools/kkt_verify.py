"""Independent KKT verification at a settled point: least-squares
multipliers on the UNSPLIT problem. split_equalities makes the dual set
unbounded (each equality's two active one-sided rows admit any common
increment), so Ipopt's returned duals produce a meaningless stationarity
number even when a certificate exists. This computes the certificate the
solve earned."""
import numpy as np


def verify_kkt(problem, x, act_tol=1e-6):
    """Return (stationarity, feasibility, mults) with least-squares duals.

    Columns: every equality (free sign) + every ACTIVE inequality
    (|log g| <= act_tol, sign-constrained >= 0) + every variable sitting on
    a positivity floor (its bound gradient is a unit vector).
    """
    from lcsolver.solvers.ipopt.sia import _log_g
    from scipy.optimize import lsq_linear
    x = np.asarray(x, dtype=float)
    n = problem.n
    g0 = np.asarray(problem.objective.log_grad(x), dtype=float)

    cols, sign_free, feas = [], [], 0.0
    for c in problem.constraints:
        g = _log_g(c, x)
        if c.operator == '==':
            feas = max(feas, abs(g))
            cols.append(np.asarray(c.body.log_grad(x), dtype=float))
            sign_free.append(True)
        else:
            feas = max(feas, g)
            if g >= -act_tol:                      # active inequality
                cols.append(np.asarray(c.body.log_grad(x), dtype=float))
                sign_free.append(False)
    # positivity floors: x_j at the solver floor acts as an active bound
    for j in range(n):
        if x[j] <= 1.1e-9:
            e = np.zeros(n); e[j] = 1.0
            cols.append(e); sign_free.append(False)
    if not cols:
        return float(np.max(np.abs(g0))), feas, np.zeros(0)
    A = np.column_stack(cols)
    # column scaling is what makes this work at aircraft size: gradient
    # columns span ~8 decades and unscaled bounded-lsq stalls at garbage
    scale = np.maximum(np.linalg.norm(A, axis=0), 1e-12)
    As = A / scale
    lo = np.array([-np.inf if s else 0.0 for s in sign_free])
    hi = np.full(len(cols), np.inf)
    r = lsq_linear(As, -g0, bounds=(lo, hi), tol=1e-12, max_iter=3000,
                   lsq_solver="lsmr", lsmr_tol=1e-12)
    resid = g0 + As @ r.x
    return float(np.max(np.abs(resid))), feas, r.x / scale
