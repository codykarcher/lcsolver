"""Make a seed self-consistent before handing it to the solver.

The problem this solves is not distance, it is INCONSISTENCY. reference.json
is a converged optimum of a related model, so most of its values are good,
but any variable this project changed -- or any variable downstream of one --
now contradicts the rest. Fixing them by hand does not converge: pinning
w_db moved the violation onto t_db and theta_db, which are defined FROM it,
and there is a long tail of such dependents.

Gauss-Newton on the equality subsystem repairs all of them at once. The
equalities ARE the definitions -- t_db = 2 dP w_db/sigma, theta_db = w_db/R,
mac = f(taper, c_root), l_ht = f(dx_lead, y_mac, chord) -- so restoring
h(x) = 0 propagates every pinned or corrected value through to everything
that depends on it.

This is the NORMAL step of the composite-step method, used for what it is
actually good at. As a feasibility method it disappointed: it left the
inequalities untouched at e^57 because a least-norm step lands wherever it
starts. As a seed repair that is precisely the property wanted -- stay near
the reference, just stop contradicting yourself.
"""
from __future__ import annotations

import numpy as np


def _posy(posy, x, n):
    vals = np.array([c * np.prod(x ** a) for c, a in posy.terms])
    tot = vals.sum()
    if tot <= 0:
        return -700.0, np.zeros(n)
    w = vals / tot
    g = np.zeros(n)
    for wk, (_c, a) in zip(w, posy.terms):
        g += wk * np.asarray(a, dtype=float)
    return float(np.log(tot)), g


def _con(con, x, n):
    b = con.body
    if getattr(b, 'p', None) is not None:
        lp, gp = _posy(b.p, x, n)
        lq, gq = _posy(b.q, x, n)
        return lp - lq, gp - gq
    return _posy(b, x, n)


def repair(problem, x0, iters=12, tol=1e-9, verbose=False):
    """Return a copy of ``x0`` with every equality satisfied.

    Least-norm Gauss-Newton in log space, so the repaired point stays as
    close to the seed as the equalities allow.
    """
    n = problem.n
    eq = [i for i, c in enumerate(problem.constraints) if c.operator == '==']
    if not eq:
        return np.array(x0, dtype=float)
    lo = np.array([b[0] if b and b[0] else 1e-30
                   for b in (problem.bounds or [(None, None)] * n)])
    hi = np.array([b[1] if b and b[1] else 1e30
                   for b in (problem.bounds or [(None, None)] * n)])
    x = np.array(x0, dtype=float).copy()

    def res(x):
        r = np.zeros(len(eq)); J = np.zeros((len(eq), n))
        for k, i in enumerate(eq):
            r[k], J[k] = _con(problem.constraints[i], x, n)
        return r, J

    for it in range(iters):
        r, J = res(x)
        nrm = float(np.max(np.abs(r)))
        if verbose:
            print(f"    repair {it:2d}  max|h| = {nrm:.3e}")
        if nrm <= tol:
            break
        d = -np.linalg.pinv(J, rcond=1e-10) @ r
        big = np.max(np.abs(d))
        if big > 1.0:
            d *= 1.0 / big
        for alpha in (1.0, 0.5, 0.25, 0.1):
            xn = np.clip(x * np.exp(alpha * d), lo * 1.000001, hi * 0.999999)
            if float(np.max(np.abs(res(xn)[0]))) < nrm:
                x = xn
                break
        else:
            break
    return x
