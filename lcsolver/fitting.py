#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Fitting black-box data with functions a geometric program can hold.

A constraint that exists only as a callback has to be linearized, and a
linearization is a local model: it is valid near the current iterate, it may be
optimistic, and so it drags a trust region and a step-rejection test along with
it. A constraint that is a *posynomial* needs none of that -- it goes into the
convex subproblem exactly as written.

So the useful thing to do with sampled black-box data is not to interpolate it
accurately but to express it in a form the subproblem can hold exactly.
Max-affine and softmax-affine functions are that form: their logarithms are
convex, which is precisely the condition for a geometric program
(Hoburg, Kirschen and Boyd, *Optimization and Engineering* 17(4), 2016).

Why a plain fit is not enough
-----------------------------
Fitting accurately is not the same as fitting *safely*. A least-squares fit
sits above the data about half the time and below it the other half. Where it
sits below, the subproblem is exactly representing a constraint that is easier
than the truth, and the optimizer will happily walk into the gap: the returned
point can be infeasible for the real problem by the size of the fitting error,
with nothing in the solve to say so. The subproblem is convex, the solve is
exact, the certificate is valid -- for the wrong function.

The models in this repository already acknowledge this. Every one of them
inflates its fit by ``mfac = 1 + rms_err`` before using it, which is a scalar
safety margin applied after the fact. That is the right instinct implemented as
a heuristic: it is neither guaranteed (the worst-case error can exceed the RMS
error) nor tight (most of the domain is inflated more than it needs to be).

A conservative fit
------------------
:func:`fit_max_affine` can instead constrain the fit to lie on one side of the
data:

    conservative='upper'   the fit is >= every sample
    conservative='lower'   the fit is <= every sample
    conservative=None      ordinary least squares (the default)

For a constraint used as ``w >= fit(u)`` -- which is how a GP-compatible fit
enters a model, and how every fit in this repository is used -- ``'upper'`` is
the safe side: the surrogate demands at least as much as the truth does, so a
point feasible for the surrogate is feasible for the truth at every sampled
input. That is the same relationship the AGM condensation has to a signomial
constraint, and it earns the same guarantee: the subproblem becomes an inner
approximation, and the solver's certificate transfers to the original problem.

The price is a deliberately biased fit and therefore a conservative optimum.
:func:`fit_max_affine` reports the bias it introduced so the trade is visible
rather than assumed.

What this does not do
---------------------
The guarantee is at the *sampled points*. A convex function that upper-bounds a
sample set can still dip below the true function between samples if the truth
is not itself convex in log space. Sampling density is the user's
responsibility, and :func:`fit_report` exists to make the residual distribution
easy to look at.
"""
from __future__ import annotations

import numpy as np

__all__ = ["fit_max_affine", "evaluate_fit", "fit_report", "fit_constraints"]


def _design(u):
    """``[u | 1]`` -- the affine design matrix for log-inputs ``u``."""
    u = np.atleast_2d(np.asarray(u, dtype=float))
    return np.hstack([u, np.ones((u.shape[0], 1))])


def _ls(A, w):
    return np.linalg.lstsq(A, w, rcond=None)[0]


def _ls_one_sided(A, w, side):
    """Least squares subject to ``A z >= w`` (or ``<=``), on the given rows.

    Small and dense -- one plane at a time, ``d + 1`` unknowns -- so a generic
    SLSQP is ample. Falls back to an unconstrained fit shifted just far enough
    to satisfy the constraint, which is always feasible and never worse than
    the heuristic margin it replaces.
    """
    z0 = _ls(A, w)
    sign = 1.0 if side == 'upper' else -1.0

    def shifted():
        gap = np.max(sign * (w - A @ z0)) if len(w) else 0.0
        z = z0.copy()
        if gap > 0:
            z[-1] += sign * gap          # raise/lower the intercept
        return z

    try:
        from scipy.optimize import minimize
    except Exception:
        return shifted()

    def obj(z):
        r = A @ z - w
        return float(r @ r)

    def jac(z):
        return 2.0 * A.T @ (A @ z - w)

    cons = [{'type': 'ineq',
             'fun': (lambda z: sign * (A @ z - w)),
             'jac': (lambda z: sign * A)}]
    try:
        res = minimize(obj, shifted(), jac=jac, constraints=cons,
                       method='SLSQP', options={'maxiter': 200, 'ftol': 1e-12})
        if res.success and np.all(sign * (A @ res.x - w) >= -1e-9):
            return res.x
    except Exception:
        pass
    return shifted()


def fit_max_affine(x, y, K=2, conservative=None, max_iter=50, seed=0,
                   tol=1e-10):
    """Fit ``y = g(x)`` with a max-affine function of ``log x``.

    Parameters
    ----------
    x : array ``(m, d)`` or ``(m,)``
        Sample inputs. Strictly positive: the fit is in log space.
    y : array ``(m,)``
        Sample outputs. Strictly positive.
    K : int
        Number of affine pieces. ``K = 1`` gives a monomial fit.
    conservative : ``None``, ``'upper'`` or ``'lower'``
        Side of the data the fit is constrained to lie on. ``'upper'`` is the
        safe choice for a constraint of the form ``w >= fit(u)``; see the
        module docstring.
    max_iter : int
        Partition/refit sweeps.

    Returns
    -------
    dict
        In the same shape the models here already use --  ``ftype='MA'``,
        ``K``, ``d``, ``c``, ``e`` -- so it drops into :func:`fit_constraints`
        and into any model written against a gpfit-style fit. Diagnostics are
        added alongside: ``rms_err``, ``max_err``, ``violations`` (samples on
        the wrong side, which a conservative fit drives to zero), and
        ``bias``, the mean signed log residual, which is what conservatism
        costs.

    Notes
    -----
    The partition/refit sweep is the alternating algorithm of Magnani and Boyd:
    assign each point to the piece that is highest there, refit each piece on
    the points assigned to it, repeat. With ``conservative`` set, the refit is
    one-sided. Constraining only the assigned piece is enough for the max --
    if the piece responsible for a point is above it, the max is too.
    """
    if conservative not in (None, 'upper', 'lower', 'shift', 'shift-lower'):
        raise ValueError("conservative must be None, 'upper', 'lower', "
                         f"'shift' or 'shift-lower'; got {conservative!r}")
    # Two ways to get to one side of the data, and neither dominates:
    #
    #   'upper'  constrain each affine piece as it is fitted
    #   'shift'  fit freely, then translate the surface clear of the data
    #
    # On data that really is log-convex, 'upper' is tighter -- the pieces can
    # sit just above with little slack. On the Hoburg drag black box, which is
    # three-dimensional and not log-convex, 'upper' is far worse and gets worse
    # as pieces are added: a piece forced above its own points pokes above the
    # data everywhere else too, and the max takes the worst of them. Measured
    # there, 'shift' costs 2.5% in the optimum against 'upper' at 50%.
    #
    # So measure, do not assume. `fit_report` prints the bias for exactly this.
    shift_only = conservative in ('shift', 'shift-lower')
    side = ('upper' if conservative in ('upper', 'shift') else
            'lower' if conservative in ('lower', 'shift-lower') else None)
    fit_side = None if shift_only else side
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(y, dtype=float).ravel()
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("max-affine fitting is done in log space, so every "
                         "sample input and output must be strictly positive")
    if len(y) != x.shape[0]:
        raise ValueError(f"x has {x.shape[0]} rows but y has {len(y)}")
    m, d = x.shape
    K = max(1, min(int(K), m))

    u, w = np.log(x), np.log(y)
    A = _design(u)

    # Start from a partition along the dominant input direction, which beats a
    # random split badly on the monotone data these fits are usually built from.
    order = np.argsort(u[:, 0])
    labels = np.zeros(m, dtype=int)
    for k, chunk in enumerate(np.array_split(order, K)):
        labels[chunk] = k

    Z = np.zeros((K, d + 1))
    prev = None
    for _ in range(max_iter):
        for k in range(K):
            rows = labels == k
            if rows.sum() < d + 1:                # too few to determine a plane
                rows = np.ones(m, dtype=bool)
            Z[k] = (_ls(A[rows], w[rows]) if fit_side is None
                    else _ls_one_sided(A[rows], w[rows], fit_side))
        vals = A @ Z.T                            # (m, K)
        labels = np.argmax(vals, axis=1)
        fit = vals.max(axis=1)
        obj = float(np.sum((fit - w) ** 2))
        if prev is not None and abs(prev - obj) <= tol * max(1.0, abs(prev)):
            break
        prev = obj

    # A max-affine fit is only guaranteed one-sided if every point is actually
    # covered. The partition can leave a point below the piece it was assigned
    # to; one uniform shift fixes it and keeps the fit valid.
    resid = (A @ Z.T).max(axis=1) - w
    if side == 'upper' and resid.min() < 0:
        Z[:, -1] += -resid.min()
        resid = (A @ Z.T).max(axis=1) - w
    elif side == 'lower' and resid.max() > 0:
        Z[:, -1] -= resid.max()
        resid = (A @ Z.T).max(axis=1) - w

    wrong = (int(np.sum(resid < -1e-12)) if side == 'upper'
             else int(np.sum(resid > 1e-12)) if side == 'lower'
             else 0)
    return {
        'ftype': 'MA', 'K': K, 'd': d, 'a1': 1.0,
        'c': [float(np.exp(z[-1])) for z in Z],
        'e': [[float(v) for v in z[:-1]] for z in Z],
        'rms_err': float(np.sqrt(np.mean(resid ** 2))),
        'max_err': float(np.max(np.abs(resid))),
        'bias': float(np.mean(resid)),
        'violations': wrong,
        'conservative': conservative,
    }


def evaluate_fit(fit, x):
    """The fitted function at ``x``; the inverse of what the model constrains."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None] if fit['d'] > 1 or x.shape[0] != fit['d'] else x[None, :]
    u = np.log(x)
    A = _design(u)
    Z = np.hstack([np.asarray(fit['e'], dtype=float),
                   np.log(np.asarray(fit['c'], dtype=float))[:, None]])
    lo = (A @ Z.T)
    if fit['ftype'] == 'MA':
        w = lo.max(axis=1)
    elif fit['ftype'] == 'SMA':
        a = float(fit['a1'])
        w = np.log(np.sum(np.exp(a * lo), axis=1)) / a
    else:
        raise ValueError(f"cannot evaluate ftype {fit['ftype']!r} here")
    return np.exp(w)


def fit_report(fit, x=None, y=None):
    """A short human-readable summary, for putting next to a solve."""
    lines = [f"{fit['ftype']} fit, K={fit['K']}, d={fit['d']}"
             + (f", conservative={fit['conservative']!r}"
                if fit.get('conservative') else ", least squares")]
    lines.append(f"  rms log error   {fit['rms_err']:.4g}")
    lines.append(f"  max log error   {fit['max_err']:.4g}")
    lines.append(f"  mean bias       {fit['bias']:+.4g}"
                 + ("   (the cost of being conservative)"
                    if fit.get('conservative') else ""))
    if fit.get('conservative'):
        lines.append(f"  samples on the wrong side  {fit['violations']}"
                     + ("   <- should be zero" if fit['violations'] else ""))
    if x is not None and y is not None:
        pred = evaluate_fit(fit, x)
        r = np.log(pred) - np.log(np.asarray(y, dtype=float).ravel())
        lines.append(f"  checked on {len(r)} samples: "
                     f"min {r.min():+.4g}  max {r.max():+.4g}")
    return "\n".join(lines)


def fit_constraints(fit, ivar, dvars, mfac=None):
    """LCsolver constraints for a fit, in the form the subproblem holds exactly.

    ``ivar`` is the dependent quantity and ``dvars`` the independent ones. The
    constraint asserts ``ivar >= fit(dvars)``, which is how a GP-compatible fit
    is used: the optimizer pushes the modelled quantity down against the fit.

    ``mfac`` inflates the fit by a constant factor. It defaults to ``1`` for a
    conservative fit, which needs no margin because it already bounds the data,
    and to ``1 + rms_err`` otherwise, which is the heuristic the models in this
    repository have been using. Passing it explicitly overrides both.
    """
    if mfac is None:
        mfac = 1.0 if fit.get('conservative') else 1.0 + fit['rms_err']
    K, e, c = fit['K'], fit['e'], fit['c']
    lhs = ivar / mfac
    monos = []
    for k in range(K):
        m = c[k]
        for i, u in enumerate(dvars):
            m = m * u ** e[k][i]
        monos.append(m)

    if fit['ftype'] == 'MA':
        return [lhs >= m for m in monos]
    if fit['ftype'] == 'SMA':
        return [lhs ** fit['a1'] >= sum(monos)]
    if fit['ftype'] == 'ISMA':
        return [1 >= sum(m / lhs ** fit['a1'][k] for k, m in enumerate(monos))]
    raise ValueError(f"unknown fit type {fit['ftype']!r}")
