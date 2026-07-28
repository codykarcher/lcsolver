"""A gpkit solver plugin that solves each GP with IPOPT instead of cvxopt.

Why this exists
---------------
SPaircraft's D8 model compiles to ~1200 free variables in ~4700 posynomial
inequalities. cvxopt's ``solvers.gp`` forms a *dense* KKT system and dies on
it: first with ``Rank(A) < p`` (its linear-equality block, built from gpkit's
monomial equalities, is rank deficient), and then -- with ``use_leqs=False``,
which routes those equalities through the LSE block instead -- with a plain
non-convergence. The published SPaircraft results came from MOSEK, which is
not installed here and needs a license.

So we swap only the numerics. gpkit still assembles the model, still runs the
sequential-GP loop, still packs and checks the result; this plugin receives the
same ``(c, A, k)`` triple cvxopt would have and returns the same four keys.
The gpkit *model* -- the thing an EDI rebuild is verified against -- is
untouched, which is the whole point: a reference solution is only worth
anything if the reference model is unmodified.

The transform
-------------
With ``y = log(x)`` a GP is convex:

    minimize    log sum_{j in cost}      exp(log c_j + a_j . y)
    subject to  log sum_{j in posy i}    exp(log c_j + a_j . y) <= 0

which is what gets built here, one Pyomo expression per posynomial. The outer
log is kept (rather than the equivalent ``sum exp(...) <= 1``) purely for
scaling: cost monomials here run to 1e5 and up, and the log keeps every
residual IPOPT sees within a couple of orders of magnitude of 1.

Duals
-----
gpkit wants ``la``, the posynomial sensitivities, in log space -- which is
exactly the multiplier on ``h_i(y) <= 0`` above, so IPOPT's constraint duals
come back unmodified. ``la[0]`` is the cost's own sensitivity and is always
1.0 by construction. Sign conventions differ between IPOPT builds, so the
magnitude is taken.

Warm start
----------
gpkit's SGP loop calls this once per outer iteration with a slightly different
A each time, and does not thread an initial guess through. The previous
primal is cached module-level and reused, which is what makes the outer loop
converge in minutes rather than not at all.
"""
from __future__ import annotations

import numpy as np
import pyomo.environ as pyo
from gpkit.exceptions import UnknownInfeasible

# Cache of the last primal, keyed by problem width. The SGP loop changes A
# between calls but not the variable set, so the previous optimum is an
# excellent start for the next subproblem.
_LAST: dict[int, np.ndarray] = {}

# log(x) is confined to this box. Physical quantities live nowhere near the
# edges -- gpkit's own Bounded() wrapper uses 1e-30..1e30 -- so this only
# stops IPOPT from wandering into exp() overflow on early iterations.
_YMAX = 100.0


def reset_cache():
    _LAST.clear()


def optimize(*, c, A, k, meq_idxs=None, p_idxs=None, verbose=False, **kwargs):
    """Solve one GP. Signature and return value match gpkit.solvers.cvxopt."""
    del meq_idxs, p_idxs, kwargs  # only needed by solvers with an equality block

    A = A.tocsr()
    n_mon, n_var = A.shape
    log_c = np.log(np.asarray(c, dtype=float))
    k = list(k)

    # Row spans: group 0 is the cost, groups 1.. are the constraints.
    starts = np.cumsum([0] + k)
    if starts[-1] != n_mon:
        raise UnknownInfeasible(
            f"monomial count mismatch: sum(k)={starts[-1]} but A has {n_mon} rows")

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n_var - 1)
    y0 = _LAST.get(n_var)
    m.y = pyo.Var(m.I, bounds=(-_YMAX, _YMAX),
                  initialize=(lambda _m, i: float(y0[i])) if y0 is not None else 0.0)

    indptr, indices, data = A.indptr, A.indices, A.data

    def lse(lo, hi):
        """log sum_{j in [lo,hi)} exp(log c_j + a_j . y), as a Pyomo expression."""
        terms = []
        for j in range(lo, hi):
            s, e = indptr[j], indptr[j + 1]
            expo = log_c[j] + sum(float(data[t]) * m.y[int(indices[t])]
                                  for t in range(s, e))
            terms.append(pyo.exp(expo))
        if len(terms) == 1:
            # A monomial: log(exp(z)) is z. Keeping it symbolic rather than
            # letting IPOPT round-trip it through exp/log costs nothing and
            # makes the ~half of these constraints that are monomial exactly
            # linear, which IPOPT exploits.
            j = lo
            s, e = indptr[j], indptr[j + 1]
            return log_c[j] + sum(float(data[t]) * m.y[int(indices[t])]
                                  for t in range(s, e))
        return pyo.log(sum(terms))

    m.obj = pyo.Objective(expr=lse(starts[0], starts[1]), sense=pyo.minimize)

    n_con = len(k) - 1
    m.C = pyo.RangeSet(0, n_con - 1) if n_con else pyo.RangeSet(0, -1)
    if n_con:
        def _rule(_m, i):
            return lse(starts[i + 1], starts[i + 2]) <= 0.0
        m.con = pyo.Constraint(m.C, rule=_rule)

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    opt = pyo.SolverFactory("ipopt")
    opt.options["tol"] = 1e-9
    opt.options["constr_viol_tol"] = 1e-9
    opt.options["max_iter"] = 3000
    # The log-space GP is well scaled already; IPOPT's gradient-based rescaling
    # only muddies the duals gpkit reads back out.
    opt.options["nlp_scaling_method"] = "none"
    res = opt.solve(m, tee=verbose, load_solutions=False)

    tc = res.solver.termination_condition
    if tc not in (pyo.TerminationCondition.optimal,
                  pyo.TerminationCondition.locallyOptimal,
                  pyo.TerminationCondition.feasible):
        raise UnknownInfeasible(f"ipopt terminated with {tc}")
    m.solutions.load_from(res)

    y = np.array([pyo.value(m.y[i]) for i in range(n_var)])
    _LAST[n_var] = y

    la = np.ones(len(k))
    if n_con:
        for i in range(n_con):
            la[i + 1] = abs(m.dual.get(m.con[i], 0.0))

    return dict(status="optimal",
                objective=float(np.exp(pyo.value(m.obj))),
                primal=y,          # log space, as gpkit expects
                la=la)


# gpkit looks up solvers by name off gpkit.solvers; registering here lets a
# caller pass solver=ipopt_gp.optimize directly, which is what we do.
optimize.name = "ipopt_gp"
