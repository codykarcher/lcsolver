#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Sensitivity of the optimum to the Constants of a formulation.

This is the LCsolver analogue of the sensitivity report that `GPkit` prints for a
geometric program: for every ``Constant`` in the model it reports how strongly
the optimal objective responds to that constant. The reported quantity is the
log-log sensitivity (an elasticity)

.. math::

    s_c \\;=\\; \\frac{d \\log f^*}{d \\log c}
          \\;=\\; \\frac{c}{f^*}\\, \\frac{d f^*}{d c}

so ``s_c = 2`` means "a 1% increase in this constant raises the optimum by
about 2%", and the sign tells you the direction. Elasticities are unitless,
which makes them comparable across constants carrying different physical units
-- the main reason this, rather than the raw derivative, is the headline number.

How it works
------------
The naive way to obtain these numbers is to perturb each constant and re-solve,
which costs ``2N`` solves for ``N`` constants and is both slow and limited by
the solver's convergence tolerance. Instead this module uses the envelope
theorem, which gives the answer *exactly* from a single solve.

Write each constraint in the form Pyomo stores it, ``body`` versus ``bound``
(the bound being ``lower`` or ``upper``), and let :math:`\\lambda_i` be the
constraint's dual. At the optimum the primal variables are stationary, so the
total derivative of the optimum with respect to a constant :math:`\\theta`
collapses to the *partial* derivatives at fixed :math:`x^*`:

.. math::

    \\frac{d f^*}{d \\theta}
      = \\frac{\\partial f}{\\partial \\theta}
      - \\sum_i \\lambda_i \\frac{\\partial (\\text{body}_i - \\text{bound}_i)}
                                  {\\partial \\theta}

Every partial derivative here is taken symbolically with Pyomo's reverse-mode
differentiation, so the result carries no truncation error whatsoever: it is the
exact derivative of the model, not a difference quotient. One reverse sweep
gives the derivatives with respect to every constant in an expression at once,
so the cost is one solve plus one walk per active constraint -- independent of
how many constants the model has.

Where the duals come from
-------------------------
The formula needs duals keyed by Pyomo constraint, and LCsolver has several solver
backends. Rather than reach into each backend's transformed problem, this module
obtains them in one of two backend-agnostic ways:

``suffix``
    If the model carries a populated ``dual`` Suffix -- which the IPOPT route
    imports directly from the solver -- those duals are used as-is.

``kkt``
    Otherwise (the cvxopt LP/QP/GP/SP backends, which solve a transformed
    problem and return only a raw vector) the duals are recovered from the
    primal solution by solving the KKT stationarity condition

    .. math:: \\nabla_x f = \\sum_{i \\in \\mathcal{A}} \\lambda_i \\nabla_x
              (\\text{body}_i - \\text{bound}_i)

    in the least-squares sense over the active set :math:`\\mathcal{A}`. This
    needs nothing but the primal solution, so it works for every backend
    uniformly and does not depend on any solver's internal row ordering.

For a signomial program (SP/SLCP) the duals describe the final convex
subproblem, so the sensitivities are a local approximation about the returned
point rather than a global statement. This is the intended reading, and it
matches how such sensitivities are used in practice; ``result['approximate']``
flags it.
"""

import collections
import math
import warnings

import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap
from pyomo.core.expr.calculus.derivatives import differentiate, Modes
from pyomo.core.expr import identify_mutable_parameters
from pyomo.common.dependencies import numpy as np


__all__ = ["sensitivities", "constraint_duals", "dual_ambiguity",
           "format_sensitivities"]


#: Relative tolerance for deciding that a constraint is binding.
#:
#: This must be looser than the accuracy of the primal solution, not tighter.
#: An interior-point solve returns a point satisfying its constraints to a few
#: parts in 1e6, so a 1e-6 test misclassifies genuinely-tight constraints as
#: slack; dropping even one of them from the stationarity system corrupts every
#: recovered dual. Measured on the aircraft GP, the recovered duals are correct
#: to 8e-6 and completely insensitive to this value anywhere in 1e-5 .. 1e-2,
#: so 1e-4 sits in the middle of a wide plateau.
ACTIVE_RTOL = 1e-4

#: Relative size of a constant's null-space component above which its
#: sensitivity is reported as undetermined rather than returned as a number.
#:
#: When the active-constraint Jacobian is rank deficient the duals are not
#: unique -- any vector from the null space can be added to them and
#: stationarity still holds. `lstsq` picks the minimum-norm member of that
#: family, silently, so an undetermined sensitivity comes back looking like an
#: ordinary answer. A sensitivity is determined exactly when the constant's
#: gradient vector over the active set is orthogonal to that null space, which
#: is what this measures.
#:
#: The value sits in an empty band rather than being tuned. Measured on
#: SPaircraft, whose active set is rank deficient by 23, the 195 constants
#: land either below 1e-8 or above 1e-4 and NOTHING falls in between; any
#: threshold inside that decade selects the same 40. Below the band is
#: orthogonality blurred by conditioning -- the scaled system has a condition
#: number of 2e17, so a true zero shows up as 1e-12 rather than 1e-16 -- and
#: above it is a real component. Hiding those 40 takes the largest reported
#: sensitivity from 315.86, which is not a credible log-log sensitivity, to
#: 1.71, which is.
DUAL_AMBIGUITY_TOL = 1e-6

#: Relative stationarity residual above which the recovered duals are reported
#: as untrustworthy. A converged solve sits several orders of magnitude below
#: this; the aircraft GP reaches 1.6e-6.
KKT_RESIDUAL_WARN = 1e-3


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _d(expr, wrt):
    """Exact partial derivative of a Pyomo expression, as a float.

    Returns 0.0 when the expression is absent or does not involve ``wrt``.
    """
    if expr is None:
        return 0.0
    if not hasattr(expr, 'is_expression_type'):
        return 0.0                              # a plain number
    try:
        deriv = differentiate(expr, wrt=wrt, mode=Modes.reverse_symbolic)
    except Exception:
        return 0.0
    try:
        return float(pyo.value(deriv))
    except Exception:
        return 0.0


def _grad(expr, wrt_list):
    """Gradient of a Pyomo expression with respect to a list of variables."""
    if expr is None or not hasattr(expr, 'is_expression_type'):
        return np.zeros(len(wrt_list))
    try:
        derivs = differentiate(expr, wrt_list=wrt_list, mode=Modes.reverse_symbolic)
    except Exception:
        return np.zeros(len(wrt_list))
    out = np.zeros(len(wrt_list))
    for i, dv in enumerate(derivs):
        try:
            out[i] = float(pyo.value(dv))
        except Exception:
            out[i] = 0.0
    return out


def _bound_of(con):
    """The bound a constraint is measured against, and whether it is an equality.

    Pyomo hoists a mutable Param out of the body and into the bound, so the
    parameter dependence of a constraint such as ``x >= p`` lives entirely in
    ``con.lower``. Both sides therefore have to be differentiated.
    """
    lower, upper = con.lower, con.upper
    if lower is not None and upper is not None:
        # Equality, or a ranged constraint. For a range, the binding side is
        # whichever the body currently sits on.
        try:
            if pyo.value(lower) == pyo.value(upper):
                return upper, True
        except Exception:
            return upper, True
        try:
            b = pyo.value(con.body)
            return (lower, False) if abs(b - pyo.value(lower)) < abs(b - pyo.value(upper)) \
                else (upper, False)
        except Exception:
            return upper, False
    if upper is not None:
        return upper, False
    return lower, False


def _residual_scale(con, variables):
    """Natural magnitude of a constraint, for a scale-aware feasibility test.

    Comparing a residual against the bound alone is useless when the bound is
    zero, which is the normal case here: LCsolver moves everything to one side, so a
    constraint reads ``body <= 0``. A weight constraint on an aircraft then has
    terms of order 1e4 N and an interior-point solver leaves a residual of order
    1e-1 -- tight to five significant figures, yet an absolute test would call
    it slack. The scale used instead is the magnitude of the largest term,
    ``max_j |x_j d(body)/dx_j|``, which is the quantity the residual should be
    judged against.
    """
    g = _grad(con.body, variables)
    xs = np.array([abs(pyo.value(v)) if v.value is not None else 0.0
                   for v in variables])
    terms = np.abs(g) * xs
    return float(terms.max()) if terms.size and terms.max() > 0 else 1.0


def _is_active(con, rtol=ACTIVE_RTOL, variables=None, scale=None):
    """True if the constraint is binding at the current point."""
    bound, is_eq = _bound_of(con)
    if is_eq:
        return True
    if bound is None:
        return False
    try:
        b, bd = pyo.value(con.body), pyo.value(bound)
    except Exception:
        return False
    if scale is None:
        scale = (_residual_scale(con, variables) if variables is not None
                 else max(1.0, abs(bd)))
    return abs(b - bd) <= rtol * max(1.0, scale)


def _param_gradient(expr, index):
    """``{name: d expr / d constant}`` for every Constant appearing in ``expr``.

    One reverse sweep answers for every constant at once, so this is called
    once per constraint rather than once per (constraint, constant) pair. The
    difference is the whole cost of the routine: an aircraft model has a couple
    of hundred constants and a couple of hundred active constraints, and the
    pairwise form walks the same expressions tens of thousands of times to
    learn -- for nearly all of them -- that the constant does not appear.
    Restricting each walk to the constants actually present is the other half:
    a constraint typically mentions two or three.
    """
    out = {}
    if expr is None or not hasattr(expr, 'is_expression_type'):
        return out                                  # a plain number
    try:
        seen, present = set(), []
        for p in identify_mutable_parameters(expr):
            if id(p) in index and id(p) not in seen:
                seen.add(id(p))
                present.append(p)
    except Exception:
        present = []
    if not present:
        return out
    try:
        derivs = differentiate(expr, wrt_list=present,
                               mode=Modes.reverse_symbolic)
    except Exception:
        return out
    for p, dv in zip(present, derivs):
        try:
            out[index[id(p)]] = float(pyo.value(dv))
        except Exception:
            pass
    return out


def _constants(model):
    """Every LCsolver Constant, as individual ParamData, keyed by name.

    An LCsolver ``Constant`` is a mutable Pyomo ``Param``; an indexed Constant
    contributes one entry per element, matching how GPkit reports vectors.
    """
    out = {}
    for p in model.component_objects(pyo.Param, active=True, descend_into=True):
        if not p.mutable:
            continue                             # an immutable Param is a literal
        for idx in p:
            pd = p[idx]
            out[pd.name] = pd
    return out


def _active_constraints(model, rtol=ACTIVE_RTOL, variables=None):
    if variables is None:
        variables = _variables(model)
    return [c for c in model.component_data_objects(pyo.Constraint, active=True,
                                                    descend_into=True)
            if _is_active(c, rtol, variables)]


def _variables(model):
    return list(model.component_data_objects(pyo.Var, active=True,
                                             descend_into=True))


def _objective(model):
    for o in model.component_data_objects(pyo.Objective, active=True,
                                          descend_into=True):
        return o
    raise ValueError("the model has no active objective")


# ---------------------------------------------------------------------------
# duals
# ---------------------------------------------------------------------------
def _duals_from_suffix(model):
    """Duals imported from the solver, if the model carries a populated Suffix."""
    suffix = getattr(model, 'dual', None)
    if suffix is None or len(suffix) == 0:
        return None
    duals = ComponentMap()
    for con in model.component_data_objects(pyo.Constraint, active=True,
                                            descend_into=True):
        if con in suffix:
            duals[con] = float(suffix[con])
    return duals if len(duals) else None


#: A KKT stationarity system, kept so that the ambiguity test can reuse it.
KKTSystem = collections.namedtuple("KKTSystem", "A_s rhs_s col_scale owners")


def _kkt_system(model, rtol=ACTIVE_RTOL):
    """Assemble the scaled stationarity system ``A_s lambda_s = rhs_s``.

    Separated from solving it because :func:`dual_ambiguity` needs the same
    matrix, and assembling it is the expensive half -- one symbolic gradient
    per active constraint.
    """
    obj = _objective(model)
    variables = _variables(model)
    if not variables:
        return None

    columns, owners = [], []
    for con in _active_constraints(model, rtol, variables):
        bound, _ = _bound_of(con)
        columns.append(_grad(con.body, variables) - _grad(bound, variables))
        owners.append(con)

    # Active variable bounds participate in stationarity too.
    for i, v in enumerate(variables):
        try:
            val = pyo.value(v)
        except Exception:
            continue
        for bnd in (v.lb, v.ub):
            if bnd is None:
                continue
            if abs(val - bnd) <= rtol * max(1.0, abs(bnd)):
                e = np.zeros(len(variables))
                e[i] = 1.0
                columns.append(e)
                owners.append(None)              # discarded afterwards

    if not columns:
        return None

    A = np.column_stack(columns)
    rhs = _grad(obj.expr, variables)

    # Equilibrate before solving. In an engineering model the variables span
    # many orders of magnitude (a Reynolds number next to a skin-friction
    # coefficient), which makes the raw stationarity system badly conditioned
    # and the recovered duals meaningless. Scaling each row by its variable's
    # value turns the system into one about *relative* changes, where every
    # entry is O(1); scaling each column by the constraint's own magnitude does
    # the same across constraints. Both are undone afterwards, so the duals
    # returned are in natural units.
    xs = np.array([abs(pyo.value(v)) if v.value is not None else 0.0
                   for v in variables])
    row_scale = np.where(xs > 0, xs, 1.0)
    col_scale = np.abs(A * row_scale[:, None]).max(axis=0)
    col_scale = np.where(col_scale > 0, col_scale, 1.0)

    return KKTSystem(A_s=A * row_scale[:, None] / col_scale[None, :],
                     rhs_s=rhs * row_scale, col_scale=col_scale, owners=owners)


def _duals_from_kkt(model, rtol=ACTIVE_RTOL):
    """Recover duals from the primal solution via KKT stationarity.

    Solves ``grad f = sum_i lambda_i grad(body_i - bound_i)`` in the least-squares
    sense over the active set, in the same sign convention Pyomo's ``dual``
    Suffix uses. Active variable bounds are included as columns so that
    stationarity can actually be met; their multipliers are then discarded,
    since a variable bound is a number and cannot depend on a Constant.
    """
    system = _kkt_system(model, rtol)
    duals = ComponentMap()
    if system is None:
        return duals
    A_s, rhs_s, col_scale, owners = (system.A_s, system.rhs_s,
                                     system.col_scale, system.owners)

    try:
        lam_s, *_ = np.linalg.lstsq(A_s, rhs_s, rcond=None)
    except np.linalg.LinAlgError:
        return duals
    lam = lam_s / col_scale

    # How well the recovered duals actually satisfy stationarity, measured in
    # the scaled (relative) space. A large value means the active set is wrong
    # or the point is not a local optimum -- either way the sensitivities built
    # from these duals are not trustworthy, so record it rather than let the
    # failure pass silently.
    denom = np.linalg.norm(rhs_s)
    resid = float(np.linalg.norm(A_s @ lam_s - rhs_s) / (denom if denom > 0 else 1.0))
    _duals_from_kkt.last_residual = resid

    for owner, value in zip(owners, lam):
        if owner is not None:
            duals[owner] = float(value)
    return duals


def dual_ambiguity(model, rtol=ACTIVE_RTOL, system=None, constants=None):
    """How undetermined each constant's sensitivity is, as a relative size.

    Returns ``{name: r}`` where ``r`` is the fraction of the constant's
    active-set gradient vector that lies in the null space of the stationarity
    system. ``r == 0`` means the sensitivity is the same for every valid choice
    of duals; ``r > 0`` means it is not determined by the problem at all, and
    the number that comes back is an artefact of which dual vector was picked.

    A degenerate active set is the normal state of an engineering model, not a
    pathology -- SPaircraft carries 23 degrees of freedom -- so this is
    reported rather than treated as a failure.
    """
    if system is None:
        system = _kkt_system(model, rtol)
    if system is None:
        return {}
    if constants is None:
        constants = _constants(model)
    index = {id(pd): n for n, pd in constants.items()}

    A_s, col_scale, owners = system.A_s, system.col_scale, system.owners
    try:
        _u, sv, Vt = np.linalg.svd(A_s, full_matrices=True)
    except np.linalg.LinAlgError:
        return {}
    tol = max(A_s.shape) * (sv[0] if len(sv) else 0.0) * np.finfo(float).eps
    rank = int((sv > tol).sum())
    if rank >= A_s.shape[1]:
        return dict.fromkeys(constants, 0.0)     # full rank: nothing ambiguous
    null_basis = Vt[rank:].T

    # The constant's gradient over the active set. A variable bound is a
    # number and cannot depend on a Constant, so those columns stay zero.
    grads = {name: np.zeros(len(owners)) for name in constants}
    for i, con in enumerate(owners):
        if con is None:
            continue
        bound, _ = _bound_of(con)
        g = _param_gradient(con.body, index)
        for name, dv in _param_gradient(bound, index).items():
            g[name] = g.get(name, 0.0) - dv
        for name, dv in g.items():
            grads[name][i] = dv

    out = {}
    for name, d in grads.items():
        u = d / col_scale
        norm = float(np.linalg.norm(u))
        out[name] = (0.0 if norm == 0.0
                     else float(np.linalg.norm(null_basis.T @ u) / norm))
    return out


def constraint_duals(model, method='auto', rtol=ACTIVE_RTOL):
    """Duals for every active constraint, keyed by Pyomo constraint.

    Parameters
    ----------
    model : Formulation
        A model holding a solution.
    method : {'auto', 'suffix', 'kkt'}
        Where to get the duals. ``'auto'`` prefers a populated ``dual`` Suffix
        (the IPOPT route) and falls back to KKT recovery from the primal
        solution (the cvxopt backends).
    rtol : float
        Relative tolerance for deciding whether a constraint is binding.
    """
    if method not in ('auto', 'suffix', 'kkt'):
        raise ValueError("method must be 'auto', 'suffix', or 'kkt'; "
                         f"got {method!r}")
    if method in ('auto', 'suffix'):
        duals = _duals_from_suffix(model)
        if duals is not None:
            return duals
        if method == 'suffix':
            raise RuntimeError(
                "the model carries no populated 'dual' Suffix. Solve with the "
                "IPOPT route, or use method='kkt' to recover duals from the "
                "primal solution.")
    return _duals_from_kkt(model, rtol)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def _fd_sensitivities(model, fstar, normalized=True, rel_step=0.01,
                      abs_step=1e-6, solve_fn=None):
    """Central-difference sensitivities by re-solving the model per Constant.

    Duals are never consulted, so this is immune to the degenerate-active-set
    failure of KKT recovery. Cost: two solves per Constant. Requires that the
    solve backend writes the solution back onto the model (every LCsolver backend
    does), and leaves the model re-solved at the baseline on exit.
    """
    if solve_fn is None:
        from lcsolver.solvers.solver import cvxopt_solve as solve_fn
    obj = _objective(model)
    sens = {}
    for name, pd in _constants(model).items():
        c0 = float(pyo.value(pd))
        h = abs(c0) * rel_step if c0 != 0.0 else abs_step
        try:
            pd.set_value(c0 + h)
            solve_fn(model)
            f_up = float(pyo.value(obj))
            pd.set_value(c0 - h)
            solve_fn(model)
            f_dn = float(pyo.value(obj))
        except Exception:
            pd.set_value(c0)
            solve_fn(model)                       # restore before propagating
            raise
        pd.set_value(c0)
        dfdc = (f_up - f_dn) / (2.0 * h)
        if normalized:
            sens[name] = dfdc * c0 / fstar if fstar != 0.0 else float('nan')
        else:
            sens[name] = dfdc
    solve_fn(model)                               # leave baseline solution
    return {'sensitivities': sens, 'objective': fstar, 'normalized': normalized,
            'method': 'fd', 'approximate': False}


def sensitivities(model, normalized=True, method='auto', rtol=ACTIVE_RTOL,
                  duals=None, approximate=None, check_ambiguity=True):
    """Sensitivity of the optimum to every Constant in the model.

    Parameters
    ----------
    model : Formulation
        A model that already holds a solution. Every LCsolver backend writes the
        solution back onto the model, so this is the state after ``solve(f)``.
    normalized : bool
        When True (the default) report the log-log sensitivity
        ``d log f* / d log c``, which is unitless and therefore comparable
        across constants. When False report the raw derivative ``d f* / d c``,
        which carries units of ``[objective]/[constant]``.
    method : {'auto', 'suffix', 'kkt', 'fd'}
        How to obtain the constraint duals; see :func:`constraint_duals`.
        ``'fd'`` bypasses duals entirely and central-differences the optimum
        with respect to each Constant by re-solving the model. It is the slow,
        assumption-free fallback for the degenerate-active-set case in which
        dual recovery is ambiguous (large stationarity residual): exactness of
        the dual route is traded for ~2 extra solves per Constant. The model is
        left holding the baseline solution afterwards.
    rtol : float
        Relative tolerance for the active-set test.
    duals : ComponentMap, optional
        Precomputed duals, to avoid recovering them again.
    approximate : bool, optional
        Marks the result as a local approximation. Set automatically for a
        signomial program, whose duals come from the final convex subproblem.
    check_ambiguity : bool
        Test which sensitivities the problem actually determines. A degenerate
        active set leaves the duals non-unique, and a sensitivity that depends
        on which dual vector was chosen is an artefact rather than an answer.
        Costs one SVD of the stationarity system -- 0.2s on SPaircraft -- and
        is what keeps a meaningless +315 out of the table.

    Returns
    -------
    dict
        ``{'sensitivities': {name: value}, 'objective': f*, 'normalized': bool,
        'method': str, 'approximate': bool, 'ambiguous': [name, ...],
        'ambiguity': {name: float}}``

    Notes
    -----
    The result is exact for LP, QP and GP: every partial derivative is taken
    symbolically, so there is no step size to choose and no truncation error.
    For a signomial program it describes the final convex subproblem and is
    therefore local to the returned point.
    """
    # Frame selection. Suffix duals (the IPOPT route) belong to the original
    # model's constraint objects, so that path stays on the original model.
    # KKT recovery, however, must run on the UNIT-CORRECTED twin: the raw
    # model's constraint expressions mix declared units (a Prouty weight
    # coefficient in lb/ft^2.3 beside a chord in m), so evaluating them raw
    # gives numbers in no consistent frame -- genuinely tight constraints look
    # slack, the recovered active set is wrong, and the stationarity system
    # goes inconsistent (observed as a residual of 0.19 on the first model
    # with non-SI Constants). unit_corrector folds the conversion factors into
    # the expressions as literals, the cloned variable values ride along
    # unchanged, and the Params keep their names and magnitudes, so the
    # normalized log-log sensitivities are identical to those defined on the
    # declared-units model. (method='fd' re-solves through the same correction
    # and needs neither.)
    if method != 'fd' and duals is None:
        use_suffix = (method in ('auto', 'suffix')
                      and _duals_from_suffix(model) is not None)
        if not use_suffix:
            from lcsolver.presolve.unitCorrector import unit_corrector
            model = unit_corrector(model)

    obj = _objective(model)
    try:
        fstar = float(pyo.value(obj))
    except Exception as e:
        raise RuntimeError(
            "could not evaluate the objective; the model does not appear to "
            f"hold a solution ({type(e).__name__}: {e})")

    if method == 'fd':
        return _fd_sensitivities(model, fstar, normalized=normalized)

    # A signomial program is solved as a sequence of convex subproblems, so its
    # duals belong to the last of those. Flag that unless the caller has said
    # otherwise, so the approximation is never silently presented as exact.
    if approximate is None:
        structure = getattr(model, '_edi_last_problem_structure', None)
        if structure is not None and 'signomial' in str(structure):
            approximate = True

    if duals is None:
        _duals_from_kkt.last_residual = None
        duals = constraint_duals(model, method=method, rtol=rtol)
        used = 'suffix' if (method != 'kkt' and _duals_from_suffix(model)) else 'kkt'
    else:
        used = 'given'
    residual = getattr(_duals_from_kkt, 'last_residual', None)

    # A sensitivity built on duals that do not satisfy stationarity is not a
    # sensitivity, it is a plausible-looking number. Both failure modes below
    # are reachable from a solve that did not converge, so say so loudly rather
    # than return a table of quiet zeros.
    if len(duals) == 0:
        warnings.warn(
            "[LC-W302] no binding constraints were found, so every sensitivity will be "
            "the objective's own dependence on each constant. If the model was "
            "expected to have active constraints, the solve probably did not "
            "converge.", RuntimeWarning, stacklevel=2)
    elif residual is not None and residual > KKT_RESIDUAL_WARN:
        warnings.warn(
            f"[LC-W302] the recovered duals satisfy the stationarity condition only to "
            f"a relative residual of {residual:.2e}. The returned "
            f"sensitivities are unreliable; this usually means the solve did "
            f"not converge or the active set is ambiguous.",
            RuntimeWarning, stacklevel=2)

    constants = _constants(model)
    index = {id(pd): name for name, pd in constants.items()}

    # partial of the objective, at fixed x*
    totals = dict.fromkeys(constants, 0.0)
    for name, dv in _param_gradient(obj.expr, index).items():
        totals[name] += dv
    # minus the duals times the partial of each active constraint residual
    for con, lam in duals.items():
        bound, _ = _bound_of(con)
        g = _param_gradient(con.body, index)
        for name, dv in _param_gradient(bound, index).items():
            g[name] = g.get(name, 0.0) - dv
        for name, dv in g.items():
            totals[name] -= lam * dv

    out = {}
    for name, pd in constants.items():
        total = totals[name]

        if normalized:
            try:
                cval = float(pyo.value(pd))
            except Exception:
                cval = float('nan')
            if fstar == 0 or cval == 0 or not math.isfinite(cval):
                out[name] = float('nan')
            else:
                out[name] = total * cval / fstar
        else:
            out[name] = total

    # Which of these the problem actually determines. Reported alongside
    # rather than dropped here: a caller asking for raw numbers should get
    # them, and the display layer decides what to show.
    ambiguity, ambiguous = {}, []
    if check_ambiguity:
        try:
            ambiguity = dual_ambiguity(model, rtol=rtol, constants=constants)
        except Exception:
            ambiguity = {}
        ambiguous = sorted(n for n, r in ambiguity.items()
                           if r > DUAL_AMBIGUITY_TOL)
        if ambiguous:
            warnings.warn(
                f"[LC-W303] {len(ambiguous)} of {len(out)} sensitivities are not "
                f"determined by the problem: the active set is degenerate, so "
                f"the duals are not unique and these values depend on which "
                f"dual vector was recovered. They are listed under "
                f"'ambiguous' and are hidden from the printed table.",
                RuntimeWarning, stacklevel=2)

    return {'sensitivities': out,
            'objective': fstar,
            'normalized': bool(normalized),
            'method': used,
            'stationarity_residual': residual,
            'ambiguity': ambiguity,
            'ambiguous': ambiguous,
            'approximate': bool(approximate) if approximate is not None else False}


def format_sensitivities(result, tol=1e-8, width=72):
    """Render a sensitivity result as a sorted, human-readable table."""
    sens = result['sensitivities']
    kind = 'd log(f*) / d log(c)' if result['normalized'] else 'd f* / d c'
    lines = []
    lines.append('=' * width)
    lines.append(f"Sensitivities to constants    [{kind}]")
    lines.append(f"objective = {result['objective']:.6g}"
                 + (f"    duals: {result['method']}" if result.get('method') else ''))
    if result.get('approximate'):
        lines.append("NOTE: signomial program -- these are a local approximation "
                     "from the\n      final convex subproblem.")
    lines.append('=' * width)

    ordered = sorted(sens.items(), key=lambda kv: -abs(kv[1])
                     if kv[1] == kv[1] else 0)
    negligible = 0
    for name, value in ordered:
        if value != value:                       # NaN
            lines.append(f"  {name:<28s}      n/a")
            continue
        if abs(value) < tol:
            negligible += 1
            continue
        bar = '+' if value > 0 else '-'
        lines.append(f"  {name:<28s} {value:+10.4f}   {bar * min(int(abs(value) * 20) + 1, 30)}")
    if negligible:
        lines.append(f"  ({negligible} constant(s) with |sensitivity| < {tol:g} omitted)")
    lines.append('=' * width)
    return '\n'.join(lines)
