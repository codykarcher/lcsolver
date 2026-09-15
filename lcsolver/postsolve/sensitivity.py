#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Sensitivity of the optimum to the Constants of a formulation.

The GPkit-style report: for every Constant, the log-log sensitivity

    s_c = d log f* / d log c = (c / f*) df*/dc

which is unitless, so comparable across constants with different units.
Computed exactly from a single solve via the envelope theorem: with lambda_i
the constraint duals,

    df*/dtheta = df/dtheta - sum_i lambda_i d(body_i - bound_i)/dtheta

Every partial is taken symbolically (Pyomo reverse mode), so no truncation
error and no re-solves. Duals come from a populated ``dual`` Suffix (the
IPOPT route) or, for the cvxopt backends, are recovered from the primal
solution by least-squares KKT stationarity over the active set. For an SP
the duals describe the final convex subproblem, so the sensitivities are
local to the returned point; ``result['approximate']`` flags it.
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


# Relative tolerance for calling a constraint binding. Must be looser than the
# primal accuracy (~1e-6 from interior point) or tight constraints get dropped,
# corrupting every recovered dual. On the aircraft GP the duals are correct to
# 8e-6 and insensitive anywhere in 1e-5..1e-2, so 1e-4 sits mid-plateau.
ACTIVE_RTOL = 1e-4

# Fraction of a constant's active-set gradient lying in the stationarity null
# space above which its sensitivity is reported undetermined (rank-deficient
# duals). Sits in an empty band: on SPaircraft the 195 constants land below
# 1e-8 or above 1e-4, nothing between; hiding those 40 takes 315.86 -> 1.71.
DUAL_AMBIGUITY_TOL = 1e-6

# Relative stationarity residual above which recovered duals are reported as
# untrustworthy. A converged solve sits far below this (aircraft GP: 1.6e-6).
KKT_RESIDUAL_WARN = 1e-3


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _d(expr, wrt):
    """Exact partial d(expr)/d(wrt) as a float; 0.0 when absent or independent."""
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

    Pyomo hoists a mutable Param into the bound (``x >= p`` lives in
    ``con.lower``), so both sides have to be differentiated.
    """
    lower, upper = con.lower, con.upper
    if lower is not None and upper is not None:
        # equality or ranged; for a range, the binding side is whichever
        # the body currently sits on
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

    The bound is routinely 0 (LCsolver moves everything to one side), so judge
    the residual against the largest term, ``max_j |x_j d(body)/dx_j|`` --
    a 1e4 N weight constraint with a 1e-1 residual is tight, not slack.
    """
    g = _grad(con.body, variables)
    xs = np.array([abs(pyo.value(v)) if v.value is not None else 0.0
                   for v in variables])
    terms = np.abs(g) * xs
    return float(terms.max()) if terms.size and terms.max() > 0 else 1.0


def _is_active(con, rtol=ACTIVE_RTOL, variables=None, scale=None):
    """True if the constraint is binding at the current point.

    Compared against the constraint's own natural magnitude only. An earlier
    ``rtol * max(1.0, scale)`` floor made the test absolute below unit scale:
    on the oxygenator GP it declared a slack 1e-12 bound active, made the
    active set rank deficient, and gave a wrong-signed dual. No floor:
    agreement with finite differences to ~1e-5.
    """
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
        if variables is not None:
            scale = _residual_scale(con, variables)
        else:
            # no variables to measure terms against, so the bound is the only
            # scale; fall back to 1.0 only when it is exactly 0, never floor
            # a bound that is merely small
            scale = abs(bd) if bd else 1.0
    # guard against a caller-supplied nonpositive scale
    return abs(b - bd) <= rtol * (scale if scale > 0 else 1.0)


def _param_gradient(expr, index):
    """``{name: d expr / d constant}`` for every Constant appearing in ``expr``.

    One reverse sweep per constraint, restricted to the constants actually
    present (typically two or three); the pairwise form walks the same
    expressions tens of thousands of times on an aircraft model.
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
    """Every LCsolver Constant (mutable Param) as ParamData keyed by name;
    an indexed Constant contributes one entry per element, GPkit-style."""
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


def _greybox_blocks(model):
    """Every active ExternalGreyBoxBlock on the model; empty without pynumero."""
    try:
        from pyomo.contrib.pynumero.interfaces.external_grey_box import (
            ExternalGreyBoxBlock)
    except Exception:
        return []
    return list(model.component_data_objects(ExternalGreyBoxBlock,
                                             descend_into=True, active=True))


def _greybox_gradients(model, variables):
    """``[(block, [grad_1, ..., grad_m])]`` for each grey box on the model.

    One gradient per output over ``variables``: +1 on the output variable,
    -d(out)/d(in) on each input, i.e. the equality ``out - box(in) == 0`` in
    the body - bound sign convention. Matched by name so it survives the
    unit-corrected clone; the jacobian usually comes from the box's cache,
    and a box that cannot be evaluated is skipped rather than losing the pass.
    """
    index = {v.name: i for i, v in enumerate(variables)}
    out = []
    for blk in _greybox_blocks(model):
        try:
            bb = blk.get_external_model()
            ins, outs = list(bb.input_names()), list(bb.output_names())
            if any(n not in index for n in ins + outs):
                continue
            bb.set_input_values(np.array([pyo.value(variables[index[n]])
                                          for n in ins], dtype=float))
            J = bb.evaluate_jacobian_outputs()
            J = J.toarray() if hasattr(J, 'toarray') else np.asarray(J)
        except Exception:
            continue
        grads = []
        for r, o in enumerate(outs):
            g = np.zeros(len(variables))
            g[index[o]] = 1.0
            for c, n in enumerate(ins):
                g[index[n]] -= float(J[r, c])
            grads.append(g)
        out.append((blk, grads))
    return out


# A KKT stationarity system, kept so that the ambiguity test can reuse it.
KKTSystem = collections.namedtuple("KKTSystem", "A_s rhs_s col_scale owners")


def _kkt_system(model, rtol=ACTIVE_RTOL):
    """Assemble the scaled stationarity system ``A_s lambda_s = rhs_s``.

    Separate from solving because dual_ambiguity needs the same matrix, and
    assembly (one symbolic gradient per active constraint) is the expensive half.
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

    # A RuntimeConstraint is an ExternalGreyBoxBlock, not a pyo.Constraint, so
    # the loop above never sees it; without its column the stationarity system
    # is inconsistent (every black-box model fired LC-W302) and the ordinary
    # duals come out wrong. Each output is the equality out - box(in) == 0:
    # +1 on the output variable, -J on the inputs. The grey-box multipliers
    # are kept -- a box may declare Constants, whose sensitivities chain
    # through lambda -- and the owner marker records which block and row.
    for _blk, grads in _greybox_gradients(model, variables):
        for _r, g in enumerate(grads):
            columns.append(g)
            owners.append(('gb', _blk, _r))

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

    # Equilibrate before solving: the variables span many orders of magnitude,
    # so scale rows by variable value (relative changes, O(1) entries) and
    # columns by constraint magnitude. Both are undone afterwards, so the
    # duals come back in natural units.
    xs = np.array([abs(pyo.value(v)) if v.value is not None else 0.0
                   for v in variables])
    row_scale = np.where(xs > 0, xs, 1.0)
    col_scale = np.abs(A * row_scale[:, None]).max(axis=0)
    col_scale = np.where(col_scale > 0, col_scale, 1.0)

    return KKTSystem(A_s=A * row_scale[:, None] / col_scale[None, :],
                     rhs_s=rhs * row_scale, col_scale=col_scale, owners=owners)


def _duals_from_kkt(model, rtol=ACTIVE_RTOL):
    """Recover duals from the primal solution via KKT stationarity.

    Least-squares solve of ``grad f = sum_i lambda_i grad(body_i - bound_i)``
    over the active set, in Pyomo's dual-Suffix sign convention. Active
    variable bounds are included as columns so stationarity can be met, then
    their multipliers are discarded (a bound cannot depend on a Constant).
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

    # Stationarity residual in the scaled space. Large means a wrong active
    # set or not a local optimum; record it rather than fail silently.
    denom = np.linalg.norm(rhs_s)
    resid = float(np.linalg.norm(A_s @ lam_s - rhs_s) / (denom if denom > 0 else 1.0))
    _duals_from_kkt.last_residual = resid

    greybox_duals = []
    for owner, value in zip(owners, lam):
        if owner is None:
            continue
        if isinstance(owner, tuple) and owner and owner[0] == 'gb':
            greybox_duals.append((owner[1], owner[2], float(value)))
            continue
        duals[owner] = float(value)
    _duals_from_kkt.last_greybox_duals = greybox_duals
    return duals


def dual_ambiguity(model, rtol=ACTIVE_RTOL, system=None, constants=None):
    """How undetermined each constant's sensitivity is, as a relative size.

    Returns ``{name: r}``, the fraction of the constant's active-set gradient
    lying in the stationarity null space. r == 0: same answer for every valid
    dual vector; r > 0: the number is an artefact of which duals were picked.
    A degenerate active set is normal (SPaircraft: 23 dof), so this is
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

    # The constant's gradient over the active set; variable-bound columns
    # stay zero (a bound cannot depend on a Constant).
    grads = {name: np.zeros(len(owners)) for name in constants}
    for i, con in enumerate(owners):
        if con is None:
            continue
        if isinstance(con, tuple) and con and con[0] == 'gb':
            # grey-box row g = out - box(in, c): gradient in a declared
            # constant is -d(box)/d(c), read from the box itself
            _blk, _r = con[1], con[2]
            try:
                cj = _blk.get_external_model().constant_jacobian()
            except Exception:
                continue
            for name, colv in cj.items():
                if name in grads:
                    grads[name][i] = -float(colv[_r])
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

    ``method='auto'`` prefers a populated ``dual`` Suffix (the IPOPT route)
    and falls back to KKT recovery from the primal (the cvxopt backends);
    'suffix' or 'kkt' force one. ``rtol`` is the active-set tolerance.
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

    Never consults duals, so immune to the degenerate-active-set failure of
    KKT recovery. Costs two solves per Constant; leaves the model re-solved
    at the baseline on exit.
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
    """Sensitivity of the optimum to every Constant; call after ``solve(f)``.

    ``normalized=True`` gives the unitless ``d log f* / d log c``; False the
    raw ``d f* / d c``. ``method`` is 'auto'/'suffix'/'kkt' as in
    :func:`constraint_duals`, or 'fd' to central-difference by re-solving
    (~2 solves per Constant; the assumption-free fallback when dual recovery
    is ambiguous). ``check_ambiguity`` costs one SVD (0.2s on SPaircraft) and
    flags sensitivities the problem does not determine -- it keeps a
    meaningless +315 out of the table. Exact for LP/QP/GP; for an SP, local
    to the final convex subproblem. Returns a dict with 'sensitivities',
    'objective', 'normalized', 'method', 'stationarity_residual', 'ambiguity',
    'ambiguous', and 'approximate'.
    """
    # Frame selection. Suffix duals (IPOPT) belong to the original model's
    # constraint objects, so that path stays there. KKT recovery must run on
    # the UNIT-CORRECTED twin: raw expressions mix declared units, so tight
    # constraints look slack and the stationarity system goes inconsistent
    # (residual 0.19 on the first non-SI model). unit_corrector folds the
    # conversions in as literals; names, magnitudes, and the normalized
    # sensitivities are unchanged. Only KKT recovery produces grey-box row
    # duals, which a box that declares Constants needs: such a model goes
    # KKT on 'auto', and an explicit 'suffix' is honoured but warned.
    _gb_constants = any(
        getattr(blk.get_external_model(), 'constantParams_optimization', None)
        for blk in _greybox_blocks(model))
    if method != 'fd' and duals is None:
        use_suffix = (method in ('auto', 'suffix')
                      and _duals_from_suffix(model) is not None
                      and not (_gb_constants and method == 'auto'))
        if _gb_constants and method == 'suffix':
            warnings.warn(
                "[LC-W312] this model's black box(es) declare Constants, "
                "whose sensitivity contribution needs KKT-recovered "
                "grey-box duals; method='suffix' cannot include it, so "
                "those sensitivities are missing the path through the box. "
                "Use method='auto' or 'kkt'.", RuntimeWarning, stacklevel=2)
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

    # SP duals belong to the last convex subproblem; flag the approximation
    # unless the caller has said otherwise.
    if approximate is None:
        structure = getattr(model, '_edi_last_problem_structure', None)
        if structure is not None and 'signomial' in str(structure):
            approximate = True

    if duals is None:
        _duals_from_kkt.last_residual = None
        _duals_from_kkt.last_greybox_duals = []
        _method = ('kkt' if (_gb_constants and method == 'auto')
                   else method)
        duals = constraint_duals(model, method=_method, rtol=rtol)
        used = ('suffix' if (_method != 'kkt' and _duals_from_suffix(model))
                else 'kkt')
    else:
        used = 'given'
    residual = getattr(_duals_from_kkt, 'last_residual', None)

    # Duals that don't satisfy stationarity give plausible-looking garbage;
    # both failure modes are reachable from a non-converged solve, so warn
    # loudly rather than return a table of quiet zeros.
    if len(duals) == 0:
        warnings.warn(
            "[LC-W302] no binding constraints were found, so every sensitivity will be "
            "the objective's own dependence on each constant. If the model was "
            "expected to have active constraints, the solve probably did not "
            "converge.", RuntimeWarning, stacklevel=2)
    elif residual is not None and residual > KKT_RESIDUAL_WARN:
        warnings.warn(
            f"[LC-W302] the recovered duals satisfy the stationarity condition only to "
            f"a relative residual of {residual:.2e}. EVERY returned "
            f"sensitivity (all constants) is unreliable; this usually means "
            f"the solve did not converge or the active set is ambiguous.",
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

    # ... and the same term for grey-box rows whose box declares Constants:
    # dg/dc = -d(box)/d(c), so the contribution is +lambda * d(box)/d(c)
    if used == 'kkt':
        for _blk, _r, lam in (getattr(_duals_from_kkt,
                                      'last_greybox_duals', None) or []):
            try:
                cj = _blk.get_external_model().constant_jacobian()
            except Exception:
                continue
            for name, colv in cj.items():
                if name in totals:
                    totals[name] += lam * float(colv[_r])

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

    # Report which of these the problem actually determines; the display
    # layer decides what to hide.
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
                f"determined by the problem: "
                + ", ".join(ambiguous[:8])
                + (", ..." if len(ambiguous) > 8 else "")
                + ". The active set is degenerate, so the duals are not "
                  "unique and these values depend on which dual vector was "
                  "recovered. They are hidden from the printed table.",
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
