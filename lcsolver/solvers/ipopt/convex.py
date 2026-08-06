#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Solve structured (LP / QP / GP) formulations with IPOPT instead of cvxopt.

This is an *option*, not a replacement: ``cvxopt`` remains the default backend for
structured problems. IPOPT is offered because it is an interior-point code with a
sparse linear-algebra backend (MA27/MA57/MUMPS) and generally scales better on
larger problems than the dense cvxopt path.

Linear and quadratic programs are already convex in their natural variables, so
they are handed to IPOPT unchanged.

Geometric programs are *not* convex in their natural variables, and handing the
raw model to a general NLP solver would forfeit the global-optimality guarantee
that motivates using a GP at all. Instead the standard logarithmic change of
variables is applied here. With :math:`x_j = e^{t_j}`, a monomial

.. math::  c_k \\prod_j x_j^{a_{kj}}  \\;=\\;  e^{\\,b_k + a_k^\\top t},
           \\qquad b_k = \\log c_k,

so a posynomial becomes a sum of exponentials of affine functions, which is
convex. The GP standard form ``posynomial <= 1`` therefore becomes

.. math::  \\sum_k e^{\\,b_k + a_k^\\top t} \\;\\le\\; 1,

and the objective is the same sum, minimized. Both are convex in :math:`t`, so
IPOPT converges to the global optimum.

Both are formed under the outer logarithm -- ``log sum_k exp(...) <= 0`` -- as
in the textbook presentation. Dropping it, as an earlier version of this
module did, leaves IPOPT evaluating the posynomial itself; on a model with
large coefficients or exponents that value is ``e`` raised to several hundred
and overflows during a line search, which surfaces as "Invalid number in NLP
function or derivative" rather than as the arithmetic problem it is. The
concern that motivated dropping it -- ``log`` of something driven toward zero
-- is handled by the log-space box below, which stops ``t`` running to minus
infinity. Single-term groups are monomials and are emitted as affine
constraints directly, so they never form ``exp`` at all.

Monomial equality constraints are affine in :math:`t` and are passed through as
such. A multi-term equality is not a valid geometric program and is rejected.
"""

import math

import pyomo.environ as pyo


# ---------------------------------------------------------------------------
def _group_rows(rows):
    """Group monomial rows by constraint index. Row = [idx, coeff, *exponents]."""
    groups = {}
    for r in rows:
        groups.setdefault(int(r[0]), []).append((float(r[1]), [float(e) for e in r[2:]]))
    return groups


# Box on the log-space variables.
#
# In log space a GP is a sum of exp(log c + a.t), and nothing bounds t unless
# the model says so. Where a model carries large exponents -- SPaircraft's
# vertical tail drag fit has tau**133.8 and M**1022.7 -- a line search can
# walk a.t past 709, exp() overflows to inf, and IPOPT aborts with "Invalid
# number in NLP function or derivative". That reads like an infeasible model
# rather than an arithmetic overflow, which makes it expensive to optimization_check.
#
# A single box cannot serve: the variable carrying the 1022.7 exponent needs
# one three hundred times tighter than a variable appearing linearly. So the
# bound is derived per column from that column's own largest exponent, as the
# widest interval over which every monomial containing it stays evaluable:
#
#     |t_j| <= EXP_LIMIT / max_k |a_kj|
#
# clamped to [MIN_BOX, MAX_BOX] so that a variable appearing only with small
# exponents is not left effectively unbounded, and one with a huge exponent
# still gets room to move. For SPaircraft this puts Mach in [0.5, 2.0] and
# tail thickness in [0.005, 180] -- both far wider than any physical answer.
#
# This bounds the *iterates*, which is the part a constraint cannot do:
# IPOPT satisfies constraints only in the limit, so a monomial constraint on
# the same variable does not stop an intermediate point from overflowing.
EXP_LIMIT = 500.0
MIN_BOX = 0.5
MAX_BOX = 200.0


def _log_box(n, groups):
    """Per-column log-space bounds that keep every monomial evaluable."""
    amax = [0.0] * n
    for terms in groups.values():
        for _, a in terms:
            for j, aj in enumerate(a):
                if abs(aj) > amax[j]:
                    amax[j] = abs(aj)
    out = []
    for j in range(n):
        b = MAX_BOX if amax[j] <= 0 else EXP_LIMIT / amax[j]
        out.append(min(MAX_BOX, max(MIN_BOX, b)))
    return out


def solve_gp_rows_ipopt(rows, relations, x0=None, tee=False, options=None,
                        method='auto', executable=None, form='auto'):
    """Solve a geometric program given only its monomial rows, with IPOPT.

    This is the row-level core of :func:`solve_gp_ipopt`, split out so that it
    can also serve as the inner solve of the signomial (PCCP) loop. That loop
    linearizes about a moving point and *adds slack columns*, so its
    subproblems have more variables than the original model and no
    ``structures['variables']`` list to size them by — the width comes from
    the rows themselves and the warm start from the current iterate.

    Parameters
    ----------
    rows : list of ``[constraint_index, coefficient, *exponents]``
    relations : operator per constraint index 1..N
    x0 : optional warm start in the ORIGINAL (not log) variables

    Returns a dict with ``status``, ``primal objective``, ``x``.
    """
    groups = _group_rows(rows)
    if 0 not in groups:
        raise ValueError('no objective monomials found in the GP structure')

    # width is set by the rows, not by any external variable list
    n = max(len(a) for terms in groups.values() for _, a in terms)

    m = pyo.ConcreteModel()
    m.J = pyo.RangeSet(0, n - 1)

    def _t0(_m, j):
        if x0 is not None and j < len(x0):
            try:
                v = float(x0[j])
                if v > 0:
                    return math.log(v)
            except Exception:
                pass
        return 0.0

    box = _log_box(n, groups)
    m.t = pyo.Var(m.J, initialize=_t0,
                  bounds=lambda _m, j: (-box[j], box[j]))
    return _build_and_solve_gp(m, n, groups, relations, tee, options,
                               method, executable, form)


# How the posynomials are written for IPOPT.
#
#   'sum'  ->  sum_k exp(b_k + a_k.t)        <= 1     (the original form)
#   'lse'  ->  log sum_k exp(b_k + a_k.t)    <= 0
#
# Neither is right for every model, which is why this is a choice rather than
# a rewrite.
#
# 'sum' hands IPOPT the posynomial itself. That is better conditioned for the
# common case, where every log c + a.t is O(1..30): residuals stay near 1 and
# IPOPT's restoration phase behaves. The JHO sailplane is such a model
# (max |log c| = 30, max |a| = 19) and it solves with 'sum' and fails under
# 'lse'.
#
# 'lse' is needed once the arguments get large. SPaircraft reaches
# log c = 176 with exponents to 1022.7, so log c + a.t lands near 676 against
# an overflow threshold of 709; one line-search step under 'sum' produces inf
# and IPOPT aborts with "Invalid number in NLP function or derivative". Under
# the logarithm the same quantity is ~676 and stays finite.
#
# 'auto' picks 'lse' only when the row data says the plain sum is at risk,
# falls back to 'lse' if 'sum' fails outright, and VERIFIES a claimed 'sum'
# success with a warm-started 'lse' polish -- because 'sum' can also fail
# silently: on a model whose optimum sits at small absolute scale (the wind
# turbine COE model, objective ~1e-7 in corrected units) the sum-form KKT
# residuals deflate below IPOPT's tolerances and it certifies a point 7x off
# the optimum.  See _build_and_solve_gp.
GP_FORM_AUTO_LOGC = 100.0     # |log c| above which 'auto' switches to 'lse'
GP_FORM_AUTO_EXPONENT = 100.0  # |exponent| likewise


def _auto_form(groups):
    """Choose 'sum' or 'lse' from the magnitudes actually present."""
    max_logc = 0.0
    max_a = 0.0
    for terms in groups.values():
        for c, a in terms:
            if c:
                max_logc = max(max_logc, abs(math.log(abs(c))))
            for aj in a:
                if aj:
                    max_a = max(max_a, abs(aj))
    if max_logc > GP_FORM_AUTO_LOGC or max_a > GP_FORM_AUTO_EXPONENT:
        return 'lse'
    return 'sum'


# Relative objective improvement above which the 'lse' verification solve is
# taken to have caught a false 'sum' optimum (see _build_and_solve_gp).  The
# wind-turbine GP that motivated the check shows a factor of ~7; genuine
# agreement between the forms is within solver tolerance, so anything beyond
# this is a 'sum' failure, and a warning names it.
GP_SUM_VERIFY_RTOL = 1e-4


def _build_and_solve_gp(m, n, groups, relations, tee, options, method,
                        executable, form='auto'):
    """Shared objective/constraint assembly and IPOPT call."""
    if form not in ('auto', 'sum', 'lse'):
        raise ValueError("form must be 'auto', 'sum' or 'lse'")
    chosen = _auto_form(groups) if form == 'auto' else form
    try:
        res = _assemble_and_solve(m, n, groups, relations, tee, options,
                                  method, executable, chosen)
    except Exception:
        if form != 'auto' or chosen == 'lse':
            raise
        # 'sum' failed and we had not already tried the logarithm; the usual
        # cause is an overflow the magnitude test did not predict.
        m.del_component(m.obj)
        m.del_component(m.cons)
        return _assemble_and_solve(m, n, groups, relations, tee, options,
                                   method, executable, 'lse')

    if form != 'auto' or chosen != 'sum':
        return res

    # 'sum' CLAIMED success; verify it, because 'sum' can also fail
    # SILENTLY.  Without the outer logarithm every quantity IPOPT sees is
    # the posynomial itself, and on a model whose optimum lives at small
    # absolute scale the whole KKT system deflates with it: the wind
    # turbine COE model (objective ~1e-7 in corrected base units) returns
    # "EXIT: Optimal Solution Found" with every residual below tolerance
    # at a point 7x above the true optimum -- the tolerances are simply
    # larger than the numbers that would have to move.  No property of the
    # returned point distinguishes this (the constraints are feasible and
    # nothing is overflowed), so the check is a second solve in 'lse'
    # form, warm-started at the 'sum' answer: log-space values are O(1-30)
    # where IPOPT's tolerances mean something.  If 'sum' was right the
    # polish converges in a handful of iterations to the same point; if
    # not, it walks to the real optimum.  If 'lse' itself fails -- the JHO
    # sailplane does, which is why 'sum' exists -- the 'sum' answer is
    # kept, so models that only 'sum' can solve behave exactly as before.
    m.del_component(m.obj)
    m.del_component(m.cons)
    try:
        res_lse = _assemble_and_solve(m, n, groups, relations, tee, options,
                                      method, executable, 'lse')
    except Exception:
        return res
    if res_lse['primal objective'] < res['primal objective'] * (1.0 - GP_SUM_VERIFY_RTOL):
        import warnings
        warnings.warn(
            "the 'sum'-form GP solve reported optimality at an objective of "
            f"{res['primal objective']:.6g}, but the 'lse' verification solve "
            f"reached {res_lse['primal objective']:.6g}; the 'sum' result was "
            "a false optimum (its KKT residuals deflated below IPOPT's "
            "tolerances) and the 'lse' result is returned instead.",
            RuntimeWarning)
        return res_lse
    return res


def _assemble_and_solve(m, n, groups, relations, tee, options, method,
                        executable, form):

    def _affine(t, c, a):
        return math.log(c) + sum(a[j] * t[j] for j in range(n) if a[j])

    def _lse(t, terms):
        """log sum_k exp(b_k + a_k . t) for one constraint/objective group.

        The outer logarithm matters numerically. Without it the value handed
        to IPOPT is the posynomial itself, which for a model carrying large
        coefficients or exponents is e raised to several hundred: SPaircraft
        reaches log c = 176 with exponents to 1022.7, so log c + a.t lands
        around 676 against an overflow threshold of 709. One line-search step
        tips it to inf and IPOPT aborts with "Invalid number in NLP function
        or derivative". Under the logarithm the same quantity is ~676, and
        every residual IPOPT sees stays within a couple of orders of 1.

        Taking the log is free mathematically -- log is monotone, so
        minimizing a positive sum and minimizing its logarithm have the same
        minimizer, and `posy <= 1` is exactly `log(posy) <= 0`.

        A single-term group is a monomial, whose log is already affine; it is
        returned as such rather than round-tripped through exp/log, which
        keeps roughly half the constraints of a typical GP linear.
        """
        if len(terms) == 1:
            c, a = terms[0]
            return _affine(t, c, a)
        return pyo.log(sum(pyo.exp(_affine(t, c, a)) for c, a in terms))

    def _body(t, terms):
        return _lse(t, terms) if form == 'lse' else _posy(t, terms)

    def _posy(t, terms):
        return sum(pyo.exp(_affine(t, c, a)) for c, a in terms)

    # objective: the posynomial, or its logarithm (same minimizer either way)
    m.obj = pyo.Objective(expr=_body(m.t, groups[0]), sense=pyo.minimize)

    # constraints: index i in 1..N maps to relations[i-1]
    m.cons = pyo.ConstraintList()
    n_ineq = n_eq = 0
    for idx in sorted(k for k in groups if k != 0):
        rel = relations[idx - 1] if (idx - 1) < len(relations) else '<='
        terms = groups[idx]
        if rel == '==':
            if len(terms) != 1:
                raise ValueError(
                    f'constraint {idx} is an equality with {len(terms)} terms; only '
                    'monomial equalities are valid in a geometric program')
            c, a = terms[0]
            # c * prod x^a == 1  ->  log(c) + a.t == 0   (affine, hence convex)
            m.cons.add(math.log(c) + sum(a[j] * m.t[j] for j in range(n)) == 0)
            n_eq += 1
        else:
            # posy <= 1, or equivalently log(posy) <= 0
            m.cons.add(_body(m.t, terms) <= (0.0 if form == 'lse' else 1.0))
            n_ineq += 1

    # ---- solve -----------------------------------------------------------
    from edi.solvers.ipopt.ipopt_solver_interface import (
        _executable_available, _summarize)
    from pyomo.opt import TerminationCondition

    route = method
    if route == 'auto':
        route = 'pyomo' if _executable_available('ipopt') else 'cyipopt'
    if route == 'pyomo':
        opt = (pyo.SolverFactory('ipopt', executable=executable)
               if executable else pyo.SolverFactory('ipopt'))
    else:
        opt = pyo.SolverFactory('cyipopt')
    if not opt.available(exception_flag=False):
        raise RuntimeError(
            'no usable IPOPT installation found for the convex backend; install '
            'the ipopt executable or `pip install cyipopt`')
    for k, v in (options or {}).items():
        opt.options[k] = v

    results = opt.solve(m, tee=tee) if route == 'cyipopt' else opt.solve(m, tee=tee)
    summary = _summarize(results)

    tc = summary['termination_condition']
    if tc not in (str(TerminationCondition.optimal),
                  str(TerminationCondition.locallyOptimal),
                  str(TerminationCondition.feasible)):
        raise RuntimeError(
            f'IPOPT did not converge on the log-transformed GP: '
            f'termination_condition={tc}. {summary["message"]}'.strip())

    # ---- map back to the original variables -------------------------------
    x = [math.exp(pyo.value(m.t[j])) for j in range(n)]
    # Under 'lse' the objective solved is log(posynomial); callers want the
    # posynomial itself.
    raw_obj = pyo.value(m.obj)
    if form == 'lse':
        obj = math.exp(raw_obj) if raw_obj < 700 else float('inf')
    else:
        obj = raw_obj
    res = {
        'status': 'optimal',
        'primal objective': obj,
        'gp form': form,
        'x': x,
        'solver': f'ipopt ({route}, log-transformed)',
        'problem_structure': 'geometric_program',
        'termination_condition': tc,
        'n_ineq': n_ineq,
        'n_eq': n_eq,
    }

    return res


def solve_gp_ipopt(structures, model=None, tee=False, options=None,
                   method='auto', executable=None, form='auto'):
    """Solve a detected geometric program with IPOPT in log space.

    Returns a dict shaped like the other EDI backends: ``status``,
    ``primal objective`` (in the ORIGINAL variables), ``x`` (original
    variables, in ``structures['variables']`` order), and ``solution``.
    """
    gp = structures['Geometric_Program']
    if not gp[0]:
        raise ValueError('the formulation was not detected as a geometric program')

    variables = structures['variables']
    x0 = []
    for v in variables:
        try:
            val = pyo.value(v)
        except Exception:
            val = None
        x0.append(val if (val is not None and val > 0) else None)

    res = solve_gp_rows_ipopt(gp[1], gp[2], x0=x0, tee=tee, options=options,
                              method=method, executable=executable, form=form)
    if model is not None:
        from edi.solvers.writeback import write_solution
        res['solution'] = write_solution(structures, res, model=model)
    return res


# ---------------------------------------------------------------------------
def solve_lp_qp_ipopt(m, **kwargs):
    """LP/QP with IPOPT.

    These are already convex in their natural variables, so no transformation is
    needed: the model goes to IPOPT unchanged and the solution is loaded back by
    Pyomo in the usual way.
    """
    from edi.solvers.ipopt import ipopt_solve
    return ipopt_solve(m, **kwargs)
