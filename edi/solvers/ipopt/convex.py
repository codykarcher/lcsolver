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
# rather than an arithmetic overflow, which makes it expensive to diagnose.
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
                        method='auto', executable=None):
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
                               method, executable)


def _build_and_solve_gp(m, n, groups, relations, tee, options, method,
                        executable):
    """Shared objective/constraint assembly and IPOPT call."""

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

    # objective: log of the posynomial (convex in t, same minimizer)
    m.obj = pyo.Objective(expr=_lse(m.t, groups[0]), sense=pyo.minimize)

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
            # posy <= 1  <=>  log(posy) <= 0
            m.cons.add(_lse(m.t, terms) <= 0.0)
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
    # The objective is solved as log(posynomial); callers want the posynomial.
    log_obj = pyo.value(m.obj)
    res = {
        'status': 'optimal',
        'primal objective': math.exp(log_obj) if log_obj < 700 else float('inf'),
        'log primal objective': log_obj,
        'x': x,
        'solver': f'ipopt ({route}, log-transformed)',
        'problem_structure': 'geometric_program',
        'termination_condition': tc,
        'n_ineq': n_ineq,
        'n_eq': n_eq,
    }

    return res


def solve_gp_ipopt(structures, model=None, tee=False, options=None,
                   method='auto', executable=None):
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
                              method=method, executable=executable)
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
