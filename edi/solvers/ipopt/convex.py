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
IPOPT converges to the global optimum. (The outer logarithm usually seen in
textbook presentations is omitted deliberately: minimizing a positive sum and
minimizing its logarithm give the same minimizer, and dropping it avoids a
``log`` of a quantity that the solver may drive toward zero.)

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

    m.t = pyo.Var(m.J, initialize=_t0)
    return _build_and_solve_gp(m, n, groups, relations, tee, options,
                               method, executable)


def _build_and_solve_gp(m, n, groups, relations, tee, options, method,
                        executable):
    """Shared objective/constraint assembly and IPOPT call."""

    def _posy(t, terms):
        """Sum of exp(b_k + a_k . t) for one constraint/objective group."""
        return sum(pyo.exp(math.log(c) + sum(a[j] * t[j] for j in range(n)))
                   for c, a in terms)

    # objective: the posynomial itself (convex in t)
    m.obj = pyo.Objective(expr=_posy(m.t, groups[0]), sense=pyo.minimize)

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
            m.cons.add(_posy(m.t, terms) <= 1.0)
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
    res = {
        'status': 'optimal',
        'primal objective': pyo.value(m.obj),
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
