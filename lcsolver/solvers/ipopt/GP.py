#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Solve structured (LP / QP / GP) formulations with IPOPT instead of cvxopt.

An option, not a replacement: cvxopt stays the default backend. IPOPT is
interior-point with sparse linear algebra (MA27/MA57/MUMPS) and scales better
on larger problems. LP/QP are convex as-is and go to IPOPT unchanged.

A GP is not convex in its natural variables, so the standard log change of
variables is applied: with x_j = exp(t_j) a monomial c_k prod_j x_j^a_kj
becomes exp(b_k + a_k.t) with b_k = log c_k, a posynomial becomes a sum of
exponentials of affine functions (convex), and posy <= 1 becomes
sum_k exp(b_k + a_k.t) <= 1 -- so IPOPT gets the global optimum.

Both objective and constraints go in under the outer logarithm
(log sum exp <= 0); dropping it overflows on large coefficients/exponents
("Invalid number in NLP function or derivative"), and the log-space box below
stops t running to -inf. Monomials stay affine; multi-term equalities are
rejected (not a valid GP).
"""

import math

import pyomo.environ as pyo

from lcsolver.core.errors import SolverUnavailable


# ---------------------------------------------------------------------------
def _group_rows(rows):
    """Group monomial rows by constraint index. Row = [idx, coeff, *exponents]."""
    groups = {}
    for r in rows:
        groups.setdefault(int(r[0]), []).append((float(r[1]), [float(e) for e in r[2:]]))
    return groups


# Box on the log-space variables. Large exponents (SPaircraft's tail drag
# fit has tau**133.8 and M**1022.7) let a line search walk a.t past 709 and
# exp() overflows, so bound each column by its own largest exponent:
# |t_j| <= EXP_LIMIT / max_k |a_kj|, clamped to [MIN_BOX, MAX_BOX]. This
# bounds the ITERATES, which a constraint cannot do -- IPOPT satisfies
# constraints only in the limit.
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
                        method='auto', executable=None, form='auto',
                        linear_solver=None, linear_solver_library=None):
    """Solve a GP given only its monomial rows, with IPOPT.

    Row-level core of solve_gp_ipopt, split out to serve as the PCCP inner
    solve -- those subproblems add slack columns and have no
    structures['variables'] list, so width comes from the rows and the warm
    start from the iterate. rows = [constraint_index, coeff, *exponents],
    relations = operator per constraint 1..N, x0 in the original (not log)
    variables. Returns a dict with status, primal objective, x.
    """
    if linear_solver is not None:
        # Resolve here, at the top of the chain, against the same route
        # _assemble_and_solve will choose; the choice travels in options
        from lcsolver.environment import (
            linear_solver_library_option,
            require_linear_solver,
        )
        from lcsolver.solvers.ipopt.NLP import _executable_available
        _route = (method if method != 'auto'
                  else ('pyomo' if _executable_available('ipopt')
                        else 'cyipopt'))
        options = dict(options or {})
        options['linear_solver'] = require_linear_solver(
            linear_solver, route=_route,
            executable=executable if _route == 'pyomo' else None,
            library=linear_solver_library)
        if linear_solver_library is not None:
            options[linear_solver_library_option(
                options['linear_solver'])] = str(linear_solver_library)
        from lcsolver.environment import apply_linear_solver_defaults
        apply_linear_solver_defaults(options, options['linear_solver'])

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
    res = _build_and_solve_gp(m, n, groups, relations, tee, options,
                              method, executable, form)
    if isinstance(res, dict):
        # first question when two machines disagree; None = build default
        res['linear_solver'] = (options or {}).get('linear_solver')
    return res


# How the posynomials are written for IPOPT.
#
#   'sum'  ->  sum_k exp(b_k + a_k.t)        <= 1     (the original form)
#   'lse'  ->  log sum_k exp(b_k + a_k.t)    <= 0
#
# Neither fits every model. 'sum' is better conditioned when log c + a.t is
# O(1..30): the JHO sailplane solves with 'sum' and fails under 'lse'.
# 'lse' is needed once arguments get large: SPaircraft hits log c = 176 with
# exponents to 1022.7, and one 'sum' line-search step overflows ("Invalid
# number in NLP function or derivative"). 'auto' picks 'lse' when the row
# data says 'sum' is at risk, falls back to 'lse' on failure, and verifies a
# claimed 'sum' success with a warm-started 'lse' polish -- 'sum' can fail
# SILENTLY when the optimum sits at small scale (wind turbine COE, ~1e-7:
# KKT residuals deflate below tolerance at a point 7x off the optimum).
# See _build_and_solve_gp.
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


# Relative improvement above which the 'lse' verification caught a false
# 'sum' optimum. The motivating wind-turbine GP shows ~7x; genuine agreement
# is within solver tolerance, so anything beyond this warns.
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
        # 'sum' failed before trying the log form; usually an overflow the
        # magnitude test did not predict.
        m.del_component(m.obj)
        m.del_component(m.cons)
        return _assemble_and_solve(m, n, groups, relations, tee, options,
                                   method, executable, 'lse')

    if form != 'auto' or chosen != 'sum':
        return res

    # 'sum' claimed success; verify, because 'sum' can fail SILENTLY: at
    # small absolute scale the whole KKT system deflates below tolerance
    # (wind turbine COE, ~1e-7, certified 7x off the optimum) and nothing
    # about the returned point distinguishes it. So re-solve in 'lse' form
    # warm-started at the 'sum' answer -- if 'sum' was right the polish
    # converges to the same point in a few iterations; if 'lse' itself
    # fails (JHO sailplane), keep the 'sum' answer as before.
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
            "[LC-W204] the 'sum'-form GP solve reported optimality at an objective of "
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
        """log sum_k exp(b_k + a_k . t) for one group. The outer log keeps
        large arguments finite (see the form comment above) and is free
        mathematically -- log is monotone, posy <= 1 is log(posy) <= 0.
        A single-term group is a monomial: returned affine, not
        round-tripped through exp/log, keeping ~half a typical GP linear.
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
    from lcsolver.solvers.ipopt.NLP import (
        _executable_available, _summarize)
    from pyomo.opt import TerminationCondition

    route = method
    if route == 'auto':
        route = 'pyomo' if _executable_available('ipopt') else 'cyipopt'
    if route == 'pyomo':
        from lcsolver.environment import ipopt_solver_factory
        opt = ipopt_solver_factory(executable)
    else:
        opt = pyo.SolverFactory('cyipopt')
        from lcsolver.environment import _pynumero_asl_available
        if opt.available(exception_flag=False) and not _pynumero_asl_available():
            raise SolverUnavailable(
                "the in-process (cyipopt) route needs Pyomo's PyNumero ASL "
                "library, which is not installed. Run "
                "`lcsolver-install-solvers`.")
    if not opt.available(exception_flag=False):
        raise SolverUnavailable(
            'no usable IPOPT installation found for the convex backend; run '
            '`lcsolver-install-solvers`')

    # The two routes take options differently: only the AMPL one has an
    # `options` mapping; PyomoCyIpoptSolver takes them as a solve() argument
    # and raises AttributeError on `opt.options[...]`. An MA27 user sets
    # linear_solver on the cyipopt route, so this has to be right on both.
    if route == 'cyipopt':
        results = opt.solve(m, tee=tee, options=dict(options or {}))
    else:
        from lcsolver.environment import ipopt_launch
        for k, v in (options or {}).items():
            opt.options[k] = v
        with ipopt_launch((options or {}).get('linear_solver'), executable):
            results = opt.solve(m, tee=tee)
    summary = _summarize(results)

    tc = summary['termination_condition']
    if tc not in (str(TerminationCondition.optimal),
                  str(TerminationCondition.locallyOptimal),
                  str(TerminationCondition.feasible)):
        from lcsolver.environment import linear_solver_failure_note
        raise RuntimeError(
            f'IPOPT did not converge on the log-transformed GP: '
            f'termination_condition={tc}. {summary["message"]}'.strip()
            + linear_solver_failure_note(
                (options or {}).get('linear_solver'),
                executable if route == 'pyomo' else None))

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
                   method='auto', executable=None, form='auto',
                   linear_solver=None, linear_solver_library=None):
    """Solve a detected GP with IPOPT in log space. Returns a dict shaped
    like the other backends: status, primal objective (original variables),
    x (in structures['variables'] order), solution."""
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
                              method=method, executable=executable, form=form,
                              linear_solver=linear_solver,
                              linear_solver_library=linear_solver_library)
    if model is not None:
        from lcsolver.postsolve.writeback import write_solution
        res['solution'] = write_solution(structures, res, model=model)
    return res


# ---------------------------------------------------------------------------
def solve_lp_qp_ipopt(m, structure='linear_program', **kwargs):
    """LP/QP with IPOPT. Already convex in the natural variables, so the
    model goes in unchanged. structure (what the detector found) is stamped
    onto the result -- ipopt_solve labels everything nonlinear_program,
    which contradicted structure_report() on a detected LP."""
    from lcsolver.solvers.ipopt import ipopt_solve

    res = ipopt_solve(m, **kwargs)
    try:
        res['problem_structure'] = structure
        m._edi_last_problem_structure = structure
    except Exception:
        pass
    return res
