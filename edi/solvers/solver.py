#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2023
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.dependencies import attempt_import
# from edi.structure.structureDetector import structure_detector
from edi.structure.structureDetector import structure_detector
from edi.units.unitCorrector import unit_corrector
from edi.solvers.writeback import write_solution


cvxopt, cvxopt_available = attempt_import( "cvxopt" )
if not cvxopt_available:
    raise ImportError('The CVXOPT solver requires cvxopt')

def cvxopt_solve(m, write_back=True):
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['maxiters'] = 100
    cvxopt.solvers.options['feastol'] = 1e-6
    cvxopt.printing.options['width'] = -1

    # print('walking units')
    m_corrected_units = unit_corrector(m)
    # print('walking structure')
    structures = structure_detector(m_corrected_units)
    _raise_if_infeasible(structures)
    # print('Detected problem structure')
    
    # structures = structure_detector(m)
    # print(structures)
    if structures['Linear_Program'][0]:
        # from edi.solvers.cvxopt import solve_LP
        from edi.solvers.cvxopt.LP import solve_LP
        res = solve_LP(structures)
        res['problem_structure'] = 'linear_program'
    elif structures['Quadratic_Program'][0]:
        # from edi.solvers.cvxopt import solve_QP
        from edi.solvers.cvxopt.QP import solve_QP
        res = solve_QP(structures)
        res['problem_structure'] = 'quadratic_program'
    elif structures['Geometric_Program'][0]:
        # from edi.solvers.cvxopt import solve_GP
        from edi.solvers.cvxopt.GP import solve_GP
        res = solve_GP(structures)
        res['problem_structure'] = 'geometric_program'
    elif structures['Signomial_Program'][0]:
        # from edi.solvers.cvxopt.SP import solve_SP
        from edi.solvers.cvxopt.SP import solve_SP
        res = solve_SP(structures,m)
        res['problem_structure'] = 'signomial_program_pccp'
    else:
        raise ValueError('Could not convert the formulation to a valid CVXOPT structure (LP,QP,GP,SP)')

    # cvxopt can return a non-converged point with status 'unknown' and no
    # exception. Surface that rather than letting it pass for a solution.
    if res.get('status') not in ('optimal', None):
        import warnings
        warnings.warn(
            f"cvxopt returned status={res.get('status')!r}; the reported point may "
            f"be infeasible or non-optimal. Consider convex_backend='ipopt'.",
            RuntimeWarning, stacklevel=2)

    # Record which structure was solved. `sensitivities` reads this to tell
    # whether the duals came from a genuinely convex solve or from the final
    # subproblem of a signomial sequence, which is only a local approximation.
    try:
        m._edi_last_problem_structure = res['problem_structure']
    except Exception:
        pass

    # Write the solution back onto the Pyomo model. Without this the solve
    # succeeds but pyo.value(m.x) still returns the initial guess, because the
    # cvxopt backends work in a transformed space and return only a raw vector.
    if write_back:
        try:
            res['solution'] = write_solution(structures, res, model=m)
        except Exception as e:                      # never lose a good solve
            res['solution'] = None
            res['writeback_error'] = f"{type(e).__name__}: {e}"
            # Say so. A failed write-back leaves the model holding its initial
            # guess while the solve reports success, so `pyo.value(m.x)` gives
            # a plausible wrong number and nothing anywhere indicates it. The
            # error was recorded in a dict key that nothing reads.
            import warnings
            warnings.warn(
                f"the solve succeeded but writing the solution back onto the "
                f"model failed ({type(e).__name__}: {e}). pyo.value() will "
                f"return the initial guess, not the solution; the values are "
                f"in result['x'].", RuntimeWarning, stacklevel=2)

    return res


def _raise_if_infeasible(structures):
    """Turn the detector's infeasibility proof into an error, not a fallback.

    The detector can prove a model infeasible before any solve: a constraint
    with no variables that evaluates false. It says so, and every caller used
    to read that only as "no structure here" and hand the model to a general
    NLP solver, which reported `termination_condition=infeasible` and lost the
    sentence naming the constraint.
    """
    if isinstance(structures, dict) and structures.get('infeasible'):
        from edi.presolve import InfeasibleProblem
        raise InfeasibleProblem(
            structures.get('message', 'the model has no feasible point'))


def _run_diagnostics(structures, level):
    """Structural checks on the way into a solve.

    `level` is 'warn' (default), 'print', or 'off'. The checks cost a fraction
    of a second and catch the modelling errors that otherwise present as a
    strange answer: a variable nothing bounds, one nothing determines, one
    computed and never read. Running them by default is the point -- as
    opt-in tools nobody ran them.

    'warn' reports only what is actionable, so a clean model stays silent.
    """
    if level in (None, 'off', False):
        return None
    import warnings

    from edi.presolve import diagnose

    try:
        rep = diagnose(structures, quiet=True)
    except Exception:
        return None                      # never fail a solve over a check
    if level == 'print':
        print(rep)
        return rep
    problems = []
    if rep.empty_columns:
        problems.append(f"{len(rep.empty_columns)} variables appear in no "
                        f"constraint ({', '.join(rep.empty_columns[:3])})")
    if rep.unbounded_above:
        problems.append(f"{len(rep.unbounded_above)} variables are not upper "
                        f"bounded ({', '.join(rep.unbounded_above[:3])})")
    if rep.unbounded_below:
        problems.append(f"{len(rep.unbounded_below)} variables are not lower "
                        f"bounded ({', '.join(rep.unbounded_below[:3])})")
    if problems:
        warnings.warn(
            "model diagnostics: " + "; ".join(problems)
            + ". Call edi.presolve.diagnose(structures) for the full report.",
            RuntimeWarning, stacklevel=3)
    return rep


def solve(m, solver='auto', convex_backend='ipopt', diagnostics='warn',
          **kwargs):
    """Solve an EDI Formulation, choosing a backend automatically.

    ``solver='auto'`` routes a detected LP, QP, GP or SP to the convex backend
    named by ``convex_backend``, and everything else -- including any
    formulation with a black-box constraint -- to IPOPT. Pass
    ``solver='cvxopt'``, ``'ipopt-convex'`` or ``'ipopt'`` to force one.

    ``convex_backend`` defaults to **ipopt**. A geometric program is solved in
    log space where it is convex, so the global-optimality guarantee is the
    same either way, and a signomial program runs the same PCCP loop with an
    IPOPT geometric-program solve underneath instead of a cvxopt one.

    The default used to be cvxopt, and was changed because cvxopt fails on
    models this repository is built around: on SPaircraft it returns
    ``status='unknown'`` and ``solve_GP`` raises, where the IPOPT route solves
    it. An interior-point method in log space also tolerates the wide variable
    boxes these models carry far better. cvxopt remains available and is still
    the faster choice on a small, well-scaled program.

    ``diagnostics`` runs the structural checks before solving: ``'warn'``
    (the default) reports only what is actionable, so a clean model stays
    silent; ``'print'`` shows the full report; ``'off'`` skips them. They cost
    a fraction of a second and catch the modelling errors that otherwise
    present as a strange answer rather than as an error.

    In every case the solution is written back onto the model, so
    ``pyo.value(m.x)`` returns the optimum after a successful solve.
    """
    from edi.presolve import InfeasibleProblem
    from edi.solvers.ipopt import ipopt_solve

    if diagnostics not in (None, 'off', False):
        try:
            _run_diagnostics(structure_detector(unit_corrector(m),
                                                bounds_as_rows=False),
                             diagnostics)
        except InfeasibleProblem:
            raise
        except Exception:
            pass                         # a check must never block a solve

    if solver == 'cvxopt':
        return cvxopt_solve(m, **kwargs)
    if solver == 'ipopt-convex':
        return _convex_ipopt(m, **kwargs)
    if solver == 'ipopt':
        return ipopt_solve(m, **kwargs)
    if solver != 'auto':
        raise ValueError(f"solver must be 'auto', 'cvxopt', or 'ipopt'; got {solver!r}")

    import warnings

    try:
        # Bind the corrected clone to a local: `structures['variables']` holds
        # only the VarData objects, and if the clone were collected here their
        # parent components would go with it.
        corrected = unit_corrector(m)
        structures = structure_detector(corrected)
        _raise_if_infeasible(structures)
        structured = any(structures[k][0] for k in
                         ('Linear_Program', 'Quadratic_Program',
                          'Geometric_Program', 'Signomial_Program'))
    except InfeasibleProblem:
        # A proof of infeasibility is an answer, not a reason to try a
        # different solver. Falling back here would replace "constraint X is
        # false as written" with whatever a general NLP solver says about a
        # problem that has no solution.
        raise
    except Exception as e:
        warnings.warn(
            f"structure detection failed ({type(e).__name__}: {e}); "
            f"solving with IPOPT instead.", RuntimeWarning, stacklevel=2)
        structured = False

    if structured:
        try:
            if convex_backend == 'ipopt':
                return _convex_ipopt(m, structures=structures, **kwargs)
            return cvxopt_solve(m, **kwargs)
        except Exception as e:
            # Fall through to plain IPOPT, but say why: a silent fallback turns
            # a bug in the structured path into a confusing IPOPT failure.
            warnings.warn(
                f"the structured backend failed ({type(e).__name__}: {e}); "
                f"falling back to IPOPT on the raw model.",
                RuntimeWarning, stacklevel=2)
    return ipopt_solve(m, **kwargs)


def _solve_sp(structures, m, sp_method='sia', **kwargs):
    """Solve a signomial program. SIA by default.

    A signomial has no convex form, so both routes here iterate on convex
    sub-problems; they differ in what they can tell you when they stop.

    ``'sia'`` -- sequential inner approximation. Terminates on a genuine KKT
    residual for the ORIGINAL problem: stationarity, primal feasibility and
    complementarity, all evaluated with the true constraint functions. On
    SPaircraft it reaches a certified KKT point in 149 iterations and about
    twenty seconds.

    ``'pccp'`` -- the penalty convex-concave loop, kept for comparison. It
    stops when the objective stops changing, which says "I stopped moving"
    rather than "I am optimal", and says nothing at all about feasibility. On
    SPaircraft it takes 180 seconds to reach a point that is less feasible than
    SIA's and carries no certificate.

    That is the whole reason for the default: not speed, though SIA is faster
    here, but that one of them can answer whether it arrived.
    """
    from edi.solvers.writeback import write_solution

    if sp_method == 'pccp':
        from edi.solvers.cvxopt.SP import solve_SP
        from edi.solvers.ipopt.convex import solve_gp_rows_ipopt

        def _inner(rows, relations, x0=None):
            return solve_gp_rows_ipopt(rows, relations, x0=x0)

        res = solve_SP(structures, m, gp_solver=_inner,
                       **{k: v for k, v in kwargs.items()
                          if k in ('reltol', 'var_reltol', 'max_iter',
                                   'use_pccp', 'penalty_exponent')})
        res['solver'] = 'ipopt (PCCP, log-transformed subproblems)'
        res['problem_structure'] = 'signomial_program_pccp'
        res['solution'] = write_solution(structures, res, model=m)
        return res

    if sp_method != 'sia':
        raise ValueError(f"sp_method must be 'sia' or 'pccp'; got {sp_method!r}")

    from edi.solvers.ipopt.slcp_bridge import solve_sia

    result = solve_sia(structures, **{k: v for k, v in kwargs.items()
                                      if k in ('x0', 'options', 'sp_form',
                                               'presolve', 'split_equalities')})
    res = {
        'x': list(result.x),
        'primal objective': result.objective,
        'status': 'optimal' if result.converged else result.status,
        'solver': 'ipopt (SIA, sequential inner approximation)',
        'problem_structure': 'signomial_program_sia',
        'converged': result.converged,
        'iterations': result.iterations,
        'max_violation': result.max_violation,
        'stationarity': result.stationarity,
        'complementarity': result.complementarity,
        'result': result,
    }
    if not result.converged:
        import warnings
        warnings.warn(
            f"SIA did not converge: {result.status}. The returned point is "
            f"feasible to {result.max_violation:.2e} with a stationarity "
            f"residual of {result.stationarity:.2e}; it is the best iterate, "
            "not a certified optimum.", RuntimeWarning, stacklevel=3)
    res['solution'] = write_solution(structures, res, model=m)
    return res


def _convex_ipopt(m, structures=None, **kwargs):
    """Solve a structured formulation with IPOPT rather than cvxopt.

    A geometric program is solved in log space, where it is convex, so the
    global-optimality guarantee is preserved. Linear and quadratic programs are
    already convex in their natural variables and go to IPOPT unchanged.
    """
    from edi.solvers.ipopt.convex import solve_gp_ipopt, solve_lp_qp_ipopt
    from edi.solvers.ipopt import ipopt_solve

    if structures is None:
        structures = structure_detector(unit_corrector(m))
    _raise_if_infeasible(structures)
    if structures['Geometric_Program'][0]:
        return solve_gp_ipopt(structures, model=m, **kwargs)
    if structures['Linear_Program'][0] or structures['Quadratic_Program'][0]:
        return solve_lp_qp_ipopt(m, **kwargs)
    if structures['Signomial_Program'][0]:
        return _solve_sp(structures, m, **kwargs)
    # Nothing structured left to exploit.
    return ipopt_solve(m, **kwargs)






