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
# from edi.presolve.structureDetector import structure_detector
from edi.presolve.structureDetector import structure_detector
from edi.presolve.unitCorrector import unit_corrector
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
        from edi.presolve.reductions import InfeasibleProblem
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

    from edi.presolve.reductions import optimization_check

    try:
        rep = optimization_check(structures)
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
            + ". Call edi.presolve.reductions.optimization_check(structures) for the full report.",
            RuntimeWarning, stacklevel=3)
    return rep


def _ipopt_available():
    """Is there any usable IPOPT -- the executable, or cyipopt?

    Asked before dispatching rather than discovered by catching the failure,
    because the two outcomes want different fallbacks. A structured problem
    with no IPOPT should go to cvxopt; a structured problem whose IPOPT path
    has a *bug* should not, since cvxopt would likely hit the same modelling
    error and report it less clearly.

    Cheap and not cached: Pyomo's own availability check is a PATH lookup, and
    caching it would make an IPOPT installed mid-session invisible.
    """
    from edi.solvers.ipopt.ipopt_solver_interface import _executable_available
    if _executable_available('ipopt'):
        return True
    try:
        import pyomo.environ as pyo
        return bool(pyo.SolverFactory('cyipopt').available(exception_flag=False))
    except Exception:
        return False


def _mark_solved(m):
    """Record that this model's variable values are an answer, not a guess.

    `optimization_check(f)` needs to know: the post-solve checks (cancellation, the
    positivity floor) read the current values, and run against an unsolved
    model they describe the author's initial guess while looking exactly like
    they describe the optimum.
    """
    try:
        m._edi_solved = True
    except Exception:
        pass


def _attach_sensitivities(m, res, wanted):
    """Post-solve reporting: holographic checks, then sensitivities.

    Compute sensitivities onto the model, so `f.solution` carries them.

    Default-on because they are the reason to state a quantity as a Constant
    rather than a literal, and as an opt-in nobody ran them. They cost one SVD
    and one walk per active constraint on top of a solve that already
    happened: 2.1s against 18.0s on SPaircraft, and unmeasurable on a model of
    ordinary size.

    Never fatal. A solve that produced an answer must return it even if the
    duals cannot be recovered from it.
    """
    import warnings
    _mark_solved(m)

    # Holographic constraints are checked on EVERY solve, not only when a
    # diagnostic is asked for. An active one means the answer is sitting on a
    # limit that was declared never to bind -- the edge of a fit, a numerical
    # box -- and nothing else about the solve looks wrong when that happens.
    try:
        from edi.solvers.holographic import (format_holographic,
                                             holographic_report)
        active = holographic_report(m)
        n_tot = len(getattr(m, '_holographic', ()) or ())
        if isinstance(res, dict):
            res['holographic_active'] = active
        m._holographic_cache = active
        if active:
            warnings.warn(
                f"{len(active)} of {n_tot} holographic constraints are ACTIVE "
                f"at the solution ("
                + ", ".join(d['name'] for d in active[:3])
                + (", ..." if len(active) > 3 else "")
                + "). These were declared as limits that should not bind, so "
                  "the optimum is on a boundary of the model's validity rather "
                  "than of the design. Read solution.summary() for the detail.",
                RuntimeWarning, stacklevel=3)
    except Exception:
        pass                                  # a check must never lose a solve

    if not wanted or not isinstance(res, dict):
        return res
    try:
        from edi.solvers.sensitivity import sensitivities as _sens
        out = _sens(m)
    except Exception:
        return res
    res['sensitivities'] = out['sensitivities']
    res['sensitivity_detail'] = out
    try:
        m._sensitivity_cache = out['sensitivities']
        m._ambiguous_cache = out.get('ambiguous')
    except Exception:
        pass
    return res


def _apply_start(m, start):
    """Put a starting point onto the model.

    Accepts a FeasibilityResult (or anything with an ``x``), or a plain
    sequence in ``structures['variables']`` order. Writing onto the model is
    not a shortcut: it is where the backends read their initial point, and it
    also means `pyo.value(m.x)` agrees with what the solve was told.
    """
    import numpy as _np

    from edi.solvers.writeback import write_solution

    x = getattr(start, 'x', start)
    x = _np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return
    st = structure_detector(unit_corrector(m))
    n = len(st.get('variables') or [])
    if x.size < n:
        raise ValueError(
            f"start has {x.size} values but the model has {n} variables. It "
            f"must be in structures['variables'] order -- a FeasibilityResult "
            f"already is.")
    write_solution(st, {'x': x[:n]}, model=m)


def solve(m, solver='auto', convex_backend='ipopt', diagnostics='warn',
          sensitivities=True, structures=None, start=None, **kwargs):
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

    ``sensitivities`` computes the sensitivity of the optimum to every Constant
    and attaches it to the result and to ``f.solution``. It is on by default --
    the numbers are the reason to declare a Constant rather than write a
    literal, and the cost is a small fraction of the solve. Pass ``False`` to
    skip it, which is worth doing in a loop that solves the same model many
    times and never reads them.

    ``structures`` accepts a structure you have already detected, and skips the
    detection here. The chain a plain ``solve(f)`` runs is::

        corrected  = unit_corrector(f)          # validate and convert units
        structures = structure_detector(corrected)
        optimization_check(structures)                    # the pre-solve checks
        <backend>(f, structures=structures)     # cvxopt / IPOPT / SLCP / SIA
        sensitivities(f)                        # duals, then write-back

    Running those yourself and passing the result back is worth doing when you
    want to look at the middle of it, and when the walk is expensive: it is
    four to six seconds on SPaircraft against an eleven-second solve, so
    detecting once and reusing it is most of a third off a optimization_check-then-solve.

    Pass structures from ``structure_detector(corrected)`` with its default
    ``bounds_as_rows=True``. The split form is for the presolve, and the
    backends read bounds out of the rows -- handing them the split form is
    caught and refused rather than silently solving an unbounded relaxation.
    ``optimization_check`` reads either form, so the default is the one to share.

    ``start`` sets the point the solve begins from, which the backends
    otherwise take from the model's current values. It accepts a
    :class:`~edi.presolve.feasibilityCheck.FeasibilityResult`, so the feasibility
    solve composes with this one::

        result = feasibility(f)
        if result:
            solve(f, start=result)

    -- worth doing on a model where the author's guesses are not feasible, and
    the only way to start from a feasible point without hand-editing every
    guess. A plain sequence or array in ``structures['variables']`` order works
    too.

    In every case the solution is written back onto the model, so
    ``pyo.value(m.x)`` returns the optimum after a successful solve.
    """
    import warnings

    from edi.presolve.reductions import InfeasibleProblem
    from edi.solvers.ipopt import ipopt_solve
    from edi.presolve.unitCorrector import UnitMismatch

    # Detect once and use the result for both the checks and the solve. These
    # used to be two separate walks of the model, because `optimization_check` needs
    # bounds separated from the rows and the structured backends read them out
    # of the rows -- but `optimization_check` folds single-variable rows into bounds
    # itself, so it reads either form and returns the same report. The walk is
    # not cheap: on SPaircraft it is four to six seconds, against an
    # eleven-second solve.
    want_checks = diagnostics not in (None, 'off', False)
    # Bind the corrected clone to a local: `structures['variables']` holds only
    # the VarData objects, and if the clone were collected here their parent
    # components would go with it.
    if start is not None:
        # Applied by writing onto the model, because that is where every
        # backend reads its initial point from. Done before detection so the
        # detected structures carry the new values.
        _apply_start(m, start)

    corrected = None
    detection_failed = None
    if structures is not None:
        # Supplied by the caller. Check the form now rather than letting a
        # backend discover it: the failure mode otherwise is an answer to a
        # problem with no variable bounds, which looks entirely reasonable.
        if isinstance(structures, dict) and structures.get('bounds') is not None:
            raise ValueError(
                "solve() was given structures built with bounds_as_rows=False. "
                "The backends read variable bounds out of the constraint rows, "
                "so solving these would ignore every bound and answer a "
                "different question. Re-run structure_detector(corrected) with "
                "its default bounds_as_rows=True; optimization_check() reads that form "
                "too.")
        _raise_if_infeasible(structures)
    elif want_checks or solver == 'auto':
        try:
            corrected = unit_corrector(m)
            structures = structure_detector(corrected)
            _raise_if_infeasible(structures)
        except InfeasibleProblem:
            # A proof of infeasibility is an answer, not a reason to try a
            # different solver. Falling back here would replace "constraint X
            # is false as written" with whatever a general NLP solver says
            # about a problem that has no solution.
            raise
        except UnitMismatch:
            # Nor is a dimensional error. A model whose constraints do not
            # balance dimensionally has no meaning to solve for, and IPOPT
            # will happily return numbers for it -- observed on an example
            # whose coordinate arrays were bare floats standing for metres:
            # the fallback reported lengths of 1e5 m with no indication that
            # anything was wrong. The unit report says which constraints and
            # what the correction is; that is the answer here.
            raise
        except Exception as e:
            detection_failed = e

    if want_checks and structures is not None:
        try:
            _run_diagnostics(structures, diagnostics)
        except InfeasibleProblem:
            raise
        except Exception:
            pass                         # a check must never block a solve

    if solver == 'cvxopt':
        return _attach_sensitivities(m, cvxopt_solve(m, **kwargs), sensitivities)
    if solver == 'ipopt-convex':
        return _attach_sensitivities(m, _convex_ipopt(m, **kwargs), sensitivities)
    if solver == 'ipopt':
        return _attach_sensitivities(m, ipopt_solve(m, **kwargs), sensitivities)
    if solver != 'auto':
        raise ValueError(f"solver must be 'auto', 'cvxopt', or 'ipopt'; got {solver!r}")

    if detection_failed is not None:
        warnings.warn(
            f"structure detection failed ({type(detection_failed).__name__}: "
            f"{detection_failed}); solving with IPOPT instead.",
            RuntimeWarning, stacklevel=2)
        structured = False
    else:
        structured = any(structures[k][0] for k in
                         ('Linear_Program', 'Quadratic_Program',
                          'Geometric_Program', 'Signomial_Program'))

    if structured:
        backend = convex_backend
        if backend == 'ipopt' and not _ipopt_available():
            # A detected LP/QP/GP/SP does not need IPOPT -- cvxopt solves the
            # same convex problem to the same optimum. Falling back to it is
            # far better than failing, but say so: IPOPT is the default for
            # good reasons (it is faster on large models and is the only route
            # for a black-box constraint), so a silent downgrade would hide a
            # missing install for as long as the models stayed convex.
            warnings.warn(
                'no usable IPOPT installation was found; solving this '
                'structured problem with cvxopt instead. The answer is the '
                'same, but IPOPT is the default backend and is required for '
                'black-box constraints. Install the ipopt executable or '
                '`pip install cyipopt` -- see docs/ipopt.rst.',
                RuntimeWarning, stacklevel=2)
            backend = 'cvxopt'
        try:
            if backend == 'ipopt':
                return _attach_sensitivities(
                    m, _convex_ipopt(m, structures=structures, **kwargs),
                    sensitivities)
            return _attach_sensitivities(m, cvxopt_solve(m, **kwargs),
                                         sensitivities)
        except Exception as e:
            # Fall through, but say why: a silent fallback turns a bug in the
            # structured path into a confusing failure further down.
            where = ('IPOPT on the raw model' if _ipopt_available()
                     else 'cvxopt' if backend == 'ipopt' else 'nothing else')
            warnings.warn(
                f"the structured backend failed ({type(e).__name__}: {e}); "
                f"falling back to {where}.",
                RuntimeWarning, stacklevel=2)
            if not _ipopt_available() and backend == 'ipopt':
                return _attach_sensitivities(m, cvxopt_solve(m, **kwargs),
                                             sensitivities)

    if not _ipopt_available():
        # Nothing left to try. cvxopt cannot take a general NLP, so this is a
        # real dead end rather than another fallback -- name it as one.
        raise RuntimeError(
            'this model needs IPOPT and no usable installation was found. '
            + ('It is not a detected LP, QP, GP or SP, so cvxopt cannot solve '
               'it. ' if not structured else '')
            + "Install the 'ipopt' executable and put it on PATH, or "
              '`pip install cyipopt`. See docs/ipopt.rst.')
    return _attach_sensitivities(m, ipopt_solve(m, **kwargs), sensitivities)


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






