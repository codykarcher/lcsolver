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
# from lcsolver.presolve.structureDetector import structure_detector
from lcsolver.presolve.structureDetector import structure_detector
from lcsolver.presolve.unitCorrector import unit_corrector
from lcsolver.solvers.writeback import write_solution


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
        # from lcsolver.solvers.cvxopt import solve_LP
        from lcsolver.solvers.cvxopt.LP import solve_LP
        res = solve_LP(structures)
        res['problem_structure'] = 'linear_program'
    elif structures['Quadratic_Program'][0]:
        # from lcsolver.solvers.cvxopt import solve_QP
        from lcsolver.solvers.cvxopt.QP import solve_QP
        res = solve_QP(structures)
        res['problem_structure'] = 'quadratic_program'
    elif structures['Geometric_Program'][0]:
        # from lcsolver.solvers.cvxopt import solve_GP
        from lcsolver.solvers.cvxopt.GP import solve_GP
        res = solve_GP(structures)
        res['problem_structure'] = 'geometric_program'
    elif structures['Signomial_Program'][0]:
        # from lcsolver.solvers.cvxopt.SP import solve_SP
        from lcsolver.solvers.cvxopt.SP import solve_SP
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
        from lcsolver.presolve.reductions import InfeasibleProblem
        raise InfeasibleProblem(
            structures.get('message', 'the model has no feasible point'))


class PresolveError(RuntimeError):
    """The pre-solve checks found the stated problem ill-posed.

    Raised (batched -- every finding in one message) before any solver runs.
    Pass ``diagnostics='warn'`` to solve() to demote this to a warning.
    """


class SolveResult(dict):
    """What ``solve`` returns: the solver's result dict with attribute access.

    Every existing key lookup (``res['x']``, ``res['status']``) works
    unchanged -- this IS a dict. Attribute access reaches the same keys
    (``res.status``), plus two conveniences: ``res.solution`` is the rich
    printable :class:`~lcsolver.objects.solution.Solution` attached to the
    model by the solve (``res['solution']`` remains the flat name->value
    write-back dict), and ``res.objective`` reads the objective value.
    """

    def __init__(self, data=None, model=None):
        super().__init__(data or {})
        self.__dict__['_model'] = model

    @property
    def solution(self):
        sol = getattr(self.__dict__.get('_model'), 'solution', None)
        return sol if sol is not None else self.get('solution')

    @property
    def objective(self):
        return self.get('primal objective', self.get('objective'))

    def summary(self, *args, **kwargs):
        """The solution summary -- same as ``f.solution.summary(...)``."""
        sol = self._rich()
        return sol.summary(*args, **kwargs)

    # -- named access with units -------------------------------------------
    #
    # One calling convention for all four accessors:
    #   sol.variables()                 -> the full flat dict, keyed by the
    #                                      dotted display name ('wing.AR');
    #                                      never nested sub-dicts
    #   sol.variables('wing.AR')        -> the single quantity
    #   sol.variables(['wing.AR', 'S']) -> {name: quantity} for those names
    # Values come back as PINT quantities (pyomo's own registry:
    # pyomo.environ.units.pint_registry), so a dict of them prints readably
    # and `.to('ft')` / `.magnitude` work directly; a dimensionless quantity
    # comes back as a plain float. Names are accepted in dotted display form
    # ('wing.AR') or the flat internal form ('wing_AR').

    def _rich(self):
        sol = self.solution
        if sol is None or not hasattr(sol, 'variables'):
            raise AttributeError(
                'this result carries no rich Solution to read from')
        return sol

    @staticmethod
    def _quantity(value, units):
        if units is None or str(units) in ('dimensionless', 'None', ''):
            return value
        from pyomo.environ import units as _pu
        return value * _pu.pint_registry(str(units))

    @staticmethod
    def _pick(sol, mapping, names, one):
        """Apply the shared calling convention to ``mapping``.

        ``mapping`` is {flat_name: payload}; ``one(flat_name)`` renders a
        single payload. Lookup accepts flat or dotted names.
        """
        def resolve(name):
            if name in mapping:
                return name
            for k in mapping:
                if sol.display_name(k) == name:
                    return k
            raise KeyError(
                f'{name!r} is not in this solution; known names: '
                + ', '.join(sorted(sol.display_name(k) for k in mapping)[:8])
                + (', ...' if len(mapping) > 8 else ''))

        if names is None:
            return {sol.display_name(k): one(k) for k in mapping}
        if isinstance(names, str):
            return one(resolve(names))
        return {name: one(resolve(name)) for name in names}

    def variables(self, names=None):
        """Design-variable values with their units. See the class note."""
        sol = self._rich()
        return self._pick(sol, sol.variables, names,
                          lambda k: self._quantity(sol.variables[k].value,
                                                   sol.variables[k].units))

    def constants(self, names=None):
        """Constant values with their units. Same convention as variables()."""
        sol = self._rich()
        return self._pick(sol, sol.constants, names,
                          lambda k: self._quantity(sol.constants[k].value,
                                                   sol.constants[k].units))

    def sensitivities(self, names=None):
        """Log-log sensitivities: d log(objective) / d log(constant).

        The scaled percent-per-percent numbers of the printed table, as plain
        floats. Same calling convention as variables(). For the dimensioned
        d(objective)/d(constant), use dimensioned_sensitivities().
        """
        sol = self._rich()
        sens = sol.sensitivities or {}
        return self._pick(sol, sens, names, lambda k: sens[k])

    def dimensioned_sensitivities(self, names=None):
        """Dimensioned sensitivities: d(objective) / d(constant), with units.

        Converts the log-log sensitivity S = dln(f)/dln(c) to
        df/dc = S * f / c, carried in objective-units per constant-units.
        Same calling convention as variables().
        """
        sol = self._rich()
        sens = sol.sensitivities or {}
        if sol.objective is None:
            raise AttributeError(
                'this solution carries no objective value, so sensitivities '
                'cannot be dimensioned')

        def one(k):
            from pyomo.environ import units as _pu
            entry = sol.constants[k]
            value = sens[k] * sol.objective / entry.value
            obj_u, c_u = sol.objective_units, entry.units
            obj_dimless = (obj_u is None
                           or str(obj_u) in ('dimensionless', 'None', ''))
            c_dimless = (c_u is None
                         or str(c_u) in ('dimensionless', 'None', ''))
            if obj_dimless and c_dimless:
                return value
            if c_dimless:
                return value * _pu.pint_registry(str(obj_u))
            if obj_dimless:
                return value / _pu.pint_registry(str(c_u))
            return (value * _pu.pint_registry(str(obj_u))
                    / _pu.pint_registry(str(c_u)))

        return self._pick(sol, sens, names, one)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _run_diagnostics(structures, level):
    """Structural checks on the way into a solve.

    ``'error'`` (the default): findings that make the stated problem
    ill-posed -- a variable in no constraint, a variable unbounded above or
    below -- raise a :class:`PresolveError` naming every one, batched, with
    the fix. A clean model stays silent. ``'warn'`` demotes those errors to a
    RuntimeWarning (the old behavior); ``'print'`` prints the full report;
    ``'off'`` skips the checks.

    The checks cost a fraction of a second and catch the modelling errors
    that otherwise present as a strange answer. Variables computed by a
    black-box constraint are accounted for and do not trip the gate.
    """
    if level in (None, 'off', False):
        return None
    import warnings

    from lcsolver.presolve.reductions import presolve_check

    try:
        rep = presolve_check(structures)
    except Exception:
        return None                # never fail a solve over a broken check
    if level == 'print':
        print(rep)
        return rep
    problems = []
    if rep.empty_columns:
        problems.append(
            f"{len(rep.empty_columns)} variables appear in no constraint "
            f"({', '.join(rep.empty_columns[:3])}) -- remove them or "
            "constrain them")
    if rep.unbounded_above:
        problems.append(
            f"{len(rep.unbounded_above)} variables are not upper bounded "
            f"({', '.join(rep.unbounded_above[:3])}) -- add bounds=[lo, hi] "
            "to the Variable or a constraint that limits them")
    if rep.unbounded_below:
        problems.append(
            f"{len(rep.unbounded_below)} variables are not lower bounded "
            f"({', '.join(rep.unbounded_below[:3])}) -- add bounds=[lo, hi] "
            "to the Variable or a constraint that limits them")
    if problems:
        msg = ("pre-solve check: " + "; ".join(problems)
               + ". Call lcsolver.presolve.reductions.presolve_check(f) for the "
                 "full report"
               + (", or pass diagnostics='warn' to solve() to demote this "
                  "error to a warning." if level == 'error' else "."))
        if level == 'error':
            raise PresolveError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
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
    from lcsolver.solvers.ipopt.ipopt_solver_interface import _executable_available
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

    # Wrap the raw backend dict so callers get attribute access and
    # `res.solution` (the rich printable Solution) -- see SolveResult. All
    # further writes below go through normal dict item assignment either way.
    if isinstance(res, dict) and not isinstance(res, SolveResult):
        res = SolveResult(res, model=m)

    # Holographic constraints are checked on EVERY solve, not only when a
    # diagnostic is asked for. An active one means the answer is sitting on a
    # limit that was declared never to bind -- the edge of a fit, a numerical
    # box -- and nothing else about the solve looks wrong when that happens.
    try:
        from lcsolver.solvers.holographic import (format_holographic,
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

    # The post-solve quality checks -- variables the optimum does not
    # determine, cancelling signomial terms, variables on the positivity
    # floor -- run automatically on a converged solve and ride back on the
    # result. Informational, not warnings: only the floor check (a symptom of
    # the ALGORITHM pinning a variable, not the model) warns.
    try:
        status = str(res.get('status', '')) if isinstance(res, dict) else ''
        converged = (res.get('converged') is True
                     or status.startswith('optimal')
                     or 'converged' in status) if isinstance(res, dict) else False
        if converged:
            from lcsolver.presolve.reductions import postsolve_check
            post = postsolve_check(m)
            res['quality'] = {'degenerate': post.degenerate,
                              'cancelling': post.cancelling,
                              'at_floor': post.at_floor}
            res['quality_text'] = post.post_solve_text()
            if post.at_floor:
                warnings.warn(
                    f"{len(post.at_floor)} variables are resting on the "
                    "solver's positivity floor ("
                    + ", ".join(nm for nm, _ in post.at_floor[:3])
                    + (", ..." if len(post.at_floor) > 3 else "")
                    + ") -- pinned by the algorithm, not the model. See "
                      "res['quality_text'] for the post-solve report.",
                    RuntimeWarning, stacklevel=3)
    except Exception:
        pass                                  # a check must never lose a solve

    if not wanted or not isinstance(res, dict):
        return res
    try:
        from lcsolver.solvers.sensitivity import sensitivities as _sens
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

    from lcsolver.solvers.writeback import write_solution

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


def solve(m, solver='auto', convex_backend='ipopt', diagnostics='error',
          sensitivities=True, structures=None, start=None, **kwargs):
    """Solve an LCsolver Formulation, choosing a backend automatically.

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

    ``diagnostics`` runs the structural checks before solving: ``'error'``
    (the default) raises a :class:`PresolveError` -- every finding batched in
    one message -- when the stated problem is ill-posed (variables in no
    constraint, variables unbounded either way); a clean model stays silent.
    ``'warn'`` demotes those errors to a RuntimeWarning; ``'print'`` shows
    the full report; ``'off'`` skips the checks. They cost a fraction of a
    second and catch the modelling errors that otherwise present as a
    strange answer rather than as an error.

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
    :class:`~lcsolver.presolve.feasibilityCheck.FeasibilityResult`, so the feasibility
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

    from lcsolver.presolve.reductions import InfeasibleProblem
    from lcsolver.solvers.ipopt import ipopt_solve
    from lcsolver.presolve.unitCorrector import UnitMismatch

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
        except (InfeasibleProblem, PresolveError):
            # The gate is the point: an ill-posed problem stops here, before
            # any solver spends time on it.
            raise
        except Exception:
            pass                         # a broken check must not block a solve

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

    from lcsolver.solvers.ipopt.ipopt_solver_interface import _has_greybox
    if _has_greybox(m):
        # A black-box (grey-box) constraint cannot enter the algebraic convex
        # backends, and letting the structured route run without it would
        # solve a relaxation and report it as the optimum. When the algebraic
        # part is a GP or SP, the model as a whole is an SP with opaque rows:
        # route it to SIA, which imposes each black box through its
        # linearization inside the trust-region loop (the grey-box rows are
        # appended in slcp_bridge.build_problem). Anything else falls through
        # to raw IPOPT via cyipopt, the only other route that can evaluate a
        # Python black box.
        if (structures is not None and detection_failed is None
                and (structures['Geometric_Program'][0]
                     or structures['Signomial_Program'][0])):
            return _attach_sensitivities(
                m, _solve_sp(structures, m, **kwargs), sensitivities)
        if not _ipopt_available():
            raise RuntimeError(
                'this model has black-box constraints and its algebraic part '
                'is not a detected GP/SP, so it needs the raw IPOPT route -- '
                'and no usable IPOPT installation was found. '
                '`pip install cyipopt`; see docs/ipopt.rst.')
        return _attach_sensitivities(m, ipopt_solve(m, **kwargs),
                                     sensitivities)

    cvxopt_failure = None
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
            if backend == 'cvxopt':
                cvxopt_failure = e
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
        if cvxopt_failure is not None:
            raise RuntimeError(
                f'cvxopt failed on this structured problem '
                f'({type(cvxopt_failure).__name__}: {cvxopt_failure}), and '
                'no usable IPOPT installation was found to try instead. '
                'IPOPT is the preferred backend -- it solves models cvxopt '
                'fails on. Install the ipopt executable and put it on PATH, '
                'or `pip install cyipopt`. See docs/ipopt.rst.'
            ) from cvxopt_failure
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
    from lcsolver.solvers.writeback import write_solution

    if sp_method == 'pccp':
        from lcsolver.solvers.cvxopt.SP import solve_SP
        from lcsolver.solvers.ipopt.convex import solve_gp_rows_ipopt

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

    from lcsolver.solvers.ipopt.slcp_bridge import solve_sia

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
        status = str(result.status or '')
        msg = (f"SIA did not converge: {status}. The returned point is "
               f"feasible to {result.max_violation:.2e} with a stationarity "
               f"residual of {result.stationarity:.2e}; it is the best "
               "iterate, not a certified optimum.")
        # Say what to DO about it, keyed on how it failed.
        remedies = []
        if 'phase 1' in status:
            report = getattr(result, 'infeasibility_report', None)
            if report:
                res['infeasibility_report'] = report
                msg += ("\nWhere feasibility fails (the L1-elastic optimum's "
                        "unclosable rows):\n" + str(report))
            remedies.append(
                'the blocking constraints named above are where the model '
                'is inconsistent -- check their signs, units and bounds; if '
                'the model should be feasible, start closer (better guesses '
                'or start=)')
        if 'did not converge within' in status:
            remedies.append(
                'raise options=SIAOptions() max_iterations, or try the '
                'conservative mode (SIAOptions with condense_numerator='
                'False), which trades speed for a feasibility-preserving '
                'iteration')
        if 'trust region collapsed' in status:
            remedies.append(
                'try the conservative mode (SIAOptions with '
                'condense_numerator=False) and run postsolve_check(f) -- a '
                'collapsing trust region often means a degenerate or '
                'unopposed variable')
        if 'sub-problem failure' in status:
            remedies.append(
                'check variable scaling and initial guesses; a sub-problem '
                'failure is usually a wildly scaled column or a guess '
                'decades from feasibility')
        if remedies:
            msg += '\nWhat to try: ' + '; '.join(remedies) + '.'
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    res['solution'] = write_solution(structures, res, model=m)
    return res


def _convex_ipopt(m, structures=None, **kwargs):
    """Solve a structured formulation with IPOPT rather than cvxopt.

    A geometric program is solved in log space, where it is convex, so the
    global-optimality guarantee is preserved. Linear and quadratic programs are
    already convex in their natural variables and go to IPOPT unchanged.
    """
    from lcsolver.solvers.ipopt.convex import solve_gp_ipopt, solve_lp_qp_ipopt
    from lcsolver.solvers.ipopt import ipopt_solve

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






