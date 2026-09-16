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

import numpy as np
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.dependencies import attempt_import
# from lcsolver.presolve.structureDetector import structure_detector
from lcsolver.presolve.structureDetector import structure_detector
from lcsolver.presolve.unitCorrector import unit_corrector
from lcsolver.postsolve.writeback import write_solution
from lcsolver.core.errors import SolverUnavailable


cvxopt, cvxopt_available = attempt_import( "cvxopt" )


def _require_cvxopt():
    """Raise only when a cvxopt backend is actually asked for.

    Checked here, not at module import: raising at import made a missing
    cvxopt (--no-deps installs) break `import lcsolver` entirely.
    """
    if not cvxopt_available:
        raise ImportError(
            "the cvxopt backend requires cvxopt, which is not importable. It "
            "is a dependency of lcsolver, so this usually means the install "
            "was done with --no-deps or cvxopt was removed afterwards; "
            "`pip install cvxopt` restores it. The IPOPT backend "
            "(convex_backend='ipopt', the default) does not need cvxopt.")


def cvxopt_solve(m, write_back=True, structures=None):
    _require_cvxopt()
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['maxiters'] = 100
    cvxopt.solvers.options['feastol'] = 1e-6
    cvxopt.printing.options['width'] = -1

    # solve() hands down its already-detected structures; a direct caller detects here
    if structures is None:
        structures = structure_detector(unit_corrector(m))
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
        # from lcsolver.solvers.sequential.pccp import solve_SP
        from lcsolver.solvers.sequential.pccp import solve_SP
        res = solve_SP(structures,m)
        res['problem_structure'] = 'signomial_program_pccp'
    else:
        raise ValueError('Could not convert the formulation to a valid CVXOPT structure (LP,QP,GP,SP)')

    # cvxopt can return status 'unknown' with no exception; don't let it pass for a solution
    if res.get('status') not in ('optimal', None):
        import warnings
        warnings.warn(
            f"[LC-W205] cvxopt returned status={res.get('status')!r}; the reported point may "
            f"be infeasible or non-optimal. Consider convex_backend='ipopt'.",
            RuntimeWarning, stacklevel=2)

    # record which structure was solved; `sensitivities` reads this to tell
    # convex duals from a signomial sequence's local approximation
    try:
        m._edi_last_problem_structure = res['problem_structure']
    except Exception:
        pass

    # write the solution back onto the model; without this pyo.value(m.x)
    # still returns the initial guess (backends work in a transformed space)
    if write_back:
        try:
            res['solution'] = write_solution(structures, res, model=m)
        except Exception as e:                      # never lose a good solve
            res['solution'] = None
            res['writeback_error'] = f"{type(e).__name__}: {e}"
            # warn: a silent failed write-back leaves pyo.value() returning a
            # plausible wrong number
            import warnings
            warnings.warn(
                f"[LC-W206] the solve succeeded but writing the solution back onto the "
                f"model failed ({type(e).__name__}: {e}). pyo.value() will "
                f"return the initial guess, not the solution; the values are "
                f"in result['x'].", RuntimeWarning, stacklevel=2)

    return res


def _raise_if_infeasible(structures):
    """Turn the detector's infeasibility proof into an error, not a fallback.

    Falling through to an NLP solver used to lose the sentence naming the
    false constraint.
    """
    if isinstance(structures, dict) and structures.get('infeasible'):
        from lcsolver.presolve.reductions import InfeasibleProblem
        raise InfeasibleProblem(
            structures.get('message', 'the model has no feasible point'))


class PresolveError(RuntimeError):
    """Pre-solve checks found the problem ill-posed. Raised before any solver
    runs, all findings batched; diagnostics='warn' demotes it to a warning."""


class SolveResult(dict):
    """The solver's result dict with attribute access. Still a dict, so
    res['x'] etc. work unchanged; res.solution is the rich printable Solution
    (res['solution'] stays the flat write-back dict), res.objective the value.
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
        """The objective value, with units (pint), like the accessors."""
        sol = self.__dict__.get('_model')
        sol = getattr(sol, 'solution', None)
        if sol is not None and getattr(sol, 'objective', None) is not None:
            return self._quantity(sol.objective,
                                  getattr(sol, 'objective_units', None))
        return self.get('primal objective', self.get('objective'))

    @property
    def messages(self):
        """Warnings the solve raised, captured instead of printed."""
        return self.get('messages', [])

    @property
    def optimality_status(self):
        """True only for a certified optimum ('optimal', SIA KKT convergence,
        or an explicit converged flag); False for best-iterate and ambiguous."""
        if self.get('converged') is True:
            return True
        status = str(self.get('status', '')).lower()
        return status == 'optimal' or status.startswith('converged')

    @property
    def report(self):
        """The Report section text: what was detected/prescribed, what ran."""
        sol = self.solution
        if sol is None or not getattr(sol, 'report', None):
            return None
        return '\n'.join(line.strip() for line in sol._report_lines())

    def summary(self, *args, **kwargs):
        """The solution summary -- same as ``f.solution.summary(...)``."""
        sol = self._rich()
        return sol.summary(*args, **kwargs)

    # -- named access with units -------------------------------------------
    #
    # One calling convention for all four accessors:
    #   sol.variables()                 -> full flat dict, keyed by dotted
    #                                      display name ('wing.AR')
    #   sol.variables('wing.AR')        -> the single quantity
    #   sol.variables(['wing.AR', 'S']) -> {name: quantity} for those names
    # Values are pint quantities (pyomo.environ.units.pint_registry), so
    # .to('ft') / .magnitude work; a dimensionless scalar is a plain float.
    # An array variable is ONE entry with an ndarray magnitude, pint even
    # when dimensionless. Names accepted dotted ('wing.AR') or flat ('wing_AR').

    def _rich(self):
        sol = self.solution
        if sol is None or not hasattr(sol, 'variables'):
            raise AttributeError(
                'this result carries no rich Solution to read from')
        return sol

    @staticmethod
    def _quantity(value, units):
        from pyomo.environ import units as _pu
        if units is None or str(units) in ('dimensionless', 'None', ''):
            # dimensionless scalar -> plain float; dimensionless ARRAY stays
            # pint, since a bare ndarray breaks callers reading .magnitude
            if isinstance(value, np.ndarray):
                return value * _pu.pint_registry.dimensionless
            return value
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


def _run_diagnostics(structures, level, peel_outputs=False):
    """Structural checks before a solve. 'error' raises PresolveError on
    ill-posed findings (unconstrained/unbounded variables), batched; 'warn'
    demotes to RuntimeWarning; 'print' shows the report; 'off' skips.
    Black-box-computed variables do not trip the gate.
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
    from lcsolver.core import codes

    # Unbounded gates as an error EXCEPT for output-only variables
    # (rep.output_columns; grey-box-fed already excluded): they can't affect
    # the optimum, so demote to a note and let the central peel in
    # _solve_impl remove them. Demote ONLY when the peel will run
    # (peel_outputs) -- with presolve bypassed it stays an error.
    peelable = set(rep.output_columns or []) if peel_outputs else set()
    unbounded_above = [n for n in rep.unbounded_above if n not in peelable]
    unbounded_below = [n for n in rep.unbounded_below if n not in peelable]
    demoted = [n for n in rep.unbounded_above + rep.unbounded_below
               if n in peelable]
    if demoted:
        warnings.warn(codes.tag(codes.OUTPUT_ONLY,
            f"pre-solve note: {len(demoted)} variable(s) are computed by a "
            f"constraint nothing else uses ({', '.join(demoted[:3])}) -- "
            "not relevant to the optimum, so not gated as an error; the "
            "variable and its defining constraint are peeled from the "
            "solve, and the value is recovered from that constraint "
            "afterwards. Remove them (or put the variable to use) to "
            "silence this note."), RuntimeWarning, stacklevel=3)

    problems = []
    if rep.empty_columns:
        problems.append(
            f"{len(rep.empty_columns)} variables appear in no constraint "
            f"({', '.join(rep.empty_columns[:3])}) -- remove them or "
            "constrain them")
    if unbounded_above:
        problems.append(
            f"{len(unbounded_above)} variables are not upper bounded "
            f"({', '.join(unbounded_above[:3])}) -- add bounds=[lo, hi] "
            "to the Variable or a constraint that limits them")
    if unbounded_below:
        problems.append(
            f"{len(unbounded_below)} variables are not lower bounded "
            f"({', '.join(unbounded_below[:3])}) -- add bounds=[lo, hi] "
            "to the Variable or a constraint that limits them")
    if problems:
        msg = ("pre-solve check: " + "; ".join(problems)
               + ". Call lcsolver.presolve.reductions.presolve_check(f) for the "
                 "full report"
               + (", or pass diagnostics='warn' to solve() to demote this "
                  "error to a warning." if level == 'error' else "."))
        if level == 'error':
            raise PresolveError(codes.tag(codes.ILL_POSED, msg))
        warnings.warn(codes.tag(codes.PRESOLVE_FINDINGS, msg),
                      RuntimeWarning, stacklevel=3)
    return rep


def _ipopt_available():
    """Any usable IPOPT -- the executable, or cyipopt?

    Asked before dispatching, not caught after: no-IPOPT should fall back to
    cvxopt, an IPOPT-path bug should not. Not cached, so a mid-session
    install is seen.
    """
    from lcsolver.solvers.ipopt.NLP import _executable_available
    if _executable_available('ipopt'):
        return True
    try:
        import pyomo.environ as pyo
        return bool(pyo.SolverFactory('cyipopt').available(exception_flag=False))
    except Exception:
        return False


def _mark_solved(m):
    """Record that the model's values are an answer, not a guess -- the
    post-solve checks read current values and must not describe an initial
    guess as if it were the optimum."""
    try:
        m._edi_solved = True
    except Exception:
        pass





def _check_structures_match(structures, m):
    """Refuse structures detected from a DIFFERENT formulation.

    Undetectable downstream: the backends read the structures' clone, so a
    same-shape model writes another model's answer with no error. Once cost a
    deck sweep returning the first deck's weight for every case.
    """
    clone = structures.get('model') if hasattr(structures, 'get') else None
    stamped = getattr(clone, '_edi_source_identity', None)
    mine = getattr(m, '_edi_identity', None)
    # `mine is None` is a mismatch, not a free pass: detection stamps its
    # source model, so an unstamped model can't be where these came from
    if stamped is not None and stamped != mine:
        raise ValueError(
            "structures= were detected from a different formulation than the "
            "one being solved. The backends read their variables off those "
            "structures, so this would return the other model's answer with no "
            "error. Re-detect against this model: "
            "structures=structure_detector(unit_corrector(f)).")
    stamped_rev = getattr(clone, '_edi_source_revision', None)
    mine_rev = getattr(m, '_edi_revision', 0)
    if stamped_rev is not None and stamped_rev != mine_rev:
        raise ValueError(
            f"structures= were detected before load_constants was called "
            f"(structures at revision {stamped_rev}, model at {mine_rev}). "
            "Their clone still holds the old constant values, so this would "
            "answer the previous deck's question. Re-detect after loading the "
            "deck: structures=structure_detector(unit_corrector(f)).")


def _attach_sensitivities(m, res, wanted, skip_degeneracy_check=False,
                          structures=None):
    """Post-solve reporting: holographic checks, then sensitivities onto the
    model so f.solution carries them.

    Default-on: sensitivities are the reason to declare a Constant, and as an
    opt-in nobody ran them. Cost 2.1s against 18.0s on SPaircraft. Never
    fatal -- a solve that produced an answer must return it.
    """
    import warnings
    _mark_solved(m)

    # wrap the raw backend dict for attribute access (see SolveResult);
    # writes below use normal dict item assignment either way
    if isinstance(res, dict) and not isinstance(res, SolveResult):
        res = SolveResult(res, model=m)

    # stash how the solve went for the solution's Report section
    try:
        req = getattr(m, '_solve_request', None) or {}
        n_greybox = 0
        try:
            from pyomo.contrib.pynumero.interfaces.external_grey_box import (
                ExternalGreyBoxBlock,
            )
            n_greybox = sum(1 for _ in m.component_data_objects(
                ExternalGreyBoxBlock, descend_into=True, active=True))
        except Exception:
            pass
        if isinstance(res, dict):
            m._solve_report = {
                'structure': res.get('problem_structure'),
                'solver': res.get('solver'),
                'status': res.get('status'),
                'gp_form': res.get('gp form'),
                'requested_solver': req.get('solver'),
                'convex_backend': req.get('convex_backend'),
                'greybox': n_greybox,
            }
    except Exception:
        pass

    # holographic constraints are checked on EVERY solve: an active one means
    # the answer sits on a limit declared never to bind, and nothing else
    # about the solve looks wrong when that happens
    try:
        from lcsolver.postsolve.holographic import (format_holographic,
                                             holographic_report,
                                             holographic_total)
        active = holographic_report(m)
        n_tot = holographic_total(m)
        if isinstance(res, dict):
            res['holographic_active'] = active
        m._holographic_cache = active
        if active:
            binds = "; ".join(
                (f"{d['name']}: {d['expr']}" if d.get('expr') else d['name'])
                + f" (at {d['value']:.6g}, margin {d['margin']:+.2e})"
                for d in active[:3])
            warnings.warn(
                f"[LC-W301] {len(active)} of {n_tot} holographic constraints are ACTIVE "
                f"at the solution -- {binds}"
                + ("; ..." if len(active) > 3 else "")
                + ". These were declared as limits that should not bind, so "
                  "the optimum is on a boundary of the model's validity rather "
                  "than of the design. Read solution.summary() for the detail.",
                RuntimeWarning, stacklevel=3)
    except Exception:
        pass                                  # a check must never lose a solve

    # post-solve quality checks (degenerate variables, cancelling terms,
    # positivity floor) run on a converged solve and ride back on the result;
    # only the floor check warns -- it's the algorithm's fault, not the model's
    try:
        status = str(res.get('status', '')) if isinstance(res, dict) else ''
        converged = (res.get('converged') is True
                     or status.startswith('optimal')
                     or 'converged' in status) if isinstance(res, dict) else False
        if converged:
            from lcsolver.presolve.reductions import postsolve_check
            # feed the checks the already-detected structures; handing the
            # model re-detects from scratch, about half the wall clock of the
            # whole solve on a few-thousand-row model. write_solution already
            # brought the detected clone to the solution.
            reuse = (structures is not None
                     and getattr(structures.get('model', None),
                                 '_edi_solved', False))
            post = postsolve_check(
                structures if reuse else m,
                skip_degeneracy_check=skip_degeneracy_check)
            res['quality'] = {'degenerate': post.degenerate,
                              'cancelling': post.cancelling,
                              'at_floor': post.at_floor}
            res['quality_text'] = post.post_solve_text()
            if post.at_floor:
                warnings.warn(
                    f"[LC-W304] {len(post.at_floor)} variables are resting on the "
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
        from lcsolver.postsolve.sensitivity import sensitivities as _sens
        out = _sens(m)
    except Exception as exc:
        # never fatal, but never SILENT: an unexplained empty table once cost
        # a day of diagnosis (unit-corrector recursion)
        res['sensitivity_detail'] = {
            'method': 'failed',
            'error': f'{type(exc).__name__}: {exc}',
        }
        warnings.warn(
            "[LC-W302] sensitivities could not be computed at this "
            f"solution ({type(exc).__name__}: {exc}). The solve itself is "
            "unaffected; res['sensitivity_detail']['error'] carries the "
            "cause.",
            RuntimeWarning, stacklevel=3)
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
    """Put a starting point onto the model. Accepts a FeasibilityResult (or
    anything with an ``x``) or a sequence in structures['variables'] order.
    Written onto the model because that's where the backends read x0."""
    import numpy as _np

    from lcsolver.postsolve.writeback import write_solution

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
          sensitivities=True, structures=None, start=None, quiet=True,
          skip_degeneracy_check=False, **kwargs):
    """Solve a Formulation. See ``_solve_impl`` below for the full story.

    ``quiet`` (default True) captures the solve's warnings into
    ``result['messages']`` instead of printing them; errors still raise.
    """
    if not quiet:
        return _solve_impl(m, solver=solver, convex_backend=convex_backend,
                           diagnostics=diagnostics,
                           sensitivities=sensitivities, structures=structures,
                           skip_degeneracy_check=skip_degeneracy_check,
                           start=start, **kwargs)
    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        res = _solve_impl(m, solver=solver, convex_backend=convex_backend,
                          diagnostics=diagnostics,
                          sensitivities=sensitivities, structures=structures,
                          skip_degeneracy_check=skip_degeneracy_check,
                          start=start, **kwargs)
    # tagged messages ([LC-Wxxx]) stand alone; untagged third-party warnings
    # keep their category as context
    msgs = [str(w.message) if str(w.message).startswith('[LC-')
            else f'{w.category.__name__}: {w.message}' for w in caught]
    if isinstance(res, dict):
        res['messages'] = msgs
    try:
        m._solve_messages = msgs
    except Exception:
        pass
    return res


def _solve_impl(m, solver='auto', convex_backend='ipopt', diagnostics='error',
                sensitivities=True, structures=None, start=None,
                skip_degeneracy_check=False, **kwargs):
    """Solve an LCsolver Formulation, choosing a backend automatically.

    'auto' routes a detected LP/QP/GP/SP to ``convex_backend`` (default ipopt:
    a GP is convex in log space either way, and cvxopt fails on SPaircraft-
    class models with status 'unknown') and everything else, black boxes
    included, to IPOPT; force with solver='cvxopt'/'ipopt-convex'/'ipopt'.
    ``diagnostics``: 'error' (default) / 'warn' / 'print' / 'off' -- see
    _run_diagnostics. ``sensitivities`` (default on) attaches d(obj)/d(Constant)
    to the result and f.solution. ``structures`` takes a pre-detected form
    (from structure_detector(unit_corrector(f)), default bounds_as_rows=True;
    the split form is refused) to skip the walk -- worth a third of an
    optimization_check-then-solve on SPaircraft. ``start`` takes a
    FeasibilityResult or a sequence in structures['variables'] order.
    The solution is written back, so pyo.value(m.x) returns the optimum.
    """
    import warnings

    from lcsolver.presolve.reductions import InfeasibleProblem
    from lcsolver.solvers.ipopt import ipopt_solve

    # remembered so the Report section can say auto-detected vs prescribed
    try:
        m._solve_request = {'solver': solver, 'convex_backend': convex_backend}
    except Exception:
        pass
    from lcsolver.presolve.unitCorrector import UnitMismatch

    # Detect once for both the checks and the solve -- the walk is 4-6s on
    # SPaircraft against an 11s solve, and optimization_check reads either form.
    # Validate sp_method HERE, not in _solve_sp: inside the structured-backend
    # try a ValueError gets caught, retried on the raw NLP route, and dies as
    # an unrelated TypeError. A misspelled option must not be retried.
    _skipdeg = skip_degeneracy_check
    _sp_method = kwargs.get('sp_method', 'sia')
    if _sp_method not in ('sia', 'pccp'):
        raise ValueError(
            f"sp_method must be 'sia' or 'pccp'; got {_sp_method!r}")

    # linear_solver selects IPOPT's inner linear solver. Validate the NAME
    # here (same reason as sp_method); availability is probed at the backend.
    # cvxopt has no such option, so pairing them is a mistake in the call
    _linear_solver = kwargs.get('linear_solver')
    if _linear_solver is not None:
        from lcsolver.environment import KNOWN_IPOPT_LINEAR_SOLVERS
        _linear_solver = str(_linear_solver).strip().lower()
        if _linear_solver not in KNOWN_IPOPT_LINEAR_SOLVERS:
            raise ValueError(
                'unknown linear_solver %r. IPOPT linear solvers are: %s'
                % (kwargs['linear_solver'],
                   ', '.join(KNOWN_IPOPT_LINEAR_SOLVERS)))
        kwargs['linear_solver'] = _linear_solver
        if solver == 'cvxopt' or convex_backend == 'cvxopt':
            raise ValueError(
                "linear_solver=%r selects IPOPT's inner linear solver, and "
                "cvxopt has no such option. Drop the argument, or use the "
                "IPOPT backend." % _linear_solver)

    # linear_solver_library supplies the dlopened library (hsllib for
    # ma57/77/86/97, pardisolib for pardiso); meaningless alone or with a
    # compiled-in solver, so refuse both here in the caller's own frame
    _ls_library = kwargs.get('linear_solver_library')
    if _ls_library is not None:
        if _linear_solver is None:
            raise ValueError(
                'linear_solver_library was given without linear_solver; '
                'name the solver the library provides (ma57/ma77/ma86/ma97 '
                'for a CoinHSL library, pardiso for Panua Pardiso)')
        from lcsolver.environment import linear_solver_library_option
        if linear_solver_library_option(_linear_solver) is None:
            raise ValueError(
                '%r is compiled into IPOPT, not loaded from a library; '
                'linear_solver_library applies only to ma57/ma77/ma86/ma97 '
                'and pardiso' % _linear_solver)

    # stamp finite-difference permission onto every grey-box model NOW,
    # before detection clones the formulation, so the clones carry it
    _fd_flag = kwargs.pop('allow_blackbox_finite_difference', None)
    if _fd_flag is not None:
        try:
            from pyomo.contrib.pynumero.interfaces.external_grey_box import (
                ExternalGreyBoxBlock,
            )
            for _blk in m.component_data_objects(ExternalGreyBoxBlock,
                                                 descend_into=True,
                                                 active=True):
                _ex = _blk.get_external_model()
                if _ex is not None:
                    _ex.allow_finite_difference = bool(_fd_flag)
        except ImportError:
            pass

    want_checks = diagnostics not in (None, 'off', False)
    # bind the corrected clone to a local: structures['variables'] holds bare
    # VarData, and collecting the clone would collect their parents
    if start is not None:
        # before detection, so the detected structures carry the new values
        _apply_start(m, start)

    corrected = None
    detection_failed = None
    if structures is not None:
        # caller-supplied; check the form now -- the failure mode otherwise
        # is a reasonable-looking answer to a problem with no bounds
        if isinstance(structures, dict) and structures.get('bounds') is not None:
            raise ValueError(
                "solve() was given structures built with bounds_as_rows=False. "
                "The backends read variable bounds out of the constraint rows, "
                "so solving these would ignore every bound and answer a "
                "different question. Re-run structure_detector(corrected) with "
                "its default bounds_as_rows=True; optimization_check() reads that form "
                "too.")
        _check_structures_match(structures, m)
        _raise_if_infeasible(structures)
    elif want_checks or solver == 'auto':
        try:
            corrected = unit_corrector(m)
            structures = structure_detector(corrected)
            _raise_if_infeasible(structures)
        except InfeasibleProblem:
            # a proof of infeasibility is an answer, not a reason to fall back
            raise
        except UnitMismatch:
            # so is a dimensional error: IPOPT happily returns numbers for a
            # model that doesn't balance (once reported 1e5 m lengths)
            raise
        except Exception as e:
            detection_failed = e

    # a block that never received its inputs posted no rows, and the solve
    # would answer the reduced problem without complaint; unconditional --
    # one attribute walk against a confident wrong number
    from lcsolver.presolve.reductions import unbuilt_blocks_check
    unbuilt_blocks_check(m)

    # detection is settled (supplied, detected, or failed); the post-solve
    # checks read this rather than re-deriving it
    _st = structures

    # Central presolve: the output-only peel runs HERE, once, so every
    # structured backend consumes the same reduced structures. An output-only
    # variable is a free column the solver parks anywhere; _finish restores
    # it from its defining constraint. Grey-box-referenced columns are never
    # peeled; presolve=False bypasses the peel and the gate above keeps the
    # dangling variable an error.
    _presolve = bool(kwargs.pop('presolve', True))
    _peeled = None
    _full_structures = structures
    _protect = None
    _can_peel = (structures is not None and detection_failed is None
                 and solver != 'ipopt'
                 and (structures['Geometric_Program'][0]
                      or structures['Signomial_Program'][0]))
    if _presolve and _can_peel:
        from lcsolver.solvers.sequential.bridge import _greybox_protected
        _protect = _greybox_protected(structures)
    _will_peel = _presolve and _can_peel and _protect is not None

    if want_checks and structures is not None:
        try:
            _run_diagnostics(structures, diagnostics, peel_outputs=_will_peel)
        except (InfeasibleProblem, PresolveError):
            # the gate is the point: stop before any solver spends time
            raise
        except Exception:
            pass                         # a broken check must not block a solve

    if _will_peel:
        from lcsolver.presolve.reductions import peel_output_columns
        try:
            reduced, removed = peel_output_columns(structures,
                                                   protect=_protect)
        except Exception:
            removed = None               # a broken peel must not block a solve
        if removed:
            structures, _peeled = reduced, removed

    def _finish(res):
        """Restore centrally peeled variables into a backend result: values
        recovered from the defining constraints go back into res['x'], and
        the write-back re-runs against the FULL structures."""
        if not _peeled or not isinstance(res, dict) or res.get('x') is None:
            return res
        import numpy as np

        from lcsolver.postsolve.writeback import write_solution
        from lcsolver.presolve.reductions import restore_columns
        # ravel: cvxopt returns x as a column matrix, whose rows are
        # length-1 sequences, not floats
        res['x'] = list(restore_columns(
            _peeled, np.asarray(res['x'], dtype=float).ravel(),
            n_original=len(_full_structures['variables'])))
        try:
            res['solution'] = write_solution(_full_structures, res, model=m)
        except Exception:
            pass                  # the reduced write-back already succeeded
        return res

    if solver == 'cvxopt':
        return _attach_sensitivities(m, _finish(cvxopt_solve(m, structures=structures, **_strip_routing_kwargs(kwargs))), sensitivities, _skipdeg, _st)
    if solver == 'ipopt-convex':
        return _attach_sensitivities(m, _finish(_convex_ipopt(m, structures=structures, presolve=_presolve, **kwargs)), sensitivities, _skipdeg, _st)
    if solver == 'ipopt':
        return _attach_sensitivities(m, ipopt_solve(m, linear_solver=kwargs.get('linear_solver'), linear_solver_library=kwargs.get('linear_solver_library'), **_strip_routing_kwargs(kwargs)), sensitivities, _skipdeg, _st)
    if solver != 'auto':
        raise ValueError(f"solver must be 'auto', 'cvxopt', or 'ipopt'; got {solver!r}")

    if detection_failed is not None:
        warnings.warn(
            f"[LC-W102] structure detection failed ({type(detection_failed).__name__}: "
            f"{detection_failed}); solving with IPOPT instead.",
            RuntimeWarning, stacklevel=2)
        structured = False
    else:
        structured = any(structures[k][0] for k in
                         ('Linear_Program', 'Quadratic_Program',
                          'Geometric_Program', 'Signomial_Program'))

    from lcsolver.solvers.ipopt.NLP import _has_greybox
    if _has_greybox(m):
        # A black box can't enter the algebraic convex backends, and running
        # the structured route without it would solve a relaxation. GP/SP
        # algebraic part -> SIA, which linearizes each black box inside the
        # trust-region loop (rows appended in slcp_bridge.build_problem);
        # anything else -> raw IPOPT via cyipopt, the only other route that
        # can evaluate a Python black box.
        if (structures is not None and detection_failed is None
                and (structures['Geometric_Program'][0]
                     or structures['Signomial_Program'][0])):
            return _attach_sensitivities(
                m, _finish(_solve_sp(structures, m, presolve=_presolve,
                                     **kwargs)),
                sensitivities, _skipdeg, _st)
        if structures is not None and detection_failed is None:
            # not-GP/SP, so this grey-box model takes the raw route --
            # probably a surprise; say why (the detector's blame list) and
            # that any SIAOptions are about to be discarded
            from lcsolver.solvers.sequential.sia import SIAOptions
            blockers = (structures.get('blockers') or {}).get(
                'Signomial_Program') or []
            why = '; '.join(
                f'{nm} {reason}' for nm, reason, _ in blockers[:3])
            msg = ("[LC-W207] this model has black-box constraints but its "
                   "algebraic part was not detected as a GP or SP, so it is "
                   "being solved on the raw IPOPT route rather than SIA"
                   + (f': {why}' if why else '.'))
            if isinstance(kwargs.get('options'), SIAOptions):
                msg += (' The SIAOptions passed via options= are IGNORED on '
                        'this route.')
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        if not _ipopt_available():
            raise SolverUnavailable(
                'this model has black-box constraints and its algebraic part '
                'is not a detected GP/SP, so it needs the raw IPOPT route -- '
                'and no usable IPOPT installation was found. Run '
                '`lcsolver-install-solvers`; see docs/ipopt.rst.')
        return _attach_sensitivities(m, ipopt_solve(m, linear_solver=kwargs.get('linear_solver'), linear_solver_library=kwargs.get('linear_solver_library'), **_strip_routing_kwargs(kwargs)),
                                     sensitivities, _skipdeg, _st)

    cvxopt_failure = None
    if structured:
        backend = convex_backend
        if backend == 'ipopt' and not _ipopt_available():
            # cvxopt solves the same convex problem to the same optimum, so
            # fall back -- but say so, or the missing IPOPT install stays
            # hidden as long as the models stay convex
            warnings.warn(
                '[LC-W202] no usable IPOPT installation was found; solving this '
                'structured problem with cvxopt instead. The answer is the '
                'same, but IPOPT is the default backend and is required for '
                'black-box constraints. Run `lcsolver-install-solvers` -- see '
                'docs/ipopt.rst.'
                + (' The requested linear_solver=%r is an IPOPT option and '
                   'is IGNORED by cvxopt.' % kwargs['linear_solver']
                   if kwargs.get('linear_solver') else ''),
                RuntimeWarning, stacklevel=2)
            backend = 'cvxopt'
        try:
            if backend == 'ipopt':
                return _attach_sensitivities(
                    m, _finish(_convex_ipopt(m, structures=structures,
                                             presolve=_presolve, **kwargs)),
                    sensitivities, _skipdeg, _st)
            return _attach_sensitivities(m, _finish(cvxopt_solve(m, structures=structures, **_strip_routing_kwargs(kwargs))),
                                         sensitivities, _skipdeg, _st)
        except Exception as e:
            from lcsolver.presolve.reductions import InfeasibleProblem
            if isinstance(e, InfeasibleProblem):
                # infeasibility is an ANSWER, not a reason to try another
                # backend -- raw IPOPT would return numbers for a design
                # that does not exist. Same policy as the presolve gate.
                raise
            # fall through, but say why: a silent fallback turns a
            # structured-path bug into a confusing failure further down
            if backend == 'cvxopt':
                cvxopt_failure = e
            where = ('IPOPT on the raw model' if _ipopt_available()
                     else 'cvxopt' if backend == 'ipopt' else 'nothing else')
            warnings.warn(
                f"[LC-W201] the structured backend failed ({type(e).__name__}: {e}); "
                f"falling back to {where}.",
                RuntimeWarning, stacklevel=2)
            if not _ipopt_available() and backend == 'ipopt':
                return _attach_sensitivities(m, _finish(cvxopt_solve(m, structures=structures, **_strip_routing_kwargs(kwargs))),
                                             sensitivities, _skipdeg, _st)

    if not _ipopt_available():
        # nothing left to try: cvxopt can't take a general NLP, so this is a
        # real dead end -- name it as one
        if cvxopt_failure is not None:
            raise SolverUnavailable(
                f'cvxopt failed on this structured problem '
                f'({type(cvxopt_failure).__name__}: {cvxopt_failure}), and '
                'no usable IPOPT installation was found to try instead. '
                'IPOPT is the preferred backend -- it solves models cvxopt '
                'fails on. Run `lcsolver-install-solvers`. See docs/ipopt.rst.'
            ) from cvxopt_failure
        raise SolverUnavailable(
            'this model needs IPOPT and no usable installation was found. '
            + ('It is not a detected LP, QP, GP or SP, so cvxopt cannot solve '
               'it. ' if not structured else '')
            + 'Run `lcsolver-install-solvers`. See docs/ipopt.rst.')
    return _attach_sensitivities(m, ipopt_solve(m, linear_solver=kwargs.get('linear_solver'), linear_solver_library=kwargs.get('linear_solver_library'), **_strip_routing_kwargs(kwargs)), sensitivities, _skipdeg, _st)


# help(solve) should tell the whole story, not just the wrapper's.
solve.__doc__ = (solve.__doc__ or '') + '\n' + (_solve_impl.__doc__ or '')


def _strip_routing_kwargs(kwargs):
    """Drop routing/options kwargs the raw backends reject -- an unexpected
    'sp_method' from the fallback used to mask every structured-path error."""
    drop = ('sp_method', 'options', 'sia_options', 'sp_form', 'presolve',
            'split_equalities', 'pair_equalities', 'linear_solver',
            'linear_solver_library')
    return {k: v for k, v in kwargs.items() if k not in drop}


def _solve_sp(structures, m, sp_method='sia', linear_solver=None,
              linear_solver_library=None, **kwargs):
    """Solve a signomial program. SIA by default.

    'sia' terminates on a genuine KKT residual for the ORIGINAL problem
    (SPaircraft: certified in 149 iterations, ~20s). 'pccp' stops when the
    objective stops changing -- "I stopped moving", not "I am optimal"
    (SPaircraft: 180s, less feasible, no certificate). SIA is the default
    because it can answer whether it arrived, not because it's faster.
    """
    from lcsolver.postsolve.writeback import write_solution

    # resolve the linear solver once and thread it in: SIA via
    # SIAOptions.ipopt_options (an explicit user setting wins), PCCP via
    # the inner GP solves. Told nothing, pick LCsolver's default explicitly
    # so the choice is recorded and SPRAL's env/defaults apply
    from lcsolver.environment import default_linear_solver, require_linear_solver
    if linear_solver is None:
        linear_solver = default_linear_solver('pyomo')
    if linear_solver is not None:
        linear_solver = require_linear_solver(linear_solver, route='pyomo',
                                              library=linear_solver_library)

    if sp_method == 'pccp':
        from lcsolver.solvers.sequential.pccp import solve_SP
        from lcsolver.solvers.ipopt.GP import solve_gp_rows_ipopt

        def _inner(rows, relations, x0=None):
            return solve_gp_rows_ipopt(
                rows, relations, x0=x0, linear_solver=linear_solver,
                linear_solver_library=linear_solver_library)

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

    from lcsolver.solvers.sequential.bridge import solve_sia

    if linear_solver is not None:
        from lcsolver.environment import linear_solver_library_option
        from lcsolver.solvers.sequential.sia import SIAOptions
        _opts = kwargs.get('options')
        if not isinstance(_opts, SIAOptions):
            _opts = SIAOptions()
            kwargs['options'] = _opts
        _opts.ipopt_options.setdefault('linear_solver', linear_solver)
        if linear_solver_library is not None:
            _opts.ipopt_options.setdefault(
                linear_solver_library_option(linear_solver),
                str(linear_solver_library))
        # a solver set straight on SIAOptions.ipopt_options wins; validate
        # it the same way so SPRAL's runtime env gets set for it too
        _eff = _opts.ipopt_options['linear_solver']
        if _eff != linear_solver:
            require_linear_solver(_eff, route='pyomo')
        from lcsolver.environment import apply_linear_solver_defaults
        apply_linear_solver_defaults(_opts.ipopt_options, _eff)

    result = solve_sia(structures, **{k: v for k, v in kwargs.items()
                                      if k in ('x0', 'options', 'sp_form',
                                               'presolve', 'split_equalities',
                                               'pair_equalities')})
    _final_viol = float(getattr(result, 'max_violation', 0.0) or 0.0)
    if (getattr(result, 'phase1_feasible', None) is False
            and not result.converged
            and _final_viol > 10.0 * 1e-6):
        # no feasible point and no recovery: an ANSWER, not a partial
        # result. Surface the elastic Phase-I diagnosis instead of
        # returning the best infeasible iterate as if it were a design.
        from lcsolver.presolve.reductions import InfeasibleProblem
        report = getattr(result, 'infeasibility_report', None) or result.status
        raise InfeasibleProblem(
            'SIA Phase I could not find a feasible point and the solve did '
            'not recover.\n' + str(report))
    res = {
        'x': list(result.x),
        'primal objective': result.objective,
        'linear_solver': (kwargs['options'].ipopt_options or {}).get(
            'linear_solver') if kwargs.get('options') is not None else None,
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
        msg = (f"[LC-W203] SIA did not converge: {status}. The returned point is "
               f"feasible to {result.max_violation:.2e} with a stationarity "
               f"residual of {result.stationarity:.2e}; it is the best "
               "iterate, not a certified optimum.")
        # say what to DO about it, keyed on how it failed
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


def _convex_ipopt(m, structures=None, presolve=True, **kwargs):
    """Solve a structured formulation with IPOPT rather than cvxopt.

    A GP is solved in log space where it is convex, so global optimality is
    preserved; LPs/QPs go to IPOPT unchanged. ``presolve`` routes only to the
    signomial path -- the central peel in _solve_impl has already run.
    """
    from lcsolver.solvers.ipopt.GP import solve_gp_ipopt, solve_lp_qp_ipopt
    from lcsolver.solvers.ipopt import ipopt_solve

    if structures is None:
        structures = structure_detector(unit_corrector(m))
    _raise_if_infeasible(structures)

    # `options` is overloaded by history: an IPOPT dict on the convex paths,
    # SIAOptions on the signomial path. Route it by TYPE so SIA parameters
    # don't crash a pure-GP solve into the raw-IPOPT fallback; `sia_options`
    # is the unambiguous spelling, forwarded only on the signomial path.
    from lcsolver.solvers.sequential.sia import SIAOptions
    _sia_opts = kwargs.pop('sia_options', None)
    # mirror the dispatch order below: a GP is also a detected SP but is
    # SOLVED as a GP, so the SIA meaning applies only when SP is the route
    _is_sp = bool(structures['Signomial_Program'][0]
                  and not structures['Geometric_Program'][0]
                  and not structures['Linear_Program'][0]
                  and not structures['Quadratic_Program'][0])
    if _is_sp:
        if _sia_opts is not None:
            kwargs['options'] = _sia_opts
        elif isinstance(kwargs.get('options'), dict):
            kwargs.pop('options')          # an IPOPT dict means nothing to SIA
    elif isinstance(kwargs.get('options'), SIAOptions):
        kwargs.pop('options')              # SIA parameters mean nothing here

    if structures['Geometric_Program'][0]:
        return solve_gp_ipopt(structures, model=m, **kwargs)
    if structures['Linear_Program'][0] or structures['Quadratic_Program'][0]:
        return solve_lp_qp_ipopt(
            m,
            structure=('linear_program' if structures['Linear_Program'][0]
                       else 'quadratic_program'),
            **kwargs)
    if structures['Signomial_Program'][0]:
        return _solve_sp(structures, m, presolve=presolve, **kwargs)
    # Nothing structured left to exploit.
    return ipopt_solve(m, linear_solver=kwargs.get('linear_solver'), linear_solver_library=kwargs.get('linear_solver_library'), **_strip_routing_kwargs(kwargs))






