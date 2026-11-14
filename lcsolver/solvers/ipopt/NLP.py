#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""IPOPT interface for LCsolver formulations.

IPOPT solves the general NLP directly (no structure required), so it's the
default for unstructured formulations and the only option for grey-box ones.
Two routes: 'pyomo' (AMPL interface driving the ipopt executable; preferred)
and 'cyipopt' (in-process via pynumero; used when the executable is missing,
and required for ExternalGreyBoxBlock models -- the AMPL route can't evaluate
a Python black box). Either way IPOPT sees the model in consistent base units
(unit_corrector) and the solution is written back onto the caller's model, in
each variable's declared units.
"""

import warnings

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

from lcsolver.core.errors import SolverUnavailable


# ---------------------------------------------------------------------------
def resolve_executable(executable=None):
    """Which ipopt binary: explicit arg, else LCSOLVER_IPOPT_EXECUTABLE,
    else PATH. The env var exists because conda activate prepends to PATH
    every shell, so a source-built MA27 ipopt silently loses to the conda
    MUMPS one; pinning the path survives the next terminal window."""
    if executable:
        return executable
    from lcsolver.environment import ipopt_executable
    return ipopt_executable()


def ma27_available(executable=None):
    """Does the ipopt executable carry HSL MA27? Probed with a one-variable
    solve, cached for the session; only consulted on a FAILED solve to
    sharpen the error, so the probe never costs the success path."""
    from lcsolver.environment import linear_solver_available
    return linear_solver_available('ma27', resolve_executable(executable))


def has_greybox(model):
    """True if the model contains any ExternalGreyBoxBlock (a black-box constraint)."""
    try:
        from pyomo.contrib.pynumero.interfaces.external_grey_box import (
            ExternalGreyBoxBlock,
        )
    except Exception:
        return False
    for _ in model.component_data_objects(ExternalGreyBoxBlock, descend_into=True,
                                          active=True):
        return True
    return False


def executable_available(name='ipopt'):
    try:
        executable = resolve_executable() if name == 'ipopt' else None
        opt = (pyo.SolverFactory(name, executable=executable) if executable
               else pyo.SolverFactory(name))
        return bool(opt.available(exception_flag=False))
    except Exception:
        return False


def unit_corrected_for_ipopt(m):
    """The model IPOPT should see: every row in consistent base units.

    The AMPL writer drops units, so a row mixing ft and m reaches IPOPT as
    bare numbers and it solves a different problem without complaint. A
    dimensional error raises. A model the corrector can't walk for any other
    reason (a Constraint indexed over several members) keeps the old
    behaviour, solved as written, but says so.
    """
    from lcsolver.presolve.unitCorrector import UnitMismatch, unit_corrector

    if getattr(m, '_lc_unit_corrected', False):
        return m
    try:
        return unit_corrector(m)
    except UnitMismatch:
        raise
    except Exception as exc:
        warnings.warn(
            f"[LC-W208] unit correction failed ({type(exc).__name__}: {exc}); "
            "solving the model as written, so any row mixing units is NOT "
            "converted.", RuntimeWarning, stacklevel=3)
        return m


def write_back_to_caller(solved, m):
    """Copy primal values and duals from the corrected clone onto ``m``.

    Variables keep their declared units through unit_corrector, so values
    copy over unchanged. Duals are those of the base-unit rows.
    """
    from lcsolver.postsolve.writeback import resolve_on_model

    for v in solved.component_data_objects(pyo.Var, descend_into=True, active=True):
        target = resolve_on_model(m, v)
        if target is not None and v.value is not None:
            target.set_value(v.value, skip_validation=True)

    source = getattr(solved, 'dual', None)
    if source is None:
        return
    if not hasattr(m, 'dual'):
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.dual.clear()
    for con in solved.component_data_objects(pyo.Constraint, descend_into=True, active=True):
        target = resolve_on_model(m, con)
        if target is not None and con in source:
            m.dual[target] = source[con]


def summarize_results(results):
    """Condense a Pyomo results object into a plain dict."""
    out = {'solver': None, 'status': None, 'termination_condition': None,
           'objective': None, 'message': None}
    try:
        solver = results.solver[0] if len(results.solver) else results.solver
        out['status'] = str(getattr(solver, 'status', None))
        out['termination_condition'] = str(
            getattr(solver, 'termination_condition', None))
        out['message'] = str(getattr(solver, 'message', '') or '')
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
def ipopt_solve(m, method='auto', tee=False, executable=None, options=None,
                load_solutions=True, linear_solver=None,
                linear_solver_library=None):
    """Solve an LCsolver Formulation with IPOPT.

    method: 'auto' picks cyipopt for grey-box models or a missing ipopt
    executable, 'pyomo' otherwise. options are IPOPT options; linear_solver
    ('mumps', 'ma27', ...) is validated against the actual build so an
    absent solver raises SolverUnavailable naming what IS available, and
    overrides options['linear_solver']. Returns a summary dict (route,
    status, termination condition, objective, {name: value} solution);
    raises RuntimeError when no IPOPT is usable or the solve fails, and
    UnitMismatch when the model's units don't balance. The reported objective
    is in base units, like 'primal objective' on the structured backends.
    """
    options = dict(options or {})
    greybox = has_greybox(m)
    # Resolved once here so the route choice, the solve and the MA27 diagnosis
    # all talk about the same binary.
    executable = resolve_executable(executable)

    # ---- choose the route -------------------------------------------------
    if method == 'auto':
        if greybox:
            route = 'cyipopt'          # AMPL route cannot evaluate a Python black box
        elif executable_available('ipopt'):
            route = 'pyomo'            # preferred
        else:
            route = 'cyipopt'          # fall back
    else:
        route = method
    if route not in ('pyomo', 'cyipopt'):
        raise ValueError(f"method must be 'auto', 'pyomo', or 'cyipopt'; got {method!r}")

    if route == 'pyomo' and greybox:
        raise RuntimeError(
            "this model contains black-box (grey-box) constraints, which the "
            "AMPL-based 'pyomo' route cannot evaluate; use method='cyipopt'")

    if linear_solver is None and not options.get('linear_solver'):
        # explicit so the choice is recorded and SPRAL's env/defaults apply
        from lcsolver.environment import default_linear_solver
        linear_solver = default_linear_solver(
            route, executable if route == 'pyomo' else None)
    if linear_solver is not None:
        from lcsolver.environment import (
            linear_solver_library_option,
            require_linear_solver,
        )
        options['linear_solver'] = require_linear_solver(
            linear_solver, route=route,
            executable=executable if route == 'pyomo' else None,
            library=linear_solver_library)
        if linear_solver_library is not None:
            options[linear_solver_library_option(
                options['linear_solver'])] = str(linear_solver_library)
        from lcsolver.environment import apply_linear_solver_defaults
        apply_linear_solver_defaults(options, options['linear_solver'])

    # IPOPT solves the corrected model; the caller's gets the answer at the end
    caller, m = m, unit_corrected_for_ipopt(m)

    # Ask for constraint duals: free, and what postsolve.sensitivity uses;
    # without the Suffix they'd have to be reconstructed from the primal.
    if not hasattr(m, 'dual'):
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # ---- solve ------------------------------------------------------------
    if route == 'pyomo':
        from lcsolver.environment import ipopt_solver_factory
        opt = ipopt_solver_factory(executable)
        if not opt.available(exception_flag=False):
            raise SolverUnavailable(
                "the 'ipopt' executable was not found. Run "
                "`lcsolver-install-solvers` to get one, pass "
                "executable='/path/to/ipopt', or use method='cyipopt'.")
        for k, v in options.items():
            opt.options[k] = v
        # Defer loading: with load_solutions=True Pyomo raises a bare
        # ValueError inside solutions.load_from on any bad status, before the
        # clean "IPOPT did not converge" diagnosis below can run. Solve without
        # loading, check the termination condition, then load explicitly.
        from lcsolver.environment import ipopt_launch
        with ipopt_launch(options.get('linear_solver'), executable):
            results = opt.solve(m, tee=tee, load_solutions=False)
    else:
        opt = pyo.SolverFactory('cyipopt')
        # cyipopt importable is not enough: PyNumero needs a compiled ASL
        # library that ships with neither pyomo nor cyipopt, and pyomo's
        # own error for its absence is unhelpful.
        from lcsolver.environment import pynumero_asl_available
        if opt.available(exception_flag=False) and not pynumero_asl_available():
            raise SolverUnavailable(
                "the in-process (cyipopt) route needs Pyomo's PyNumero ASL "
                "library, which is not installed. `pyomo build-extensions` "
                "compiles it, or run `lcsolver-install-solvers`.")
        if not opt.available(exception_flag=False):
            raise SolverUnavailable(
                "neither the 'ipopt' executable nor cyipopt is available. "
                "Run `lcsolver-install-solvers` to install both; note that "
                "`pip install cyipopt` on its own compiles against an IPOPT "
                "that has to exist already.")
        # PyomoCyIpoptSolver has no `options` mapping (opt.options[k] raises
        # AttributeError) -- options go as a solve() argument. Black-box
        # models are forced onto this route, so it has to accept
        # linear_solver.
        results = opt.solve(m, tee=tee, options=dict(options))

    # ---- interpret --------------------------------------------------------
    summary = summarize_results(results)
    summary['solver'] = route
    summary['problem_structure'] = 'nonlinear_program'
    # first question when two machines disagree; None = build default
    summary['linear_solver'] = options.get('linear_solver')

    tc = summary['termination_condition']
    ok = tc in (str(TerminationCondition.optimal),
                str(TerminationCondition.locallyOptimal),
                str(TerminationCondition.feasible))
    if not ok:
        msg = (f"IPOPT did not converge: termination_condition={tc}, "
               f"status={summary['status']}. {summary['message']}".strip())
        if options.get('linear_solver'):
            from lcsolver.environment import linear_solver_failure_note
            msg += linear_solver_failure_note(
                options['linear_solver'],
                executable if route == 'pyomo' else None)
        elif route == 'pyomo' and not ma27_available(executable):
            msg += (
                "\nNote: this IPOPT build appears to lack the HSL MA27 "
                "linear solver, so it is running MUMPS (the shipped "
                "default). MA27 is markedly more robust on these problems "
                "-- see docs/ipopt.rst and utilities/install_ipopt.sh for "
                "building IPOPT with it.")
        raise RuntimeError(msg)

    if route == 'pyomo' and load_solutions:
        m.solutions.load_from(results)
    if m is not caller and (route == 'cyipopt' or load_solutions):
        write_back_to_caller(m, caller)

    # Pyomo has already written the solution onto the model; report it in the
    # same {name: value} form the other LCsolver backends use.
    sol = {}
    for v in m.component_data_objects(pyo.Var, descend_into=True, active=True):
        try:
            sol[v.name] = pyo.value(v)
        except Exception:
            sol[v.name] = None
    summary['solution'] = sol

    try:
        obj = next(m.component_data_objects(pyo.Objective, descend_into=True,
                                            active=True))
        summary['objective'] = pyo.value(obj)
    except Exception:
        summary['objective'] = None

    return summary


# older scripts in the lc* repos import this by its former name
_executable_available = executable_available
