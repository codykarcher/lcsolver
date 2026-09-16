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
a Python black box). Either way the solution is written back onto the model.
"""

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

from lcsolver.core.errors import SolverUnavailable


# ---------------------------------------------------------------------------
def _resolve_executable(executable=None):
    """Which ipopt binary: explicit arg, else LCSOLVER_IPOPT_EXECUTABLE,
    else PATH. The env var exists because conda activate prepends to PATH
    every shell, so a source-built MA27 ipopt silently loses to the conda
    MUMPS one; pinning the path survives the next terminal window."""
    if executable:
        return executable
    from lcsolver.environment import ipopt_executable
    return ipopt_executable()


def _ma27_available(executable=None):
    """Does the ipopt executable carry HSL MA27? Probed with a one-variable
    solve, cached for the session; only consulted on a FAILED solve to
    sharpen the error, so the probe never costs the success path."""
    from lcsolver.environment import linear_solver_available
    return linear_solver_available('ma27', _resolve_executable(executable))


def _has_greybox(model):
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


def _executable_available(name='ipopt'):
    try:
        executable = _resolve_executable() if name == 'ipopt' else None
        opt = (pyo.SolverFactory(name, executable=executable) if executable
               else pyo.SolverFactory(name))
        return bool(opt.available(exception_flag=False))
    except Exception:
        return False


def _summarize(results):
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
    raises RuntimeError when no IPOPT is usable or the solve fails.
    """
    options = dict(options or {})
    greybox = _has_greybox(m)
    # Resolved once here so the route choice, the solve and the MA27 diagnosis
    # all talk about the same binary.
    executable = _resolve_executable(executable)

    # ---- choose the route -------------------------------------------------
    if method == 'auto':
        if greybox:
            route = 'cyipopt'          # AMPL route cannot evaluate a Python black box
        elif _executable_available('ipopt'):
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
        from lcsolver.environment import _pynumero_asl_available
        if opt.available(exception_flag=False) and not _pynumero_asl_available():
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
    summary = _summarize(results)
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
        elif route == 'pyomo' and not _ma27_available(executable):
            msg += (
                "\nNote: this IPOPT build appears to lack the HSL MA27 "
                "linear solver, so it is running MUMPS (the shipped "
                "default). MA27 is markedly more robust on these problems "
                "-- see docs/ipopt.rst and utilities/install_ipopt.sh for "
                "building IPOPT with it.")
        raise RuntimeError(msg)

    if route == 'pyomo' and load_solutions:
        m.solutions.load_from(results)

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
