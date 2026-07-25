#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""IPOPT interface for EDI formulations.

Unlike the cvxopt backends, which require the formulation to fall into a
recognised structure (LP, QP, GP, SP) and solve a transformed problem, IPOPT
solves the general nonlinear program directly. That makes it the natural default
for a formulation that is not in one of those classes, and the only option for a
formulation containing black-box (grey-box) constraints.

Two routes to IPOPT are supported, in this order of preference:

``pyomo``
    ``pyo.SolverFactory('ipopt')``, Pyomo's own AMPL-based interface, driving the
    ``ipopt`` executable. This is the preferred route: it is pure Pyomo, it
    handles the whole modelling language, and it loads the solution back onto the
    model itself. It requires the ``ipopt`` binary on PATH (or an explicit path).

``cyipopt``
    ``pyo.SolverFactory('cyipopt')``, backed by ``pyomo.contrib.pynumero``. Used
    automatically when the ``ipopt`` executable is unavailable, and used
    *preferentially* when the model contains ``ExternalGreyBoxBlock`` components,
    because the AMPL route cannot evaluate a Python black box.

In both cases the solution is written back onto the model, so after a successful
solve ``pyo.value(f.x)`` returns the optimum rather than the initial guess.
"""

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition


# ---------------------------------------------------------------------------
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
        opt = pyo.SolverFactory(name)
        return bool(opt.available(exception_flag=False))
    except Exception:
        return False


def _summarise(results):
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
                load_solutions=True):
    """Solve an EDI ``Formulation`` with IPOPT.

    Parameters
    ----------
    m : Formulation
        The model. Because ``Formulation`` subclasses ``ConcreteModel`` it is
        passed straight to Pyomo.
    method : {'auto', 'pyomo', 'cyipopt'}
        Which route to IPOPT to use. ``'auto'`` (the default) selects
        ``cyipopt`` when the model contains black-box constraints or when the
        ``ipopt`` executable is not available, and ``pyomo`` otherwise.
    tee : bool
        Stream solver output.
    executable : str, optional
        Explicit path to the ``ipopt`` binary (``'pyomo'`` route only).
    options : dict, optional
        IPOPT options, e.g. ``{'tol': 1e-8, 'max_iter': 500}``.
    load_solutions : bool
        Load the solution onto the model. Left at ``True`` in normal use.

    Returns
    -------
    dict
        Summary containing the route used, solver status, termination condition,
        objective value, and a ``{variable_name: value}`` mapping.

    Raises
    ------
    RuntimeError
        If no usable IPOPT installation can be found, or if the solve does not
        reach an optimal termination condition.
    """
    options = dict(options or {})
    greybox = _has_greybox(m)

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

    # Ask the solver for constraint duals. They cost nothing extra and are what
    # `edi.solvers.sensitivity` uses to report how the optimum responds to each
    # Constant; without the Suffix those duals would have to be reconstructed
    # from the primal solution.
    if not hasattr(m, 'dual'):
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # ---- solve ------------------------------------------------------------
    if route == 'pyomo':
        opt = (pyo.SolverFactory('ipopt', executable=executable)
               if executable else pyo.SolverFactory('ipopt'))
        if not opt.available(exception_flag=False):
            raise RuntimeError(
                "the 'ipopt' executable was not found. Install IPOPT and put it "
                "on PATH, pass executable='/path/to/ipopt', or use "
                "method='cyipopt' (pip install cyipopt).")
        for k, v in options.items():
            opt.options[k] = v
        results = opt.solve(m, tee=tee, load_solutions=load_solutions)
    else:
        opt = pyo.SolverFactory('cyipopt')
        if not opt.available(exception_flag=False):
            raise RuntimeError(
                "neither the 'ipopt' executable nor cyipopt is available. "
                "Install one of them: a system IPOPT build, or `pip install cyipopt`.")
        for k, v in options.items():
            opt.options[k] = v
        results = opt.solve(m, tee=tee)

    # ---- interpret --------------------------------------------------------
    summary = _summarise(results)
    summary['solver'] = route
    summary['problem_structure'] = 'nonlinear_program'

    tc = summary['termination_condition']
    ok = tc in (str(TerminationCondition.optimal),
                str(TerminationCondition.locallyOptimal),
                str(TerminationCondition.feasible))
    if not ok:
        raise RuntimeError(
            f"IPOPT did not converge: termination_condition={tc}, "
            f"status={summary['status']}. {summary['message']}".strip())

    # Pyomo has already written the solution onto the model; report it in the
    # same {name: value} form the other EDI backends use.
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
