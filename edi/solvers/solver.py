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

    # Write the solution back onto the Pyomo model. Without this the solve
    # succeeds but pyo.value(m.x) still returns the initial guess, because the
    # cvxopt backends work in a transformed space and return only a raw vector.
    if write_back:
        try:
            res['solution'] = write_solution(structures, res, model=m)
        except Exception as e:                      # never lose a good solve
            res['solution'] = None
            res['writeback_error'] = f"{type(e).__name__}: {e}"

    return res


def solve(m, solver='auto', convex_backend='cvxopt', **kwargs):
    """Solve an EDI Formulation, choosing a backend automatically.

    solver='auto' routes to cvxopt when the formulation is detected as an LP,
    QP, GP, or SP, and to IPOPT otherwise (including any formulation containing
    black-box constraints). Pass solver='cvxopt' or solver='ipopt' to force one.

    In every case the solution is written back onto the model, so pyo.value(m.x)
    returns the optimum after a successful solve.
    """
    from edi.solvers.ipopt import ipopt_solve

    if solver == 'cvxopt':
        return cvxopt_solve(m, **kwargs)
    if solver == 'ipopt-convex':
        return _convex_ipopt(m, **kwargs)
    if solver == 'ipopt':
        return ipopt_solve(m, **kwargs)
    if solver != 'auto':
        raise ValueError(f"solver must be 'auto', 'cvxopt', or 'ipopt'; got {solver!r}")

    try:
        structures = structure_detector(unit_corrector(m))
        structured = any(structures[k][0] for k in
                         ('Linear_Program', 'Quadratic_Program',
                          'Geometric_Program', 'Signomial_Program'))
    except Exception:
        structured = False

    if structured:
        try:
            if convex_backend == 'ipopt':
                return _convex_ipopt(m, structures=structures, **kwargs)
            return cvxopt_solve(m, **kwargs)
        except Exception:
            pass                                    # fall through to plain IPOPT
    return ipopt_solve(m, **kwargs)


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
    if structures['Geometric_Program'][0]:
        return solve_gp_ipopt(structures, model=m, **kwargs)
    if structures['Linear_Program'][0] or structures['Quadratic_Program'][0]:
        return solve_lp_qp_ipopt(m, **kwargs)
    # Signomial: no convex form exists, so hand the raw model to IPOPT and be
    # explicit that global optimality is not claimed.
    return ipopt_solve(m, **kwargs)






