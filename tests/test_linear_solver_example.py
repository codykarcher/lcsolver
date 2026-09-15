"""The bundled MUMPS-vs-MA27 disagreement (examples/data/d8_sia_subproblem.nl).

The model is the SIA sub-problem at iteration 2 of the SPaircraft D8.2 deck,
captured at SIA's own tolerances. Measured on Ipopt 3.14.20: MUMPS 5.9.1
declares it locally infeasible (falsely -- the full D8 run then stops
uncertified, 17% high), MA27 solves it to optimality (and the full run
certifies at 21,384 lbf). These tests replay the file on whatever linear
solvers this machine's IPOPT carries, so the example cannot rot silently.
"""
import os

import pytest

try:
    import pyomo.environ as pyo

    from lcsolver.environment import (
        ipopt_executable,
        linear_solver_available,
    )
    available = True
except Exception:                                    # pragma: no cover
    available = False

NL = os.path.join(os.path.dirname(__file__), os.pardir, 'examples', 'data',
                  'd8_sia_subproblem.nl')
TOLS = {'tol': 1e-12, 'constr_viol_tol': 1e-12,
        'acceptable_constr_viol_tol': 1e-10}


def _termination(linear_solver):
    exe = ipopt_executable()
    opt = pyo.SolverFactory('ipopt', executable=exe)
    for k, v in TOLS.items():
        opt.options[k] = v
    opt.options['linear_solver'] = linear_solver
    results = opt.solve(NL, tee=False, load_solutions=False)
    return str(results.solver.termination_condition)


def _has(linear_solver):
    try:
        return bool(linear_solver_available(linear_solver,
                                            ipopt_executable()))
    except Exception:
        return False


@pytest.mark.skipif(not available, reason='LCsolver import failed')
def test_the_subproblem_file_is_bundled():
    assert os.path.exists(NL)
    assert os.path.getsize(NL) > 10_000


@pytest.mark.skipif(not available or not _has('ma27'),
                    reason='needs an IPOPT with ma27')
def test_ma27_solves_the_d8_subproblem():
    assert _termination('ma27') in ('optimal', 'locallyOptimal')


@pytest.mark.skipif(not available or not _has('mumps'),
                    reason='needs an IPOPT with mumps')
def test_mumps_falsely_declares_it_infeasible():
    """If a future MUMPS version starts solving this, the example's claim
    has expired and both it and this test should be refreshed with a new
    capture -- that is a finding, not a nuisance."""
    assert _termination('mumps') != 'optimal'


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
