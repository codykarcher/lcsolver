"""The bundled SPRAL crash (examples/data/d8_spral_crash.nl).

A SIA sub-problem from the SPaircraft D8.2 deck on which SPRAL v2023.03.29
kills the ipopt executable (SIGBUS/SIGSEGV) at IPOPT's tolerances: SPRAL
declares the KKT system singular (estimated rank ~1490 of 1709) and dies on
the regularized re-factorization that follows. MA27 and MUMPS solve the
same file. The bookend to the MUMPS sentinel: that one is a false verdict,
this one is a dead process. Deterministic, single-threaded, independent of
the OpenMP binding; avoided by spral_scaling=mc64 or spral_u=0.5, which is
what LCsolver's SPRAL defaults carry.
"""
import os

import pytest

try:
    import pyomo.environ as pyo

    from lcsolver.environment import (
        IpoptCrashed,
        ipopt_executable,
        ipopt_launch,
        linear_solver_available,
        linear_solver_default_options,
    )
    available = True
except Exception:                                    # pragma: no cover
    available = False

NL = os.path.join(os.path.dirname(__file__), os.pardir, 'examples', 'data',
                  'd8_spral_crash.nl')
TOLS = {'tol': 1e-12, 'constr_viol_tol': 1e-12,
        'acceptable_constr_viol_tol': 1e-10}


def _solver(linear_solver, extra=None):
    exe = ipopt_executable()
    opt = pyo.SolverFactory('ipopt', executable=exe)
    for k, v in TOLS.items():
        opt.options[k] = v
    opt.options['linear_solver'] = linear_solver
    for k, v in (extra or {}).items():
        opt.options[k] = v
    return opt


def _has(linear_solver):
    try:
        return bool(linear_solver_available(linear_solver,
                                            ipopt_executable()))
    except Exception:
        return False


@pytest.mark.skipif(not available, reason='LCsolver import failed')
def test_the_crash_file_is_bundled():
    assert os.path.exists(NL)
    assert os.path.getsize(NL) > 10_000


@pytest.mark.skipif(not available or not _has('ma27'),
                    reason='needs an IPOPT with ma27')
def test_ma27_solves_it():
    results = _solver('ma27').solve(NL, tee=False, load_solutions=False)
    assert str(results.solver.termination_condition) in ('optimal',
                                                         'locallyOptimal')


@pytest.mark.skipif(not available or not _has('spral'),
                    reason='needs an IPOPT with spral')
class TestSpral:

    def test_bare_spral_crashes_and_the_crash_is_named(self):
        """Bare = IPOPT's own SPRAL defaults, none of LCsolver's."""
        with pytest.raises(IpoptCrashed) as ctx:
            with ipopt_launch('spral'):
                _solver('spral').solve(NL, tee=False, load_solutions=False)
        msg = str(ctx.value)
        assert 'SIGBUS' in msg or 'SIGSEGV' in msg
        assert "linear_solver='spral'" in msg
        assert 'MA27' in msg

    def test_lcsolver_spral_defaults_solve_it(self):
        extra = linear_solver_default_options('spral')
        assert extra, 'SPRAL defaults are empty; the crash is unguarded'
        results = _solver('spral', extra).solve(NL, tee=False,
                                                load_solutions=False)
        assert str(results.solver.termination_condition) in (
            'optimal', 'locallyOptimal')


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
