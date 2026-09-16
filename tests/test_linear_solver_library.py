"""Runtime-loaded linear solvers: the linear_solver_library plumbing.

MA57/77/86/97 and Pardiso are dlopened via the ``hsllib``/``pardisolib``
option, exposed as ``solve(f, linear_solver=..., linear_solver_library=...)``
-- validated in the caller's frame, probed with the library in place, and
threaded to every route including SIA's sub-problems. Covered here without
any licensed library, so the machinery is ready on a machine that has one.
"""
import os

import pytest

try:
    import lcsolver
    from lcsolver import Formulation
    from lcsolver.core.errors import SolverUnavailable
    from lcsolver import environment as env
    from lcsolver.environment import (
        RUNTIME_LOADED_LINEAR_SOLVERS,
        linear_solver_library_option,
        require_linear_solver,
    )
    available = True
except Exception:                                    # pragma: no cover
    available = False


def _gp():
    f = Formulation()
    x = f.Variable('x', 1.0, '-', 'x', bounds=[0.1, 10.0])
    y = f.Variable('y', 1.0, '-', 'y', bounds=[0.1, 10.0])
    f.Objective(x + y)
    f.Constraint(x * y >= 1.0)
    return f


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestOptionMapping:

    def test_hsl_solvers_map_to_hsllib(self):
        for name in ('ma57', 'ma77', 'ma86', 'ma97'):
            assert linear_solver_library_option(name) == 'hsllib'

    def test_pardiso_maps_to_pardisolib(self):
        assert linear_solver_library_option('pardiso') == 'pardisolib'

    def test_compiled_in_solvers_have_no_library_option(self):
        for name in ('ma27', 'mumps', 'spral', 'pardisomkl', 'wsmp'):
            assert linear_solver_library_option(name) is None


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestCallValidation:

    def test_library_without_solver_is_refused(self):
        with pytest.raises(ValueError, match='without linear_solver'):
            lcsolver.solve(_gp(), linear_solver_library='/tmp/lib.dylib')

    def test_library_with_compiled_in_solver_is_refused(self):
        with pytest.raises(ValueError, match='compiled into IPOPT'):
            lcsolver.solve(_gp(), linear_solver='ma27',
                           linear_solver_library='/tmp/lib.dylib')

    def test_missing_library_file_is_named(self):
        with pytest.raises(SolverUnavailable, match='does not exist'):
            require_linear_solver('ma97',
                                  library='/nowhere/libcoinhsl.dylib')

    def test_require_refuses_library_for_compiled_in(self):
        with pytest.raises(ValueError, match='runtime-loaded'):
            require_linear_solver('mumps', library='/tmp/lib.dylib')


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestProbePlumbing:

    def test_probe_receives_the_library_option(self, monkeypatch, tmp_path):
        lib = tmp_path / 'libcoinhsl.dylib'
        lib.write_bytes(b'not really a library')
        seen = {}

        def fake_available(name, executable=None, library=None):
            seen['name'], seen['library'] = name, library
            return True

        monkeypatch.setattr(env, 'linear_solver_available', fake_available)
        out = require_linear_solver('ma97', library=str(lib))
        assert out == 'ma97'
        assert seen['library'] == str(lib)

    def test_unavailable_runtime_solver_hints_at_the_library(self,
                                                             monkeypatch):
        monkeypatch.setattr(env, 'linear_solver_available',
                            lambda name, executable=None, library=None:
                            name == 'ma27')
        with pytest.raises(SolverUnavailable) as ctx:
            require_linear_solver('ma97')
        assert 'linear_solver_library' in str(ctx.value)

    def test_spral_request_sets_the_omp_environment(self, monkeypatch):
        monkeypatch.delenv('OMP_CANCELLATION', raising=False)
        monkeypatch.delenv('OMP_PROC_BIND', raising=False)
        monkeypatch.setattr(env, 'linear_solver_available',
                            lambda name, executable=None, library=None: True)
        require_linear_solver('spral')
        assert os.environ['OMP_CANCELLATION'] == 'TRUE'
        assert os.environ['OMP_PROC_BIND'] == 'TRUE'


def _ipopt_here():
    try:
        from lcsolver.solvers.solver import any_ipopt_available
        return bool(any_ipopt_available())
    except Exception:
        return False


@pytest.mark.skipif(not available or not _ipopt_here(),
                    reason='needs a usable IPOPT')
def test_a_library_lacking_the_solver_fails_the_probe():
    """The MA27-only CoinHSL archive is a real library that genuinely lacks
    ma97 -- handing it as hsllib must fail the probe cleanly, not crash."""
    lib = os.path.expanduser(
        '~/software/ipopt/HSL_build/lib/libcoinhsl.dylib')
    if not os.path.exists(lib):
        pytest.skip('no local coinhsl library to try')
    with pytest.raises(SolverUnavailable, match='ma97'):
        lcsolver.solve(_gp(), linear_solver='ma97',
                       linear_solver_library=lib)


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
