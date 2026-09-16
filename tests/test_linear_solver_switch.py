"""The linear_solver switch: solve(f, linear_solver='ma27'|'mumps'|...).

The switch validates the NAME before any backend runs, probes AVAILABILITY
against the build the chosen route actually uses, and threads the choice
through every IPOPT call site (raw NLP, log-space GP, SIA sub-problems,
PCCP inner solves). cvxopt has no such option, so pairing them is refused.
"""
import pytest

try:
    import lcsolver
    from lcsolver import Formulation
    from lcsolver.core.errors import SolverUnavailable
    from lcsolver import environment as env
    from lcsolver.environment import (
        KNOWN_IPOPT_LINEAR_SOLVERS,
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


def _sp():
    f = Formulation()
    x = f.Variable('x', 1.0, '-', 'x', bounds=[1e-3, 10.0])
    y = f.Variable('y', 1.0, '-', 'y', bounds=[1e-3, 10.0])
    f.Objective(x)
    f.ConstraintList([x >= y / (1 + y), y >= 0.5])
    return f


def _ipopt_here():
    try:
        from lcsolver.solvers.solver import any_ipopt_available
        return bool(any_ipopt_available())
    except Exception:
        return False


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestNameValidation:
    """Pure logic -- runs everywhere, no IPOPT needed."""

    def test_unknown_name_is_refused_before_any_backend(self):
        with pytest.raises(ValueError, match='unknown linear_solver'):
            lcsolver.solve(_gp(), linear_solver='ma28')

    def test_cvxopt_solver_with_linear_solver_is_refused(self):
        with pytest.raises(ValueError, match='cvxopt has no such option'):
            lcsolver.solve(_gp(), solver='cvxopt', linear_solver='ma27')

    def test_cvxopt_backend_with_linear_solver_is_refused(self):
        with pytest.raises(ValueError, match='cvxopt has no such option'):
            lcsolver.solve(_gp(), convex_backend='cvxopt',
                           linear_solver='ma27')

    def test_require_rejects_an_unknown_name(self):
        with pytest.raises(ValueError, match='unknown linear_solver'):
            require_linear_solver('umfpack')

    def test_names_are_normalized(self, monkeypatch):
        monkeypatch.setattr(env, 'linear_solver_available',
                            lambda name, executable=None, library=None: True)
        assert require_linear_solver(' MA27 ') == 'ma27'

    def test_unavailable_solver_names_what_is_available(self, monkeypatch):
        monkeypatch.setattr(
            env, 'linear_solver_available',
            lambda name, executable=None, library=None: name == 'ma27')
        with pytest.raises(SolverUnavailable) as ctx:
            require_linear_solver('pardiso')
        msg = str(ctx.value)
        assert "'pardiso'" in msg
        assert 'ma27' in msg                     # says what IS there

    def test_known_list_covers_the_headline_three(self):
        for name in ('mumps', 'ma27', 'pardiso'):
            assert name in KNOWN_IPOPT_LINEAR_SOLVERS


@pytest.mark.skipif(not available, reason='LCsolver import failed')
@pytest.mark.skipif(not available or not _ipopt_here(),
                    reason='needs a usable IPOPT')
class TestRealSolves:

    def _an_available_solver(self):
        exe = env.ipopt_executable()
        for name in ('ma27', 'mumps'):
            if env.linear_solver_available(name, exe):
                return name
        pytest.skip('neither ma27 nor mumps available in this build')

    def test_gp_solves_with_an_explicit_linear_solver(self):
        name = self._an_available_solver()
        res = lcsolver.solve(_gp(), linear_solver=name)
        assert res['status'] in ('ok', 'optimal')
        assert abs(float(res.objective) - 2.0) < 1e-6
        assert res.get('linear_solver') == name

    def test_raw_route_records_the_linear_solver(self):
        name = self._an_available_solver()
        from lcsolver.solvers.ipopt.NLP import ipopt_solve
        res = ipopt_solve(_gp(), linear_solver=name)
        assert res['linear_solver'] == name

    def test_missing_solver_raises_solverunavailable(self):
        exe = env.ipopt_executable()
        missing = next((s for s in ('mumps', 'ma27', 'pardiso')
                        if not env.linear_solver_available(s, exe)), None)
        if missing is None:
            pytest.skip('every headline solver is available in this build')
        with pytest.raises(SolverUnavailable):
            lcsolver.solve(_gp(), linear_solver=missing)

    def test_sia_route_threads_the_choice_into_its_options(self):
        name = self._an_available_solver()
        from lcsolver.solvers.sequential.sia import SIAOptions
        opts = SIAOptions(max_iterations=50)
        res = lcsolver.solve(_sp(), options=opts, linear_solver=name)
        assert opts.ipopt_options['linear_solver'] == name
        assert abs(float(res['primal objective']) - 1.0 / 3.0) < 1e-4

    def test_an_explicit_sia_ipopt_option_wins(self):
        name = self._an_available_solver()
        from lcsolver.solvers.sequential.sia import SIAOptions
        opts = SIAOptions(max_iterations=50)
        opts.ipopt_options['linear_solver'] = name  # user's own setting
        lcsolver.solve(_sp(), options=opts, linear_solver=name)
        assert opts.ipopt_options['linear_solver'] == name


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
