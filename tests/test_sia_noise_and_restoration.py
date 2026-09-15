"""The endgame lessons from the MSES camber run, as behavior.

Four mechanisms: a declared analysis_noise floors the feasibility band the
guards and the relative-change stop measure against; near-tolerance
violations are repaired in place instead of by a Phase-I hop; tau decays
inside the relief band instead of parking at its escalated value; and the
relative-change status says when the trust radius forced the small step.
"""
import numpy as np
import pytest

try:
    import lcsolver
    from lcsolver import Formulation
    from lcsolver.solvers.sequential.sia import (
        SIAOptions,
        SIAResult,
        _relative_change_converged,
    )
    available = True
except Exception:                                    # pragma: no cover
    available = False


class _StubProblem:
    """objective_value and violation of a fabricated iterate history."""

    def __init__(self, viol):
        self._viol = viol

    def objective_value(self, x):
        return float(np.sum(x))

    def constraint_values(self, x):
        return []


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestOptions:

    def test_new_options_exist_with_safe_defaults(self):
        o = SIAOptions()
        assert o.analysis_noise == 0.0            # off by default
        assert o.tau_relief_band == 10.0
        assert o.restoration_inplace_band == 100.0

    def test_options_are_settable(self):
        o = SIAOptions(analysis_noise=1e-4)
        assert o.analysis_noise == 1e-4


def _converge_pair(viol, binding=False, **opt):
    """Run the stop rule over two nearly identical accepted iterates."""
    import lcsolver.solvers.sequential.sia as sia
    problem = _StubProblem(viol)
    real = sia._violation
    sia._violation = lambda p, x: viol
    try:
        options = SIAOptions(objective_reltol=1e-3, variable_reltol=1e-2,
                             **opt)
        res = SIAResult()
        res.history = [np.array([1.0, 2.0]), np.array([1.0, 2.0000001])]
        res.objectives = [3.0, 3.0000001]
        res._relative_change_prev = (res.history[0], res.objectives[0])
        res._rel_step_binding = binding
        fired = _relative_change_converged(problem, res, options, k=9)
        return fired, res
    finally:
        sia._violation = real


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestRelativeChangeGate:

    def test_infeasible_point_still_refused(self):
        fired, _ = _converge_pair(viol=1e-3)
        assert not fired

    def test_analysis_noise_floors_the_gate(self):
        fired, res = _converge_pair(viol=1e-5, analysis_noise=1e-4)
        assert fired
        assert 'analysis noise' in res.status

    def test_true_feasible_status_carries_no_noise_note(self):
        fired, res = _converge_pair(viol=1e-9)
        assert fired
        assert 'analysis noise' not in res.status

    def test_binding_radius_is_named_in_the_status(self):
        fired, res = _converge_pair(viol=1e-9, binding=True)
        assert fired
        assert 'trust radius was binding' in res.status

    def test_free_step_carries_no_binding_note(self):
        fired, res = _converge_pair(viol=1e-9, binding=False)
        assert fired
        assert 'trust radius' not in res.status


def _ipopt_here():
    try:
        from lcsolver.solvers.solver import _ipopt_available
        return bool(_ipopt_available())
    except Exception:
        return False


@pytest.mark.skipif(not available or not _ipopt_here(),
                    reason='needs a usable IPOPT')
def test_the_small_sp_anchor_still_converges():
    """The mechanisms are defaults-on for restoration/zigzag/tau; the tiny
    SP anchor must land on its analytic optimum unchanged."""
    f = Formulation()
    x = f.Variable('x', 1.0, '-', 'x', bounds=[1e-3, 10.0])
    y = f.Variable('y', 1.0, '-', 'y', bounds=[1e-3, 10.0])
    f.Objective(x)
    f.ConstraintList([x >= y / (1 + y), y >= 0.5])
    sol = lcsolver.solve(f, options=SIAOptions(max_iterations=60))
    assert abs(float(sol['primal objective']) - 1.0 / 3.0) < 1e-4


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
