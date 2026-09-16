"""Value-only queries never pay for derivatives.

Derivatives are the expensive half of a black-box evaluation (a
finite-difference sweep on a values-only box, an adjoint solve in a real
analysis), and the sequential solvers ask for values alone far more often
than they linearize: line searches, violation checks, restores.  Three
layers make that cheap: evaluate_outputs fills the cache values-only, a
later jacobian request at the same point upgrades in place, and the SLCP
Signomial wrappers route value queries through a values-only callback.
"""
import numpy as np
import pytest

try:
    import lcsolver
    from lcsolver import BlackBoxFunctionModel, Formulation
    from lcsolver.solvers.sequential.slcp import CachedSignomial, Signomial
    available = True
except Exception:                                    # pragma: no cover
    available = False


class _CountingSquare(BlackBoxFunctionModel if available else object):
    """y = x**2, values only, counting every BlackBox call."""

    def __init__(self):
        super().__init__()
        self.inputs.append('x', units='-', description='in')
        self.outputs.append(name='y', units='-')
        self.availableDerivative = 0
        self.calls = 0

    def BlackBox(self, x):
        self.calls += 1
        x = self.sanitizeInputs(x, strip_units=True)
        return float(x) ** 2


def _wired():
    f = Formulation()
    f.Variable('x', 2.0, '-', 'in', bounds=[0.5, 4.0])
    f.Variable('y', 4.0, '-', 'out', bounds=[0.1, 20.0])
    f.Objective(f.y)
    box = _CountingSquare()
    f.RuntimeConstraint([f.y], ['=='], [f.x], box)
    from pyomo.contrib.pynumero.interfaces.external_grey_box import (
        ExternalGreyBoxBlock,
    )
    blk = next(iter(f.component_data_objects(ExternalGreyBoxBlock)))
    return blk.get_external_model()


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestGreyboxSplit:

    def test_values_cost_one_call_and_need_no_permission(self):
        ex = _wired()
        ex.set_input_values(np.array([3.0]))
        assert float(ex.evaluate_outputs()[0]) == 9.0
        assert ex.calls == 1                     # no FD sweep, no error

    def test_values_are_cached_at_the_point(self):
        ex = _wired()
        ex.set_input_values(np.array([3.0]))
        ex.evaluate_outputs()
        ex.evaluate_outputs()
        assert ex.calls == 1

    def test_jacobian_still_needs_permission(self):
        ex = _wired()
        ex.set_input_values(np.array([3.0]))
        ex.evaluate_outputs()                    # fine
        with pytest.raises(ValueError) as ctx:
            ex.evaluate_jacobian_outputs()       # not fine
        assert 'allow_blackbox_finite_difference' in str(ctx.value)

    def test_jacobian_upgrade_reuses_the_cached_values(self):
        ex = _wired()
        ex.allow_finite_difference = True
        ex.set_input_values(np.array([3.0]))
        ex.evaluate_outputs()
        assert ex.calls == 1
        jac = ex.evaluate_jacobian_outputs()
        # central differences on one input: 2 calls, base NOT re-evaluated
        assert ex.calls == 3
        assert abs(float(jac.todense()[0, 0]) - 6.0) < 1e-5


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestSignomialValuePath:

    def _pair(self):
        counts = {'full': 0, 'value': 0}

        def fn(x):
            counts['full'] += 1
            return float(x[0]) ** 2, np.array([2.0 * x[0]])

        def fn_value(x):
            counts['value'] += 1
            return float(x[0]) ** 2

        return counts, fn, fn_value

    def test_value_query_skips_the_full_callback(self):
        counts, fn, fv = self._pair()
        s = Signomial(fn, 1, fn_value=fv)
        assert s(np.array([3.0])) == 9.0
        assert counts == {'full': 0, 'value': 1}

    def test_cached_value_then_gradient_upgrades(self):
        counts, fn, fv = self._pair()
        s = CachedSignomial(fn, 1, fn_value=fv)
        x = np.array([3.0])
        assert s(x) == 9.0                       # value path
        assert s(x) == 9.0                       # cache hit
        assert counts == {'full': 0, 'value': 1}
        assert s.grad(x)[0] == 6.0               # upgrade in place
        assert counts == {'full': 1, 'value': 1}
        assert s(x) == 9.0                       # upgraded entry serves values
        assert counts == {'full': 1, 'value': 1}

    def test_without_fn_value_behavior_is_unchanged(self):
        counts, fn, _ = self._pair()
        s = CachedSignomial(fn, 1)
        x = np.array([3.0])
        assert s(x) == 9.0
        assert s.grad(x)[0] == 6.0
        assert counts == {'full': 1, 'value': 0}


def _ipopt_here():
    try:
        from lcsolver.solvers.solver import any_ipopt_available
        return bool(any_ipopt_available())
    except Exception:
        return False


@pytest.mark.skipif(not available or not _ipopt_here(),
                    reason='needs a usable IPOPT')
def test_fd_solve_still_lands_on_the_answer():
    """End to end: the split must not change what an FD-box solve finds."""
    f = Formulation()
    f.Variable('x', 2.0, '-', 'in', bounds=[0.5, 4.0])
    f.Variable('y', 4.0, '-', 'out', bounds=[0.1, 20.0])
    f.Objective(f.y)
    f.RuntimeConstraint([f.y], ['=='], [f.x], _CountingSquare())
    res = lcsolver.solve(f, allow_blackbox_finite_difference=True)
    assert res['status'] in ('ok', 'optimal')
    import pyomo.environ as pyo
    assert abs(pyo.value(f.y) - pyo.value(f.x) ** 2) < 1e-5


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
