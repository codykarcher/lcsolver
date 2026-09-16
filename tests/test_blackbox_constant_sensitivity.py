"""Constants consumed inside a black box earn honest reported sensitivities.

A Constant wired into a RuntimeConstraint reaches the box as a trailing
argument; its jacobian columns are chain-ruled by the KKT pass into the
reported d(objective)/d(constant), same as if written algebraically.
Anchor model: min y s.t. y == c*x**2, x >= 2, so d log(f*)/d log(c) = 1.
"""
import numpy as np
import pytest

try:
    import pyomo.environ as pyo

    import lcsolver
    from lcsolver import BlackBoxFunctionModel, Formulation
    available = True
except Exception:                                    # pragma: no cover
    available = False


def _ipopt_here():
    try:
        from lcsolver.solvers.solver import any_ipopt_available
        return bool(any_ipopt_available())
    except Exception:
        return False


class _ScaledSquare(BlackBoxFunctionModel if available else object):
    """y = c * x**2, with c a declared CONSTANT and analytic derivatives."""

    def __init__(self):
        super().__init__()
        self.inputs.append('x', units='-', description='in')
        self.constants.append('c', units='-', description='scale factor')
        self.outputs.append(name='y', units='-')
        self.availableDerivative = 1

    def BlackBox(self, x, c):
        x, c = self.sanitizeInputs(x, c, strip_units=True)
        x, c = float(x), float(c)
        return self.packOutputs(c * x ** 2, [2.0 * c * x, x ** 2])


class _ScaledSquareNoDeriv(BlackBoxFunctionModel if available else object):
    """Same physics, values only -- the FD fallback must cover c too."""

    def __init__(self):
        super().__init__()
        self.inputs.append('x', units='-', description='in')
        self.constants.append('c', units='-', description='scale factor')
        self.outputs.append(name='y', units='-')
        self.availableDerivative = 0

    def BlackBox(self, x, c):
        x, c = self.sanitizeInputs(x, c, strip_units=True)
        return float(c) * float(x) ** 2


def _model(box, cval=1.5):
    f = Formulation()
    f.Variable('x', 2.0, '-', 'in', bounds=[2.0, 10.0])
    f.Variable('y', 4.0 * cval, '-', 'out', bounds=[0.1, 1000.0])
    f.Constant('c', cval, '-', 'scale factor')
    f.Objective(f.y)
    f.RuntimeConstraint([f.y], ['=='], [f.x], box, constants=[f.c])
    return f


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestWiring:

    def test_mismatched_count_is_refused(self):
        f = Formulation()
        f.Variable('x', 2.0, '-', 'in')
        f.Variable('y', 4.0, '-', 'out')
        f.Objective(f.y)
        with pytest.raises(ValueError, match='positional'):
            f.RuntimeConstraint([f.y], ['=='], [f.x], _ScaledSquare())

    def test_non_param_is_refused(self):
        f = Formulation()
        f.Variable('x', 2.0, '-', 'in')
        f.Variable('y', 4.0, '-', 'out')
        f.Variable('z', 1.0, '-', 'not a constant')
        f.Objective(f.y)
        with pytest.raises(ValueError, match='Constants'):
            f.RuntimeConstraint([f.y], ['=='], [f.x], _ScaledSquare(),
                                constants=[f.z])

    def test_constant_jacobian_is_exposed(self):
        box = _ScaledSquare()
        f = _model(box)
        from pyomo.contrib.pynumero.interfaces.external_grey_box import (
            ExternalGreyBoxBlock,
        )
        blk = next(iter(f.component_data_objects(ExternalGreyBoxBlock)))
        ex = blk.get_external_model()
        ex.set_input_values(np.array([2.0]))
        cj = ex.constant_jacobian()
        assert set(cj) == {'c'}
        assert abs(float(cj['c'][0]) - 4.0) < 1e-12    # d(c x^2)/dc = x^2

    def test_optimizer_jacobian_excludes_the_constant_column(self):
        box = _ScaledSquare()
        f = _model(box)
        from pyomo.contrib.pynumero.interfaces.external_grey_box import (
            ExternalGreyBoxBlock,
        )
        blk = next(iter(f.component_data_objects(ExternalGreyBoxBlock)))
        ex = blk.get_external_model()
        ex.set_input_values(np.array([2.0]))
        jac = ex.evaluate_jacobian_outputs().todense()
        assert jac.shape == (1, 1)                     # x only, never c
        assert abs(float(jac[0, 0]) - 6.0) < 1e-12     # 2*c*x = 2*1.5*2


@pytest.mark.skipif(not available or not _ipopt_here(),
                    reason='needs a usable IPOPT')
class TestReportedSensitivity:

    def test_the_constant_earns_its_exact_sensitivity(self):
        f = _model(_ScaledSquare(), cval=1.5)
        res = lcsolver.solve(f)
        sens = res['sensitivities']
        assert 'c' in sens, f'c missing from {sorted(sens)}'
        assert abs(float(sens['c']) - 1.0) < 1e-3      # d ln f*/d ln c == 1

    def test_the_fd_route_reports_it_too(self):
        f = _model(_ScaledSquareNoDeriv(), cval=2.5)
        res = lcsolver.solve(f, allow_blackbox_finite_difference=True)
        sens = res['sensitivities']
        assert abs(float(sens['c']) - 1.0) < 1e-3

    def test_a_box_without_constants_is_unchanged(self):
        class _Plain(BlackBoxFunctionModel):
            def __init__(self):
                super().__init__()
                self.inputs.append('x', units='-', description='in')
                self.outputs.append(name='y', units='-')
                self.availableDerivative = 1

            def BlackBox(self, x):
                x = self.sanitizeInputs(x, strip_units=True)
                return self.packOutputs(float(x) ** 2, [2.0 * float(x)])

        f = Formulation()
        f.Variable('x', 2.0, '-', 'in', bounds=[2.0, 10.0])
        f.Variable('y', 4.0, '-', 'out', bounds=[0.1, 1000.0])
        f.Objective(f.y)
        f.RuntimeConstraint([f.y], ['=='], [f.x], _Plain())
        res = lcsolver.solve(f)
        assert res['status'] in ('ok', 'optimal')


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
