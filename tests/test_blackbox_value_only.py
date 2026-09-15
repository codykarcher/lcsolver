"""Values-only black boxes: refuse silently missing derivatives, allow FD.

A box with ``availableDerivative = 0`` gives no jacobian, so the solve must
ERROR naming the box and the fix -- unless the modeller explicitly grants
``allow_blackbox_finite_difference=True``, which builds a central-difference
jacobian shaped exactly as packOutputs would shape an author-supplied one.
"""
import numpy as np
import pytest

try:
    import lcsolver
    from lcsolver import BlackBoxFunctionModel, Formulation
    available = True
except Exception:                                    # pragma: no cover
    available = False


def _ipopt_here():
    try:
        from lcsolver.solvers.solver import _ipopt_available
        return bool(_ipopt_available())
    except Exception:
        return False


class _SquareNoDeriv(BlackBoxFunctionModel if available else object):
    """y = x**2, values only."""

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


class _VectorNoDeriv(BlackBoxFunctionModel if available else object):
    """y_i = c * x_i**2 + sum(x): vector in/out plus a scalar input,
    values only -- exercises every finite-difference block shape."""

    def __init__(self, n=3):
        super().__init__()
        self.inputs.append('x', units='-', description='in', size=n)
        self.inputs.append('c', units='-', description='scale')
        self.outputs.append(name='y', units='-', size=n)
        self.availableDerivative = 0

    def BlackBox(self, x, c):
        x, c = self.sanitizeInputs(x, c, strip_units=True)
        x = np.asarray(x, dtype=float)
        return float(c) * x ** 2 + x.sum()


def _wire(box, n=None):
    f = Formulation()
    if n is None:
        f.Variable('x', 2.0, '-', 'in', bounds=[0.5, 4.0])
        f.Variable('y', 4.0, '-', 'out', bounds=[0.1, 20.0])
        f.Objective(f.y)
        f.RuntimeConstraint([f.y], ['=='], [f.x], box)
    else:
        f.Variable('x', [1.0, 2.0, 3.0], '-', 'in', size=n,
                   bounds=[0.5, 4.0])
        f.Variable('c', 2.0, '-', 'scale', bounds=[0.5, 4.0])
        f.Variable('y', [1.0, 1.0, 1.0], '-', 'out', size=n,
                   bounds=[0.1, 200.0])
        f.Objective(f.sum(f.y))
        f.RuntimeConstraint([f.y], ['=='] * n, [f.x, f.c], box)
    return f


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestRefusal:

    def test_jacobian_request_without_permission_raises(self):
        box = _SquareNoDeriv()
        f = _wire(box)
        from pyomo.contrib.pynumero.interfaces.external_grey_box import (
            ExternalGreyBoxBlock,
        )
        blk = next(iter(f.component_data_objects(ExternalGreyBoxBlock)))
        ex = blk.get_external_model()
        ex.set_input_values(np.array([2.0]))
        with pytest.raises(ValueError) as ctx:
            ex.evaluate_jacobian_outputs()
        msg = str(ctx.value)
        assert 'availableDerivative=0' in msg
        assert 'allow_blackbox_finite_difference' in msg

    def test_direct_evaluation_still_works_without_permission(self):
        """A values-only box is still a fine function to CALL."""
        box = _SquareNoDeriv()
        assert float(box.BlackBox(3.0)) == 9.0


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestFiniteDifference:

    def test_fd_jacobian_matches_the_analytic_one(self):
        box = _SquareNoDeriv()
        f = _wire(box)
        from pyomo.contrib.pynumero.interfaces.external_grey_box import (
            ExternalGreyBoxBlock,
        )
        blk = next(iter(f.component_data_objects(ExternalGreyBoxBlock)))
        ex = blk.get_external_model()
        ex.allow_finite_difference = True
        ex.set_input_values(np.array([2.0]))
        jac = ex.evaluate_jacobian_outputs().todense()
        assert abs(float(jac[0, 0]) - 4.0) < 1e-5      # d(x^2)/dx at 2

    def test_fd_blocks_take_every_shape(self):
        box = _VectorNoDeriv()
        f = _wire(box, n=3)
        from pyomo.contrib.pynumero.interfaces.external_grey_box import (
            ExternalGreyBoxBlock,
        )
        blk = next(iter(f.component_data_objects(ExternalGreyBoxBlock)))
        ex = blk.get_external_model()
        ex.allow_finite_difference = True
        x = np.array([1.0, 2.0, 3.0])
        ex.set_input_values(np.concatenate([x, [2.0]]))
        jac = np.asarray(ex.evaluate_jacobian_outputs().todense())
        # d y_i / d x_j = 2*c*x_i * delta_ij + 1 ; d y_i / d c = x_i**2
        expected = 2.0 * 2.0 * np.diag(x) + 1.0
        assert np.allclose(jac[:, :3], expected, rtol=1e-4, atol=1e-4)
        assert np.allclose(jac[:, 3], x ** 2, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not available or not _ipopt_here(),
                    reason='needs a usable IPOPT')
class TestThroughSolve:

    def test_solve_errors_without_the_flag(self):
        f = _wire(_SquareNoDeriv())
        with pytest.raises(Exception) as ctx:
            lcsolver.solve(f)
        assert 'allow_blackbox_finite_difference' in str(ctx.value)

    def test_solve_succeeds_with_the_flag(self):
        box = _SquareNoDeriv()
        f = _wire(box)
        res = lcsolver.solve(f, allow_blackbox_finite_difference=True)
        assert res['status'] in ('ok', 'optimal')
        import pyomo.environ as pyo
        assert abs(pyo.value(f.y) - pyo.value(f.x) ** 2) < 1e-5
        assert box.calls == 0      # the CLONE ran, the original never did


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
