"""The failure chain behind a grey-box model silently landing on raw IPOPT.

Found via an MSES multifidelity model: a single variable declared with
``bounds=[0.0, 0.1]`` cleared the SP flag with no message (a zero lower bound
is not representable in log space), which sent the grey-box solve down the
raw IPOPT route instead of SIA, silently discarding the caller's SIAOptions
-- and the first visible symptom was an AttributeError from deep inside
Pyomo's unit converter, because the model also packed np.diag matrices where
scalar-input jacobian blocks belong.

Three defenses, one per link in that chain: the detector BLAMES the
nonpositive bound, the solver WARNS when a grey-box model falls to the raw
route, and packOutputs VALIDATES jacobian block shapes in the modeller's own
stack frame.
"""
import numpy as np
import pytest

try:
    import pyomo.environ as pyo

    import lcsolver
    from lcsolver import BlackBoxFunctionModel, Formulation
    from lcsolver.presolve.structureDetector import structure_detector
    from lcsolver.presolve.unitCorrector import unit_corrector
    from lcsolver.solvers.sequential.sia import SIAOptions
    available = True
except Exception:                                    # pragma: no cover
    available = False


def _gp_with_bound(lb):
    f = Formulation()
    x = f.Variable('x', 1.0, '-', 'x', bounds=[lb, 10.0])
    y = f.Variable('y', 1.0, '-', 'y', bounds=[0.1, 10.0])
    f.Objective(x + y)
    f.Constraint(x * y >= 1.0)
    return f


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestNonpositiveBoundIsBlamed:

    def test_zero_lower_bound_clears_gp_sp_with_blame(self):
        s = structure_detector(unit_corrector(_gp_with_bound(0.0)))
        assert not s['Geometric_Program'][0]
        assert not s['Signomial_Program'][0]
        blockers = s.get('blockers', {}).get('Signomial_Program', [])
        assert blockers, 'the disqualification must be blamed, not silent'
        names = [nm for nm, _, _ in blockers]
        assert any("variable 'x'" in nm for nm in names)
        reasons = ' '.join(reason for _, reason, _ in blockers)
        assert 'nonpositive lower bound' in reasons

    def test_negative_lower_bound_is_blamed_too(self):
        s = structure_detector(unit_corrector(_gp_with_bound(-1.0)))
        assert not s['Signomial_Program'][0]
        blockers = s.get('blockers', {}).get('Signomial_Program', [])
        assert blockers
        reasons = ' '.join(reason for _, reason, _ in blockers)
        assert 'nonpositive lower bound' in reasons

    def test_positive_lower_bound_stays_gp(self):
        s = structure_detector(unit_corrector(_gp_with_bound(0.1)))
        assert s['Geometric_Program'][0]
        assert s['Signomial_Program'][0]


class _Square(BlackBoxFunctionModel):
    """y_i = c * x_i**2 -- one vector pair, one scalar input."""

    def __init__(self, n=3):
        super().__init__()
        self.inputs.append('x', units='-', description='x', size=n)
        self.inputs.append('c', units='-', description='scale')
        self.outputs.append(name='y', units='-', size=n)
        self.availableDerivative = 1
        self._n = n

    def BlackBox(self, x, c):
        x, c = self.sanitizeInputs(x, c, strip_units=True)
        x = np.asarray(x, dtype=float)
        return self.packOutputs(c * x**2, [np.diag(2.0 * c * x), x**2])


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestJacobianBlockShapes:

    def test_correct_shapes_pass(self):
        box = _Square()
        vals, jac = box.BlackBox(np.array([1.0, 2.0, 3.0]), 2.0)
        assert len(jac) == 2

    def test_diag_on_a_scalar_input_block_is_named(self):
        box = _Square()
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError) as ctx:
            # np.diag on the scalar-input block: the mistake this guards
            box.packOutputs(2.0 * x**2,
                            [np.diag(4.0 * x), np.diag(x**2)])
        msg = str(ctx.value)
        assert 'd(y)/d(c)' in msg
        assert '(3,)' in msg and '(3, 3)' in msg
        assert 'np.diag' in msg

    def test_wrong_vector_length_is_named(self):
        box = _Square()
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError) as ctx:
            box.packOutputs(2.0 * x**2,
                            [np.diag(4.0 * x), np.array([1.0, 2.0])])
        assert 'd(y)/d(c)' in str(ctx.value)

    def test_scalar_scalar_block_is_unchecked_shape_empty(self):
        box = BlackBoxFunctionModel()
        box.inputs.append('u', units='-', description='u')
        box.outputs.append(name='v', units='-')
        box.availableDerivative = 1
        vals, jac = box.packOutputs(4.0, [2.0])
        assert pyo.value(jac[0]) == 2.0


class _PassThrough(BlackBoxFunctionModel):
    """out == in, scalar; enough grey box to trigger routing."""

    def __init__(self):
        super().__init__()
        self.inputs.append('xin', units='-', description='in')
        self.outputs.append(name='xout', units='-')
        self.availableDerivative = 1

    def BlackBox(self, xin):
        # a single input comes back bare, not as a tuple
        xin = self.sanitizeInputs(xin, strip_units=True)
        return self.packOutputs(float(xin), [1.0])


@pytest.mark.skipif(not available, reason='LCsolver import failed')
def test_greybox_fallback_to_raw_route_warns():
    """A grey-box model whose algebra is not GP/SP must say so when it
    reroutes, and say that the SIAOptions are discarded.

    solve() runs quiet by default -- every warning is captured into
    ``messages`` rather than emitted -- so the warning is read back from
    the model, which carries the captured list either way.
    """
    from lcsolver.core.errors import SolverUnavailable

    f = _gp_with_bound(0.0)            # zero bound: not-SP, with blame
    f.Variable('xin', 1.0, '-', 'in', bounds=[0.5, 2.0])
    f.Variable('xout', 1.0, '-', 'out', bounds=[0.5, 2.0])
    f.RuntimeConstraint([f.xout], ['=='], [f.xin], _PassThrough())
    try:
        lcsolver.solve(f, options=SIAOptions(max_iterations=2))
    except SolverUnavailable as exc:
        pytest.skip(f'raw grey-box route not runnable here: {exc}')
    msgs = getattr(f, '_solve_messages', [])
    lc207 = [t for t in msgs if 'LC-W207' in t]
    assert lc207, f'expected an LC-W207 message, got: {msgs}'
    assert 'SIAOptions' in lc207[0]
    assert "variable 'x'" in lc207[0]


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
