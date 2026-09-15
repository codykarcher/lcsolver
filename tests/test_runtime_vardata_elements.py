"""Elements of an indexed Var wire into a RuntimeConstraint as scalars.

The per-segment idiom -- one black box per mission segment, wired from a
slice: for A, CL in zip(f.Alpha[:], f.C_L[:]) -- hands RuntimeConstraint
bare VarData elements, not ScalarVars. Those must be accepted everywhere a
scalar is (validation, unwrapping, fillCache), or the multifidelity CAPS
examples cannot be written per segment.
"""
import numpy as np
import pytest

try:
    import lcsolver
    from lcsolver import BlackBoxFunctionModel, Formulation
    available = True
except Exception:                                    # pragma: no cover
    available = False


class _Seg(BlackBoxFunctionModel if available else object):
    """CL = 0.1*a + 0.3 for one segment."""

    def __init__(self):
        super().__init__()
        self.inputs.append('a', units='-', description='aoa-like')
        self.outputs.append(name='CL', units='-')
        self.availableDerivative = 1

    def BlackBox(self, a):
        a = float(self.sanitizeInputs(a, strip_units=True))
        return self.packOutputs(0.1 * a + 0.3, [0.1])


def _build(n=3):
    f = Formulation()
    f.Variable('a', [1.0] * n, '-', 'aoa', size=n, bounds=[0.1, 8.0])
    f.Variable('CL', [0.4] * n, '-', 'lift', size=n, bounds=[0.05, 2.0])
    f.Objective(f.sum(f.a))
    f.Constraint(f.sum(f.CL) >= 1.5)
    boxes = []
    for i, (a, cl) in enumerate(zip(f.a[:], f.CL[:])):
        box = _Seg()
        f.RuntimeConstraint([cl], ['=='], [a], box)
        boxes.append(box)
    return f, boxes


def _ipopt_here():
    try:
        from lcsolver.solvers.solver import _ipopt_available
        return bool(_ipopt_available())
    except Exception:
        return False


@pytest.mark.skipif(not available, reason='LCsolver import failed')
def test_elements_wire_and_unwrap_by_name():
    f, boxes = _build()
    assert len(boxes) == 3
    for i, box in enumerate(boxes):
        assert box.input_names() == [f'a[{i}]']
        assert box.output_names() == [f'CL[{i}]']


@pytest.mark.skipif(not available, reason='LCsolver import failed')
def test_elements_evaluate_through_fillcache():
    f, boxes = _build()
    boxes[1].set_input_values(np.array([2.0]))
    out = boxes[1].evaluate_outputs()
    assert abs(out[0] - 0.5) < 1e-12               # 0.1*2 + 0.3
    jac = boxes[1].evaluate_jacobian_outputs().todense()
    assert abs(float(jac[0, 0]) - 0.1) < 1e-12


@pytest.mark.skipif(not available or not _ipopt_here(),
                    reason='needs a usable IPOPT')
def test_the_per_segment_model_solves():
    f, _ = _build()
    from lcsolver.solvers.sequential.sia import SIAOptions
    sol = lcsolver.solve(f, options=SIAOptions(max_iterations=60))
    assert sol.get('status') in ('optimal', 'ok', 'converged')


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
