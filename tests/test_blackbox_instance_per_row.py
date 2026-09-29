"""A box object may be wired into ONE RuntimeConstraint.

setOptimizationVariables and set_external_model both store the wiring on the
box object, so sharing one box between rows leaves every block pointing at the
last row's variables. That failure is silent -- the solver reports the right
number of grey-box rows and converges -- so the wiring has to raise.
"""
import unittest

try:
    import pyomo.environ as pyo  # noqa: F401
    from lcsolver import Formulation, BlackBoxFunctionModel, BBVariable
    available = True
except Exception:                                    # pragma: no cover
    available = False


if available:
    class _Square(BlackBoxFunctionModel):
        def __init__(self):
            super().__init__()
            self.inputs = [BBVariable(name='x', units='', size=0,
                                      description='in')]
            self.outputs = [BBVariable(name='y', units='', size=0,
                                       description='out')]
            self.availableDerivative = 1

        def BlackBox(self, *args, **kwargs):
            x = self.parseInputs(*args, **kwargs).x
            return self.packOutputs([x ** 2], [[2.0 * x]])


@unittest.skipIf(not available, 'lcsolver import failed')
class TestOneBoxPerRow(unittest.TestCase):

    def _f(self, n):
        f = Formulation()
        f.Variable(name='x', guess=2.0, units='', description='x')
        for i in range(n):
            f.Variable(name='y%d' % i, guess=4.0, units='', description='y')
        return f

    def test_two_rows_sharing_one_box_raise(self):
        f = self._f(2)
        box = _Square()
        f.RuntimeConstraint([f.y0], ['>='], [f.x], box)
        with self.assertRaises(ValueError) as cm:
            f.RuntimeConstraint([f.y1], ['>='], [f.x], box)
        self.assertIn('OWN box instance', str(cm.exception))

    def test_a_box_per_row_is_accepted(self):
        f = self._f(2)
        f.RuntimeConstraint([f.y0], ['>='], [f.x], _Square())
        f.RuntimeConstraint([f.y1], ['>='], [f.x], _Square())
        self.assertEqual(len(f._runtimeConstraint_keys), 2)


if __name__ == '__main__':
    unittest.main()
