"""The Solution object: what a solve produced, held on its own."""
import unittest
import warnings

try:
    import pyomo.environ as pyo
    from pyomo.environ import units

    from edi import Formulation
    from edi.objects.solution import Entry, Solution
    from edi.solvers import solver as solver_module
    available = True
except Exception:                                    # pragma: no cover
    available = False


def _model():
    """min S  s.t.  S*AR >= k_area, AR <= AR_max.  Optimum S = k/AR_max."""
    f = Formulation()
    w = f.group('wing')
    ar = w.Variable('AR', 11.0, '-', 'aspect ratio', bounds=[1.0, 20.0])
    s = w.Variable('S', 100.0, 'm^2', 'wing area', bounds=[1.0, 1000.0])
    k = f.Constant('k_area', 400.0, 'm^2', 'required lifting area')
    cap = f.Constant('AR_max', 12.0, '-', 'structural limit')
    f.Objective(s)
    f.Constraint(s * ar >= k)
    f.Constraint(ar <= cap)
    return f


@unittest.skipIf(not available, 'EDI import failed')
class TestSolution(unittest.TestCase):

    def _solved(self):
        f = _model()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            solver_module.solve(f, diagnostics='off')
        return f

    def test_solution_reads_the_solved_model(self):
        """Not the unit-corrected clone, which is never solved."""
        f = self._solved()
        sol = f.solution

        self.assertAlmostEqual(sol['wing_S'], 400.0 / 12.0, places=4)
        self.assertAlmostEqual(sol['wing_AR'], 12.0, places=4)
        self.assertAlmostEqual(sol.objective, 400.0 / 12.0, places=4)

    def test_units_and_descriptions_are_carried(self):
        sol = self._solved().solution
        self.assertEqual(sol.variables['wing_S'].description, 'wing area')
        self.assertIn('m', str(sol.variables['wing_S'].units))
        self.assertEqual(sol.variables['wing_AR'].description, 'aspect ratio')

    def test_constants_are_separate_from_variables(self):
        sol = self._solved().solution
        self.assertEqual(set(sol.variables), {'wing_AR', 'wing_S'})
        self.assertEqual(set(sol.constants), {'k_area', 'AR_max'})
        self.assertAlmostEqual(sol['k_area'], 400.0)

    def test_printing_shows_every_section(self):
        text = str(self._solved().solution)
        for heading in ('Objective', 'Variables', 'Constants', 'Sensitivities'):
            self.assertIn(heading, text)
        self.assertIn('wing_S', text)
        self.assertIn('wing area', text)

    def test_sensitivities_are_attached_and_correct(self):
        """Doubling the required area doubles the wing; doubling the aspect
        ratio limit halves it. So +1 and -1 exactly."""
        f = self._solved()
        sol = f.solution_with_sensitivities()

        self.assertIsNotNone(sol.sensitivities)
        self.assertAlmostEqual(sol.sensitivities['k_area'], 1.0, places=3)
        self.assertAlmostEqual(sol.sensitivities['AR_max'], -1.0, places=3)
        self.assertIn('k_area', str(sol))

    def test_missing_names_raise_rather_than_return_none(self):
        sol = self._solved().solution
        with self.assertRaises(KeyError):
            sol['not_a_variable']
        self.assertIsNone(sol.get('not_a_variable'))
        self.assertIn('wing_S', sol)

    def test_reserved_names_are_refused(self):
        """A component called `solution` would shadow `f.solution`."""
        f = Formulation()
        for name in ('solution', 'sensitivities'):
            with self.assertRaises(ValueError) as ctx:
                f.Variable(name, 1.0, '-', 'shadows the accessor')
            self.assertIn('reserved', str(ctx.exception))
        with self.assertRaises(ValueError):
            f.Constant('solution', 1.0, '-', 'shadows the accessor')
        with self.assertRaises(ValueError):
            f.group('solution')

    def test_solution_stands_alone(self):
        """It is an ordinary object: constructible and printable without a model."""
        sol = Solution(objective=2.0, objective_units=None,
                       variables={'x': Entry('x', 1.0, None, 'a thing')},
                       constants={'c': Entry('c', 3.0, None, 'a constant')},
                       sensitivities={'c': 0.5})
        self.assertEqual(sol['x'], 1.0)
        self.assertIn('a thing', str(sol))
        self.assertIn('0.5000', str(sol))
        self.assertEqual(sol.to_dict(), {'x': 1.0, 'c': 3.0})


if __name__ == '__main__':
    unittest.main()
