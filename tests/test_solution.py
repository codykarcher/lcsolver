"""The Solution object: what a solve produced, held on its own."""
import unittest
import warnings

try:
    import pyomo.environ as pyo
    from pyomo.environ import units

    from lcsolver import Formulation
    from lcsolver.objects.solution import Entry, Solution
    from lcsolver.solvers import solver as solver_module
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


@unittest.skipIf(not available, 'LCsolver import failed')
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
        self.assertIn('wing.S', text)          # dotted for display
        self.assertNotIn('wing_S', text)
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

    def test_grouped_names_print_dotted_and_sort_after_ungrouped(self):
        """`wing_box_t_cap` shows as `wing.box.t_cap`, and a model's own
        quantities come first so they are not buried among namespaced ones."""
        f = Formulation()
        f.Variable('W_total', 1000.0, 'N', 'total weight', bounds=[1.0, 1e6])
        w = f.group('wing')
        w.Variable('AR', 11.0, '-', 'aspect ratio', bounds=[1.0, 20.0])
        w.group('box').Variable('t_cap', 0.01, 'm', 'cap', bounds=[1e-4, 1.0])
        lg = f.group('landing_gear')
        lg.Variable('d_strut', 0.1, 'm', 'strut', bounds=[0.01, 1.0])
        f.Objective(f.W_total)
        f.Constraint(f.W_total >= 1000.0 * units.N)

        sol = f.solution
        self.assertEqual(sol.display_name('wing_box_t_cap'), 'wing.box.t_cap')
        # a group whose own name contains an underscore must keep it
        self.assertEqual(sol.display_name('landing_gear_d_strut'),
                         'landing_gear.d_strut')
        self.assertEqual(sol.display_name('W_total'), 'W_total')

        shown = [ln for ln in str(sol).splitlines() if '  :  ' in ln]
        order = [ln.split('  :  ')[0].strip() for ln in shown]
        self.assertEqual(order[0], 'W_total')       # ungrouped first
        self.assertEqual(order[1:], ['landing_gear.d_strut', 'wing.AR',
                                     'wing.box.t_cap'])

    def test_indexing_still_uses_the_real_name(self):
        """Dots are for reading. The stored name is what everything else uses."""
        f = Formulation()
        w = f.group('wing')
        w.Variable('AR', 11.0, '-', 'aspect ratio', bounds=[1.0, 20.0])
        f.Objective(f.wing_AR)
        sol = f.solution
        self.assertAlmostEqual(sol['wing_AR'], 11.0)
        self.assertIn('wing_AR', sol)

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


@unittest.skipIf(not available, 'LCsolver import failed')
class TestEmptyPrefixGroup(unittest.TestCase):
    """A group that namespaces nothing must not claim the whole model.

    A builder shared between a standalone model and a larger one that mounts
    it under a prefix gets prefix='' in the standalone case. Every name starts
    with the empty string.
    """

    def test_an_empty_prefix_does_not_swallow_every_name(self):
        from lcsolver.objects.solution import Solution, Entry
        sol = Solution(objective=1.0,
                       variables={'x': Entry('x', 1.0), 'y': Entry('y', 2.0)},
                       groups=[('', 'eng')])
        self.assertEqual(sol.display_name('x'), 'x')
        self.assertEqual(sol.display_name('y'), 'y')

    def test_a_real_prefix_still_applies(self):
        from lcsolver.objects.solution import Solution, Entry
        sol = Solution(objective=1.0,
                       variables={'Eng_M': Entry('Eng_M', 1.0)},
                       groups=[('Eng_', 'eng')])
        self.assertEqual(sol.display_name('Eng_M'), 'eng.M')


@unittest.skipIf(not available, 'LCsolver import failed')
class TestSensitivityDisplay(unittest.TestCase):
    """`top`, the threshold, and hiding what the problem does not determine."""

    @staticmethod
    def _sens(text):
        """Just the sensitivity block: the names also appear above it."""
        return text.split('Sensitivities', 1)[1]

    def _solution(self, n=6, ambiguous=()):
        from lcsolver.objects.solution import Solution, Entry
        sens = {f'c{i}': (n - i) * 1.0 for i in range(n)}
        constants = {k: Entry(k, 1.0, None, '') for k in sens}
        return Solution(objective=1.0, constants=constants,
                        sensitivities=sens, ambiguous=ambiguous)

    def test_top_keeps_the_n_largest(self):
        text = self._sens(self._solution().summary(top=2))
        self.assertIn('c0', text)                 # 6.0, largest
        self.assertIn('c1', text)                 # 5.0
        self.assertNotIn('c4', text)
        self.assertIn('4 smaller not shown', text)

    def test_top_selects_on_magnitude_not_display_order(self):
        """Grouped names sort last for display but must not sort last for `top`."""
        from lcsolver.objects.solution import Solution, Entry
        sens = {'plain': 0.5, 'wing_AR': 9.0}
        sol = Solution(objective=1.0,
                       constants={k: Entry(k, 1.0) for k in sens},
                       sensitivities=sens, groups=[('wing_', 'wing')])
        text = self._sens(sol.summary(top=1))
        self.assertIn('wing.AR', text)            # the larger one
        self.assertNotIn('plain', text)

    def test_a_threshold_drops_the_small_ones(self):
        text = self._sens(self._solution().summary(sensitivity_tol=4.5))
        self.assertIn('c0', text)
        self.assertNotIn('c3', text)
        self.assertIn('omitted', text)

    def test_undetermined_sensitivities_are_hidden_by_default(self):
        text = self._sens(self._solution(ambiguous={'c0'}).summary())
        self.assertNotIn('c0', text)
        self.assertIn('not determined by the problem', text)

    def test_they_can_be_shown_and_are_marked(self):
        text = self._sens(
            self._solution(ambiguous={'c0'}).summary(show_ambiguous=True))
        line = [l for l in text.splitlines() if 'c0' in l][0]
        self.assertTrue(line.rstrip().endswith('?'))

    def test_an_indexed_variable_prints_one_row_per_element(self):
        """`get_variables` yields components; an indexed one is not a number.

        Asking Pyomo to evaluate it raises and logs a page of ERROR lines.
        """
        from lcsolver.solvers.solver import solve

        f = Formulation()
        x = f.Variable('x', 1.0, 'm', 'a scalar')
        v = f.Variable('v', 1.0, 'm', 'a vector', size=3)
        f.Objective(x + sum(v[i] for i in range(3)))
        f.Constraint(x >= 2.0 * units.m)
        for i in range(3):
            f.Constraint(v[i] >= (i + 1) * 1.0 * units.m)
        solve(f, sensitivities=False)

        text = f.solution.summary()
        for i in range(3):
            self.assertIn(f'v[{i}]', text)
        self.assertIn('a vector', text)           # description from the parent
