#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Vectors and matrices: elementwise constraints, reductions, sequences.

The language rule the summing guard depends on is pinned separately, in
``test_vector_language_guarantee.py``.
"""

import numpy as np
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.environ import units

try:
    from edi import Formulation
    from edi.objects.vector import ShapeMismatch
    from edi.solvers.solver import solve

    available = True
except Exception:                                    # pragma: no cover
    available = False


@unittest.skipIf(not available, 'EDI import failed')
class TestSummingGuard(unittest.TestCase):
    """`sum(x)` used to return the sum of the index keys, silently.

    A real model asserted `L_dist_sum == sum(L_dist)` and got
    `L_dist_sum == 10`; only a unit mismatch elsewhere caught it. Iteration
    still yields keys, as Pyomo does -- it is only the accidental addition
    that is refused.
    """

    def _f(self):
        f = Formulation()
        x = f.Variable('x', 1.0, 'm', 'x', size=3)
        return f, x

    def test_bare_sum_raises_and_names_the_alternative(self):
        f, x = self._f()
        with self.assertRaises(TypeError) as ctx:
            sum(x)
        self.assertIn('f.sum(x)', str(ctx.exception))

    def test_accumulating_by_hand_is_caught_too(self):
        f, x = self._f()
        with self.assertRaises(TypeError):
            total = 0
            for i in x:
                total += i

    def test_iteration_still_yields_keys(self):
        """Pyomo's convention is preserved; nothing has to be un-learned."""
        f, x = self._f()
        self.assertEqual([int(i) for i in x], [0, 1, 2])
        self.assertEqual([str(x[i]) for i in x], ['x[0]', 'x[1]', 'x[2]'])

    def test_index_arithmetic_still_works(self):
        """`x[i-1]` is the whole point of keeping key iteration."""
        f, x = self._f()
        pairs = [str(x[i - 1]) for i in x if i > 0]
        self.assertEqual(pairs, ['x[0]', 'x[1]'])

    def test_the_explicit_forms_work(self):
        f, x = self._f()
        self.assertEqual(str(f.sum(x)), 'x[0] + x[1] + x[2]')
        self.assertEqual(str(sum(x.values())), 'x[0] + x[1] + x[2]')
        self.assertEqual(str(sum(x[i] for i in x)), 'x[0] + x[1] + x[2]')

    def test_constants_are_guarded_the_same_way(self):
        f = Formulation()
        c = f.Constant('c', 2.0, 'm', 'c', size=3)
        with self.assertRaises(TypeError):
            sum(c)
        self.assertEqual(str(f.sum(c)), 'c[0] + c[1] + c[2]')

    def test_a_multidimensional_key_is_guarded_with_the_same_message(self):
        f = Formulation()
        M = f.Variable('M', 1.0, 'm', 'M', size=[2, 2])
        with self.assertRaises(TypeError) as ctx:
            sum(M)
        self.assertIn('f.sum', str(ctx.exception))


@unittest.skipIf(not available, 'EDI import failed')
class TestElementwiseConstraints(unittest.TestCase):

    def _f(self):
        f = Formulation()
        x = f.Variable('x', 1.0, 'm', 'x', size=3)
        y = f.Variable('y', 1.0, 'm', 'y', size=3)
        return f, x, y

    def test_comparison_builds_one_constraint_per_element(self):
        f, x, y = self._f()
        cons = x >= y
        self.assertEqual(cons.shape, (3,))
        self.assertEqual([str(c) for c in cons],
                         ['y[0]  <=  x[0]', 'y[1]  <=  x[1]', 'y[2]  <=  x[2]'])

    def test_the_elements_are_the_models_own_variables(self):
        """Not copies -- a constraint from the array constrains the model."""
        f, x, y = self._f()
        self.assertIs(x[0:1][0], x[0])

    def test_equality_builds_equality_constraints(self):
        f, x, y = self._f()
        from pyomo.core.expr.relational_expr import EqualityExpression
        cons = x == y
        self.assertTrue(all(isinstance(c, EqualityExpression) for c in cons))

    def test_a_scalar_applies_to_every_element(self):
        f, x, y = self._f()
        cons = x <= 5.0 * units.m
        self.assertEqual(len(cons), 3)

    def test_strict_inequalities_are_refused(self):
        """An optimizer cannot enforce `<`; saying so beats a silent `<=`."""
        f, x, y = self._f()
        for op in (lambda: x < y, lambda: x > y, lambda: x != y):
            with self.assertRaises(TypeError):
                op()


@unittest.skipIf(not available, 'EDI import failed')
class TestElementwiseArithmetic(unittest.TestCase):
    """Building the expressions, not just comparing them.

    Comparisons alone were not enough to retire a single loop: a model writes
    `V == M * a`, and with arithmetic missing from the component that raised,
    so `for i in range(N)` had to stay to do the multiplying.
    """

    def _f(self):
        f = Formulation()
        T = f.Variable('T', 300.0, 'K', 'temperature', size=3)
        M = f.Variable('M', 0.8, '-', 'mach', size=3)
        R = f.Constant('R', 287.0, 'J/kg/K', 'gas constant')
        return f, T, M, R

    def test_vector_times_vector(self):
        f, T, M, R = self._f()
        self.assertEqual(str((M * T)[0]), 'M[0]*T[0]')

    def test_scalar_constant_times_vector_either_way_round(self):
        f, T, M, R = self._f()
        self.assertEqual(str((R * T)[0]), 'R*T[0]')
        # Pyomo normalises the operand order, so this is the same expression.
        self.assertEqual(str((T * R)[0]), 'R*T[0]')

    def test_plain_numbers(self):
        f, T, M, R = self._f()
        self.assertEqual(str((2 * T)[0]), '2*T[0]')
        self.assertEqual(str((T / 2.0)[0]), '0.5*T[0]')

    def test_powers(self):
        f, T, M, R = self._f()
        self.assertEqual(str((T ** 1.5)[0]), 'T[0]**1.5')

    def test_addition_and_subtraction_and_negation(self):
        f, T, M, R = self._f()
        self.assertEqual(str((T + T)[0]), 'T[0] + T[0]')
        self.assertIn('T[0]', str((T - T)[0]))
        self.assertIn('T[0]', str((-T)[0]))

    def test_a_whole_physical_relation(self):
        """The flight-state constraint that motivated this."""
        f, T, M, R = self._f()
        a = f.Variable('a', 300.0, 'm/s', 'speed of sound', size=3)
        cons = a == (R * T) ** 0.5
        self.assertEqual(str(cons[0]), 'a[0]  ==  (R*T[0])**0.5')

    def test_arithmetic_respects_the_shape_rule(self):
        f, T, M, R = self._f()
        M2 = f.Variable('M2', 1.0, 'm', 'M2', size=[2, 3])
        with self.assertRaises(ShapeMismatch):
            M2 == T * 2


@unittest.skipIf(not available, 'EDI import failed')
class TestKeywordNamedQuantity(unittest.TestCase):
    """A quantity may be named for a Python keyword.

    A wing taper ratio is `lambda` in every reference this repository checks
    against, and the flat name is load-bearing -- the gpkit cross-check maps
    `\\lambda` onto it. `wing.lambda` is a syntax error, so the trailing
    underscore PEP 8 prescribes for the collision is accepted instead.
    """

    def test_a_keyword_name_is_reachable_with_a_trailing_underscore(self):
        f = Formulation()
        wing = f.group('wing')
        taper = wing.Variable('lambda', 0.25, '-', 'taper ratio')
        self.assertIs(f.wing.lambda_, taper)
        self.assertEqual(taper.name, 'wing_lambda')

    def test_ordinary_names_are_unaffected(self):
        f = Formulation()
        wing = f.group('wing')
        ar = wing.Variable('AR', 11.0, '-', 'aspect ratio')
        self.assertIs(f.wing.AR, ar)

    def test_a_typo_still_raises(self):
        """The rule is narrow: only an actual keyword loses its underscore."""
        f = Formulation()
        wing = f.group('wing')
        wing.Variable('AR', 11.0, '-', 'aspect ratio')
        with self.assertRaises(AttributeError):
            f.wing.nonexistent_


@unittest.skipIf(not available, 'EDI import failed')
class TestSlicingAndSequences(unittest.TestCase):

    def _f(self, n=4):
        f = Formulation()
        return f, f.Variable('x', 1.0, 'm', 'x', size=n)

    def test_slicing(self):
        f, x = self._f()
        self.assertEqual([str(v) for v in x[1:]], ['x[1]', 'x[2]', 'x[3]'])
        self.assertEqual([str(v) for v in x[:2]], ['x[0]', 'x[1]'])

    def test_negative_indexing(self):
        f, x = self._f()
        self.assertEqual(str(x[-1]), 'x[3]')
        self.assertEqual(str(x[-2]), 'x[2]')

    def test_a_positive_index_still_returns_the_variable_itself(self):
        f, x = self._f()
        self.assertEqual(str(x[0]), 'x[0]')
        self.assertNotIsInstance(x[0], np.ndarray)

    def test_a_monotone_sequence_is_one_expression(self):
        """The case mission and dynamics models need."""
        f, x = self._f()
        cons = x[1:] >= x[:-1]
        self.assertEqual([str(c) for c in cons],
                         ['x[0]  <=  x[1]', 'x[1]  <=  x[2]', 'x[2]  <=  x[3]'])

    def test_a_recurrence_reads_as_written(self):
        f, x = self._f()
        burn = f.Constant('burn', 0.1, 'm', 'burn', size=3)
        cons = x[1:] == x[:-1] - burn
        self.assertEqual(len(cons), 3)

    def test_it_solves(self):
        f, x = self._f()
        f.Objective(f.sum(x))
        f.ConstraintList(x[1:] >= x[:-1])
        f.ConstraintList([x[0] >= 2.0 * units.m])
        solve(f, sensitivities=False)
        vals = [pyo.value(x[i]) for i in range(4)]
        for v in vals:
            self.assertAlmostEqual(v, 2.0, places=4)


@unittest.skipIf(not available, 'EDI import failed')
class TestMatrices(unittest.TestCase):

    def _f(self):
        f = Formulation()
        M = f.Variable('M', 1.0, 'm', 'M', size=[2, 3])
        return f, M

    def test_shape(self):
        f, M = self._f()
        self.assertEqual(M.shape, (2, 3))

    def test_rows_and_columns(self):
        f, M = self._f()
        self.assertEqual([str(v) for v in M[0, :]],
                         ['M[0,0]', 'M[0,1]', 'M[0,2]'])
        self.assertEqual([str(v) for v in M[:, 1]], ['M[0,1]', 'M[1,1]'])

    def test_reductions_along_an_axis(self):
        f, M = self._f()
        down = f.sum(M, axis=0)
        across = f.sum(M, axis=1)
        self.assertEqual(len(down), 3)
        self.assertEqual(len(across), 2)
        self.assertEqual(str(down[0]), 'M[0,0] + M[1,0]')
        self.assertEqual(str(across[0]), 'M[0,0] + M[0,1] + M[0,2]')

    def test_a_full_reduction_is_an_expression_not_an_array(self):
        """numpy keeps the array type through .sum(); Pyomo cannot use that."""
        f, M = self._f()
        self.assertNotIsInstance(f.sum(M), np.ndarray)

    def test_sequences_along_a_chosen_axis(self):
        f, M = self._f()
        cons = M[:, 1:] >= M[:, :-1]
        self.assertEqual(cons.shape, (2, 2))

    def test_three_dimensions(self):
        f = Formulation()
        T = f.Variable('T', 1.0, 'm', 'T', size=[2, 3, 4])
        self.assertEqual(T.shape, (2, 3, 4))
        self.assertEqual(f.sum(T, axis=2).shape, (2, 3))
        self.assertEqual((T[:, :, 1:] >= T[:, :, :-1]).shape, (2, 3, 3))


@unittest.skipIf(not available, 'EDI import failed')
class TestBroadcasting(unittest.TestCase):
    """Shapes are never expanded silently.

    numpy would spread a length-3 vector across the rows of a 2x3 matrix just
    as readily as down its columns, and whichever the author meant, the other
    reading is a different model that solves without complaint.
    """

    def _f(self):
        f = Formulation()
        M = f.Variable('M', 1.0, 'm', 'M', size=[2, 3])
        percol = f.Constant('percol', 5.0, 'm', 'per-column cap', size=3)
        perrow = f.Constant('perrow', 9.0, 'm', 'per-row cap', size=2)
        return f, M, percol, perrow

    def test_an_ambiguous_comparison_raises(self):
        f, M, percol, perrow = self._f()
        with self.assertRaises(ShapeMismatch) as ctx:
            M <= percol
        self.assertIn('broadcast_rows', str(ctx.exception))

    def test_broadcast_rows_repeats_the_vector_as_each_row(self):
        f, M, percol, perrow = self._f()
        cons = M <= f.broadcast_rows(percol, 2)
        self.assertEqual(cons.shape, (2, 3))
        self.assertEqual(str(cons[0, 0]), 'M[0,0]  <=  percol[0]')
        self.assertEqual(str(cons[0, 1]), 'M[0,1]  <=  percol[1]')
        self.assertEqual(str(cons[1, 0]), 'M[1,0]  <=  percol[0]')

    def test_broadcast_cols_repeats_the_vector_as_each_column(self):
        f, M, percol, perrow = self._f()
        cons = M <= f.broadcast_cols(perrow, 3)
        self.assertEqual(cons.shape, (2, 3))
        self.assertEqual(str(cons[0, 0]), 'M[0,0]  <=  perrow[0]')
        self.assertEqual(str(cons[0, 1]), 'M[0,1]  <=  perrow[0]')
        self.assertEqual(str(cons[1, 0]), 'M[1,0]  <=  perrow[1]')

    def test_the_wrong_helper_is_a_shape_error_not_a_wrong_model(self):
        f, M, percol, perrow = self._f()
        with self.assertRaises(ShapeMismatch):
            M <= f.broadcast_rows(perrow, 2)      # perrow is length 2, not 3

    def test_a_matching_length_still_raises_without_a_helper(self):
        """A square matrix is where a silent broadcast would hurt most."""
        f = Formulation()
        S = f.Variable('S', 1.0, 'm', 'S', size=[3, 3])
        v = f.Constant('v', 1.0, 'm', 'v', size=3)
        with self.assertRaises(ShapeMismatch):
            S <= v


@unittest.skipIf(not available, 'EDI import failed')
class TestConstraintListAcceptsArrays(unittest.TestCase):

    def test_a_flat_array_of_constraints(self):
        f = Formulation()
        x = f.Variable('x', 1.0, 'm', 'x', size=3)
        y = f.Variable('y', 1.0, 'm', 'y', size=3)
        f.Objective(f.sum(x))
        f.ConstraintList(x >= y)
        self.assertEqual(len(list(f.component_data_objects(pyo.Constraint))), 3)

    def test_a_matrix_of_constraints_is_flattened(self):
        f = Formulation()
        M = f.Variable('M', 1.0, 'm', 'M', size=[2, 3])
        cap = f.Constant('cap', 5.0, 'm', 'cap', size=3)
        f.Objective(f.sum(M))
        f.ConstraintList(M <= f.broadcast_rows(cap, 2))
        self.assertEqual(len(list(f.component_data_objects(pyo.Constraint))), 6)

    def test_arrays_and_plain_constraints_can_be_mixed(self):
        f = Formulation()
        x = f.Variable('x', 1.0, 'm', 'x', size=3)
        f.Objective(f.sum(x))
        f.ConstraintList([x[1:] >= x[:-1], x[0] >= 1.0 * units.m])
        self.assertEqual(len(list(f.component_data_objects(pyo.Constraint))), 3)


@unittest.skipIf(not available, 'EDI import failed')
class TestGroupsExposeTheHelpers(unittest.TestCase):

    def test_a_group_forwards_the_vector_operations(self):
        f = Formulation()
        wing = f.group('wing')
        x = wing.Variable('x', 1.0, 'm', 'x', size=3)
        self.assertEqual(str(wing.sum(x)), 'wing_x[0] + wing_x[1] + wing_x[2]')
        self.assertEqual(wing.broadcast_rows(x, 2).shape, (2, 3))


if __name__ == '__main__':
    unittest.main()
