#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""User names for constraints.

The Pyomo component keeps its generated constraint_N name; the user name is a
map on the side, so nothing downstream changes and a name is a lookup.
"""
import unittest
import warnings

import pyomo.environ as pyo

from lcsolver import Formulation, BlackBoxFunctionModel
from lcsolver.presolve import unit_corrector


class Square(BlackBoxFunctionModel):
    def __init__(self):
        super().__init__()
        self.description = 'y = x**2'
        self.inputs.append(name='x', units='-', description='')
        self.outputs.append(name='y', units='-', description='')
        self.availableDerivative = 1

    def BlackBox(self, x):
        x = self.sanitizeInputs(x, strip_units=True)
        return self.packOutputs(x**2, [2 * x])


def build():
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='-', size=2, bounds=[0.1, 10.0], description='')
    y = f.Variable(name='y', guess=1.0, units='-', description='')
    f.Objective(y)
    return f, x, y


class TestNamedConstraints(unittest.TestCase):

    def test_constraint_takes_a_name_and_keeps_its_number(self):
        f, x, y = build()
        generated = f.Constraint(y >= x[0], assignedName='floor')
        self.assertEqual(generated, 'constraint_1')
        self.assertEqual(f.constraint_names(), {'floor': 'constraint_1'})
        self.assertIs(f.constraint('floor'), f.constraint_1)

    def test_unnamed_rows_keep_the_numbering(self):
        f, x, y = build()
        f.Constraint(y >= x[0], assignedName='floor')
        f.Constraint(y >= x[1])
        f.Constraint(y <= 5.0, assignedName='cap')
        self.assertEqual(f._allConstraint_keys, ['constraint_1', 'constraint_2', 'constraint_3'])
        self.assertEqual(f.constraint_names(), {'floor': 'constraint_1', 'cap': 'constraint_3'})

    def test_constraint_dict_dispatches_like_constraint_list(self):
        f, x, y = build()
        f.ConstraintDict({
            'floor': y >= x[0],                        # scalar row
            'cap':   x <= 5.0,                         # vector row -> one per element
            'box':   [[y], '==', [x[0]], Square()],    # black box, positional form
            'box2':  dict(outputs=[y], operators=['=='], inputs=[x[1]], black_box=Square()),
        })
        self.assertEqual(f.constraint_names(), {
            'floor': 'constraint_1',
            'cap':   ['constraint_2', 'constraint_3'],
            'box':   'constraint_4',
            'box2':  'constraint_5',
        })
        self.assertEqual([c.name for c in f.constraint('cap')], ['constraint_2', 'constraint_3'])
        self.assertEqual(f._runtimeConstraint_keys, ['constraint_4', 'constraint_5'])

    def test_runtime_constraint_takes_a_name_after_constants(self):
        """assignedName is the sixth positional slot, after constants, so a
        five-element positional black-box spec still means constants."""
        f, x, y = build()
        f.RuntimeConstraint([y], ['=='], [x[0]], Square(), None, 'box')
        f.RuntimeConstraint([y], ['=='], [x[1]], Square(), assignedName='box2')
        self.assertEqual(f.constraint_names(), {'box': 'constraint_1', 'box2': 'constraint_2'})

    def test_duplicate_name_raises_on_every_path(self):
        f, x, y = build()
        f.Constraint(y >= x[0], assignedName='row')
        with self.assertRaisesRegex(ValueError, "'row' is already used"):
            f.Constraint(y >= x[1], assignedName='row')
        with self.assertRaisesRegex(ValueError, "'row' is already used"):
            f.ConstraintDict({'row': x <= 5.0})
        with self.assertRaisesRegex(ValueError, "'row' is already used"):
            f.RuntimeConstraint([y], ['=='], [x[0]], Square(), assignedName='row')

    def test_unknown_name_names_the_known_ones(self):
        f, x, y = build()
        f.Constraint(y >= x[0], assignedName='floor')
        with self.assertRaisesRegex(KeyError, "no constraint named 'flor'.*'floor'"):
            f.constraint('flor')

    def test_holographic_paths_carry_the_name_and_the_flag(self):
        f, x, y = build()
        f.HolographicConstraint(y <= 100.0, assignedName='ceiling')
        f.HolographicConstraintDict({'edge': x[1] <= 8.0})
        self.assertEqual(f.constraint_names(), {'ceiling': 'constraint_1', 'edge': 'constraint_2'})
        self.assertEqual(f._holographic, {'constraint_1', 'constraint_2'})

    def test_group_prefixes_the_name_on_every_path(self):
        f, x, y = build()
        wing = f.group('wing')
        wing.Constraint(y >= x[0], assignedName='floor')
        wing.ConstraintDict({'cap': x <= 5.0})
        wing.HolographicConstraint(y <= 100.0, assignedName='ceiling')
        wing.HolographicConstraintDict({'edge': x[1] <= 8.0})
        wing.RuntimeConstraint([y], ['=='], [x[1]], Square(), assignedName='box')
        self.assertEqual(sorted(f.constraint_names()),
                         ['wing_box', 'wing_cap', 'wing_ceiling', 'wing_edge', 'wing_floor'])
        self.assertEqual(f._holographic, {'constraint_4', 'constraint_5'})

    def test_names_survive_the_unit_corrector(self):
        f, x, y = build()
        f.ConstraintDict({'floor': y >= x[0], 'cap': x <= 5.0})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            corrected = unit_corrector(f)
        self.assertEqual(corrected.constraint_names(), f.constraint_names())
        self.assertEqual(corrected.constraint('floor').name, 'constraint_1')
        self.assertIsNot(corrected.constraint('floor'), f.constraint('floor'))

    def test_a_named_row_evaluates_like_any_other(self):
        f, x, y = build()
        f.ConstraintDict({'floor': y >= 2.0 * x[0]})
        x[0].value, y.value = 3.0, 6.0
        con = f.constraint('floor')
        self.assertAlmostEqual(pyo.value(con.body) - pyo.value(con.upper), 0.0)


if __name__ == '__main__':
    unittest.main()
