"""The unit checker should say what is wrong and what would fix it."""
import unittest

try:
    import pyomo.environ as pyo
    from pyomo.environ import units

    from edi import Formulation
    from edi.units.unitCorrector import UnitMismatch, unit_corrector
    available = True
except Exception:                                    # pragma: no cover
    available = False


@unittest.skipIf(not available, 'EDI import failed')
class TestUnitMessages(unittest.TestCase):

    def test_a_mismatch_names_both_sides_and_the_correction(self):
        f = Formulation()
        s = f.Variable('S', 100.0, 'm^2', 'wing area')
        ar = f.Variable('AR', 11.0, '-', 'aspect ratio')
        f.Objective(s)
        f.Constraint(50.0 <= s * ar)

        with self.assertRaises(UnitMismatch) as ctx:
            unit_corrector(f)
        msg = str(ctx.exception)

        self.assertIn('constraint', msg)          # which constraint
        self.assertIn('S*AR', msg)                # what it says
        self.assertIn('dimensionless', msg)       # left units
        self.assertIn('m**2', msg)                # right units
        self.assertIn('multiply the left side', msg)   # what to do about it

    def test_unrelated_units_report_the_ratio(self):
        f = Formulation()
        length = f.Variable('L', 1.0, 'm', 'length')
        time = f.Variable('T', 1.0, 's', 'time')
        f.Objective(length)
        f.Constraint(length >= time)

        with self.assertRaises(UnitMismatch) as ctx:
            unit_corrector(f)
        msg = str(ctx.exception)
        self.assertIn('m/s', msg)

    def test_an_objective_mismatch_is_reported_as_the_objective(self):
        f = Formulation()
        length = f.Variable('L', 1.0, 'm', 'length')
        area = f.Variable('A', 1.0, 'm^2', 'area')
        f.Objective(length + area)
        f.Constraint(length >= 1.0 * units.m)

        with self.assertRaises(UnitMismatch) as ctx:
            unit_corrector(f)
        self.assertIn('objective', str(ctx.exception))

    def test_equivalent_units_are_not_a_mismatch(self):
        """N against kg*m/s**2 is the same dimension written two ways."""
        f = Formulation()
        force = f.Variable('F', 1.0, 'N', 'force')
        mass = f.Variable('m', 1.0, 'kg', 'mass')
        accel = f.Variable('a', 1.0, 'm/s^2', 'acceleration')
        f.Objective(force)
        f.Constraint(force >= mass * accel)
        unit_corrector(f)                         # must not raise

    def test_the_underlying_error_is_kept_but_trimmed(self):
        """Useful for debugging the walker; not hundreds of pointer addresses."""
        f = Formulation()
        length = f.Variable('L', 1.0, 'm', 'length')
        time = f.Variable('T', 1.0, 's', 'time')
        f.Objective(length)
        f.Constraint(length >= time)

        with self.assertRaises(UnitMismatch) as ctx:
            unit_corrector(f)
        msg = str(ctx.exception)
        self.assertIn('underlying:', msg)
        self.assertNotIn('0x', msg)               # no object addresses
        for line in msg.splitlines():
            self.assertLess(len(line), 180)


    def test_every_mismatch_is_reported_not_just_the_first(self):
        """One round trip per bad constraint is a bad way to fix a model."""
        f = Formulation()
        area = f.Variable('S', 100.0, 'm^2', 'wing area')
        time = f.Variable('T', 1.0, 's', 'time')
        f.Objective(area)
        f.Constraint(50.0 <= area)          # bad
        f.Constraint(area >= time)          # bad
        f.Constraint(area >= 1.0 * units.m ** 2)   # fine
        f.Constraint(area >= time * time)   # bad

        with self.assertRaises(UnitMismatch) as ctx:
            unit_corrector(f)
        msg = str(ctx.exception)

        self.assertIn('3 unit errors', msg)
        for name in ('constraint_1', 'constraint_2', 'constraint_4'):
            self.assertIn(name, msg)
        self.assertNotIn('constraint_3', msg)      # the good one

    def test_a_lone_mismatch_is_not_dressed_up_as_a_list(self):
        f = Formulation()
        area = f.Variable('S', 100.0, 'm^2', 'wing area')
        f.Objective(area)
        f.Constraint(50.0 <= area)

        with self.assertRaises(UnitMismatch) as ctx:
            unit_corrector(f)
        msg = str(ctx.exception)
        self.assertTrue(msg.startswith('Error in units for'))
        self.assertNotIn('unit errors:', msg)


if __name__ == '__main__':
    unittest.main()
