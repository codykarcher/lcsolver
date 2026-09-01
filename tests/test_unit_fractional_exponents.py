"""Fractional unit exponents must not fail the checker on float roundoff.

A Constant in [N/W^0.803] times [W]^0.803 is newtons, but pint compares unit
exponents as exact floats and the seconds exponent comes out
-1.9999999999999998 rather than -2.0.  The checker then raised a mismatch
whose report -- pint's formatter rounds for display -- read `[N] =/= [N]`.
The walker now compares dimensions with a tolerance on the exponents.
"""
import unittest

try:
    import pyomo.environ as pyo
    from pyomo.environ import units

    from lcsolver import Formulation
    from lcsolver.presolve.unitCorrector import UnitMismatch, unit_corrector
    from lcsolver.presolve.unitWalker import _units_match, _is_dimensionless
    available = True
except Exception:                                    # pragma: no cover
    available = False


@unittest.skipIf(not available, 'LCsolver import failed')
class TestFractionalExponents(unittest.TestCase):

    def test_engine_weight_curve_fit_balances(self):
        """The case this was found in: W_eng >= k_ew * P_max**0.803."""
        f = Formulation()
        w_eng = f.Variable('W_eng', 5000.0, 'N', 'engine weight')
        p_max = f.Variable('P_max', 1e5, 'W', 'max engine power')
        k_ew = f.Constant('k_ew', 0.0372, 'N/W^(0.803)',
                          'constant for engine weight')
        f.Objective(w_eng)
        f.Constraint(w_eng >= k_ew * p_max**0.803)
        unit_corrector(f)                         # must not raise

    def test_fractional_exponents_in_a_sum(self):
        """The same roundoff inside a sum node rather than across sides."""
        f = Formulation()
        w = f.Variable('W', 5000.0, 'N', 'weight')
        p = f.Variable('P', 1e5, 'W', 'power')
        k = f.Constant('k', 0.0372, 'N/W^(0.803)', 'curve-fit constant')
        w0 = f.Constant('W_0', 100.0, 'N', 'fixed weight')
        f.Objective(w)
        f.Constraint(w >= w0 + k * p**0.803)
        unit_corrector(f)                         # must not raise

    def test_a_real_mismatch_still_raises(self):
        """The tolerance must not swallow an actual wrong dimension."""
        f = Formulation()
        w = f.Variable('W', 5000.0, 'N', 'weight')
        p = f.Variable('P', 1e5, 'W', 'power')
        k = f.Constant('k', 0.0372, 'N/W^(0.803)', 'curve-fit constant')
        f.Objective(w)
        f.Constraint(w >= k * p**0.804)           # wrong exponent
        with self.assertRaises(UnitMismatch):
            unit_corrector(f)

    def test_units_match_tolerates_roundoff(self):
        # Built the way the product handler builds it: str-concat and reparse,
        # which is the path that leaves second**-1.9999999999999998 behind.
        reg = units.pint_registry
        u1 = (1.0 * reg('N/W**0.803')).to_base_units().units
        u2 = (1.0 * reg('W')).to_base_units().units ** 0.803
        a = reg(str(u1) + '*' + str(u2)).units
        n = (1.0 * reg('N')).to_base_units().units
        self.assertNotEqual(a, n)                 # pint itself says no
        self.assertTrue(_units_match(a, n))       # the walker says yes

    def test_units_match_rejects_a_real_difference(self):
        reg = units.pint_registry
        self.assertFalse(_units_match(reg('m').units, reg('s').units))
        self.assertFalse(_units_match(reg('m').units, reg('m**1.001').units))

    def test_dimensionless_with_residue(self):
        reg = units.pint_registry
        u = reg('kg**0.803').units * reg('kg').units ** -0.803
        self.assertTrue(_is_dimensionless(u))
        self.assertFalse(_is_dimensionless(reg('kg**0.001').units))


if __name__ == '__main__':
    unittest.main()
