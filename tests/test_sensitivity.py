#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Tests for sensitivity of the optimum to the Constants.

The problems here are chosen so the sensitivity is known in closed form, which
makes these tests statements about correctness rather than regression snapshots.
Where a closed form is not available (the aircraft GP) the analytic result is
checked against a finite-difference re-solve instead.
"""

import warnings

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.environ import units
from pyomo.common.dependencies import attempt_import
from pyomo.core.base.units_container import pint_available

cvxopt, cvxopt_available = attempt_import("cvxopt")
numpy, numpy_available = attempt_import("numpy")

try:
    from edi import Formulation, sensitivities, format_sensitivities

    edi_available = True
except Exception:
    edi_available = False


def _ipopt_available():
    try:
        return bool(pyo.SolverFactory('ipopt').available(exception_flag=False))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# models with known sensitivities
# ---------------------------------------------------------------------------
def _lp():
    """min x + y  s.t.  x >= a, y >= b.

    f* = a + b, so d log f*/d log a = a/(a+b) = 1/3 and likewise 2/3 for b.
    """
    f = Formulation()
    f.Variable(name='x', guess=5.0, units='m', description='x')
    f.Variable(name='y', guess=5.0, units='m', description='y')
    f.Constant(name='a', value=1.0, units='m', description='a')
    f.Constant(name='b', value=2.0, units='m', description='b')
    f.Objective(f.x + f.y)
    f.ConstraintList([f.x >= f.a, f.y >= f.b])
    return f


def _qp():
    """min x^2 + y^2  s.t.  x + y >= a.

    x = y = a/2 so f* = a^2/2 and d log f*/d log a = 2.
    """
    f = Formulation()
    f.Variable(name='x', guess=5.0, units='', description='x')
    f.Variable(name='y', guess=5.0, units='', description='y')
    f.Constant(name='a', value=4.0, units='', description='a')
    f.Objective(f.x ** 2 + f.y ** 2)
    f.ConstraintList([f.x + f.y >= f.a])
    return f


def _gp():
    """min x + y  s.t.  x*y >= a.

    x = y = sqrt(a) so f* = 2 sqrt(a) and d log f*/d log a = 1/2.
    """
    f = Formulation()
    f.Variable(name='x', guess=1.0, units='', description='x')
    f.Variable(name='y', guess=1.0, units='', description='y')
    f.Constant(name='a', value=2.0, units='', description='a')
    f.Objective(f.x + f.y)
    f.ConstraintList([f.x * f.y >= f.a])
    return f


def _objective_only():
    """min a*x s.t. x >= 2. The constant sits in the objective, not a constraint."""
    f = Formulation()
    f.Variable(name='x', guess=1.0, units='', description='x')
    f.Constant(name='a', value=3.0, units='', description='a')
    f.Objective(f.a * f.x)
    f.ConstraintList([f.x >= 2.0 * units.dimensionless])
    return f


@unittest.skipIf(not edi_available, 'EDI import failed')
@unittest.skipIf(not pint_available, 'Testing units requires pint')
@unittest.skipIf(not numpy_available, 'sensitivity requires numpy')
class TestSensitivityKnownAnswers(unittest.TestCase):
    """Closed-form checks across problem classes and both solver cores."""

    def _solve_cvxopt(self, f):
        from edi.solvers.solver import cvxopt_solve
        cvxopt_solve(f)

    def _solve_ipopt(self, f):
        from edi.solvers.ipopt import ipopt_solve
        ipopt_solve(f)

    # ---- cvxopt core --------------------------------------------------
    @unittest.skipIf(not cvxopt_available, 'cvxopt is not installed')
    def test_lp_cvxopt(self):
        f = _lp()
        self._solve_cvxopt(f)
        s = sensitivities(f)['sensitivities']
        self.assertAlmostEqual(s['a'], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(s['b'], 2.0 / 3.0, places=5)

    @unittest.skipIf(not cvxopt_available, 'cvxopt is not installed')
    def test_qp_cvxopt(self):
        f = _qp()
        self._solve_cvxopt(f)
        self.assertAlmostEqual(sensitivities(f)['sensitivities']['a'], 2.0, places=5)

    @unittest.skipIf(not cvxopt_available, 'cvxopt is not installed')
    def test_gp_cvxopt(self):
        f = _gp()
        self._solve_cvxopt(f)
        self.assertAlmostEqual(sensitivities(f)['sensitivities']['a'], 0.5, places=5)

    # ---- ipopt core ---------------------------------------------------
    @unittest.skipIf(not _ipopt_available(), 'the ipopt executable is not available')
    def test_lp_ipopt(self):
        f = _lp()
        self._solve_ipopt(f)
        s = sensitivities(f)
        self.assertEqual(s['method'], 'suffix')
        self.assertAlmostEqual(s['sensitivities']['a'], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(s['sensitivities']['b'], 2.0 / 3.0, places=5)

    @unittest.skipIf(not _ipopt_available(), 'the ipopt executable is not available')
    def test_qp_ipopt(self):
        f = _qp()
        self._solve_ipopt(f)
        self.assertAlmostEqual(sensitivities(f)['sensitivities']['a'], 2.0, places=5)

    @unittest.skipIf(not _ipopt_available(), 'the ipopt executable is not available')
    def test_gp_ipopt(self):
        f = _gp()
        self._solve_ipopt(f)
        self.assertAlmostEqual(sensitivities(f)['sensitivities']['a'], 0.5, places=5)

    # ---- structural cases ---------------------------------------------
    @unittest.skipIf(not _ipopt_available(), 'the ipopt executable is not available')
    def test_constant_in_objective(self):
        """A constant appearing only in the objective still gets a sensitivity.

        f* = 2a, so the elasticity is exactly 1. This is the term the envelope
        formula picks up from df/dtheta rather than from any dual.
        """
        f = _objective_only()
        self._solve_ipopt(f)
        self.assertAlmostEqual(sensitivities(f)['sensitivities']['a'], 1.0, places=5)

    @unittest.skipIf(not _ipopt_available(), 'the ipopt executable is not available')
    def test_unnormalized_is_raw_derivative(self):
        """normalized=False must give df*/dc, not the elasticity."""
        f = _objective_only()
        self._solve_ipopt(f)
        raw = sensitivities(f, normalized=False)
        self.assertFalse(raw['normalized'])
        # f* = 2a  =>  df*/da = 2
        self.assertAlmostEqual(raw['sensitivities']['a'], 2.0, places=5)

    @unittest.skipIf(not cvxopt_available, 'cvxopt is not installed')
    def test_both_cores_agree(self):
        """The two dual routes are independent and must give the same answer."""
        if not _ipopt_available():
            self.skipTest('the ipopt executable is not available')
        f1 = _gp(); self._solve_cvxopt(f1)
        f2 = _gp(); self._solve_ipopt(f2)
        s1 = sensitivities(f1)
        s2 = sensitivities(f2)
        self.assertEqual(s1['method'], 'kkt')
        self.assertEqual(s2['method'], 'suffix')
        self.assertAlmostEqual(s1['sensitivities']['a'],
                               s2['sensitivities']['a'], places=5)


@unittest.skipIf(not edi_available, 'EDI import failed')
@unittest.skipIf(not pint_available, 'Testing units requires pint')
@unittest.skipIf(not numpy_available, 'sensitivity requires numpy')
class TestSensitivityAgainstFiniteDifference(unittest.TestCase):
    """A realistic GP, checked against perturb-and-re-solve.

    This is the test that would catch an error in the dual bookkeeping: the
    finite difference knows nothing about duals, the envelope theorem, or the
    active set.
    """

    @unittest.skipIf(not cvxopt_available, 'cvxopt is not installed')
    def test_gp_matches_finite_difference(self):
        from edi.solvers.solver import cvxopt_solve
        import numpy as np

        def build():
            f = Formulation()
            f.Variable(name='x', guess=1.0, units='', description='x')
            f.Variable(name='y', guess=1.0, units='', description='y')
            f.Constant(name='a', value=2.0, units='', description='a')
            f.Constant(name='b', value=3.0, units='', description='b')
            f.Objective(f.x + f.y)
            f.ConstraintList([f.x * f.y >= f.a, f.x / f.y >= 1.0 / f.b])
            return f

        f = build()
        cvxopt_solve(f)
        analytic = sensitivities(f)['sensitivities']

        eps = 1e-3
        for name in ('a', 'b'):
            vals = []
            for scale in (1 + eps, 1 - eps):
                g = build()
                p = g.find_component(name)
                p.value = pyo.value(p) * scale
                cvxopt_solve(g)
                vals.append(pyo.value(
                    next(g.component_data_objects(pyo.Objective, active=True))))
            fd = ((np.log(vals[0]) - np.log(vals[1]))
                  / (np.log(1 + eps) - np.log(1 - eps)))
            self.assertAlmostEqual(analytic[name], fd, places=4,
                                   msg=f"constant {name}: analytic {analytic[name]} "
                                       f"vs finite difference {fd}")


@unittest.skipIf(not edi_available, 'EDI import failed')
@unittest.skipIf(not pint_available, 'Testing units requires pint')
@unittest.skipIf(not numpy_available, 'sensitivity requires numpy')
class TestSensitivityInterface(unittest.TestCase):
    """The public surface and its failure behaviour."""

    @unittest.skipIf(not cvxopt_available, 'cvxopt is not installed')
    def test_formulation_method(self):
        f = _gp()
        from edi.solvers.solver import cvxopt_solve
        cvxopt_solve(f)
        self.assertAlmostEqual(f.sensitivities()['sensitivities']['a'], 0.5,
                               places=5)

    @unittest.skipIf(not cvxopt_available, 'cvxopt is not installed')
    def test_result_keys(self):
        f = _gp()
        from edi.solvers.solver import cvxopt_solve
        cvxopt_solve(f)
        r = sensitivities(f)
        for key in ('sensitivities', 'objective', 'normalized', 'method',
                    'approximate', 'stationarity_residual'):
            self.assertIn(key, r)

    @unittest.skipIf(not cvxopt_available, 'cvxopt is not installed')
    def test_format_is_printable(self):
        f = _gp()
        from edi.solvers.solver import cvxopt_solve
        cvxopt_solve(f)
        text = format_sensitivities(sensitivities(f))
        self.assertIn('a', text)
        self.assertIn('Sensitivities', text)

    def test_unsolved_model_warns_rather_than_lying(self):
        """An unsolved model has no binding constraints; that must not pass silently."""
        f = _gp()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sensitivities(f)
            self.assertTrue(any(issubclass(x.category, RuntimeWarning) for x in w),
                            "expected a RuntimeWarning for a model with no active "
                            "constraints")

    def test_suffix_method_without_duals_raises(self):
        f = _gp()
        with self.assertRaises(RuntimeError):
            sensitivities(f, method='suffix')

    def test_bad_method_raises(self):
        f = _gp()
        with self.assertRaises(ValueError):
            sensitivities(f, method='nonsense')


if __name__ == '__main__':
    unittest.main()
