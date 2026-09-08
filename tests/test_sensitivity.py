#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
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
    from lcsolver import Formulation, sensitivities, format_sensitivities

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


@unittest.skipIf(not edi_available, 'LCsolver import failed')
@unittest.skipIf(not pint_available, 'Testing units requires pint')
@unittest.skipIf(not numpy_available, 'sensitivity requires numpy')
class TestSensitivityKnownAnswers(unittest.TestCase):
    """Closed-form checks across problem classes and both solver cores."""

    def _solve_cvxopt(self, f):
        from lcsolver.solvers.solver import cvxopt_solve
        cvxopt_solve(f)

    def _solve_ipopt(self, f):
        from lcsolver.solvers.ipopt import ipopt_solve
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


@unittest.skipIf(not edi_available, 'LCsolver import failed')
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
        from lcsolver.solvers.solver import cvxopt_solve
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


@unittest.skipIf(not edi_available, 'LCsolver import failed')
@unittest.skipIf(not pint_available, 'Testing units requires pint')
@unittest.skipIf(not numpy_available, 'sensitivity requires numpy')
class TestSensitivityInterface(unittest.TestCase):
    """The public surface and its failure behavior."""

    @unittest.skipIf(not cvxopt_available, 'cvxopt is not installed')
    def test_formulation_method(self):
        f = _gp()
        from lcsolver.solvers.solver import cvxopt_solve
        cvxopt_solve(f)
        self.assertAlmostEqual(f.sensitivities()['sensitivities']['a'], 0.5,
                               places=5)

    @unittest.skipIf(not cvxopt_available, 'cvxopt is not installed')
    def test_result_keys(self):
        f = _gp()
        from lcsolver.solvers.solver import cvxopt_solve
        cvxopt_solve(f)
        r = sensitivities(f)
        for key in ('sensitivities', 'objective', 'normalized', 'method',
                    'approximate', 'stationarity_residual'):
            self.assertIn(key, r)

    @unittest.skipIf(not cvxopt_available, 'cvxopt is not installed')
    def test_format_is_printable(self):
        f = _gp()
        from lcsolver.solvers.solver import cvxopt_solve
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


class TestParameterGradient(unittest.TestCase):
    """The one-walk-per-expression gradient the envelope sum is built on.

    Its predecessor asked for one constant at a time, which cost a full walk
    per (constraint, constant) pair -- almost all of them returning zero
    because the constant was not in that constraint. These are the cases where
    the batched form could differ from the pairwise one.
    """

    def _index(self, f):
        from lcsolver.postsolve.sensitivity import _constants
        return {id(pd): n for n, pd in _constants(f).items()}

    def test_a_constant_appearing_twice_is_not_counted_twice(self):
        """The walker yields a repeated Param once per occurrence."""
        from lcsolver.postsolve.sensitivity import _param_gradient
        f = Formulation()
        x = f.Variable('x', 1.0, '')
        a = f.Constant('a', 3.0, '')
        f.Objective(x)
        g = _param_gradient(a * x + a * x, self._index(f))
        self.assertAlmostEqual(g['a'], 2.0, places=12)     # d(2ax)/da = 2x

    def test_constants_absent_from_an_expression_are_omitted(self):
        from lcsolver.postsolve.sensitivity import _param_gradient
        f = Formulation()
        x = f.Variable('x', 2.0, '')
        a = f.Constant('a', 3.0, '')
        b = f.Constant('b', 5.0, '')
        f.Objective(x)
        g = _param_gradient(a * x, self._index(f))
        self.assertIn('a', g)
        self.assertNotIn('b', g)                # contributes nothing, not zero

    def test_a_plain_number_has_no_gradient(self):
        """Bounds are often literals, and were reaching the walker as floats."""
        from lcsolver.postsolve.sensitivity import _param_gradient
        f = Formulation()
        x = f.Variable('x', 1.0, '')
        f.Constant('a', 3.0, '')
        f.Objective(x)
        self.assertEqual(_param_gradient(4.0, self._index(f)), {})
        self.assertEqual(_param_gradient(None, self._index(f)), {})

    def test_it_agrees_with_differentiating_one_at_a_time(self):
        from lcsolver.postsolve.sensitivity import _param_gradient, _d, _constants
        f = Formulation()
        x = f.Variable('x', 2.0, '')
        y = f.Variable('y', 3.0, '')
        a = f.Constant('a', 3.0, '')
        b = f.Constant('b', 5.0, '')
        f.Objective(x)
        expr = a * x ** 2 + b * y + a * b * x * y
        constants = _constants(f)
        batched = _param_gradient(expr, {id(pd): n for n, pd in constants.items()})
        for name, pd in constants.items():
            self.assertAlmostEqual(batched[name], _d(expr, pd), places=10)


@unittest.skipIf(not _ipopt_available(),
                 'the ambiguity measure is read off the duals the IPOPT route '
                 'returns; the cvxopt fallback recovers a different dual vector')
class TestDualAmbiguity(unittest.TestCase):
    """Which sensitivities the problem actually determines.

    A rank-deficient active set leaves the duals non-unique: any null-space
    vector can be added and stationarity still holds. `lstsq` returns the
    minimum-norm member of that family without saying so, so an undetermined
    sensitivity comes back looking like an ordinary number. On SPaircraft that
    produced a reported +315 -- not a credible log-log sensitivity -- from an
    active set rank deficient by 23.
    """

    def test_a_nondegenerate_problem_reports_nothing_ambiguous(self):
        from lcsolver.solvers.ipopt import ipopt_solve
        from lcsolver.postsolve.sensitivity import DUAL_AMBIGUITY_TOL

        f = _gp()
        ipopt_solve(f)
        res = sensitivities(f)
        self.assertEqual(res['ambiguous'], [])
        for r in res['ambiguity'].values():
            self.assertLessEqual(r, DUAL_AMBIGUITY_TOL)

    def test_a_duplicated_constraint_makes_its_constant_undetermined(self):
        """State the same binding constraint twice and the duals split freely.

        Either multiplier can take the other's share, so any sensitivity that
        reads them individually is an artefact of the split.
        """
        from lcsolver.solvers.ipopt import ipopt_solve

        f = Formulation()
        x = f.Variable('x', 1.0, '', 'x')
        a = f.Constant('a', 2.0, '', 'a')
        b = f.Constant('b', 2.0, '', 'b')
        f.Objective(x)
        f.Constraint(x >= a)
        f.Constraint(x >= b)            # the same constraint, said twice
        ipopt_solve(f)

        res = sensitivities(f)
        self.assertTrue(res['ambiguous'],
                        'a duplicated binding constraint leaves the duals '
                        'undetermined, and that should be reported')

    def test_the_ambiguity_measure_is_a_relative_size(self):
        from lcsolver.solvers.ipopt import ipopt_solve
        from lcsolver.postsolve.sensitivity import dual_ambiguity

        f = _gp()
        ipopt_solve(f)
        for r in dual_ambiguity(f).values():
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0 + 1e-9)

    def test_it_can_be_switched_off(self):
        from lcsolver.solvers.ipopt import ipopt_solve

        f = _gp()
        ipopt_solve(f)
        res = sensitivities(f, check_ambiguity=False)
        self.assertEqual(res['ambiguity'], {})
        self.assertEqual(res['ambiguous'], [])


class TestActiveSetIsScaleInvariant(unittest.TestCase):
    """A slack bound on a small-magnitude variable must read as slack.

    The active-set test compares a residual against the constraint's own
    natural magnitude. It used to compare against ``rtol * max(1.0, scale)``,
    whose floor turned that relative test into an absolute one for anything
    below unit scale -- invisible in aerospace, where variables are forces and
    speeds, and immediate outside it. A dimensionless damage index of order
    1e-5 sitting on a slack 1e-12 lower bound was declared active, which added
    a column to the stationarity system, which made the active set rank
    deficient, which made the duals non-unique. The failure then surfaced as
    LC-W303 correctly hiding sensitivities that had genuinely become
    undetermined -- so the visible symptom was two steps from the cause.
    """

    @staticmethod
    def _one_constraint(expr_of):
        """A Formulation carrying a single constraint, and that constraint.

        ``Formulation.Constraint`` returns the constraint's NAME, not the
        component, so reach for the component itself.
        """
        f = Formulation()
        y = f.Variable('y', 1.0e-5, '', 'a dimensionless index of order 1e-5')
        f.Objective(y)
        f.Constraint(expr_of(y))
        return f, y, list(f.get_constraints())[0]

    def test_a_slack_bound_far_below_unit_scale_is_not_active(self):
        """The unit that fails directly, with no solve in the way."""
        from lcsolver.postsolve.sensitivity import _is_active, ACTIVE_RTOL

        f, y, con = self._one_constraint(lambda v: v >= 1.0e-12)
        y.set_value(2.0e-5)

        self.assertFalse(
            _is_active(con, rtol=ACTIVE_RTOL, variables=[y]),
            'y is 2e-5 and the bound is 1e-12 -- seven orders of magnitude '
            'of slack must not read as binding')

    def test_the_no_variables_fallback_is_also_scale_invariant(self):
        """The same trap sits in the branch that has no variables to measure."""
        from lcsolver.postsolve.sensitivity import _is_active, ACTIVE_RTOL

        f, y, con = self._one_constraint(lambda v: v >= 1.0e-12)
        y.set_value(2.0e-5)

        self.assertFalse(
            _is_active(con, rtol=ACTIVE_RTOL),
            'the bound is the only scale available, and 2e-5 is nowhere near '
            '1e-12 by any measure')

    def test_a_genuinely_tight_small_scale_bound_is_still_active(self):
        """The fix must not go the other way and lose real active bounds."""
        from lcsolver.postsolve.sensitivity import _is_active, ACTIVE_RTOL

        f, y, con = self._one_constraint(lambda v: v >= 1.0e-5)
        y.set_value(1.0e-5 * (1 + 1e-9))     # converged onto its bound

        self.assertTrue(
            _is_active(con, rtol=ACTIVE_RTOL, variables=[y]),
            'a bound the solver has converged onto is active regardless of '
            'the magnitude it is expressed in')

    @unittest.skipUnless(edi_available, "lcsolver not available")
    @unittest.skipUnless(_ipopt_available(), "ipopt not available")
    def test_a_small_scale_slack_bound_does_not_poison_the_duals(self):
        """End to end: the spurious column used to cost every sensitivity.

        Without the slack bound this problem has one active constraint for one
        free dimension and ``d log x / d log a == 1`` exactly. Adding a bound
        that is seven orders of magnitude slack must not change that.
        """
        from lcsolver.solvers.ipopt import ipopt_solve

        f = Formulation()
        x = f.Variable('x', 1.0, '', 'x')
        y = f.Variable('y', 1.0e-5, '', 'a small dimensionless companion')
        a = f.Constant('a', 2.0, '', 'a')
        f.Objective(x)
        f.ConstraintList([
            x >= a,
            y == 1.0e-5 * x,
            y >= 1.0e-12,               # slack by seven orders of magnitude
            y <= 1.0e-2,
        ])
        ipopt_solve(f)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = sensitivities(f)

        self.assertEqual(
            res['ambiguous'], [],
            'the slack bound is not part of the active set, so nothing here '
            'is degenerate and every sensitivity is determined')
        self.assertAlmostEqual(res['sensitivities']['a'], 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
