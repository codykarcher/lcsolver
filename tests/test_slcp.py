#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Tests for the Sequential Log-Convex Programming solver."""

import math

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import, numpy as np, numpy_available

slcp, slcp_available = attempt_import('lcsolver.solvers.ipopt.slcp')


def _ipopt_available():
    try:
        return pyo.SolverFactory('ipopt').available(exception_flag=False)
    except Exception:
        return False


ipopt_available = _ipopt_available()


def _xy_problem():
    """min x*y  s.t.  1/x <= 1, 2/y <= 1.  Optimum (1, 2), objective 2."""
    from lcsolver.solvers.ipopt.slcp import Constraint, Posynomial, Problem

    objective = Posynomial([(1.0, [1, 1])], 2)
    constraints = [
        Constraint(Posynomial([(1.0, [-1, 0])], 2), '<='),
        Constraint(Posynomial([(2.0, [0, -1])], 2), '<='),
    ]
    return Problem(2, objective, constraints, names=['x', 'y'])


@unittest.skipIf(not numpy_available, 'SLCP requires numpy')
@unittest.skipIf(not slcp_available, 'could not import the SLCP module')
class TestSLCPComponents(unittest.TestCase):
    """The problem-description primitives."""

    def test_posynomial_value_and_gradient(self):
        from lcsolver.solvers.ipopt.slcp import Posynomial

        p = Posynomial([(2.0, [1, 0]), (3.0, [0, 2])], 2)
        x = np.array([5.0, 4.0])
        self.assertAlmostEqual(p(x), 2 * 5 + 3 * 16)
        # d/dx = 2, d/dy = 6y = 24
        self.assertTrue(np.allclose(p.grad(x), [2.0, 24.0]))

    def test_log_gradient_matches_equation_11(self):
        """d log f(e^y)/dy_i = x_i/f * df/dx_i."""
        from lcsolver.solvers.ipopt.slcp import Posynomial

        p = Posynomial([(2.0, [1, 0]), (3.0, [0, 2])], 2)
        x = np.array([5.0, 4.0])
        expected = x * p.grad(x) / p(x)
        self.assertTrue(np.allclose(p.log_grad(x), expected))

    def test_negative_coefficient_rejected(self):
        from lcsolver.solvers.ipopt.slcp import Posynomial

        self.assertRaises(ValueError, Posynomial, [(-1.0, [1, 0])], 2)

    def test_multiterm_posynomial_equality_rejected(self):
        """A multi-term posynomial equality is not GP-compatible."""
        from lcsolver.solvers.ipopt.slcp import Constraint, Posynomial

        body = Posynomial([(1.0, [1, 0]), (1.0, [0, 1])], 2)
        self.assertRaises(ValueError, Constraint, body, '==')

    def test_posynomial_is_exact_in_logspace(self):
        from lcsolver.solvers.ipopt.slcp import Constraint, Posynomial, Signomial

        posy = Constraint(Posynomial([(1.0, [1, 1])], 2), '<=')
        self.assertTrue(posy.exact_in_logspace)

        sig = Constraint(Signomial(lambda x: (1.0, np.zeros(2)), 2), '<=')
        self.assertFalse(sig.exact_in_logspace)


@unittest.skipIf(not numpy_available, 'SLCP requires numpy')
@unittest.skipIf(not slcp_available, 'could not import the SLCP module')
@unittest.skipIf(not ipopt_available, 'SLCP sub-problems require IPOPT')
class TestSLCPSolve(unittest.TestCase):
    """End-to-end solves."""

    def test_slcp_reaches_known_optimum(self):
        r = slcp.solve(_xy_problem(), [3.0, 3.0], method='slcp')
        self.assertTrue(r.converged, msg=r.status)
        self.assertAlmostEqual(r.objective, 2.0, places=5)
        self.assertAlmostEqual(r.x[0], 1.0, places=4)
        self.assertAlmostEqual(r.x[1], 2.0, places=4)

    def test_all_three_methods_agree(self):
        problem = _xy_problem()
        objectives = []
        for method in ('slcp', 'lsqp', 'sqp'):
            r = slcp.solve(problem, [3.0, 3.0], method=method)
            self.assertTrue(r.converged, msg=f'{method}: {r.status}')
            objectives.append(r.objective)
        for value in objectives:
            self.assertAlmostEqual(value, 2.0, places=4)

    def test_max_violation_reported_on_every_solve(self):
        """A caller must always be able to tell whether x is usable."""
        for method in ('slcp', 'lsqp', 'sqp'):
            r = slcp.solve(_xy_problem(), [3.0, 3.0], method=method)
            self.assertIsNotNone(r.max_violation, msg=f'{method}: {r.status}')
            self.assertLessEqual(r.max_violation, 1e-6, msg=f'{method}: {r.status}')

    def test_stalled_infeasible_is_not_reported_as_converged(self):
        """A collapsed step at an INFEASIBLE point is a stall, not convergence.

        Regression for the original behaviour, which set converged=True on step
        magnitude alone. A signomial constraint that cannot be satisfied
        anywhere (body >= 2 > 1 for every x) drives the iterates to a standstill
        while remaining infeasible; the result must say so.
        """
        def impossible(x):
            g = np.zeros(2)
            return 2.0, g          # body == 2 > 1 always, gradient 0 -> no way out

        problem = slcp.Problem(
            2,
            slcp.Posynomial([(1.0, [1, 0]), (1.0, [0, 1])], 2),
            [slcp.Constraint(slcp.Signomial(impossible, 2), '<=')],
        )
        r = slcp.solve(problem, [1.0, 1.0], method='slcp',
                       options=slcp.Options(max_iterations=50))
        self.assertFalse(r.converged, msg=r.status)
        self.assertIsNotNone(r.max_violation)
        self.assertGreater(r.max_violation, 1e-6)

    def test_feasibility_tolerance_is_configurable(self):
        opts = slcp.Options(feasibility_tolerance=1e-3)
        self.assertEqual(opts.feasibility_tolerance, 1e-3)
        self.assertEqual(slcp.Options().feasibility_tolerance, 1e-6)

    def test_agm_condensation_is_a_tight_underestimator(self):
        """q_hat <= q everywhere, with equality at the linearization point."""
        n = 2
        pp = slcp.Posynomial([(1.0, [1, 0])], n)
        qq = slcp.Posynomial([(1.0, [0, 0]), (2.0, [0, 1]), (0.5, [1, 1])], n)
        pr = slcp.PosynomialRatio(pp, qq, n)
        xk = np.array([1.5, 0.8])
        c, a = pr.condensed_q(xk)
        qhat = lambda x: c * np.prod(np.asarray(x, dtype=float) ** a)
        self.assertAlmostEqual(qhat(xk), qq(xk), places=10)      # tight at x_k
        for x in ([2.2, 0.35], [0.4, 3.0], [1.0, 1.0], [5.0, 0.1]):
            self.assertLessEqual(qhat(x), qq(x) + 1e-12)         # under-estimator

    def test_sp_form_matches_black_box_optimum(self):
        """p/q <= 1 and the 2-q<=1 black box must find the same optimum."""
        n = 2
        obj = slcp.Posynomial([(1.0, [1, 0]), (1.0, [0, 1])], n)
        q = slcp.Posynomial([(1.0, [1, 1])], n)
        pp = slcp.Posynomial([(1.0, [0, 0])], n)

        def bb(x):
            return 2.0 - x[0] * x[1], np.array([-x[1], -x[0]])

        r_bb = slcp.solve(slcp.Problem(n, obj, [slcp.Constraint(slcp.Signomial(bb, n), '<=')]),
                          [3.0, 0.4], method='slcp')
        r_sp = slcp.solve(slcp.Problem(n, obj, [slcp.Constraint(slcp.PosynomialRatio(pp, q, n), '<=')]),
                          [3.0, 0.4], method='slcp')
        self.assertTrue(r_bb.converged and r_sp.converged)
        self.assertAlmostEqual(r_bb.objective, 2.0, places=5)
        self.assertAlmostEqual(r_sp.objective, 2.0, places=5)

    def test_sp_form_multiterm_q_converges_efficiently(self):
        """A multi-term q makes the AGM condensation a real approximation,
        iterated to tightness. SP-form constraints always enter the Reduced
        Lagrangian (they are only PARTLY exact -- q's curvature is condensed
        away and BFGS is the only thing left to supply it); measured, omitting
        them failed to converge from every start."""
        n = 3
        obj = slcp.Posynomial([(1.0, [1, 0, 0]), (1.0, [0, 1, 0]), (1.0, [0, 0, 1])], n)
        q = slcp.Posynomial([(0.6, [1, 1, 0]), (0.5, [0, 1, 1]),
                             (0.4, [1, 0, 1]), (0.3, [2, 0, 0])], n)
        pp = slcp.Posynomial([(1.0, [0, 0, 0])], n)
        prob = slcp.Problem(n, obj, [slcp.Constraint(slcp.PosynomialRatio(pp, q, n), '<=')])
        for start in ([0.5, 0.5, 0.5], [0.3, 0.9, 0.6], [1.0, 0.4, 0.4]):
            r = slcp.solve(prob, start, method='slcp',
                           options=slcp.Options(max_iterations=300))
            self.assertTrue(r.converged, msg=f'{start}: {r.status}')
            self.assertLess(r.iterations, 100)
            self.assertLessEqual(r.max_violation, 1e-6)

    def test_sp_form_equality_rejected(self):
        n = 2
        pp = slcp.Posynomial([(1.0, [0, 0])], n)
        q = slcp.Posynomial([(1.0, [1, 1])], n)
        self.assertRaises(ValueError, slcp.Constraint,
                          slcp.PosynomialRatio(pp, q, n), '==')

    def test_rejects_unknown_method(self):
        self.assertRaises(ValueError, slcp.solve, _xy_problem(), [3.0, 3.0],
                          **{'method': 'nonsense'})

    def test_rejects_nonpositive_start(self):
        """The log transform is undefined for a non-positive iterate."""
        self.assertRaises(ValueError, slcp.solve, _xy_problem(), [0.0, 1.0])

    def test_history_records_every_iterate(self):
        r = slcp.solve(_xy_problem(), [3.0, 3.0], method='slcp')
        self.assertEqual(len(r.history), r.iterations + 1)
        self.assertTrue(np.allclose(r.history[0], [3.0, 3.0]))

    def test_signomial_constraint_is_honoured(self):
        """A non-GP-compatible constraint, supplied as a callback.

        min x  s.t.  2/(x + y) <= 1 and y <= 1, so x >= 1 at the optimum.
        The ratio is not a sum of monomials, hence a Signomial.
        """
        from lcsolver.solvers.ipopt.slcp import (Constraint, Posynomial, Problem,
                                            Signomial)

        def ratio(x):
            denom = x[0] + x[1]
            g = np.array([-2.0 / denom ** 2, -2.0 / denom ** 2])
            return 2.0 / denom, g

        problem = Problem(
            2,
            Posynomial([(1.0, [1, 0])], 2),
            [Constraint(Signomial(ratio, 2), '<='),
             Constraint(Posynomial([(1.0, [0, 1])], 2), '<=')],
            names=['x', 'y'])

        r = slcp.solve(problem, [4.0, 0.5], method='slcp')
        self.assertTrue(r.converged, msg=r.status)
        self.assertGreaterEqual(r.x[0] + r.x[1], 2.0 - 1e-4)
        self.assertLessEqual(r.x[1], 1.0 + 1e-6)

    def test_reduced_lagrangian_excludes_exact_constraints(self):
        """Paper Equation 14: posynomials are omitted from the BFGS Hessian.

        With the reduced flag set, an exactly-imposed constraint must contribute
        nothing to the gradient, so the result equals the objective's own log
        gradient regardless of the multiplier.
        """
        problem = _xy_problem()
        x = np.array([2.0, 3.0])
        mults = np.array([5.0, 7.0])

        reduced = slcp._lagrangian_gradient(problem, x, mults, 'slcp', True)
        full = slcp._lagrangian_gradient(problem, x, mults, 'slcp', False)

        self.assertTrue(np.allclose(reduced, problem.objective.log_grad(x)))
        self.assertFalse(np.allclose(reduced, full))


@unittest.skipIf(not numpy_available, 'SLCP requires numpy')
@unittest.skipIf(not slcp_available, 'could not import the SLCP module')
class TestDampedBFGS(unittest.TestCase):
    """Paper Equation 13."""

    def test_preserves_positive_definiteness_on_negative_curvature(self):
        """Damping is what keeps B usable when s'z < 0."""
        B = np.eye(2)
        s = np.array([1.0, 0.0])
        z = np.array([-5.0, 0.0])          # negative curvature
        B_new = slcp._damped_bfgs(B, s, z)
        eigenvalues = np.linalg.eigvalsh(B_new)
        self.assertTrue((eigenvalues > 0).all(),
                        msg=f'B lost positive definiteness: {eigenvalues}')

    def test_matches_plain_bfgs_when_curvature_is_good(self):
        """theta = 1 when s'z >= 0.2 s'Bs, so damping is inactive."""
        B = np.eye(2)
        s = np.array([1.0, 0.0])
        z = np.array([2.0, 0.0])
        B_new = slcp._damped_bfgs(B, s, z)
        Bs = B @ s
        expected = (B - np.outer(Bs, Bs) / (s @ Bs)
                    + np.outer(z, z) / (s @ z))
        self.assertTrue(np.allclose(B_new, expected))


if __name__ == '__main__':
    unittest.main()
