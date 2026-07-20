#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Tests for the Sequential Log-Convex Programming solver."""

import math

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import, numpy as np, numpy_available

slcp, slcp_available = attempt_import('edi.solvers.ipopt.slcp')


def _ipopt_available():
    try:
        return pyo.SolverFactory('ipopt').available(exception_flag=False)
    except Exception:
        return False


ipopt_available = _ipopt_available()


def _xy_problem():
    """min x*y  s.t.  1/x <= 1, 2/y <= 1.  Optimum (1, 2), objective 2."""
    from edi.solvers.ipopt.slcp import Constraint, Posynomial, Problem

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
        from edi.solvers.ipopt.slcp import Posynomial

        p = Posynomial([(2.0, [1, 0]), (3.0, [0, 2])], 2)
        x = np.array([5.0, 4.0])
        self.assertAlmostEqual(p(x), 2 * 5 + 3 * 16)
        # d/dx = 2, d/dy = 6y = 24
        self.assertTrue(np.allclose(p.grad(x), [2.0, 24.0]))

    def test_log_gradient_matches_equation_11(self):
        """d log f(e^y)/dy_i = x_i/f * df/dx_i."""
        from edi.solvers.ipopt.slcp import Posynomial

        p = Posynomial([(2.0, [1, 0]), (3.0, [0, 2])], 2)
        x = np.array([5.0, 4.0])
        expected = x * p.grad(x) / p(x)
        self.assertTrue(np.allclose(p.log_grad(x), expected))

    def test_negative_coefficient_rejected(self):
        from edi.solvers.ipopt.slcp import Posynomial

        self.assertRaises(ValueError, Posynomial, [(-1.0, [1, 0])], 2)

    def test_multiterm_posynomial_equality_rejected(self):
        """A multi-term posynomial equality is not GP-compatible."""
        from edi.solvers.ipopt.slcp import Constraint, Posynomial

        body = Posynomial([(1.0, [1, 0]), (1.0, [0, 1])], 2)
        self.assertRaises(ValueError, Constraint, body, '==')

    def test_posynomial_is_exact_in_logspace(self):
        from edi.solvers.ipopt.slcp import Constraint, Posynomial, Signomial

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
        from edi.solvers.ipopt.slcp import (Constraint, Posynomial, Problem,
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
