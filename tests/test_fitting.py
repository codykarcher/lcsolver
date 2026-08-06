#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Fitting black-box data with functions a geometric program can hold exactly.

The property under test is not accuracy but *sidedness*. A fit used as
`w >= fit(u)` that dips below the truth lets the optimizer walk into the gap
and return a point that is infeasible for the real problem, with a valid
certificate for the wrong function.
"""

import numpy as np
import pyomo.common.unittest as unittest

from lcsolver.fitting import (evaluate_fit, fit_constraints, fit_max_affine,
                         fit_report)


def _convex_data(n=60):
    """y = x^2 + 1/x -- log-convex, so a max-affine fit can track it."""
    x = np.geomspace(0.3, 3.0, n)
    return x, x ** 2 + 1.0 / x


class TestMaxAffineFit(unittest.TestCase):

    def test_a_monomial_is_recovered_exactly(self):
        """K=1 on monomial data is a straight line in log space."""
        x = np.geomspace(1.0, 10.0, 20)
        y = 3.0 * x ** 1.7
        fit = fit_max_affine(x, y, K=1)
        self.assertAlmostEqual(fit['c'][0], 3.0, places=6)
        self.assertAlmostEqual(fit['e'][0][0], 1.7, places=6)
        self.assertLess(fit['rms_err'], 1e-10)

    def test_more_pieces_fit_better(self):
        x, y = _convex_data()
        errs = [fit_max_affine(x, y, K=k)['rms_err'] for k in (1, 2, 4)]
        self.assertLess(errs[1], errs[0])
        self.assertLess(errs[2], errs[1])

    def test_multidimensional(self):
        rng = np.random.default_rng(0)
        X = np.exp(rng.uniform(-1, 1, size=(80, 3)))
        y = X[:, 0] ** 1.2 * X[:, 1] ** -0.4 + X[:, 2] ** 2
        fit = fit_max_affine(X, y, K=3)
        self.assertEqual(fit['d'], 3)
        self.assertLess(fit['rms_err'], 0.2)

    def test_evaluate_round_trips(self):
        x, y = _convex_data()
        fit = fit_max_affine(x, y, K=3)
        pred = evaluate_fit(fit, x)
        self.assertEqual(pred.shape, y.shape)
        self.assertLess(np.abs(np.log(pred) - np.log(y)).max(),
                        fit['max_err'] + 1e-9)


class TestConservativeFit(unittest.TestCase):
    """The property the whole module exists for."""

    def _residuals(self, fit, x, y):
        return np.log(evaluate_fit(fit, x)) - np.log(y)

    def test_least_squares_lands_on_both_sides(self):
        x, y = _convex_data()
        r = self._residuals(fit_max_affine(x, y, K=3), x, y)
        self.assertGreater((r < 0).sum(), 0)
        self.assertGreater((r > 0).sum(), 0)

    def test_an_upper_fit_is_never_below_the_data(self):
        x, y = _convex_data()
        for mode in ('upper', 'shift'):
            fit = fit_max_affine(x, y, K=3, conservative=mode)
            r = self._residuals(fit, x, y)
            self.assertGreaterEqual(r.min(), -1e-9, f'mode={mode}')
            self.assertEqual(fit['violations'], 0, f'mode={mode}')

    def test_a_lower_fit_is_never_above_the_data(self):
        x, y = _convex_data()
        for mode in ('lower', 'shift-lower'):
            fit = fit_max_affine(x, y, K=3, conservative=mode)
            r = self._residuals(fit, x, y)
            self.assertLessEqual(r.max(), 1e-9, f'mode={mode}')

    def test_conservatism_costs_bias(self):
        """It is a trade, and the reported bias is what is paid."""
        x, y = _convex_data()
        self.assertGreater(fit_max_affine(x, y, K=3,
                                          conservative='shift')['bias'], 0)
        self.assertAlmostEqual(fit_max_affine(x, y, K=3)['bias'], 0.0,
                               places=6)

    def test_neither_conservative_mode_dominates(self):
        """Which is tighter depends on the data, so both are offered.

        On log-convex data like this, constraining each piece as it is fitted
        is tighter -- the pieces sit just above with little slack. On the
        three-dimensional Hoburg drag black box the same choice is far worse
        (50% in the optimum against 2.5%), because a piece forced above its own
        points pokes above the data everywhere else and the max takes the worst
        of them. Both are conservative; the cost is what differs, and
        `fit_report` prints it.
        """
        x, y = _convex_data()
        shift = fit_max_affine(x, y, K=4, conservative='shift')
        piece = fit_max_affine(x, y, K=4, conservative='upper')
        for fit in (shift, piece):
            r = self._residuals(fit, x, y)
            self.assertGreaterEqual(r.min(), -1e-9)
            self.assertGreater(fit['bias'], 0.0)

    def test_multidimensional_conservatism(self):
        rng = np.random.default_rng(1)
        X = np.exp(rng.uniform(-1, 1, size=(120, 2)))
        y = X[:, 0] ** 1.5 + X[:, 1] ** -0.8
        fit = fit_max_affine(X, y, K=4, conservative='shift')
        self.assertGreaterEqual(self._residuals(fit, X, y).min(), -1e-9)


class TestFitConstraints(unittest.TestCase):

    def test_a_conservative_fit_needs_no_extra_margin(self):
        """The mfac heuristic exists to cover a two-sided fit."""
        x, y = _convex_data()
        loose = fit_max_affine(x, y, K=2)
        tight = fit_max_affine(x, y, K=2, conservative='shift')

        from lcsolver import Formulation
        f = Formulation()
        w = f.Variable('w', 1.0, '-', 'dependent')
        u = f.Variable('u', 1.0, '-', 'independent')
        self.assertEqual(len(fit_constraints(loose, w, [u])), loose['K'])
        self.assertEqual(len(fit_constraints(tight, w, [u])), tight['K'])

    def test_the_report_mentions_the_trade(self):
        x, y = _convex_data()
        text = fit_report(fit_max_affine(x, y, K=2, conservative='shift'), x, y)
        self.assertIn('conservative', text)
        self.assertIn('bias', text)


class TestInputChecking(unittest.TestCase):

    def test_non_positive_data_is_refused(self):
        with self.assertRaises(ValueError):
            fit_max_affine(np.array([1.0, -1.0]), np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            fit_max_affine(np.array([1.0, 2.0]), np.array([1.0, 0.0]))

    def test_mismatched_lengths_are_refused(self):
        with self.assertRaises(ValueError):
            fit_max_affine(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    def test_an_unknown_mode_is_refused(self):
        x, y = _convex_data()
        with self.assertRaises(ValueError):
            fit_max_affine(x, y, conservative='sideways')


if __name__ == '__main__':
    unittest.main()
