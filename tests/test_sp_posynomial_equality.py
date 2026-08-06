"""A posynomial equality's reverse direction must not invert term by term.

Regression test for a bug in ``solve_SP``: an equality whose numerator had
been reduced to a *sum* of monomials was routed through the monomial-equality
branch, which forms the reverse direction by inverting each row separately --
coefficient reciprocated, exponents negated. That is only valid for a single
monomial, because ``sum(1/m_i) != 1/sum(m_i)``.

The effect was silent and scale-dependent. ``b == a - k*c`` becomes
``(b + k*c)/a == 1`` after the negative term is moved across, and the reverse
direction should be ``a/(b + k*c) <= 1``. The buggy form produced
``a/b + a/(k*c) <= 1`` instead, which agrees only when one term dominates and
is wrong by orders of magnitude otherwise.

Found in the SPaircraft rebuild, where the fuselage wingbox station
``x_b == x_wing - c_0 r_w/2`` came out as ``x_wing/x_b + 4 x_wing/c_0 <= 1``,
evaluating to 13.8 instead of 1 at a point known to satisfy the model.
"""
import pyomo.environ as pyo
import pytest

from lcsolver import Formulation
from lcsolver.solvers.solver import solve

M = pyo.units.m


def _solve_difference(k: float, a_val: float, c_val: float) -> float:
    """min b subject to b == a - k*c, with a and c pinned.

    Only the *reverse* direction of the equality stops the objective driving
    b to zero, so this pins down exactly the branch under test.
    """
    f = Formulation()
    a = f.Variable(name="a", guess=a_val, units="m", description="")
    c = f.Variable(name="c", guess=c_val, units="m", description="")
    b = f.Variable(name="b", guess=max(a_val - k * c_val, 1e-3), units="m",
                   description="")
    f.Objective(b)
    f.ConstraintList([b == a - k * c,
                      a == a_val * M,
                      c == c_val * M])
    solve(f, solver="ipopt-convex")
    return float(pyo.value(b))


@pytest.mark.parametrize("k, a_val, c_val", [
    (0.25, 20.92, 6.594),      # the SPaircraft case: terms of similar size
    (0.5, 100.0, 1.0),         # subtracted term much smaller
    (0.5, 100.0, 190.0),       # subtracted term dominant
    (1e-3, 50.0, 1000.0),      # tiny coefficient, large variable
])
def test_reverse_direction_uses_the_sum_not_the_terms(k, a_val, c_val):
    expected = a_val - k * c_val
    assert expected > 0, "test case must stay positive"
    got = _solve_difference(k, a_val, c_val)
    assert got == pytest.approx(expected, rel=1e-5), (got, expected)


def test_equivalent_spellings_agree():
    """b == a - k*c and b + k*c == a are the same constraint."""
    f1 = Formulation()
    a1 = f1.Variable(name="a", guess=100.0, units="m", description="")
    c1 = f1.Variable(name="c", guess=190.0, units="m", description="")
    b1 = f1.Variable(name="b", guess=5.0, units="m", description="")
    f1.Objective(b1)
    f1.ConstraintList([b1 == a1 - 0.5 * c1, a1 == 100.0 * M, c1 == 190.0 * M])
    solve(f1, solver="ipopt-convex")

    f2 = Formulation()
    a2 = f2.Variable(name="a", guess=100.0, units="m", description="")
    c2 = f2.Variable(name="c", guess=190.0, units="m", description="")
    b2 = f2.Variable(name="b", guess=5.0, units="m", description="")
    f2.Objective(b2)
    f2.ConstraintList([b2 + 0.5 * c2 == a2, a2 == 100.0 * M, c2 == 190.0 * M])
    solve(f2, solver="ipopt-convex")

    assert float(pyo.value(b1)) == pytest.approx(float(pyo.value(b2)), rel=1e-6)


def test_monomial_equality_still_uses_the_exact_reciprocal():
    """A single-monomial equality keeps the exact term-by-term inversion."""
    f = Formulation()
    x = f.Variable(name="x", guess=2.0, units="m", description="")
    y = f.Variable(name="y", guess=3.0, units="m", description="")
    f.Objective(y)
    # y == 6 m^2 / x is a monomial equality; with x pinned, y is determined.
    f.ConstraintList([y * x == 6.0 * M ** 2, x == 2.0 * M])
    solve(f, solver="ipopt-convex")
    assert float(pyo.value(y)) == pytest.approx(3.0, rel=1e-6)
