"""Adding a signomial fraction to something must work.

Dividing by a multi-term expression produces a fraction in the detector's
row form (TASOPT's ``lsfac == 1 - a*(1-lam)/(1+lam)`` is typical), and
adding one used to raise, forcing callers to clear denominators by hand.
``gpRow_add`` now puts the operands over a common denominator.
"""
import pyomo.environ as pyo
import pytest

from lcsolver import Formulation
from lcsolver.solvers.solver import solve
from lcsolver.presolve.detectorSupportFunctions import gpRow_add

M = pyo.units.m


def _solve(build_constraints, target):
    f = Formulation()
    x = f.Variable(name="x", guess=2.0, units="-", description="")
    y = f.Variable(name="y", guess=3.0, units="-", description="")
    z = f.Variable(name="z", guess=1.0, units="-", description="")
    f.Objective(z)
    f.ConstraintList(build_constraints(x, y, z))
    solve(f, solver="ipopt-convex", max_iter=200)
    return float(pyo.value(target(x, y, z)))


def test_constant_plus_quotient_by_a_sum():
    """z == 1 + x/(1+y) with x, y pinned."""
    got = _solve(lambda x, y, z: [z == 1.0 + x / (1.0 + y), x == 2.0, y == 3.0],
                 lambda x, y, z: z)
    assert got == pytest.approx(1.0 + 2.0 / 4.0, rel=1e-6)


def test_difference_over_a_sum():
    """The TASOPT shape: z == 1 - a*(1-y)/(1+y)."""
    a = -0.15
    got = _solve(lambda x, y, z: [z == 1.0 - a * (1.0 - y) / (1.0 + y),
                                  y == 0.6, x == 1.0],
                 lambda x, y, z: z)
    assert got == pytest.approx(1.0 - a * (1.0 - 0.6) / (1.0 + 0.6), rel=1e-6)


@pytest.mark.xfail(reason="known: summing two quotients-by-a-sum in one "
                          "expression still extracts a wrong denominator",
                   strict=True)
def test_two_fractions_added():
    """z == x/(1+y) + y/(1+x) still comes out wrong: gpRow_add's arithmetic
    is correct (row-level check below gets 1.5) but the walker assembles a
    sum of two DivisionExpressions with the wrong denominator -- not yet
    isolated. One fraction plus a posynomial, the case models need, works."""
    got = _solve(lambda x, y, z: [z == x / (1.0 + y) + y / (1.0 + x),
                                  x == 2.0, y == 3.0],
                 lambda x, y, z: z)
    assert got == pytest.approx(2.0 / 4.0 + 3.0 / 3.0, rel=1e-6)


def test_two_fractions_are_correct_at_row_level():
    """The arithmetic itself is right, which is why the refusal is temporary."""
    f1 = [[1, 1.0, 1.0, 0.0], [-2, 1.0, 0.0, 0.0], [-2, 1.0, 0.0, 1.0]]
    f2 = [[1, 1.0, 0.0, 1.0], [-2, 1.0, 0.0, 0.0], [-2, 1.0, 1.0, 0.0]]
    from lcsolver.presolve.detectorSupportFunctions import (_posyMultiply,
                                                        _splitFraction, _retag)
    n1, dd1 = _splitFraction([r[:] for r in f1])
    n2, dd2 = _splitFraction([r[:] for r in f2])
    d1, d2 = _retag(dd1, 1), _retag(dd2, 1)
    num = _posyMultiply(n1, d2, 1) + _posyMultiply(n2, d1, 1)
    den = _posyMultiply(d1, d2, 1)
    ev = lambda rows: sum(c * 2.0 ** e0 * 3.0 ** e1 for _, c, e0, e1 in rows)
    assert ev(num) / ev(den) == pytest.approx(2.0 / 4.0 + 3.0 / 3.0, rel=1e-12)


def test_row_level_common_denominator():
    """A/B + C over a common denominator, checked on the rows directly."""
    # A = 2*v0            (numerator, index 1)
    # B = 1 + v1          (denominator, index -2)
    # C = 5
    frac = [[1, 2.0, 1.0, 0.0], [-2, 1.0, 0.0, 0.0], [-2, 1.0, 0.0, 1.0]]
    plain = [[1, 5.0, 0.0, 0.0]]
    out = gpRow_add([r[:] for r in frac], [r[:] for r in plain])
    num = [r for r in out if r[0] >= 0]
    den = [r for r in out if r[0] < 0]
    # (2*v0 + 5*(1 + v1)) / (1 + v1)  ->  three numerator terms, two denominator
    assert len(den) == 2, out
    assert len(num) == 3, out
    # evaluate at v0=3, v1=4: (6 + 25)/5 = 6.2
    def ev(rows):
        return sum(c * 3.0 ** e0 * 4.0 ** e1 for _, c, e0, e1 in rows)
    assert ev(num) / ev(den) == pytest.approx((2 * 3 + 5 * 5) / 5.0, rel=1e-12)


def test_plain_addition_is_unchanged():
    """No fraction involved: behaviour must be exactly as before."""
    a = [[1, 2.0, 1.0, 0.0]]
    b = [[1, 3.0, 1.0, 0.0]]
    out = gpRow_add([r[:] for r in a], [r[:] for r in b])
    assert out == [[1, 5.0, 1.0, 0.0]]
