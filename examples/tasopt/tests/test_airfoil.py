"""Airfoil database (airtable.f, airfun.f, spline.f) vs the compiled Fortran.

Reads the shipped ``air/C.air`` transonic table -- 39 Mach x 7 cl x 7 tau,
three functions, Re = 2e7 -- checks a sample of the spline derivative arrays
``airtable`` builds, then evaluates ``airfun`` over an 80-point grid.

The database path is absolute in ``fortran_ref/drv_airfoil.f``; if the TASOPT
tree moves, regenerate with::

    gfortran -fdefault-real-8 -O0 -o drv_airfoil \\
        drv_airfoil.f airtable.f airfun.f spline.f
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.aero.airfoil import (CL_PENALTY, TAU_PENALTY, airfun, airtable,
                                    spline, trisol)

DATA = Path(__file__).parent / "data"
AIR = Path("/Users/codykarcher/Desktop/Tasopt2.16/air/C.air")
RTOL = 1e-13

pytestmark = pytest.mark.skipif(not AIR.exists(),
                                reason="airfoil database not present")


@pytest.fixture(scope="module")
def table():
    return airtable(AIR)


# The seven derivative entries the driver prints, in 0-based indices.
DERIVS = {
    "A_M_1": ("A_M", (4, 2, 3, 0)), "A_cl_1": ("A_cl", (4, 2, 3, 0)),
    "A_tau_1": ("A_tau", (4, 2, 3, 1)), "A_M_cl_1": ("A_M_cl", (8, 3, 2, 1)),
    "A_M_tau_1": ("A_M_tau", (8, 3, 2, 2)),
    "A_cl_tau_1": ("A_cl_tau", (8, 3, 2, 0)),
    "A_Mclt_1": ("A_M_cl_tau", (11, 4, 4, 1)),
}


def test_matches_fortran(table):
    n = 0
    with (DATA / "airfoil_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0] == "name":
                continue
            name = row[0].strip()
            if name == "ARe":
                assert table.ARe == pytest.approx(float(row[1]), rel=RTOL)
                n += 1
            elif name in DERIVS:
                attr, idx = DERIVS[name]
                assert getattr(table, attr)[idx] == pytest.approx(
                    float(row[1]), rel=RTOL), name
                n += 1
            else:
                i, j, k = int(name[1:3]), int(name[3:5]), int(name[5:7])
                g = airfun(0.35 + 0.13 * j, 0.085 + 0.018 * k,
                           0.30 + 0.13 * i, table)
                for a, b in zip((g.cdf, g.cdp, g.cdw, g.cm),
                                map(float, row[1:5])):
                    assert a == pytest.approx(b, rel=RTOL, abs=1e-300), name
                    n += 1
    assert n == 328


def test_grid_shape(table):
    assert (len(table.AMa), len(table.Acl), len(table.Atau)) == (39, 7, 7)
    assert table.nAfun == 3
    assert table.ARe == 2.0e7


def test_interpolation_reproduces_the_knots(table):
    """At a grid point the tri-cubic must return the tabulated value exactly."""
    for (i, j, k) in ((10, 2, 3), (20, 4, 5), (5, 1, 1)):
        g = airfun(table.Acl[j], table.Atau[k], table.AMa[i], table)
        assert g.cdf == pytest.approx(table.A[i, j, k, 0], rel=1e-12)
        assert g.cdp == pytest.approx(table.A[i, j, k, 1], rel=1e-12)
        assert g.cm == pytest.approx(table.A[i, j, k, 2], rel=1e-12)


def test_wave_drag_is_always_zero(table):
    """cdw is hard-wired to 0 in airfun.f -- it is not read from the table."""
    for M in (0.2, 0.6, 0.78, 0.85):
        assert airfun(0.55, 0.12, M, table).cdw == 0.0


def test_drag_rises_through_the_transonic_range(table):
    """The database itself carries the drag rise, even though cdw does not."""
    d = [airfun(0.55, 0.13, M, table).cdf + airfun(0.55, 0.13, M, table).cdp
         for M in (0.60, 0.70, 0.78)]
    assert d[2] > d[1] > d[0]


def test_out_of_range_penalty_is_quadratic(table):
    """Leaving the database adds a fence to cdp; it is not physics."""
    hi = table.Acl[-1]
    for d in (0.1, 0.2):
        g = airfun(hi + d, 0.12, 0.7, table)
        assert g.penalty == pytest.approx(CL_PENALTY * d ** 2, rel=1e-12)
    tmax = table.Atau[-1]
    g = airfun(0.55, tmax + 0.01, 0.7, table)
    assert g.penalty == pytest.approx(TAU_PENALTY * 1e-4, rel=1e-12)
    # In range, no fence at all.
    assert airfun(0.55, 0.12, 0.7, table).penalty == 0.0


def test_penalty_is_included_in_reported_cdp(table):
    hi = table.Acl[-1]
    inside = airfun(hi, 0.12, 0.7, table)
    outside = airfun(hi + 0.2, 0.12, 0.7, table)
    assert outside.cdp > inside.cdp
    assert outside.cdp - outside.penalty == pytest.approx(
        outside.cdp - CL_PENALTY * 0.04, rel=1e-12)


def test_spline_of_a_quadratic_is_exact():
    """The end condition is zero *third* derivative, so quadratics are exact.

    A cubic is not -- it has a constant nonzero third derivative, which the
    end condition contradicts, so the fit is off near the ends.
    """
    s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    x = [2.0 + 3.0 * v - 0.5 * v ** 2 for v in s]
    for a, b in zip(spline(x, s), [3.0 - 1.0 * v for v in s]):
        assert a == pytest.approx(b, rel=1e-10)


def test_spline_of_a_cubic_is_not_exact_at_the_ends():
    s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    x = [2.0 + 3.0 * v - 0.5 * v ** 2 + 0.25 * v ** 3 for v in s]
    got = spline(x, s)
    want = [3.0 - 1.0 * v + 0.75 * v ** 2 for v in s]
    assert got[0] != pytest.approx(want[0], rel=1e-3)
    assert got[3] == pytest.approx(want[3], rel=0.05)   # interior is close


def test_spline_of_two_points_is_a_straight_line():
    xs = spline([1.0, 4.0], [0.0, 2.0])
    assert xs[0] == pytest.approx(1.5, rel=1e-12)
    assert xs[1] == pytest.approx(1.5, rel=1e-12)


def test_trisol_solves_a_known_system():
    #  [2 0 0; 1 2 0; 0 1 2] in (a=diag, b=sub, c=super) form
    a, b, c, d = [2.0, 2.0, 2.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], \
                 [3.0, 4.0, 5.0]
    x = trisol(a, b, c, d)
    assert 2 * x[0] + x[1] == pytest.approx(3.0, rel=1e-12)
    assert x[0] + 2 * x[1] + x[2] == pytest.approx(4.0, rel=1e-12)
    assert x[1] + 2 * x[2] == pytest.approx(5.0, rel=1e-12)
