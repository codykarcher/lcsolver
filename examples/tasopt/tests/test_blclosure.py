"""Boundary-layer closure relations (blsys.f) vs the compiled Fortran.

The nine correlations over a 40-point grid in shape parameter, Reynolds number
and Mach number, spanning both branches of every routine that has two.

Reference regenerated with::

    gfortran -fdefault-real-8 -O0 -o drv_blclosure drv_blclosure.f blsys.f
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tasopt_py.aero import blclosure as B

DATA = Path(__file__).parent / "data"
RTOL = 1e-13


def _grid(i, j):
    return 1.2 + 0.6 * i, 10.0 ** (1.0 + 0.8 * j), 0.04 * j


def test_matches_fortran():
    n = 0
    with (DATA / "blclosure_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "name":
                continue
            name, i, j, ref = (row[0].strip(), int(row[1]), int(row[2]),
                               float(row[3]))
            hk, rt, msq = _grid(i, j)
            got = {
                "hkin": lambda: B.hkin(hk, msq),
                "hsl": lambda: B.hsl(hk, rt, msq),
                "hst": lambda: B.hst(hk, rt, msq),
                "cfl": lambda: B.cfl(hk, rt, msq),
                "cft": lambda: B.cft(hk, rt, msq),
                "dil": lambda: B.dil(hk, rt),
                "dilw": lambda: B.dilw(hk, rt),
                "hct": lambda: B.hct(hk, msq),
                "dit": lambda: B.dit(1.7 + 0.1 * i, 0.1 * j, 0.002, 0.01 * j),
            }[name]()
            assert got == pytest.approx(ref, rel=RTOL), f"{name} i={i} j={j}"
            n += 1
    assert n == 360


def test_both_branches_of_every_two_branch_routine_are_exercised():
    """hsl/cfl/dil switch on Hk, hst on Hk vs Ho -- the grid must span both."""
    hks = [_grid(i, 1)[0] for i in range(1, 9)]
    assert min(hks) < 4.0 < max(hks)      # dil
    assert min(hks) < 4.35 < max(hks)     # hsl
    assert min(hks) < 5.5 < max(hks)      # cfl


def test_shape_parameters_are_continuous_across_their_branch_points():
    for f, cut in ((lambda h: B.hsl(h, 1e4, 0.0), 4.35),
                   (lambda h: B.cfl(h, 1e4, 0.0), 5.5),
                   (lambda h: B.dil(h, 1e4), 4.0)):
        lo, hi = f(cut - 1e-7), f(cut + 1e-7)
        assert lo == pytest.approx(hi, rel=1e-5)


def test_hst_reynolds_floor():
    """Rtheta dependence is limited below 200 -- source note dated 12/4/94."""
    assert B.hst(2.0, 50.0, 0.0) == pytest.approx(B.hst(2.0, 200.0, 0.0),
                                                  rel=1e-14)
    assert B.hst(2.0, 400.0, 0.0) != pytest.approx(B.hst(2.0, 200.0, 0.0),
                                                   rel=1e-6)


def test_hst_branch_switch_follows_ho():
    """Ho = 3 + 400/Rt above Rt = 400, and 4 below it."""
    # Just either side of Ho at a high Reynolds number.
    rt = 1e5
    ho = 3.0 + 400.0 / rt
    assert B.hst(ho - 1e-6, rt, 0.0) == pytest.approx(
        B.hst(ho + 1e-6, rt, 0.0), rel=1e-4)


def test_cft_clamps_keep_it_finite():
    """log(Rt/Fc) floors at 3 and the exponent argument at -20."""
    assert math.isfinite(B.cft(2.0, 1.0, 0.0))       # tiny Reynolds number
    assert math.isfinite(B.cft(30.0, 1e6, 0.0))      # huge shape parameter
    # Below the Reynolds floor the answer stops moving.
    assert B.cft(2.0, 1.0, 0.0) == pytest.approx(B.cft(2.0, 5.0, 0.0),
                                                 rel=1e-12)


def test_wake_dissipation_ignores_mach():
    """dilw calls hsl with Msq forced to zero, whatever the real Mach."""
    import inspect
    assert "hsl(hk, rt, 0.0)" in inspect.getsource(B.dilw)


def test_friction_falls_with_reynolds_number():
    for f in (lambda rt: B.cfl(2.5, rt, 0.0), lambda rt: B.cft(2.0, rt, 0.2)):
        assert f(1e6) < f(1e4)


def test_laminar_friction_goes_negative_when_separated():
    """The Falkner-Skan fit passes through zero near Hk = 4 and stays
    negative beyond it -- which is how the march detects separation."""
    assert B.cfl(3.0, 1e4, 0.0) > 0.0
    assert B.cfl(6.0, 1e4, 0.0) < 0.0


def test_hct_vanishes_in_incompressible_flow():
    assert B.hct(2.0, 0.0) == 0.0


# --- the derivative forms -------------------------------------------------

def _d_pairs(hk, rt, msq):
    return ((B.hkin(hk, msq), B.hkin_d(hk, msq)),
            (B.hsl(hk, rt, msq), B.hsl_d(hk, rt, msq)),
            (B.hst(hk, rt, msq), B.hst_d(hk, rt, msq)),
            (B.cfl(hk, rt, msq), B.cfl_d(hk, rt, msq)),
            (B.cft(hk, rt, msq), B.cft_d(hk, rt, msq)),
            (B.dil(hk, rt), B.dil_d(hk, rt)))


def test_derivative_forms_return_the_same_value():
    """The two forms are written out separately, so pin them together."""
    for i in range(1, 9):
        for j in range(5):
            hk, rt, msq = _grid(i, j)
            for value, withd in _d_pairs(hk, rt, msq):
                assert withd[0] == value


def test_derivatives_agree_with_finite_differences():
    """A blunt check that the transcribed analytic derivatives are the
    derivatives of the values above them, away from the branch points."""
    def fd(f, args, k, h):
        up, dn = list(args), list(args)
        up[k], dn[k] = args[k] + h, args[k] - h
        return (f(*up) - f(*dn)) / (2.0 * h)

    for hk, rt, msq in ((1.8, 1.0e4, 0.30), (2.6, 5.0e5, 0.64),
                        (3.9, 8.0e2, 0.10), (5.0, 2.0e5, 0.50)):
        checks = (
            (lambda h, m: B.hkin(h, m), B.hkin_d(hk, msq)[1:], (hk, msq)),
            (lambda k, r, m: B.hsl(k, r, m), B.hsl_d(hk, rt, msq)[1:2],
             (hk, rt, msq)),
            (lambda k, r, m: B.hst(k, r, m), B.hst_d(hk, rt, msq)[1:],
             (hk, rt, msq)),
            (lambda k, r, m: B.cfl(k, r, m), B.cfl_d(hk, rt, msq)[1:3],
             (hk, rt, msq)),
            (lambda k, r, m: B.cft(k, r, m), B.cft_d(hk, rt, msq)[1:],
             (hk, rt, msq)),
            (lambda k, r: B.dil(k, r), B.dil_d(hk, rt)[1:], (hk, rt)),
        )
        for f, ders, args in checks:
            for k, want in enumerate(ders):
                step = 1e-6 * max(abs(args[k]), 1.0)
                assert fd(f, args, k, step) == pytest.approx(
                    want, rel=2e-5, abs=1e-12), f"{f} arg {k} at hk={hk}"
