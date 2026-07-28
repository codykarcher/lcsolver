"""Planform sizing and centroid geometry vs the compiled Fortran.

``tailpo.f``, ``surfdx.f``, ``wingsc.f`` (both ``wingsc`` and ``wingAc``).

Reference regenerated with::

    gfortran -fdefault-real-8 -O0 -o drv_planform \\
        drv_planform.f tailpo.f surfdx.f wingsc.f
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tasopt_py.aero.loading import planform_integrals
from tasopt_py.structures.planform import (chord_integrals, surfdx, tailpo,
                                           wingAc, wingsc)

DATA = Path(__file__).parent / "data"
RTOL = 1e-13


def _wing_inputs(i):
    return dict(W=600000.0 + 40000.0 * i, CL=0.50 + 0.03 * i,
                qinf=11000.0 + 500.0 * i, AR=8.0 + 0.5 * i,
                etasi=0.25 + 0.02 * i, bo=3.2 + 0.15 * i,
                lambdat=0.15 + 0.02 * i, lambdas=0.65 + 0.03 * i)


def test_matches_fortran():
    n = 0
    with (DATA / "planform_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0] == "name":
                continue
            kind, i = row[0][:6], int(row[0][6:])
            if kind == "tailpo":
                g = tailpo(25.0 + 4.0 * i, 4.0 + 0.6 * i, 0.20 + 0.05 * i,
                           12000.0 + 900.0 * i, 0.6 + 0.1 * i)
                got = (g.b, g.co, g.po)
            elif kind == "surfdx":
                g = surfdx(33.0 + 1.5 * i, 9.0 + 0.6 * i, 3.2 + 0.15 * i,
                           0.15 + 0.02 * i, 0.65 + 0.03 * i, 22.0 + 2.0 * i)
                got = (g.dx, g.macco)
            else:
                kw = _wing_inputs(i)
                w = wingsc(**kw)
                if kind == "wingsc":
                    got = (w.S, w.b, w.bs, w.co)
                else:
                    a = wingAc(kw["W"], kw["CL"], kw["qinf"], w.b, w.bs,
                               kw["bo"], kw["lambdat"], kw["lambdas"])
                    got = (a.S, a.AR, a.bs, a.co)
            for a, b in zip(got, map(float, row[1:1 + len(got)])):
                assert a == pytest.approx(b, rel=RTOL), row[0]
                n += 1
    assert n == 52


def test_wingsc_delivers_the_requested_lift():
    kw = _wing_inputs(2)
    w = wingsc(**kw)
    assert w.S * kw["qinf"] * kw["CL"] == pytest.approx(kw["W"], rel=1e-14)
    assert w.b ** 2 / w.S == pytest.approx(kw["AR"], rel=1e-14)


def test_wingsc_and_wingac_are_inverses():
    """Feeding wingsc's span back into wingAc must return its aspect ratio."""
    kw = _wing_inputs(3)
    w = wingsc(**kw)
    a = wingAc(kw["W"], kw["CL"], kw["qinf"], w.b, w.bs, kw["bo"],
               kw["lambdat"], kw["lambdas"])
    assert a.AR == pytest.approx(kw["AR"], rel=1e-13)
    assert a.S == pytest.approx(w.S, rel=1e-14)
    assert a.co == pytest.approx(w.co, rel=1e-14)


def test_break_is_pushed_out_to_the_side_of_body():
    """A break requested inside the fuselage is moved to bo."""
    kw = dict(_wing_inputs(1), etasi=0.01)
    w = wingsc(**kw)
    assert w.bs == kw["bo"]


def test_root_chord_reproduces_the_area():
    """S = Kc * co * b is the definition of Kc, so it must close."""
    kw = _wing_inputs(4)
    w = wingsc(**kw)
    Kc, _, _ = chord_integrals(w.b, w.bs, kw["bo"], kw["lambdat"],
                               kw["lambdas"])
    assert Kc * w.co * w.b == pytest.approx(w.S, rel=1e-14)


def test_chord_integral_agrees_with_the_loading_module():
    """Kc is built independently in aero.loading; the two must not drift."""
    b, bs, bo, lt, ls = 35.0, 10.5, 3.6, 0.18, 0.70
    Kc, _, _ = chord_integrals(b, bs, bo, lt, ls)
    Kc2, _, _ = planform_integrals(b, bs, bo, lt, ls, 0.9, 1.2,
                                   b ** 2 / 124.0, -0.3, -0.05)
    assert Kc == pytest.approx(Kc2, rel=1e-15)


def test_unswept_surface_has_no_centroid_offset():
    g = surfdx(35.0, 10.5, 3.6, 0.18, 0.70, 0.0)
    assert g.dx == 0.0
    assert 0.0 < g.macco < 1.0        # mac is shorter than the root chord


def test_centroid_offset_is_linear_in_tan_sweep():
    a = surfdx(35.0, 10.5, 3.6, 0.18, 0.70, 20.0)
    b = surfdx(35.0, 10.5, 3.6, 0.18, 0.70, 40.0)
    ratio = math.tan(math.radians(40.0)) / math.tan(math.radians(20.0))
    assert b.dx / a.dx == pytest.approx(ratio, rel=1e-14)


def test_rectangular_planform_has_unit_mac():
    """With no taper and no centre section, mac equals the root chord."""
    g = surfdx(30.0, 3.0, 3.0, 1.0, 1.0, 0.0)
    assert g.macco == pytest.approx(1.0, rel=1e-14)


def test_tailpo_scales_with_dynamic_pressure():
    a = tailpo(31.0, 5.5, 0.25, 12000.0, 0.8)
    b = tailpo(31.0, 5.5, 0.25, 24000.0, 0.8)
    assert b.po / a.po == pytest.approx(2.0, rel=1e-14)
    assert b.b == a.b and b.co == a.co
