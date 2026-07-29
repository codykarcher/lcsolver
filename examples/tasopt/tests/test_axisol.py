"""Axisymmetric potential flow (axisol.f) vs the compiled Fortran.

A 737-like fuselage in four configurations: tail tapering to a point and to an
edge, at M = 0.8 and M = 0.2, and a coarser panelling with blunter end
sections.

Reference regenerated with::

    gfortran -fdefault-real-8 -O0 -o drv_axisol drv_axisol.f axisol.f gaussn.f
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tasopt_py.aero.axisol import COSINE_SPACING, axisol, vline, vsurf

DATA = Path(__file__).parent / "data"
RTOL = 1e-13

BODY = dict(xnose=0.0, xend=37.8, xblend1=6.1, xblend2=30.0, Amax=12.0)
CASES = {1: dict(iclose=0, Mach=0.80, nc=30, anose=1.8, btail=1.6),
         2: dict(iclose=1, Mach=0.80, nc=30, anose=1.8, btail=1.6),
         3: dict(iclose=0, Mach=0.20, nc=30, anose=1.8, btail=1.6),
         4: dict(iclose=0, Mach=0.80, nc=20, anose=2.2, btail=2.0)}


def _run(ic):
    kw = CASES[ic]
    return axisol(BODY["xnose"], BODY["xend"], BODY["xblend1"],
                  BODY["xblend2"], BODY["Amax"], kw["anose"], kw["btail"],
                  kw["iclose"], kw["Mach"], kw["nc"], 200)


def test_matches_fortran():
    got = {ic: _run(ic) for ic in CASES}
    n = 0
    with (DATA / "axisol_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic, i = int(row[0]), int(row[1])
            g = got[ic]
            if i == 0:
                pairs = [(float(g.nl), float(row[2])),
                         (float(g.ilte), float(row[3]))]
            else:
                pairs = list(zip((g.x[i - 1], g.z[i - 1], g.s[i - 1],
                                  g.dy[i - 1], g.q[i - 1]),
                                 map(float, row[2:7])))
            for a, b in pairs:
                assert a == pytest.approx(b, rel=RTOL, abs=1e-300), \
                    f"case {ic} i={i}"
                n += 1
    assert n == 873


def test_point_counts():
    """nl = nc + nc/2 + 2 surface-plus-wake points, TE at nc + 1."""
    for ic, kw in CASES.items():
        g = _run(ic)
        assert g.ilte == kw["nc"] + 1
        assert g.nl == kw["nc"] + kw["nc"] // 2 + 2


def test_barrel_radius_is_the_equivalent_round_body():
    g = _run(1)
    Rcyl = math.sqrt(BODY["Amax"] / math.pi)
    mid = [g.z[i] for i in range(g.ilte)
           if BODY["xblend1"] < g.x[i] < BODY["xblend2"]]
    assert mid
    for zz in mid:
        assert zz == pytest.approx(Rcyl, rel=1e-14)


def test_stagnation_at_the_nose_and_acceleration_over_the_barrel():
    g = _run(1)
    assert g.q[0] == 0.0                       # nose stagnation point
    barrel = [g.q[i] for i in range(1, g.ilte)
              if BODY["xblend1"] < g.x[i] < BODY["xblend2"]]
    assert max(barrel) > 1.0                   # flow speeds up over the body


def test_compressibility_raises_the_peak_speed():
    """Prandtl-Glauert: the same body at higher Mach accelerates the flow more."""
    slow = max(_run(3).q)
    fast = max(_run(1).q)
    assert fast > slow


def test_trailing_edge_and_wake_are_held_off_the_axis():
    """The line source is singular on the axis, so neither may sit at z = 0."""
    g = _run(1)
    assert g.z[g.ilte - 1] == pytest.approx(0.25 * g.z[g.ilte - 2], rel=1e-14)
    for i in range(g.ilte, g.nl):
        assert g.z[i] == pytest.approx(0.125 * g.z[g.ilte - 2], rel=1e-14)
        assert g.z[i] > 0.0


def test_edge_tail_carries_a_lateral_width():
    point, edge = _run(1), _run(2)
    assert all(v == 0.0 for v in point.dy)
    assert max(edge.dy) > 0.0
    # Only aft of the second blend point.
    for i in range(edge.ilte):
        if edge.x[i] < BODY["xblend2"]:
            assert edge.dy[i] == 0.0


def test_arc_length_is_monotone():
    for ic in CASES:
        g = _run(ic)
        assert all(b > a for a, b in zip(g.s, g.s[1:]))


def test_cosine_spacing_is_the_live_option():
    """axisol.f carries a uniform-spacing option, commented out."""
    assert COSINE_SPACING is True


def test_source_line_reduces_to_the_incompressible_kernel():
    """At M = 0 the Prandtl-Glauert factor is 1 and beta drops out."""
    u, v, w = vline(5.0, 0.0, 2.0, 0.0, 1.0, 1.0)
    assert u != 0.0 and w != 0.0
    assert v == 0.0                     # on the y = 0 plane


def test_source_panel_matches_a_line_as_it_narrows():
    """A panel of vanishing width induces the same velocity as a line."""
    line = vline(5.0, 0.0, 2.0, 0.0, 1.0, 0.6)
    panel = vsurf(5.0, 0.0, 2.0, 0.0, 1.0, -1e-6, 1e-6, 0.6)
    for a, b in zip(line, panel):
        assert a == pytest.approx(b, rel=1e-6, abs=1e-12)
