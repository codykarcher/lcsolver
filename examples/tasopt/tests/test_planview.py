"""The aircraft outline and plot files (airpic.f, pltwrt) against the 737.

``tests/data/737_2.plt`` and ``737_3.plt`` are the Matlab parameter and
geometry files the shipped program wrote for the 737. This checks the port
reproduces both, line for line: the 28-column parameter row and all 34
geometry points -- six wing, six tail, fourteen fuselage, eight nacelle.

``airpic`` is worth having beyond the plot files. It is the only place in
TASOPT that turns a sized aircraft into a picture, and :func:`airpic` hands
back the four polylines directly, so a converged case can be drawn without
going through a Matlab file.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tasopt_py.model import indices as I
from tasopt_py.output import report
from tasopt_py.planview import (PLOT_COLUMNS, PLOT_XAXIS, airpic, plot_row,
                                pltwrt)
from tasopt_py.run import run_case

DATA = Path(__file__).parent / "data"
TAS = Path("/Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.tas")
REF2 = DATA / "737_2.plt"
REF3 = DATA / "737_3.plt"

pytestmark = pytest.mark.skipif(
    not TAS.exists() or not REF2.exists() or not REF3.exists(),
    reason="737 case or plot files not present")


@pytest.fixture(scope="module")
def drawn():
    r = run_case(TAS)
    # tasopt.f writes the .out before the plot files, and that is where the
    # fleet PFEI gets stored -- pltwrt's first column depends on it.
    report(r.case, r)
    m = r.case.missions[0]
    view = airpic(r.case.pari, r.case.parg)
    row, geom = pltwrt(view, r.case.parg, m.parm, m.para, m.pare, 0.0)
    return r, view, row, geom


def test_parameter_row_matches_the_fortran(drawn):
    _, _, row, _ = drawn
    ref = REF2.read_text().splitlines()
    assert row == ref[1]


def test_geometry_matches_the_fortran(drawn):
    _, _, _, geom = drawn
    ref = REF3.read_text().splitlines()
    assert len(geom) == len(ref) - 1 == 34
    for k, (got, want) in enumerate(zip(geom, ref[1:])):
        assert got == want, f"geometry line {k + 1}"


def test_the_polyline_counts_are_what_the_header_declares(drawn):
    _, view, _, _ = drawn
    ni, nj, nw, nh, nf, ne = (int(t) for t in
                              REF3.read_text().splitlines()[0].split())
    assert (nw, nh, nf, ne) == (6, 6, 14, 8)
    assert len(view.wing) == nw
    assert len(view.htail) == nh
    assert len(view.fuselage) == nf
    assert len(view.nacelle) == ne


def test_the_outline_is_the_starboard_half(drawn):
    """Only y >= 0, except the nacelle, which is drawn about its own axis."""
    _, view, _, _ = drawn
    for poly in (view.wing, view.htail, view.fuselage):
        assert all(y >= 0.0 for _, y in poly)
    assert min(y for _, y in view.nacelle) < 0.0


def test_the_fuselage_closes_on_the_axis(drawn):
    _, view, _, _ = drawn
    assert view.fuselage[0][1] == pytest.approx(0.0, abs=1e-12)  # nose tip
    assert view.fuselage[-1][1] == 0.0                           # tail close
    assert view.fuselage[-1][0] == view.fuselage[-2][0]


def test_the_wing_spans_the_aircraft(drawn):
    r, view, _, _ = drawn
    assert max(y for _, y in view.wing) == pytest.approx(
        r.case.parg[I.IGB] / 2.0)


def test_the_nacelle_can_be_placed_at_an_engine(drawn):
    r, view, _, _ = drawn
    xe, ye = r.case.parg[I.IGXENG], r.case.parg[I.IGYENG]
    moved = view.nacelle_at(xe, ye)
    assert moved[0][0] == pytest.approx(view.nacelle[0][0] + xe)
    assert moved[0][1] == pytest.approx(view.nacelle[0][1] + ye)


def test_the_drawing_axis_does_not_follow_the_models(drawn):
    """airpic pins both surfaces at a hard-wired 40% chord and never reads
    ``Xaxis``, which is what the structures and moments use. The 737 happens
    to set Xaxis to 0.40 as well, so the two coincide here -- but changing
    Xaxis moves the modelled planform and leaves the drawn one alone."""
    r, _, _, _ = drawn
    assert PLOT_XAXIS == 0.40
    assert r.case.parg[I.IGXAXIS] == 0.40      # coincidence on this case

    before = airpic(r.case.pari, r.case.parg).wing
    r.case.parg[I.IGXAXIS] = 0.25
    after = airpic(r.case.pari, r.case.parg).wing
    assert after == before


def test_the_parameter_row_is_the_documented_columns(drawn):
    r, _, _, _ = drawn
    m = r.case.missions[0]
    row = plot_row(r.case.parg, m.parm, m.para, m.pare, 0.0)
    # The leading value is the swept j parameter; the rest are the 28.
    assert len(row) == len(PLOT_COLUMNS) + 1
    assert row[1] == pytest.approx(r.case.parg[I.IGPFEI])
    assert row[4] == pytest.approx(r.case.parg[I.IGWMTO] / 4.44822, rel=1e-9)
    assert PLOT_COLUMNS[0] == "PFEI" and PLOT_COLUMNS[-1] == "sweep [deg]"
