"""Drawing a sized aircraft (tasopt_py.plot) against picwrt's own output.

TASOPT's five plot emitters write instructions for other programs -- gnuplot,
idraw, Matlab -- and none of them draws anything. :mod:`tasopt_py.plot` draws
the pictures instead, so the thing to check is that it draws *the reference
program's* picture rather than a redrawing of it.

``tests/data/737_g.plt`` is the gnuplot file ``picwrt`` wrote for the 737: the
whole outline, both halves, both engines, and the neutral-point and CG marks,
in feet about the neutral point. The tests below pull the line data straight
back off the matplotlib axes and compare it to that file, point for point.

The remaining tests are about the panels that have no reference file, where
what can be checked is that the right quantity reached the right axis.
"""
from __future__ import annotations

from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

from tasopt_py.model import indices as I                       # noqa: E402
from tasopt_py.plot import (compressor_maps, engine_deck,      # noqa: E402
                            figure_for, fuselage_bl,
                            mission_profile, plan_view, sweep)
from tasopt_py.run import run_case                             # noqa: E402

DATA = Path(__file__).parent / "data"
TAS = Path("/Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.tas")
REF = DATA / "737_g.plt"

pytestmark = pytest.mark.skipif(not TAS.exists() or not REF.exists(),
                                reason="737 case or gnuplot file not present")


@pytest.fixture(scope="module")
def flown():
    return run_case(TAS, off_design=False)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def reference_polylines():
    """``737_g.plt`` as a list of ``[(y, x), ...]``, blank-line separated."""
    polys, cur = [], []
    for ln in REF.read_text().splitlines():
        if not ln.strip():
            if cur:
                polys.append(cur)
                cur = []
            continue
        a, b = ln.split()
        cur.append((float(a), float(b)))
    if cur:
        polys.append(cur)
    return polys


def drawn_polylines(ax):
    return [list(zip(*ln.get_data())) for ln in ax.lines]


def test_the_outline_is_picwrts_outline(flown):
    """Every polyline picwrt writes, drawn, in its order and its units."""
    ax = plan_view(flown.case.pari, flown.case.parg, fill=False)
    got = drawn_polylines(ax)
    want = reference_polylines()

    assert len(want) == 12        # 3 surfaces x 2 halves, 2 engines, 4 marks
    assert len(got) == len(want)
    # The outline and engines come first, in picwrt's order; the four marks
    # are compared below.
    for k in range(8):
        assert len(got[k]) == len(want[k]), f"polyline {k}"
        for (gy, gx), (wy, wx) in zip(got[k], want[k]):
            assert gy == pytest.approx(wy, abs=5e-4)
            assert gx == pytest.approx(wx, abs=5e-4)


def test_the_neutral_point_and_cg_marks_match(flown):
    ax = plan_view(flown.case.pari, flown.case.parg, fill=False)
    got = drawn_polylines(ax)
    want = reference_polylines()
    # picwrt writes NP, CG aft, CG fwd as two-point ticks, then the CG range
    # along the centreline. plan_view draws them in the same order.
    for k in (8, 9, 10):
        for (gy, gx), (wy, wx) in zip(got[k], want[k]):
            assert gy == pytest.approx(wy, abs=5e-4)
            assert gx == pytest.approx(wx, abs=5e-4)
    # The CG range line is the last thing drawn and runs along y = 0.
    line = got[-1]
    assert all(y == 0.0 for y, _ in line)
    assert len(line) == 2


def test_the_origin_defaults_to_the_neutral_point(flown):
    """As picwrt's call site does -- `xorg = parg(igxNP)`. The NP tick then
    sits at zero, which is what makes the CG margins readable off the
    picture."""
    parg = flown.case.parg
    ax = plan_view(flown.case.pari, parg, fill=False)
    np_tick = drawn_polylines(ax)[8]
    assert all(x == pytest.approx(0.0, abs=1e-9) for _, x in np_tick)

    ax2 = plan_view(flown.case.pari, parg, xorg=0.0, fill=False)
    np_tick2 = drawn_polylines(ax2)[8]
    assert all(x == pytest.approx(-parg[I.IGXNP] / 0.3048, rel=1e-9)
               for _, x in np_tick2)


def test_the_drawing_is_mirrored_about_the_centreline(flown):
    """picwrt writes each surface twice, +y then -y. airpic only builds the
    starboard half."""
    ax = plan_view(flown.case.pari, flown.case.parg, fill=False)
    got = drawn_polylines(ax)
    for k in (0, 2, 4):
        for (y1, x1), (y2, x2) in zip(got[k], got[k + 1]):
            assert y2 == pytest.approx(-y1)
            assert x2 == pytest.approx(x1)


def test_both_engines_are_placed(flown):
    """`yoffe = yeng*(1 - 2*frac)`, so two engines land at +yeng and -yeng."""
    parg = flown.case.parg
    assert int(parg[I.IGNENG] + 0.01) == 2
    ax = plan_view(flown.case.pari, parg, fill=False)
    got = drawn_polylines(ax)
    y1 = sum(y for y, _ in got[6]) / len(got[6])
    y2 = sum(y for y, _ in got[7]) / len(got[7])
    assert y1 == pytest.approx(-y2)
    assert abs(y1) == pytest.approx(parg[I.IGYENG] / 0.3048, rel=1e-6)


def test_metres_are_available_too(flown):
    ax = plan_view(flown.case.pari, flown.case.parg, units="m", fill=False)
    span_ft = max(y for y, _ in drawn_polylines(
        plan_view(flown.case.pari, flown.case.parg, fill=False))[0])
    span_m = max(y for y, _ in drawn_polylines(ax)[0])
    assert span_m == pytest.approx(span_ft * 0.3048, rel=1e-9)


def test_an_unknown_unit_is_refused(flown):
    with pytest.raises(ValueError, match="units must be"):
        plan_view(flown.case.pari, flown.case.parg, units="cubits")


# --- the panels without a reference file ----------------------------------

def test_the_mission_profile_plots_the_flown_points(flown):
    """Static and rotation sit at or behind zero range and would fold the
    profile back on itself, so they are left out -- prfwrt tabulates them,
    but there is nothing to plot them against."""
    m = flown.case.missions[0]
    axes = mission_profile(flown.case.parg, m.parm, m.para, m.pare)
    R, alt = axes[0].lines[0].get_data()
    assert len(R) == 14                          # 17 points less ST, RO, and
    assert R[0] == pytest.approx(0.0)            # ... starting at takeoff
    assert R[-1] == pytest.approx(
        m.para[I.IARANGE, I.IPDESCENTN] * 0.000539975)
    assert max(alt) == pytest.approx(
        max(m.para[I.IAALT, ip] for ip in range(1, I.IPTOTAL)) / 304.8)


def test_the_boundary_layer_marks_the_trailing_edge(flown):
    """Everything downstream of it is wake and the closure switches there."""
    bl = flown.fuselage_bl
    axes = fuselage_bl(bl)
    xte = bl.x[bl.iblte - 1] / 0.3048
    for a in axes:
        assert any(vl.get_xdata()[0] == pytest.approx(xte)
                   for vl in a.lines if len(set(vl.get_xdata())) == 1)


def test_the_compressor_maps_mark_the_design_point(flown):
    m = flown.case.missions[0]
    axes = compressor_maps(m.pare)
    for a, idx in zip(axes, (I.IEPIFD, I.IEPILCD, I.IEPIHCD)):
        star = [ln for ln in a.lines if ln.get_marker() == "*"]
        assert len(star) == 1
        x, y = star[0].get_data()
        assert x[0] == pytest.approx(1.0)
        assert y[0] == pytest.approx(m.pare[idx, I.IPCRUISE1])


def test_the_compressor_maps_split_the_mission_into_segments(flown):
    """Four tracks per map, so it is visible which excursion is takeoff and
    which is descent -- the blank lines tfwrt writes, drawn."""
    m = flown.case.missions[0]
    axes = compressor_maps(m.pare)
    labels = [ln.get_label() for ln in axes[0].lines]
    assert labels == ["ground", "climb", "cruise", "descent", "design"]


def test_the_engine_deck_draws_one_curve_per_flight_condition():
    from tasopt_py.enginedeck import engine_deck as build
    from tasopt_py.tasfile import read_tase

    tase = TAS.with_suffix(".tase")
    if not tase.exists():
        pytest.skip("no .tase file")
    r = run_case(TAS, off_design=False)
    m = r.case.missions[0]
    deck = build(r.case.pari, r.case.parg, m.para, m.pare, read_tase(tase))

    ax = engine_deck(deck)
    assert len(ax.lines) == 9                    # 3 altitudes x 3 Mach
    F, tsfc = ax.lines[0].get_data()
    assert len(F) == 5                           # 5 throttle settings
    assert list(F) == sorted(F)                  # thrust increases with it


def test_a_swept_column_that_does_not_exist_is_refused(flown):
    with pytest.raises(ValueError, match="not a swept column"):
        sweep([flown], "PFEI", "wingspan")


def test_a_sweep_of_one_run_is_a_single_point(flown):
    ax = sweep([flown], "AR", "L/D")
    x, y = ax.lines[0].get_data()
    assert len(x) == 1
    assert x[0] == pytest.approx(flown.case.parg[I.IGAR])
    m = flown.case.missions[0]
    assert y[0] == pytest.approx(m.para[I.IACL, I.IPCRUISE1]
                                 / m.para[I.IACD, I.IPCRUISE1])
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("AR", "L/D")


def test_the_swept_pfei_column_needs_the_report_to_have_been_written(flown):
    """pltwrt's PFEI column reads parg, and only output.report puts the fleet
    value there -- so on a run with no report it is the 9e307 fill value.
    Inherited from the Fortran (DISCREPANCIES.md §43), not introduced here,
    and pinned so it is not mistaken for a plotting bug."""
    from tasopt_py.output import report
    from tasopt_py.tasfile import BIGNUM

    fresh = run_case(TAS, off_design=False)
    assert fresh.case.parg[I.IGPFEI] == BIGNUM
    ax = sweep([fresh], "AR", "PFEI")
    assert ax.lines[0].get_data()[1][0] == BIGNUM

    # Only the design mission was flown here, so report on that one alone --
    # the second mission's arrays are still unset and engwrt would divide by
    # zero walking them.
    fresh.case.missions = fresh.case.missions[:1]
    report(fresh.case, fresh)
    ax2 = sweep([fresh], "AR", "PFEI")
    assert ax2.lines[0].get_data()[1][0] == pytest.approx(7.849124, abs=1e-6)


def test_the_summary_page_builds(flown):
    """It is convenience only, but it should not raise -- it is the first
    thing anyone will try."""
    fig = figure_for(flown)
    assert len(fig.axes) >= 12
    assert "737-800" in fig._suptitle.get_text()


def test_the_summary_page_does_not_depend_on_a_written_report(flown):
    """Its PFEI is summed rather than read out of parg, so a figure works on
    a run that has not written a .out -- see the previous test."""
    fresh = run_case(TAS, off_design=False)
    title = figure_for(fresh)._suptitle.get_text()
    assert "PFEI 7.8491" in title
    assert "174,979" in title
