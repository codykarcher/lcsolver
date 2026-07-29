"""The compressor-map file (``mapwrt``) against the real TASOPT 737 run.

``tests/data/tfan_800.dat`` is the file the shipped program writes for the
737 -- but only after you edit the source. ``Ltfwrite`` is hard-wired
``.false.`` in ``tasopt.f`` with the ``.true.`` line commented out directly
beneath it, and its ``getLval`` read is commented out of ``getparm.f``, so no
``.tas`` file can switch it on. The reference here was produced by flipping
both lines, rebuilding, running, and putting the source back
(``DISCREPANCIES.md`` §45).

That is worth the trouble because ``mapwrt`` is the only place TASOPT reports
where each compressor sits *on its map* -- corrected mass flow and speed as
fractions of design, against pressure ratio. Those are the coordinates a
compressor map is drawn in, and :func:`tasopt_py.output.map_point` hands them
back as numbers so a converged case can be plotted on one.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tasopt_py.model import indices as I
from tasopt_py.output import map_point, mapwrt, tfwrt
from tasopt_py.run import run_case

DATA = Path(__file__).parent / "data"
TAS = Path("/Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.tas")
REF = DATA / "tfan_800.dat"

pytestmark = pytest.mark.skipif(not TAS.exists() or not REF.exists(),
                                reason="737 case or tfan file not present")


@pytest.fixture(scope="module")
def flown():
    return run_case(TAS, off_design=False)


def test_the_whole_file_matches_the_fortran(flown):
    m = flown.case.missions[0]
    got = tfwrt(flown.case.parg, m.para, m.pare)
    want = REF.read_text().splitlines()
    assert len(got) == len(want)
    for k, (a, b) in enumerate(zip(got, want)):
        assert a == b, f"line {k + 1}"


def test_the_header_is_written_only_for_the_first_row(flown):
    pare = flown.case.missions[0].pare
    assert len(mapwrt(pare.column(I.IPSTATIC), 1)) == 3
    assert len(mapwrt(pare.column(I.IPSTATIC), 0)) == 1
    assert len(mapwrt(pare.column(I.IPSTATIC))) == 1


def test_the_segments_are_separated_by_blank_lines(flown):
    """Three blanks, before climb, cruise and descent -- what tells a
    plotting program where to lift the pen."""
    m = flown.case.missions[0]
    got = tfwrt(flown.case.parg, m.para, m.pare)
    assert sum(1 for ln in got if ln == "") == 3


def test_the_design_point_sits_at_the_design_coordinates(flown):
    """Cruise 1 is where the engine was sized, so the fan and LPC should be
    at exactly 1.0 in both corrected mass flow and corrected speed. The HPC
    is not -- tfsize sizes it at its own design corrected flow, and the
    cruise point is off that by 2.8%."""
    pare = flown.case.missions[0].pare
    m = map_point(pare.column(I.IPCRUISE1))
    assert m.fan[0] == pytest.approx(1.0, abs=1e-12)
    assert m.fan[2] == pytest.approx(1.0, abs=1e-12)
    assert m.lpc[0] == pytest.approx(1.0, abs=1e-12)
    assert m.lpc[2] == pytest.approx(1.0, abs=1e-12)
    assert m.hpc[0] == pytest.approx(1.02838, abs=1e-5)


def test_the_map_coordinates_are_per_component_not_per_core(flown):
    """mapwrt scales the fan's corrected flow by the bypass ratio, so it is
    the flow through the fan rather than through the core. Getting this wrong
    puts the fan a factor of six off its map."""
    pare = flown.case.missions[0].pare
    col = pare.column(I.IPCRUISE1)
    m = map_point(col)
    # Undo the design normalisation and the BPR, and the fan and LPC
    # corrected flows are the same core flow at (nearly) the same station.
    fan = m.fan[0] * col[I.IEMBFD] / col[I.IEBPR]
    lpc = m.lpc[0] * col[I.IEMBLCD]
    assert fan == pytest.approx(lpc, rel=1e-9)


def test_the_takeoff_point_is_the_highest_pressure_ratio(flown):
    """Which is what the file is for -- seeing the excursion the engine makes
    over the mission, and how close takeoff runs to the top of the map."""
    pare = flown.case.missions[0].pare
    pts = {ip: map_point(pare.column(ip))
           for ip in range(I.IPSTATIC, I.IPDESCENTN + 1)}
    hottest = max(pts, key=lambda ip: pts[ip].OPR)
    assert hottest == I.IPCLIMBN
    assert pts[I.IPTAKEOFF].fan[1] > pts[I.IPCRUISE1].fan[1]
    assert pts[I.IPDESCENTN].fan[1] < pts[I.IPCRUISE1].fan[1]
