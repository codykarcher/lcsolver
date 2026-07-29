"""Walking a tank along a mission, against TASOPT.jl.

The tank models elsewhere answer point questions. A mission is not a point:
fuel drains, altitude and Mach change, the heat leak follows. This supplies
the fuel demand and heat rate as functions of *time*.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.cryo.mission_tank import (fuel_flow_at, heat_rate_at)

REF = Path(__file__).parent / "data" / "tanktools_ref.csv"

pytestmark = pytest.mark.skipif(not REF.exists(),
                                reason="tanktools reference absent")

TIMES = [0.0, 600.0, 1800.0, 5400.0, 12000.0, 18000.0, 21600.0]
MDOTS = [0.30, 0.28, 0.24, 0.18, 0.12, 0.02, 0.0]
MDOTS0 = [0.30, 0.28, 0.00, 0.18, 0.12, 0.02, 0.0]
QS = [3400.0, 3300.0, 3000.0, 2600.0, 2550.0, 2900.0, 3450.0]


def test_interpolations_match_tasopt_jl_exactly():
    n = 0
    for r in csv.DictReader(REF.open()):
        t, want = float(r["t"]), float(r["val"])
        if r["kind"] == "mdot":
            got = fuel_flow_at(t, TIMES, MDOTS)
        elif r["kind"] == "mdot0":
            got = fuel_flow_at(t, TIMES, MDOTS0)
        else:
            got = heat_rate_at(t, TIMES, QS)
        assert got == want, (r["kind"], t)
        n += 1
    assert n == 24


def test_fuel_flow_is_exponential_and_heat_rate_is_linear():
    """Different on purpose. Fuel flow falls roughly geometrically as the
    aircraft lightens, so a straight line would understate it mid-segment;
    heat rate tracks altitude and speed, which the mission already
    discretises finely."""
    t = 0.5 * (TIMES[3] + TIMES[4])          # midpoint of a long cruise leg
    m0, mf = MDOTS[3], MDOTS[4]
    linear = 0.5 * (m0 + mf)
    geometric = (m0 * mf) ** 0.5
    assert fuel_flow_at(t, TIMES, MDOTS) == pytest.approx(geometric)
    assert fuel_flow_at(t, TIMES, MDOTS) < linear      # the understatement

    Q = heat_rate_at(t, TIMES, QS)
    assert Q == pytest.approx(0.5 * (QS[3] + QS[4]))


def test_stations_are_returned_exactly():
    for i, t in enumerate(TIMES):
        assert fuel_flow_at(t, TIMES, MDOTS) == MDOTS[i]
        assert heat_rate_at(t, TIMES, QS) == QS[i]


def test_a_segment_touching_zero_reads_zero_throughout():
    """Both ends, for different reasons -- see the module docstring. Worth
    pinning because it is a real understatement of descent-idle fuel burn,
    not a rounding matter."""
    # Ends at zero: the [1800, 5400] segment of MDOTS0 starts at 0.
    assert fuel_flow_at(3000.0, TIMES, MDOTS0) == 0.0
    # Starts at 0.28 and ends at 0 -- the unguarded case.
    assert fuel_flow_at(1200.0, TIMES, MDOTS0) == 0.0
    # ...but the station values themselves are untouched.
    assert fuel_flow_at(600.0, TIMES, MDOTS0) == 0.28
    # And the last real segment of MDOTS ends at zero too.
    assert fuel_flow_at(20000.0, TIMES, MDOTS) == 0.0
    assert fuel_flow_at(18000.0, TIMES, MDOTS) == 0.02


def test_a_time_outside_the_mission_is_refused():
    """The reference's loop falls through and returns `nothing`, which then
    fails as a type error somewhere less obvious."""
    with pytest.raises(ValueError, match="outside the mission"):
        fuel_flow_at(-1.0, TIMES, MDOTS)
    with pytest.raises(ValueError, match="outside the mission"):
        heat_rate_at(30000.0, TIMES, QS)


def test_the_heat_rate_dips_in_cruise_and_rises_either_side():
    """Which is what makes precomputing it per mission point worth doing:
    it is not monotone, so a two-point approximation would be wrong."""
    assert QS[4] < QS[0]           # cruise colder than takeoff
    assert QS[-1] > QS[4]          # descent warmer again
    mid = heat_rate_at(15000.0, TIMES, QS)
    assert QS[4] < mid < QS[5]
