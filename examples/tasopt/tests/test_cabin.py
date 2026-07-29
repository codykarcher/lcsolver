"""Cabin layout and sizing, against TASOPT.jl.

TASOPT 2.16 does not lay a cabin out. The cabin length comes from
``xshell1``/``xshell2`` in the case file and nothing checks the passengers
fit -- change the fuselage radius and the cabin length does not move, because
there is no seat pitch to move it.

v3 seats the aircraft: seats abreast from the cabin width, rows from the
passenger count, cabin length from the rows. So diameter, pitch and length
are finally connected.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tasopt_py.structures.cabin import (AISLE_HALFWIDTH, FUSE_OFFSET,
                                        SEAT_LAYOUTS, SEAT_PITCH, SEAT_WIDTH,
                                        arrange_seats, find_cabin_width,
                                        find_floor_angles, place_cabin_seats,
                                        seats_abreast)

REF = Path(__file__).parent / "data" / "cabin_ref.csv"

pytestmark = pytest.mark.skipif(not REF.exists(),
                                reason="cabin reference absent")


def _rows(kind):
    return [r for r in csv.DictReader(REF.open()) if r["kind"] == kind]


def test_seats_abreast_matches():
    for r in _rows("abreast"):
        assert seats_abreast(float(r["a"])) == int(float(r["r1"]))


def test_seat_positions_match():
    """The window seats, exactly. The *sum* is not compared relatively --
    the seats are symmetric about the centreline so it is ~1e-16, and a
    relative comparison of floating-point noise means nothing."""
    for r in _rows("seaty"):
        w = float(r["a"])
        ys = arrange_seats(seats_abreast(w), w)
        assert ys[0] == float(r["r1"])
        assert ys[-1] == float(r["r2"])
        assert sum(ys) == pytest.approx(float(r["r3"]), abs=1e-12)


def test_cabin_placement_matches():
    for r in _rows("place"):
        lc, xs, n = place_cabin_seats(int(float(r["a"])), float(r["b"]))
        assert lc == pytest.approx(float(r["r1"]), rel=1e-14)
        assert n == int(float(r["r2"]))
        assert len(xs) == int(float(r["r3"]))


def test_cabin_width_matches():
    for r in _rows("width"):
        got = find_cabin_width(float(r["a"]), float(r["b"]),
                               int(float(r["c"])), float(r["d"]), 0.5)
        assert got == pytest.approx(float(r["r1"]), rel=1e-14)


def test_floor_angles_match():
    for r in _rows("angle1"):
        got = find_floor_angles(False, float(r["a"]), 0.0,
                                h_seat=float(r["b"]))
        assert got == float(r["r1"])
    for r in _rows("angle2"):
        a, b = find_floor_angles(True, float(r["a"]), 0.0,
                                 theta1=float(r["b"]),
                                 d_floor=float(r["c"]))
        assert a == pytest.approx(float(r["r1"]), rel=1e-14)
        assert b == pytest.approx(float(r["r2"]), rel=1e-14)


# --- what the layout model says -------------------------------------------

def test_seats_are_symmetric_about_the_centreline():
    for w in (3.0, 4.0, 5.5, 7.5):
        ys = arrange_seats(seats_abreast(w), w)
        assert ys[0] == pytest.approx(-ys[-1])
        assert sum(ys) == pytest.approx(0.0, abs=1e-12)


def test_spare_width_goes_to_the_aisles_not_the_walls():
    """A wide cabin gets wide aisles; the window seat stays put against the
    wall offset."""
    narrow = arrange_seats(6, 3.9)
    wide = arrange_seats(6, 4.6)
    # Window seat is the same distance from the wall in both.
    assert narrow[0] + 3.9 / 2.0 == pytest.approx(wide[0] + 4.6 / 2.0)
    # ...and the aisle has absorbed the difference.
    gap_n = narrow[3] - narrow[2]
    gap_w = wide[3] - wide[2]
    assert gap_w > gap_n


def test_a_wider_cabin_seats_more_people():
    assert (seats_abreast(3.0) < seats_abreast(4.5)
            < seats_abreast(6.0) < seats_abreast(7.5))


def test_the_single_deck_floor_angle_maximises_cabin_width():
    """The angle is chosen, not given: -asin(h_seat/2R) puts the floor half a
    seat height below centre so the widest point of the circle lands at
    shoulder level. Checked by sweeping."""
    R, h_seat = 1.9, 0.5
    best = find_floor_angles(False, R, 0.0, h_seat=h_seat)
    w_best = find_cabin_width(R, 0.0, 0, best, h_seat)
    for k in range(-40, 41):
        theta = best + k * 0.005
        try:
            w = find_cabin_width(R, 0.0, 0, theta, h_seat)
        except ValueError:
            continue                       # asin out of range
        assert w <= w_best * (1.0 + 1e-12)


def test_the_width_is_the_narrower_of_floor_and_shoulder():
    """A seat needs room at both heights, so the model takes the minimum --
    which is why raising the seat height can *narrow* the cabin."""
    R = 1.9
    wide = find_cabin_width(R, 0.0, 0, 0.0, 0.1)
    narrow = find_cabin_width(R, 0.0, 0, 0.0, 0.9)
    assert narrow < wide


def test_a_double_bubble_is_wider_by_the_web_offset():
    R, wfb = 1.9, 0.6
    single = find_cabin_width(R, wfb, 0, -0.1, 0.5)
    double = find_cabin_width(R, wfb, 1, -0.1, 0.5)
    assert double - single == pytest.approx(2.0 * wfb)


def test_the_cabin_length_is_the_last_seat_not_the_last_row():
    """`lcabin = xseats[end]`, so it measures to the *front* of the final row
    and omits that row's pitch and anything behind it. Worth pinning: it
    makes the cabin one pitch shorter than a seat-count argument suggests."""
    lc, xs, n = place_cabin_seats(180, 3.5)
    rows = math.ceil(180 / n)
    assert len(xs) == rows
    assert lc == xs[-1]

    # Reconstruct it: the front offset, one pitch per gap, plus half a pitch
    # for each of the two emergency rows.
    assert lc == pytest.approx(xs[0] + (rows - 1) * SEAT_PITCH
                               + 2 * SEAT_PITCH / 2.0)
    # A length that included the final row's own pitch would be one more.
    assert lc + SEAT_PITCH > lc


def test_emergency_exits_sit_at_fixed_row_numbers():
    """Rows 12-13 up to ten abreast, 19-20 above -- not derived from the
    passenger count, so a 100-seat and a 300-seat aircraft put them in the
    same place."""
    _, xs, n = place_cabin_seats(200, 3.5)
    assert n <= 10
    pitches = [b - a for a, b in zip(xs, xs[1:])]
    # Rows 12 and 13 (1-based) are reached by the 11th and 12th gaps.
    assert pitches[10] == pytest.approx(1.5 * SEAT_PITCH)
    assert pitches[11] == pytest.approx(1.5 * SEAT_PITCH)
    assert pitches[0] == pytest.approx(SEAT_PITCH)


def test_the_layout_table_stops_at_sixteen_abreast():
    assert max(SEAT_LAYOUTS) == 16
    assert SEAT_LAYOUTS[10] == (3, 4, 3)        # a twin-aisle widebody row
    assert sum(SEAT_LAYOUTS[10]) == 10
    for n, layout in SEAT_LAYOUTS.items():
        assert sum(layout) == n
    with pytest.raises(ValueError, match="layout table stops"):
        seats_abreast(20.0)


def test_the_defaults_are_the_sources():
    assert SEAT_PITCH == pytest.approx(30.0 * 0.0254)
    assert SEAT_WIDTH == pytest.approx(19.0 * 0.0254)
    assert AISLE_HALFWIDTH == pytest.approx(10.0 * 0.0254)
    assert FUSE_OFFSET == pytest.approx(6.0 * 0.0254)
