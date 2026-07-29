"""Cabin layout and sizing -- ``size_cabin.jl``.

TASOPT 2.16 does not lay a cabin out. The cabin length comes from
``xshell1``/``xshell2`` in the ``.tas`` file, the payload from ``Wpay``, and
nothing checks that the passengers actually fit. Change the fuselage radius
and the cabin length does not move; change the seat pitch and nothing
happens, because there is no seat pitch.

v3 seats the aircraft. Given a passenger count and a fuselage cross-section
it works out how many seats fit abreast, how many rows that needs, and how
long the cabin therefore is -- so fuselage diameter, seat pitch and cabin
length are finally connected.

What sets the width
-------------------
The cabin is not as wide as the fuselage. Two things narrow it:

* **Where the floor sits.** A floor below the centreline is in a narrower
  part of the circle. The single-deck case chooses the angle that *maximises*
  cabin width, which is ``-asin(h_seat / 2R)`` -- half a seat height below
  centre, so the widest point of the circle falls at shoulder height rather
  than at the floor.
* **Shoulder room, not floor room.** :func:`find_cabin_width` takes the
  **narrower** of the width at the floor and the width at seat height,
  because a seat needs both.

A double bubble adds ``nfweb * 2 * wfb`` on top, which is the whole point of
the shape.

Things worth knowing
--------------------
* **The cabin length is the last seat's station**, not the last row's back.
  ``lcabin = xseats[end]``, so it measures to the front of the final row and
  omits both that row's pitch and the galley behind it.
* **Emergency exits are at fixed row numbers** -- rows 12-13 for ten-abreast
  or fewer, 19-20 above that -- and add half a pitch each. Not derived from
  the passenger count, so a 100-seat aircraft and a 300-seat one put them in
  the same place.
* The seat layout table stops at 16 abreast.

Verified against TASOPT.jl; see ``tests/test_cabin.py``.
"""
from __future__ import annotations

import math

__all__ = ["SEAT_LAYOUTS", "seats_abreast", "arrange_seats",
           "place_cabin_seats", "find_cabin_width", "find_floor_angles",
           "IN_TO_M", "FT_TO_M"]

IN_TO_M = 0.0254
FT_TO_M = 0.3048

#: How many seats sit in each block between aisles, by seats abreast.
#: ``[3, 4, 3]`` is a twin-aisle ten-abreast row. From ``constants.jl``.
SEAT_LAYOUTS = {
    1: (1, 0), 2: (1, 1), 3: (2, 1), 4: (2, 2), 5: (3, 2), 6: (3, 3),
    7: (2, 3, 2), 8: (2, 4, 2), 9: (3, 3, 3), 10: (3, 4, 3), 11: (3, 5, 3),
    12: (3, 3, 3, 3), 13: (3, 4, 3, 3), 14: (3, 4, 4, 3), 15: (3, 5, 4, 3),
    16: (3, 5, 5, 3),
}

#: Defaults, all from the source's keyword arguments.
SEAT_PITCH = 30.0 * IN_TO_M
SEAT_WIDTH = 19.0 * IN_TO_M
AISLE_HALFWIDTH = 10.0 * IN_TO_M
FUSE_OFFSET = 6.0 * IN_TO_M
FRONT_SEAT_OFFSET = 10.0 * FT_TO_M


def _aisle_flag(idx: int, layout) -> float:
    """1.0 if there is an aisle to the left of seat ``idx`` (1-based)."""
    running = 0
    for block in layout:
        running += block
        if idx - running == 1:
            return 1.0
    return 0.0


def seats_abreast(cabin_width: float, seat_width: float = SEAT_WIDTH,
                  aisle_halfwidth: float = AISLE_HALFWIDTH,
                  fuse_offset: float = FUSE_OFFSET) -> int:
    """The most seats that fit across ``cabin_width``.

    Widens one seat at a time until the required width exceeds what is
    available, then backs off by one. Each extra seat may also add an aisle,
    which is why the layout table is consulted rather than assuming a fixed
    aisle count.
    """
    n = 1
    Dmin = n * seat_width + 2.0 * aisle_halfwidth + 2.0 * fuse_offset
    while cabin_width > Dmin or math.isclose(cabin_width, Dmin,
                                             rel_tol=1e-9, abs_tol=1e-12):
        n += 1
        if n not in SEAT_LAYOUTS:
            raise ValueError(
                f"cabin width {cabin_width:.3f} m would seat more than "
                f"{max(SEAT_LAYOUTS)} abreast, which is where the layout "
                "table stops")
        layout = SEAT_LAYOUTS[n]
        Dmin = (n * seat_width
                + (len(layout) - 1) * 2.0 * aisle_halfwidth
                + 2.0 * fuse_offset)
    return n - 1


def arrange_seats(seats_per_row: int, cabin_width: float,
                  seat_width: float = SEAT_WIDTH,
                  aisle_halfwidth: float = AISLE_HALFWIDTH,
                  fuse_offset: float = FUSE_OFFSET) -> list:
    """Lateral seat centre positions, m, measured from the centreline.

    Any width left over after the seats and the minimum aisles is given
    **to the aisles**, split equally -- so a wide cabin gets wide aisles
    rather than a gap at the walls.
    """
    layout = SEAT_LAYOUTS[seats_per_row]
    n_aisles = len(layout) - 1
    Dmin = (seats_per_row * seat_width + n_aisles * 2.0 * aisle_halfwidth
            + 2.0 * fuse_offset)
    exp_aisle = aisle_halfwidth + (cabin_width - Dmin) / (2.0 * n_aisles)

    y = [fuse_offset + seat_width / 2.0]        # the first window seat
    for i in range(2, seats_per_row + 1):
        y.append(y[-1] + seat_width
                 + _aisle_flag(i, layout) * 2.0 * exp_aisle)
    return [v - cabin_width / 2.0 for v in y]


def place_cabin_seats(pax: int, cabin_width: float,
                      seat_pitch: float = SEAT_PITCH,
                      seat_width: float = SEAT_WIDTH,
                      aisle_halfwidth: float = AISLE_HALFWIDTH,
                      fuse_offset: float = FUSE_OFFSET,
                      front_seat_offset: float = FRONT_SEAT_OFFSET) -> tuple:
    """``(lcabin, xseats, seats_per_row)``.

    ``lcabin`` is the **last seat's station**, not the back of the last row,
    so it omits that row's pitch and anything behind it.
    """
    n = seats_abreast(cabin_width, seat_width, aisle_halfwidth, fuse_offset)
    rows = math.ceil(pax / n)

    # Fixed row numbers, not derived from the passenger count.
    emergency_rows = (12, 13) if n <= 10 else (19, 20)

    xseats = [front_seat_offset]
    for r in range(2, rows + 1):
        extra = seat_pitch / 2.0 if r in emergency_rows else 0.0
        xseats.append(xseats[-1] + seat_pitch + extra)
    return xseats[-1], xseats, n


def find_cabin_width(Rfuse: float, wfb: float, nfweb: int, theta: float,
                     h_seat: float) -> float:
    """Effective cabin width, m, for a floor at angle ``theta``.

    Takes the **narrower** of the width at the floor and the width at seat
    height, because a seat needs room at both.
    """
    theta_seat = math.asin((h_seat + Rfuse * math.sin(theta)) / Rfuse)
    cos_theta = min(math.cos(theta), math.cos(theta_seat))
    return nfweb * 2.0 * wfb + 2.0 * Rfuse * cos_theta


def find_floor_angles(is_doubledecker: bool, Rfuse: float, dRfuse: float,
                      theta1: float = 0.0, h_seat: float = 0.0,
                      d_floor: float = 0.0):
    """Angular position of each deck, rad from the bubble centre.

    Single deck: the angle is *chosen* to maximise cabin width, which puts
    the floor half a seat height below centre so the circle's widest point
    lands at shoulder level. Double decker: the main deck angle is an input
    and only the upper deck is derived.
    """
    if not is_doubledecker:
        return -math.asin(h_seat / (2.0 * Rfuse))
    theta2 = math.asin((Rfuse * math.sin(theta1) + d_floor) / Rfuse)
    return theta1, theta2
