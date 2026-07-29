"""Scaled fuselage cross-sections, against TASOPT.jl -- and against 2.16.

A cryogenic tank takes the fuselage's shape scaled down, so the tank sizing
needs perimeter and area at an arbitrary radius. v3 factors that into an
object; 2.16 writes the same area expression out inline in three places.

Both are checked here. The v3 check says the port matches the reference; the
2.16 check says the reference matches the Fortran the rest of this port is
verified against, which is what makes it safe to use one class for both.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tasopt_py.cryo.geometry import CrossSection, scaled_cross_section

REF = Path(__file__).parent / "data" / "xsection_ref.csv"

pytestmark = pytest.mark.skipif(not REF.exists(),
                                reason="cross-section reference absent")


def test_scaled_sections_match_tasopt_jl_exactly():
    n = 0
    for r in csv.DictReader(REF.open()):
        cs = CrossSection(float(r["R"]), float(r["dR"]), float(r["wfb"]),
                          int(r["nwebs"]))
        p, a = scaled_cross_section(cs, float(r["Rscale"]))
        assert p == float(r["perim"]), (r["kind"], r["Rscale"])
        assert a == float(r["area"]), (r["kind"], r["Rscale"])
        n += 1
    assert n == 24


def test_it_agrees_with_2_16s_inline_area():
    """The Fortran writes this out in fusew.f, wsize.f and aswout.f:

        thetafb = asin(wfb/Rfuse)
        sin2t   = 2*sin(thetafb)*cos(thetafb)
        Afuse   = (pi + 2*thetafb + sin2t)*Rfuse**2 + 2*Rfuse*dRfuse

    which is this class at n_webs = 1. If these ever diverge, one of the two
    ports is wrong.
    """
    for Rfuse, dRfuse, wfb in ((1.9558, 0.38, 0.6), (2.2, 0.0, 0.9),
                               (1.75, 0.5, 0.45)):
        cs = CrossSection(Rfuse, dRfuse, wfb, n_webs=1)
        thetafb = math.asin(wfb / Rfuse)
        hfb = math.sqrt(Rfuse ** 2 - wfb ** 2)
        sin2t = 2.0 * (wfb / Rfuse) * (hfb / Rfuse)
        inline = ((math.pi + 2.0 * thetafb + sin2t) * Rfuse ** 2
                  + 2.0 * Rfuse * dRfuse)
        assert cs.area == pytest.approx(inline, rel=1e-14)


def test_a_circular_section_is_a_circle():
    cs = CrossSection(radius=2.0)
    assert cs.area == pytest.approx(math.pi * 4.0)
    assert cs.perimeter == pytest.approx(4.0 * math.pi)
    assert cs.theta_web == 0.0 and cs.sin2theta == 0.0


def test_a_downward_shift_adds_a_rectangle():
    """dRfuse stretches the section vertically: +2 R dR of area and +2 dR of
    perimeter, exactly."""
    a = CrossSection(radius=2.0)
    b = CrossSection(radius=2.0, bubble_lower_downward_shift=0.5)
    assert b.area - a.area == pytest.approx(2.0 * 2.0 * 0.5)
    assert b.perimeter - a.perimeter == pytest.approx(2.0 * 0.5)


def test_scaling_preserves_similarity():
    """Every length scales together, so area is exactly quadratic in R and
    perimeter exactly linear -- for the double bubble as much as the circle.
    That is worth asserting rather than assuming: scaling the radius but not
    the bubble offset would change the web angle and break both."""
    cs = CrossSection(1.9, 0.38, 0.6, n_webs=1)
    p0, a0 = scaled_cross_section(cs, 1.9)
    for k in (0.5, 0.8, 1.2, 2.0):
        p, a = scaled_cross_section(cs, 1.9 * k)
        assert p == pytest.approx(p0 * k, rel=1e-14)
        assert a == pytest.approx(a0 * k ** 2, rel=1e-14)


def test_more_webs_enclose_more_area():
    R, dR, wfb = 2.2, 0.5, 0.9
    areas = [CrossSection(R, dR, wfb, n).area for n in (0, 1, 2)]
    assert areas[0] < areas[1] < areas[2]


def test_degenerate_sections_are_refused():
    with pytest.raises(ValueError, match="radius must be positive"):
        CrossSection(radius=0.0)
    with pytest.raises(ValueError, match="exceeds radius"):
        CrossSection(radius=1.0, bubble_center_y_offset=1.5, n_webs=1)
    # ...but an offset larger than the radius is harmless with no webs,
    # because nothing takes its arcsine.
    assert CrossSection(radius=1.0, bubble_center_y_offset=1.5).area > 0.0
