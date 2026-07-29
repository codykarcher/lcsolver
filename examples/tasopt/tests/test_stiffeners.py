"""Cryogenic-tank stiffener rings, against TASOPT.jl.

Barron, *Cryogenic Systems* (1985) ch. 7. A tank slung inside a fuselage
carries its contents through a few discrete supports; between them the shell
would ovalise, so stiffener rings hold it round and have to be sized for the
moment the support reactions put into them.

All three routines agree with the reference to machine precision or exactly.
The interesting tests are the ones about *how* the reference searches for its
maximum, because that is where the physics stops and an assumption starts.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tasopt_py.cryo.stiffeners import (PREF, find_K1_head, stiffener_weight,
                                       stiffeners_bending_moment,
                                       stiffeners_bending_moment_outer)
from tasopt_py.cryo.fuel_thermo import P_ATM

DATA = Path(__file__).parent / "data"
MOMENTS = DATA / "stiffeners_ref.csv"
K1 = DATA / "k1head_ref.csv"
WEIGHTS = DATA / "stiffw_ref.csv"

pytestmark = pytest.mark.skipif(not MOMENTS.exists(),
                                reason="stiffener references absent")


def test_bending_moments_match_tasopt_jl():
    n = 0
    for r in csv.DictReader(MOMENTS.open()):
        if r["kind"] == "inner":
            phi, k = stiffeners_bending_moment(float(r["a1"]))
        else:
            phi, k = stiffeners_bending_moment_outer(float(r["a1"]),
                                                     float(r["a2"]))
        assert phi == pytest.approx(float(r["phimax"]), rel=1e-14)
        assert k == pytest.approx(float(r["kmax"]), rel=1e-14)
        n += 1
    assert n == 25


def test_stiffener_weights_match_exactly():
    n = 0
    for r in csv.DictReader(WEIGHTS.open()):
        got = stiffener_weight(
            r["kind"], float(r["W"]), float(r["Rtank"]), float(r["perim"]),
            float(r["s_a"]), float(r["rho"]), float(r["th1"]),
            float(r["th2"]), float(r["Nstiff"]), float(r["l_cyl"]),
            float(r["E"]))
        assert got == float(r["Wstiff"])
        n += 1
    assert n == 16


def test_head_radius_parameter_matches_exactly():
    n = 0
    for r in csv.DictReader(K1.open()):
        assert find_K1_head(float(r["AR"])) == float(r["K1"])
        n += 1
    assert n == 11


# --- how the reference finds its maximum ----------------------------------

def _k_inner(theta, phi):
    """Barron Eq. (7.4) / (7.5) as the reference codes them."""
    if 0.0 <= phi < theta:
        return (0.5 * math.cos(phi) + phi * math.sin(phi)
                - (math.pi - theta) * math.sin(theta) + math.cos(theta)
                + math.cos(phi) * math.sin(theta) ** 2)
    return (0.5 * math.cos(phi) - (math.pi - phi) * math.sin(phi)
            + theta + math.cos(theta)
            + math.cos(phi) * math.sin(theta) ** 2)


def test_the_coded_moment_distribution_is_discontinuous_at_the_support():
    """§56. The two branches do not agree at ``phi = theta``: they jump by
    0.08 in k at a typical support angle and by 1.0 at 2.5 rad. A bending
    moment around a continuous ring cannot be discontinuous, so one of the
    two expressions is wrong -- which is what the reference's own TODO
    suspects about Barron Eq. (7.5).

    It is not academic. The strict ``<`` means ``phi = theta`` takes the
    upper branch, which is the larger one, and that angle is in the search
    list -- so for support angles up to about 1.4 rad **the sizing maximum
    lands exactly on the jump**.
    """
    for theta, expect in ((0.5, 0.260), (1.0, 0.159), (1.2, 0.082),
                          (2.5, 1.004)):
        lo = _k_inner(theta, theta - 1e-12)
        hi = _k_inner(theta, theta)
        assert hi - lo == pytest.approx(expect, abs=1e-3), theta

    # ...and the located maximum is that upper-branch value at phi = theta.
    for theta in (0.8, 1.0, 1.2):
        phi, k = stiffeners_bending_moment(theta)
        assert phi == theta
        assert k == pytest.approx(_k_inner(theta, theta), rel=1e-14)


def test_the_inner_search_is_narrow_by_design():
    """`philist = [theta; LinRange(1.1,1.2,21); pi]` -- 23 angles, 21 of them
    inside a 0.1 rad window, "taking advantage of known form of maximum". So
    the search is sharp near 1.15 rad and blind everywhere else, and it
    relies on the maximum being at the support angle, at pi, or in that
    window."""
    fine = 4000
    for theta in (0.8, 1.2, 1.5, 2.0, 2.5):
        _, k = stiffeners_bending_moment(theta)
        swept = max(abs(_k_inner(theta, math.pi * i / fine))
                    for i in range(fine + 1))
        # The windowed search is never far below a fine sweep...
        assert k >= swept * 0.99, theta
        # ...and can sit slightly *above* it, because it evaluates exactly at
        # phi = theta where the sweep straddles the discontinuity.
        assert k <= swept * 1.01, theta


def test_the_outer_search_sweeps_the_whole_half_circle():
    """361 points from 0 to pi, so no window assumption -- and the located
    maximum really is on that grid."""
    phi, k = stiffeners_bending_moment_outer(0.6, 2.6)
    step = math.pi / 360.0
    assert phi == pytest.approx(round(phi / step) * step, abs=1e-12)
    assert 0.0 <= phi <= math.pi
    assert k > 0.0


def test_ties_take_the_first_index_as_julia_does():
    """`findmax` returns the first maximum. It matters because the inner
    search puts the support angle and a uniform sweep in one list, so a tie
    between them is reachable."""
    from tasopt_py.cryo.stiffeners import _absmax
    assert _absmax([1.0, 2.0, 2.0, 1.0]) == (2.0, 1)
    assert _absmax([-3.0, 3.0]) == (3.0, 0)


# --- the source's own doubts and this port's departures --------------------

def test_the_reference_doubts_barrons_equation_and_uses_it_anyway():
    """`stiffeners_bendingM` carries a TODO saying Eq. (7.5) 'does not match
    the curves in Fig. 7.3' and suspects an error in Barron. Reproduced as
    written -- changing it would move every inner-tank stiffener weight away
    from the reference. This pins the value so that if anyone ever does
    correct it, the effect is visible."""
    _, k = stiffeners_bending_moment(1.2)
    assert k == pytest.approx(0.24868, abs=1e-5)


def test_an_out_of_range_head_aspect_ratio_is_refused():
    """A departure. TASOPT.jl prints 'ARtank of heads not supported' to
    stdout and returns K1 = 1.0, which is neither an interpolation nor a
    failure -- it silently puts a wrong head radius into the pressure sizing,
    in the middle of an optimisation where nobody reads stdout."""
    for ar in (0.5, 0.99, 3.01, 5.0):
        with pytest.raises(ValueError, match="outside Barron"):
            find_K1_head(ar)
    assert find_K1_head(1.0) == 0.50
    assert find_K1_head(3.0) == 1.36


def test_an_unknown_tank_type_is_refused():
    with pytest.raises(ValueError, match="inner.*outer"):
        stiffener_weight("middle", 1e5, 1.5, 9.4, 1.7e8, 2825.0, 1.0)


def test_the_two_atmospheric_constants_differ():
    """The reference carries `pref = 101320` in constants.jl and
    `p_atm = 101325` used by the fuel fits. Five pascals apart, and both are
    live. Pinned so it is not 'tidied' into one."""
    assert PREF == 101320.0
    assert P_ATM == 101325.0
    assert PREF != P_ATM


# --- physical sanity ------------------------------------------------------

def test_an_outer_ring_is_heavier_than_an_inner_one_at_equal_load():
    """The outer vessel must also resist collapse under external pressure --
    Barron Eq. (7.11) at four atmospheres -- so it needs more section."""
    common = dict(W=5.0e5, Rtank=2.2, perim=2 * math.pi * 2.2, s_a=1.7e8,
                  rho_stiff=2825.0)
    inner = stiffener_weight("inner", theta1=1.0, **common)
    outer = stiffener_weight("outer", theta1=1.0, theta2=2.2, Nstiff=2.0,
                             l_cyl=5.0, E=7.3e10, **common)
    assert outer > inner


def test_weight_grows_with_load_and_radius():
    def w(W, R):
        return stiffener_weight("inner", W, R, 2 * math.pi * R, 1.7e8,
                                2825.0, 1.2)
    assert w(5e5, 1.5) > w(1e5, 1.5)
    assert w(1e5, 2.2) > w(1e5, 1.5)
