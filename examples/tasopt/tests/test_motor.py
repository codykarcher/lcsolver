"""Permanent-magnet motor losses, against TASOPT.jl.

Nothing in TASOPT 2.16 corresponds to any of this -- the Fortran has no
electrical machine of any kind.

The four loss mechanisms scale differently, and that is the whole design
story: ohmic loss falls with speed at fixed power while core and windage
losses rise, windage as the *cube*. So a motor has a best speed rather than
simply wanting to be fast.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tasopt_py.propsys.motor import (MU_0, STEELS, _solve_friction,
                                     airgap_flux, core_loss,
                                     cross_sectional_area, eddy_loss,
                                     hysteresis_loss, ohmic_loss,
                                     remanent_flux, slot_resistance,
                                     windage_loss)

REF = Path(__file__).parent / "data" / "motor_ref.csv"

pytestmark = pytest.mark.skipif(not REF.exists(),
                                reason="motor reference absent")

KH, KE, ALPHA, RHO_STEEL = STEELS["M19"]


def _rows(kind):
    return [r for r in csv.DictReader(REF.open()) if r["kind"] == kind]


def test_the_steel_constants_are_the_references():
    r = _rows("steel")[0]
    assert (float(r["a"]), float(r["b"]), float(r["c"]),
            float(r["d"])) == (KH, KE, ALPHA, RHO_STEEL)
    # The Steinmetz exponent is not 2 -- which is why hysteresis and eddy
    # losses are written with different powers of B.
    assert ALPHA == pytest.approx(1.793)


def test_all_loss_models_match_tasopt_jl():
    for r in _rows("airgap"):
        assert airgap_flux(float(r["a"]), float(r["b"]),
                           float(r["c"])) == float(r["r1"])
    for r in _rows("ohmic"):
        assert ohmic_loss(float(r["a"]), float(r["b"]),
                          int(float(r["c"]))) == float(r["r1"])
    for r in _rows("hyst"):
        assert hysteresis_loss(float(r["a"]), float(r["b"]), float(r["c"]),
                               KH, ALPHA) == float(r["r1"])
    for r in _rows("eddy"):
        assert eddy_loss(float(r["a"]), float(r["b"]), float(r["c"]),
                         KE) == float(r["r1"])
    for r in _rows("windage"):
        got = windage_loss(float(r["a"]), float(r["b"]), float(r["c"]),
                           0.3, 1.225, 1.5e-5)
        assert got == pytest.approx(float(r["r1"]), rel=1e-13)


# --- what the loss scalings mean ------------------------------------------

def test_the_loss_exponents_are_what_the_design_turns_on():
    """Ohmic falls with speed at fixed power; core and windage rise. That
    trade is why a motor has a best speed."""
    mass, B = 30.0, 1.4
    slow, fast = 200.0, 800.0

    # Core loss rises: hysteresis linearly, eddy quadratically.
    assert (hysteresis_loss(mass, fast, B, KH, ALPHA)
            / hysteresis_loss(mass, slow, B, KH, ALPHA)
            == pytest.approx(fast / slow))
    assert (eddy_loss(mass, fast, B, KE) / eddy_loss(mass, slow, B, KE)
            == pytest.approx((fast / slow) ** 2))

    # Windage rises faster still -- the cube, before the friction
    # coefficient's own weak decline.
    w1 = windage_loss(500.0, 0.2, 0.003, 0.3, 1.225, 1.5e-5)
    w2 = windage_loss(1000.0, 0.2, 0.003, 0.3, 1.225, 1.5e-5)
    # Just under 2^3: the Omega^3 is exact, but doubling the speed also
    # doubles the gap Reynolds number, and the friction coefficient falls
    # with it (0.00457 -> 0.00396). So the nominal cube is softened.
    assert 6.9 < w2 / w1 < 7.0

    # Ohmic falls: at fixed power, twice the speed is half the torque and
    # half the current, so a quarter of the loss.
    assert (ohmic_loss(50.0, 1e-3) / ohmic_loss(100.0, 1e-3)
            == pytest.approx(0.25))


def test_windage_goes_as_the_fourth_power_of_radius():
    a = windage_loss(1000.0, 0.15, 0.003, 0.3, 1.225, 1.5e-5)
    b = windage_loss(1000.0, 0.30, 0.003, 0.3, 1.225, 1.5e-5)
    # Just under 2^4, and for the same reason as the speed case: the radius
    # enters the Reynolds number too, so Cf falls as the rotor grows.
    assert 13.9 < b / a < 14.0


def test_the_airgap_flux_has_a_ceiling():
    """A magnetic-circuit result: however thick the magnet, B_gap cannot
    exceed mu0 M. So there is no point making magnets arbitrarily thick."""
    M = 1.0e6
    ceiling = MU_0 * M
    for thickness in (0.005, 0.02, 0.1, 10.0):
        assert airgap_flux(M, thickness, 0.003) < ceiling
    assert airgap_flux(M, 10.0, 0.003) == pytest.approx(ceiling, rel=1e-3)


def test_a_bigger_airgap_costs_flux():
    assert airgap_flux(1.0e6, 0.02, 0.002) > airgap_flux(1.0e6, 0.02, 0.006)


def test_hot_magnets_lose_field():
    """Neodymium loses about 0.1%/K, so a motor 100 K above its base
    temperature has lost a tenth of its field."""
    Br = 1.2
    cold = remanent_flux(Br, 0.1, 293.15, Tbase=20.0)
    hot = remanent_flux(Br, 0.1, 393.15, Tbase=20.0)
    assert cold == pytest.approx(Br)
    assert (cold - hot) / cold == pytest.approx(0.10, abs=0.005)


def test_annulus_area_and_slot_resistance():
    assert cross_sectional_area(0.2, 0.1) == pytest.approx(
        math.pi * (0.04 - 0.01))
    assert slot_resistance(1.68e-8, 1.0e-4, 0.5) == pytest.approx(8.4e-5)


# --- the reference's windage solve fails where this one does not ----------

def test_the_bracketed_friction_solve_works_where_the_references_does_not():
    """§67. Vrancik's correlation is implicit in Cf and has a logarithm, so
    the residual is undefined at Cf <= 0. `find_zero(res, 1e-2)` -- what the
    source uses -- steps negative and throws a DomainError for gap Reynolds
    numbers above roughly 30000.

    That is not an exotic corner. A 1500 rad/s rotor (about 14,000 rpm) with
    a 0.15 m gap radius is Re = 30000, which is an ordinary aircraft-scale
    motor. Five of the eight cases in the reference dump are past it, and the
    dump had to be taken with a bracketed solve to produce them at all.
    """
    for Re in (1.0e4, 3.0e4, 1.0e5, 1.0e6, 1.0e7):
        Cf = _solve_friction(Re)
        assert 1.0e-4 < Cf < 1.0e-1
        # It really solves Vrancik's equation.
        assert (1.0 / math.sqrt(Cf) - 2.04
                - 1.768 * math.log(Re * math.sqrt(Cf))) == pytest.approx(
                    0.0, abs=1e-9)

    # Friction falls with Reynolds number, as it should.
    assert _solve_friction(1.0e4) > _solve_friction(1.0e6)


def test_a_stationary_rotor_is_refused():
    """Re = 0 and Vrancik takes its logarithm."""
    with pytest.raises(ValueError, match="Reynolds"):
        windage_loss(0.0, 0.2, 0.003, 0.3, 1.225, 1.5e-5)


def test_core_loss_sums_the_two_mechanisms():
    m, f, B = 30.0, 400.0, 1.4
    assert core_loss(m, f, B) == pytest.approx(
        hysteresis_loss(m, f, B, KH, ALPHA) + eddy_loss(m, f, B, KE))
    # Which dominates depends on frequency -- eddy wins at high speed.
    lo_h = hysteresis_loss(m, 100.0, B, KH, ALPHA)
    lo_e = eddy_loss(m, 100.0, B, KE)
    hi_h = hysteresis_loss(m, 2000.0, B, KH, ALPHA)
    hi_e = eddy_loss(m, 2000.0, B, KE)
    assert lo_h > lo_e
    assert hi_e > hi_h
