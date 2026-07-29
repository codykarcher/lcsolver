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


# --- geometric sizing ------------------------------------------------------

def _reference_design():
    """TASOPT.jl's own PMSM defaults, read off a constructed object."""
    from tasopt_py.propsys.motor import MotorDesign
    return MotorDesign(U_max=200.0, J_max=5.0e6, N_pole_pairs=8,
                       N_slots=48, N_slots_per_phase=16,
                       N_energized_slots=32, airgap_thickness=0.002,
                       magnet_thickness=0.02, magnet_M=860400.0,
                       teeth_thickness=0.02, winding_kpf=0.35)


#: What ``size_PMSM!(PMSM{Motor}(), 8000.0, 1e6)`` produces.
REF_MOTOR = dict(radius_gap=0.23873241463784303, length=0.24827400601201793,
                 mass=256.6153902181946, B_gap=0.9829186613788256,
                 A_slot=0.000365875345018842, I=640.2818537829736,
                 R=0.0008394188164472384)
REF_MASSES = dict(rotor=64.94563517235221, stator=86.45575348301549,
                  magnet=53.528860202473076, teeth=26.833726758615875,
                  windings=24.819879933423234, shaft=0.031534668314668224)


def test_the_sized_motor_matches_tasopt_jl():
    """All thirteen outputs -- the geometry, the six component masses, the
    slot current and the phase resistance -- to machine precision."""
    from tasopt_py.propsys.motor import size_motor

    m = size_motor(_reference_design(), 8000.0, 1.0e6)
    got = dict(radius_gap=m.radius_gap, length=m.length, mass=m.mass,
               B_gap=m.B_gap, A_slot=m.A_slot, I=m.I,
               R=m.phase_resistance)
    for k, want in REF_MOTOR.items():
        assert got[k] == pytest.approx(want, rel=1e-13), k
    for k, want in REF_MASSES.items():
        assert m.masses[k] == pytest.approx(want, rel=1e-13), k


def test_faster_motors_are_smaller():
    """radius_gap = U_max / Omega, so the tip-speed limit alone sets the
    diameter. That is the entire argument for high-speed machines, and it is
    why they need gearboxes."""
    from tasopt_py.propsys.motor import size_motor

    d = _reference_design()
    slow = size_motor(d, 4000.0, 1.0e6)
    fast = size_motor(d, 12000.0, 1.0e6)
    assert fast.radius_gap == pytest.approx(slow.radius_gap / 3.0, rel=1e-12)
    assert fast.mass < slow.mass


def test_the_windings_are_at_cooling_air_temperature_not_ambient():
    """363.15 K, tied to the cooling air. Worth 28% on the phase resistance
    against a 20 C assumption, and therefore 28% on the ohmic loss."""
    from tasopt_py.propsys.motor import MotorDesign, size_motor

    d = _reference_design()
    assert d.winding_T == pytest.approx(363.15)
    hot = size_motor(d, 8000.0, 1.0e6)
    cold = size_motor(MotorDesign(**{**d.__dict__, "winding_T": 293.15}),
                      8000.0, 1.0e6)
    assert hot.phase_resistance / cold.phase_resistance == pytest.approx(
        1.0 + 0.00404 * 70.0, rel=1e-12)


def test_the_three_sizing_failures_are_reported():
    """The reference errors on each of these too; this raises with a reason.
    They are the real limits of the machine, not numerical accidents."""
    from tasopt_py.propsys.motor import MotorDesign, size_motor

    d = _reference_design()

    # The shaft binds at high speed *and* high power, which is not the
    # obvious direction. A slow machine is not shaft-limited -- the radius
    # goes as 1/Omega, so slowing it down makes the bore bigger faster than
    # it makes the torque bigger. The binding group is P * Omega^2.
    assert size_motor(d, 400.0, 5.0e6).radius_gap > 4.0    # huge, but fine
    with pytest.raises(ValueError, match="shaft too thin"):
        size_motor(d, 50000.0, 2.0e7)

    # Push the tip-speed limit far enough down and the rotor bore closes
    # up -- but not before the shaft binds. At U_max = 25 it is the shaft
    # that fails; the bore only goes negative below about 19 m/s, where the
    # 20 mm magnet plus the yoke exceed the whole radius.
    with pytest.raises(ValueError, match="shaft too thin"):
        size_motor(MotorDesign(**{**d.__dict__, "U_max": 25.0}),
                   8000.0, 1.0e6)
    with pytest.raises(ValueError, match="rotor bore closes up"):
        size_motor(MotorDesign(**{**d.__dict__, "U_max": 15.0}),
                   8000.0, 1.0e6)

    # A thick tooth at high current density makes the winding's own field
    # exceed saturation, and the tooth-area expression goes imaginary.
    with pytest.raises(ValueError, match="winding's own field"):
        size_motor(MotorDesign(**{**d.__dict__, "J_max": 1.0e9}),
                   8000.0, 1.0e6)


def test_the_stator_and_rotor_dominate_the_mass():
    """Back iron, not copper. The windings are under 10% of the machine,
    which is why the flux limits matter more than the current limit."""
    from tasopt_py.propsys.motor import size_motor

    m = size_motor(_reference_design(), 8000.0, 1.0e6)
    iron = m.masses["rotor"] + m.masses["stator"] + m.masses["teeth"]
    assert iron / m.mass > 0.65
    assert m.masses["windings"] / m.mass < 0.10
    # And the specific power that falls out.
    assert 1.0e6 / m.mass / 1000.0 == pytest.approx(3.9, abs=0.2)
