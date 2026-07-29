"""Heat leak into a cryogenic tank, against TASOPT.jl.

This is what turns a tank from a structure into a *boil-off rate*, and so
what couples hydrogen to the mission: fuel that boils must be vented or
burned, so the aircraft carries more than it uses. TASOPT 2.16 has no tank
and therefore none of this.

All four routines agree with the reference to machine precision. The rest of
the tests are about what the model assumes, because several of those
assumptions are fixed in the source and worth having on the record.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.atmosphere import atmos
from tasopt_py.cryo.thermal import (SIGMA_SB, TREF, VACUUM_EMISSIVITY,
                                    VACUUM_PRESSURE, freestream_heat_coeff,
                                    gas_Pr, tank_heat_coeff,
                                    vacuum_resistance)

REF = Path(__file__).parent / "data" / "thermal_ref.csv"

pytestmark = pytest.mark.skipif(not REF.exists(),
                                reason="thermal reference absent")


def _rows(kind):
    return [r for r in csv.DictReader(REF.open()) if r["fn"] == kind]


def test_air_properties_match():
    for r in _rows("gasPr"):
        _, Pr, _, _, mu, k = gas_Pr(float(r["a"]))
        assert Pr == pytest.approx(float(r["r1"]), rel=1e-14)
        assert mu == pytest.approx(float(r["r2"]), rel=1e-14)
        assert k == pytest.approx(float(r["r3"]), rel=1e-14)


def test_freestream_coefficients_match():
    n = 0
    for r in _rows("fshc"):
        h, Tair, Taw = freestream_heat_coeff(float(r["a"]), 288.2,
                                             float(r["b"]), 20.0, 250.0, 1.9)
        assert h == pytest.approx(float(r["r1"]), rel=1e-14)
        assert Tair == pytest.approx(float(r["r2"]), rel=1e-14)
        assert Taw == pytest.approx(float(r["r3"]), rel=1e-14)
        n += 1
    assert n == 9                       # 3 altitudes x 3 Mach


def test_liquid_side_coefficients_match():
    for r in _rows("thc"):
        got = tank_heat_coeff(float(r["b"]), int(float(r["a"])),
                              float(r["c"]), float(r["d"]))
        assert got == float(r["r1"])


def test_vacuum_resistances_match():
    for r in _rows("vac"):
        got = vacuum_resistance(float(r["a"]), float(r["b"]),
                                float(r["c"]), float(r["d"]))
        assert got == float(r["r1"])


# --- what the model assumes -----------------------------------------------

def test_the_air_model_is_the_simple_one_not_tasopts_own_mixture():
    """`gasPr("air_simple")` uses constant cp = 1005 and R = 287.1, where
    `gasPr("air")` sums the real five-species mixture. The reference picks
    the simple one here and says why -- speed.

    That makes the tank's air properties inconsistent with the rest of
    TASOPT, which uses the full mixture everywhere: R = 287.1 against the
    286.857 that mission.py derives from sea-level conditions. A 0.08%
    difference, and it only touches the tank's air-side film coefficient,
    but it is a real seam between two parts of the same program.
    """
    R, _, gamma, cp, _, _ = gas_Pr(288.2)
    assert R == 287.1
    assert cp == 1005.0
    assert gamma == pytest.approx(cp / (cp - R))

    sl = atmos(0.0)
    R_mixture = sl.p / (sl.rho * sl.T)
    assert R_mixture == pytest.approx(286.857, abs=0.01)
    assert abs(R - R_mixture) / R_mixture == pytest.approx(8.5e-4, abs=2e-4)


def test_zero_mach_switches_to_natural_convection():
    """The switch is exact equality on M, so a taxiing aircraft at M = 0.01
    is on the forced-convection branch and a stationary one is not. The two
    do not meet in the limit -- the coefficient jumps."""
    still, _, _ = freestream_heat_coeff(0.0, 288.2, 0.0, 20.0, 250.0, 1.9)
    crawling, _, _ = freestream_heat_coeff(0.0, 288.2, 1e-6, 20.0, 250.0, 1.9)
    assert still > 0.0 and crawling > 0.0
    # Forced convection at a millionth of Mach 1 is far *below* natural.
    assert crawling < still / 100.0


def test_the_recovery_temperature_rises_with_mach():
    """Taw is what the driving temperature difference is taken against, so
    at speed the tank is fighting a hotter wall than the static air."""
    for M in (0.0, 0.3, 0.8):
        _, Tair, Taw = freestream_heat_coeff(11000.0, 288.2, M, 20.0,
                                             250.0, 1.9)
        assert Taw >= Tair
    _, Tair, Taw = freestream_heat_coeff(11000.0, 288.2, 0.8, 20.0, 250.0,
                                         1.9)
    assert Taw - Tair == pytest.approx(24.0, abs=3.0)


def test_only_two_fuels_have_liquid_side_properties():
    """Hard-wired NIST point values at 2 atm -- 120 K for methane, 20 K for
    hydrogen -- which do not move with tank pressure even though everything
    around them does."""
    assert tank_heat_coeff(100.0, 40, 20.4, 5.0) > 0.0
    assert tank_heat_coeff(150.0, 11, 111.5, 5.0) > 0.0
    with pytest.raises(ValueError, match="no liquid-side properties"):
        tank_heat_coeff(100.0, 24, 280.0, 5.0)      # kerosene


def test_the_vacuum_gap_is_not_dominated_by_radiation_alone():
    """Radiation and residual-gas conduction act in parallel, and it is worth
    knowing which one sets the answer. Radiation carries about **77%** of the
    heat at a 20 K to 300 K gap -- the larger share, but not so much larger
    that the assumed vacuum quality stops mattering.

    I guessed the gas term was negligible and it is not: a ten-times better
    vacuum raises the resistance by 26%, which is a real design lever. The
    assumed 1e-2 Pa carries a TODO in the source wondering whether it should
    be an input, and this is why it should.
    """
    import math

    import tasopt_py.cryo.thermal as T

    Tc, Th, Si, So = 20.0, 300.0, 40.0, 45.0
    eps = T.VACUUM_EMISSIVITY
    Fe = 1.0 / (1.0 / eps + Si / So * (1.0 / eps - 1.0))
    hrad = T.SIGMA_SB * Fe * (Tc ** 2 + Th ** 2) * (Tc + Th)
    R_rad = 1.0 / (hrad * Si)

    R = vacuum_resistance(Tc, Th, Si, So)
    assert R < R_rad                       # parallel paths, so lower than
    # Radiation's share of the conductance.
    assert R / R_rad == pytest.approx(0.769, abs=0.01)

    p0 = T.VACUUM_PRESSURE
    try:
        T.VACUUM_PRESSURE = p0 / 10.0
        better = vacuum_resistance(Tc, Th, Si, So)
    finally:
        T.VACUUM_PRESSURE = p0
    assert (better - R) / R == pytest.approx(0.26, abs=0.03)


def test_the_vacuum_gap_constants_are_the_sources():
    assert VACUUM_PRESSURE == 1.0e-2        # ~1e-4 Torr, Brewer (1991)
    assert VACUUM_EMISSIVITY == 0.04        # polished aluminium
    assert SIGMA_SB == 5.670374419e-8
    assert TREF == 288.2


# --- the atmosphere gained an argument ------------------------------------

def test_atmos_gained_a_hot_day_offset_that_defaults_to_2_16():
    """TASOPT 2.16's atmos.f takes altitude alone and always returns the
    standard atmosphere. v3 adds a temperature offset, because a cryogenic
    tank has to be designed for a hot day on the ground. Default zero
    reproduces the Fortran exactly, which is what keeps 737.out
    byte-identical."""
    assert atmos(3.0) == atmos(3.0, 0.0)

    std = atmos(0.0)
    hot = atmos(0.0, 15.0)
    assert hot.T == pytest.approx(std.T + 15.0)
    assert hot.p == std.p                    # pressure is untouched
    assert hot.rho < std.rho                 # ...so it thins
    assert hot.a > std.a
