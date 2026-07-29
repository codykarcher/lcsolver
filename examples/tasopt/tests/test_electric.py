"""Inverter and cable, against TASOPT.jl.

TASOPT 2.16 has no electrical system at all -- only shaft-power offtakes that
vanish from the cycle without going anywhere. That is a fair model of a
bleed-driven accessory and useless for an aircraft where electricity is the
propulsion.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.propsys import (Cable, Inverter, operate_inverter,
                               resistivity, size_cable, size_inverter)

DATA = Path(__file__).parent / "data"
INV = DATA / "electric_ref.csv"
CAB = DATA / "cable_ref.csv"

pytestmark = pytest.mark.skipif(not INV.exists(),
                                reason="electric references absent")


def test_inverter_matches_tasopt_jl_exactly():
    n = 0
    for r in csv.DictReader(INV.open()):
        a, b, c, d = (float(r[x]) for x in "abcd")
        inv = Inverter()
        size_inverter(inv, a, b)
        if r["kind"] == "invsize":
            assert inv.mass == float(r["r1"])
            assert inv.P_input == float(r["r2"])
            assert inv.P_input / inv.P == float(r["r3"])
        else:
            operate_inverter(inv, c * a, d)
            assert inv.P / inv.P_input == float(r["r1"])
        n += 1
    assert n == 36


def test_cable_matches_tasopt_jl_exactly():
    n = 0
    for r in csv.DictReader(CAB.open()):
        s = size_cable(Cable(), float(r["P"]), float(r["V"]), float(r["l"]))
        assert s.mass == float(r["mass"])
        assert s.R == float(r["R"])
        assert s.W == float(r["W"])
        n += 1
    assert n == 12


# --- what the models say --------------------------------------------------

def test_inverter_efficiency_falls_off_at_both_ends():
    """The 1/x term is the point of the fit: a lightly loaded inverter is
    inefficient, which a plain polynomial in load fraction would miss."""
    inv = Inverter()
    size_inverter(inv, 1.0e6, 200.0)
    etas = {frac: operate_inverter(inv, frac * 1.0e6, 200.0)
            for frac in (0.1, 0.2, 0.5, 1.0)}
    assert etas[0.1] < etas[0.2] < etas[0.5]
    # ...and the peak is interior, not at full load.
    assert etas[0.5] > etas[1.0]


def test_the_three_fit_points_are_reproduced():
    """The curve is constructed to pass through 10%, 20% and 100% exactly,
    so those are the places to check the solve rather than the shape."""
    inv = Inverter()
    size_inverter(inv, 1.0e6, 200.0)
    f_switch = 200.0 * inv.kcf
    eta100 = -2.5e-7 * f_switch + 0.995
    assert operate_inverter(inv, 1.0e6, 200.0) == pytest.approx(eta100,
                                                                rel=1e-12)
    assert operate_inverter(inv, 0.2e6, 200.0) == pytest.approx(
        eta100 - 0.0017, rel=1e-12)
    assert operate_inverter(inv, 0.1e6, 200.0) == pytest.approx(
        eta100 - 0.0097, rel=1e-12)


def test_switching_faster_costs_efficiency():
    """2.5e-7 per Hz of switching frequency, and the switching frequency is
    20x the electrical one -- so a 1 kHz fundamental costs half a point."""
    inv = Inverter()
    size_inverter(inv, 1.0e6, 200.0)
    slow = operate_inverter(inv, 1.0e6, 200.0)
    fast = operate_inverter(inv, 1.0e6, 1000.0)
    assert fast < slow
    assert slow - fast == pytest.approx(2.5e-7 * 20.0 * 800.0, rel=1e-9)


def test_inverter_mass_is_a_flat_specific_power():
    inv = Inverter()
    size_inverter(inv, 1.0e6, 200.0)
    assert inv.mass == pytest.approx(1.0e6 / 19.0e3)


def test_zero_load_is_refused():
    """The fit has a 1/x term and diverges there."""
    inv = Inverter()
    size_inverter(inv, 1.0e6, 200.0)
    with pytest.raises(ValueError, match="load fraction"):
        operate_inverter(inv, 0.0, 200.0)


def test_raising_cable_voltage_trades_conductor_against_insulation():
    """Two independent limits: conductor area from current density,
    insulation thickness from dielectric strength. So more volts means less
    copper and more plastic -- there is an optimum, not a trend."""
    a = size_cable(Cable(), 2.0e6, 540.0, 20.0)
    b = size_cable(Cable(), 2.0e6, 3000.0, 20.0)
    assert b.A_con < a.A_con                 # less current
    assert b.t_ins > a.t_ins                 # more insulation
    assert b.R > a.R                         # thinner conductor, more ohms


def test_there_is_an_optimum_cable_voltage():
    """The consequence of the trade above, checked rather than asserted:
    mass falls then rises."""
    masses = [(V, size_cable(Cable(), 2.0e6, V, 20.0).mass)
              for V in (300.0, 1000.0, 5000.0, 10000.0, 30000.0, 100000.0)]
    best = min(masses, key=lambda t: t[1])
    assert best[0] not in (300.0, 100000.0)   # interior, not on a bound
    # It is a long way up: 30 kV for a 2 MW, 20 m run. Copper dominates
    # until the insulation thickness starts to matter, which for a 10 MV/m
    # polyamide is only at tens of kilovolts.
    assert best[0] == 30000.0
    assert best[1] < 0.05 * masses[0][1]      # 50x lighter than at 300 V


def test_cable_loss_is_quadratic_in_power():
    """Ohmic, so a cable run above its design power loses disproportionately
    more -- which is why size_cable returns something that can be asked."""
    s = size_cable(Cable(), 1.0e6, 1000.0, 20.0)
    assert s.efficiency(1.0e6) > s.efficiency(2.0e6)
    loss1 = 1.0e6 * (1.0 - s.efficiency(1.0e6))
    loss2 = 2.0e6 * (1.0 - s.efficiency(2.0e6))
    assert loss2 / loss1 == pytest.approx(4.0, rel=1e-9)
    assert not s.exceeds_design(1.0e6)
    assert s.exceeds_design(2.0e6)


def test_conductor_resistivity_rises_with_temperature():
    c = Cable()
    assert resistivity(c, 293.15) == pytest.approx(1.68e-8, rel=1e-9)
    assert resistivity(c, 373.15) > resistivity(c, 293.15)
    # The default conductor temperature is 50 C, not room temperature.
    assert c.Tcon == pytest.approx(323.1)


def test_zero_voltage_is_refused():
    with pytest.raises(ValueError, match="voltage must be positive"):
        size_cable(Cable(), 1.0e6, 0.0, 20.0)
