"""PEM fuel cell, against TASOPT.jl.

The piece that makes a hydrogen-*electric* aircraft close: it turns the
hydrogen the cryogenic tank carries into electricity, without a turbine.
Nothing in TASOPT 2.16 corresponds to any of it.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.engine_v3.fuelcell import (T_VAP, V_HEAT, cell_voltage_simple,
                                          conductivity_nafion,
                                          conductivity_pbi, lambda_water,
                                          power_density, stack_operate,
                                          stack_size, stack_weight,
                                          water_sat_pressure)

REF = Path(__file__).parent / "data" / "fuelcell_ref.csv"

pytestmark = pytest.mark.skipif(not REF.exists(),
                                reason="fuel cell reference absent")


def _rows(kind):
    return [r for r in csv.DictReader(REF.open()) if r["kind"] == kind]


def test_everything_matches_tasopt_jl():
    for r in _rows("volt"):
        assert cell_voltage_simple(*(float(r[x]) for x in "abcd")) == float(
            r["r1"])
    for r in _rows("psat"):
        assert water_sat_pressure(float(r["a"])) == float(r["r1"])
    for r in _rows("nafion"):
        assert conductivity_nafion(float(r["a"]),
                                   float(r["b"])) == float(r["r1"])
    for r in _rows("pbi"):
        assert conductivity_pbi(float(r["a"]), float(r["b"]),
                                float(r["c"])) == pytest.approx(
            float(r["r1"]), rel=1e-14)
    for r in _rows("lam"):
        assert lambda_water(float(r["a"])) == float(r["r1"])
    for r in _rows("weight"):
        assert stack_weight(float(r["a"]), float(r["b"]), 2.5e-5, 3e-4,
                            3e-4, float(r["c"])) == float(r["r1"])


# --- what the polarisation model says -------------------------------------

def test_voltage_falls_with_current_and_the_losses_dominate_differently():
    """Activation loss is logarithmic, so it bites hardest at low current;
    ohmic is linear; concentration diverges near the limiting current."""
    T, pH2, pair = 353.0, 2.0e5, 1.5e5
    vs = [cell_voltage_simple(j, T, pH2, pair)
          for j in (500.0, 5000.0, 15000.0, 19000.0)]
    assert vs[0] > vs[1] > vs[2] > vs[3]
    # A real cell voltage, well below the 1.229 V reversible value.
    assert 0.5 < vs[1] < 0.9


def test_power_density_has_a_maximum_which_is_why_there_are_two_roots():
    """The voltage falls faster than the current rises past a point, so jV
    peaks and collapses. Sizing a stack for a given power therefore has two
    solutions, and only the lower current is physical."""
    T, pH2, pair = 353.0, 2.0e5, 1.5e5
    peak = max(power_density(j, T, pH2, pair)
               for j in range(1000, 19900, 100))
    assert power_density(1000.0, T, pH2, pair) < peak
    assert power_density(19800.0, T, pH2, pair) < peak
    # Two distinct currents give the same power, either side of the peak.
    target = 0.9 * peak
    low = [j for j in range(1000, 19900, 50)
           if power_density(j, T, pH2, pair) > target]
    assert low[0] < low[-1]


def test_the_off_design_solve_takes_the_low_current_root():
    """The reference starts find_zero from 1 A/m^2 with a comment warning
    that the wrong root is reachable. This brackets below the peak, which
    cannot land on the wrong side."""
    T, pH2, pair = 353.0, 2.0e5, 1.5e5
    n_cells, A_cell, _ = stack_size(1.0e6, 800.0, 8000.0, T, pH2, pair)
    V_stack, Q = stack_operate(1.0e6, n_cells, A_cell, T, pH2, pair)
    assert V_stack == pytest.approx(800.0, rel=1e-6)
    assert Q > 0.0
    # Half power gives a *higher* stack voltage -- less current, less loss.
    V_half, _ = stack_operate(0.5e6, n_cells, A_cell, T, pH2, pair)
    assert V_half > V_stack


def test_asking_for_more_power_than_the_cell_can_make_is_refused():
    T, pH2, pair = 353.0, 2.0e5, 1.5e5
    n_cells, A_cell, _ = stack_size(1.0e6, 800.0, 8000.0, T, pH2, pair)
    with pytest.raises(ValueError, match="exceeds the cell"):
        stack_operate(1.0e8, n_cells, A_cell, T, pH2, pair)


def test_a_fuel_cell_makes_more_heat_than_electricity_at_low_voltage():
    """The gap between the thermoneutral voltage and the operating one is
    heat. At 0.7 V per cell with V_heat = 1.482, more than half the enthalpy
    becomes heat -- which is why a fuel-cell aircraft needs the heat
    exchangers."""
    T, pH2, pair = 353.0, 2.0e5, 1.5e5
    n_cells, A_cell, Q = stack_size(1.0e6, 800.0, 12000.0, T, pH2, pair)
    V_cell = 800.0 / n_cells
    assert V_cell < 0.75
    assert Q > 1.0e6                    # more heat than the 1 MW electrical
    assert Q / 1.0e6 == pytest.approx(V_HEAT["LT-PEMFC"] / V_cell - 1.0,
                                      rel=1e-6)


def test_the_product_phase_switch_nearly_cancels_itself():
    """Above 383.15 K the water leaves as vapour: the reversible voltage
    drops from 1.229 to 1.185 V *and* the reaction entropy changes by a
    factor of nearly four, from -163.23 to -44.34 J/mol/K.

    Those two nearly cancel at the switch temperature. The 44 mV drop in E0
    is offset by 52 mV of recovered temperature derating, leaving a step of
    only 8 mV -- and it goes *up*, not down. Worth pinning: the model is far
    better behaved across the phase change than picking either constant in
    isolation would suggest, and a reader checking only E0 would expect the
    opposite sign.
    """
    j, pH2, pair = 5000.0, 2.0e5, 1.5e5
    below = cell_voltage_simple(j, T_VAP - 0.01, pH2, pair)
    above = cell_voltage_simple(j, T_VAP + 0.01, pH2, pair)
    assert above > below                        # up, not down
    assert above - below == pytest.approx(0.0083, abs=0.002)

    # The two contributions, separately, to show the cancellation.
    from tasopt_py.engine_v3.fuelcell import FARADAY
    dT = T_VAP - 298.15
    liquid = 1.229 + (-163.23) / (2 * FARADAY) * dT
    vapour = 1.185 + (-44.34) / (2 * FARADAY) * dT
    assert 1.185 - 1.229 == pytest.approx(-0.044)          # E0 falls
    assert vapour - liquid == pytest.approx(0.0083, abs=0.002)


def test_higher_pressure_helps_but_only_logarithmically():
    """Nernst, so doubling the pressure is worth tens of millivolts."""
    j, T = 5000.0, 353.0
    lo = cell_voltage_simple(j, T, 1.5e5, 1.0e5)
    hi = cell_voltage_simple(j, T, 3.0e5, 2.0e5)
    assert hi > lo
    assert 0.01 < hi - lo < 0.06


# --- three things in the reference worth knowing --------------------------

def test_the_saturation_pressure_is_discontinuous_at_100_C():
    """§71. Two correlations spliced at 100 C -- Huang (2018) below, Jiao
    and Li (2010) above -- and they do not meet. The pressure *drops* by
    860 Pa as the temperature rises through the splice."""
    just_below = water_sat_pressure(373.1499)
    just_above = water_sat_pressure(373.15)
    assert just_below > just_above
    assert just_below - just_above == pytest.approx(860.0, abs=50.0)
    # Both branches are individually monotone, so it is the splice.
    assert water_sat_pressure(372.0) < water_sat_pressure(373.0)
    assert water_sat_pressure(374.0) < water_sat_pressure(375.0)


def test_nafion_conductivity_goes_negative_when_dry():
    """The linear fit crosses zero at lambda = 0.634 and keeps going. A dry
    membrane is very resistive, not conductive in reverse. The reference
    does not guard it, and neither does this port -- but it is pinned."""
    assert conductivity_nafion(350.0, 2.0) > 0.0
    assert conductivity_nafion(350.0, 0.634) == pytest.approx(0.0, abs=1e-3)
    assert conductivity_nafion(350.0, 0.0) < 0.0


def test_lambda_calc_has_an_unreachable_branch():
    """§72. The reference writes

        if a < 1 ... elseif a >= 1 ... else (extrapolated for robustness
        if a < 1) ...

    The third branch is commented as handling `a < 1`, which the *first*
    branch already caught, so it can never run. The intended guard was
    presumably for negative activity, and there is none."""
    # The two live branches.
    assert lambda_water(0.5) == pytest.approx(0.043 + 17.81 * 0.5
                                              - 39.85 * 0.25 + 36.0 * 0.125)
    assert lambda_water(2.0) == pytest.approx(14.0 + 1.4)
    # A negative activity falls through the cubic, which the "robustness"
    # branch was meant to catch and does not.
    assert lambda_water(-0.5) < 0.0
