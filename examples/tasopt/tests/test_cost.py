"""Aircraft acquisition cost, against TASOPT.jl.

TASOPT 2.16 has no cost model. v3 has two, and **both are disclaimed by their
own authors** -- "Unused and Unvetted", "not endorsed by the current dev
team", and in ``CostVal``'s case the docstring adds *"Honestly, not a clue."*

They are in different states of disrepair, and the tests say which:

* ``CostVal`` runs but hard-wires every input to a 737 MAX9.
* ``CostEst`` cannot be called at all -- it indexes turboelectric parameters
  that have been removed from the model.

The correlations underneath are real (DAPCA IV as published in Raymer, plus
Birkler for the turbine), so they are ported with their inputs as arguments.
"""
from __future__ import annotations

import pytest

from tasopt_py.cost import COST_WARNING, cost_val_baseline
from tasopt_py.cost.dapca import (COMPLEXITY, FTA, PPI_1999, PPI_2007,
                                  WRAP_RATES, airframe_cost, dapca_hours,
                                  engine_cost, weighted_weight)

#: What TASOPT.jl's CostVal(500.0) returns.
REF_500 = (1.9011798278205987e7, 5.9291545957872584e7, 2.1866609748185635e7)


def test_the_hardwired_baseline_matches_tasopt_jl_exactly():
    assert cost_val_baseline(500.0) == REF_500


def test_the_disclaimer_is_carried_not_buried():
    """The reference's own warning is part of the port's public surface, so
    it cannot be used without seeing it."""
    assert "not endorsed" in COST_WARNING
    assert "Unvetted" in COST_WARNING


def test_the_baseline_is_a_function_of_production_quantity_alone():
    """Because every other input is a literal in the reference's body. That
    is the reason the correlations are re-exposed with arguments."""
    a = cost_val_baseline(200.0)
    b = cost_val_baseline(1000.0)
    assert a != b
    # Per-aircraft cost falls with volume, for both labour pools.
    assert b[0] < a[0] and b[1] < a[1]
    # ...but the engines do not, because they are not on a learning curve
    # here at all.
    assert a[2] == b[2]


# --- what the model actually says -----------------------------------------

def test_structure_is_costed_by_complexity_not_just_mass():
    """The weighting factors span a factor of twenty. A kilogram of
    empennage costs six times a kilogram of engine and twenty times a
    kilogram of landing gear -- which is what makes the model say anything
    beyond 'heavier is dearer'."""
    assert COMPLEXITY["tails"] / COMPLEXITY["engines"] == pytest.approx(6.05,
                                                                        abs=0.1)
    assert COMPLEXITY["tails"] / COMPLEXITY["gear"] > 20.0

    base = dict(fuselage=20000.0, tails=2000.0, wing=12000.0,
                engines=8000.0, gear=1000.0, systems=900.0)
    W0 = weighted_weight(**base)
    # A tonne added to the tails costs far more than a tonne on the gear.
    heavier_tail = weighted_weight(**{**base, "tails": 3000.0})
    heavier_gear = weighted_weight(**{**base, "gear": 2000.0})
    assert heavier_tail - W0 == pytest.approx(COMPLEXITY["tails"] * 1000.0)
    assert (heavier_tail - W0) / (heavier_gear - W0) > 20.0


def test_the_learning_curves_differ_by_labour_pool():
    """Manufacturing scales as Q^0.641 so its per-aircraft cost falls
    steeply with volume; engineering is Q^0.163, nearly fixed, because it is
    largely done once."""
    lo = dapca_hours(30000.0, 855.0, 100.0)
    hi = dapca_hours(30000.0, 855.0, 1000.0)
    HE_lo, _, HM_lo, _ = lo
    HE_hi, _, HM_hi, _ = hi
    # Per aircraft.
    assert (HE_hi / 1000.0) / (HE_lo / 100.0) == pytest.approx(
        10.0 ** (0.163 - 1.0), rel=1e-9)
    assert (HM_hi / 1000.0) / (HM_lo / 100.0) == pytest.approx(
        10.0 ** (0.641 - 1.0), rel=1e-9)
    # Engineering per aircraft falls much faster than manufacturing.
    assert (HE_hi / 1000.0) / (HE_lo / 100.0) < (HM_hi / 1000.0) / (
        HM_lo / 100.0)


def test_quality_control_is_a_flat_fraction_of_manufacturing():
    _, _, HM, HQ = dapca_hours(30000.0, 855.0, 500.0)
    assert HQ == pytest.approx(0.133 * HM)


def test_birklers_engine_correlation_goes_negative_for_a_small_engine():
    """The -2228 constant. A reminder that this is a regression over a fleet
    of large turbofans, not a physical model -- it should not be applied to
    a small engine, and nothing stops you."""
    assert engine_cost(130.4, 1.0, 1804.0) > 0.0
    assert engine_cost(1.0, 0.3, 400.0) < 0.0


def test_the_avionics_adjustment_is_a_chosen_markup():
    """cA = 4/3, 'such that avionics costs are around 25% of flyaway cost'.
    It is a number picked to make an answer come out, not a correlation, and
    it multiplies the entire airframe cost."""
    a = airframe_cost(30000.0, 855.0, 500.0, 220, avionics_adj=1.0)
    b = airframe_cost(30000.0, 855.0, 500.0, 220, avionics_adj=4.0 / 3.0)
    assert b.unit_cost / a.unit_cost == pytest.approx(4.0 / 3.0)


def test_the_two_inflation_indices_are_both_live():
    """1999 and 2007 bases, both used -- the tank and fuel-cell correlations
    are 2007-based, the DAPCA ones 1999."""
    assert PPI_1999 == pytest.approx(242.8 / 144.8)
    assert PPI_2007 == pytest.approx(242.8 / 186.8)
    assert PPI_1999 > PPI_2007


def test_the_flight_test_fleet_is_fixed():
    """Six aircraft, 'assume upper end due to novelty of propulsion system
    tech' -- not an input, so it cannot be varied for a conventional
    design."""
    assert FTA == 6


def test_the_wrap_rates_are_the_sources():
    assert WRAP_RATES == {"RE": 86.0, "RT": 88.0, "RQ": 81.0, "RM": 73.0}
