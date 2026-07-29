"""Saturated mixture and tank pressure evolution, against TASOPT.jl.

A cryogenic tank holds liquid *and* its vapour on the saturation line, and
the split between them is what makes tank pressure a state variable. Heat
boils liquid to vapour, vapour takes far more room than the liquid it came
from, the pressure rises. Nothing in TASOPT 2.16 corresponds to any of it.

32 mixtures and 64 pressure-derivative states, all exact.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.cryo.fuel_thermo import P_ATM
from tasopt_py.cryo.mixture import (convert_beta_same_rho, from_p_beta)
from tasopt_py.cryo.pressure import (dbeta_dt, dp_dt, mdot_boiloff,
                                     venting_mass_flow)

DATA = Path(__file__).parent / "data"
MIX = DATA / "mixture_ref.csv"
BETA = DATA / "betaconv_ref.csv"
PRESS = DATA / "pressure_ref.csv"

pytestmark = pytest.mark.skipif(not MIX.exists(),
                                reason="mixture references absent")

FIELDS = ("x", "T", "rho", "h", "u", "u_p", "phi", "rho_star", "hvap")


def test_mixtures_match_tasopt_jl_exactly():
    n = 0
    for r in csv.DictReader(MIX.open()):
        m = from_p_beta(r["species"], float(r["p_atm"]) * P_ATM,
                        float(r["beta"]))
        for name in FIELDS:
            assert getattr(m, name) == float(r[name]), (r["species"], name)
        n += 1
    assert n == 32


def test_beta_conversions_match():
    n = 0
    for r in csv.DictReader(BETA.open()):
        b = convert_beta_same_rho(r["species"], float(r["p_atm"]) * P_ATM,
                                  float(r["p0_atm"]) * P_ATM,
                                  float(r["beta0"]))
        assert b == pytest.approx(float(r["beta"]), rel=1e-14)
        n += 1
    assert n == 16


def test_pressure_derivatives_match_exactly():
    n = 0
    for r in csv.DictReader(PRESS.open()):
        m = from_p_beta(r["species"], float(r["p_atm"]) * P_ATM,
                        float(r["beta"]))
        Q, W, md, xo, mv, xv, V, al = (
            float(r[k]) for k in ("Q", "W", "mdot", "xout", "mvent",
                                  "xvent", "V", "alpha"))
        dp = dp_dt(m, Q, W, md, xo, mv, xv, V, al)
        db = dbeta_dt(m, dp, md + mv, V)
        assert dp == float(r["dpdt"])
        assert db == float(r["dbdt"])
        assert venting_mass_flow(m, Q, W, md, xo, xv) == float(
            r["mvent_req"])
        assert mdot_boiloff(m, db, dp, md * (1.0 - xo), V) == float(
            r["mboil"])
        n += 1
    assert n == 64


# --- what the model says --------------------------------------------------

def test_volume_and_mass_fractions_are_wildly_different_for_hydrogen():
    """beta is by volume, x is by mass, and saturated LH2 vapour is 53 times
    less dense than the liquid. At 95% liquid by volume the vapour is under
    0.1% by mass. Confusing the two is the obvious way to get this wrong."""
    m = from_p_beta("H2", P_ATM, 0.95)
    assert m.beta == 0.95
    assert m.x < 1.0e-3
    assert m.liquid.rho / m.gas.rho == pytest.approx(53.7, abs=1.0)


def test_venting_vapour_is_far_more_effective_than_drawing_liquid():
    """Everything turns on hvap*(x + rho_star). rho_star is 0.019 for LH2 at
    one atmosphere, so venting vapour removes about fifty times more of the
    pressure-raising phase per kilogram than drawing liquid off does."""
    m = from_p_beta("H2", P_ATM, 0.95)
    assert m.rho_star == pytest.approx(0.019, abs=0.003)
    liquid_route = m.hvap * (0.0 + m.rho_star)
    vapour_route = m.hvap * (1.0 + m.rho_star)
    assert vapour_route / liquid_route > 40.0


def test_heat_raises_pressure_and_drawing_fuel_lowers_it():
    m = from_p_beta("H2", P_ATM, 0.95)
    heated = dp_dt(m, 2600.0, 0.0, 0.0, 0.0, 0.0, 1.0, 50.0)
    assert heated > 0.0
    # Venting vapour against the same heat load turns it around.
    vented = dp_dt(m, 2600.0, 0.0, 0.0, 0.0, 0.02, 1.0, 50.0)
    assert vented < heated


def test_the_venting_rate_holds_the_pressure_constant():
    """The definition, checked rather than assumed: vent at exactly the rate
    venting_mass_flow returns and dp/dt should vanish."""
    m = from_p_beta("H2", P_ATM, 0.95)
    Q, W, md, xo, xv, V = 2600.0, 0.0, 0.02, 0.0, 1.0, 50.0
    mv = venting_mass_flow(m, Q, W, md, xo, xv)
    assert mv > 0.0
    assert dp_dt(m, Q, W, md, xo, mv, xv, V) == pytest.approx(0.0, abs=1e-12)


def test_no_venting_is_needed_when_the_draw_already_cools_the_tank():
    """Clamped at zero rather than going negative -- a negative vent rate
    would mean sucking vapour back in."""
    m = from_p_beta("H2", P_ATM, 0.95)
    # Draw a lot of vapour off against a small heat load.
    assert venting_mass_flow(m, 10.0, 0.0, 1.0, 1.0, 1.0) == 0.0


def test_a_full_tank_is_refused():
    """At beta = 1 there is no vapour and the quality is undefined -- the
    reference divides by (1 - beta) with nothing stopping it."""
    with pytest.raises(ValueError, match="liquid fill fraction"):
        from_p_beta("H2", P_ATM, 1.0)
    assert from_p_beta("H2", P_ATM, 0.999).x > 0.0


def test_a_sealed_tank_conserves_density_when_pressure_changes():
    """convert_beta_same_rho's contract: same mass, same volume, so the bulk
    density is fixed and the fill fraction has to move."""
    rho0 = from_p_beta("H2", P_ATM, 0.95).rho
    b = convert_beta_same_rho("H2", 2.0 * P_ATM, P_ATM, 0.95)
    m = from_p_beta("H2", 2.0 * P_ATM, b)
    assert m.rho == pytest.approx(rho0, rel=1e-9)
    assert b > 0.95          # the liquid thins, so more of it is needed


def test_a_mixture_is_immutable_and_moved_by_value():
    """The reference mutates in place (`update_pβ!`); this returns a new
    value. Sharing a mutable state between an integrator's stages is a good
    way to get a wrong derivative."""
    m = from_p_beta("H2", P_ATM, 0.95)
    n = m.at(2.0 * P_ATM, 0.9)
    assert m.p == P_ATM and m.beta == 0.95      # unchanged
    assert n.p == 2.0 * P_ATM and n.beta == 0.9
    with pytest.raises(Exception):
        m.p = 5.0
