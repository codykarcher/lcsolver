"""Saturated cryogenic-fuel properties, against TASOPT.jl.

Nothing in TASOPT 2.16 corresponds to this. The Fortran carries fuel as a
density and a temperature and never asks what phase it is in, which is a fair
description of kerosene and useless for a fluid that boils at 20 K.

Two things are checked here, and the second is the more interesting one.

**The values**, against ``tests/data/fuel_thermo_ref.csv``, dumped from the
running Julia package by ``julia_ref/dump_fuel_thermo.jl``. Agreement is
around 1e-13, not exact, and that is expected: this port evaluates the fits
by Horner where the Julia writes ``a*x^6 + b*x^5 + ...`` term by term, so the
two differ in association order and therefore in the last couple of bits.

**The derivatives**, which this port does not copy. TASOPT.jl writes ``ρ_p``
and ``u_p`` out by hand, term by term with the powers brought down manually;
:mod:`tasopt_py.cryo.fuel_thermo` differentiates the value polynomial
instead. Those are independent calculations, so agreeing to 1e-13 says the
reference's hand-differentiation is right -- eight polynomials of up to sixth
order, forty-odd terms, all correct.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.cryo import gas_properties, liquid_properties
from tasopt_py.cryo.fuel_thermo import (P_ATM, P_MAX_ATM, P_MIN_ATM, SPECIES,
                                        _polyder, _polyval)
from tasopt_py.cryo.fuel_thermo_fits import FITS

REF = Path(__file__).parent / "data" / "fuel_thermo_ref.csv"

pytestmark = pytest.mark.skipif(not REF.exists(),
                                reason="fuel thermo reference absent")


def rows():
    return list(csv.DictReader(REF.open()))


def _call(kind, species, p):
    f = gas_properties if kind == "gas" else liquid_properties
    return f(species, p)


def test_values_match_tasopt_jl():
    got = rows()
    assert len(got) == 48                    # 2 species x 2 phases x 12 p
    for r in got:
        s = _call(r["kind"], r["species"], float(r["p_atm"]) * P_ATM)
        for name, v in (("Tsat", s.Tsat), ("rho", s.rho), ("h", s.h),
                        ("u", s.u)):
            assert v == pytest.approx(float(r[name]), rel=1e-12), (
                r["kind"], r["species"], r["p_atm"], name)


def test_the_independently_differentiated_derivatives_agree():
    """The point of the exercise: these were not copied from the reference,
    they were differentiated from the value fits. Agreement to 1e-12 across
    48 states says TASOPT.jl's hand-written derivative expressions are
    correct."""
    for r in rows():
        s = _call(r["kind"], r["species"], float(r["p_atm"]) * P_ATM)
        assert s.rho_p == pytest.approx(float(r["rho_p"]), rel=1e-12)
        assert s.u_p == pytest.approx(float(r["u_p"]), rel=1e-12)


def test_the_derivatives_are_also_right_by_finite_difference():
    """An independent check that does not involve the reference at all, so a
    shared mistake in both would still be caught."""
    for species in SPECIES:
        for kind in ("gas", "liquid"):
            p = 2.0 * P_ATM
            dp = 1.0e-3 * P_ATM
            a = _call(kind, species, p - dp)
            b = _call(kind, species, p + dp)
            s = _call(kind, species, p)
            assert s.rho_p == pytest.approx((b.rho - a.rho) / (2 * dp),
                                            rel=1e-6)
            assert s.u_p == pytest.approx((b.u - a.u) / (2 * dp), rel=1e-6)


def test_hydrogen_boils_at_twenty_kelvin():
    """The number that makes the whole tank module necessary."""
    s = liquid_properties("H2", P_ATM)
    assert s.Tsat == pytest.approx(20.36, abs=0.05)
    assert s.rho == pytest.approx(70.86, rel=1e-3)      # ~1/12 of kerosene
    g = gas_properties("H2", P_ATM)
    assert g.Tsat == pytest.approx(s.Tsat, rel=1e-9)    # same saturation line
    assert g.rho == pytest.approx(1.3205, rel=1e-3)


def test_the_fits_are_about_one_percent_off_nist():
    """Worth pinning, because it sets what any tank result built on them can
    mean. TASOPT.jl's own unit tests compare against NIST at `atol = 1e-1` on
    density and `rtol = 1e-2` on energies for exactly this reason -- these are
    polynomial fits to the saturation line, not an equation of state.

    NIST saturated H2 at 1 atm: Tsat 20.369 K, vapour 1.3322 kg/m^3, liquid
    70.848 kg/m^3.
    """
    g = gas_properties("H2", P_ATM)
    l = liquid_properties("H2", P_ATM)
    assert abs(l.Tsat - 20.369) < 0.05                  # ~0.05% on Tsat
    assert abs(l.rho - 70.848) / 70.848 < 2e-4          # liquid is good
    # ...the vapour density is the loosest of them, ~0.9% low.
    assert 5e-3 < abs(g.rho - 1.3322) / 1.3322 < 2e-2


def test_methane_boils_much_higher_and_is_much_denser():
    s = liquid_properties("CH4", P_ATM)
    assert s.Tsat == pytest.approx(111.5, abs=1.0)
    assert s.rho == pytest.approx(422.6, rel=1e-2)
    assert s.rho > 5.0 * liquid_properties("H2", P_ATM).rho


def test_lh2_is_accepted_as_a_name_for_the_liquid():
    a = liquid_properties("LH2", P_ATM)
    b = liquid_properties("h2", P_ATM)
    assert a == b


def test_an_unknown_species_is_refused():
    """TASOPT.jl's `if` chain has no `else`, so an unknown species leaves the
    locals undefined and it errors somewhere less obvious. This says so."""
    with pytest.raises(ValueError, match="no saturation fit"):
        gas_properties("N2", P_ATM)


def test_outside_the_fit_range_is_refused():
    """A deliberate departure from the reference, which does not check. The
    fits were made on 0.1-10 atm; a sixth-order polynomial a decade outside
    that is not a physical model, and returning one silently is worse than
    refusing."""
    for p_atm in (0.05, 12.0, 40.0):
        with pytest.raises(ValueError, match="outside the"):
            liquid_properties("H2", p_atm * P_ATM)
    # ...and the endpoints themselves are fine.
    assert liquid_properties("H2", P_MIN_ATM * P_ATM).rho > 0.0
    assert liquid_properties("H2", P_MAX_ATM * P_ATM).rho > 0.0


def test_how_far_the_extrapolation_would_have_gone():
    """Quantifies why the guard is there rather than asserting a taste. At
    40 atm the unguarded liquid-density fit returns a value it should not."""
    c = FITS[("liquid", "H2")]["rho"]
    assert _polyval(c, 40.0) > 1.0e3          # denser than water, for LH2
    # At the top of the fit range it is still sane -- LH2 thins as it warms.
    assert _polyval(c, P_MAX_ATM) == pytest.approx(49.6, abs=1.0)


def test_the_fits_are_generated_with_the_orders_the_source_has():
    for key, orders in ((("gas", "H2"), {"Tsat": 6, "rho": 3, "h": 6, "u": 6}),
                        (("liquid", "H2"),
                         {"Tsat": 6, "rho": 6, "h": 6, "u": 6})):
        for name, n in orders.items():
            assert len(FITS[key][name]) - 1 == n, (key, name)


def test_polyder_is_the_derivative():
    assert _polyder([2.0, 3.0, 4.0]) == [4.0, 3.0]     # d/dx(2x^2+3x+4)
    assert _polyval(_polyder([1.0, 0.0, 0.0]), 3.0) == 6.0
