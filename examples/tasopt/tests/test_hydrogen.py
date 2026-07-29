"""Hydrogen, against TASOPT.jl -- the first thing here not from the Fortran.

TASOPT 2.16 **cannot burn hydrogen at all**. This is not a matter of accuracy:

* ``gasfun.f`` defines eleven species -- N2, O2, Ar, CO2, H2O, and the
  hydrocarbons CH4 through C14H30. There is no ``gas_H2``.
* ``gaschem``, which supplies the atom counts that let ``gasfun`` balance a
  combustion reaction, ends at ``igas = 24`` with
  ``write(*,*) 'GASCHEM: undefined gas index:' ; stop``.

So a hydrogen case does not give a bad answer, it halts the program.
TASOPT.jl adds one species -- ``H2`` at ``igas = 40`` -- and that is the whole
thermodynamic difference. The table is lifted out of ``gasdata.jl`` by
``tools/gen_h2_table.py`` rather than transcribed, and checked here against
values dumped from the running Julia package (``julia_ref/dump_gas_h2.jl``).

``gas_burn`` also gains a ninth argument in v3, ``hvap`` -- the fuel's heat of
vaporisation, which a cryogenic fuel has to pay before it can burn. Zero
reproduces the Fortran.

Both agree with TASOPT.jl **bit for bit**, which is not luck: the port's cubic
Hermite interpolation is the same algorithm operating on the same table, so
anything other than an exact match would mean one of the two had been
transcribed wrong.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.gas.mixture import gas_burn, gasfuel
from tasopt_py.gas.properties import IGAS, NCHON, gasfun, gaschem
from tasopt_py.gas.tables import GASES

DATA = Path(__file__).parent / "data"
H2_REF = DATA / "gas_h2_ref.csv"
BURN_REF = DATA / "gas_burn_h2_ref.csv"

#: ``airfrac.inc``'s live composition -- N2, O2, CO2, H2O, Ar, fuel.
ALPHA = [0.7532, 0.2315, 0.0006, 0.0020, 0.0127, 0.0]
NSPEC = 6

I_H2 = 40


def test_2_16_has_eleven_species_and_this_port_now_has_twelve():
    """Stated as a fact about the source, so that if someone later 'tidies'
    H2 out of the tables the reason it is there is on the record."""
    fortran = {"N2", "O2", "CO2", "H2O", "AR",
               "CH4", "C2H6", "C3H8", "C4H10", "C8H18", "C14H30"}
    assert fortran <= set(GASES)
    assert set(GASES) - fortran == {"H2"}
    assert IGAS[I_H2] == "H2"


def test_the_table_came_from_the_generator_not_a_keyboard():
    h2 = GASES["H2"]
    assert h2["ndim"] == 45
    assert h2["r"] == 4124.9            # 8314.46 / 2.016
    assert h2["hform"] == 0.0
    for col in ("t", "tl", "cp", "cpt", "h", "s"):
        assert len(h2[col]) == 45
    # The table starts at the normal boiling point, not at 175 K like the
    # Fortran's -- which is the whole point of having it.
    assert h2["t"][0] == pytest.approx(20.369)
    assert h2["t"][-1] == 1800.0
    assert h2["tl"] == pytest.approx(
        [__import__("math").log(t) for t in h2["t"]], rel=2e-4)


@pytest.mark.skipif(not H2_REF.exists(), reason="H2 reference absent")
def test_properties_match_tasopt_jl_bit_for_bit():
    rows = list(csv.DictReader(H2_REF.open()))
    assert len(rows) == 25
    for r in rows:
        g = gasfun(I_H2, float(r["t"]))
        for name, got in (("s", g.s), ("s_t", g.s_t), ("h", g.h),
                          ("h_t", g.h_t), ("cp", g.cp), ("r", g.r)):
            assert got == float(r[name]), (r["t"], name)


@pytest.mark.skipif(not H2_REF.exists(), reason="H2 reference absent")
def test_the_reference_covers_knots_interior_and_extrapolation():
    """A table interpolator that is only checked between knots will not
    notice a wrong endpoint slope, and TASOPT extrapolates rather than
    clamping outside the tabulated range."""
    ts = [float(r["t"]) for r in csv.DictReader(H2_REF.open())]
    knots = set(GASES["H2"]["t"])
    assert any(t in knots for t in ts)                  # on a knot
    assert any(t not in knots for t in ts)              # between knots
    assert min(ts) < GASES["H2"]["t"][0]                # below the table
    assert max(ts) > GASES["H2"]["t"][-1]               # above it


def test_the_combustion_reaction_balances():
    """H2 + 1/2 O2 -> H2O, and no carbon anywhere: 1 kg of hydrogen needs
    7.937 kg of oxygen and makes 8.937 kg of water."""
    assert gaschem(I_H2) == (0, 2, 0, 0)
    assert NCHON["H2"] == (0, 2, 0, 0)

    g = gasfuel(I_H2, NSPEC)
    n2, o2, co2, h2o, ar, fuel = g
    assert o2 == pytest.approx(-7.93662, rel=1e-5)
    assert h2o == pytest.approx(8.93662, rel=1e-5)
    assert co2 == 0.0 and n2 == 0.0 and ar == 0.0
    assert sum(g) == pytest.approx(1.0)


def test_kerosene_still_makes_carbon_dioxide():
    """The obvious control -- so a hydrogen result of 'no CO2' means the
    chemistry works, not that the CO2 slot is broken."""
    g = gasfuel(24, NSPEC)              # C14H30
    assert g[2] > 3.0                   # CO2 per unit fuel mass
    assert sum(g) == pytest.approx(1.0)


@pytest.mark.skipif(not BURN_REF.exists(), reason="burn reference absent")
def test_burning_it_matches_tasopt_jl():
    rows = list(csv.DictReader(BURN_REF.open()))
    assert len(rows) == 32
    seen_h2, seen_hvap = 0, 0
    for r in rows:
        ifuel, hvap = int(r["ifuel"]), float(r["hvap"])
        beta = [0.0] * NSPEC
        beta[NSPEC - 1] = 1.0
        f, lam = gas_burn(ALPHA, beta, gasfuel(ifuel, NSPEC), NSPEC, ifuel,
                          float(r["to"]), float(r["tf"]), float(r["t"]), hvap)
        assert f == float(r["f"]), (ifuel, hvap, r["t"])
        for k in range(NSPEC):
            assert lam[k] == float(r[f"lam{k + 1}"]), (ifuel, k)
        seen_h2 += ifuel == I_H2
        seen_hvap += hvap != 0.0
    assert seen_h2 == 16 and seen_hvap == 8


def test_hvap_defaults_to_the_fortrans_behaviour():
    """2.16's gas_burn has eight arguments and assumes the fuel arrives as a
    gas. The ninth is TASOPT.jl's; omitting it must reproduce 2.16 exactly,
    which is what keeps the 737 byte-identical."""
    beta = [0.0] * NSPEC
    beta[NSPEC - 1] = 1.0
    gamma = gasfuel(24, NSPEC)
    a = gas_burn(ALPHA, beta, gamma, NSPEC, 24, 700.0, 280.0, 1400.0)
    b = gas_burn(ALPHA, beta, gamma, NSPEC, 24, 700.0, 280.0, 1400.0, 0.0)
    assert a[0] == b[0] and a[1] == b[1]


def test_vaporisation_costs_fuel():
    """Boiling the liquid is charged against the fuel, so a cryogenic fuel
    needs more of it for the same temperature rise. ~446 kJ/kg for LH2 is
    about 0.4% of its heating value -- small, and it scales with fuel flow."""
    beta = [0.0] * NSPEC
    beta[NSPEC - 1] = 1.0
    gamma = gasfuel(I_H2, NSPEC)
    dry, _ = gas_burn(ALPHA, beta, gamma, NSPEC, I_H2, 700.0, 20.4, 1400.0)
    wet, _ = gas_burn(ALPHA, beta, gamma, NSPEC, I_H2, 700.0, 20.4, 1400.0,
                      446.0e3)
    assert wet > dry
    assert (wet - dry) / dry == pytest.approx(0.004, abs=0.002)


def test_hydrogen_needs_far_less_fuel_by_mass():
    """The headline reason to care: per unit mass, hydrogen carries about
    2.8x the energy of kerosene, so the fuel/air ratio for the same burner
    temperature rise is correspondingly smaller."""
    beta = [0.0] * NSPEC
    beta[NSPEC - 1] = 1.0
    fk, _ = gas_burn(ALPHA, beta, gasfuel(24, NSPEC), NSPEC, 24,
                     700.0, 280.0, 1400.0)
    fh, _ = gas_burn(ALPHA, beta, gasfuel(I_H2, NSPEC), NSPEC, I_H2,
                     700.0, 280.0, 1400.0)
    assert 2.5 < fk / fh < 3.1
