"""Heat exchanger correlations, against TASOPT.jl.

A fuel-cell aircraft rejects more heat than it makes electricity, and a
cryogenic tank has hydrogen cold enough to be a heat sink -- so heat
exchangers become a sizing driver rather than an accessory. TASOPT 2.16 has
none.

The geometry is a staggered tube bank in cross flow: Blasius/Colburn inside
the tubes, Zukauskas (1987) across them, Gunter and Shaw (1945) for the
pressure drop.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tasopt_py.engine_v3.heat_exchanger import (SAFETY_FACTOR, TMIN_TUBE,
                                                colburn_j_pipe, hx_weight,
                                                nusselt_staggered,
                                                pressure_drop_staggered,
                                                tube_thickness)

REF = Path(__file__).parent / "data" / "hx_ref.csv"

pytestmark = pytest.mark.skipif(not REF.exists(),
                                reason="heat exchanger reference absent")


def _rows(kind):
    return [r for r in csv.DictReader(REF.open()) if r["kind"] == kind]


def test_all_correlations_match_tasopt_jl_exactly():
    n = 0
    for r in _rows("jpipe"):
        j, Cf = colburn_j_pipe(float(r["a"]))
        assert j == float(r["r1"]) and Cf == float(r["r2"])
        n += 1
    for r in _rows("nu"):
        assert nusselt_staggered(*(float(r[x]) for x in "abcde")) == float(
            r["r1"])
        n += 1
    for r in _rows("dp"):
        assert pressure_drop_staggered(
            *(float(r[x]) for x in "abcdefghi")) == float(r["r1"])
        n += 1
    assert n == 112


def test_the_colburn_analogy_ties_heat_to_friction():
    """j = Cf/2 exactly, which is why there is no separate internal
    heat-transfer correlation. It assumes Pr near one, and is a real
    approximation for a liquid coolant."""
    for Re in (1.0e3, 1.0e5):
        j, Cf = colburn_j_pipe(Re)
        assert j == Cf / 2.0
    # Blasius: friction falls as Re^-0.25.
    j1, _ = colburn_j_pipe(1.0e4)
    j2, _ = colburn_j_pipe(1.0e5)
    assert j1 / j2 == pytest.approx(10.0 ** 0.25, rel=1e-12)


def test_the_nusselt_branches_are_well_matched_at_their_boundaries():
    """Zukauskas switches coefficient and exponents at Re = 40, 1000 and
    2e5, and above 1000 also on the pitch ratio crossing 2 -- four
    discontinuities in all.

    Measured, every one of them is **under 1.2%**:

        Re = 40      -1.05%
        Re = 1000    +1.17%   (both C1/m and the row factor C2 switch here)
        Re = 2e5     +0.37%
        xt/xl = 2    -0.43%

    So the fit is nearly continuous, which is worth recording as a *good*
    property rather than assuming otherwise: an optimiser can cross these
    without the objective stepping visibly. Contrast the fuel cell's
    saturation-pressure splice (§71), which jumps a comparable 0.85% but in
    the *wrong direction* on a monotone quantity.
    """
    Pr, NL, xl = 0.7, 8.0, 1.25

    def step(a_args, b_args):
        a, b = nusselt_staggered(*a_args), nusselt_staggered(*b_args)
        assert a != b                       # they really are different
        return (b - a) / a

    assert step((39.9, Pr, NL, 1.5, xl),
                (40.1, Pr, NL, 1.5, xl)) == pytest.approx(-0.0105, abs=2e-3)
    assert step((999.0, Pr, NL, 1.5, xl),
                (1001.0, Pr, NL, 1.5, xl)) == pytest.approx(0.0117,
                                                            abs=2e-3)
    assert step((1.999e5, Pr, NL, 1.5, xl),
                (2.001e5, Pr, NL, 1.5, xl)) == pytest.approx(0.0037,
                                                             abs=2e-3)
    assert step((5000.0, Pr, NL, 2.49, 1.25),
                (5000.0, Pr, NL, 2.51, 1.25)) == pytest.approx(-0.0043,
                                                               abs=2e-3)


def test_the_row_count_factor_saturates():
    """C2 approaches 1 as rows are added, so a deep bank behaves like an
    infinite one -- there is little to gain past a dozen rows."""
    Pr, Re, xt, xl = 0.7, 5000.0, 1.5, 1.25
    few = nusselt_staggered(Re, Pr, 2.0, xt, xl)
    many = nusselt_staggered(Re, Pr, 20.0, xt, xl)
    lots = nusselt_staggered(Re, Pr, 100.0, xt, xl)
    assert few < many < lots
    assert (lots - many) / many < 0.01          # already saturated


def test_pressure_drop_switches_from_laminar_to_turbulent_at_re_200():
    """f = 90/Re below, 0.96 Re^-0.145 above. Unlike the Nusselt switches,
    these two nearly meet -- 0.45 against 0.44 at Re = 200."""
    args = dict(G=100.0, L=0.3, rho=1.2, Dv=0.01, tD_o=0.006,
                xt_D=1.5, xl_D=1.25, mu_ratio=1.0)
    below = pressure_drop_staggered(Re=199.9, **args)
    above = pressure_drop_staggered(Re=200.1, **args)
    assert abs(above - below) / below < 0.03    # nearly continuous


def test_pressure_drop_is_quadratic_in_mass_velocity():
    args = dict(Re=5000.0, L=0.3, rho=1.2, Dv=0.01, tD_o=0.006,
                xt_D=1.5, xl_D=1.25, mu_ratio=1.0)
    a = pressure_drop_staggered(G=100.0, **args)
    b = pressure_drop_staggered(G=200.0, **args)
    assert b / a == pytest.approx(4.0, rel=1e-12)


# --- tube sizing and weight -----------------------------------------------

def test_the_hoop_stress_expression_has_a_pole():
    """t goes as C K / (K - 2 K C)^2 with C = SF dp / (2 YTS), which
    diverges as C approaches 0.5 -- that is, as the design pressure
    approaches the yield stress. The reference does not check; this does."""
    YTS = 3.0e8
    assert tube_thickness(50.0, 1.0e6, YTS) >= TMIN_TUBE
    # Walk C toward 0.5 and watch it blow up.
    thin = tube_thickness(50.0, 0.10 * YTS, YTS)
    thick = tube_thickness(50.0, 0.45 * YTS, YTS)
    assert thick > 10.0 * thin
    with pytest.raises(ValueError, match="pole at C = 0.5"):
        tube_thickness(50.0, 0.5 * YTS, YTS)
    assert SAFETY_FACTOR == 2.0


def test_the_minimum_thickness_floors_a_lightly_loaded_tube():
    """30 BWG, from Brewer (1991). A low-pressure exchanger is set by
    manufacturability, not stress."""
    assert tube_thickness(50.0, 1.0e3, 3.0e8) == TMIN_TUBE
    assert TMIN_TUBE == 3.0e-4


def test_weight_counts_tube_metal_only():
    """No coolant inventory, no headers as geometry, no fins -- so it is a
    lower bound on installed mass."""
    W = hx_weight(N_tubes_tot=2000.0, tD_o=0.006, tD_i=0.0054, length=0.4,
                  rho=2700.0, fouter=0.2)
    V = 2000.0 * math.pi * (0.006 ** 2 - 0.0054 ** 2) / 4.0 * 0.4
    assert W == pytest.approx(9.81 * 2700.0 * V * 1.2)


def test_a_shaft_through_the_exchanger_adds_its_own_weight():
    """An exchanger in the core wraps the shaft and forces it longer."""
    plain = hx_weight(2000.0, 0.006, 0.0054, 0.4, 2700.0, 0.2)
    with_shaft = hx_weight(2000.0, 0.006, 0.0054, 0.4, 2700.0, 0.2,
                           shaft=(0.3, 0.08, 7850.0))
    added = with_shaft - plain
    assert added == pytest.approx(9.81 * 7850.0 * 0.3 * 0.08 ** 2
                                  * math.pi / 4.0)
    assert added > 0.0
