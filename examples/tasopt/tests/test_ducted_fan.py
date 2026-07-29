"""Ducted fan weight, against TASOPT.jl.

An electrically driven fan and nacelle with no core -- what a fuel cell or
battery drives. No counterpart in TASOPT 2.16, whose only engine is a
turbofan.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tasopt_py.engine_v3.ducted_fan import (AR_FAN, BLADE_SOLIDITY, KTECH,
                                            ducted_fan_weight)

REF = Path(__file__).parent / "data" / "dfan_ref.csv"

pytestmark = pytest.mark.skipif(not REF.exists(),
                                reason="ducted fan reference absent")


def test_weights_match_tasopt_jl_exactly():
    n = 0
    for r in csv.DictReader(REF.open()):
        g = ducted_fan_weight(float(r["Dfan"]), float(r["Nmech"]),
                              float(r["neng"]), float(r["rSnace"]),
                              float(r["fpylon"]))
        assert g.Weng == float(r["Weng"])
        assert g.Wnac == float(r["Wnac"])
        assert g.Webare == float(r["Webare"])
        assert g.Snace1 == float(r["Snace1"])
        n += 1
    assert n == 48


def test_the_nacelle_is_counted_about_1_9_times():
    """§70. Wnace enters Weng once inside Wfan, at 0.8(1+fpylon), and once
    again on its own. TASOPT 2.16's tfweight.f counts it once. Reproduced
    and pinned rather than corrected, because `Wfan` may be intended as
    'fan module including its share of nacelle structure' -- but the
    arithmetic is worth being explicit about."""
    for fpylon in (0.0, 0.1, 0.15):
        g = ducted_fan_weight(2.0, 6000.0, 1.0, 14.0, fpylon)
        expected_multiple = 1.0 + 0.8 * (1.0 + fpylon)
        # Back out the nacelle count by differencing against zero fan mass.
        # Weng - Webare is exactly one nacelle...
        assert g.Weng - g.Webare == pytest.approx(g.Wnac)
        # ...and Webare contains 0.8(1+fpylon) more of it.
        implied = g.Wnac * expected_multiple
        assert implied > g.Wnac
    g = ducted_fan_weight(2.0, 6000.0, 1.0, 14.0, 0.1)
    assert 1.0 + 0.8 * 1.1 == pytest.approx(1.88)


def test_bare_engine_weight_is_neither_bare_nor_consistent():
    """Webare excludes the nacelle proper but includes 0.8 of it, so it is
    not Weng - Wnac in spirit even though it is in arithmetic."""
    g = ducted_fan_weight(2.0, 6000.0, 2.0, 14.0, 0.1)
    assert g.Webare == pytest.approx(g.Weng - g.Wnac)
    # ...but it still contains nacelle mass, so it is larger than the fan
    # blades alone.
    Utip = 2.0 / 2.0 * (2.0 * math.pi * 6000.0 / 60.0)
    mfan = KTECH * (135.0 * 2.0 ** 2.7 / math.sqrt(AR_FAN)
                    * (BLADE_SOLIDITY / 1.25) ** 0.3 * (Utip / 350.0) ** 0.3)
    assert g.Webare > mfan * 9.81 * 2.0


def test_half_the_blade_mass_is_a_technology_factor():
    """ktech = 0.5, flat, fixed in the source -- so half the fan mass is an
    assumption rather than a calculation, and it is not an input."""
    assert KTECH == 0.5
    assert AR_FAN == 3.0
    assert BLADE_SOLIDITY == 0.4


def test_fan_mass_grows_steeply_with_diameter():
    """D^2.7 on the blades plus D^2 on the nacelle area, so a 40% bigger fan
    is more than twice the weight."""
    small = ducted_fan_weight(1.4, 6000.0, 2.0, 14.0, 0.1)
    big = ducted_fan_weight(2.0, 6000.0, 2.0, 14.0, 0.1)
    assert big.Weng / small.Weng > 2.0


def test_tip_speed_matters_only_weakly():
    """Utip^0.3 -- doubling shaft speed adds about 23% to the blade mass,
    which is why the correlation is dominated by diameter."""
    slow = ducted_fan_weight(2.0, 4000.0, 2.0, 14.0, 0.1)
    fast = ducted_fan_weight(2.0, 8000.0, 2.0, 14.0, 0.1)
    assert 1.0 < fast.Weng / slow.Weng < 1.15


def test_the_nacelle_area_split_is_fixed():
    """40% inlet, 20% cowl, 40% exhaust, and the three have different areal
    densities -- so the split is doing real work, but it is not an input."""
    g = ducted_fan_weight(2.0, 6000.0, 1.0, 16.0, 0.1)
    assert g.Snace1 == pytest.approx(16.0 * 0.25 * math.pi * 4.0)


def test_heat_exchanger_weight_is_added_to_both_totals():
    """The reference sums it over the engine's heat exchangers; taken as an
    argument here until hxfun is ported."""
    without = ducted_fan_weight(2.0, 6000.0, 2.0, 14.0, 0.1)
    with_hx = ducted_fan_weight(2.0, 6000.0, 2.0, 14.0, 0.1, W_HX=5000.0)
    assert with_hx.Weng - without.Weng == pytest.approx(5000.0)
    assert with_hx.Webare - without.Webare == pytest.approx(5000.0)
    assert with_hx.Wnac == without.Wnac        # nacelle is unaffected
