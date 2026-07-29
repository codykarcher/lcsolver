"""Landing gear sizing, against TASOPT.jl.

TASOPT 2.16 sizes gear by fixed mass fraction -- ``flgnose`` and ``flgmain``
times MTOW, and that is the whole model. The gear has no length, so nothing
connects it to the aircraft it holds up: change the fan diameter or the
tailstrike angle and the weight does not move.

v3 keeps that and adds Raymer (2012) correlations, which need a length, which
has to come from geometry. That closes a loop 2.16 does not have.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tasopt_py.structures.landing_gear import (HISTORICAL_CORRELATIONS,
                                               IN_TO_M, LOAD_FACTOR,
                                               MASS_FRACTIONS, Gear,
                                               LandingGear, size_landing_gear)

REF = Path(__file__).parent / "data" / "lg_ref.csv"

pytestmark = pytest.mark.skipif(not REF.exists(),
                                reason="landing gear reference absent")


def test_the_raymer_correlations_match_tasopt_jl():
    """Driven through the public entry point, with a geometry chosen so the
    clearance requirement sets the nose length directly."""
    n = 0
    for r in csv.DictReader(REF.open()):
        nsm, nwm, nwn = int(r["nsm"]), int(r["nwm"]), int(r["nwn"])
        gear = LandingGear(
            model=HISTORICAL_CORRELATIONS,
            nose_gear=Gear(number_struts=1, wheels_per_strut=nwn),
            main_gear=Gear(number_struts=nsm, wheels_per_strut=nwm // nsm),
            engine_ground_clearance=float(r["lgn"]))
        Wn, Wm = size_landing_gear(gear, float(r["WMTO"]),
                                   Vstall=float(r["Vstall"]))
        assert Wn == float(r["Wn"])
        # With no dihedral the main gear is the same length as the nose,
        # so its correlation is exercised at lgn rather than lgm.
        assert gear.main_gear.length == gear.nose_gear.length
        n += 1
    assert n == 32


def test_the_mass_fraction_model_is_2_16s():
    """Unchanged, and the geometry arguments are ignored -- so a case that
    used 2.16's model gets 2.16's answer."""
    gear = LandingGear(
        model=MASS_FRACTIONS,
        nose_gear=Gear(overall_mass_fraction=0.011),
        main_gear=Gear(overall_mass_fraction=0.044))
    Wn, Wm = size_landing_gear(gear, 7.8e5, dfan=99.0, Vstall=1.0)
    assert Wn == pytest.approx(7.8e5 * 0.011)
    assert Wm == pytest.approx(7.8e5 * 0.044)


def test_the_length_is_the_larger_of_clearance_and_tailstrike():
    """Both constraints are real and either can bind."""
    def nose_length(clearance, tailstrike_deg, x_end, Rfuse):
        gear = LandingGear(
            model=HISTORICAL_CORRELATIONS,
            engine_ground_clearance=clearance,
            tailstrike_angle=math.radians(tailstrike_deg))
        size_landing_gear(gear, 7.8e5, x_end=x_end, Rfuse=Rfuse,
                          Vstall=70.0)
        return gear.nose_gear.length

    # Clearance binding: a tall nacelle, a short tailcone.
    assert nose_length(3.0, 10.0, 30.0, 1.9) == pytest.approx(3.0)
    # Tailstrike binding: a long tailcone at a steep rotation angle.
    long_tail = nose_length(1.0, 14.0, 40.0, 1.9)
    assert long_tail > 1.0
    assert long_tail == pytest.approx(40.0 * math.tan(math.radians(14.0))
                                      - 3.8)


def test_a_bigger_fan_makes_heavier_gear():
    """The coupling 2.16 cannot express: the fan diameter enters the
    clearance requirement, which sets the length, which sets the weight."""
    def weight(dfan):
        gear = LandingGear(model=HISTORICAL_CORRELATIONS,
                           engine_ground_clearance=0.5)
        return size_landing_gear(gear, 7.8e5, dfan=dfan, Vstall=70.0)[0]

    assert weight(2.0) > weight(1.5) > weight(1.0)


def test_dihedral_lifts_the_engine_and_shortens_the_gear():
    def nose_length(dihedral_deg):
        gear = LandingGear(model=HISTORICAL_CORRELATIONS,
                           engine_ground_clearance=0.5,
                           wing_dihedral_angle=math.radians(dihedral_deg))
        size_landing_gear(gear, 7.8e5, dfan=1.8, span=35.0, etas=0.35,
                          Vstall=70.0)
        return gear.nose_gear.length

    assert nose_length(5.0) < nose_length(0.0)


def test_the_inch_conversion_is_v3s_not_2_16s():
    """§62. v3 has `in_to_m = 0.0254` and divides; 2.16 carries the
    reciprocal directly as `in_m = 39.37`, a truncated 39.37007874. They
    differ in the sixth digit, and this module uses v3's -- which is what
    makes the correlation match the reference exactly rather than to 1e-6."""
    from tasopt_py.output import IN_M

    assert IN_TO_M == 0.0254
    assert IN_M == 39.37
    assert 1.0 / IN_TO_M != IN_M
    assert abs(1.0 / IN_TO_M - IN_M) / IN_M == pytest.approx(2.0e-6,
                                                             rel=0.01)


def test_the_load_factor_is_fixed_in_the_source():
    """3 x 1.5, not an input -- so a case that wanted a different ultimate
    factor cannot say so."""
    assert LOAD_FACTOR == 4.5


def test_an_unknown_model_is_refused():
    with pytest.raises(ValueError, match="landing gear model"):
        LandingGear(model="guesswork")
