"""The tank sizing driver, against TASOPT.jl.

Ties the structural, thermal and pressure models into one sized tank and
places it in the fuselage. Nothing in TASOPT 2.16 corresponds to any of it.

Two modes, and the second is what makes a hydrogen aircraft designable rather
than merely analysable:

* **Fixed insulation** -- thicknesses given, report what leaks in.
* **Sized insulation** -- a boil-off *rate* given, solve for the thickness.
  That inverts the whole thermal model.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.cryo.geometry import CrossSection
from tasopt_py.cryo.sizing import (FT_TO_M, insulation_increment, needs_vacuum,
                                   size_tank, tank_stations)
from tasopt_py.cryo.tank import FuselageTank

REF = Path(__file__).parent / "data" / "insul_ref.csv"

pytestmark = pytest.mark.skipif(not REF.exists(),
                                reason="insulation reference absent")


def _tank(**kw):
    base = dict(rhofuel=70.8, rhofuelgas=1.33, ullage_frac=0.05,
                pvent=2.0e5, Tfuel=20.4, clearance_fuse=0.1, ARtank=2.0,
                ew=0.9, ftankadd=0.1, theta_inner=1.2,
                inner_material="Al-2219-T87", t_insul=[0.05, 0.10],
                material_insul=["polyurethane27", "polyurethane32"])
    base.update(kw)
    return FuselageTank(**base)


def test_the_insulation_increment_matches_tasopt_jl():
    """The inverse solve: given a boil-off rate, how much insulation. Three
    cases including one where the answer is *negative* -- the given stack is
    already better than the target, so it can be thinned."""
    n = 0
    for r in csv.DictReader(REF.open()):
        tank = _tank(Wfuelintank=float(r["Wfuel"]),
                     t_insul=[float(r["t1"]), float(r["t2"])])
        dt = insulation_increment(
            float(r["R"]), CrossSection(float(r["R"])), tank,
            z=float(r["z"]), TSL=float(r["TSL"]), Mair=float(r["M"]),
            xftank=float(r["xftank"]), ifuel=40,
            boiloff_percent=float(r["boiloff"]), hvap=float(r["hvap"]),
            iinsuldes=(0, 1), qfac=float(r["qfac"]))
        assert dt == pytest.approx(float(r["dt"]), abs=1e-11), r["case"]
        n += 1
    assert n == 3


def test_a_negative_increment_is_a_real_answer():
    """Case b asks for 0.5%/hr from a stack already good enough for less, so
    the solve thins it. Reproduced rather than clamped -- clamping would
    silently make a lighter design impossible to find."""
    rows = {r["case"]: r for r in csv.DictReader(REF.open())}
    assert float(rows["b"]["dt"]) < 0.0


def test_the_increment_solve_leaves_its_input_alone():
    """It perturbs the thicknesses to evaluate residuals; the caller's tank
    must come back as it went in, or a second call gets a different answer."""
    tank = _tank(Wfuelintank=3000.0 * 9.81)
    before = list(tank.t_insul)
    insulation_increment(1.9, CrossSection(1.9), tank, z=11000.0, TSL=288.2,
                         Mair=0.8, xftank=20.0, ifuel=40,
                         boiloff_percent=0.2, hvap=446.0e3, iinsuldes=(0, 1))
    assert tank.t_insul == before


def test_the_forward_and_inverse_models_agree():
    """The strongest check available without building a whole aircraft: size
    the insulation for a target boil-off, then run the *forward* heat-leak
    model on the result and confirm it returns the heat that boil-off
    implies."""
    tank = _tank()
    s = size_tank(1.9, CrossSection(1.9), tank, Wfuel=3000.0 * 9.81,
                  placement="rear", x_start_cylinder=6.0, l_cabin=20.0,
                  sizes_insulation=True, boiloff_percent=0.2,
                  iinsuldes=(0, 1))
    implied = 0.2 * 3000.0 / 100.0 / 3600.0 * 446.0e3      # W
    assert s.Q == pytest.approx(implied, rel=1e-6)
    assert sum(s.t_insul) > 0.15                # thicker than it started


def test_a_tighter_boiloff_target_needs_more_insulation():
    def total(boiloff):
        tank = _tank()
        s = size_tank(1.9, CrossSection(1.9), tank, Wfuel=3000.0 * 9.81,
                      x_start_cylinder=6.0, l_cabin=20.0,
                      sizes_insulation=True, boiloff_percent=boiloff,
                      iinsuldes=(0, 1))
        return sum(s.t_insul)

    assert total(0.2) > total(0.3) > total(0.5) > total(1.0)


def test_an_unachievable_boiloff_target_fails_loudly():
    """0.15%/hr from a 1.9 m fuselage on polyurethane is not reachable: the
    solve drives the insulation thicker than the fuselage radius, so the
    tank radius goes negative.

    The reference computes on regardless -- nothing checks the sign -- and
    returns a tank with a negative radius and whatever weight follows from
    it. This port's CrossSection refuses a non-positive radius, so the
    impossible request fails where it becomes impossible.
    """
    tank = _tank()
    with pytest.raises((ValueError, ZeroDivisionError)):
        size_tank(1.9, CrossSection(1.9), tank, Wfuel=3000.0 * 9.81,
                  x_start_cylinder=6.0, l_cabin=20.0,
                  sizes_insulation=True, boiloff_percent=0.15,
                  iinsuldes=(0, 1))


def test_what_boiloff_costs_in_insulation():
    """The trade the module exists to quantify, on a 3 t LH2 tank in a 1.9 m
    fuselage. Halving the boil-off roughly doubles the insulation."""
    tank = _tank()

    def sized(boiloff):
        t = _tank()
        return size_tank(1.9, CrossSection(1.9), t, Wfuel=3000.0 * 9.81,
                         x_start_cylinder=6.0, l_cabin=20.0,
                         sizes_insulation=True, boiloff_percent=boiloff,
                         iinsuldes=(0, 1))

    loose, tight = sized(1.0), sized(0.5)
    assert sum(loose.t_insul) == pytest.approx(0.056, abs=0.01)
    assert sum(tight.t_insul) == pytest.approx(0.143, abs=0.01)
    assert sum(tight.t_insul) / sum(loose.t_insul) > 2.0
    assert tight.Winsftank > loose.Winsftank


def test_only_the_flagged_layers_are_designed():
    """A single scalar increment is added to every layer in iinsuldes, so
    the layer *proportions* are an input and only the total is designed."""
    tank = _tank()
    s = size_tank(1.9, CrossSection(1.9), tank, Wfuel=3000.0 * 9.81,
                  x_start_cylinder=6.0, l_cabin=20.0,
                  sizes_insulation=True, boiloff_percent=0.2,
                  iinsuldes=(1,))
    assert s.t_insul[0] == 0.05                 # untouched
    assert s.t_insul[1] > 0.10


# --- placement ------------------------------------------------------------

def test_the_three_placements():
    ft = FT_TO_M
    assert ft == pytest.approx(0.3048)

    fwd = tank_stations("front", 6.0, 6.0, 20.0)
    assert fwd == (6.0 + ft + 3.0, 0.0, 1, 0)

    aft = tank_stations("rear", 6.0, 6.0, 20.0)
    assert aft == (0.0, 6.0 + 20.0 + ft + 3.0, 0, 1)

    both = tank_stations("both", 6.0, 6.0, 20.0)
    assert both[0] == 6.0 + ft + 3.0
    assert both[1] == 6.0 + ft + 6.0 + ft + 20.0 + ft + 3.0
    assert both[2:] == (1, 1)


def test_both_tanks_put_the_fuel_centroid_between_them():
    tank = _tank()
    s = size_tank(1.9, CrossSection(1.9), tank, Wfuel=3000.0 * 9.81,
                  tank_count=2, placement="both", x_start_cylinder=6.0,
                  l_cabin=20.0)
    assert s.xftank > 0.0 and s.xftankaft > s.xftank
    assert s.xfuel == pytest.approx(0.5 * (s.xftank + s.xftankaft))
    # Two tanks, so each holds half the fuel and the pair weighs twice one.
    assert s.Wftank == pytest.approx(2.0 * s.Wtank)


def test_an_unknown_placement_is_refused():
    with pytest.raises(ValueError, match="tank placement"):
        tank_stations("middle", 6.0, 6.0, 20.0)


# --- the vacuum jacket ----------------------------------------------------

def test_a_vacuum_layer_triggers_an_outer_vessel():
    """Which is the whole reason size_outer_tank exists -- the outer vessel
    is what holds the vacuum."""
    assert needs_vacuum(["polyurethane27", "vacuum"])
    assert needs_vacuum(["microspheres"])
    assert not needs_vacuum(["polyurethane27", "polyurethane32"])

    geom = dict(x_start_cylinder=6.0, l_cabin=20.0)
    plain = size_tank(1.9, CrossSection(1.9), _tank(),
                      Wfuel=3000.0 * 9.81, **geom)
    assert plain.outer is None

    vac = size_tank(1.9, CrossSection(1.9),
                    _tank(material_insul=["vacuum", "polyurethane32"]),
                    Wfuel=3000.0 * 9.81, **geom)
    assert vac.outer is not None
    assert vac.Wtank > plain.Wtank              # the jacket costs weight
    assert vac.Q < plain.Q                      # ...and buys heat leak


def test_the_capacity_exceeds_the_fuel_it_was_sized_for():
    """Because the volume includes ullage, so the tank holds more liquid
    than the design load if it were filled solid."""
    tank = _tank()
    Wfuel = 3000.0 * 9.81
    s = size_tank(1.9, CrossSection(1.9), tank, Wfuel=Wfuel,
                  x_start_cylinder=6.0, l_cabin=20.0)
    assert s.Wfmax > Wfuel
    assert s.Wfmax / Wfuel == pytest.approx(1.05, abs=0.02)


def test_a_zero_running_length_is_refused():
    """The Meador-Smart skin friction goes as Re^-0.139, and Re is zero at
    xftank = 0. The reference does not check, so it fails as a division by
    zero somewhere less obvious."""
    from tasopt_py.cryo.thermal import freestream_heat_coeff

    with pytest.raises(ValueError, match="running length"):
        freestream_heat_coeff(11000.0, 288.2, 0.8, 0.0)
    # ...but at rest there is no forced convection, so it is fine.
    assert freestream_heat_coeff(0.0, 288.2, 0.0, 0.0, 250.0, 1.9)[0] > 0.0
