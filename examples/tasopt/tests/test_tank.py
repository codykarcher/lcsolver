"""Inner cryogenic vessel sizing, against TASOPT.jl.

The first thing in this port that sizes a fuel *tank*. TASOPT 2.16 has none --
fuel is a weight and a volume in the wing box, and the only fuel properties in
a ``.tas`` file are a density and a temperature.

Three configurations are checked against the reference (circular fuselage,
double bubble, and a larger tank at a different head aspect ratio), all six
outputs each, and they agree to machine precision.

The rest of the tests are about the model rather than the arithmetic: what the
ullage fraction buys, why the tank is smaller than the fuselage, and what the
quarter-of-ultimate allowable stress means.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.cryo.geometry import CrossSection
from tasopt_py.cryo.material_data import MATERIALS
from tasopt_py.cryo.tank import (FuselageTank, material, size_inner_tank)

REF = Path(__file__).parent / "data" / "inner_tank_ref.csv"

pytestmark = pytest.mark.skipif(not REF.exists(),
                                reason="inner tank reference absent")

OUTPUTS = ("Wtank", "Winsul", "Vfuel", "Rtank_outer", "l_tank", "l_cyl")


def _from_row(r):
    cs = CrossSection(float(r["Rfuse"]), float(r["dRfuse"]),
                      float(r["wfb"]), int(r["nwebs"]))
    tank = FuselageTank(
        Wfuelintank=float(r["Wfuel"]), rhofuel=float(r["rhof"]),
        rhofuelgas=float(r["rhog"]), ullage_frac=float(r["ullage"]),
        pvent=float(r["pvent"]), clearance_fuse=float(r["clear"]),
        ARtank=float(r["AR"]), ew=float(r["ew"]),
        ftankadd=float(r["ftankadd"]), theta_inner=float(r["theta"]),
        inner_material="Al-2219-T87",
        t_insul=[float(r["tins1"]), float(r["tins2"])],
        material_insul=["polyurethane27", "polyurethane32"])
    return float(r["Rfuse"]), cs, tank


def test_sizing_matches_tasopt_jl():
    n = 0
    for r in csv.DictReader(REF.open()):
        Rfuse, cs, tank = _from_row(r)
        g = size_inner_tank(Rfuse, cs, tank)
        got = dict(zip(OUTPUTS, (g.Wtank, g.Winsul_sum, g.Vfuel,
                                 g.Rtank_outer, g.l_tank, g.l_cyl)))
        for name in OUTPUTS:
            assert got[name] == pytest.approx(float(r[name]), rel=1e-14), (
                r["case"], name)
        n += 1
    assert n == 3


def test_all_three_fuselage_shapes_are_covered():
    cases = {r["case"] for r in csv.DictReader(REF.open())}
    assert cases == {"single", "bubble", "bigger"}
    shapes = {int(r["nwebs"]) for r in csv.DictReader(REF.open())}
    assert shapes == {0, 1}          # circular and double bubble


# --- what the model actually says -----------------------------------------

def _realistic():
    """A plausible LH2 narrowbody tank: ~3 t of fuel in a 1.9 m fuselage."""
    cs = CrossSection(1.9)
    tank = FuselageTank(
        Wfuelintank=3000.0 * 9.81, rhofuel=70.8, rhofuelgas=1.33,
        ullage_frac=0.05, pvent=2.0e5, clearance_fuse=0.1, ARtank=2.0,
        ew=0.9, ftankadd=0.1, theta_inner=1.2,
        inner_material="Al-2219-T87", t_insul=[0.05, 0.10],
        material_insul=["polyurethane27", "polyurethane32"])
    return 1.9, cs, tank


def test_a_realistic_tank_comes_out_a_sensible_size():
    """Three tonnes of LH2 is about 44 m^3 -- the reason a hydrogen airliner
    needs a fuselage tank rather than a wet wing."""
    g = size_inner_tank(*_realistic())
    assert 40.0 < g.Vfuel < 50.0
    assert 4.0 < g.l_cyl < 5.0                    # 6.1 m overall
    assert 6.0 < g.l_tank < 6.2
    assert g.Rtank_outer == pytest.approx(1.9 - 0.15 - 0.1)
    # Tank plus insulation is a serious fraction of the fuel it carries --
    # about 36% here, against a few percent for a kerosene wet wing. That
    # ratio is most of why LH2 aircraft are hard.
    assert 0.3 < g.Wtank / (3000.0 * 9.81) < 0.45


def test_the_tank_is_smaller_than_the_fuselage_by_insulation_and_clearance():
    Rfuse, cs, tank = _realistic()
    g = size_inner_tank(Rfuse, cs, tank)
    assert g.Rtank_outer == pytest.approx(
        Rfuse - sum(tank.t_insul) - tank.clearance_fuse)


def test_thicker_insulation_costs_volume_and_weight():
    Rfuse, cs, tank = _realistic()
    thin = size_inner_tank(Rfuse, cs, tank)
    tank.t_insul = [0.10, 0.20]
    thick = size_inner_tank(Rfuse, cs, tank)
    # Same fuel, so the same volume -- but a narrower tank, so it gets longer.
    assert thick.Vfuel == pytest.approx(thin.Vfuel)
    assert thick.Rtank_outer < thin.Rtank_outer
    assert thick.l_cyl > thin.l_cyl
    assert thick.Winsul_sum > thin.Winsul_sum


def test_ullage_makes_the_tank_bigger():
    """The tank holds a saturated mixture, and vapour is 53x less dense than
    liquid, so a few percent of ullage buys a few percent of volume."""
    Rfuse, cs, tank = _realistic()
    tank.ullage_frac = 0.0
    full = size_inner_tank(Rfuse, cs, tank)
    tank.ullage_frac = 0.05
    ullaged = size_inner_tank(Rfuse, cs, tank)
    assert ullaged.Vfuel > full.Vfuel
    assert (ullaged.Vfuel - full.Vfuel) / full.Vfuel == pytest.approx(
        0.05, abs=0.01)


def test_higher_vent_pressure_makes_a_heavier_vessel():
    Rfuse, cs, tank = _realistic()
    tank.pvent = 1.5e5
    low = size_inner_tank(Rfuse, cs, tank)
    tank.pvent = 4.0e5
    high = size_inner_tank(Rfuse, cs, tank)
    assert high.Wtank > low.Wtank


def test_the_insulation_surface_areas_grow_outward():
    """One entry per interface, starting at the metal wall, so N layers give
    N+1 areas and each is larger than the last."""
    Rfuse, cs, tank = _realistic()
    g = size_inner_tank(Rfuse, cs, tank)
    assert len(g.Shead_insul) == len(tank.t_insul) + 1
    assert all(b > a for a, b in zip(g.Shead_insul, g.Shead_insul[1:]))


# --- the material database ------------------------------------------------

def test_materials_are_generated_from_v3s_database():
    """2.16 has no material database at all -- rhocap, Ecap and friends are
    scalars in the .tas file, so there is no way to say 'Al-2219-T87'."""
    assert len(MATERIALS) == 29
    al = material("Al-2219-T87")
    assert al["rho"] == 2840.0
    assert al["UTS"] == 476.0e6
    assert al["YTS"] == 393.0e6
    assert al["E"] == 73.1e9


def test_insulators_carry_conductivity_fits():
    insulators = [k for k, v in MATERIALS.items()
                  if "conductivity_coeffs" in v]
    assert set(insulators) >= {"polyurethane27", "polyurethane32",
                               "mylar", "vacuum"}


def test_an_unknown_material_is_refused():
    with pytest.raises(KeyError, match="no material"):
        material("unobtainium")


def test_mismatched_insulation_lists_are_refused():
    with pytest.raises(ValueError, match="thicknesses but"):
        FuselageTank(t_insul=[0.05, 0.1], material_insul=["polyurethane27"])
