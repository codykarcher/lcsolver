"""Inner cryogenic vessel sizing -- ``size_inner_tank`` in ``tankWmech.jl``.

The first thing in this port that sizes a *fuel tank*. TASOPT 2.16 has none:
fuel is a weight and a volume in the wing box, and the only fuel properties
in a ``.tas`` file are a density and a temperature. That is a fair model of
kerosene in a wet wing and a useless one for liquid hydrogen, which boils at
20 K, is a twelfth the density, and has to be carried in an insulated
pressure vessel inside the fuselage.

What this sizes
---------------
A cylindrical vessel with ellipsoidal heads, sitting inside the fuselage and
taking its cross-section (so a double-bubble fuselage gets a double-bubble
tank), wrapped in one or more insulation layers:

* wall and head thickness from the vent pressure, Barron (1985) Eqs. (7.1)
  and (7.2), with the allowable stress **a quarter** of ultimate (Barron
  p. 359);
* cylinder length from the volume the fuel needs, allowing for ullage --
  the tank is not full of liquid, it holds a saturated mixture;
* two main stiffener rings (:mod:`tasopt_py.cryo.stiffeners`);
* the insulation layers built outward from the metal wall, each contributing
  a cylindrical shell and two ellipsoidal caps.

The materials come from :mod:`~tasopt_py.cryo.material_data`, generated from
v3's TOML database -- the first place in the port that names a material
rather than passing a density.

Things worth knowing
--------------------
* **The head thickness uses the tank's *outer* radius while the wall
  thickness that produced it used the same.** Barron's Eq. (7.2) is written
  in terms of the inner diameter; the reference passes ``2 * Rtank_outer``
  to both. Reproduced.
* **``Wfuel`` already includes boil-off.** The comment says so; the caller
  is expected to have added it.
* The ullage fraction mixes liquid and vapour densities linearly to get a
  bulk density, which is what sets the volume. At the usual 5% ullage and
  LH2's density ratio of 54, the vapour is worth about 0.1% of the mass and
  5% of the volume.

Verified against TASOPT.jl; see ``tests/test_tank.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import CrossSection, scaled_cross_section
from .material_data import MATERIALS
from .stiffeners import GEE, stiffener_weight

__all__ = ["FuselageTank", "InnerTank", "size_inner_tank", "material"]


def material(name: str) -> dict:
    """A material's properties by name, from v3's database."""
    try:
        return MATERIALS[name]
    except KeyError:
        raise KeyError(
            f"no material {name!r} in the database; "
            f"{len(MATERIALS)} are defined") from None


@dataclass
class FuselageTank:
    """The tank-design inputs -- v3's ``fuselage_tank``, the fields used here.

    TASOPT 2.16 has no equivalent, so this is additive: it sits alongside
    ``parg`` rather than replacing anything, which is what keeps the
    byte-exact 737 regression intact.
    """
    #: Weight of fuel in one tank, N. Already includes boil-off.
    Wfuelintank: float = 0.0
    #: Saturated liquid and vapour densities, kg/m^3.
    rhofuel: float = 0.0
    rhofuelgas: float = 0.0
    #: Fraction of the tank volume occupied by vapour.
    ullage_frac: float = 0.0
    #: Vent pressure, Pa -- what the vessel is sized to hold.
    pvent: float = 0.0
    #: Radial gap between the fuselage inner wall and the tank, m.
    clearance_fuse: float = 0.0
    #: Head aspect ratio: major diameter over head depth.
    ARtank: float = 2.0
    #: Weld efficiency.
    ew: float = 0.9
    #: Additional-weight fraction for fittings, supports and the like.
    ftankadd: float = 0.0
    #: Angular position of the main support rings, rad from the bottom.
    theta_inner: float = 0.0
    #: Inner-vessel material name, into :data:`MATERIALS`.
    inner_material: str = "Al-2219-T87"
    #: Insulation layer thicknesses, m, innermost first.
    t_insul: list = field(default_factory=list)
    #: Insulation material names, one per layer.
    material_insul: list = field(default_factory=list)

    def __post_init__(self):
        if len(self.t_insul) != len(self.material_insul):
            raise ValueError(
                f"{len(self.t_insul)} insulation thicknesses but "
                f"{len(self.material_insul)} materials")


@dataclass(frozen=True)
class InnerTank:
    """What :func:`size_inner_tank` works out."""
    Wtank: float          # total inner-vessel weight incl. insulation, N
    Winsul_sum: float     # insulation weight, N
    Vfuel: float          # volume the saturated mixture occupies, m^3
    Shead_insul: list     # head surface area at each insulation interface
    Rtank_outer: float    # outer radius of the metal vessel, m
    l_tank: float         # overall length including heads and insulation, m
    l_cyl: float          # length of the cylindrical portion, m


def size_inner_tank(Rfuse: float, cross_section: CrossSection,
                    tank: FuselageTank) -> InnerTank:
    """Size the inner (pressure) vessel and its insulation.

    ``Rfuse`` is the fuselage inner radius and ``cross_section`` its shape;
    the tank is that shape scaled down by the insulation thickness and the
    clearance.
    """
    t_cond = list(tank.t_insul)
    thickness_insul = sum(t_cond)

    alloy = material(tank.inner_material)
    sigskin = alloy["UTS"]
    rhoskin = alloy["rho"]

    dp = tank.pvent
    AR = tank.ARtank
    weld_eff = tank.ew

    Rtank_outer = Rfuse - thickness_insul - tank.clearance_fuse
    # Barron (1985) p. 359: allowable stress is a quarter of ultimate.
    s_a = sigskin / 4.0

    # Barron Eq. (7.1) and (7.2). Both are given the *outer* diameter.
    tskin = dp * (2.0 * Rtank_outer) / (2.0 * s_a * weld_eff + 0.8 * dp)
    Rtank = Rtank_outer - tskin
    Lhead = Rtank / AR
    K = (1.0 / 6.0) * (AR ** 2 + 2.0)
    t_head = (dp * (2.0 * Rtank_outer) * K
              / (2.0 * s_a * weld_eff + 2.0 * dp * (K - 0.1)))

    perim_tank, Atank = scaled_cross_section(cross_section, Rtank)

    # The tank holds a saturated mixture, not pure liquid.
    rho_mix = ((1.0 - tank.ullage_frac) * tank.rhofuel
               + tank.ullage_frac * tank.rhofuelgas)
    Vfuel = tank.Wfuelintank / (GEE * rho_mix)

    # Two ellipsoidal caps plus however much cylinder is left over.
    V_ellipsoid = 2.0 * (Atank * Rtank / AR) / 3.0
    V_cylinder = Vfuel - 2.0 * V_ellipsoid
    l_cyl = V_cylinder / Atank

    Scyl = perim_tank * l_cyl
    Shead = 2.0 * Atank * (0.333 + 0.667 * (Lhead / Rtank) ** 1.6) ** 0.625

    Whead = rhoskin * GEE * Shead * t_head
    Wcyl = rhoskin * GEE * Scyl * tskin

    # The support rings carry the vessel *and its contents*.
    Winnertank = Wcyl + 2.0 * Whead + tank.Wfuelintank
    Nmain = 2.0
    Wstiff = Nmain * stiffener_weight(
        "inner", Winnertank / Nmain, Rtank, perim_tank, s_a, rhoskin,
        tank.theta_inner)

    Wtank = (Wcyl + 2.0 * Whead + Wstiff) * (1.0 + tank.ftankadd)

    # Insulation, built outward from the metal wall.
    N = len(t_cond)
    Shead_insul = [0.0] * (N + 1)
    Winsul_sum = 0.0
    L = Lhead + tskin
    Ro = Ri = Rtank_outer
    _, Ao = scaled_cross_section(cross_section, Ro)
    Shead_insul[0] = 2.0 * Ao * (0.333 + 0.667 * (L / Ro) ** 1.6) ** 0.625

    for n in range(N):
        Ro = Ro + t_cond[n]
        L = L + t_cond[n]
        _, Ao = scaled_cross_section(cross_section, Ro)
        _, Ai = scaled_cross_section(cross_section, Ri)
        rho_insul = material(tank.material_insul[n])["rho"]

        Vcyl_insul = l_cyl * (Ao - Ai)
        Shead_insul[n + 1] = (2.0 * Ao
                              * (0.333 + 0.667 * (L / Ro) ** 1.6) ** 0.625)
        # Closed form for the volume of an ellipsoidal shell of thickness t.
        Area_coeff = Shead_insul[n + 1] / Ro ** 2
        Vhead_insul = ((Shead_insul[n] + Shead_insul[n + 1]) / 2.0
                       - Area_coeff / 6.0 * t_cond[n] ** 2) * t_cond[n]

        Winsul_sum += (Vcyl_insul + 2.0 * Vhead_insul) * rho_insul * GEE
        Ri = Ro

    Wtank = Wtank + Winsul_sum
    l_tank = l_cyl + 2.0 * Lhead + 2.0 * thickness_insul + 2.0 * t_head

    return InnerTank(Wtank=Wtank, Winsul_sum=Winsul_sum, Vfuel=Vfuel,
                     Shead_insul=Shead_insul, Rtank_outer=Rtank_outer,
                     l_tank=l_tank, l_cyl=l_cyl)
