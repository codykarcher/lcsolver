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

import math
from dataclasses import dataclass, field

from .geometry import CrossSection, scaled_cross_section
from .material_data import MATERIALS
from .stiffeners import GEE, PREF, find_K1_head, stiffener_weight

__all__ = ["FuselageTank", "InnerTank", "size_inner_tank", "material",
           "OuterTank", "size_outer_tank"]


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
    #: Outer (vacuum jacket) vessel material.
    outer_material: str = "Al-2219-T87"
    #: Angular positions of the outer vessel's two support rings, rad.
    theta_outer: tuple = (1.0, 2.0)
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


@dataclass(frozen=True)
class OuterTank:
    """What :func:`size_outer_tank` works out for the vacuum jacket."""
    Wtank: float          # total outer-vessel weight, N
    Wcyl: float           # cylindrical portion, N
    Whead: float          # one elliptical head, N
    Wstiff: float         # all stiffeners, N
    Souter: float         # total surface area, m^2
    Shead: float          # one head, m^2
    Scyl: float           # cylindrical portion, m^2
    t_cyl: float          # cylinder wall thickness, m
    t_head: float         # head thickness, m
    l_outer: float        # overall length, m


def _solve_thickness_ratio(f, lo: float = 1.0e-9, hi: float = 1.0) -> float:
    """The wall thickness ratio that makes the buckling residual vanish.

    A bracketed bisection, where the reference uses ``Roots.find_zero(f,
    1e-3)`` -- Order0, a secant/Steffensen hybrid started from a guess.

    Using a different algorithm is safe *here* in a way it was not for
    ``blax``: this is a genuine root of a smooth monotonic function, not a
    capped iteration stopped on step size, so any solver that converges lands
    on the same number. The port's own tests confirm agreement to 1e-14.

    Bracketing rather than guessing also means it cannot wander past the
    **pole**. The residual's denominator, ``L/Do - 0.45 sqrt(t/D)``, vanishes
    at ``t/D = (L/Do / 0.45)^2`` and the expression changes sign across it
    for no physical reason. On a closely stiffened vessel -- short spans, so
    small ``L/Do`` -- that pole sits at ``t/D`` well below 1, so a naive
    bracket of ``[0, 1]`` straddles it and either fails or converges to
    nonsense. The caller passes ``hi`` just below it.
    """
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0.0:
        raise ValueError(
            f"the buckling residual does not change sign on [{lo}, {hi}] "
            f"(f = {flo:.4g}, {fhi:.4g}); no wall thickness satisfies the "
            "collapse condition for this geometry")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0 or (hi - lo) < 1.0e-16 * max(1.0, mid):
            return mid
        if flo * fm < 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def size_outer_tank(Rfuse: float, cross_section: CrossSection,
                    tank: FuselageTank, Winnertank: float, l_cyl: float,
                    Ninterm: float) -> OuterTank:
    """Size the outer vacuum-jacket vessel.

    Unlike the inner vessel, which is pressurised from within, this one is
    loaded from *outside* -- it holds vacuum against the cabin -- so it is
    sized against **collapse**, not burst. The wall thickness comes from the
    critical buckling pressure of a stiffened cylinder (Barron 1985, Eq.
    (7.11) at four atmospheres), which is implicit in ``t/D`` and has to be
    solved for.

    ``Ninterm`` is the number of intermediate stiffener rings, on top of the
    two main ones. Those intermediate rings carry **no load** -- they exist
    only to shorten the unsupported span and so allow a thinner wall -- which
    is what makes the number of them worth optimising.
    """
    alloy = material(tank.outer_material)
    poiss = alloy["nu"]
    Eouter = alloy["E"]
    rho_outer = alloy["rho"]
    s_a = alloy["UTS"] / 4.0

    theta1, theta2 = tank.theta_outer
    Nmain = 2.0
    pc = 4.0 * PREF                     # Barron Eq. (7.11)

    Rtank_outer = Rfuse - tank.clearance_fuse
    Do = 2.0 * Rtank_outer
    Nstiff = Nmain + Ninterm
    # Two of the stiffeners are at the ends, so the skin is divided into
    # Nstiff - 1 spans.
    L = l_cyl / (Nstiff - 1.0)
    L_Do = L / Do

    def residual(t_D):
        return (2.42 * Eouter * t_D ** 2.5
                / ((1.0 - poiss ** 2) ** 0.75
                   * (L_Do - 0.45 * math.sqrt(t_D)))
                - pc)

    # The denominator vanishes here; search strictly below it.
    t_D_pole = (L_Do / 0.45) ** 2
    t_Do = _solve_thickness_ratio(residual, hi=min(1.0, 0.999 * t_D_pole))
    t_cyl = t_Do * Do

    K1 = find_K1_head(tank.ARtank)
    t_head = K1 * Do * math.sqrt(pc * math.sqrt(3.0 * (1.0 - poiss ** 2))
                                 / (0.5 * Eouter))

    perim_vessel, Avessel = scaled_cross_section(cross_section, Rtank_outer)
    Shead = (2.0 * Avessel
             * (0.333 + 0.667 * (1.0 / tank.ARtank) ** 1.6) ** 0.625)
    Scyl = perim_vessel * l_cyl
    Souter = Scyl + 2.0 * Shead

    Wcyl = Scyl * t_cyl * rho_outer * GEE
    Whead = Shead * t_head * rho_outer * GEE

    common = (Rtank_outer, perim_vessel, s_a, rho_outer, theta1, theta2,
              Nstiff, l_cyl, Eouter)
    # Each main ring carries half the inner vessel; the intermediate rings
    # carry nothing at all.
    Wmainstiff = stiffener_weight("outer", Winnertank / Nmain, *common)
    Wintermstiff = stiffener_weight("outer", 0.0, *common)
    Wstiff = Nmain * Wmainstiff + Ninterm * Wintermstiff

    Wtank = (Wcyl + 2.0 * Whead + Wstiff) * (1.0 + tank.ftankadd)
    l_outer = l_cyl + Do / tank.ARtank + 2.0 * t_head

    return OuterTank(Wtank=Wtank, Wcyl=Wcyl, Whead=Whead, Wstiff=Wstiff,
                     Souter=Souter, Shead=Shead, Scyl=Scyl, t_cyl=t_cyl,
                     t_head=t_head, l_outer=l_outer)


def optimize_outer_tank(Rfuse: float, cross_section: CrossSection,
                        tank: FuselageTank, Winnertank: float, l_cyl: float,
                        Ninterm0: float = 1.0, *, lower: float = 0.0,
                        upper: float = 50.0, tol: float = 1.0e-9,
                        maxeval: int = 100) -> float:
    """The number of intermediate stiffener rings that minimises tank weight.

    There is a real trade here. Each ring costs its own weight, but shortens
    the unsupported span, which lets the wall be thinner over the whole
    cylinder. Below the optimum the wall dominates; above it the rings do.

    The reference minimises with NLopt's Nelder-Mead over a **continuous**
    ``Ninterm`` bounded to [0, 50] -- so the answer is a fractional number of
    rings, which is then used as one. That is reproduced rather than rounded:
    rounding would change every outer-vessel weight, and the fractional count
    is what the reference's own tank weights are built on.

    This uses a golden-section search rather than a simplex, because a
    one-dimensional bounded minimisation is what this is; the reference
    reaches for a general-purpose simplex because NLopt was already a
    dependency. Both converge to the same interior minimum -- see
    ``tests/test_tank.py`` -- and golden section cannot wander outside the
    bounds, which Nelder-Mead on a boxed problem can.
    """
    def weight(N):
        return size_outer_tank(Rfuse, cross_section, tank, Winnertank,
                               l_cyl, N).Wtank

    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lower, upper
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = weight(c), weight(d)

    for _ in range(maxeval):
        if abs(b - a) < tol * max(1.0, abs(a) + abs(b)):
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = weight(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = weight(d)

    return 0.5 * (a + b)
