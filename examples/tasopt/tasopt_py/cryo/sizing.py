"""Tank sizing driver -- ``tanksize.jl``.

The piece that ties the rest of :mod:`tasopt_py.cryo` together: given how
much fuel a tank must hold and where it sits, work out the vessel, the
insulation, the heat leak, and where the whole assembly lands in the
fuselage.

Two modes, and the second is the interesting one
------------------------------------------------
* **Fixed insulation.** The layer thicknesses are given; size the vessel and
  report what leaks in.
* **Sized insulation.** A *boil-off rate* is given -- a percent of the fuel
  load per hour -- and the insulation thickness is solved for. That inverts
  the whole thermal model: the heat rate is known from the boil-off, and the
  unknown is how much insulation delivers exactly that resistance.

The second is what makes a hydrogen aircraft designable rather than merely
analysable, and it is why :func:`tasopt_py.cryo.thermal.residuals_Q` takes an
optional known heat rate.

A single scalar thickness increment is solved for and added to **every**
layer flagged in ``iinsuldes``, rather than each being sized independently.
So the layer *proportions* are an input and only the total is designed.

Placement
---------
``front``, ``rear`` or ``both``. The tank is set one foot clear of the cabin
at whichever end, and ``both`` splits the fuel between two tanks with the
cabin between them. The one-foot gaps are hard-wired.

Nothing here exists in TASOPT 2.16.

Verified against TASOPT.jl; see ``tests/test_tank_sizing.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .stiffeners import GEE
from .tank import (InnerTank, OuterTank, optimize_outer_tank,
                   size_inner_tank, size_outer_tank)
from .thermal import (ThermalParams, _newton, freestream_heat_coeff,
                      residuals_Q, tank_heat_leak)

__all__ = ["SizedTank", "size_tank", "insulation_increment",
           "tank_stations", "FT_TO_M", "VACUUM_MATERIALS"]

#: The hard-wired clearance between the tank and the cabin, one foot.
FT_TO_M = 0.3048

#: Insulation materials that require an outer vacuum vessel.
VACUUM_MATERIALS = ("vacuum", "microspheres")


@dataclass(frozen=True)
class SizedTank:
    """One sized tank and where it sits."""
    inner: InnerTank
    outer: OuterTank = None       # None when no vacuum jacket is needed
    Wtank: float = 0.0            # one tank, N, inner + outer
    ltank: float = 0.0            # overall length, m
    Rtank: float = 0.0            # overall radius, m
    Wfmax: float = 0.0            # max fuel capacity, all tanks, N
    Wftank: float = 0.0           # all tanks, N
    Winsftank: float = 0.0        # insulation, all tanks, N
    xftank: float = 0.0           # forward tank centroid, m (0 if none)
    xftankaft: float = 0.0        # aft tank centroid, m (0 if none)
    xfuel: float = 0.0            # fuel centroid, m
    t_insul: tuple = ()           # the thicknesses actually used, m
    Q: float = 0.0                # heat leak into one tank, W


def needs_vacuum(materials) -> bool:
    """Whether any insulation layer requires an outer vacuum vessel."""
    return any(m.lower() in VACUUM_MATERIALS for m in materials)


def insulation_increment(Rfuse, cross_section, tank, *, z, TSL, Mair,
                         xftank, ifuel, boiloff_percent, hvap,
                         iinsuldes, qfac=1.0) -> float:
    """The thickness to add to each designed layer to hit a boil-off rate.

    ``boiloff_percent`` is percent of the tank's fuel load **per hour**. The
    heat that corresponds to is fixed, so this solves for the insulation
    that delivers it -- the inverse of :func:`tank_heat_leak`.
    """
    t_cond = list(tank.t_insul)
    _, _, Taw = freestream_heat_coeff(z, TSL, Mair, xftank)
    dT = Taw - tank.Tfuel

    # Boil-off rate to heat rate. The /100/3600 is percent-per-hour to a
    # fraction per second; qfac is the same valve-leakage allowance the
    # forward problem applies, divided out here because it is applied on the
    # way back.
    mdot = boiloff_percent * tank.Wfuelintank / (GEE * 100.0) / 3600.0
    Q = mdot * hvap / qfac

    def residual(x):
        dt = x[0]
        t_all = list(t_cond)
        for i in iinsuldes:
            t_all[i] += dt
        tank.t_insul = t_all
        inner = size_inner_tank(Rfuse, cross_section, tank)
        p = ThermalParams(
            l_cyl=inner.l_cyl, l_tank=inner.l_tank,
            r_tank=inner.Rtank_outer, Shead=inner.Shead_insul,
            t_cond=t_all, material=tank.material_insul, Tfuel=tank.Tfuel,
            z=z, TSL=TSL, Mair=Mair, xftank=xftank, ifuel=ifuel,
            cross_section=cross_section)
        return residuals_Q(x[1:], p, Q_known=Q)

    guess = [0.0, tank.Tfuel + 1.0]
    total = sum(t_cond)
    for i in range(len(t_cond)):
        guess.append(tank.Tfuel + dT * sum(t_cond[:i + 1]) / total)
    guess[-1] -= 1.0

    try:
        sol = _newton(residual, guess, ftol=1.0e-7)
    finally:
        tank.t_insul = t_cond          # leave the input as we found it
    return sol[0]


def tank_stations(placement: str, ltank: float, x_start_cylinder: float,
                  l_cabin: float) -> tuple:
    """``(xftank, xftankaft, n_front, n_aft)`` for a placement.

    One foot of clearance between the tank and the cabin at each interface,
    hard-wired in the source.
    """
    ft = FT_TO_M
    if placement == "front":
        return x_start_cylinder + ft + ltank / 2.0, 0.0, 1, 0
    if placement == "rear":
        return (0.0,
                x_start_cylinder + l_cabin + ft + ltank / 2.0, 0, 1)
    if placement == "both":
        return (x_start_cylinder + ft + ltank / 2.0,
                x_start_cylinder + ft + ltank + ft + l_cabin + ft
                + ltank / 2.0, 1, 1)
    raise ValueError(
        f"tank placement must be 'front', 'rear' or 'both', not "
        f"{placement!r}")


def size_tank(Rfuse: float, cross_section, tank, *, Wfuel: float,
              tank_count: int = 1, placement: str = "rear",
              x_start_cylinder: float = 0.0, l_cabin: float = 0.0,
              z: float = 11000.0, TSL: float = 288.2, Mair: float = 0.8,
              ifuel: int = 40, qfac: float = 1.0,
              sizes_insulation: bool = False, boiloff_percent: float = 0.0,
              hvap: float = 446.0e3, iinsuldes=()) -> SizedTank:
    """Size one tank and place it. ``Wfuel`` is the total across all tanks."""
    tank.Wfuelintank = Wfuel / tank_count

    # The heat-leak station depends on where the tank ends up, but the
    # placement depends on the tank length, which depends on the heat leak.
    # The reference breaks that loop by using the *previous* iteration's
    # station, which for a standalone call is whatever was passed in.
    xftank_heat = (x_start_cylinder + l_cabin) if placement == "rear" \
        else x_start_cylinder

    if sizes_insulation:
        dt = insulation_increment(
            Rfuse, cross_section, tank, z=z, TSL=TSL, Mair=Mair,
            xftank=xftank_heat, ifuel=ifuel,
            boiloff_percent=boiloff_percent, hvap=hvap,
            iinsuldes=iinsuldes, qfac=qfac)
        t_all = list(tank.t_insul)
        for i in iinsuldes:
            t_all[i] += dt
        tank.t_insul = t_all

    inner = size_inner_tank(Rfuse, cross_section, tank)

    outer = None
    if needs_vacuum(tank.material_insul):
        Winner_tot = inner.Wtank + tank.Wfuelintank
        lcyl2 = inner.l_cyl + 2.0 * inner.l_tank - 2.0 * inner.l_cyl
        Ninterm = optimize_outer_tank(Rfuse, cross_section, tank,
                                      Winner_tot, lcyl2)
        outer = size_outer_tank(Rfuse, cross_section, tank, Winner_tot,
                                lcyl2, Ninterm)
        Wtank = inner.Wtank + outer.Wtank
        ltank = outer.l_outer
        Rtank = Rfuse - tank.clearance_fuse
    else:
        Wtank = inner.Wtank
        ltank = inner.l_tank
        Rtank = inner.Rtank_outer

    xftank, xftankaft, n_front, n_aft = tank_stations(
        placement, ltank, x_start_cylinder, l_cabin)
    xfuel = ((n_front * xftank + n_aft * xftankaft) / (n_front + n_aft))

    p = ThermalParams(
        l_cyl=inner.l_cyl, l_tank=inner.l_tank, r_tank=inner.Rtank_outer,
        Shead=inner.Shead_insul, t_cond=list(tank.t_insul),
        material=tank.material_insul, Tfuel=tank.Tfuel, z=z, TSL=TSL,
        Mair=Mair, xftank=xftank_heat, ifuel=ifuel,
        cross_section=cross_section)
    Q = tank_heat_leak(p, qfac)

    return SizedTank(
        inner=inner, outer=outer, Wtank=Wtank, ltank=ltank, Rtank=Rtank,
        Wfmax=inner.Vfuel * tank.rhofuel * GEE * tank_count,
        Wftank=tank_count * Wtank,
        Winsftank=tank_count * inner.Winsul_sum,
        xftank=xftank, xftankaft=xftankaft, xfuel=xfuel,
        t_insul=tuple(tank.t_insul), Q=Q)
