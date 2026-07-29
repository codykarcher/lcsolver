"""Tank pressure evolution -- ``pressure.jl``.

A sealed cryogenic tank does not hold a constant pressure. Heat leaks in and
boils liquid to vapour; the vapour occupies far more volume than the liquid
it came from; the pressure climbs, and with it the saturation temperature.
Draw fuel off and the pressure falls instead. This module is the two coupled
ordinary differential equations that describe that -- pressure and liquid
fill fraction against time -- plus the two things you can do about a rising
pressure: vent, or let it rise.

The state is ``(p, beta)``. The forcing is the heat rate ``Q``, any work
``W`` put in, and the mass flows out: ``mdot`` drawn to the engines at
quality ``xout``, and ``mdot_vent`` released at quality ``xvent``.

Why the quality of the outflow matters
--------------------------------------
Everything turns on the group ``hvap * (x + rho_star)``. Drawing off *vapour*
(``x = 1``) removes far more of the pressure-raising phase per kilogram than
drawing off liquid (``x = 0``), so venting vapour is a much more efficient
way to hold pressure down -- and drawing liquid to the engines does almost
nothing to relieve it. ``rho_star = rho_g/(rho_l - rho_g)`` is small for
hydrogen (about 0.019 at one atmosphere), so the two cases differ by a factor
of fifty.

Nothing here exists in TASOPT 2.16.

Verified against TASOPT.jl; see ``tests/test_pressure.py``.
"""
from __future__ import annotations

__all__ = ["dp_dt", "dbeta_dt", "venting_mass_flow", "mdot_boiloff"]


def dp_dt(mixture, Q: float, W: float, mdot: float, xout: float,
          mdot_vent: float, xvent: float, V: float,
          alpha: float = 1.0) -> float:
    """Rate of change of tank pressure, Pa/s.

    ``alpha`` scales the whole thing; the reference uses it to model a tank
    that is not perfectly mixed, where only part of the heat reaches the
    bulk.
    """
    return (alpha * mixture.phi / V
            * (Q + W
               - mdot * mixture.hvap * (xout + mixture.rho_star)
               - mdot_vent * mixture.hvap * (xvent + mixture.rho_star)))


def dbeta_dt(mixture, dpdt: float, mdot_tot: float, V: float) -> float:
    """Rate of change of the liquid fill fraction, 1/s.

    Two effects: mass leaving lowers the bulk density, and a pressure change
    moves both phases' densities. The second is why this needs ``dpdt``.
    """
    drho_dt = -mdot_tot / V
    return ((drho_dt
             - mixture.beta * mixture.liquid.rho_p * dpdt
             - (1.0 - mixture.beta) * mixture.gas.rho_p * dpdt)
            / (mixture.liquid.rho - mixture.gas.rho))


def venting_mass_flow(mixture, Q: float, W: float, mdot: float,
                      xout: float, xvent: float) -> float:
    """Mass flow that must be vented to hold the pressure constant, kg/s.

    Clamped at zero: if the fuel being drawn off already removes more energy
    than is coming in, the pressure is falling and there is nothing to vent.
    """
    mdot_vent = ((Q + W - mdot * mixture.hvap * (xout + mixture.rho_star))
                 / (mixture.hvap * (xvent + mixture.rho_star)))
    return max(mdot_vent, 0.0)


def mdot_boiloff(mixture, dbetadt: float, dpdt: float, mdot_liq: float,
                 V: float) -> float:
    """Rate at which liquid becomes vapour, kg/s.

    Derived from the liquid inventory rather than from the heat directly:
    the liquid volume is ``beta V``, so its mass changes with ``beta``, with
    the liquid density (which moves with pressure), and with whatever liquid
    is drawn off.
    """
    return -(dbetadt * V * mixture.liquid.rho
             + mixture.beta * V * dpdt * mixture.liquid.rho_p
             + mdot_liq)
