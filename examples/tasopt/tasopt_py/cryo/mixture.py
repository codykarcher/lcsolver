"""Saturated liquid/vapour mixture in a tank -- ``mixture.jl``.

A cryogenic tank does not hold liquid. It holds liquid *and* its vapour, in
equilibrium on the saturation line, and the split between them is what makes
tank pressure a state variable rather than a constant. Heat leaking in boils
liquid into vapour; vapour takes more room than the liquid it came from; the
pressure rises; the saturation temperature rises with it.

This module carries that state. Two numbers define it -- the pressure ``p``
and the **liquid fill fraction** ``beta``, the fraction of the tank *volume*
that is liquid -- and everything else follows from the saturation properties
at that pressure.

Two fractions, easily confused
------------------------------
``beta`` is by **volume**; the vapour quality ``x`` is by **mass**. They are
wildly different for hydrogen: at 95% liquid by volume the vapour is 0.1% by
mass, because saturated LH2 vapour is 53 times less dense than the liquid.
Both are carried, and :func:`from_p_beta` converts.

Nothing here exists in TASOPT 2.16, which has no tank.

Verified against TASOPT.jl; see ``tests/test_mixture.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .fuel_thermo import SaturatedPhase, gas_properties, liquid_properties

__all__ = ["SaturatedMixture", "from_p_beta", "convert_beta_same_rho"]


@dataclass(frozen=True)
class SaturatedMixture:
    """A saturated two-phase mixture at a given pressure and fill fraction."""
    gas: SaturatedPhase
    liquid: SaturatedPhase
    species: str
    beta: float          # liquid fraction by VOLUME
    x: float             # vapour quality, by MASS
    T: float             # saturation temperature, K
    p: float             # Pa
    rho: float           # bulk density, kg/m^3
    h: float             # specific enthalpy, J/kg
    u: float             # specific internal energy, J/kg
    u_p: float           # du/dp, J/(kg Pa)
    phi: float           # 1/(rho u_p) -- the energy derivative group
    rho_star: float      # rho_g / (rho_l - rho_g)
    hvap: float          # enthalpy of vaporisation, J/kg

    def at(self, p: float, beta: float) -> "SaturatedMixture":
        """The same species at a new pressure and fill fraction.

        The reference mutates in place (``update_pβ!``); this returns a new
        value, because a mixture is a state and sharing a mutable one between
        an integrator's stages is a good way to get a wrong derivative.
        """
        return from_p_beta(self.species, p, beta)


def from_p_beta(species: str, p: float, beta: float) -> SaturatedMixture:
    """Build the mixture at pressure ``p`` with liquid volume fraction
    ``beta``."""
    if not (0.0 <= beta < 1.0):
        raise ValueError(
            f"liquid fill fraction must be in [0, 1), got {beta}; at beta = 1 "
            "there is no vapour and the quality is undefined")

    gas = gas_properties(species, p)
    liquid = liquid_properties(species, p)

    # Quality: mass of vapour over total mass.
    x = 1.0 / (1.0 + (liquid.rho / gas.rho) * (beta / (1.0 - beta)))
    rho = 1.0 / (x / gas.rho + (1.0 - x) / liquid.rho)
    h = x * gas.h + (1.0 - x) * liquid.h
    u = x * gas.u + (1.0 - x) * liquid.u

    # How the quality moves with pressure at fixed volume fraction, which is
    # what makes u_p a mixture property rather than a phase one.
    x_p = ((-x / gas.rho ** 2 * gas.rho_p
            - (1.0 - x) / liquid.rho ** 2 * liquid.rho_p)
           / (1.0 / liquid.rho - 1.0 / gas.rho))
    u_p = (x * gas.u_p + (1.0 - x) * liquid.u_p
           + x_p * (gas.u - liquid.u))

    return SaturatedMixture(
        gas=gas, liquid=liquid, species=species, beta=beta, x=x,
        T=gas.Tsat, p=p, rho=rho, h=h, u=u, u_p=u_p,
        phi=1.0 / (rho * u_p),
        rho_star=gas.rho / (liquid.rho - gas.rho),
        hvap=gas.h - liquid.h)


def convert_beta_same_rho(species: str, p: float, p0: float,
                          beta0: float) -> float:
    """The fill fraction at ``p`` that has the same bulk density as
    ``beta0`` at ``p0``.

    Used when a tank is sealed and its pressure changes: the mass and the
    volume are both fixed, so the density is, and the fill fraction has to
    move to match.
    """
    rho_target = from_p_beta(species, p0, beta0).rho
    gas = gas_properties(species, p)
    liquid = liquid_properties(species, p)
    # rho = beta rho_l + (1 - beta) rho_g, solved for beta.
    return ((rho_target - gas.rho) / (liquid.rho - gas.rho))
