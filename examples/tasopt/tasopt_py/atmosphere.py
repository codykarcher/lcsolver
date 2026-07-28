"""Standard atmosphere — port of TASOPT ``src/atmos.f``.

TASOPT does not use the piecewise-linear ICAO standard atmosphere. It uses a
*smooth* approximation, which matters because the whole point of TASOPT is
gradient-based optimization: a piecewise definition has a kink at the
tropopause that upsets the optimizer.

Two smoothings are involved.

**Temperature** blends the troposphere lapse into the isothermal
stratosphere with a softplus:

    T(h) = Tblend * log(1 + exp((TSL + Tlapse*h - Tpause)/Tblend)) + Tpause

As ``Tblend -> 0`` this becomes ``max(TSL + Tlapse*h, Tpause)``, the exact
ICAO form; at the ``Tblend = 2 K`` TASOPT uses, the corner is rounded over
roughly a 2 K band around the tropopause.

**Pressure** is a fitted rational-exponent expression rather than the exact
hydrostatic integral:

    p(h) = pSL * exp(-0.118 h/(1 + 0.002 h) - 0.00198 h^2/(1 + 0.0006 h^2))

The header claims validity to h = 20 km for T and rho, and to h = 70 km for p.

Density then follows from ``rho = gam*p/((gam-1)*cp*T)``, which is the ideal
gas law with ``R = (gam-1)/gam * cp`` rather than a separately declared R.
Viscosity is Sutherland's law.

Units in, units out (as in the Fortran):
    h [km]  ->  T [K], p [Pa], rho [kg/m^3], a [m/s], mu [kg/(m s)]
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Constants exactly as in atmos.f
P_SL = 1.0132e5    # Pa
T_SL = 288.2       # K
T_PAUSE = 216.65   # K
T_BLEND = 2.0      # K, tropopause blending range
T_LAPSE = -6.5     # K/km
MU_SL = 1.78e-5    # kg/(m s)
T_SUTH = 110.0     # K
CP = 1004.0        # J/(kg K)
GAMMA = 1.4


@dataclass(frozen=True)
class AtmosphereState:
    T: float      # K
    p: float      # Pa
    rho: float    # kg/m^3
    a: float      # m/s
    mu: float     # kg/(m s)


def atmos(h: float) -> AtmosphereState:
    """Atmospheric state at altitude *h* in kilometres."""
    x = (T_SL + T_LAPSE * h - T_PAUSE) / T_BLEND
    # log1p(exp(x)) overflows for large x; it is asymptotically x there.
    # The Fortran does not guard this, but at h well below the tropopause
    # x grows and exp(x) overflows in double precision past x ~ 709.
    # Keeping the guard costs nothing and makes the function total.
    softplus = x + math.log1p(math.exp(-x)) if x > 0.0 else math.log1p(math.exp(x))
    T = softplus * T_BLEND + T_PAUSE

    p = P_SL * math.exp(
        -0.11800 * h / (1.0 + 0.0020 * h)
        - 0.00198 * h**2 / (1.0 + 0.0006 * h**2)
    )
    rho = GAMMA * p / ((GAMMA - 1.0) * CP * T)
    a = math.sqrt(GAMMA * p / rho)
    mu = MU_SL * math.sqrt(T / T_SL) ** 3 * (T_SL + T_SUTH) / (T + T_SUTH)
    return AtmosphereState(T=T, p=p, rho=rho, a=a, mu=mu)


__all__ = ["atmos", "AtmosphereState", "P_SL", "T_SL", "T_PAUSE", "GAMMA", "CP"]
