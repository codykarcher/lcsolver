"""Heat leak into a cryogenic tank -- ``tankWthermal.jl``.

This is what turns a tank from a *structure* into a *boil-off rate*, and so
what finally couples hydrogen to the mission: fuel that boils has to be
vented or burned, so the aircraft carries more of it than it uses.

The heat path runs from the freestream to the liquid, in series:

1. **Air to the outer wall.** Forced convection in flight (Meador-Smart
   reference-temperature method with the Chilton-Colburn analogy, Anderson
   p. 1056) or natural convection on a horizontal cylinder when stationary
   (Holman p. 334). The switch is at ``M == 0``.
2. **Through the insulation**, layer by layer -- and through a vacuum gap if
   there is one, where the transport is radiation in parallel with residual
   gas conduction, Barron (1985).
3. **Inner wall to the liquid**, natural convection at the tank scale.

Nothing here exists in TASOPT 2.16, which has no tank and therefore no
boil-off.

Things worth knowing
--------------------
* **``gasPr("air_simple")`` uses a constant ``cp = 1005`` and
  ``R = 287.1``**, where ``gasPr("air")`` sums the real five-species mixture.
  The reference picks the simple one here for speed, and says so. That makes
  the air properties inconsistent with the rest of TASOPT, which uses the
  full mixture everywhere -- 287.1 against the 286.857 that
  :mod:`tasopt_py.sizing.mission` derives. Reproduced, with both available.
* **The liquid-side properties are hard-wired NIST point values** at 2 atm,
  taken at 120 K for methane and 20 K for hydrogen. They do not move with
  tank pressure even though everything around them does.
* The vacuum gap assumes 1e-2 Pa (about 1e-4 Torr, following Brewer 1991),
  polished-aluminium emissivity of 0.04, and accommodation coefficients of
  0.9 and 1.0. All are fixed in the source; the pressure carries a ``TODO``
  wondering whether it should be an input.

Verified against TASOPT.jl; see ``tests/test_thermal.py``.
"""
from __future__ import annotations

import math

from ..atmosphere import atmos
from .stiffeners import GEE

__all__ = ["gas_Pr", "freestream_heat_coeff", "tank_heat_coeff",
           "vacuum_resistance", "SIGMA_SB", "TREF",
           "VACUUM_PRESSURE", "VACUUM_EMISSIVITY"]

#: ``constants.jl``. Note ``Tref`` is 288.2, not the 288.15 of the standard
#: atmosphere -- the reference carries its own.
TREF = 288.2
SIGMA_SB = 5.670374419e-8

#: The vacuum gap's assumed state, all fixed in the source.
VACUUM_PRESSURE = 1.0e-2        # Pa, ~1e-4 Torr (Brewer 1991)
VACUUM_EMISSIVITY = 0.04        # highly polished aluminium
VACUUM_ACCOM_OUTER = 0.9
VACUUM_ACCOM_INNER = 1.0
VACUUM_RGAS = 287.05
VACUUM_GAMMA = 1.4

#: Sutherland constants and the simple-air properties, from ``gasPr``.
_AIR = dict(mu0=1.716e-5, S_mu=111.0, K0=0.0241, S_k=194.0, T0=273.0,
            R=287.1, cp=1005.0)

#: Liquid-side properties, hard-wired NIST point values at 2 atm.
#: ``(Pr, beta [1/K], nu [m^2/s], k [W/m/K])``.
_LIQUID = {
    11: (2.0, 3.5e-3, 2.4e-7, 0.17185),      # CH4 at 120 K
    40: (1.3, 15.0e-3, 2.0e-7, 0.10381),     # LH2 at 20 K
}


def gas_Pr(T: float) -> tuple:
    """``gasPr("air_simple", T)`` -- ``(R, Pr, gamma, cp, mu, k)``.

    Constant ``cp`` and ``R``, with Sutherland's laws for viscosity and
    conductivity. The reference offers a full-mixture ``"air"`` variant and
    deliberately does not use it here, "to save some computational time by
    using a constant cp for air".
    """
    a = _AIR
    mu = a["mu0"] * (T / a["T0"]) ** 1.5 * ((a["T0"] + a["S_mu"])
                                            / (T + a["S_mu"]))
    k = a["K0"] * (T / a["T0"]) ** 1.5 * ((a["T0"] + a["S_k"])
                                          / (T + a["S_k"]))
    cp, R = a["cp"], a["R"]
    Pr = cp * mu / k
    gamma = cp / (cp - R)
    return R, Pr, gamma, cp, mu, k


def freestream_heat_coeff(z: float, TSL: float, M: float, xftank: float,
                          Tw: float = TREF, Rfuse: float = 1.0) -> tuple:
    """Air-side heat transfer coefficient, W/(m^2 K).

    Returns ``(h, Tair, Taw)`` -- the coefficient, the freestream static
    temperature, and the adiabatic wall temperature that the driving
    difference is taken against.

    ``M == 0`` switches to natural convection on a horizontal cylinder; any
    forward speed uses the reference-temperature forced-convection model.
    """
    at = atmos(z / 1000.0, TSL - TREF)
    Tair, p, a = at.T, at.p, at.a
    u = M * a

    R, Pr, gamma, cp, _, _ = gas_Pr(Tair)
    r = Pr ** (1.0 / 3.0)                    # turbulent recovery factor
    Taw = Tair * (1.0 + r * M ** 2 * (gamma - 1.0) / 2.0)

    # Meador-Smart reference temperature.
    T_s = Tair * (0.5 * (1.0 + Tw / Tair)
                  + 0.16 * r * (gamma - 1.0) / 2.0 * M ** 2)
    _, Pr_s, _, cp, mu_s, k_s = gas_Pr(T_s)
    rho_s = p / (R * T_s)

    if M == 0.0:
        L = 2.0 * Rfuse
        beta = 1.0 / Tair
        nu = mu_s / rho_s
        Gr = GEE * beta * abs(Tair - Tw) * L ** 3 / nu ** 2
        Ra = Gr * Pr_s
        # Holman p. 334, the 1e9 < Ra correlation.
        Nu = 0.13 * Ra ** 0.333
        h = Nu * k_s / L
    else:
        Re = rho_s * u * xftank / mu_s
        cf = 0.02296 / Re ** 0.139           # Meador-Smart
        St = cf / (2.0 * Pr_s ** (2.0 / 3.0))   # Chilton-Colburn
        h = St * rho_s * u * cp

    return h, Tair, Taw


def tank_heat_coeff(T_w: float, ifuel: int, Tfuel: float,
                    ltank: float) -> float:
    """Liquid-side heat transfer coefficient, W/(m^2 K).

    Natural convection at the tank length scale, with a length-based Nusselt
    correlation. ``ifuel`` is 11 for methane or 40 for hydrogen -- the same
    gas indices the engine uses.
    """
    try:
        Pr_l, beta, nu_l, k = _LIQUID[ifuel]
    except KeyError:
        raise ValueError(
            f"no liquid-side properties for ifuel = {ifuel}; only CH4 (11) "
            "and H2 (40) are tabulated. TASOPT.jl's `if` chain has no "
            "`else`, so an unknown fuel leaves the properties undefined."
        ) from None

    Ra_l = (GEE * beta * abs(T_w - Tfuel) * ltank ** 3 * Pr_l / nu_l ** 2)
    Nu_l = 0.0605 * Ra_l ** (1.0 / 3.0)
    return Nu_l * k / ltank


def vacuum_resistance(Tcold: float, Thot: float, S_inner: float,
                      S_outer: float) -> float:
    """Thermal resistance of a vacuum gap, K/W -- Barron (1985).

    Radiation and residual-gas conduction act in **parallel**, so the two
    resistances combine as ``R1 R2 / (R1 + R2)``. Radiation dominates at
    these temperatures; the gas term is what makes an imperfect vacuum cost
    something.
    """
    eps = VACUUM_EMISSIVITY
    Fe = 1.0 / (1.0 / eps + S_inner / S_outer * (1.0 / eps - 1.0))
    hrad = SIGMA_SB * Fe * (Tcold ** 2 + Thot ** 2) * (Tcold + Thot)
    R_rad = 1.0 / (hrad * S_inner)

    Fa = 1.0 / (1.0 / VACUUM_ACCOM_INNER
                + S_inner / S_outer * (1.0 / VACUUM_ACCOM_OUTER - 1.0))
    G = ((VACUUM_GAMMA + 1.0) / (VACUUM_GAMMA - 1.0)
         * math.sqrt(VACUUM_RGAS / (8.0 * math.pi * Thot)) * Fa)
    R_conv = 1.0 / (G * VACUUM_PRESSURE * S_inner)

    return R_conv * R_rad / (R_conv + R_rad)
