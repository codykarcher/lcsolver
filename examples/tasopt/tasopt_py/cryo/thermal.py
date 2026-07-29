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
from dataclasses import dataclass

from ..atmosphere import atmos
from .geometry import scaled_cross_section
from .material_data import MATERIALS
from .stiffeners import GEE

__all__ = ["gas_Pr", "freestream_heat_coeff", "tank_heat_coeff",
           "vacuum_resistance", "thermal_conductivity",
           "insulation_resistances", "residuals_Q", "tank_heat_leak",
           "ThermalParams", "SIGMA_SB", "TREF",
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


def thermal_conductivity(name: str, T: float) -> float:
    """Insulation conductivity at temperature ``T``, W/(m K).

    A polynomial in temperature, ``sum(c[i] T^i)``, with the coefficients
    from v3's material database. Cryogenic insulation conductivity varies
    strongly over the 20-300 K range the tank spans, which is why this is a
    fit rather than a constant.
    """
    props = MATERIALS.get(name)
    if props is None:
        raise KeyError(f"no material {name!r} in the database")
    coeffs = props.get("conductivity_coeffs")
    if coeffs is None:
        raise KeyError(
            f"{name!r} has no conductivity fit; it is not an insulator")
    return sum(c * T ** i for i, c in enumerate(coeffs))


def insulation_resistances(T_w: float, T_ins, p) -> list:
    """Thermal resistance of each insulation layer, K/W.

    Cylindrical and end-cap paths act in **parallel** within a layer -- heat
    can go out through the barrel or through the heads -- so they combine as
    ``R_cyl R_ends / (R_cyl + R_ends)``. A vacuum layer instead uses
    :func:`vacuum_resistance`.

    Reproduces §61: a vacuum layer does **not** advance the running radius
    or the previous-layer temperature, because those two updates sit inside
    the non-vacuum branch. Any layer outboard of a vacuum gap is therefore
    computed at the wrong radius and against the wrong inner temperature.
    """
    r_inner = p.r_tank
    T_prev = T_w
    out = []
    for i, t in enumerate(p.t_cond):
        name = p.material[i]
        if name.lower() == "vacuum":
            S_inner = p.perim_R * p.l_cyl * r_inner + 2.0 * p.Shead[i]
            S_outer = (p.perim_R * p.l_cyl * (r_inner + t)
                       + 2.0 * p.Shead[i + 1])
            out.append(vacuum_resistance(T_prev, T_ins[i], S_inner, S_outer))
            # r_inner and T_prev are deliberately not advanced here -- §61.
        else:
            k = thermal_conductivity(name, (T_ins[i] + T_prev) / 2.0)
            R_cyl = (math.log((r_inner + t) / r_inner)
                     / (p.perim_R * p.l_cyl * k))
            Area_coeff = p.Shead[i] / r_inner ** 2
            R_ends = t / (k * (p.Shead[i + 1] + p.Shead[i]
                               - Area_coeff * t ** 2))
            out.append(R_ends * R_cyl / (R_ends + R_cyl))
            r_inner += t
            T_prev = T_ins[i]
    return out


@dataclass
class ThermalParams:
    """The tank geometry and flight condition the heat path is solved on."""
    l_cyl: float                 # cylindrical length of the inner tank, m
    l_tank: float                # overall inner-tank length, m
    r_tank: float                # inner-tank radius, m
    Shead: list                  # head area at each insulation interface
    t_cond: list                 # insulation layer thicknesses, m
    material: list               # insulation material names
    Tfuel: float                 # liquid temperature, K
    z: float                     # altitude, m
    TSL: float                   # sea-level design temperature, K
    Mair: float                  # freestream Mach
    xftank: float                # tank CG station, m
    ifuel: int
    cross_section: object        # the fuselage CrossSection

    @property
    def perim_R(self) -> float:
        """Perimeter over radius -- 2 pi for a circle, more for a bubble."""
        perim, _ = scaled_cross_section(self.cross_section, self.r_tank)
        return perim / self.r_tank


def residuals_Q(x, p: ThermalParams) -> list:
    """The heat-path residual. ``x = [Q, T_w, T_ins...]``.

    The circuit is air, then each insulation layer, then the liquid film, in
    series. The first residual sets the total heat rate against the driving
    temperature difference over the total resistance; the rest march the
    temperature outward from the fuel through each resistance in turn, so
    every interface temperature is consistent with the heat passing through
    it.
    """
    Q = x[0]
    T_w = x[1]
    T_ins = list(x[2:])
    Tfuse = x[-1]                # the outermost interface is the fuselage

    Rfuse = p.cross_section.radius
    h_air, _, Taw = freestream_heat_coeff(p.z, p.TSL, p.Mair, p.xftank,
                                          Tfuse, Rfuse)
    dT = Taw - p.Tfuel

    perim_inner, _ = scaled_cross_section(p.cross_section, p.r_tank)
    S_int = perim_inner * p.l_cyl + 2.0 * p.Shead[0]

    # Air-side resistance is taken over the *fuselage* perimeter, not the
    # tank's -- the tank is heated through the fuselage skin around it.
    Rair = 1.0 / (h_air * p.cross_section.perimeter * p.l_tank)
    h_liq = tank_heat_coeff(T_w, p.ifuel, p.Tfuel, p.l_tank)
    R_liq = 1.0 / (h_liq * S_int)

    R_ins = insulation_resistances(T_w, T_ins, p)
    Req = sum(R_ins) + R_liq + Rair

    F = [Q - dT / Req, 0.0] + [0.0] * len(T_ins)
    T_calc = p.Tfuel + R_liq * Q
    F[1] = T_w - T_calc
    for i in range(len(T_ins)):
        T_calc = T_calc + R_ins[i] * Q
        F[i + 2] = T_ins[i] - T_calc
    return F


def _solve(A, b) -> list:
    """``A x = b`` by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for k in range(n):
        piv = max(range(k, n), key=lambda i: abs(M[i][k]))
        if M[piv][k] == 0.0:
            raise ZeroDivisionError(
                "singular Jacobian in the tank heat-path solve")
        M[k], M[piv] = M[piv], M[k]
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= f * M[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j]
                              for j in range(i + 1, n))) / M[i][i]
    return x


def _newton(f, x0, xtol: float = 1.0e-7, ftol: float = 1.0e-6,
            maxiter: int = 200) -> list:
    """A damped Newton with numerical Jacobian, for the heat path.

    The reference uses ``NLsolve.nlsolve`` (trust-region) at the same
    tolerances. This is a genuine root of a smooth system rather than a
    capped iteration, so the solver choice does not change the answer -- the
    tests confirm agreement with the reference.

    The elimination here is a plain partial-pivot solve rather than
    :func:`tasopt_py.linalg.gaussn`. That is a deliberate difference from how
    ``blax`` is handled: there the Newton is *capped* and stops on step size,
    so where it lands depends on the iterate path and therefore on the
    elimination, and the Fortran's had to be reproduced exactly. Here the
    system is solved to convergence, so it does not.
    """
    x = list(x0)
    n = len(x)
    for _ in range(maxiter):
        F = f(x)
        if max(abs(v) for v in F) < ftol:
            return x
        A = [[0.0] * n for _ in range(n)]
        for j in range(n):
            h = 1.0e-7 * max(abs(x[j]), 1.0)
            xp = list(x)
            xp[j] += h
            Fp = f(xp)
            for i in range(n):
                A[i][j] = (Fp[i] - F[i]) / h
        step = _solve(A, [-v for v in F])

        # Damp so a wild first step cannot drive a temperature negative.
        scale = 1.0
        for i in range(n):
            if x[i] + step[i] <= 0.0 < x[i]:
                scale = min(scale, 0.5 * x[i] / abs(step[i]))
        for i in range(n):
            x[i] += scale * step[i]
        if max(abs(scale * s) for s in step) < xtol:
            return x
    return x


def tank_heat_leak(p: ThermalParams, qfac: float = 1.0) -> float:
    """Heat rate into the tank, W.

    ``qfac`` is the reference's allowance for extra leakage through valves
    and penetrations that the one-dimensional circuit does not model
    (Verstraete Eq. 3.20). It multiplies the answer.
    """
    _, _, Taw = freestream_heat_coeff(p.z, p.TSL, p.Mair, p.xftank)
    thickness = sum(p.t_cond)
    dT = Taw - p.Tfuel

    # The reference's initial guess: a nominal 0.01 K/W total resistance,
    # the wall a degree above the fuel, and the interfaces spread linearly
    # through the insulation.
    guess = [dT / 0.01, p.Tfuel + 1.0]
    for i in range(len(p.t_cond)):
        guess.append(p.Tfuel + dT * sum(p.t_cond[:i + 1]) / thickness)
    guess[-1] = guess[-1] - 1.0

    sol = _newton(lambda x: residuals_Q(x, p), guess)
    return qfac * sol[0]
