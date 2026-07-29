"""Heat exchanger correlations -- ``hxfun.jl``.

A fuel-cell aircraft rejects more heat than it makes electricity (see
:mod:`tasopt_py.engine_v3.fuelcell`), and a cryogenic tank has hydrogen cold
enough to be worth using as a heat sink. So heat exchangers stop being an
accessory and become a sizing driver. TASOPT 2.16 has none.

The geometry is a bank of **staggered circular tubes in cross flow**: coolant
inside the tubes, engine air across them. That fixes which correlations
apply:

* **Inside the tubes** -- Blasius friction with the Colburn analogy, so the
  heat transfer follows from the friction and no separate correlation is
  needed.
* **Across the bank** -- Žukauskas (1987) for the Nusselt number, which is
  piecewise in Reynolds number with *five* branches, and Gunter and Shaw
  (1945) for the pressure drop, piecewise with two.

Where the branches matter
-------------------------
Žukauskas switches its leading coefficient and both exponents at Re = 40,
1000 and 2e5, and above Re = 1000 it also switches on the **tube pitch
ratio** ``xt_D/xl_D`` crossing 2. So a design that drifts across a pitch
ratio of 2 during an optimisation sees the exponent on Reynolds number jump
from 0.6 to 0.6 but the coefficient from ``0.35 (xt/xl)^0.2`` to a flat 0.4 --
a small step, but a step.

The row-count factor ``C2`` switches at Re = 1000 too, and its two forms are
genuinely different functions of the row count, not a smooth blend.

Verified against TASOPT.jl; see ``tests/test_heat_exchanger.py``.
"""
from __future__ import annotations

import math

__all__ = ["colburn_j_pipe", "nusselt_staggered", "pressure_drop_staggered",
           "tube_thickness", "hx_weight", "TMIN_TUBE", "SAFETY_FACTOR"]

#: Minimum tube wall thickness, m -- 30 BWG, from Brewer (1991).
TMIN_TUBE = 3.0e-4
#: Applied to the design pressure difference in the hoop-stress balance.
SAFETY_FACTOR = 2.0


def colburn_j_pipe(Re_D: float) -> tuple:
    """``(j, Cf)`` inside a smooth tube.

    Blasius friction, then the Colburn analogy ``j = Cf/2``. That analogy is
    why there is no separate internal heat-transfer correlation: momentum
    and heat transport are assumed similar, which holds for ``Pr`` near one
    and is a real approximation for a liquid coolant.
    """
    Cf = 0.0791 * Re_D ** -0.25
    return Cf / 2.0, Cf


def nusselt_staggered(Re_D: float, Pr: float, N_L: float, xt_D: float,
                      xl_D: float) -> float:
    """Nusselt number across a staggered tube bank -- Žukauskas (1987).

    ``N_L`` is the number of tube rows, ``xt_D`` and ``xl_D`` the transverse
    and longitudinal pitch ratios.
    """
    if Re_D > 1000.0:
        C2 = 1.0 - math.exp(-N_L ** (1.0 / math.sqrt(3.0)))
    else:
        C2 = 1.0 - math.exp(-math.sqrt(3.0 * N_L ** (1.0 / math.sqrt(2.0))))

    if Re_D < 40.0:
        C1, m, n = 1.04, 0.4, 0.36
    elif Re_D < 1000.0:
        C1, m, n = 0.71, 0.5, 0.36
    elif Re_D <= 2.0e5 and xt_D / xl_D < 2.0:
        C1, m, n = 0.35 * (xt_D / xl_D) ** 0.2, 0.6, 0.36
    elif Re_D <= 2.0e5:
        C1, m, n = 0.4, 0.6, 0.36
    else:
        C1, m, n = 0.031 * (xt_D / xl_D) ** 0.2, 0.8, 0.4

    return C1 * C2 * Re_D ** m * Pr ** n


def pressure_drop_staggered(Re: float, G: float, L: float, rho: float,
                            Dv: float, tD_o: float, xt_D: float,
                            xl_D: float, mu_ratio: float) -> float:
    """Pressure drop across a staggered bank, Pa -- Gunter and Shaw (1945).

    ``G`` is the mass velocity, ``Dv`` the volumetric hydraulic diameter and
    ``mu_ratio`` the bulk-to-wall viscosity ratio.
    """
    f2 = 90.0 / Re if Re <= 200.0 else 0.96 * Re ** -0.145
    return (G ** 2 * L / (Dv * rho) * f2
            * (Dv / (xt_D * tD_o)) ** 0.4
            * (xl_D / xt_D) ** 0.6
            * mu_ratio ** -0.14)


def tube_thickness(K: float, dp: float, YTS: float) -> float:
    """Tube wall thickness, m, from a hoop-stress balance.

    ``K = pi b n_stages / (4 xt_D A_cc)`` relates the outer diameter to the
    geometry. The result is floored at :data:`TMIN_TUBE`.

    The hoop expression has a **pole**: with ``C = SF dp / (2 YTS)`` the
    thickness goes as ``C K / (K - 2 K C)^2``, which diverges as ``C``
    approaches 0.5 -- that is, as the design pressure approaches the yield
    stress. The reference does not check; this does.
    """
    C = SAFETY_FACTOR * dp / (2.0 * YTS)
    if C >= 0.5:
        raise ValueError(
            f"design pressure difference {dp:.3g} Pa against a yield stress "
            f"of {YTS:.3g} Pa gives C = {C:.3f}; the hoop-stress expression "
            "has a pole at C = 0.5 and no tube thickness satisfies it")
    thoop = C * K / (K - 2.0 * K * C) ** 2
    return max(thoop, TMIN_TUBE)


def hx_weight(N_tubes_tot: float, tD_o: float, tD_i: float, length: float,
              rho: float, fouter: float, gee: float = 9.81,
              shaft: tuple = None) -> float:
    """Heat exchanger weight, N.

    Tube metal only, times a fraction for headers and mounting. ``shaft``,
    if given, is ``(L, D_i, rho_shaft)`` for an exchanger sitting around an
    engine shaft, which adds the extra shaft length it forces.

    Note what is *not* counted: no coolant inventory, no headers as
    geometry, no fins. So this is a lower bound on installed mass.
    """
    V_t = N_tubes_tot * math.pi * (tD_o ** 2 - tD_i ** 2) / 4.0 * length
    W = gee * rho * V_t * (1.0 + fouter)
    if shaft is not None:
        L, D_i, rho_shaft = shaft
        W += gee * rho_shaft * L * D_i ** 2 * math.pi / 4.0
    return W


# --------------------------------------------------------------------------
# Effectiveness-NTU, both directions
# --------------------------------------------------------------------------

def _capacity_rates(C_c: float, C_p: float) -> tuple:
    """``(C_min, C_max, C_r, coolant_is_min)``."""
    C_min, C_max = min(C_c, C_p), max(C_c, C_p)
    if C_max <= 0.0:
        raise ValueError("both heat capacity rates are zero or negative")
    return C_min, C_max, C_min / C_max, C_c <= C_p


def max_effectiveness(C_r: float, coolant_is_min: bool) -> float:
    """The effectiveness at which NTU goes to infinity.

    A single-pass cross-flow exchanger with one stream mixed cannot reach
    ``eps = 1`` however large it is made -- the mixed stream carries its
    outlet temperature everywhere, which caps what the unmixed stream can
    exchange with it. The cap depends on **which** stream is mixed, and the
    two expressions are not the same function:

    * coolant is ``C_min`` (so ``C_max`` mixed): ``(1 - exp(-C_r)) / C_r``
    * coolant is ``C_max`` (so ``C_min`` mixed): ``1 - exp(-1/C_r)``

    At ``C_r = 1`` both give 0.632; they diverge as the capacity rates
    separate, and at ``C_r = 0.2`` they are 0.906 and 0.993.
    """
    if C_r <= 0.0:
        raise ValueError(f"capacity rate ratio must be positive, got {C_r}")
    if coolant_is_min:
        return (1.0 - math.exp(-C_r)) / C_r
    return 1.0 - math.exp(-1.0 / C_r)


def NTU_from_effectiveness(eps: float, C_c: float, C_p: float) -> tuple:
    """``(NTU, eps_used)`` for a required effectiveness -- the sizing
    direction.

    ``eps`` is **clipped to 99% of the achievable maximum** if it is asked
    for above it, exactly as the reference does. That is a silent
    substitution: a caller that asks for 0.95 on a geometry that can only
    reach 0.906 gets 0.897 back and no indication, so the returned value is
    handed back here for the caller to check.
    """
    C_min, C_max, C_r, coolant_is_min = _capacity_rates(C_c, C_p)
    eps_max = max_effectiveness(C_r, coolant_is_min)
    if eps > eps_max:
        eps = 0.99 * eps_max

    if coolant_is_min:
        NTU = -math.log(1.0 + math.log(1.0 - C_r * eps) / C_r)
    else:
        NTU = -1.0 / C_r * math.log(1.0 + C_r * math.log(1.0 - eps))
    return NTU, eps


def effectiveness_from_NTU(NTU: float, C_c: float, C_p: float) -> float:
    """Effectiveness for a given NTU -- the off-design direction.

    The exact inverse of :func:`NTU_from_effectiveness` on the branch that
    applies, which the tests check by round trip.
    """
    _, _, C_r, coolant_is_min = _capacity_rates(C_c, C_p)
    if coolant_is_min:
        return (1.0 - math.exp(-C_r * (1.0 - math.exp(-NTU)))) / C_r
    return 1.0 - math.exp(-1.0 / C_r * (1.0 - math.exp(-C_r * NTU)))


def heat_transfer(eps: float, C_c: float, C_p: float, Tp_in: float,
                  Tc_in: float) -> tuple:
    """``(Q, Tp_out, Tc_out)`` from an effectiveness.

    ``Qmax = C_min (Tp_in - Tc_in)`` is the thermodynamic ceiling: the most
    either stream could exchange if the exchanger were infinitely large.
    """
    C_min, _, _, _ = _capacity_rates(C_c, C_p)
    Q = eps * C_min * (Tp_in - Tc_in)
    return Q, Tp_in - Q / C_p, Tc_in + Q / C_c
