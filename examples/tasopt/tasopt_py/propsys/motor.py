"""Permanent-magnet motor losses -- ``PMSM.jl``.

The loss physics of a permanent-magnet synchronous machine, which is what
sets both its efficiency and how hard it is to cool. Nothing in TASOPT 2.16
corresponds to it -- the Fortran has no electrical machine of any kind.

Four loss mechanisms, and they scale differently
------------------------------------------------
=================  ============================  =========================
loss               scales as                     what it is
=================  ============================  =========================
ohmic              ``I^2 R``                     current through windings
hysteresis         ``m f B^alpha``               re-magnetising the steel
eddy               ``m f^2 B^2``                 currents induced in it
windage            ``Cf rho Omega^3 R^4 l``      air dragged in the gap
=================  ============================  =========================

The exponents are what matter for design. Ohmic loss falls with speed at
fixed power (less torque, less current), while the core and windage losses
all *rise* with it -- windage as the **cube** of speed. That is the whole
reason a motor has a best speed rather than simply wanting to be fast, and
the tests exercise it.

Hysteresis carries a Steinmetz exponent ``alpha`` of about 1.8 rather than a
round 2, which is why the flux density is raised to a material constant here
and squared in the eddy term.

The windage coefficient is implicit
-----------------------------------
Vrancik (1968) gives the skin friction in the annular gap as

    1/sqrt(Cf) = 2.04 + 1.768 ln(Re sqrt(Cf))

which has to be solved for ``Cf``. The reference uses ``find_zero(res,
1e-2)``; this brackets instead, for the same reason as the tank's buckling
solve -- see :func:`_solve_friction`.

Verified against TASOPT.jl; see ``tests/test_motor.py``.
"""
from __future__ import annotations

import math

__all__ = ["MU_0", "airgap_flux", "remanent_flux", "ohmic_loss",
           "hysteresis_loss", "eddy_loss", "core_loss", "windage_loss",
           "slot_resistance", "cross_sectional_area", "STEELS"]

#: Vacuum permeability, N/A^2. NIST.
MU_0 = 1.25663706127e-6

#: Electrical steels, from v3's material database.
#: ``(k_hysteresis, k_eddy, steinmetz_alpha, density)``.
STEELS = {
    "M19": (0.02351412, 7.0963515e-5, 1.793, 7750.0),
}


def airgap_flux(M: float, thickness: float, airgap: float) -> float:
    """Flux density in the airgap, T.

    A magnetic-circuit result assuming no MMF drop across the steel, so it
    is the magnet's own field divided between magnet and gap by their
    thicknesses. The consequence is a hard ceiling: however thick the magnet,
    ``B_gap`` cannot exceed ``mu0 M``.
    """
    return MU_0 * M * thickness / (thickness + airgap)


def remanent_flux(Br: float, alpha: float, T: float,
                  Tbase: float = 20.0) -> float:
    """Magnet remanent flux at temperature ``T``, T.

    Linear in temperature, with ``alpha`` a percentage per kelvin -- which is
    why the expression divides by 100. Neodymium loses roughly 0.1%/K, so a
    motor run 100 K hot has lost a tenth of its field.
    """
    return Br * (1.0 - alpha * (T - (273.15 + Tbase)) / 100.0)


def ohmic_loss(I: float, phase_resistance: float,
               energized_phases: int = 2) -> float:
    """Resistive loss, W. Two phases are energised at a time by default."""
    return I ** 2 * (energized_phases * phase_resistance)


def hysteresis_loss(mass: float, f: float, B: float, k_h: float,
                    alpha: float) -> float:
    """Hysteresis loss in a steel component, W.

    Linear in frequency and raised to a Steinmetz exponent in flux density
    -- about 1.8, not 2.
    """
    return mass * k_h * f * B ** alpha


def eddy_loss(mass: float, f: float, B: float, k_e: float) -> float:
    """Eddy-current loss in a steel component, W. Quadratic in both."""
    return mass * k_e * f ** 2 * B ** 2


def core_loss(mass: float, f: float, B: float, steel: str = "M19") -> float:
    """Hysteresis plus eddy loss for one component, W."""
    k_h, k_e, alpha, _ = STEELS[steel]
    return (hysteresis_loss(mass, f, B, k_h, alpha)
            + eddy_loss(mass, f, B, k_e))


def _solve_friction(Re: float, lo: float = 1.0e-8,
                    hi: float = 1.0) -> float:
    """Vrancik's skin-friction coefficient, bracketed.

    ``1/sqrt(Cf) = 2.04 + 1.768 ln(Re sqrt(Cf))``. The reference solves it
    with ``find_zero(res, 1e-2)`` from a guess. Bracketing is used here for
    the same reason as the tank's buckling solve: the residual has a
    logarithm, so it is undefined below ``Cf = 0`` and an unbracketed solver
    started from a guess can step there.
    """
    def res(Cf):
        return 1.0 / math.sqrt(Cf) - 2.04 - 1.768 * math.log(
            Re * math.sqrt(Cf))

    flo, fhi = res(lo), res(hi)
    if flo * fhi > 0.0:
        raise ValueError(
            f"Vrancik's friction residual does not change sign on "
            f"[{lo}, {hi}] at Re = {Re:.4g}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = res(mid)
        if fm == 0.0 or (hi - lo) < 1.0e-16 * max(1.0, mid):
            return mid
        if flo * fm < 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def windage_loss(Omega: float, radius_gap: float, airgap: float,
                 length: float, rho: float, nu: float) -> float:
    """Air-friction loss in the rotor gap, W -- Vrancik (1968).

    ``Omega`` is in rad/s. The **cube** of speed and the **fourth power** of
    radius, which is why a fast large-diameter rotor is dominated by it.
    """
    Re = Omega * radius_gap * airgap / nu
    if Re <= 0.0:
        raise ValueError(
            f"windage needs a positive gap Reynolds number, got {Re}; "
            "Vrancik's correlation takes its logarithm")
    Cf = _solve_friction(Re)
    return Cf * math.pi * rho * Omega ** 3 * radius_gap ** 4 * length


def slot_resistance(resistivity: float, area: float, length: float) -> float:
    """Resistance of one winding slot, ohm."""
    return resistivity * length / area


def cross_sectional_area(Ro: float, Ri: float) -> float:
    """Annulus area, m^2."""
    return math.pi * (Ro ** 2 - Ri ** 2)
