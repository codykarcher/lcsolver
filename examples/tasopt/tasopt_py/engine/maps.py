"""Compressor and fan efficiency map — a port of ``ecmap`` from ``tfmap.f``.

Efficiency as a function of pressure ratio and corrected mass flow, relative
to the design point. ``tfsize`` calls it three times, once each for the fan,
the low-pressure compressor and the high-pressure compressor.

The shape
---------
Two normalised coordinates:

    p = (pi - 1) / (piD - 1)        pressure ratio, relative to design
    m = mb / mbD                    corrected mass flow, relative to design

and efficiency falls away from the design point along two independent
penalties:

    e1 = 1 - CK * |p/m^(a+da-1) - m|^c        off the working line
           - DK * |m/mo - 1|^d                 off the design mass flow

plus a linear trend in pressure ratio, ``effK*(pi - piK)``, which lets a
family of engines share one map shape while their peak efficiency drifts with
size.

The ``sign()`` factors
----------------------
``psgn`` and ``msgn`` carry the sign of each bracket *outward* so that the
fractional powers ``c`` and ``d`` act on a positive quantity. Without them
``(negative)**3.0`` would be fine but ``(negative)**2.5`` would not, and the
map constants make ``c`` and ``d`` freely choosable. The effect is an odd
symmetric penalty: falling below the design mass flow is penalised the same
way as exceeding it.

The penalties are switched off in TASOPT 2.16
---------------------------------------------
``tfmap.inc`` carries four generations of map constants, three commented out.
In the **active** set ``CK = DK = 0``, which kills both penalty terms
outright: ``e1`` is identically 1 and the efficiency collapses to

    eff = effo + effK * (pi - piK)

a straight line in pressure ratio, with no mass-flow dependence at all. The
elaborate map shape above is dead in the shipped configuration. That matters
for anyone reasoning about off-design behaviour from this routine, and it is
not visible without opening the include file.

The historical sets with nonzero ``CK``/``DK`` are kept here as
``*_PENALISED`` so the full expression can still be exercised, and the tests
use them for that.

A note on ``eo``
----------------
The source computes ``eo`` as a normalising factor and then immediately
overrides it with ``eo = 1.0``, leaving the original expression commented out
one line above. So the division by ``eo`` is a no-op. Kept here for symmetry
with the source rather than silently folded away.

Verified against the compiled Fortran; see ``tests/test_maps.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["MapEfficiency", "MapSpeed", "CMAPF", "CMAPLC", "CMAPHC",
           "CMAPF_PENALISED", "CMAPLC_PENALISED", "CMAPHC_PENALISED",
           "ecmap", "Ncmap", "MapSpeedError", "TMAPL", "TMAPH", "etmap"]

# Map constants, in the order the routine unpacks them:
#     a, b, k, mo, da, c, d, CK, DK
#
# These are the ACTIVE values from tfmap.inc -- the ones not commented out.
# Note CK = DK = 0 in all three: see the module docstring, this switches the
# entire off-design penalty off.
CMAPF = (3.50, 0.80, 0.03, 0.95, -0.50, 3.0, 6.0, 0.0, 0.0)
CMAPLC = (1.90, 1.00, 0.03, 0.95, -0.20, 3.0, 5.5, 0.0, 0.0)
CMAPHC = (1.75, 2.00, 0.03, 0.95, -0.35, 3.0, 5.0, 0.0, 0.0)

# Turbine maps -- (Pcon, Ncon) only, a far simpler shape than the compressor
# maps above. Both turbines share the same pair in tfmap.inc.
TMAPL = (0.15, 0.15)
TMAPH = (0.15, 0.15)

# Historical sets, kept commented out in tfmap.inc above the active ones.
# These DO have nonzero CK/DK, so they exercise the penalty terms; the tests
# use them for exactly that reason.
CMAPF_PENALISED = (3.50, 0.80, 0.03, 0.75, -0.50, 3.0, 6.0, 2.5, 15.0)
CMAPLC_PENALISED = (2.50, 1.00, 0.03, 0.75, -0.20, 3.0, 5.5, 4.0, 6.0)
CMAPHC_PENALISED = (1.75, 2.00, 0.03, 0.75, -0.35, 3.0, 5.0, 10.5, 3.0)


@dataclass(frozen=True)
class MapEfficiency:
    eff: float       # efficiency
    eff_pi: float    # d(eff)/d(pressure ratio)
    eff_mb: float    # d(eff)/d(corrected mass flow)


def ecmap(pi: float, mb: float, piD: float, mbD: float, Cmap,
          effo: float, piK: float, effK: float) -> MapEfficiency:
    """Compressor or fan efficiency at ``(pi, mb)``.

    ``Cmap`` is the nine-element constant set; ``effo`` the peak efficiency;
    ``piK``/``effK`` the offset and slope of the linear pressure-ratio trend.
    Derivatives come back alongside, as the Fortran returns them.
    """
    a, b, k, mo, da, c, d, CK, DK = Cmap
    adm = a + da - 1.0

    p = (pi - 1.0) / (piD - 1.0)
    p_pi = 1.0 / (piD - 1.0)

    m = mb / mbD
    m_mb = 1.0 / mbD

    # Signs pulled out so the fractional powers act on positive quantities.
    psgn = math.copysign(1.0, p / m ** adm - m)
    msgn = math.copysign(1.0, m / mo - 1.0)

    eo = 1.0        # the source overrides its own expression with this

    e1 = (1.0 - CK * (psgn * (p / m ** adm - m)) ** c
          - DK * (msgn * (m / mo - 1.0)) ** d)
    e1_p = (-CK * (psgn * (p / m ** adm - m)) ** (c - 1.0)
            * c * psgn / m ** adm)
    e1_m = (-CK * (psgn * (p / m ** adm - m)) ** (c - 1.0)
            * c * psgn * (-adm * p / m ** adm / m - 1.0)
            - DK * (msgn * (m / mo - 1.0)) ** (d - 1.0)
            * d * msgn / mo)

    eff = effo * e1 / eo + effK * (pi - piK)
    eff_p = effo * e1_p / eo
    eff_m = effo * e1_m / eo

    return MapEfficiency(eff=eff,
                         eff_pi=eff_p * p_pi + effK,
                         eff_mb=eff_m * m_mb)


class MapSpeedError(RuntimeError):
    """``Ncmap`` failed to invert the map for corrected speed."""


@dataclass(frozen=True)
class MapSpeed:
    Nb: float        # corrected wheel speed
    Nb_pi: float     # d(Nb)/d(pressure ratio)
    Nb_mb: float     # d(Nb)/d(corrected mass flow)


def Ncmap(pi: float, mb: float, piD: float, mbD: float, NbD: float,
          Cmap) -> MapSpeed:
    """Corrected wheel speed at ``(pi, mb)`` -- a port of ``Ncmap``.

    Unlike :func:`ecmap` this one has no closed form. The map is written as
    ``(p, m)`` given ``N``, so getting ``N`` back out takes a Newton solve,
    and the two branches of that map meet along the *spine* ``p = m^a``:

    * above the spine the map is single-valued in ``p``, so the residual is
      written on pressure ratio;
    * below it the same is true of ``m``, so the residual switches to mass
      flow.

    Both branches are the same surface -- the switch only picks the better
    conditioned of the two residuals, which is why the derivatives coming out
    are continuous across it.

    The step limit ``|dN| <= 0.05`` matters: the ``log(1 - (m - ms)/k)``
    on the upper branch goes to ``-inf`` as the working line approaches the
    surge boundary, and an unlimited Newton step walks straight past it into
    the domain error.

    Note this routine reads ``CK``/``DK`` out of ``Cmap`` and never uses
    them -- speed does not carry the efficiency penalties.
    """
    a, b, k, mrato, da, c, d, CK, DK = Cmap
    eps = 1.0e-11

    m = mb / mbD
    m_mb = 1.0 / mbD
    p = (pi - 1.0) / (piD - 1.0)
    p_pi = 1.0 / (piD - 1.0)

    psm = m ** a            # p on the spine at this m
    N = m ** (1.0 / b)      # initial guess, also from the spine

    res = res_N = res_m = res_p = 0.0
    for _ in range(20):
        ms = N ** b
        ms_N = b * ms / N
        ps = N ** (a * b)
        ps_N = a * b * ps / N

        if p >= psm:
            # Above the spine: residual on pressure ratio.
            res = ps + 2.0 * N * k * math.log(1.0 - (m - ms) / k) - p
            res_N = (ps_N + 2.0 * k * math.log(1.0 - (m - ms) / k)
                     + 2.0 * N * k / (1.0 - (m - ms) / k) * ms_N / k)
            res_m = -2.0 * N * k / (1.0 - (m - ms) / k) * 1.0 / k
            res_p = -1.0
        else:
            # Below the spine: residual on mass flow.
            res = ms + k * (1.0 - math.exp((p - ps) / (2.0 * N * k))) - m
            res_N = ms_N + k * (-math.exp((p - ps) / (2.0 * N * k))) \
                * (-ps_N / (2.0 * N * k) - (p - ps) / (2.0 * N * k) / N)
            res_m = -1.0
            res_p = k * (-math.exp((p - ps) / (2.0 * N * k))) / (2.0 * N * k)

        dN = -res / res_N

        rlx = 1.0
        dNlim = 0.05
        if rlx * dN > dNlim:
            rlx = dNlim / dN
        if rlx * dN < -dNlim:
            rlx = -dNlim / dN

        if abs(dN) < eps:
            break

        N = N + rlx * dN
    else:
        raise MapSpeedError(
            f"Ncmap: convergence failed. N={N!r}, dN={dN!r}")

    N_m = -res_m / res_N
    N_p = -res_p / res_N
    return MapSpeed(Nb=NbD * N,
                    Nb_pi=NbD * N_p * p_pi,
                    Nb_mb=NbD * N_m * m_mb)


def etmap(dh: float, mb: float, Nb: float, piD: float, mbD: float,
          NbD: float, ept0: float, Tmap,
          Tt: float, cpt: float, Rt: float) -> float:
    """Turbine polytropic efficiency -- a port of ``etmap``.

    Two quadratic penalties, one on pressure ratio and one on the speed-flow
    product, both centred on the design point::

        ept = ept0 (1 - Pcon (1 - prat/piD)^2 - Ncon (1 - Nb mb/(NbD mbD))^2)

    The pressure ratio is not an input -- it is reconstructed from the
    specified enthalpy change through the isentropic relation

        prat = (Tt/(Tt + dh/cpt))^(cpt/(Rt ept0))

    which is why the total state has to come in alongside. ``dh`` is negative
    for a turbine, so ``Trat > 1``: ``prat`` here is the *inverse* expansion
    ratio, matching the sign convention ``piD`` carries in ``tfoper``.

    Unlike the compressor maps this one has no dead constants -- ``Pcon`` and
    ``Ncon`` are both 0.15 in the shipped configuration, so both penalties are
    live. The Fortran also returns six derivatives, which are only used to
    build its analytic Jacobian; :func:`tasopt_py.engine.tfoper.tfoper`
    differentiates numerically, so only the value comes back here.
    """
    pcon, Ncon = Tmap[0], Tmap[1]

    Trat = Tt / (Tt + dh / cpt)
    gex = cpt / (Rt * ept0)
    prat = Trat ** gex

    Nmb = Nb * mb
    NmbD = NbD * mbD

    return ept0 * (1.0 - pcon * (1.0 - prat / piD) ** 2
                        - Ncon * (1.0 - Nmb / NmbD) ** 2)

