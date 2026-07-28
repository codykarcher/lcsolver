"""Surface profile drag — a port of ``surfcd.f``.

TASOPT computes wing and tail profile drag two ways, and the file provides
both:

``surfcd``
    Takes section friction and pressure drag coefficients (``cdf``, ``cdp``)
    as *inputs* and integrates them spanwise in closed form, assuming
    ``cd ~ Re^aRexp`` along the taper. Self-contained, and the one used when
    airfoil tables are not being consulted.

``surfcd2``
    Integrates numerically over eight spanwise stations, calling the airfoil
    database (``airfun``) at each to get ``cdf`` and ``cdp`` from the local
    lift coefficient, thickness and perpendicular Mach number. More faithful,
    and it also returns the section lift coefficients.

Both apply the same two corrections, which are the physics worth
understanding:

* **Sweep.** Pressure drag is reduced by sweep but friction drag is not, so
  only ``cdp`` carries the ``cos^3(Lambda)`` factor (as
  ``(fSuns + (1-fSuns) cos^2 L) * cos L``).
* **Root shock unsweep.** Near the wing root the shock is not swept, so
  sweep relief is lost there. ``fSuns`` is the fraction of the surface
  subject to that loss; where ``fSuns -> 1`` the pressure drag reverts to its
  unswept value. ``surfcd`` estimates it as one number for the whole surface,
  ``surfcd2`` decays it spanwise as ``exp(-(eta-etao) b / (kSuns C 2 co))``.

Planform convention, shared with ``surfw.f`` and ``surfcm.f``: the surface is
a double-taper, broken at ``bs``, with ``bo`` the fuselage-carrythrough span.
``lambda`` ratios are chords over the root chord and ``gamma`` ratios are
section loads over the root load.

Verified against the compiled Fortran; see ``tests/test_drag.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

__all__ = ["SurfCD", "SurfCD2", "surfcd", "surfcd2"]


@dataclass(frozen=True)
class SurfCD:
    CDsurf: float   # overall profile CD, referenced to S
    CDover: float   # fuselage CD added by lift carryover


@dataclass(frozen=True)
class SurfCD2:
    clpo: float     # section lift coefficients at root, break and tip
    clps: float
    clpt: float
    CDfwing: float  # friction profile cd in the perpendicular plane
    CDpwing: float  # pressure profile cd in the perpendicular plane
    CDwing: float   # overall profile CD
    CDover: float


def surfcd(S: float, b: float, bs: float, bo: float,
           lambdat: float, lambdas: float, sweep: float, co: float,
           cdf: float, cdp: float, Reco: float, Reref: float, aRexp: float,
           kSuns: float, fCDcen: float) -> SurfCD:
    """Profile CD from given section drag coefficients (``surfcd``)."""
    cosL = math.cos(sweep * math.pi / 180.0)

    etao = bo / b
    etas = bs / b

    # exposed surface area
    Ssurf = co * 0.5 * b * ((1.0 + lambdas) * (etas - etao)
                            + (lambdas + lambdat) * (1.0 - etas))

    # fraction of the exposed surface subject to root shock unsweep
    fSuns = kSuns * co ** 2 / (0.5 * Ssurf)

    # section cd with Re and sweep effects, referenced to the root chord
    Refac = (Reco / Reref) ** aRexp
    cdo = (cdf + cdp * (fSuns + (1.0 - fSuns) * cosL ** 2) * cosL) * Refac

    # Spanwise integral of cd over the taper, assuming cd ~ Re^aRexp. Both
    # factors below are singular in their exact form when the two taper
    # ratios coincide, so TASOPT switches to the asymptotic expansion within
    # 0.02 of the singularity. The thresholds are reproduced exactly: they
    # change the answer in the sixth decimal near the switch, and matching
    # the Fortran bit-for-bit means matching where it switches.
    lsxa = lambdas ** aRexp
    lxa = lambdat ** aRexp

    if abs(lambdas - 1.0) < 0.02:
        lsfac = 1.0 - aRexp * (1.0 - lambdas) / (1.0 + lambdas)
    else:
        lsfac = (2.0 / (2.0 + aRexp) * (1.0 - lsxa * lambdas ** 2)
                 / (1.0 - lambdas ** 2))

    if abs(lambdat - lambdas) < 0.02:
        lfac = 1.0 - aRexp * (lambdas - lambdat) / (lambdas + lambdat)
    else:
        lfac = (2.0 / (2.0 + aRexp)
                * (lsxa * lambdas ** 2 - lxa * lambdat ** 2)
                / (lambdas ** 2 - lambdat ** 2))

    CDsurf = (co * 0.5 * b / S) * cdo * (
        2.0 * etao * fCDcen
        + (1.0 + lambdas) * (etas - etao) * lsfac
        + (lambdas + lambdat) * (1.0 - etas) * lfac)

    # Carryover is disabled in TASOPT 2.16: the expression is present in the
    # source but commented out, with CDover hard-set to zero. Reproduced.
    CDover = 0.0
    return SurfCD(CDsurf=CDsurf, CDover=CDover)


def surfcd2(S: float, b: float, bs: float, bo: float,
            lambdat: float, lambdas: float, gammat: float, gammas: float,
            toco: float, tocs: float, toct: float,
            Mach: float, sweep: float, co: float,
            CL: float, CLhtail: float, fLo: float, fLt: float,
            Reco: float, aRexp: float, kSuns: float, fexcd: float,
            ARe: float, airfoil: Callable[[float, float, float], tuple],
            fduo: float = 0.0, fdus: float = 0.0, fdut: float = 0.0,
            n: int = 8) -> SurfCD2:
    """Profile CD by spanwise quadrature against an airfoil database.

    ``airfoil(cl, toc, Mperp) -> (cdf, cdp, cdwbar, cm)`` stands in for
    TASOPT's ``airfun``, which interpolates a four-dimensional spline table.
    Passing it in keeps this function testable without the tables and lets a
    fitted surrogate be substituted -- which is exactly what the signomial
    formulations do.

    ``fduo``/``fdus``/``fdut`` are the fractional overspeeds at root, break
    and tip -- the local acceleration over the surface, which raises both the
    section Reynolds number and the perpendicular Mach number.

    ``n`` is the station count and must be even; TASOPT uses 8, noting ~0.07%
    quadrature error against 12 (and ~0.30% at 4).
    """
    if n % 2:
        raise ValueError(f"n must be even; got {n}")

    cosL = math.cos(sweep * math.pi / 180.0)
    AR = b ** 2 / S
    etao = bo / b
    etas = bs / b

    # Kc is the area-weighted chord integral, Kp0 the load-weighted one; Kp
    # additionally carries the root and tip lift losses. Their ratio converts
    # the surface CL into the root section's lift coefficient.
    Kc = (etao
          + 0.5 * (1.0 + lambdas) * (etas - etao)
          + 0.5 * (lambdas + lambdat) * (1.0 - etas))
    Kp0 = (etao
           + 0.5 * (1.0 + gammas) * (etas - etao)
           + 0.5 * (gammas + gammat) * (1.0 - etas))
    Ko = 1.0 / (Kc * AR)
    Kp = Kp0 + fLo * etao + 2.0 * fLt * Ko * gammat * lambdat

    clp1 = (CL - CLhtail) / cosL ** 2 * S / (Kp * b * co)

    return _surfcd2_quadrature(
        n=n, etao=etao, etas=etas, b=b, co=co, cosL=cosL, Mach=Mach,
        clp1=clp1, lambdas=lambdas, lambdat=lambdat, gammas=gammas,
        gammat=gammat, toco=toco, tocs=tocs, toct=toct, Reco=Reco,
        ARe=ARe, aRexp=aRexp, kSuns=kSuns, fexcd=fexcd, airfoil=airfoil,
        fduo=fduo, fdus=fdus, fdut=fdut)


def _surfcd2_quadrature(*, n, etao, etas, b, co, cosL, Mach, clp1,
                        lambdas, lambdat, gammas, gammat,
                        toco, tocs, toct, Reco, ARe, aRexp, kSuns, fexcd,
                        airfoil, fduo=0.0, fdus=0.0, fdut=0.0):
    """Midpoint quadrature over the two panels, inner then outer."""
    clpo = clp1 / (1.0 + fduo) ** 2
    clps = clp1 * gammas / lambdas / (1.0 + fdus) ** 2
    clpt = clp1 * gammat / lambdat / (1.0 + fdut) ** 2

    CDfwing = CDpwing = CDwing = Snorm = 0.0
    half = n // 2

    for panel in (0, 1):
        for i in range(1, half + 1):
            frac = (i - 0.5) / half
            if panel == 0:      # root -> break
                eta = etao * (1.0 - frac) + etas * frac
                deta = (etas - etao) / half
                P = 1.0 - frac + gammas * frac
                C = 1.0 - frac + lambdas * frac
                toc = toco * (1.0 - frac) + tocs * frac
                fdu = fduo * (1.0 - frac) + fdus * frac
            else:               # break -> tip
                eta = etas * (1.0 - frac) + 1.0 * frac
                deta = (1.0 - etas) / half
                P = gammas * (1.0 - frac) + gammat * frac
                C = lambdas * (1.0 - frac) + lambdat * frac
                toc = tocs * (1.0 - frac) + toct * frac
                fdu = fdus * (1.0 - frac) + fdut * frac

            fSuns = math.exp(-(eta - etao) * b / (kSuns * C * 2.0 * co))

            clp = clp1 * (P / C) / (1.0 + fdu) ** 2
            Rec = Reco * C * (1.0 + fdu)
            Mperp = Mach * cosL * (1.0 + fdu)

            cdf1, cdp1, _cdwbar, _cm = airfoil(clp, toc, Mperp)

            Refac = (Rec / ARe) ** aRexp
            cdf = cdf1 * Refac * fexcd * (1.0 + fdu) ** 3
            cdp = (cdp1 * Refac * fexcd * (1.0 + fdu) ** 3
                   * (fSuns + (1.0 - fSuns) * cosL ** 2) * cosL)
            cd = cdf + cdp

            Snorm += C * deta
            CDwing += C * deta * cd
            CDfwing += C * deta * cdf
            CDpwing += C * deta * cdp

    return SurfCD2(clpo=clpo, clps=clps, clpt=clpt,
                   CDfwing=CDfwing / Snorm, CDpwing=CDpwing / Snorm,
                   CDwing=CDwing / Snorm, CDover=0.0)
