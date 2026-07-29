"""Fuselage boundary layer and wake -- ``fusebl.f``.

Ties :mod:`tasopt_py.aero.axisol` and :mod:`tasopt_py.aero.blax` together and
turns their output into the four numbers the rest of TASOPT wants, all written
back into ``para``:

==================  =========================================================
``DAfsurf``         dissipation area over the fuselage *surface*
``DAfwake``         dissipation area in the *wake*
``KAfTE``           kinetic-energy defect area at the trailing edge
``PAfinf``          far-downstream momentum defect area -- the profile drag
==================  =========================================================

``cdsum`` reads ``PAfinf`` and calls it the fuselage drag: ``CDfuse =
PAfinf/S``. There is no other route to fuselage drag in the program, the
wetted-area one through ``bodycd`` being commented out. ``tfcalc`` reads
``DAfsurf`` and ``KAfTE`` to size the boundary-layer-ingestion credit.

What happens in between
-----------------------
The double-bubble cross-section enters only through its *area*. ``Sfuse`` is
built from ``Rfuse``, ``wfb`` and ``dRfuse``, and ``axisol`` turns it straight
back into the radius of the equivalent round body, ``sqrt(Sfuse/pi)``. Only
the perimeter, ``bbl``, remembers that the section was not a circle -- and
only when ``ifclose`` says the tail closes to an edge, in which case the
lateral extent ``dybl`` adds ``4 dy`` to it.

``rnbl``, the cosine of the contour angle, is differenced from the body
coordinates with an arc-length-weighted central difference, then linearly
*extrapolated* to both endpoints and floored at zero. The extrapolation is
what sets ``rn`` at the nose, where the lateral divergence term in the BL is
strongest.

Squire-Young at the end of the wake
-----------------------------------
The last wake station is not far downstream, so the momentum defect there is
carried to infinity with Squire-Young, ``Pinf = Pend ue^Havg``. The
far-downstream kinetic-energy defect is then simply *set equal* to it
(``Kinf = Pinf``), which is the incompressible far-field identity, and the
dissipation between the last station and infinity follows from the difference.

Note ``hfb = sqrt(Rfuse^2 - wfb^2)`` uses the raw ``wfb`` while the angle just
above it uses ``wfb`` clipped to ``[0, Rfuse]``, so a fuselage with
``wfb > Rfuse`` takes the square root of a negative number. Reproduced --
no shipped case does it.

Verified against the compiled Fortran; see ``tests/test_fusebl.py``.
"""
from __future__ import annotations

import math

from ..atmosphere import atmos
from ..model import indices as I
from .axisol import axisol
from .blax import blax

__all__ = ["fusebl", "NC", "GAMSL"]

#: Control points for the potential-flow problem. Fixed in the source.
NC = 30
#: constants.inc's gamSL, filled at runtime by tasopt.f.
GAMSL = 1.4
#: fbl.inc's array dimension, and so the cap on nbl.
NBLDIM = 60


def fusebl(pari, parg, para) -> None:
    """Solve the fuselage BL at one mission point and store the four areas.

    ``para`` is a single mission-point column, as ``fusebl(pari, parg,
    para(1,ip))`` passes it.
    """
    ifclose = pari[I.IIFCLOSE]

    xnose = parg[I.IGXNOSE]
    xend = parg[I.IGXEND]
    xblend1 = parg[I.IGXBLEND1]
    xblend2 = parg[I.IGXBLEND2]

    Mach = para[I.IAMACH]
    altkm = para[I.IAALT] / 1000.0
    at = atmos(altkm)
    Reunit = Mach * at.a * at.rho / at.mu

    wfb = parg[I.IGWFB]
    Rfuse = parg[I.IGRFUSE]
    dRfuse = parg[I.IGDRFUSE]

    # Cross-sectional area of the double bubble. Note hfb uses the unclipped
    # wfb while thetafb uses the clipped one; see the module docstring.
    wfblim = max(min(wfb, Rfuse), 0.0)
    thetafb = math.asin(wfblim / Rfuse)
    hfb = math.sqrt(Rfuse ** 2 - wfb ** 2)
    sin2t = 2.0 * hfb * wfb / Rfuse ** 2
    Sfuse = ((math.pi + 2.0 * thetafb + sin2t) * Rfuse ** 2
             + 2.0 * Rfuse * dRfuse)

    anose = parg[I.IGANOSE]
    btail = parg[I.IGBTAIL]

    # --- potential-flow surface velocity, from a PG source line -----------
    g = axisol(xnose, xend, xblend1, xblend2, Sfuse, anose, btail, ifclose,
               Mach, NC, NBLDIM)
    nbl, iblte = g.nl, g.ilte
    xbl, zbl, sbl, dybl = g.x, g.z, g.s, g.dy

    # --- fuselage perimeter ------------------------------------------------
    if ifclose == 0:
        bbl = [2.0 * math.pi * z for z in zbl]
    else:
        bbl = [2.0 * math.pi * z + 4.0 * dy for z, dy in zip(zbl, dybl)]

    # --- dr/dn, the cosine of the contour angle from the axis --------------
    rnbl = [0.0] * nbl
    for i in range(1, nbl - 1):
        dxm = xbl[i] - xbl[i - 1]
        dzm = zbl[i] - zbl[i - 1]
        dsm = sbl[i] - sbl[i - 1]

        dxp = xbl[i + 1] - xbl[i]
        dzp = zbl[i + 1] - zbl[i]
        dsp = sbl[i + 1] - sbl[i]

        dxo = (dxm / dsm) * dsp + (dxp / dsp) * dsm
        dzo = (dzm / dsm) * dsp + (dzp / dsp) * dsm

        rnbl[i] = dxo / math.sqrt(dxo ** 2 + dzo ** 2)

    # Extrapolate to the endpoints, then floor at zero.
    rnbl[0] = (rnbl[1] - (0.5 * (sbl[2] + sbl[1]) - sbl[0])
               * (rnbl[2] - rnbl[1]) / (sbl[2] - sbl[1]))
    rnbl[nbl - 1] = (rnbl[nbl - 2]
                     + (sbl[nbl - 1] - 0.5 * (sbl[nbl - 2] + sbl[nbl - 3]))
                     * (rnbl[nbl - 2] - rnbl[nbl - 3])
                     / (sbl[nbl - 2] - sbl[nbl - 3]))
    rnbl[0] = max(rnbl[0], 0.0)
    rnbl[nbl - 1] = max(rnbl[nbl - 1], 0.0)

    # --- viscous/inviscid BL calculation -----------------------------------
    fex = para[I.IAFEXCDF]
    r = blax(nbl, iblte, sbl, bbl, rnbl, g.q, Reunit, Mach, fex)

    gmi = GAMSL - 1.0

    # KE defect at the TE, and the dissipation accumulated over the surface.
    i = iblte - 1                                   # 0-based
    trbl = 1.0 + 0.5 * gmi * Mach ** 2 * (1.0 - r.ue[i] ** 2)
    rhbl = trbl ** (1.0 / gmi)
    KTE = (0.5 * rhbl * r.ue[i] ** 3 * r.ts[i]
           * (bbl[i] + 2.0 * math.pi * r.ds[i]))
    Difsurf = r.ph[i]

    # Momentum and KE defects and accumulated dissipation at the wake end.
    i = nbl - 1
    trbl = 1.0 + 0.5 * gmi * Mach ** 2 * (1.0 - r.ue[i] ** 2)
    rhbl = trbl ** (1.0 / gmi)
    Pend = rhbl * r.ue[i] ** 2 * r.th[i] * (bbl[i] + 2.0 * math.pi * r.ds[i])
    Kend = (0.5 * rhbl * r.ue[i] ** 3 * r.ts[i]
            * (bbl[i] + 2.0 * math.pi * r.ds[i]))
    Difend = r.ph[i]

    # Far-downstream momentum defect via Squire-Young. Vinf = 1 here.
    Hend = r.ds[i] / r.th[i]
    Hinf = 1.0 + gmi * Mach ** 2
    Havg = 0.5 * (Hend + Hinf)
    Pinf = Pend * r.ue[i] ** Havg

    # Far-downstream KE defect, 0.5 rho V^3 Theta*.
    Kinf = Pinf

    # Dissipation downstream of the last wake point.
    tsinf = 2.0 * Kinf
    dcinf = 0.5 * gmi * Mach ** 2 * tsinf
    dDif = Kinf - Kend + 0.5 * (dcinf + r.dc[i]) * (1.0 - r.ue[i])
    Difinf = Difend + dDif

    Difwake = Difinf - Difsurf

    Vinf = 1.0
    qinf = 0.5
    para[I.IADAFSURF] = Difsurf / (qinf * Vinf)
    para[I.IADAFWAKE] = Difwake / (qinf * Vinf)
    para[I.IAKAFTE] = KTE / (qinf * Vinf)
    para[I.IAPAFINF] = Pinf / qinf
