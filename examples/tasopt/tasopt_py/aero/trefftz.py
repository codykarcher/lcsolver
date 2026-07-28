"""Trefftz-plane induced drag -- a port of ``trefftz1`` from ``trefftz.f``.

Far behind the aircraft the trailing vorticity has rolled into a flat sheet,
and the induced drag is whatever kinetic energy that sheet carries. This
routine discretises the sheet into horseshoe vortices, computes the downwash
each one induces at the others' control points, and integrates.

Several surfaces at once
------------------------
``nsurf`` streamwise-independent lifting surfaces are handled together --
wing and horizontal tail -- so the tail sees the wing's downwash and the
induced drag is that of the *combination*, not the sum of the two in
isolation. Each surface has its own vertical position ``zcent``, which is what
makes that interaction depend on tail height.

Each surface is described by four spanwise stations: centreline, side of body
``bo``, planform break ``bs``, and tip ``b/2``, with the loading given as a
root value ``po`` and the ratios ``gammas``/``gammat`` at break and tip.

The three coordinate systems
----------------------------
Panel edges are placed in ``t``, an angular coordinate with ``t = 1`` at the
centreline and ``t = 0`` at the tip, so uniform steps in ``t`` cluster panels
towards the tip where the loading gradient is. ``bunch`` warps that further
towards the centre. The physical ``y`` then follows from ``e = cos(pi t/2)``.

Separately, ``yp`` is the *wake* coordinate. The wake leaving the side of body
does not stay at ``bo`` -- it contracts onto the fuselage, and the sheet is
mapped onto an equivalent width ``bop``. Over the fuselage that mapping is the
power law ``yp = bop (y/yo)^((bo/bop)^2)``; outboard it is the
constant-area form ``yp = sqrt(y^2 - yo^2 + yop^2)``. The vortices are placed
at ``yp``, but the lift integral uses them against ``dy = yp(i+1) - yp(i)``,
so the two are consistent.

The tip factor
--------------
``gc`` carries ``sqrt(1 - ec^ktip)``, which rolls the loading off to zero at
the tip. Without it the sheet would end in a finite-strength vortex and the
downwash integral would diverge.

The image system
----------------
Every vortex is mirrored in the ``y = 0`` plane with opposite sign, which
enforces symmetry exactly rather than by relying on the discretisation.

``Lspec``
---------
With ``Lspec`` true the circulations are rescaled so each surface carries
exactly ``CLsurfsp``; the shape of the loading is kept and only its level
changes. That is how the routine is used in trim -- the loads are known and
the drag that goes with them is what is wanted.

``fLo`` is accepted and never used, matching the source's argument list.

Verified against the compiled Fortran; see ``tests/test_trefftz.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["TrefftzResult", "TrefftzError", "trefftz1", "BUNCH"]

# Centre resolution bunching, 0..1. trefftz.f carries three values with two
# commented out; 0.5 is the live one.
BUNCH = 0.5


class TrefftzError(RuntimeError):
    """Panel count exceeds what the source's fixed arrays allow."""


@dataclass
class TrefftzResult:
    CL: float
    CD: float                 # induced drag coefficient
    spanef: float             # span efficiency, CD_elliptical / CD
    CLsurf: list = field(default_factory=list)
    # Panel-level detail, as the Fortran returns it
    yc: list = field(default_factory=list)
    zc: list = field(default_factory=list)
    gc: list = field(default_factory=list)
    vc: list = field(default_factory=list)
    wc: list = field(default_factory=list)
    vnc: list = field(default_factory=list)


def _stretch(t):
    """Invert the bunching warp so the requested ``acos`` station is recovered."""
    if BUNCH <= 0.0:
        return t
    return (1.0 + BUNCH - math.sqrt((1.0 + BUNCH) ** 2
                                    - 4.0 * BUNCH * t)) * 0.5 / BUNCH


def trefftz1(nsurf, npout, npinn, npimg, Sref, bref,
             b, bs, bo, bop, zcent, po, gammat, gammas, fLo, ktip,
             Lspec, CLsurfsp, idim=None) -> TrefftzResult:
    """Induced drag of ``nsurf`` lifting surfaces in the Trefftz plane.

    ``npimg``/``npinn``/``npout`` are the panel counts over the fuselage,
    between side-of-body and break, and between break and tip.
    """
    isum = sum(npout[k] + npinn[k] + npimg[k] + 1 for k in range(nsurf))
    if idim is not None and isum > idim:
        raise TrefftzError(f"passed array overflow; increase idim to {isum}")
    if isum > 360:
        raise TrefftzError(f"local array overflow; increase jdim to {isum}")

    n = isum
    t = [0.0] * n
    y = [0.0] * n
    yp = [0.0] * n
    z = [0.0] * n
    zp = [0.0] * n
    yc = [0.0] * n
    zc = [0.0] * n
    gc = [0.0] * n
    ycp = [0.0] * n
    zcp = [0.0] * n
    ifrst = [0] * nsurf
    ilast = [0] * nsurf

    i = -1                       # 0-based; the source counts from 1
    for isurf in range(nsurf):
        e0, eo = 0.0, bo[isurf] / b[isurf]
        es, e1 = bs[isurf] / b[isurf], 1.0
        eop = bop[isurf] / b[isurf]

        t0 = 1.0
        to = _stretch(math.acos(eo) / (0.5 * math.pi))
        ts = _stretch(math.acos(es) / (0.5 * math.pi))
        t1 = 0.0

        y0, yo, ys, y1 = 0.0, 0.5 * bo[isurf], 0.5 * bs[isurf], 0.5 * b[isurf]
        yop = 0.5 * bop[isurf]
        z0 = zo = zs = z1 = zcent[isurf]
        g0 = go = po[isurf]
        gs = po[isurf] * gammas[isurf]
        g1 = po[isurf] * gammat[isurf]

        k0 = 1
        ko = 1 + npimg[isurf]
        ks = ko + npinn[isurf]
        k1 = ks + npout[isurf]

        i += 1
        ifrst[isurf] = i
        t[i], y[i], z[i], yp[i], zp[i] = t0, y0, z0, y0, z0

        for lo, hi, ka, kb, ea, eb, ya, yb_, za, zb_, ga, gb, over in (
                (t0, to, k0, ko, e0, eo, y0, yo, z0, zo, g0, go, True),
                (to, ts, ko, ks, eo, es, yo, ys, zo, zs, go, gs, False),
                (ts, t1, ks, k1, es, e1, ys, y1, zs, z1, gs, g1, False)):
            for k in range(ka + 1, kb + 1):
                i += 1
                fk = float(k - ka) / float(kb - ka)
                t[i] = lo * (1.0 - fk) + hi * fk
                tc = 0.5 * (t[i - 1] + t[i])
                e = math.cos(0.5 * math.pi
                             * (t[i] + BUNCH * t[i] * (1.0 - t[i])))
                ec = math.cos(0.5 * math.pi * (tc + BUNCH * tc * (1.0 - tc)))
                if eb - ea == 0.0:
                    fi, fc = 1.0, 0.5
                else:
                    fi = (e - ea) / (eb - ea)
                    fc = (ec - ea) / (eb - ea)
                y[i] = ya * (1.0 - fi) + yb_ * fi
                z[i] = za * (1.0 - fi) + zb_ * fi
                yc[i - 1] = ya * (1.0 - fc) + yb_ * fc
                zc[i - 1] = za * (1.0 - fc) + zb_ * fc
                gc[i - 1] = ((ga * (1.0 - fc) + gb * fc)
                             * math.sqrt(1.0 - ec ** ktip))
                if over:
                    # Wake contraction onto the fuselage.
                    yexp = (eo / eop) ** 2
                    yp[i] = yop * (y[i] / yo) ** yexp
                    ycp[i - 1] = yop * (yc[i - 1] / yo) ** yexp
                else:
                    yp[i] = math.sqrt(y[i] ** 2 - yo ** 2 + yop ** 2)
                    ycp[i - 1] = math.sqrt(yc[i - 1] ** 2 - yo ** 2
                                           + yop ** 2)
                zp[i] = z[i]
                zcp[i - 1] = zc[i - 1]

        ilast[isurf] = i
        # Dummy control point separating this surface from the next.
        yc[i] = zc[i] = gc[i] = 0.0

    ii = ilast[nsurf - 1] + 1        # count, for the 0-based loops below

    # Wake vortex strengths are the jumps in bound circulation.
    gw = [0.0] * n
    for isurf in range(nsurf):
        gw[ifrst[isurf]] = 0.0
        for i in range(ifrst[isurf] + 1, ilast[isurf]):
            gw[i] = gc[i - 1] - gc[i]
        gw[ilast[isurf]] = gc[ilast[isurf] - 1]

    if Lspec:
        # Rescale each surface's loading to the lift it is required to carry.
        for isurf in range(nsurf):
            cltest = 0.0
            for i in range(ifrst[isurf], ilast[isurf]):
                cltest += 2.0 * gc[i] * (yp[i + 1] - yp[i]) * bref / (0.5 * Sref)
            gfac = CLsurfsp[isurf] / cltest
            for i in range(ifrst[isurf], ilast[isurf] + 1):
                gc[i] *= gfac
                gw[i] *= gfac

    vc = [0.0] * n
    wc = [0.0] * n
    vnc = [0.0] * n
    CL = 0.0
    CD = 0.0
    CLsurf = [0.0] * nsurf

    for isurf in range(nsurf):
        for i in range(ifrst[isurf], ilast[isurf]):
            dy = yp[i + 1] - yp[i]
            dz = zp[i + 1] - zp[i]
            ds = math.sqrt(dy ** 2 + dz ** 2)
            vsum = wsum = 0.0
            for j in range(ii):
                # The wake vortex itself...
                yb = ycp[i] - yp[j]
                zb = zcp[i] - zp[j]
                rsq = yb ** 2 + zb ** 2
                vsum -= gw[j] * zb / rsq
                wsum += gw[j] * yb / rsq
                # ...and its image in the symmetry plane.
                yb = ycp[i] + yp[j]
                zb = zcp[i] - zp[j]
                rsq = yb ** 2 + zb ** 2
                vsum += gw[j] * zb / rsq
                wsum -= gw[j] * yb / rsq
            vc[i] = vsum * bref / (2.0 * math.pi)
            wc[i] = wsum * bref / (2.0 * math.pi)
            vnc[i] = (dy * wc[i] - dz * vc[i]) / ds
            dCL = 2.0 * gc[i] * dy * bref / (0.5 * Sref)
            dCD = -gc[i] * vnc[i] * ds * bref / (0.5 * Sref)
            CL += dCL
            CD += dCD
            CLsurf[isurf] += dCL

    AR = bref ** 2 / Sref
    spanef = (CL ** 2 / (math.pi * AR)) / CD
    return TrefftzResult(CL=CL, CD=CD, spanef=spanef, CLsurf=CLsurf,
                         yc=yc, zc=zc, gc=gc, vc=vc, wc=wc, vnc=vnc)
