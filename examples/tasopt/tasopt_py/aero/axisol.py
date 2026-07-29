"""Compressible potential flow about a quasi-axisymmetric body -- ``axisol.f``.

A piecewise-constant source line along the axis, with Prandtl-Glauert applied
by stretching the lateral coordinates by ``beta = sqrt(1 - M^2)``. Source
strengths are solved from flow tangency at the control points, then the
surface speed is evaluated at every geometry point and handed to the boundary
layer.

The body shape
--------------
Three sections, defined by exponents rather than a spline:

* nose, ``z = Rcyl (1 - f^anose)^(1/anose)`` with ``f`` running 1 -> 0
  from the nose to the first blend point -- a superellipse;
* a constant-radius barrel between the blend points;
* tail, ``z = Rcyl (1 - f^btail)``.

``Rcyl`` is the radius of the *equivalent round body*, ``sqrt(Amax/pi)``, so a
double-bubble fuselage enters only through its cross-sectional area. The
lateral extent of a non-circular tail is carried separately in ``dy``, and
where it is nonzero the singularity becomes a source *panel* (``vsurf``)
rather than a line (``vline``).

Two details that are not obvious
---------------------------------
The trailing-edge point is placed at ``0.25`` of the previous point's radius
rather than on the body, and the wake points sit at ``0.125`` of it. Both are
arbitrary small numbers that keep the source line off the axis, where the
line-source velocity is singular (``hbsq -> 0`` in :func:`vline`).

Wake x-stations mirror the surface ones about the tail:
``x[i] = 2*xend - x[2*ilte - i]``, so the wake is spaced like the aft body.

Verified against the compiled Fortran; see ``tests/test_axisol.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["AxisolResult", "axisol", "vline", "vsurf", "COSINE_SPACING"]

QOPI = 0.0795774715459476678      # 1/(4 pi)
#: axisol.f carries a uniform-spacing option, commented out; cosine is live.
COSINE_SPACING = True


@dataclass
class AxisolResult:
    nl: int                       # number of surface + wake points
    ilte: int                     # index of the trailing-edge point (1-based)
    x: list = field(default_factory=list)
    z: list = field(default_factory=list)
    s: list = field(default_factory=list)     # arc length
    dy: list = field(default_factory=list)    # half-width of an edge-type tail
    q: list = field(default_factory=list)     # speed, V/Vinf


def vline(x, y, z, x1, x2, b):
    """Velocity from a unit constant source line on the axis over ``x1..x2``.

    ``b`` is the Prandtl-Glauert factor; the lateral coordinates are stretched
    by it and the result unstretched, which is what makes a compressible
    solution out of an incompressible kernel.
    """
    bsq = b ** 2
    yb, zb = y * b, z * b
    R1 = math.sqrt((x1 - x) ** 2 + yb ** 2 + zb ** 2)
    R2 = math.sqrt((x2 - x) ** 2 + yb ** 2 + zb ** 2)
    hbsq = yb ** 2 + zb ** 2
    u0 = QOPI * (1.0 / R2 - 1.0 / R1) / bsq
    v0 = QOPI * ((x2 - x) / R2 - (x1 - x) / R1) * yb / hbsq / b
    w0 = QOPI * ((x2 - x) / R2 - (x1 - x) / R1) * zb / hbsq / b
    return u0, v0, w0


def vsurf(x, y, z, x1, x2, y1, y2, b):
    """Velocity from a unit constant source panel over ``x1..x2``, ``y1..y2``."""
    bsq = b ** 2
    y1b, y2b, zb = y1 * b, y2 * b, z * b
    R11 = math.sqrt((x1 - x) ** 2 + y1b ** 2 + zb ** 2)
    R12 = math.sqrt((x1 - x) ** 2 + y2b ** 2 + zb ** 2)
    R21 = math.sqrt((x2 - x) ** 2 + y1b ** 2 + zb ** 2)
    R22 = math.sqrt((x2 - x) ** 2 + y2b ** 2 + zb ** 2)
    g11, g12 = math.log(abs(y1b + R11)), math.log(abs(y2b + R12))
    g21, g22 = math.log(abs(y1b + R21)), math.log(abs(y2b + R22))
    h11, h12 = math.atanh((x1 - x) / R11), math.atanh((x1 - x) / R12)
    h21, h22 = math.atanh((x2 - x) / R21), math.atanh((x2 - x) / R22)
    t11 = math.atan2((x1 - x) * y1b, zb * R11)
    t12 = math.atan2((x1 - x) * y2b, zb * R12)
    t21 = math.atan2((x2 - x) * y1b, zb * R21)
    t22 = math.atan2((x2 - x) * y2b, zb * R22)
    u0 = QOPI * (g22 - g21 - g12 + g11) / (y2b - y1b) / bsq
    v0 = QOPI * (h22 - h21 - h12 + h11) / (y2b - y1b) / b
    w0 = QOPI * (t22 - t21 - t12 + t11) / (y2b - y1b) / b
    return u0, v0, w0


def _gauss(a, b):
    """Dense solve, standing in for ``gaussn``. Partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(m[i][k]))
        if abs(m[p][k]) < 1e-300:
            raise ZeroDivisionError("singular source-strength system")
        m[k], m[p] = m[p], m[k]
        for i in range(k + 1, n):
            f = m[i][k] / m[k][k]
            if f != 0.0:
                for j in range(k, n + 1):
                    m[i][j] -= f * m[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) \
            / m[i][i]
    return x


def _shape(xx, xnose, xend, xblend1, xblend2, Rcyl, anose, btail):
    """Body radius and its x-derivative at station ``xx``."""
    if xx < xblend1:
        f = 1.0 - (xx - xnose) / (xblend1 - xnose)
        f_x = -1.0 / (xblend1 - xnose)
        z = Rcyl * (1.0 - f ** anose) ** (1.0 / anose)
        z_x = -z / (1.0 - f ** anose) * f ** (anose - 1.0) * f_x
        return z, z_x, "nose"
    if xx < xblend2:
        return Rcyl, 0.0, "barrel"
    f = (xx - xblend2) / (xend - xblend2)
    f_x = 1.0 / (xend - xblend2)
    z = Rcyl * (1.0 - f ** btail)
    z_x = -Rcyl * btail * f ** (btail - 1.0) * f_x
    return z, z_x, "tail"


def axisol(xnose, xend, xblend1, xblend2, Amax, anose, btail, iclose,
           Mach, nc, nldim=None) -> AxisolResult:
    """Surface speed over a quasi-axisymmetric body and its wake.

    ``nc`` control points on the body; the result carries ``nc + nc//2 + 2``
    points, the first ``nc + 1`` of them on the surface.
    """
    if nc > 40:
        raise ValueError(f"axisol: local-array overflow, increase idim to {nc}")
    ilte = nc + 1
    nl = nc + nc // 2 + 2
    if nldim is not None and nl > nldim:
        raise ValueError(f"axisol: array overflow, increase nldim to {nl}")

    beta = math.sqrt(1.0 - Mach ** 2)
    Rcyl = math.sqrt(Amax / math.pi)          # equivalent round body

    x = [0.0] * nl
    z = [0.0] * nl
    dy = [0.0] * nl

    def frac_of(t):
        return t if not COSINE_SPACING else 0.5 * (1.0 - math.cos(math.pi * t))

    for i in range(ilte):
        frac = frac_of(float(i) / float(ilte - 1))
        x[i] = xnose * (1.0 - frac) + xend * frac
        if i == 0 or i == ilte - 1:
            z[i] = 0.0
        else:
            z[i] = _shape(x[i], xnose, xend, xblend1, xblend2, Rcyl,
                          anose, btail)[0]
        if iclose == 0 or x[i] < xblend2:
            dy[i] = 0.0
        else:
            dy[i] = Rcyl - z[i]

    # Keep the TE point off the axis, where the line source is singular.
    z[ilte - 1] = 0.25 * z[ilte - 2]
    dy[ilte - 1] = 0.0 if iclose == 0 else Rcyl - z[ilte - 1]

    for i in range(ilte, nl):
        # Wake stations mirror the surface ones about the tail.
        x[i] = 2.0 * xend - x[2 * ilte - i - 2]
        z[i] = 0.125 * z[ilte - 2]
        dy[i] = 0.0 if iclose == 0 else Rcyl - z[i]

    s = [0.0] * nl
    for i in range(nl - 1):
        s[i + 1] = s[i] + math.sqrt((x[i + 1] - x[i]) ** 2
                                    + (z[i + 1] - z[i]) ** 2)

    # --- control points, on the actual surface ---------------------------
    xc = [0.0] * nc
    zc = [0.0] * nc
    dyc = [0.0] * nc
    nxc = [0.0] * nc
    nzc = [0.0] * nc
    for i in range(nc):
        tc = 0.5 * (float(i) / float(ilte - 1) + float(i + 1) / float(ilte - 1))
        frac = frac_of(tc)
        xc[i] = xnose * (1.0 - frac) + xend * frac
        zc[i], z_x, where = _shape(xc[i], xnose, xend, xblend1, xblend2,
                                   Rcyl, anose, btail)
        dyc[i] = 0.0 if (iclose == 0 or where != "tail") else Rcyl - zc[i]
        # Outward normal of the meridian, pointing away from the axis.
        nxc[i] = z_x / math.sqrt(1.0 + z_x ** 2)
        nzc[i] = -1.0 / math.sqrt(1.0 + z_x ** 2)

    # --- flow tangency at the control points -----------------------------
    aa = [[0.0] * nc for _ in range(nc)]
    rr = [0.0] * nc
    for i in range(nc):
        for j in range(nc):
            if dyc[j] == 0.0:
                u0, v0, w0 = vline(xc[i], 0.0, zc[i], x[j], x[j + 1], beta)
            else:
                u0, v0, w0 = vsurf(xc[i], 0.0, zc[i], x[j], x[j + 1],
                                   -dyc[j], dyc[j], beta)
            aa[i][j] += u0 * nxc[i] + w0 * nzc[i]
        rr[i] = nxc[i]

    src = [-v for v in _gauss(aa, rr)]

    # --- surface and wake speeds ------------------------------------------
    q = [0.0] * nl
    for i in range(1, nl):
        ul = vl = wl = 0.0
        for j in range(nc):
            if dyc[j] == 0.0:
                u0, v0, w0 = vline(x[i], 0.0, z[i], x[j], x[j + 1], beta)
            else:
                u0, v0, w0 = vsurf(x[i], 0.0, z[i], x[j], x[j + 1],
                                   -dyc[j], dyc[j], beta)
            ul += u0 * src[j]
            vl += v0 * src[j]
            wl += w0 * src[j]
        ul += 1.0                       # freestream
        q[i] = math.sqrt(ul ** 2 + vl ** 2 + wl ** 2)

    return AxisolResult(nl=nl, ilte=ilte, x=x, z=z, s=s, dy=dy, q=q)
