"""Airfoil database: table reader and tri-cubic evaluation.

Ports ``spline``/``trisol`` from ``spline.f``, ``airtable`` from
``airtable.f`` and ``airfun`` from ``airfun.f``.

TASOPT does not compute airfoil section drag; it looks it up. A database file
tabulates three functions of (Mach, cl, tau=thickness ratio) --

    ``cdf`` friction drag, ``cdp`` pressure drag, ``cm`` pitching moment

-- on a structured grid, and this module interpolates it. The tables ship in
``air/`` (``C.air``, ``E.air``, ``B200.air``, ``Cl.air``); each was produced by
running a viscous airfoil code over the grid.

Why the cross-derivatives are precomputed
-----------------------------------------
:func:`airtable` builds not just the value array ``A`` but seven derivative
arrays: ``A_M``, ``A_cl``, ``A_tau`` and all four cross terms up to
``A_M_cl_tau``. Interpolation then never differentiates -- it evaluates a
Hermite form that already has the slopes it needs, once per axis. That makes
:func:`airfun` cheap and, more importantly, gives a result that is smooth in
all three variables, which matters because the sizing loop differentiates
through it.

The mixed second derivatives are built twice, cl-then-tau and tau-then-cl, and
**averaged**. Splining in a different order gives a different answer, and the
source declines to prefer either.

Two behaviours worth knowing about
----------------------------------
* ``cdw``, the wave-drag output, is set to zero unconditionally. It is not
  computed from the tables and never has been in this version. Compressibility
  drag reaches the aircraft through ``surfcd``'s own correlation instead.
* Outside the tabulated ``cl`` and ``tau`` range there is no extrapolation
  guard, but a quadratic penalty is added to ``cdp`` -- ``1.0*(dcl)^2`` and
  ``25.0*(dtau)^2``. That is an optimiser fence, not physics: it makes leaving
  the database expensive without making it impossible. Results outside the box
  are meaningless as drag, so the penalty is reported separately here.

Verified against the compiled Fortran; see ``tests/test_airfoil.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["AirfoilTable", "SectionDrag", "airtable", "airfun", "spline",
           "trisol", "CL_PENALTY", "TAU_PENALTY"]

CL_PENALTY = 1.0      # per unit cl^2 outside the tabulated cl range
TAU_PENALTY = 25.0    # per unit tau^2 outside the tabulated thickness range


def trisol(a, b, c, d):
    """Tridiagonal solve, in place on ``d`` -- a port of ``TRISOL``.

    Plain Thomas algorithm, no pivoting; the spline systems it is used on are
    diagonally dominant so none is needed.
    """
    kk = len(d)
    a, b, c, d = list(a), list(b), list(c), list(d)
    for k in range(1, kk):
        km = k - 1
        c[km] /= a[km]
        d[km] /= a[km]
        a[k] -= b[k] * c[km]
        d[k] -= b[k] * d[km]
    d[kk - 1] /= a[kk - 1]
    for k in range(kk - 2, -1, -1):
        d[k] -= c[k] * d[k + 1]
    return d


def spline(x, s):
    """Cubic-spline slopes ``dx/ds`` at the knots -- a port of ``SPLINE``.

    Returns first derivatives, not second, which is what the Hermite
    evaluation in :func:`airfun` wants. End condition is zero third
    derivative; with only two points it degrades to zero second derivative,
    giving a straight line.
    """
    n = len(x)
    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n
    xs = [0.0] * n
    for i in range(1, n - 1):
        dsm = s[i] - s[i - 1]
        dsp = s[i + 1] - s[i]
        b[i] = dsp
        a[i] = 2.0 * (dsm + dsp)
        c[i] = dsm
        xs[i] = 3.0 * ((x[i + 1] - x[i]) * dsm / dsp
                       + (x[i] - x[i - 1]) * dsp / dsm)

    a[0] = 1.0
    c[0] = 1.0
    xs[0] = 2.0 * (x[1] - x[0]) / (s[1] - s[0])

    b[n - 1] = 1.0
    a[n - 1] = 1.0
    xs[n - 1] = 2.0 * (x[n - 1] - x[n - 2]) / (s[n - 1] - s[n - 2])

    if n == 2:
        b[n - 1] = 1.0
        a[n - 1] = 2.0
        xs[n - 1] = 3.0 * (x[n - 1] - x[n - 2]) / (s[n - 1] - s[n - 2])

    return trisol(a, b, c, xs)


@dataclass
class AirfoilTable:
    """A tabulated airfoil, with all the spline derivatives precomputed."""
    AMa: list                 # Mach knots
    Acl: list                 # lift coefficient knots
    Atau: list                # thickness ratio knots
    ARe: float                # Reynolds number the database was run at
    nAfun: int                # number of tabulated functions (3: cdf, cdp, cm)
    A: dict = field(default_factory=dict)      # (i,j,k,l) -> value
    A_M: dict = field(default_factory=dict)
    A_cl: dict = field(default_factory=dict)
    A_tau: dict = field(default_factory=dict)
    A_M_cl: dict = field(default_factory=dict)
    A_M_tau: dict = field(default_factory=dict)
    A_cl_tau: dict = field(default_factory=dict)
    A_M_cl_tau: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SectionDrag:
    cdf: float          # friction drag coefficient
    cdp: float          # pressure drag coefficient, plus any out-of-box penalty
    cdw: float          # wave drag -- always 0.0, see the module docstring
    cm: float           # pitching moment coefficient
    penalty: float = 0.0   # the part of cdp that is the out-of-range fence


def airtable(fname) -> AirfoilTable:
    """Read an airfoil database file and build its spline derivative arrays."""
    text = Path(fname).read_text().split()
    pos = 0

    def take(n=1):
        nonlocal pos
        out = text[pos:pos + n]
        pos += n
        return out

    nAMa, nAcl, nAtau, nAfun = (int(v) for v in take(4))
    AMa = [float(v) for v in take(nAMa)]
    Acl = [float(v) for v in take(nAcl)]
    Atau = [float(v) for v in take(nAtau)]
    ARe = float(take(1)[0])

    A = {}
    # File order is tau slowest, then cl, then Mach -- the transpose of how
    # the arrays are indexed.
    for k in range(nAtau):
        for j in range(nAcl):
            for i in range(nAMa):
                for l, v in enumerate(take(nAfun)):
                    A[i, j, k, l] = float(v)

    t = AirfoilTable(AMa=AMa, Acl=Acl, Atau=Atau, ARe=ARe, nAfun=nAfun, A=A)

    for l in range(nAfun):
        # d/dMach along each (cl, tau) line
        for k in range(nAtau):
            for j in range(nAcl):
                col = [A[i, j, k, l] for i in range(nAMa)]
                for i, v in enumerate(spline(col, AMa)):
                    t.A_M[i, j, k, l] = v

        # d/dcl, and the mixed d2/dMach dcl
        for k in range(nAtau):
            for i in range(nAMa):
                f = [A[i, j, k, l] for j in range(nAcl)]
                fM = [t.A_M[i, j, k, l] for j in range(nAcl)]
                fcl, fMcl = spline(f, Acl), spline(fM, Acl)
                for j in range(nAcl):
                    t.A_cl[i, j, k, l] = fcl[j]
                    t.A_M_cl[i, j, k, l] = fMcl[j]

        # d/dtau, and the mixed d2/dMach dtau
        for j in range(nAcl):
            for i in range(nAMa):
                f = [A[i, j, k, l] for k in range(nAtau)]
                fM = [t.A_M[i, j, k, l] for k in range(nAtau)]
                ftau, fMtau = spline(f, Atau), spline(fM, Atau)
                for k in range(nAtau):
                    t.A_tau[i, j, k, l] = ftau[k]
                    t.A_M_tau[i, j, k, l] = fMtau[k]

        # The cl-tau cross term, built both ways round and averaged. Splining
        # cl-then-tau and tau-then-cl do not agree, and the source picks
        # neither.
        for i in range(nAMa):
            for j in range(nAcl):
                f = [t.A_cl[i, j, k, l] for k in range(nAtau)]
                fM = [t.A_M_cl[i, j, k, l] for k in range(nAtau)]
                ftau, fMtau = spline(f, Atau), spline(fM, Atau)
                for k in range(nAtau):
                    t.A_cl_tau[i, j, k, l] = ftau[k]
                    t.A_M_cl_tau[i, j, k, l] = fMtau[k]
            for k in range(nAtau):
                f = [t.A_tau[i, j, k, l] for j in range(nAcl)]
                fM = [t.A_M_tau[i, j, k, l] for j in range(nAcl)]
                fcl, fMcl = spline(f, Acl), spline(fM, Acl)
                for j in range(nAcl):
                    t.A_cl_tau[i, j, k, l] = 0.5 * (t.A_cl_tau[i, j, k, l]
                                                    + fcl[j])
                    t.A_M_cl_tau[i, j, k, l] = 0.5 * (t.A_M_cl_tau[i, j, k, l]
                                                      + fMcl[j])
    return t


def _bracket(v, knots):
    """Bisection for the interval containing *v*, clamped to the end intervals.

    Matches the source: no range check, so a value outside the table lands in
    the first or last interval and is extrapolated by the cubic.
    """
    lo, hi = 0, len(knots) - 1
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if v < knots[mid]:
            hi = mid
        else:
            lo = mid
    return hi


def _hermite(t, d, f0, f1, s0, s1):
    """One axis of the tri-cubic: values ``f`` and slopes ``s`` over span ``d``."""
    fxm = d * s0 - f1 + f0
    fxo = d * s1 - f1 + f0
    return (t * f1 + (1.0 - t) * f0
            + t * (1.0 - t) * ((1.0 - t) * fxm - t * fxo))


def airfun(cl: float, tau: float, Mach: float, t: AirfoilTable) -> SectionDrag:
    """Section drag and moment at ``(Mach, cl, tau)`` -- a port of ``airfun``."""
    io = _bracket(Mach, t.AMa)
    im = io - 1
    dMa = t.AMa[io] - t.AMa[im]
    tMa = (Mach - t.AMa[im]) / dMa

    jo = _bracket(cl, t.Acl)
    jm = jo - 1
    dcl = t.Acl[jo] - t.Acl[jm]
    tcl = (cl - t.Acl[jm]) / dcl

    ko = _bracket(tau, t.Atau)
    km = ko - 1
    dtau = t.Atau[ko] - t.Atau[km]
    ttau = (tau - t.Atau[km]) / dtau

    out = []
    for l in range(t.nAfun):
        Ai = {}
        Ai_cl = {}
        Ai_tau = {}
        Ai_cl_tau = {}
        # Collapse the Mach axis first, keeping cl and tau slopes alive.
        for jd in (0, 1):
            for kd in (0, 1):
                j = jo + jd - 1
                k = ko + kd - 1
                for store, val, slope in (
                        (Ai, t.A, t.A_M),
                        (Ai_cl, t.A_cl, t.A_M_cl),
                        (Ai_tau, t.A_tau, t.A_M_tau),
                        (Ai_cl_tau, t.A_cl_tau, t.A_M_cl_tau)):
                    store[jd, kd] = _hermite(
                        tMa, dMa, val[im, j, k, l], val[io, j, k, l],
                        slope[im, j, k, l], slope[io, j, k, l])

        # Then the cl axis, keeping the tau slope alive.
        Aij = {}
        Aij_tau = {}
        for kd in (0, 1):
            Aij[kd] = _hermite(tcl, dcl, Ai[0, kd], Ai[1, kd],
                               Ai_cl[0, kd], Ai_cl[1, kd])
            Aij_tau[kd] = _hermite(tcl, dcl, Ai_tau[0, kd], Ai_tau[1, kd],
                                   Ai_cl_tau[0, kd], Ai_cl_tau[1, kd])

        # Finally tau.
        out.append(_hermite(ttau, dtau, Aij[0], Aij[1],
                            Aij_tau[0], Aij_tau[1]))

    cdf, cdp, cm = out[0], out[1], out[2]

    # Out-of-database fence. Quadratic in how far outside you are.
    penalty = 0.0
    if cl < t.Acl[0]:
        penalty += CL_PENALTY * (cl - t.Acl[0]) ** 2
    elif cl > t.Acl[-1]:
        penalty += CL_PENALTY * (cl - t.Acl[-1]) ** 2
    if tau < t.Atau[0]:
        penalty += TAU_PENALTY * (tau - t.Atau[0]) ** 2
    elif tau > t.Atau[-1]:
        penalty += TAU_PENALTY * (tau - t.Atau[-1]) ** 2

    return SectionDrag(cdf=cdf, cdp=cdp + penalty, cdw=0.0, cm=cm,
                       penalty=penalty)
