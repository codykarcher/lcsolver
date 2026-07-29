"""Tabulated compressor maps -- ``map_functions.jl`` and the map modules.

**The biggest single change from TASOPT 2.16.** 2.16's ``ecmap`` is an
analytic surface: a handful of fitted constants give pressure ratio and
efficiency as smooth closed-form functions of corrected mass flow and speed.
v3 throws that away and interpolates **pyCycle's tabulated maps** bilinearly
on a (corrected speed ``N``, R-line ``R``) grid.

That is not a refinement, it is a different object, and it changes the
character of the engine model in ways worth stating plainly:

* **The map is no longer smooth.** Bilinear interpolation is C0 -- the
  gradient jumps across every grid line. Anything that differentiates the
  cycle (and ``tfoper``'s Newton solve does, analytically) now sees a
  piecewise-constant derivative in each cell rather than a continuous one.
* **The independent variables changed.** 2.16 works in (mass flow, speed);
  v3 works in (speed, R-line), where R-line is a map-parameterisation
  coordinate with no direct physical meaning. Getting from one to the other
  needs a **2-D Newton inverse** at every evaluation -- see
  :func:`find_NR_inverse`.
* **Off-map behaviour is padding, not physics.** The reference pads each
  grid with an extrapolation border -- zeros, scaled copies of the edge
  column, a 1.5x maximum -- so a solver that wanders off the map gets a
  defined number instead of an error. Those numbers are made up. See
  :data:`LARGE_FAC`.

Scaling
-------
A map is used at whatever design point the engine wants, not the one it was
tabulated at, so everything is scaled by the ratio of the wanted design
value to the map's own: mass flow by ``Wc/mbD``, speed by ``NbD/Nc``,
efficiency by ``epol0/polyeff``, and pressure ratio affinely so that
``pratio = piD`` lands exactly on the map's design ``PR``.

Verified against TASOPT.jl; see ``tests/test_maps.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .map_data import MAPS

__all__ = ["CompressorMap", "FAN_MAP", "LPC_MAP", "HPC_MAP", "MAP_BY_NAME",
           "bilinear", "bilinear_gradient", "find_NR_inverse",
           "compressor_speed_and_efficiency", "LARGE_FAC", "SMALL_FAC"]

#: Off-map padding factors. The extrapolation border is filled with
#: ``LARGE_FAC`` times the grid maximum on the high side and zeros on the
#: low side -- invented values that exist to keep a wandering solver from
#: erroring out, not to model anything.
LARGE_FAC = 1.5
SMALL_FAC = 1.01


def _pad(NcMap, RlineMap, WcMap, PRMap, polyeffMap):
    """Reproduce ``create_extrapolated_maps``.

    Adds one row and one column at each edge. The three quantities are
    padded *differently*, and none of the choices is symmetric:

    * ``Wc``: zero at low R-line and low speed, ``1.01x`` the last column at
      high R-line, ``1.5x`` the global maximum at high speed -- except the
      bottom-left corner, forced back to zero.
    * ``PR``: ``1.01x`` the first column at low R-line, a flat 1.0 at high
      R-line, 0.99 at low speed, ``1.5x`` the maximum at high speed --
      except the bottom-right corner, forced to 0.99.
    * ``polyeff``: **zero on all four edges**. So efficiency falls linearly
      to nothing the moment an iterate leaves the tabulated region, in every
      direction. That is a strong signal to an optimiser and a nonsense
      number to a physicist -- it is a fence, not a model.
    """
    mR = [0.0] + list(RlineMap) + [4.0]
    mN = [0.0] + list(NcMap) + [2.0]

    Wcmax = LARGE_FAC * max(max(r) for r in WcMap)
    mWc = [[0.0] + list(r) + [SMALL_FAC * r[-1]] for r in WcMap]
    ncol = len(mWc[0])
    mWc = [[0.0] * ncol] + mWc + [[Wcmax] * ncol]
    mWc[-1][0] = 0.0

    PRmax = LARGE_FAC * max(max(r) for r in PRMap)
    mPR = [[SMALL_FAC * r[0]] + list(r) + [1.0] for r in PRMap]
    ncol = len(mPR[0])
    mPR = [[0.99] * ncol] + mPR + [[PRmax] * ncol]
    mPR[-1][-1] = 0.99

    mPE = [[0.0] + list(r) + [0.0] for r in polyeffMap]
    ncol = len(mPE[0])
    mPE = [[0.0] * ncol] + mPE + [[0.0] * ncol]
    return mN, mR, mWc, mPR, mPE


def _cell(xs, x):
    """Index ``i`` with ``xs[i] <= x <= xs[i+1]``, clamped to the grid.

    Outside the padded grid the value is held constant at the edge, which is
    what ``Interpolations.jl``'s gridded linear interpolant does by default
    when handed an in-bounds-clamped coordinate.
    """
    if x <= xs[0]:
        return 0
    if x >= xs[-1]:
        return len(xs) - 2
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


def bilinear(xs, ys, Z, x, y) -> float:
    """Bilinear interpolation of ``Z[i][j]`` at ``(x, y)``."""
    i, j = _cell(xs, x), _cell(ys, y)
    tx = (x - xs[i]) / (xs[i + 1] - xs[i])
    ty = (y - ys[j]) / (ys[j + 1] - ys[j])
    return ((1 - tx) * ((1 - ty) * Z[i][j] + ty * Z[i][j + 1])
            + tx * ((1 - ty) * Z[i + 1][j] + ty * Z[i + 1][j + 1]))


def _cell_grad(xs, x):
    """Cell index for a *derivative* at ``x``.

    Differs from :func:`_cell` at the knots. A linear interpolant has no
    derivative exactly on a grid line -- the left and right cells disagree --
    and ``Interpolations.jl`` resolves it by taking the **left** cell. The
    values agree either way, by continuity; only the gradients do not, and
    they are what the Newton solve uses.
    """
    if x <= xs[0]:
        return 0
    if x >= xs[-1]:
        return len(xs) - 2
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] < x:
            lo = mid
        else:
            hi = mid
    return lo


def bilinear_gradient(xs, ys, Z, x, y) -> tuple:
    """``(dZ/dx, dZ/dy)``.

    Piecewise constant in each variable across a cell, and **discontinuous
    at every grid line** -- the price of a linear interpolant. A Newton
    solve that steps across a grid line sees the Jacobian change abruptly.
    """
    i, j = _cell_grad(xs, x), _cell_grad(ys, y)
    dx = xs[i + 1] - xs[i]
    dy = ys[j + 1] - ys[j]
    tx = (x - xs[i]) / dx
    ty = (y - ys[j]) / dy
    dzdx = ((1 - ty) * (Z[i + 1][j] - Z[i][j])
            + ty * (Z[i + 1][j + 1] - Z[i][j + 1])) / dx
    dzdy = ((1 - tx) * (Z[i][j + 1] - Z[i][j])
            + tx * (Z[i + 1][j + 1] - Z[i + 1][j])) / dy
    return dzdx, dzdy


@dataclass
class CompressorMap:
    """One tabulated map plus its design-point defaults."""
    name: str
    Nc: float
    Rline: float
    Wc: float
    PR: float
    polyeff: float
    mN: list
    mR: list
    mWc: list
    mPR: list
    mPE: list

    def Wc_at(self, N, R):
        return bilinear(self.mN, self.mR, self.mWc, N, R)

    def PR_at(self, N, R):
        return bilinear(self.mN, self.mR, self.mPR, N, R)

    def polyeff_at(self, N, R):
        return bilinear(self.mN, self.mR, self.mPE, N, R)

    def grad_Wc(self, N, R):
        return bilinear_gradient(self.mN, self.mR, self.mWc, N, R)

    def grad_PR(self, N, R):
        return bilinear_gradient(self.mN, self.mR, self.mPR, N, R)

    def grad_polyeff(self, N, R):
        return bilinear_gradient(self.mN, self.mR, self.mPE, N, R)


def _build(name: str) -> CompressorMap:
    d = MAPS[name]
    mN, mR, mWc, mPR, mPE = _pad(d["NcMap"], d["RlineMap"], d["WcMap"],
                                 d["PRMap"], d["polyeff_Map"])
    f = d["defaults"]
    return CompressorMap(name, f["Nc"], f["Rline"], f["Wc"], f["PR"],
                         f["polyeff"], mN, mR, mWc, mPR, mPE)


FAN_MAP = _build("Fan")
LPC_MAP = _build("LPC")
HPC_MAP = _build("HPC")
MAP_BY_NAME = {"Fan": FAN_MAP, "LPC": LPC_MAP, "HPC": HPC_MAP}


#: The padded grids span these ranges; iterates are projected into them.
#: Without this an inverse that steps outside simply diverges -- 17 of the
#: 48 reference operating points do exactly that from the default guess.
N_BOUNDS = (1.0e-4, 1.9999)
R_BOUNDS = (1.0e-4, 3.9999)


def _clamp(N, R):
    return (min(max(N, N_BOUNDS[0]), N_BOUNDS[1]),
            min(max(R, R_BOUNDS[0]), R_BOUNDS[1]))


def find_NR_inverse(m: CompressorMap, Wc: float, PR: float,
                    Ng: float = 0.5, Rg: float = 2.0,
                    tol: float = 1e-12, itmax: int = 100) -> tuple:
    """Invert the map: given ``(Wc, PR)`` find ``(N, R)`` and the Jacobian.

    Returns ``(N, R, dN_dWc, dN_dPR, dR_dWc, dR_dPR)``.

    This is the price v3 pays for tabulating the map in coordinates the
    cycle does not solve in: every map evaluation carries a nested 2-D root
    find.

    Two things make it work, and both are load-bearing:

    * **Projection onto the padded grid.** Every iterate is clamped into
      ``N_BOUNDS`` x ``R_BOUNDS`` before the residual is evaluated. Without
      it a Newton step off a flat padded region flies to infinity -- 17 of
      the 48 reference operating points diverge immediately from the
      default guess of (0.5, 2.0), including every high-pressure-ratio HPC
      point.
    * **Damping.** The interpolant is only C0, so the Jacobian is piecewise
      constant and a full Newton step routinely overshoots into a cell
      where it is wrong. Backtracking on the residual norm fixes that; the
      reference gets the same effect from NLsolve's trust region.

    A second attempt starts from (1.0, 2.0) if the first fails, exactly as
    the reference does.
    """
    def residual(N, R):
        return m.Wc_at(N, R) - Wc, m.PR_at(N, R) - PR

    def attempt(N, R):
        N, R = _clamp(N, R)
        fW, fP = residual(N, R)
        for _ in range(itmax):
            a, b = m.grad_Wc(N, R)
            c, d = m.grad_PR(N, R)
            det = a * d - b * c
            if det == 0.0:
                return None
            dN = (-fW * d + fP * b) / det
            dR = (-a * fP + c * fW) / det
            norm0 = math.hypot(fW, fP)
            step = 1.0
            for _ in range(40):            # backtrack on the residual norm
                Nn, Rn = _clamp(N + step * dN, R + step * dR)
                gW, gP = residual(Nn, Rn)
                if math.hypot(gW, gP) < norm0 or norm0 == 0.0:
                    break
                step *= 0.5
            else:
                return None
            moved = abs(Nn - N) + abs(Rn - R)
            N, R, fW, fP = Nn, Rn, gW, gP
            scale = max(1.0, abs(Wc), abs(PR))
            if math.hypot(fW, fP) <= tol * scale:
                return N, R
            if moved == 0.0:
                # Machine precision reached: the step no longer changes the
                # iterate. Accept if the residual is small, since no further
                # progress is representable.
                return (N, R) if math.hypot(fW, fP) <= 1e-8 * scale else None
        return None

    got = attempt(Ng, Rg) or attempt(1.0, 2.0)
    if got is None:
        raise ValueError(
            f"{m.name} map inverse did not converge for Wc={Wc:.6g}, "
            f"PR={PR:.6g} from (N, R) = ({Ng}, {Rg}) or (1.0, 2.0). The "
            "bilinear map is only C0, so the Jacobian is piecewise constant "
            "and Newton can stall between adjacent cells.")

    N, R = got
    a, b = m.grad_Wc(N, R)
    c, d = m.grad_PR(N, R)
    det = a * d - b * c
    if det == 0.0:
        raise ValueError(
            f"{m.name} map inverse: singular Jacobian at the solution "
            f"N={N:.6g}, R={R:.6g}; the map is flat there and the "
            "sensitivities are undefined")
    return N, R, d / det, -b / det, -c / det, a / det


def compressor_speed_and_efficiency(m: CompressorMap, pratio: float,
                                    mb: float, piD: float, mbD: float,
                                    NbD: float, epol0: float,
                                    Ng: float = 0.5, Rg: float = 2.0):
    """``(Nb, epol, dNb_dpi, dNb_dmb, depol_dpi, depol_dmb, N, R)``.

    Scales the caller's design point onto the map's own, inverts, reads the
    efficiency, and scales back -- carrying derivatives throughout, because
    ``tfoper``'s Newton needs them.

    Note the pressure-ratio scaling is **affine, not multiplicative**:
    ``PR = 1 + (pratio-1)/(piD-1) (PR_map - 1)``. It fixes the no-work point
    ``pratio = 1`` as well as the design point, which a plain ratio would
    not -- a compressor doing nothing must have a pressure ratio of one on
    any map.
    """
    dWc_dmb = m.Wc / mbD
    dNb_dN = NbD / m.Nc
    depol_dep = epol0 / m.polyeff

    Wc = mb * dWc_dmb
    dPR_dpi = (m.PR - 1.0) / (piD - 1.0)
    PR = 1.0 + (pratio - 1.0) * dPR_dpi

    N, R, dN_dw, dN_dpr, dR_dw, dR_dpr = find_NR_inverse(m, Wc, PR,
                                                         Ng=Ng, Rg=Rg)
    Nb = N * dNb_dN
    dNb_dmb = dN_dw * dWc_dmb * dNb_dN
    dNb_dpi = dN_dpr * dPR_dpi * dNb_dN

    ep = m.polyeff_at(N, R)
    dep_dN, dep_dR = m.grad_polyeff(N, R)
    epol = ep * depol_dep
    depol_dw = (dep_dN * dN_dw + dep_dR * dR_dw) * depol_dep
    depol_dpr = (dep_dN * dN_dpr + dep_dR * dR_dpr) * depol_dep
    return (Nb, epol, dNb_dpi, dNb_dmb, depol_dpr * dPR_dpi,
            depol_dw * dWc_dmb, N, R)
