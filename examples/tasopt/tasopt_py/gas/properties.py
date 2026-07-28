"""Thermally-perfect gas properties — port of TASOPT ``src/gasfun.f``.

Computes s(T), h(T), cp(T) and R for a gas with temperature-dependent
specific heat, by cubic Hermite interpolation of tabulated data.

Two details of the original that are easy to get wrong when porting:

**Enthalpy interpolates in T, entropy interpolates in log T.** That is not an
arbitrary choice — ``dh = cp dT`` but ``ds = cp dT/T = cp d(ln T)``, so using
``log T`` as the independent variable for s makes ``cp`` the correct Hermite
slope in both cases. The table therefore carries both ``t`` and its log
``tl``, and the two interpolations use different fractional coordinates
(``f`` and ``fl``).

**The Hermite form is written with cp as the endpoint slope.** For enthalpy
the tabulated slope is ``cp`` itself; for cp the slope is the separately
tabulated ``cpt = dcp/dT``.

The adiabatic pressure change over a process 1..2 with polytropic efficiency
``epol`` is, as the original's header notes::

    p2 = p1 exp[   epol   (s2-s1)/R ]    compression
    p2 = p1 exp[ (1/epol) (s2-s1)/R ]    expansion

Gas indices follow the Fortran's ``igas``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from tasopt_py.gas.tables import GASES

# igas -> table key, exactly as the if-blocks in gasfun.f
IGAS = {
    1: "N2", 2: "O2", 3: "CO2", 4: "H2O", 5: "AR",
    11: "CH4", 12: "C2H6", 13: "C3H8", 14: "C4H10",
    18: "C8H18", 24: "C14H30",
}

# number of C,H,O,N atoms per molecule (gaschem in gasfun.f)
NCHON = {
    "N2": (0, 0, 0, 2), "O2": (0, 0, 2, 0), "CO2": (1, 0, 2, 0),
    "H2O": (0, 2, 1, 0), "AR": (0, 0, 0, 0),
    "CH4": (1, 4, 0, 0), "C2H6": (2, 6, 0, 0), "C3H8": (3, 8, 0, 0),
    "C4H10": (4, 10, 0, 0), "C8H18": (8, 18, 0, 0), "C14H30": (14, 30, 0, 0),
}


@dataclass(frozen=True)
class GasState:
    s: float      # entropy-complement function, J/(kg K)
    s_t: float    # ds/dT
    h: float      # complete enthalpy, J/kg
    h_t: float    # dh/dT  (= cp)
    cp: float     # specific heat, J/(kg K)
    r: float      # gas constant, J/(kg K)


def _bracket(t: list, t1: float) -> int:
    """Binary search returning the 1-based upper bracket index, as in gasfun.f.

    The Fortran does not clamp: below the first tabulated temperature or above
    the last, the same Hermite formula is evaluated with f outside [0,1], i.e.
    it extrapolates. That behaviour is preserved here rather than clamped,
    because clamping would silently change results near the table edges.
    """
    ilow, i = 1, len(t)
    while i > ilow + 1:
        imid = (i + ilow) // 2
        if t1 < t[imid - 1]:
            i = imid
        else:
            ilow = imid
    return i


def _hermite(fr: float, d: float, y0: float, y1: float,
             m0: float, m1: float) -> float:
    """The exact cubic form used throughout gasfun.f.

        (1-fr) y0 + fr y1
          + fr(1-fr) [ (1-fr)(d*m0 - y1 + y0) - fr(d*m1 - y1 + y0) ]

    where *d* is the interval width and m0/m1 the endpoint slopes.
    """
    return ((1.0 - fr) * y0 + fr * y1
            + fr * (1.0 - fr) * ((1.0 - fr) * (d * m0 - y1 + y0)
                                 - fr * (d * m1 - y1 + y0)))


def gas_properties(key: str, t1: float) -> GasState:
    """Properties of gas *key* (e.g. ``"N2"``) at temperature *t1* [K]."""
    g = GASES[key]
    t, tl, cp, cpt, h, s = g["t"], g["tl"], g["cp"], g["cpt"], g["h"], g["s"]

    i = _bracket(t, t1)
    lo, hi = i - 2, i - 1          # 0-based neighbours of the 1-based bracket

    dt = t[hi] - t[lo]
    fr = (t1 - t[lo]) / dt
    dtl = tl[hi] - tl[lo]
    frl = (math.log(t1) - tl[lo]) / dtl

    # entropy interpolates against log T, with cp as the slope
    s1 = _hermite(frl, dtl, s[lo], s[hi], cp[lo], cp[hi])
    # enthalpy interpolates against T, with cp as the slope
    h1 = g["hform"] + _hermite(fr, dt, h[lo], h[hi], cp[lo], cp[hi])
    # cp interpolates against T, with dcp/dT as the slope
    cp1 = _hermite(fr, dt, cp[lo], cp[hi], cpt[lo], cpt[hi])

    return GasState(s=s1, s_t=cp1 / t1, h=h1, h_t=cp1, cp=cp1, r=g["r"])


def gasfun(igas: int, t: float) -> GasState:
    """``gasfun(igas, t)`` — properties by the Fortran's numeric gas index."""
    try:
        key = IGAS[igas]
    except KeyError:
        raise ValueError(f"GASFUN: undefined gas index: {igas}") from None
    return gas_properties(key, t)


def gaschem(igas: int) -> tuple:
    """``gaschem(igas)`` — (nC, nH, nO, nN) atoms per molecule."""
    try:
        return NCHON[IGAS[igas]]
    except KeyError:
        raise ValueError(f"GASCHEM: undefined gas index: {igas}") from None


__all__ = ["gasfun", "gaschem", "gas_properties", "GasState", "IGAS", "NCHON"]
