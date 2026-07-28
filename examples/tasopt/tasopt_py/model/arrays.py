"""The parameter arrays TASOPT carries every quantity in.

The whole program shares five flat arrays, positions named by ``index.inc``:

======  ==============================================================
pari    integer flags -- fuel type, engine location, sizing options
parm    per-mission quantities -- range, payload, takeoff weight
parg    geometry and sizing -- spans, areas, weights, one set per aircraft
para    aerodynamics, one column per mission point
pare    engine state, one column per mission point
======  ==============================================================

Everything is 1-based, because the ported code has to line up with the source
line by line: ``parg[IGB]`` here is ``parg(igb)`` there. Element 0 exists and
is unused, exactly as if the array were declared ``(0:n)`` -- writing to it
raises rather than silently corrupting an off-by-one.

``para`` and ``pare`` are indexed ``[index, point]``, matching Fortran's
``para(ia..., ip...)``.

Why not attributes
------------------
Named attributes would read better, but a 609-name flat namespace transcribed
by hand is 609 chances at a silent aliasing bug, and the source freely does
arithmetic on indices (``ieepsc1 + icrow - 1``). Keeping the integer indexing
makes every ported line checkable against its original.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import indices as I

__all__ = ["ParamArray", "ParamMatrix", "Aircraft"]


class ParamArray:
    """A 1-based flat array of floats."""

    __slots__ = ("_v", "_name")

    def __init__(self, n: int, name: str = "par"):
        self._v = [0.0] * (n + 1)
        self._name = name

    def __len__(self):
        return len(self._v) - 1

    def _check(self, i):
        if i < 1 or i >= len(self._v):
            raise IndexError(
                f"{self._name}[{i}] out of range 1..{len(self._v) - 1}")

    def __getitem__(self, i):
        self._check(i)
        return self._v[i]

    def __setitem__(self, i, value):
        self._check(i)
        self._v[i] = value

    def __repr__(self):
        return f"<{self._name}[1..{len(self)}]>"

    def copy(self):
        out = type(self)(len(self), self._name)
        out._v = list(self._v)
        return out


class ParamMatrix:
    """A 1-based ``[index, point]`` array of floats."""

    __slots__ = ("_v", "_n", "_np", "_name")

    def __init__(self, n: int, npoint: int, name: str = "par"):
        self._n, self._np, self._name = n, npoint, name
        self._v = [[0.0] * (npoint + 1) for _ in range(n + 1)]

    @property
    def shape(self):
        return self._n, self._np

    def _check(self, key):
        i, p = key
        if i < 1 or i > self._n:
            raise IndexError(f"{self._name}[{i},{p}] index out of "
                             f"range 1..{self._n}")
        if p < 1 or p > self._np:
            raise IndexError(f"{self._name}[{i},{p}] point out of "
                             f"range 1..{self._np}")

    def __getitem__(self, key):
        self._check(key)
        return self._v[key[0]][key[1]]

    def __setitem__(self, key, value):
        self._check(key)
        self._v[key[0]][key[1]] = value

    def column(self, p: int) -> list:
        """All indices at mission point *p*, as a 1-based ParamArray."""
        out = ParamArray(self._n, f"{self._name}(:,{p})")
        for i in range(1, self._n + 1):
            out[i] = self._v[i][p]
        return out

    def set_column(self, p: int, col: ParamArray):
        for i in range(1, self._n + 1):
            self._v[i][p] = col[i]

    def __repr__(self):
        return f"<{self._name}[1..{self._n}, 1..{self._np}]>"

    def copy(self):
        out = type(self)(self._n, self._np, self._name)
        out._v = [row[:] for row in self._v]
        return out


@dataclass
class Aircraft:
    """One aircraft: the five arrays, sized from ``index.inc``.

    ``npoint`` is how many mission points the aero and engine arrays carry.
    ``index.inc`` declares ``iptotal = 17`` named points: static, rotate,
    takeoff, cutback, five climb, **two** cruise, five descent, and a spare
    ``iptest`` slot that is not part of the mission. Two cruise points against
    five climb points is deliberate -- cruise is nearly uniform, the climb is
    not.
    """
    npoint: int = I.IPTOTAL
    pari: ParamArray = field(default=None)
    parm: ParamArray = field(default=None)
    parg: ParamArray = field(default=None)
    para: ParamMatrix = field(default=None)
    pare: ParamMatrix = field(default=None)

    def __post_init__(self):
        if self.pari is None:
            self.pari = ParamArray(I.IITOTAL, "pari")
        if self.parm is None:
            self.parm = ParamArray(I.IMTOTAL, "parm")
        if self.parg is None:
            self.parg = ParamArray(I.IGTOTAL, "parg")
        if self.para is None:
            self.para = ParamMatrix(I.IATOTAL, self.npoint, "para")
        if self.pare is None:
            self.pare = ParamMatrix(I.IETOTAL, self.npoint, "pare")

    def copy(self) -> "Aircraft":
        return Aircraft(npoint=self.npoint, pari=self.pari.copy(),
                        parm=self.parm.copy(), parg=self.parg.copy(),
                        para=self.para.copy(), pare=self.pare.copy())
