"""The off-design engine deck -- ``eopwrt`` in ``tasopt.f``.

Once an aircraft is sized, its engine is a fixed piece of hardware, and this
runs that hardware over a grid of **altitude x Mach x throttle** and reports
what it does at each point. The result is an engine deck: thrust, fuel flow,
TSFC, spool speeds, pressures and temperatures at every station, for 45
operating points on the 737.

It is the only part of TASOPT that exercises the engine away from the design
mission, and unlike the rest of the output path it *computes* something --
every point is a converged ``tfoper`` solve. That makes it useful well beyond
its text file: :func:`engine_deck` hands back the operating points as data,
which is what you would fit a surrogate to.

How a point is run
------------------
Two solves per (altitude, Mach) pair, then one per throttle setting:

1. The state is seeded from **takeoff** and run at takeoff ``Tt4`` to find
   the maximum thrust available at this flight condition. That is what the
   throttle fractions are fractions *of* -- not of sea-level static thrust,
   so a "60%" point at 20 kft is a much smaller force than at sea level.
2. The state is re-seeded from **descent** if the lowest requested throttle
   is below 0.5, otherwise from **climb 1**, on the reasoning that the
   nearest converged point makes the best starting guess.
3. Each throttle setting is then run at specified thrust, marching from the
   previous one.

The whole thing runs in the ``iptest`` slot -- the spare 17th mission point
``index.inc`` carries for exactly this -- so the mission arrays are left
alone. Cooling flows are held at whatever the sizing established
(``icool = 1``).

The grid comes from a separate ``<case>.tase`` file, read by
:func:`tasopt_py.tasfile.read_tase`. ``tasopt.f`` opens it next to the
``.tas`` and silently skips the deck if it is absent, so most cases never
produce one. The shipped 737 does: ``runs/737/737.tase`` asks for three
altitudes, three Mach numbers and five throttle settings, and the program
writes ``737.oute``.

The point labels wrap at ten
----------------------------
Each point is written under a tag ``T0``..``T9`` formed as
``mod(keFset, 10)``, so the throttle index is what identifies a point and it
repeats every ten settings. Since ``engwrt`` prefixes every one of its ~120
lines with that tag, a grid with more than ten throttle settings produces
duplicate labels. Reproduced as written; nothing shipped goes past five.

Verified against the compiled Fortran; see ``tests/test_enginedeck.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .atmosphere import atmos
from .engine.tfcalc import tfcalc
from .model import indices as I
from .output import (FT_M, HR_S, RULE1, RULE2, RULE3, VERSION, _blank, _c64,
                     _ffmt, _say, engwrt)

__all__ = ["engine_deck", "eopwrt", "DeckPoint", "EngineDeck"]


@dataclass
class DeckPoint:
    """One operating point of the deck."""
    alt: float                  # m
    mach: float
    #: The requested fraction of the maximum thrust available here.
    fset: float
    #: 1-based position of this setting on the throttle axis. The printed
    #: label is ``T{kfset % 10}``, so it wraps -- see the module docstring.
    kfset: int
    #: Maximum thrust per engine at this altitude and Mach, N.
    Fmax: float
    #: The engine state, a ``pare`` column.
    pare: object = None

    @property
    def F(self) -> float:
        """Thrust per engine, N."""
        return self.pare[I.IEFE]

    @property
    def TSFC(self) -> float:
        """Thrust-specific fuel consumption, 1/hr -- the printed units."""
        return self.pare[I.IETSFC] / HR_S

    @property
    def mfuel(self) -> float:
        """Fuel mass flow per engine, kg/s."""
        return self.pare[I.IEMCORE] * self.pare[I.IEFF]


@dataclass
class EngineDeck:
    """A whole deck: every point, and the grid it was run on."""
    grid: object = None
    points: list = field(default_factory=list)

    def at(self, alt: float, mach: float) -> list:
        """The throttle sweep at one flight condition."""
        return [p for p in self.points
                if p.alt == alt and p.mach == mach]


def engine_deck(pari, parg, para, pare, grid, *, icool: int = 1) -> EngineDeck:
    """Run the sized engine over a grid of operating points.

    ``para``/``pare`` are the design mission's matrices; they are written in
    the ``iptest`` slot only, so the mission itself is untouched.
    """
    ip = I.IPTEST
    deck = EngineDeck(grid=grid)

    def seed(ipinit):
        for ia in range(1, I.IATOTAL + 1):
            para[ia, ip] = para[ia, ipinit]
        for ie in range(1, I.IETOTAL + 1):
            pare[ie, ip] = pare[ie, ipinit]

    def condition(alt, M0, T0, p0, rho0, a0, mu0):
        para[I.IAALT, ip] = alt
        para[I.IAMACH, ip] = M0
        pare[I.IEP0, ip] = p0
        pare[I.IET0, ip] = T0
        pare[I.IEA0, ip] = a0
        pare[I.IERHO0, ip] = rho0
        pare[I.IEMU0, ip] = mu0
        pare[I.IEM0, ip] = M0
        pare[I.IEU0, ip] = M0 * a0

    for alt in grid.alt:
        at = atmos(alt / 1000.0)
        for M0 in grid.mach:
            # Max thrust here: seed from takeoff and run at takeoff Tt4.
            seed(I.IPTAKEOFF)
            condition(alt, M0, at.T, at.p, at.rho, at.a, at.mu)
            pare[I.IETT4, ip] = pare[I.IETT4, I.IPTAKEOFF]
            _run(pari, parg, para, pare, ip, 1, icool)
            Fmax = pare[I.IEFE, ip]

            # Then re-seed from whichever mission point is closest in
            # throttle to the settings being asked for.
            seed(I.IPDESCENTN if grid.fset[0] < 0.5 else I.IPCLIMB1)
            condition(alt, M0, at.T, at.p, at.rho, at.a, at.mu)

            for kfset, frac in enumerate(grid.fset, start=1):
                pare[I.IEFE, ip] = Fmax * frac
                _run(pari, parg, para, pare, ip, 2, icool)
                deck.points.append(DeckPoint(alt=alt, mach=M0, fset=frac,
                                             kfset=kfset, Fmax=Fmax,
                                             pare=pare.column(ip)))
    return deck


def _run(pari, parg, para, pare, ip, icall, icool):
    """``tfcalc`` on a column of the matrices, written back in place."""
    acol, ecol = para.column(ip), pare.column(ip)
    tfcalc(pari, parg, acol, ecol, ip, icall, icool, 1)
    para.set_column(ip, acol)
    pare.set_column(ip, ecol)


def eopwrt(case, deck) -> str:
    """The ``.oute`` file for a completed deck, as the Fortran writes it."""
    out = [_blank(), RULE1, f" TASOPT v{VERSION:5.2f}",
           _blank(), RULE2,
           " Config:  " + _c64(case.configname), _blank(),
           " Case:  " + _c64(case.casename[0]),
           "     :  " + _c64(case.casename[1]),
           _blank(), RULE2, _say("Engine off-design operation matrix")]

    alt = mach = None
    for p in deck.points:
        if p.alt != alt:
            alt, mach = p.alt, None
            out.append(_blank())
            out.append(RULE1)
            out.append(" altitude =" + _ffmt(p.alt * FT_M, 10, 0) + " ft")
        if p.mach != mach:
            mach = p.mach
            out.append(_blank())
            out.append(RULE2)
            out.append(f" Mach ={p.mach:10.4f}")
        out.append(_blank())
        out.append(RULE3)
        out.append(f" Fset ={p.fset * 100.0:10.2f} %")
        # The tag wraps at ten -- see the module docstring.
        out += engwrt(f"T{p.kfset % 10}", p.pare)

    return "".join(line + "\n" for line in out)
