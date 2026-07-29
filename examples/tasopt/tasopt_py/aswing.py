"""The ASWING export -- ``aswout.f`` and ``BOUTPUT`` in ``aswio.f``.

ASWING is Drela's aeroelastic beam code. TASOPT can hand it a sized aircraft
as an ``.asw`` input deck: the fuselage, wing, horizontal and vertical tails
as **beams** carrying spanwise distributions of stiffness, mass, inertia and
section aerodynamics, plus the point weights, engines, joints and ground
attachments that tie them together. Behind ``Laswwrite``, which ``737.tas``
sets to ``F``.

It is the largest single thing in TASOPT that is not physics -- about 2100
code lines -- and almost all of it is *translation*: taking the sizing's
``parg`` entries and re-expressing them in the variables an ASWING beam
wants. The interesting part is that the translation is lossy in places, and
several of those places are mistakes; they are listed below and reproduced.

What is exported
----------------
================  =========================================================
``Name``          the configuration name
``Unit``          length, time and force units -- all 1.0, SI
``Constant``      g, sea-level density and speed of sound
``Reference``     Sref, Cref, Bref and the moment/acceleration/velocity
                  reference points (all at the wing box)
``Weight``        eight point masses: fixed equipment, APU, hydraulics and
                  electrics, nose gear, two main gear, two engines
``Engine``        two thrust points, each with its direction and dF/dPeng
``Joint``         fuselage-to-VT-root and fuselage-to-HT-centre
``Ground``        the wing box on the fuselage, and the two wing roots
``Beam 1..4``     fuselage, wing, horizontal tail, vertical tail
================  =========================================================

A fifth beam, the strut, is written for a strut-braced wing
(``iwplan = 2``), and the vertical tail grows a horizontal cross-member on a
Pi-tail (``nvtail > 1``). The 737 has neither, so both are checked against a
second case: ``runs/D8/sd81.tas`` has both, and its 366-line deck is
reproduced byte for byte too. Between the two, every branch in ``aswout`` is
exercised.

``Strut``, ``Sensor`` and ``Jangle`` *blocks* exist in ``BOUTPUT`` but
nothing in ``aswout`` ever creates one -- the strut is a beam, not a strut
pylon -- so writing one raises rather than emitting an untested format.

How the file gets its shape
---------------------------
Two things make the format less mechanical than it looks.

**Column blocks.** A beam's variables are not written one per column across
one wide table. ``BOUTPUT`` groups them: it walks the defined variables in
index order, and for each one collects every *later* variable that has the
same number of stations **and the same station positions** (to within
1e-5 of the beam's own length), up to six at a time, and writes that group
as its own table with its own ``t`` column. So the fuselage comes out as
three tables -- 31 geometry stations, 8 structural stations, 21 aerodynamic
stations -- because those are three different griddings of the same beam.

**Per-column scale factors.** Each column is divided by a power of ten
chosen as ``10**int(log10(2*max|q|))``, written on a ``*`` line above the
table. The factor of two means a column whose largest value is 6.0 is
scaled by 10 rather than 1, which is why the deck is full of columns that
look an order of magnitude too small.

Things in the source worth knowing
----------------------------------
All of these are reproduced, with a test pinning each.

* **The fuselage's friction-drag column is evaluated at the wrong station.**
  ``aswout`` interpolates the BL dissipation coefficient onto each output
  station inside an ``if`` that tests the interval -- but the edge velocity
  ``ue`` is recomputed *outside* that test, once per interval, so after the
  loop it holds the value at the **last** BL interval rather than at the
  station. Worse, ``fi`` is not initialised, so the first station uses
  whatever the previous iteration left. ``Cdf = 2 Cdiss ue^3`` is therefore
  wrong everywhere by the cube of a velocity ratio. §46.
* **The wing's outer panel is given inboard section properties.** In the
  taper loop ``cm = cms*(1-frac) + cms*frac``, and the same for ``cdf`` and
  ``cdp`` -- so all three are pinned at their break values and never reach
  the tip. ``cmt``, ``cdft`` and ``cdpt`` are all computed just above and
  never used. This one is **latent**: ``cdf`` and ``cdp`` are hard-wired to
  the same two constants at all three stations anyway, and the 737's
  section moment is uniform across the span, so no shipped deck differs by
  a digit. It would start to matter the moment a case gave its tip its own
  airfoil. §47.
* **Three shell variables are self-referential.** In the same loop
  ``Csh = Csho*(1-frac) + Csh*frac``, and likewise ``Nsh`` and ``Atsh``:
  the right-hand side is the loop variable itself, carrying whatever the
  previous station left, not the break value ``Cshs``. §48.
* **The wing's fuel inertia is interpolated between an inertia and a
  mass.** In the mid panel
  ``mgnnf = mgnnofuel*(1-frac) + mgsfuel*frac`` -- the inboard end is the
  fuel inertia, the outboard end the fuel *mass*, where ``mgnnsfuel`` was
  meant. On the 737 that is a factor of 3.5 at the break, and it shows as
  a discontinuity in the deck: the column steps from 0.60510 to 0.17277
  across the doubled station. §49.
* **The vertical tail's ``Csh`` is assigned from itself** in the
  cross-connect loop -- ``Csh = Csh`` -- so the horizontal member between
  two fins is given the **horizontal tail's tip** shell width, left in the
  variable by the previous beam's loop. It is not garbage, and it is not
  ``Cshv``; it is a value from a different surface. Only fires on a Pi-tail
  (``nvtail > 1``), which ``runs/D8`` and ``runs/HE`` have. §50.
* The ``Unit`` block writes L, T and F but not M, although ``BOUTPUT`` uses
  ``UNITM`` to scale the density on the ``Constant`` line. Harmless here
  because everything is 1.0.
* ``EAfac`` is set to 0.001 and then immediately to 1.0 -- a debugging
  knob left switched off.

Verified against the compiled Fortran; see ``tests/test_aswing.py``.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .atmosphere import atmos
from .model import beam_indices as B
from .model import indices as I
from .output import _efmt, _gfmt

__all__ = ["aswout", "boutput", "Deck", "Beam", "Pylon", "Joint", "Ground"]

GEE = 9.81
#: ``aswout``'s local array bounds. Kept because they are what the source
#: checks against, and a case that overflows one should say so.
IBX, NBX, NJX, NGX, NPX, KPX = 100, 35, 30, 8, 50, 21

#: Pylon kinds, as ``KPTYPE`` encodes them: 1 point weight, 2 strut,
#: 10+k engine k, 100+k sensor k.
KP_WEIGHT, KP_STRUT = 1, 2
KP_ENGINE, KP_SENSOR = 10, 100


# --------------------------------------------------------------------------
# The deck's data model
# --------------------------------------------------------------------------

class Beam:
    """One ASWING beam: spanwise distributions of the variables it defines.

    ``kbnum`` is the beam's own number; ``ibeam`` is the number of the beam
    it is *aerodynamically* part of, which differs only for surfaces that
    share a lifting system -- the vertical tail carries the horizontal
    tail's, and the strut the wing's.
    """

    def __init__(self, name: str, kbnum: int, ibeam: int = None):
        self.name = name
        self.kbnum = kbnum
        self.ibeam = kbnum if ibeam is None else ibeam
        self._q = defaultdict(dict)          # j -> {station: value}
        self._t = defaultdict(dict)          # j -> {station: t}
        self.nb = {}                         # j -> station count
        self.defined = set()                 # j's that get written

    # -- filling ----------------------------------------------------------
    def put(self, ib: int, j: int, value: float, t: float = None):
        self._q[j][ib] = value
        if t is not None:
            self._t[j][ib] = t

    def put_t(self, ib: int, t: float):
        """``DO J = 1, JBTOT: TB(IB,J,IS) = t`` -- the lifting surfaces set
        one station position for every variable at once."""
        for j in range(1, B.JBTOT + 1):
            self._t[j][ib] = t

    def count(self, n: int, *js):
        for j in js:
            self.nb[j] = n

    def count_all(self, n: int):
        for j in range(1, B.JBTOT + 1):
            self.nb[j] = n

    def define(self, *js):
        self.defined.update(js)

    # -- reading ----------------------------------------------------------
    def q(self, ib: int, j: int) -> float:
        return self._q[j].get(ib, 0.0)

    def t(self, ib: int, j: int) -> float:
        return self._t[j].get(ib, 0.0)


@dataclass
class Pylon:
    """A point attachment: weight, strut, engine or sensor."""
    kptype: int
    kbeam: int
    #: ``QPYLO(0..KPX)``: index 0 is the beam station ``t``, the rest depend
    #: on the kind.
    q: list = field(default_factory=lambda: [0.0] * (KPX + 1))


@dataclass
class Joint:
    beams: tuple                 # (beam 1, beam 2)
    t: tuple                     # station on each
    kjtype: int = 0


@dataclass
class Ground:
    beam: int
    t: float
    kgtype: int = 0


@dataclass
class Deck:
    """A complete ASWING input deck, before it is written out."""
    name: str = ""
    unitl: float = 1.0
    unitm: float = 1.0
    unitt: float = 1.0
    unitf: float = 1.0
    unchl: str = "m"
    unchm: str = "kg"
    uncht: str = "s"
    unchf: str = "N"
    gee: float = GEE
    rhoSL: float = 0.0
    aSL: float = 0.0
    Sref: float = 0.0
    Cref: float = 0.0
    Bref: float = 0.0
    #: ``XYZREF(3,3)`` -- the moment, acceleration and velocity reference
    #: points, one column each. ``aswout`` puts all three at the wing box.
    xyzref: list = field(default_factory=lambda: [[0.0] * 3 for _ in range(3)])
    beams: list = field(default_factory=list)
    pylons: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    grounds: list = field(default_factory=list)
    #: Engine type per engine, ``IENGTYP``. 0 is a plain thrust point.
    engtyp: list = field(default_factory=list)


# --------------------------------------------------------------------------
# aswout.f -- build the deck
# --------------------------------------------------------------------------

def aswout(pari, parg, para, configname: str, bl=None) -> Deck:
    """Build an ASWING deck from a sized aircraft.

    ``para`` is a single mission-point column -- ``tasopt.f`` passes cruise 1
    -- and ``bl`` is the fuselage boundary-layer solution, which the
    fuselage beam's drag columns are built from. Without it those two
    columns are left at zero rather than guessed.
    """
    pi = math.pi
    Lttail = False
    Lptail = parg[I.IGNVTAIL] > 1.0001

    xnose, xend = parg[I.IGXNOSE], parg[I.IGXEND]
    xwbox = parg[I.IGXWBOX]
    xhbox, xvbox = parg[I.IGXHBOX], parg[I.IGXVBOX]
    xblend1, xblend2 = parg[I.IGXBLEND1], parg[I.IGXBLEND2]
    xshell1, xshell2 = parg[I.IGXSHELL1], parg[I.IGXSHELL2]

    ifwcen = pari[I.IIFWCEN]
    iwplan = pari[I.IIWPLAN]
    iengloc = pari[I.IIENGLOC]

    # An upturned nose is assumed when the fuselage moment is nose-up.
    xupsweep = xnose + 0.75 * (xwbox - xnose)
    if parg[I.IGCLMF0] < 0.0:
        dupsweep = 0.7 * (parg[I.IGRFUSE] + 0.5 * parg[I.IGDRFUSE])
    else:
        dupsweep = -0.2 * parg[I.IGRFUSE]

    if Lptail:
        yov = parg[I.IGWFB] + parg[I.IGRFUSE]
        ytv = 0.5 * parg[I.IGBOH]
        zov = 0.5 * parg[I.IGBOV]
        ztv = parg[I.IGBV]
        t0v = 0.0
        ttv = yov + ztv
    else:
        yov = ytv = 0.0
        zov = 0.5 * parg[I.IGBOV]
        ztv = parg[I.IGBV]
        t0v = 1.0
        ttv = ztv

    # `EAfac = 0.001` immediately overwritten by `EAfac = 1.0` -- a debug
    # knob left off.
    EAfac = 1.0

    at = atmos(0.0)
    deck = Deck(name=configname, rhoSL=at.rho, aSL=at.a,
                Sref=parg[I.IGS], Bref=parg[I.IGB], Cref=parg[I.IGCMA])
    for L in range(3):
        deck.xyzref[0][L] = xwbox
        deck.xyzref[1][L] = 0.0
        deck.xyzref[2][L] = 0.0

    # ---- ground attachments ---------------------------------------------
    # The second fuselage ground is commented out in the source.
    deck.grounds.append(Ground(beam=1, t=xwbox))
    deck.grounds.append(Ground(beam=2, t=0.5 * parg[I.IGBO]))
    deck.grounds.append(Ground(beam=2, t=-0.5 * parg[I.IGBO]))

    # ---- joints ----------------------------------------------------------
    # The two fuselage-to-wing joints above these are commented out; the
    # wing is held by its ground attachments instead.
    if Lttail:
        deck.joints.append(Joint((1, 4), (xvbox, t0v)))
        deck.joints.append(Joint((4, 3), (ttv, 0.0)))
    elif Lptail:
        deck.joints.append(Joint((1, 4), (xvbox, t0v)))
        deck.joints.append(Joint((4, 3), (ttv, 0.5 * parg[I.IGBOH])))
        deck.joints.append(Joint((4, 3), (-ttv, -0.5 * parg[I.IGBOH])))
    else:
        deck.joints.append(Joint((1, 4), (xvbox, t0v)))
        deck.joints.append(Joint((1, 3), (xhbox, 0.0)))

    # ---- point weights ---------------------------------------------------
    def weight(kbeam, t, x, y, z, Mg):
        p = Pylon(kptype=KP_WEIGHT, kbeam=kbeam)
        p.q[0], p.q[1], p.q[2], p.q[3], p.q[4] = t, x, y, z, Mg
        deck.pylons.append(p)

    Rfuse = parg[I.IGRFUSE]
    weight(1, parg[I.IGXFIX], parg[I.IGXFIX], 0.0, dupsweep, parg[I.IGWFIX])
    weight(1, parg[I.IGXAPU], parg[I.IGXAPU], 0.0, 0.0,
           parg[I.IGWPAY] * parg[I.IGFAPU])
    weight(1, parg[I.IGXHPESYS], parg[I.IGXHPESYS], 0.0, -0.8 * Rfuse,
           parg[I.IGWMTO] * parg[I.IGFHPESYS])
    weight(1, parg[I.IGXLGNOSE], parg[I.IGXLGNOSE], 0.0, -0.8 * Rfuse,
           parg[I.IGWMTO] * parg[I.IGFLGNOSE])

    xlgmain = parg[I.IGXCGAFT] + parg[I.IGDXLGMAIN]
    ylgmain = 1.2 * (parg[I.IGWFB] + Rfuse)
    zlgmain = -0.8 * (Rfuse + 0.5 * parg[I.IGDRFUSE])
    Wlg = 0.5 * parg[I.IGWMTO] * parg[I.IGFLGMAIN]
    weight(1, xwbox, xlgmain, ylgmain, zlgmain, Wlg)
    weight(1, xwbox, xlgmain, -ylgmain, zlgmain, Wlg)

    neng = int(parg[I.IGNENG] + 0.001)

    def engine_station(esgn):
        """Where an engine hangs, and on which beam."""
        if iengloc == 1:
            return (2, 0.5 * parg[I.IGBS] * esgn,
                    parg[I.IGZWING] - 0.3 * parg[I.IGCO] * parg[I.IGLAMBDAS])
        return 1, xvbox, 0.5 * Rfuse

    # ---- engines ---------------------------------------------------------
    for ieng in range(1, neng + 1):
        frac = 0.0 if neng == 1 else float(ieng - 1) / float(neng - 1)
        esgn = 1.0 - 2.0 * frac
        p = Pylon(kptype=KP_ENGINE + ieng, kbeam=0)
        deck.engtyp.append(0)
        p.q[1] = parg[I.IGXENG]
        p.q[2] = parg[I.IGYENG] * esgn
        p.q[4], p.q[5], p.q[6] = -1.0, 0.0, 0.0     # thrust direction
        p.q[7] = 1.0 / float(neng)                  # dF/dPeng
        p.q[8] = 0.0                                # dM/dPeng
        p.kbeam, p.q[0], p.q[3] = engine_station(esgn)
        deck.pylons.append(p)

    # ---- engine weights --------------------------------------------------
    for ieng in range(1, neng + 1):
        frac = 0.0 if neng == 1 else float(ieng - 1) / float(neng - 1)
        esgn = 1.0 - 2.0 * frac
        p = Pylon(kptype=KP_WEIGHT, kbeam=0)
        p.q[1] = parg[I.IGXENG]
        p.q[2] = parg[I.IGYENG] * esgn
        p.q[4] = parg[I.IGWENG] / float(neng)
        p.kbeam, p.q[0], p.q[3] = engine_station(esgn)
        deck.pylons.append(p)

    deck.beams.append(_fuselage_beam(pari, parg, bl, xnose, xend, xwbox,
                                     xblend1, xblend2, xshell1, xshell2,
                                     xupsweep, dupsweep))
    deck.beams.append(_wing_beam(parg, para, iwplan, ifwcen, xwbox, EAfac))
    htail, csh_left_behind = _htail_beam(parg, para, Lptail, EAfac)
    deck.beams.append(htail)
    deck.beams.append(_vtail_beam(parg, para, Lptail, yov, ytv, zov, ztv,
                                  t0v, csh_left_behind))
    if iwplan == 2:
        deck.beams.append(_strut_beam(parg, para, xwbox))
        yo = 0.5 * parg[I.IGBO]
        ys = 0.5 * parg[I.IGBS]
        deck.grounds.append(Ground(beam=5, t=yo))
        deck.grounds.append(Ground(beam=5, t=-yo))
        deck.joints.append(Joint((5, 2), (ys, ys)))
        deck.joints.append(Joint((5, 2), (-ys, -ys)))

    return deck


def _fuselage_beam(pari, parg, bl, xnose, xend, xwbox, xblend1, xblend2,
                   xshell1, xshell2, xupsweep, dupsweep) -> Beam:
    """Beam 1: three different griddings of the same body."""
    pi = math.pi
    beam = Beam("Fuselage", 1)

    wfb, Rfuse, dRfuse = parg[I.IGWFB], parg[I.IGRFUSE], parg[I.IGDRFUSE]
    tskin = parg[I.IGTSKIN]

    wfblim = max(min(wfb, Rfuse), 0.0)
    thetafb = math.asin(wfblim / Rfuse)
    hfb = math.sqrt(Rfuse ** 2 - wfb ** 2)
    sint, cost = wfb / Rfuse, hfb / Rfuse
    sin2t = 2.0 * sint * cost
    Afuse = ((pi + 2.0 * thetafb + sin2t) * Rfuse ** 2
             + 2.0 * Rfuse * dRfuse)
    anose, btail = parg[I.IGANOSE], parg[I.IGBTAIL]
    #: An area-equivalent circular radius, so the beam is round even when
    #: the real cross-section is a double bubble.
    Rcyl = math.sqrt(Afuse / pi)

    Wcabin = (parg[I.IGWPAY]
              + parg[I.IGWPAY] * parg[I.IGFPADD]
              + parg[I.IGWPAY] * parg[I.IGFSEAT]
              + parg[I.IGWWINDOW] + parg[I.IGWINSUL] + parg[I.IGWFLOOR])
    Wshell, Whbend = parg[I.IGWSHELL], parg[I.IGWHBEND]
    Wvbend, Wcone = parg[I.IGWVBEND], parg[I.IGWCONE]
    EIhshell, EIhbend = parg[I.IGEIHSHELL], parg[I.IGEIHBEND]
    EIvshell, EIvbend = parg[I.IGEIVSHELL], parg[I.IGEIVBEND]
    GJshell, GJcone = parg[I.IGGJSHELL], parg[I.IGGJCONE]

    # -- geometry: 31 cosine-spaced stations -------------------------------
    ni = 31
    for i in range(1, ni + 1):
        fraci = float(i - 1) / float(ni - 1)
        frac = 0.5 * (1.0 - math.cos(pi * fraci))     # ispace = 1
        x = xnose * (1.0 - frac) + xend * frac
        y = 0.0

        if x < xupsweep:
            f = 1.0 - (x - xnose) / (xupsweep - xnose)
            z = dupsweep * f ** 2
        else:
            z = 0.0

        if i == 1 or i == ni:
            rad = 0.0
        elif x < xblend1:
            f = 1.0 - (x - xnose) / (xblend1 - xnose)
            rad = Rcyl * (1.0 - f ** anose) ** (1.0 / anose)
        elif x < xblend2:
            rad = Rcyl
        else:
            f = (x - xblend2) / (xend - xblend2)
            rad = Rcyl * (1.0 - f ** btail)

        t = x
        beam.put(i, B.JXA, x, t)
        beam.put(i, B.JYA, y, t)
        beam.put(i, B.JZA, z, t)
        beam.put(i, B.JRAD, rad, t)
        beam.put(i, B.JCSH, rad + tskin, t)
        beam.put(i, B.JNSH, rad + tskin, t)
        beam.put(i, B.JASH, Afuse * tskin, t)

    geom = (B.JXA, B.JYA, B.JZA, B.JRAD, B.JCSH, B.JNSH, B.JASH)
    beam.count(ni, *geom)
    beam.define(*geom)

    # -- structure: 8 stations, doubled at each discontinuity --------------
    ni = 8
    for i in range(1, ni + 1):
        mg1 = mgcc1 = mgnn1 = mg2 = mgcc2 = mgnn2 = 0.0
        EIcc = EInn = GJ = 0.0

        if i == 1:
            x = xnose
            EIcc, EInn, GJ = 0.1 * EIhshell, 0.1 * EIvshell, 0.1 * GJshell
        elif i in (2, 3):
            x = xshell1
            mg1 = Wshell / (xshell2 - xnose)
            EIcc, EInn, GJ = EIhshell, EIvshell, GJshell
        elif i in (4, 5):
            x = xwbox
            mg1 = (Wshell / (xshell2 - xnose)
                   + Whbend * 2.0 / (xshell2 - xshell1))
            EIcc, EInn, GJ = EIhshell + EIhbend, EIvshell, GJshell
            if i == 5:
                mg1 += Wvbend * 2.0 / (xshell2 - xwbox)
                EInn = EIvshell + EIvbend
        elif i == 6:
            x = xshell2
            mg1 = Wshell / (xshell2 - xnose)
            EIcc, EInn, GJ = EIhshell, EIvshell, GJshell
        elif i == 7:
            x = xshell2
            mg1 = Wcone / (xend - xshell2)
            EIcc, EInn, GJ = EIhshell, EIvshell, GJcone
        else:
            x = xend
            EIcc, EInn, GJ = 0.2 * EIhshell, 0.2 * EIvshell, 0.5 * GJcone

        if mg1 != 0.0:
            mgcc1 = mg1 * 0.5 * (Rfuse + dRfuse) ** 2
            mgnn1 = mg1 * 0.5 * (Rfuse + wfb) ** 2
        # The cabin payload rides between the two pressure bulkheads only.
        if i in (3, 4, 5, 6):
            mg2 = Wcabin / (xshell2 - xshell1)
            mgcc2 = mg2 * 0.25 * (Rfuse + dRfuse) ** 2
            mgnn2 = mg2 * 0.50 * (Rfuse + wfb) ** 2

        t = x
        beam.put(i, B.JMG1, mg1, t)
        beam.put(i, B.JMCC1, mgcc1, t)
        beam.put(i, B.JMNN1, mgnn1, t)
        beam.put(i, B.JMG2, mg2, t)
        beam.put(i, B.JMNN2, mgnn2, t)
        beam.put(i, B.JMCC2, mgcc2, t)
        beam.put(i, B.JECC, EIcc, t)
        beam.put(i, B.JENN, EInn, t)
        beam.put(i, B.JGJ, GJ, t)

    beam.count(ni, B.JMG1, B.JMCC1, B.JMNN1, B.JMG2, B.JMCC2, B.JMNN2,
               B.JECC, B.JENN, B.JGJ)
    # JMCC1 and JMNN1 are computed but their LQBDEF lines are commented out,
    # so the shell's own rotational inertia never reaches the deck.
    beam.define(B.JMG1, B.JMG2, B.JMCC2, B.JMNN2, B.JECC, B.JENN, B.JGJ)

    # -- aerodynamics: 21 evenly spaced stations ---------------------------
    ni = 21
    drag = _fuselage_drag(bl, xnose, xend, ni)
    for i in range(1, ni + 1):
        frac = float(i - 1) / float(ni - 1)
        t = xnose * (1.0 - frac) + xend * frac
        beam.put(i, B.JCDF, drag[i - 1], t)
        beam.put(i, B.JCDP, 0.05, t)

    beam.count(ni, B.JCDF, B.JCDP)
    beam.define(B.JCDF, B.JCDP)
    return beam


def _fuselage_drag(bl, xnose, xend, ni) -> list:
    """``2 Cdiss ue^3`` at each output station -- §46, reproduced.

    Two things about this loop are wrong and both are kept, because a deck
    that differs from the reference program's is not a port of it.

    The dissipation coefficient is interpolated **inside** a test for which
    BL interval contains the station, which is right. The interpolation
    weight ``fi`` and the edge velocity ``ue`` are updated **outside** it,
    once per interval, so after the loop ``ue`` holds the value at the last
    BL interval before the trailing edge no matter where the station is.
    ``Cdf`` is therefore off by the cube of a velocity ratio everywhere.

    And ``fi`` is a subroutine local, not a loop variable, so it survives
    from one output station to the next -- and on the *first* station, if no
    interval matches, it is read before it has ever been written. That does
    not happen on the shipped 737, whose first station sits exactly on the
    first BL station; :func:`aswout` raises rather than inventing a value if
    it ever does.
    """
    if bl is None:
        return [0.0] * ni

    out = []
    fi = None                       # persists across stations, as it does
    for i in range(1, ni + 1):
        frac = float(i - 1) / float(ni - 1)
        x = xnose * (1.0 - frac) + xend * frac
        Cdiss, ue = 0.0, 0.0
        for ibl in range(1, bl.iblte):      # 1..iblte-1, 1-based
            xa, xb = bl.x[ibl - 1], bl.x[ibl]
            if xa <= x <= xb:
                fi = (x - xa) / (xb - xa)
                Cdissp = 2.0 * bl.cd[ibl] * bl.th[ibl] / bl.ts[ibl]
                if bl.ts[ibl - 1] == 0.0:
                    Cdisso = Cdissp
                else:
                    Cdisso = (2.0 * bl.cd[ibl - 1] * bl.th[ibl - 1]
                              / bl.ts[ibl - 1])
                Cdiss = Cdisso * (1.0 - fi) + Cdissp * fi
            if fi is None:
                raise ValueError(
                    "aswout: the fuselage drag loop reads its interpolation "
                    f"weight before any interval has matched, at x = {x:.4f}"
                    ". The Fortran reads an uninitialised local here; see "
                    "DISCREPANCIES.md 46.")
            ue = bl.ue[ibl - 1] * (1.0 - fi) + bl.ue[ibl] * fi
        out.append(2.0 * Cdiss * ue ** 3)
    return out


def _wing_beam(parg, para, iwplan, ifwcen, xwbox, EAfac) -> Beam:
    """Beam 2: centre section, inner panel to the break, outer panel."""
    pi = math.pi
    beam = Beam("Wing", 2)

    cosL = math.cos(parg[I.IGSWEEP] * pi / 180.0)
    tanL = math.tan(parg[I.IGSWEEP] * pi / 180.0)
    wdihed = {0: 2.0, 1: 2.5}.get(iwplan, 0.0)
    tanD = math.tan(wdihed * pi / 180.0)

    co = parg[I.IGCO]
    cs = co * parg[I.IGLAMBDAS]
    ct = co * parg[I.IGLAMBDAT]
    bo, bs, b = parg[I.IGBO], parg[I.IGBS], parg[I.IGB]
    yo, ys, yt = 0.5 * bo, 0.5 * bs, 0.5 * b

    # Washout is written as a linear function of sweep, 1 degree at the
    # mid-span station and -1.2 deg per unit tan(sweep) away from it.
    etao, etas, etat = bo / b, bs / b, 1.0
    dtwdL, etac = -1.2, 0.5
    twisto = (1.0 + dtwdL * tanL * (etao - etac)) * pi / 180.0
    twists = (1.0 + dtwdL * tanL * (etas - etac)) * pi / 180.0
    twistt = (1.0 + dtwdL * tanL * (etat - etac)) * pi / 180.0

    alphao = alphas = alphat = 1.0 * pi / 180.0
    dcldao = dcldas = dcldat = 2.0 * pi * 1.05

    cmo, cms = para[I.IACMPO], para[I.IACMPS]
    cdfo = cdfs = cdft = 0.006
    cdpo = cdps = cdpt = 0.003

    rh, wbox = parg[I.IGRH], parg[I.IGWBOX]
    hboxo, tbcapo, tbwebo = (parg[I.IGHBOXO], parg[I.IGTBCAPO],
                             parg[I.IGTBWEBO])
    havgo = hboxo * (1.0 - (1.0 - rh) / 3.0)
    Abfuelo = (wbox - 2.0 * tbwebo) * (havgo - 2.0 * tbcapo)

    hboxs, tbcaps, tbwebs = (parg[I.IGHBOXS], parg[I.IGTBCAPS],
                             parg[I.IGTBWEBS])
    havgs = hboxs * (1.0 - (1.0 - rh) / 3.0)
    Abfuels = (wbox - 2.0 * tbwebs) * (havgs - 2.0 * tbcaps)

    rhocap, rhoweb = parg[I.IGRHOCAP], parg[I.IGRHOWEB]
    rhofuel = parg[I.IGRHOFUEL]
    rfmax = parg[I.IGWFUEL] / parg[I.IGWFMAX]

    Xax = parg[I.IGXAXIS]
    Xbox0, Xbox1 = -0.5 * wbox, 0.5 * wbox
    XLE, XTE = -Xax, 1.0 - Xax
    zwing = parg[I.IGZWING]

    def panel(tbcap, tbweb, hbox, c):
        """Box mass per unit span, and the secondary-structure fractions
        that hang off it, at one spanwise station."""
        box = (2.0 * rhocap * GEE * tbcap * wbox * (c * cosL) ** 2
               + 2.0 * rhoweb * GEE * tbweb * hbox * rh * (c * cosL) ** 2)
        parts = {
            "box": box,
            "flap": box * parg[I.IGFFLAP],
            "slat": box * parg[I.IGFSLAT],
            "aile": box * parg[I.IGFAILE],
            "lete": box * parg[I.IGFLETE],
            "ribs": box * parg[I.IGFRIBS],
            "spoi": box * parg[I.IGFSPOI],
        }
        # Chordwise position of each piece, as a fraction of chord about
        # the structural axis.
        pos = {"box": 0.0,
               "flap": Xbox1 + 0.20 * (XTE - Xbox1),
               "slat": Xbox0 + 0.50 * (XLE - Xbox0),
               "aile": Xbox1 + 0.40 * (XTE - Xbox1),
               "lete": Xbox0,
               "ribs": 0.0,
               "spoi": Xbox1 + 0.05 * (XTE - Xbox1)}
        # The box and the ribs are spread over the box width rather than
        # concentrated, so their second moment is w^2/12 not x^2.
        pos2 = dict(pos)
        pos2["box"] = pos2["ribs"] = None

        mg = sum(parts.values())
        dXmg = sum(parts[k] * pos[k] for k in parts)
        dX2mg = sum(parts[k] * (wbox ** 2 / 12.0 if pos2[k] is None
                                else pos2[k] ** 2) for k in parts)
        Ccg = (dXmg / mg) * c * cosL
        mgnn = dX2mg * (c * cosL) ** 2 - mg * Ccg ** 2
        return mg, Ccg, mgnn

    mgo, Ccgo, mgnno = panel(tbcapo, tbwebo, hboxo, co)
    mgs, Ccgs, mgnns = panel(tbcaps, tbwebs, hboxs, cs)

    mgofuel = Abfuelo * rhofuel * GEE * rfmax * (co * cosL) ** 2
    mgsfuel = Abfuels * rhofuel * GEE * rfmax * (cs * cosL) ** 2
    mgnnofuel = mgofuel * (wbox * co * cosL) ** 2 / 12.0
    mgnnsfuel = mgsfuel * (wbox * cs * cosL) ** 2 / 12.0

    Csho, Cshs = 0.5 * wbox * co * cosL, 0.5 * wbox * cs * cosL
    Nsho, Nshs = 0.5 * hboxo * co * cosL, 0.5 * hboxs * cs * cosL
    Atsho = ((wbox - tbwebo) * (havgo - tbcapo) * tbcapo * (co * cosL) ** 3)
    Atshs = ((wbox - tbwebs) * (havgs - tbcaps) * tbcaps * (cs * cosL) ** 3)

    EIcco, EInno, GJo = parg[I.IGEICO], parg[I.IGEINO], parg[I.IGGJO]
    EAo = parg[I.IGECAP] * 2.0 * tbcapo * wbox * (co * cosL) ** 2
    EIccs, EInns, GJs = parg[I.IGEICS], parg[I.IGEINS], parg[I.IGGJS]
    EAs = parg[I.IGECAP] * 2.0 * tbcaps * wbox * (cs * cosL) ** 2

    def station(ib, t, **kw):
        beam.put(ib, B.JXAX, Xax)
        for j, v in kw.items():
            beam.put(ib, getattr(B, j), v)
        beam.put_t(ib, t)

    # -- centre section, 2 stations ----------------------------------------
    for i in (1, 2):
        frac = float(i - 1)
        y = yo * frac
        mgf, mgnnf = mgofuel, mgnnofuel
        if ifwcen == 0:
            mgf = mgnnf = 0.0
        station(i, y, JXA=xwbox, JYA=y, JZA=zwing, JCH=co,
                JTW=twisto, JAL=alphao, JDCL=dcldao, JCM=cmo,
                JCDF=cdfo, JCDP=cdpo, JMG1=mgo, JMNN1=mgnno, JCCG1=Ccgo,
                JMG2=mgf, JMNN2=mgnnf, JCCG2=0.0,
                JECC=EIcco, JENN=EInno, JGJ=GJo, JECS=0.0, JESN=0.0,
                JEA=EAo * EAfac, JCSH=Csho, JNSH=Nsho, JASH=Atsho)

    # -- inner panel, root to break, 4 stations ----------------------------
    # Csh/Nsh/Atsh carry over from the centre section here rather than
    # interpolating to their break values -- see §48.
    Csh, Nsh, Atsh = Csho, Nsho, Atsho
    for k, i in enumerate(range(3, 7)):
        frac = float(k) / 3.0
        y = yo * (1.0 - frac) + ys * frac
        c = co * (1.0 - frac) + cs * frac
        Csh = Csho * (1.0 - frac) + Csh * frac
        Nsh = Nsho * (1.0 - frac) + Nsh * frac
        Atsh = Atsho * (1.0 - frac) + Atsh * frac
        station(i, y,
                JXA=xwbox + (y - yo) * tanL, JYA=y,
                JZA=zwing + (y - yo) * tanD, JCH=c * cosL,
                JTW=twisto * (1.0 - frac) + twists * frac,
                JAL=alphao * (1.0 - frac) + alphas * frac,
                JDCL=dcldao * (1.0 - frac) + dcldas * frac,
                JCM=cmo * (1.0 - frac) + cms * frac,
                JCDF=cdfo * (1.0 - frac) + cdfs * frac,
                JCDP=cdpo * (1.0 - frac) + cdps * frac,
                JMG1=mgo * (1.0 - frac) + mgs * frac,
                JMNN1=mgnno * (1.0 - frac) + mgnns * frac,
                JCCG1=Ccgo * (1.0 - frac) + Ccgs * frac,
                JMG2=mgofuel * (1.0 - frac) + mgsfuel * frac,
                # §49: the inboard end is the fuel inertia, the outboard
                # end is the fuel *mass* -- `mgnnsfuel` was meant.
                JMNN2=mgnnofuel * (1.0 - frac) + mgsfuel * frac,
                JCCG2=0.0,
                JECC=EIcco * (1.0 - frac) + EIccs * frac,
                JENN=EInno * (1.0 - frac) + EInns * frac,
                JGJ=GJo * (1.0 - frac) + GJs * frac,
                JEA=EAo * (1.0 - frac) + EAs * frac,
                JCSH=Csh, JNSH=Nsh, JASH=Atsh)

    # -- outer panel, break to tip, 5 stations -----------------------------
    for k, i in enumerate(range(7, 12)):
        frac = float(k) / 4.0
        y = ys * (1.0 - frac) + yt * frac
        c = cs * (1.0 - frac) + ct * frac
        fc = c / cs
        # §47: cm, cdf and cdp are written cms*(1-frac) + cms*frac, so all
        # three stay at their break values instead of reaching the tip.
        station(i, y,
                JXA=xwbox + (y - yo) * tanL, JYA=y,
                JZA=zwing + (y - yo) * tanD, JCH=c * cosL,
                JTW=twists * (1.0 - frac) + twistt * frac,
                JAL=alphas * (1.0 - frac) + alphat * frac,
                JDCL=dcldas * (1.0 - frac) + dcldat * frac,
                JCM=cms, JCDF=cdfs, JCDP=cdps,
                JMG1=mgs * fc ** 2, JMNN1=mgnns * fc ** 4,
                JCCG1=Ccgs * fc,
                JMG2=mgsfuel * fc ** 2, JMNN2=mgnnsfuel * fc ** 4,
                JCCG2=0.0,
                JECC=EIccs * fc ** 4, JENN=EInns * fc ** 4,
                JGJ=GJs * fc ** 4, JEA=EAs * fc ** 2,
                JCSH=Cshs * fc, JNSH=Nshs * fc, JASH=Atshs * fc ** 3)

    beam.count_all(11)
    beam.define(B.JXA, B.JYA, B.JZA, B.JTW, B.JCH, B.JAL, B.JXAX, B.JDCL,
                B.JCM, B.JCDF, B.JCDP, B.JMG1, B.JMNN1, B.JCCG1, B.JMG2,
                B.JMNN2, B.JCCG2, B.JECC, B.JENN, B.JGJ, B.JEA, B.JCSH,
                B.JNSH, B.JASH)
    return beam


def _htail_beam(parg, para, Lptail, EAfac):
    """Beam 3: centre section and one tapered panel.

    Returns ``(beam, Csh)``, the second being the shell width left in the
    Fortran's local variable when this block finishes -- which the vertical
    tail then picks up. See §50.
    """
    pi = math.pi
    beam = Beam("Horizontal Tail", 3)

    cosLh = math.cos(parg[I.IGSWEEPH] * pi / 180.0)
    tanLh = math.tan(parg[I.IGSWEEPH] * pi / 180.0)
    tanDh = 0.0                                    # hdihed = 0

    coh = parg[I.IGCOH]
    cth = coh * parg[I.IGLAMBDAH]
    boh, bh = parg[I.IGBOH], parg[I.IGBH]
    yoh, yth = 0.5 * boh, 0.5 * bh

    twisth = 0.0
    alphah = -1.0 * pi / 180.0
    dcldah = 2.0 * pi * 1.05
    dcldfh, dcmdfh = 0.08, -0.012                  # the elevator
    cmh = 0.0
    cdfh = para[I.IACDFT] * para[I.IAFEXCDT]
    cdph = para[I.IACDPT] * para[I.IAFEXCDT]

    rhh, wboxh = parg[I.IGRHH], parg[I.IGWBOXH]
    hboxh, tbcaph, tbwebh = (parg[I.IGHBOXH], parg[I.IGTBCAPH],
                             parg[I.IGTBWEBH])
    havgh = hboxh * (1.0 - (1.0 - rhh) / 3.0)
    Xaxh = parg[I.IGXAXIS]
    xhbox = parg[I.IGXHBOX]
    zhtail = parg[I.IGBV] if Lptail else parg[I.IGZHTAIL]

    mghbox = (2.0 * parg[I.IGRHOCAP] * GEE * tbcaph * wboxh
              * (coh * cosLh) ** 2
              + 2.0 * parg[I.IGRHOWEB] * GEE * tbwebh * hboxh * rhh
              * (coh * cosLh) ** 2)
    mghadd = mghbox * parg[I.IGFHADD]
    mgh = mghbox + mghadd
    # Both pieces are put at the box centre, so the first moment is zero.
    dXmgh = 0.0
    dX2mgh = mghbox * wboxh ** 2 / 12.0 + mghadd * (wboxh * 0.5) ** 2
    Ccgh = (dXmgh / mgh) * coh * cosLh
    mgnnh = dX2mgh * (coh * cosLh) ** 2 - mgh * Ccgh ** 2

    Cshh = 0.5 * wboxh * coh * cosLh
    Nshh = 0.5 * hboxh * coh * cosLh
    Atshh = ((wboxh - tbwebh) * (havgh - tbcaph) * tbcaph
             * (coh * cosLh) ** 3)

    EIcch, EInnh, GJh = parg[I.IGEICH], parg[I.IGEINH], parg[I.IGGJH]
    EAh = parg[I.IGECAP] * 2.0 * tbcaph * wboxh * (coh * cosLh) ** 2

    def station(ib, t, **kw):
        beam.put(ib, B.JXAX, Xaxh)
        beam.put(ib, B.JCLF2, dcldfh)
        beam.put(ib, B.JCMF2, dcmdfh)
        for j, v in kw.items():
            beam.put(ib, getattr(B, j), v)
        beam.put_t(ib, t)

    ni = 0
    if yoh != 0.0:
        for i in (1, 2):
            y = yoh * float(i - 1)
            station(i, y, JXA=xhbox, JYA=y, JZA=zhtail, JCH=coh,
                    JTW=twisth, JAL=alphah, JDCL=dcldah, JCM=cmh,
                    JCDF=cdfh, JCDP=cdph, JMG1=mgh, JMNN1=mgnnh,
                    JCCG1=Ccgh, JECC=EIcch, JENN=EInnh, JGJ=GJh,
                    JEA=EAh * EAfac, JCSH=Cshh, JNSH=Nshh, JASH=Atshh)
        ni = 2

    for k in range(5):
        i = ni + 1 + k
        frac = float(k) / 4.0
        y = yoh * (1.0 - frac) + yth * frac
        c = coh * (1.0 - frac) + cth * frac
        fc = c / coh
        station(i, y,
                JXA=xhbox + (y - yoh) * tanLh, JYA=y,
                JZA=zhtail + (y - yoh) * tanDh, JCH=c * cosLh,
                JTW=twisth, JAL=alphah, JDCL=dcldah, JCM=cmh,
                JCDF=cdfh, JCDP=cdph,
                JMG1=mgh * fc ** 2, JMNN1=mgnnh * fc ** 4,
                JCCG1=Ccgh * fc,
                JECC=EIcch * fc ** 4, JENN=EInnh * fc ** 4,
                JGJ=GJh * fc ** 4, JEA=EAh * fc ** 2,
                JCSH=Cshh * fc, JNSH=Nshh * fc, JASH=Atshh * fc ** 3)

    beam.count_all(ni + 5)
    beam.define(B.JXA, B.JYA, B.JZA, B.JTW, B.JCH, B.JAL, B.JXAX, B.JDCL,
                B.JCLF2, B.JCMF2, B.JCM, B.JCDF, B.JCDP, B.JMG1, B.JMNN1,
                B.JCCG1, B.JECC, B.JENN, B.JGJ, B.JEA, B.JCSH, B.JNSH,
                B.JASH)
    return beam, beam.q(ni + 5, B.JCSH)


def _vtail_beam(parg, para, Lptail, yov, ytv, zov, ztv, t0v,
                csh_left_behind) -> Beam:
    """Beam 4: the fin, plus a horizontal cross-connect on a Pi-tail.

    ``ibeam`` is the horizontal tail's, because the two are aerodynamically
    one lifting system as far as ASWING is concerned.
    """
    pi = math.pi
    beam = Beam("Vertical Tail", 4, ibeam=3)

    cosLv = math.cos(parg[I.IGSWEEPV] * pi / 180.0)
    tanLv = math.tan(parg[I.IGSWEEPV] * pi / 180.0)
    cov = parg[I.IGCOV]
    ctv = cov * parg[I.IGLAMBDAV]

    twistv = alphav = 0.0
    dcldav = 2.0 * pi * 1.05
    cmv = 0.0
    cdfv = para[I.IACDFT] * para[I.IAFEXCDT]
    cdpv = para[I.IACDPT] * para[I.IAFEXCDT]

    rhv, wboxv = parg[I.IGRHV], parg[I.IGWBOXV]
    hboxv, tbcapv, tbwebv = (parg[I.IGHBOXV], parg[I.IGTBCAPV],
                             parg[I.IGTBWEBV])
    havgv = hboxv * (1.0 - (1.0 - rhv) / 3.0)
    Xaxv = parg[I.IGXAXIS]
    xvbox = parg[I.IGXVBOX]

    mgvbox = (2.0 * parg[I.IGRHOCAP] * GEE * tbcapv * wboxv
              * (cov * cosLv) ** 2
              + 2.0 * parg[I.IGRHOWEB] * GEE * tbwebv * hboxv * rhv
              * (cov * cosLv) ** 2)
    mgvadd = mgvbox * parg[I.IGFVADD]
    mgv = mgvbox + mgvadd
    dXmgv = 0.0
    dX2mgv = mgvbox * wboxv ** 2 / 12.0 + mgvadd * (wboxv * 0.5) ** 2
    Ccgv = (dXmgv / mgv) * cov * cosLv
    mgnnv = dX2mgv * (cov * cosLv) ** 2 - mgv * Ccgv ** 2

    Cshv = 0.5 * wboxv * cov * cosLv
    Nshv = 0.5 * hboxv * cov * cosLv
    Atshv = ((wboxv - tbwebv) * (havgv - tbcapv) * tbcapv
             * (cov * cosLv) ** 3)

    EIccv, EInnv, GJv = parg[I.IGEICV], parg[I.IGEINV], parg[I.IGGJV]

    def station(ib, t, **kw):
        beam.put(ib, B.JXAX, Xaxv)
        for j, v in kw.items():
            beam.put(ib, getattr(B, j), v)
        beam.put_t(ib, t)

    ni = 0
    if Lptail:
        # The horizontal member joining the two fins. §50: `Csh = Csh`
        # here, so it keeps whatever the variable last held -- which is the
        # *horizontal tail's tip* shell width, not `Cshv`.
        t1v = t0v + yov
        Csh = csh_left_behind
        for i in (1, 2):
            frac = float(i - 1)
            y = yov * frac
            station(i, t0v + y, JXA=xvbox, JYA=y, JZA=zov, JCH=cov,
                    JTW=twistv, JAL=alphav, JDCL=dcldav * 0.3, JCM=0.0,
                    JCDF=cdfv, JCDP=cdpv, JMG1=0.0, JMNN1=0.0, JCCG1=0.0,
                    JECC=0.0, JENN=0.0, JGJ=0.0,
                    JCSH=Csh, JNSH=Nshv, JASH=Atshv * 10.0)
        ni = 2
    else:
        t1v = t0v

    for k in range(5):
        i = ni + 1 + k
        frac = float(k) / 4.0
        c = cov * (1.0 - frac) + ctv * frac
        fc = c / cov
        station(i, t1v + ztv * frac,
                JXA=xvbox + ztv * frac * tanLv,
                JYA=yov * (1.0 - frac) + ytv * frac,
                JZA=zov + ztv * frac, JCH=c * cosLv,
                JTW=twistv, JAL=alphav, JDCL=dcldav, JCM=cmv,
                JCDF=cdfv, JCDP=cdpv,
                JMG1=mgv * fc ** 2, JMNN1=mgnnv * fc ** 4,
                JCCG1=Ccgv * fc,
                JECC=EIccv * fc ** 4, JENN=EInnv * fc ** 4,
                JGJ=GJv * fc ** 4,
                JCSH=Cshv * fc, JNSH=Nshv * fc, JASH=Atshv * fc ** 3)

    beam.count_all(ni + 5)
    beam.define(B.JXA, B.JYA, B.JZA, B.JTW, B.JCH, B.JAL, B.JXAX, B.JDCL,
                B.JCM, B.JCDF, B.JCDP, B.JMG1, B.JMNN1, B.JCCG1, B.JECC,
                B.JENN, B.JGJ, B.JCSH, B.JNSH, B.JASH)
    return beam


def _strut_beam(parg, para, xwbox) -> Beam:
    """Beam 5, only on a strut-braced wing (``iwplan = 2``)."""
    pi = math.pi
    beam = Beam("Strut", 5, ibeam=2)

    tanL = math.tan(parg[I.IGSWEEP] * pi / 180.0)
    tanD = math.tan(0.0)            # iwplan = 2 means wdihed = 0
    yo, ys = 0.5 * parg[I.IGBO], 0.5 * parg[I.IGBS]
    dzs = -0.1 * parg[I.IGCO] * parg[I.IGLAMBDAS]
    zwing = parg[I.IGZWING]

    xo = xwbox
    xs = xwbox + (ys - yo) * tanL
    zo = zwing - parg[I.IGZS]
    zs = zwing + (ys - yo) * tanD + dzs
    cosLs = parg[I.IGCOSLS]

    Estrut = parg[I.IGECAP]
    Gstrut = 0.5 * Estrut / (1.0 + 0.3)
    Astrut, hstrut = parg[I.IGASTRUT], parg[I.IGHSTRUT]
    cstrut = parg[I.IGCSTRUT]
    dclda = 2.0 * pi * 0.9

    Xax = 0.25
    Csh = cstrut * (1.0 - Xax)
    Nsh = cstrut * hstrut * 0.5
    EA = Estrut * Astrut
    EIcc = Estrut * 0.14 * Astrut * cstrut ** 2 * hstrut ** 2
    EInn = Estrut * 0.08 * Astrut * cstrut ** 2 * hstrut
    GJ = Gstrut * 0.40 * Astrut * cstrut ** 2 * hstrut ** 2
    mg = parg[I.IGRHOSTRUT] * GEE * Astrut
    mgcc = mg * 0.07 * hstrut ** 2 * cstrut ** 2
    mgnn = mg * 0.07 * cstrut ** 2

    common = dict(JTW=0.0, JDCL=dclda, JCDF=para[I.IACDFS],
                  JCDP=para[I.IACDPS], JMG1=mg, JMCC1=mgcc, JMNN1=mgnn,
                  JECC=EIcc, JENN=EInn, JGJ=GJ, JEA=EA, JCSH=Csh,
                  JNSH=Nsh)

    def station(ib, t, x, y, z, cnorm):
        beam.put(ib, B.JXAX, Xax)
        beam.put(ib, B.JXA, x)
        beam.put(ib, B.JYA, y)
        beam.put(ib, B.JZA, z)
        beam.put(ib, B.JCH, cnorm)
        for j, v in common.items():
            beam.put(ib, getattr(B, j), v)
        beam.put_t(ib, t)

    for i in (1, 2):
        y = yo * float(i - 1)
        station(i, y, xo, y, zo, cstrut)
    for k in range(4):
        i = 3 + k
        frac = float(k) / 3.0
        y = yo * (1.0 - frac) + ys * frac
        station(i, y, xo * (1.0 - frac) + xs * frac, y,
                zo * (1.0 - frac) + zs * frac, cstrut * cosLs)

    beam.count_all(6)
    beam.define(B.JXAX, B.JXA, B.JYA, B.JZA, B.JCH, B.JTW, B.JDCL, B.JCDF,
                B.JCDP, B.JMG1, B.JMNN1, B.JECC, B.JENN, B.JGJ, B.JEA,
                B.JCSH, B.JNSH)
    return beam


# --------------------------------------------------------------------------
# BOUTPUT -- write the deck
# --------------------------------------------------------------------------

RULE = "#" + "=" * 40
BEAM_RULE = "#" + "=" * 48
BLOCK_RULE = "#" + "-" * 34


def _qunit(deck) -> dict:
    """``QUNIT(J)`` -- what each beam variable is divided by on the way out.

    All 1.0 here because TASOPT writes SI with unit scale factors, but kept
    as the source has it so a deck in other units would come out right.
    """
    L, F, T = deck.unitl, deck.unitf, deck.unitt
    dtor = math.atan(1.0) / 45.0
    u = {j: 1.0 for j in range(1, B.JBTOT + 1)}
    for j in (B.JXA, B.JYA, B.JZA, B.JCH, B.JCCG1, B.JNCG1, B.JCCG2,
              B.JNCG2, B.JCEA, B.JNEA, B.JCSH, B.JNSH):
        u[j] = L
    u[B.JTW] = u[B.JAL] = dtor
    for j in (B.JECC, B.JENN, B.JECN, B.JGJ):
        u[j] = F * L ** 2
    u[B.JEA] = F
    u[B.JMG1] = u[B.JMG2] = F / L
    for j in (B.JMCC1, B.JMNN1, B.JMCC2, B.JMNN2):
        u[j] = F * L
    u[B.JTDE] = u[B.JTDG] = T
    u[B.JASH] = L ** 3
    return u


def _qfac(qmax: float) -> float:
    """The column scale factor: ``10**int(log10(2*qmax))``.

    The factor of two is why a column topping out at 6.0 is scaled by 10.
    """
    if qmax == 0.0:
        return 1.0
    iexp = int(math.log10(2.0 * qmax) + 100.0) - 100
    return 10.0 ** iexp


def _g11_1(v: float) -> str:
    """``G11.1``, which is how the scale factors are written."""
    return _gfmt(v, 11, 1)


def _f11_5(v: float) -> str:
    return f"{v:11.5f}"


def _g14_6(v: float) -> str:
    return _gfmt(v, 14, 6)


def _g13_5(v: float) -> str:
    return _gfmt(v, 13, 5)


def _pad(s: str, n: int) -> str:
    """A fixed-length Fortran ``character*n``, blank-padded."""
    return s[:n].ljust(n)


#: ``(heading, QPUNIT index)`` for each pylon kind, in the order BOUTPUT
#: writes them. The engine set is for ``IENGTYP = 0``, the only kind
#: ``aswout`` ever produces.
_WEIGHT_COLUMNS = ("      Xp   ", "      Yp   ", "      Zp   ",
                   "      Mg   ", "      CDA  ", "      Vol  ",
                   "      Hxg  ", "      Hyg  ", "      Hzg  ")
_ENGINE_COLUMNS = ("      Xp   ", "      Yp   ", "      Zp   ",
                   "      Tx   ", "      Ty   ", "      Tz   ",
                   "   dFdPeng ", "   dMdPeng ")


def boutput(deck) -> str:
    """Write a deck as an ``.asw`` file, as ``BOUTPUT`` writes it."""
    L, M, T, F = deck.unitl, deck.unitm, deck.unitt, deck.unitf
    out = []

    out.append(RULE)
    out.append("Name")
    out.append(_pad(deck.name, 64))
    out.append("End")

    out.append(RULE)
    out.append("Unit")
    # The M unit is never written, although the Constant line below scales
    # the density by it.
    for tag, val, name in (("L", L, deck.unchl), ("T", T, deck.uncht),
                           ("F", F, deck.unchf)):
        out.append(tag + _g13_5(val) + " " + _pad(name, 32))
    out.append("End")

    out.append(RULE)
    out.append("Constant")
    out.append("#" + "".join(_pad(h, 13) for h in
                             ("       g     ", "     rhoSL   ",
                              "     VsoSL   ")))
    out.append(" " + "".join(_g13_5(v) for v in
                             (deck.gee * T ** 2 / L,
                              deck.rhoSL * L ** 3 / M,
                              deck.aSL * T / L)))
    out.append("End")

    out.append(RULE)
    out.append("Reference")
    out.append("#" + "".join(("      Sref   ", "      Cref   ",
                              "      Bref   ")))
    out.append(" " + "".join(_g13_5(v) for v in
                             (deck.Sref / L ** 2, deck.Cref / L,
                              deck.Bref / L)))
    for k, names in enumerate((("      Xmom   ", "      Ymom   ",
                                "      Zmom   "),
                               ("      Xacc   ", "      Yacc   ",
                                "      Zacc   "),
                               ("      Xvel   ", "      Yvel   ",
                                "      Zvel   "))):
        out.append("#")
        out.append("#" + "".join(names))
        out.append(" " + "".join(_g13_5(deck.xyzref[r][k] / L)
                                 for r in range(3)))
    out.append("End")

    out += _pylon_blocks(deck)
    out += _joint_block(deck)
    out += _ground_block(deck)
    for beam in deck.beams:
        out += _beam_block(deck, beam)

    return "".join(line + "\n" for line in out)


def _pylon_blocks(deck) -> list:
    """The Weight, Strut, Engine and Sensor blocks, in that order."""
    L, T, F = deck.unitl, deck.unitt, deck.unitf
    out = []

    kinds = {1: [], 2: [], 3: [], 4: []}
    for p in deck.pylons:
        if p.kptype == KP_WEIGHT:
            kinds[1].append(p)
        elif p.kptype == KP_STRUT:
            kinds[2].append(p)
        elif p.kptype > KP_SENSOR:
            kinds[4].append(p)
        elif p.kptype > KP_ENGINE:
            kinds[3].append(p)
        else:
            raise ValueError(f"BOUTPUT: illegal pylon type {p.kptype}")

    for iptype in (1, 2, 3, 4):
        if not kinds[iptype]:
            continue
        if iptype == 1:
            nj = 9
            qpunit = [1.0, L, L, L, F, L ** 2, L ** 3] + \
                     [F * L ** 2 / T] * 3
            title, columns, leaders = "Weight", _WEIGHT_COLUMNS, (" Nbeam",)
            k0 = 2 + 6 + 14
        elif iptype == 3:
            nj = 8                      # IENGTYP = 0
            qpunit = [1.0, L, L, L, 1.0, 1.0, 1.0, F, F * L]
            title, columns = "Engine", _ENGINE_COLUMNS
            leaders = (" KPeng", " IEtyp", " Nbeam")
            k0 = 2 + 18 + 14
        else:
            raise NotImplementedError(
                f"aswout never builds pylon type {iptype}; BOUTPUT can "
                "write it but nothing here can produce one")

        out.append(RULE)
        out.append(title)
        out.append("#" + "".join(leaders) + "       t       "
                   + "".join(columns))

        # The scale-factor line: a '*', then a fixed '1.' for the t column
        # at a kind-dependent offset, then one factor per column.
        qfac = [1.0] * (nj + 1)
        for j in range(1, nj + 1):
            qmax = max(abs(p.q[j]) for p in kinds[iptype])
            qfac[j] = _qfac(qmax / qpunit[j])
        # `LINE(K0-14:K0-1) = '       1.     '`, and the first factor goes
        # at K0+1 -- so the prefix is K0 characters, one of them blank.
        line = [" "] * k0
        line[0] = "*"
        line[k0 - 15:k0 - 1] = list("       1.     ")
        line = "".join(line) + "".join(_g11_1(qfac[j])
                                       for j in range(1, nj + 1))
        out.append(line)

        for p in kinds[iptype]:
            if iptype == 1:
                head = f"{p.kbeam:6d}"
            else:
                ke = p.kptype - KP_ENGINE
                head = (f"{ke:6d}{deck.engtyp[ke - 1]:6d}"
                        f"{p.kbeam:6d}")
            out.append(head + "  " + _g14_6(p.q[0])
                       + "".join(_f11_5(p.q[j] / qfac[j] / qpunit[j])
                                 for j in range(1, nj + 1)))
        out.append("End")
    return out


def _joint_block(deck) -> list:
    if not deck.joints:
        return []
    out = [RULE, "Joint",
           "# Nbeam1 Nbeam2      t1            t2         KJtype"]
    for j in deck.joints:
        out.append(f"{j.beams[0]:7d}{j.beams[1]:7d}  "
                   + _g14_6(j.t[0]) + _g14_6(j.t[1]) + f"{j.kjtype:5d}")
    out.append("End")
    return out


def _ground_block(deck) -> list:
    if not deck.grounds:
        return []
    out = [RULE, "Ground", "#  Nbeam       t         KGtype"]
    for g in deck.grounds:
        out.append(f"{g.beam:7d}  " + _g14_6(g.t) + f"{g.kgtype:5d}")
    out.append("End")
    return out


def _beam_block(deck, beam) -> list:
    """One beam, as however many column blocks its griddings need."""
    qunit = _qunit(deck)
    out = [BEAM_RULE]
    if beam.ibeam == beam.kbnum:
        out.append(f"Beam{beam.kbnum:7d}")
    else:
        out.append(f"Beam{beam.kbnum:7d}{beam.ibeam:7d}")
    out.append(_pad(beam.name, 64))

    remaining = sorted(beam.defined)
    while remaining:
        jkey = remaining[0]
        group = [jkey]
        n = beam.nb[jkey]
        # Everything later with the same number of stations at the same
        # positions joins this block, up to six columns.
        for j in remaining[1:]:
            if beam.nb[j] != n:
                continue
            deltn = abs(beam.t(n, j) - beam.t(1, j))
            if any(abs(beam.t(i, j) - beam.t(i, jkey)) > 1e-5 * deltn
                   for i in range(1, n + 1)):
                continue
            group.append(j)
            if len(group) >= 6:
                break
        for j in group:
            remaining.remove(j)

        out.append(BLOCK_RULE)
        out.append("       t        " + "".join(B.VARS[j] for j in group))

        qfac = {}
        for j in group:
            qmax = max(abs(beam.q(i, j)) for i in range(1, n + 1))
            qfac[j] = _qfac(qmax / qunit[j])
        out.append("*      1.      "
                   + "".join(_g11_1(qfac[j]) for j in group))

        for i in range(1, n + 1):
            out.append(" " + _g14_6(beam.t(i, jkey))
                       + "".join(_f11_5(beam.q(i, j) / qfac[j] / qunit[j])
                                 for j in group))

    out.append("End")
    return out
