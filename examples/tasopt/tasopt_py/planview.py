"""The aircraft outline, and the plot files -- ``airpic.f`` and ``pltwrt``.

``airpic`` turns a sized aircraft into four polylines -- wing, horizontal
tail, fuselage and nacelle -- for drawing a plan view. It is the only place in
TASOPT that builds a *picture* of the result, and it is useful well beyond the
Matlab and gnuplot files it was written for: given a converged case,
:func:`airpic` is what you plot.

``pltwrt`` writes the three Matlab files, one per ``i``/``j`` grid point: a
label file, a 28-column parameter file, and a geometry file holding the
polylines. The columns are listed in :data:`PLOT_COLUMNS`.

Only the *starboard half* is built, from the centreline out. The fuselage
polyline is a nose superellipse, a straight barrel and a tail taper, closed
back to the axis; the nacelle is a stylised cowl outline placed at the engine
station.

Things in the source worth knowing
----------------------------------
* ``airpic`` hard-wires a lift coefficient of 0.70 and a pitching moment of
  -0.1 at the top, computes ``clp`` and ``cmp`` from them, and then never uses
  either. Leftovers from a version that drew loadings.
* The spanwise axis is placed at a hard-wired 40% chord for both wing and
  tail (``xax = 0.40``); ``Xaxis`` from the ``.tas`` file, which the
  structural and moment calculations use, is never read. The 737 sets
  ``Xaxis`` to 0.40 too, so they coincide there -- on a case that does not,
  the drawn planform is pinned differently from the modelled one.
* A whole block adjusting the tail root to follow the fuselage contour sits
  inside ``if(.false.)``, so the horizontal tail is drawn with a straight root
  whatever the fuselage does behind it.
* ``coh`` is recomputed here from ``Sh``, ``ARh`` and ``lambdah`` rather than
  read from ``parg(igcoh)``, which ``wsize`` has already set. The two agree on
  a converged case.

Verified against the compiled Fortran; see ``tests/test_planview.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .model import indices as I
from .output import _gfmt

__all__ = ["airpic", "PlanView", "pltwrt", "PLOT_COLUMNS", "NNOSE", "NTAIL",
           "PLOT_XAXIS"]

#: Points along the nose and tail profiles.
NNOSE, NTAIL = 8, 5
#: Chordwise location of the spanwise axis, for drawing only. The model's own
#: axis is ``parg(igXaxis)`` and is not this.
PLOT_XAXIS = 0.40

FT_M = 1.0 / 0.3048
LB_N = 1.0 / 4.44822
KFT_M = 1.0 / 0.3048 / 1000.0
IN_M = 39.37
NMI_M = 0.000539975
HR_S = 1.0 / 3600.0

#: What the 28 columns of the parameter plot file hold, in order.
PLOT_COLUMNS = (
    "PFEI", "2 Wfuel [lb]", "2 Wfmax [lb]", "WMTO [lb]", "AR", "L/D",
    "hboxs", "CL", "CLperp", "alt [kft]", "FPR", "BPR", "dfan [in]",
    "Weng [lb]", "FTO [lb]", "lTO [ft]", "lBF [ft]", "sin(gamma_BF)",
    "V2 [kt]", "u6 takeoff [m/s]", "u8 takeoff [m/s]", "dB sideline",
    "dB cutback", "dB flyover", "fc takeoff", "Tt4 takeoff [K]",
    "Tt4 cruise [K]", "sweep [deg]")


@dataclass
class PlanView:
    """Four starboard-half polylines, in metres, in aircraft axes."""
    wing: list = field(default_factory=list)        # [(x, y), ...]
    htail: list = field(default_factory=list)
    fuselage: list = field(default_factory=list)
    nacelle: list = field(default_factory=list)     # about its own origin

    def nacelle_at(self, xeng, yeng):
        """The nacelle outline moved to an engine station."""
        return [(xeng + x, yeng + y) for x, y in self.nacelle]


def airpic(pari, parg) -> PlanView:
    """Build the plan-view outline of a sized aircraft."""
    ifclose = pari[I.IIFCLOSE]

    Rfuse, wfb = parg[I.IGRFUSE], parg[I.IGWFB]
    anose, btail = parg[I.IGANOSE], parg[I.IGBTAIL]
    xnose, xend = parg[I.IGXNOSE], parg[I.IGXEND]
    xblend1, xblend2 = parg[I.IGXBLEND1], parg[I.IGXBLEND2]
    xwbox, xhbox = parg[I.IGXWBOX], parg[I.IGXHBOX]

    co, bo, bs, b = parg[I.IGCO], parg[I.IGBO], parg[I.IGBS], parg[I.IGB]
    sweep = parg[I.IGSWEEP]
    lambdat, lambdas = parg[I.IGLAMBDAT], parg[I.IGLAMBDAS]

    boh, Sh, ARh = parg[I.IGBOH], parg[I.IGSH], parg[I.IGARH]
    lambdah, sweeph = parg[I.IGLAMBDAH], parg[I.IGSWEEPH]

    # Recomputed here rather than read from parg(igcoh); they agree on a
    # converged case.
    bh = math.sqrt(Sh * ARh)
    coh = Sh / (boh + (bh - boh) * 0.5 * (1.0 + lambdah))

    tanL = math.tan(sweep * math.pi / 180.0)

    # A CL of 0.70 and a cm of -0.1 are set at the top of airpic.f and used
    # only to form clp and cmp, which are then never read.

    xcLE, xcTE = -PLOT_XAXIS, 1.0 - PLOT_XAXIS
    cs, ct = co * lambdas, co * lambdat
    xs = tanL * (bs - bo) / 2.0
    xt = tanL * (b - bo) / 2.0

    dx = xwbox
    wing = [(co * xcLE + dx, bo / 2.0),
            (xs + cs * xcLE + dx, bs / 2.0),
            (xt + ct * xcLE + dx, b / 2.0),
            (xt + ct * xcTE + dx, b / 2.0),
            (xs + cs * xcTE + dx, bs / 2.0),
            (co * xcTE + dx, bo / 2.0)]

    # --- fuselage: nose superellipse, barrel, tail taper, close to axis ---
    hwidth = Rfuse + wfb
    dytail = -hwidth if ifclose == 0 else -0.2 * hwidth

    fuselage = []
    for i in range(NNOSE):
        fraci = float(i) / float(NNOSE - 1)
        fracx = math.cos(0.5 * math.pi * fraci)
        fuselage.append((xblend1 + (xnose - xblend1) * fracx,
                         hwidth * (1.0 - fracx ** anose) ** (1.0 / anose)))
    for i in range(NTAIL):
        fracx = float(i) / float(NTAIL - 1)
        fuselage.append((xblend2 + (xend - xblend2) * fracx,
                         hwidth + dytail * fracx ** btail))
    fuselage.append((fuselage[-1][0], 0.0))

    # --- horizontal tail ---------------------------------------------------
    dx = xhbox
    tanLh = math.tan(sweeph * math.pi / 180.0)
    cth = coh * lambdah

    xoLEh = coh * (0.0 - PLOT_XAXIS) + dx
    xoTEh = coh * (1.0 - PLOT_XAXIS) + dx
    xtLEh = cth * (0.0 - PLOT_XAXIS) + dx + 0.5 * (bh - boh) * tanLh
    xtTEh = cth * (1.0 - PLOT_XAXIS) + dx + 0.5 * (bh - boh) * tanLh
    yoLEh = yoTEh = 0.5 * boh
    ytLEh = ytTEh = 0.5 * bh

    if ifclose == 0:
        xcLEh, xcTEh, ycLEh, ycTEh = xoLEh, xoTEh, yoLEh, yoTEh
    else:
        xcLEh = coh * (0.0 - PLOT_XAXIS) + dx + 0.5 * (0.0 - boh) * tanLh
        xcTEh = coh * (1.0 - PLOT_XAXIS) + dx + 0.5 * (0.0 - boh) * tanLh
        ycLEh = ycTEh = 0.0

    # A block here that would walk the tail root back along the fuselage
    # contour is inside `if(.false.)`, so the root stays straight.

    htail = [(xcLEh, ycLEh), (xoLEh, yoLEh), (xtLEh, ytLEh),
             (xtTEh, ytTEh), (xoTEh, yoTEh), (xcTEh, ycTEh)]

    # --- nacelle, about its own origin -------------------------------------
    dfan, rSnace = parg[I.IGDFAN], parg[I.IGRSNACE]
    Afan = 0.25 * math.pi * dfan ** 2
    lnace = dfan * rSnace * 0.15
    Snace = Afan * rSnace
    dnace = (Snace * 0.85) / (math.pi * lnace)

    xeLE, xeTE = -0.5 * lnace, 0.5 * lnace
    yeLE, yeTE = 0.5 * dnace, 0.4 * dnace
    nacelle = [(xeTE, 0.0), (xeTE, yeTE), (xeLE + 0.3 * lnace, yeLE),
               (xeLE, yeLE), (xeLE, -yeLE), (xeLE + 0.3 * lnace, -yeLE),
               (xeTE, -yeTE), (xeTE, 0.0)]

    return PlanView(wing=wing, htail=htail, fuselage=fuselage,
                    nacelle=nacelle)


def _g13(values):
    """``format(1x,30g13.5)``."""
    return " " + "".join(_gfmt(v, 13, 5) for v in values)


def plot_row(parg, parm, para, pare, parsjspec=0.0) -> list:
    """The 28 numbers ``pltwrt`` writes per grid point, plus the sweep value.

    Returned rather than written, so a caller can tabulate them however it
    likes; :func:`pltwrt` formats them the way the Matlab reader expects.
    """
    ip = I.IPCRUISE1
    LoD = para[I.IACL, ip] / para[I.IACD, ip]
    cosL = math.cos(parg[I.IGSWEEP] * math.pi / 180.0)
    CLperp = para[I.IACL, ip] / cosL ** 2
    return [parsjspec,
            parg[I.IGPFEI],
            2.0 * parg[I.IGWFUEL] * LB_N,
            2.0 * parg[I.IGWFMAX] * LB_N,
            parg[I.IGWMTO] * LB_N,
            parg[I.IGAR],
            LoD,
            parg[I.IGHBOXS],
            para[I.IACL, ip],
            CLperp,
            para[I.IAALT, ip] * KFT_M,
            pare[I.IEPIF, ip],
            pare[I.IEBPR, ip],
            parg[I.IGDFAN] * IN_M,
            parg[I.IGWENG] * LB_N,
            parm[I.IMFTO] * LB_N,
            parm[I.IMLTO] * FT_M,
            parm[I.IMLBF] * FT_M,
            math.sin(parm[I.IMGAMVBF]),
            parm[I.IMV2] * NMI_M / HR_S,
            pare[I.IEU6, I.IPTAKEOFF],
            pare[I.IEU8, I.IPTAKEOFF],
            parm[I.IMDBSL],
            parm[I.IMDBCB],
            parm[I.IMDBFO],
            pare[I.IEFC, I.IPTAKEOFF],
            pare[I.IETT4, I.IPTAKEOFF],
            pare[I.IETT4, I.IPCRUISE1],
            parg[I.IGSWEEP]]


def pltwrt(view, parg, parm, para, pare, parsjspec=0.0) -> tuple:
    """The parameter and geometry lines for one grid point, in feet.

    Returns ``(parameter_line, geometry_lines)``. The label file is static
    text and the headers depend on the grid, so a caller assembles the files;
    this is the per-point content, formatted as the Fortran writes it.
    """
    row = _g13(plot_row(parg, parm, para, pare, parsjspec))

    geom = []
    for poly in (view.wing, view.htail, view.fuselage):
        geom += [_g13([x * FT_M, y * FT_M]) for x, y in poly]
    geom += [_g13([x * FT_M, y * FT_M]) for x, y in
             view.nacelle_at(parg[I.IGXENG], parg[I.IGYENG])]
    return row, geom
