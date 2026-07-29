"""The ``.out`` report -- ``output.f`` and the writer block in ``tasopt.f``.

TASOPT writes a plain-text report of everything it computed. This module
reproduces it, byte for byte, so a ported run can be diffed against the
reference program's own ``737.out`` rather than against a handful of numbers
somebody chose to check.

What is covered
---------------
The summary the report opens with, and repeats per fleet mission:

===================================  ==========================
header, ``Fleet PFEI``               ``tasopt.f``
``Airframe parameters...``           ``geowrt``
``Fuselage BL+Wake development...``  ``blfwrt``
``Cruise performance...``            ``tasopt.f``
``Takeoff performance...``           ``tofwrt``
``Mission profile summary...``       ``prfwrt``
``Aero, Engine parameters...``       ``airwrt``, ``engwrt``
===================================  ==========================

All of it: :func:`report` reproduces the reference program's ``737.out``
byte for byte, all 4565 lines.

Two more writers live here that the shipped program cannot reach
------------------------------------------------------------------
:func:`mapwrt` (compressor operating points, ``tfan_MMM.dat``) and
:func:`prfwrt` (the mission profile, ``prof_MMM.dat``) are guarded by
``Ltfwrite`` and ``Lpfwrite``, and **both are hard-wired ``.false.``** in
``tasopt.f`` with the ``.true.`` line commented out directly beneath. Their
``getLval`` reads are commented out of ``getparm.f`` as well, so no ``.tas``
file can switch them on either. Reaching them means editing the source and
recompiling; see ``DISCREPANCIES.md`` §45. Both are ported here and both are
verified against files produced by doing exactly that. ``prfwrt`` also
appears inside the ``.out`` report, so only ``mapwrt`` is otherwise
unreachable.

Reproducing Fortran's formatted output
--------------------------------------
Three things have to be right or the diff is noise:

* ``write(lu,*)`` with no items writes a **zero-length** record, not a line
  with a leading blank. With items it prefixes one blank.
* Fixed-length ``character`` variables print padded to their declared length,
  so ``configname`` (``character*64``) always occupies 64 columns whatever is
  in it.
* Fortran's ``fw.d`` and Python's ``%w.df`` agree, including on how they round
  ties, because both go through the platform's double-to-decimal conversion.
  A value that lands exactly on a rounding boundary could still differ; none
  in the shipped case does.

``lunit``-style unit conversion is done at the print, not in the model: the
whole program is SI internally and the report is in feet and pounds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .atmosphere import atmos
from .model import indices as I

__all__ = ["report", "geowrt", "tofwrt", "prfwrt", "blfwrt", "airwrt",
           "engwrt", "mapwrt", "map_point", "MapPoint", "tfwrt", "VERSION",
           "LB_N", "FT_M", "KFT_M", "IN_M", "PSI_PA", "NMI_M", "HR_S"]

VERSION = 2.16

# constants.inc's unit conversion factors.
LB_N = 1.0 / 4.44822
FT_M = 1.0 / 0.3048
KFT_M = 1.0 / 0.3048 / 1000.0
IN_M = 39.37
PSI_PA = 0.000145038
NMI_M = 0.000539975
HR_S = 1.0 / 3600.0
GAMSL = 1.4
#: constants.inc sets Tref = TSL and pref = pSL from atmos(0).
_TREF, _PREF = atmos(0.0).T, atmos(0.0).p

#: ``write(lu,1010)`` and ``write(lu,1020)`` -- each is preceded by a blank
#: line, which is what the leading ``/`` in the format does.
RULE1 = " " + "=" * 61
RULE2 = " " + "-" * 59
RULE3 = " " + "- " * 28 + "-"   # the literal, before the list-directed blank


def _ffmt(v: float, w: int, d: int) -> str:
    """Fortran ``Fw.d``.

    Identical to Python's ``%w.df`` except at ``d = 0``, where Fortran keeps
    the decimal point -- ``3510566.`` where Python writes ``3510566``.
    """
    if d > 0:
        return f"{v:{w}.{d}f}"
    return (f"{v:.0f}.").rjust(w)


def _efmt(v: float, w: int, d: int) -> str:
    """Fortran ``Ew.d``.

    Not C's ``%E``. Fortran normalises the mantissa to ``0.1 <= m < 1`` with a
    scale factor of zero, so 2067 prints as ``0.207E+04`` where C gives
    ``2.067E+03``.
    """
    if v == 0.0:
        return ("0." + "0" * d + "E+00").rjust(w)
    e = math.floor(math.log10(abs(v))) + 1
    m = v / 10.0 ** e
    body = f"{m:.{d}f}"
    if abs(float(body)) >= 1.0:          # rounded up to 1.000
        e += 1
        m = v / 10.0 ** e
        body = f"{m:.{d}f}"
    sign = "+" if e >= 0 else "-"
    return f"{body}E{sign}{abs(e):02d}".rjust(w)


def _gfmt(v: float, w: int, d: int) -> str:
    """Fortran ``Gw.d``.

    ``F(w-4, d-N)`` plus four blanks while ``0.1 <= |v| < 10**d``, where ``N``
    is the decimal exponent; ``E(w,d)`` outside that. A zero is treated as
    ``N = 1``, which is why the column shows ``0.00`` and not ``0.000``.
    """
    if v == 0.0:
        n = 1
    else:
        # The decimal exponent is taken from the value after rounding to d
        # significant digits, so 99.999999 at d = 5 counts as 100.00 (N = 3)
        # and not as 99.9999 (N = 2).
        n = math.floor(math.log10(abs(float(f"{v:.{d - 1}e}")))) + 1
    if 0 <= n <= d:
        return _ffmt(v, w - 4, d - n) + " " * 4
    return _efmt(v, w, d)


def _blank():
    """``write(lu,*)`` with no items: a zero-length record."""
    return ""


def _say(*parts):
    """``write(lu,*) 'text'``: list-directed, so one leading blank."""
    return " " + "".join(parts)


def _c64(s: str) -> str:
    """A ``character*64`` printed whole -- padded, or truncated."""
    return s[:64].ljust(64)


def geowrt(pari, parg) -> list:
    """``geowrt`` -- airframe geometry and weights."""
    out = []

    def w(label, value, fmt, unit=""):
        out.append(f" {label}{value:{fmt}}{unit}")

    Wpadd = parg[I.IGWPAY] * parg[I.IGFPADD]
    Wseat = parg[I.IGWPAY] * parg[I.IGFSEAT]
    Wapu = parg[I.IGWPAY] * parg[I.IGFAPU]

    Whpesys = parg[I.IGWMTO] * parg[I.IGFHPESYS]
    Wlgnose = parg[I.IGWMTO] * parg[I.IGFLGNOSE]
    Wlgmain = parg[I.IGWMTO] * parg[I.IGFLGMAIN]
    Wtotadd = Whpesys + Wlgnose + Wlgmain

    Wbox = parg[I.IGWWEB] + parg[I.IGWCAP]
    Wflap = Wbox * parg[I.IGFFLAP]
    Wslat = Wbox * parg[I.IGFSLAT]
    Waile = Wbox * parg[I.IGFAILE]
    Wlete = Wbox * parg[I.IGFLETE]
    Wribs = Wbox * parg[I.IGFRIBS]
    Wspoi = Wbox * parg[I.IGFSPOI]
    Wwatt = Wbox * parg[I.IGFWATT]
    Wwing = Wbox + Wflap + Wslat + Waile + Wlete + Wribs + Wspoi + Wwatt

    Wempty = parg[I.IGWMTO] - parg[I.IGWFUEL] - parg[I.IGWPAY]
    Weadd = parg[I.IGWEBARE] * parg[I.IGFEADD]
    Wpylon = ((parg[I.IGWEBARE] + Weadd + parg[I.IGWNACE])
              * parg[I.IGFPYLON])

    for label, v in (("Wempty  +", Wempty), ("Wpay    +", parg[I.IGWPAY]),
                     ("Wfuel   +", parg[I.IGWFUEL]),
                     ("(max)    ", parg[I.IGWFMAX]),
                     ("WMTO    =", parg[I.IGWMTO])):
        w(label, v * LB_N, "9.1f", "  lb")
    out.append(_blank())
    for label, v in (("Whpesys +", Whpesys), ("Wlgnose +", Wlgnose),
                     ("Wlgmain +", Wlgmain), ("Wtotadd =", Wtotadd)):
        w(label, v * LB_N, "9.1f", "  lb")
    out.append(_blank())
    for label, v in (("Wfix    +", parg[I.IGWFIX]), ("Wapu    +", Wapu),
                     ("Wpadd   +", Wpadd), ("Wshell  +", parg[I.IGWSHELL]),
                     ("Wcone   +", parg[I.IGWCONE]),
                     ("Whbend  +", parg[I.IGWHBEND]),
                     ("Wvbend  +", parg[I.IGWVBEND]),
                     ("Wwindow +", parg[I.IGWWINDOW]),
                     ("Winsul  +", parg[I.IGWINSUL]),
                     ("Wfloor  +", parg[I.IGWFLOOR]), ("Wseat   +", Wseat),
                     ("Wfuse   =", parg[I.IGWFUSE])):
        w(label, v * LB_N, "9.1f", "  lb")
    out.append(_blank())
    for label, v in (("Wcap    +", parg[I.IGWCAP]),
                     ("Wweb    +", parg[I.IGWWEB]),
                     ("Wflap   +", Wflap), ("Wslat   +", Wslat),
                     ("Waile   +", Waile), ("Wlete   +", Wlete),
                     ("Wribs   +", Wribs), ("Wspoi   +", Wspoi),
                     ("Wwatt   +", Wwatt), ("Wwing   =", Wwing)):
        w(label, v * LB_N, "9.1f", "  lb")
    out.append(_blank())
    for label, v in (("Wstrut  =", parg[I.IGWSTRUT]),
                     ("Whtail  =", parg[I.IGWHTAIL]),
                     ("Wvtail  =", parg[I.IGWVTAIL])):
        w(label, v * LB_N, "9.1f", "  lb")
    out.append(_blank())
    for label, v in (("Webare  +", parg[I.IGWEBARE]), ("Weadd   +", Weadd),
                     ("Wnace   +", parg[I.IGWNACE]), ("Wpylon  +", Wpylon),
                     ("Weng    =", parg[I.IGWENG])):
        w(label, v * LB_N, "9.1f", "  lb")
    out.append(_blank())
    out.append(f" ifwcen  ={pari[I.IIFWCEN]:4d}")
    out.append(_blank())
    out.append(f" neng    ={int(parg[I.IGNENG]):8d}")
    w("dfan    =", parg[I.IGDFAN] * IN_M, "9.3f", "  in")
    w("lnace   =", parg[I.IGLNACE] * FT_M, "9.3f", "  ft")
    w("Snace/S =", parg[I.IGFSNACE], "9.4f")
    out.append(_blank())
    w("w/c     =", parg[I.IGWBOX], "8.3f")
    w("ho/c    =", parg[I.IGHBOXO], "8.4f")
    w("hs/c    =", parg[I.IGHBOXS], "8.4f")
    for label, idx in (("tcapo/c =", I.IGTBCAPO), ("tcaps/c =", I.IGTBCAPS),
                       ("twebo/c =", I.IGTBWEBO), ("twebs/c =", I.IGTBWEBS)):
        w(label, parg[idx], "8.5f")

    cosL = math.cos(parg[I.IGSWEEP] * math.pi / 180.0)
    co = parg[I.IGCO]
    cs = co * parg[I.IGLAMBDAS]
    ct = co * parg[I.IGLAMBDAT]
    for label, v in (("tcapo   =", parg[I.IGTBCAPO] * co * cosL),
                     ("tcaps   =", parg[I.IGTBCAPS] * cs * cosL),
                     ("tcapt   =", parg[I.IGTBCAPS] * ct * cosL),
                     ("twebo   =", parg[I.IGTBWEBO] * co * cosL),
                     ("twebs   =", parg[I.IGTBWEBS] * cs * cosL),
                     ("twebt   =", parg[I.IGTBWEBS] * ct * cosL)):
        w(label, v * IN_M, "8.5f", "  in  ")

    w("So_max  =", parg[I.IGSOMAX] * LB_N, "9.1f", "  lb  ")
    w("Ss_max  =", parg[I.IGSSMAX] * LB_N, "9.1f", "  lb  ")
    out.append(" Mo_max  ="
               + _ffmt(parg[I.IGMOMAX] * LB_N * FT_M, 9, 0) + " ft-lb")
    out.append(" Ms_max  ="
               + _ffmt(parg[I.IGMSMAX] * LB_N * FT_M, 9, 0) + " ft-lb")
    out.append(_blank())
    w("Nlift   =", parg[I.IGNLIFT], "8.3f")
    w("Nland   =", parg[I.IGNLAND], "8.3f")
    w("Vne     =", parg[I.IGVNE] * NMI_M / HR_S, "8.1f", " kt")
    out.append(_blank())

    xLEw = parg[I.IGXWBOX] - parg[I.IGXAXIS] * parg[I.IGCO]
    xLEh = parg[I.IGXHBOX] - parg[I.IGXAXIS] * parg[I.IGCOH]
    xLEv = parg[I.IGXVBOX] - parg[I.IGXAXIS] * parg[I.IGCOV]
    for label, v in (("xnose   =", parg[I.IGXNOSE]),
                     ("xend    =", parg[I.IGXEND]),
                     ("xLEw    =", xLEw), ("xLEh    =", xLEh),
                     ("xLEv    =", xLEv)):
        w(label, v * FT_M, "8.2f", "  ft ")
    out.append(_blank())
    for label, idx in (("xwing   =", I.IGXWING), ("xhtail  =", I.IGXHTAIL),
                       ("xvtail  =", I.IGXVTAIL)):
        w(label, parg[idx] * FT_M, "8.2f", "  ft ")
    out.append(_blank())
    for label, idx in (("xblend1 =", I.IGXBLEND1),
                       ("xblend2 =", I.IGXBLEND2)):
        w(label, parg[idx] * FT_M, "8.2f", "  ft ")
    out.append(_blank())
    for label, idx in (("xshell1 =", I.IGXSHELL1), ("xshell2 =", I.IGXSHELL2),
                       ("xhbend  =", I.IGXHBEND), ("xvbend  =", I.IGXVBEND),
                       ("xwbox   =", I.IGXWBOX), ("xhbox   =", I.IGXHBOX),
                       ("xvbox   =", I.IGXVBOX)):
        w(label, parg[idx] * FT_M, "8.2f", "  ft ")
    out.append(_blank())
    xLGnose = parg[I.IGXLGNOSE]
    xLGmain = parg[I.IGXCGAFT] + parg[I.IGDXLGMAIN]
    for label, v in (("xfix    =", parg[I.IGXFIX]), ("xLGnose =", xLGnose),
                     ("xLGmain =", xLGmain), ("xapu    =", parg[I.IGXAPU]),
                     ("xhpesys =", parg[I.IGXHPESYS])):
        w(label, v * FT_M, "8.2f", "  ft ")
    out.append(_blank())
    w("xeng    =", parg[I.IGXENG] * FT_M, "8.2f", "  ft ")
    w("yeng    =", parg[I.IGYENG] * FT_M, "8.2f", "  ft ")
    out.append(_blank())
    w("rpayfwd =", parg[I.IGRPAYFWD], "8.2f")
    w("rpayaft =", parg[I.IGRPAYAFT], "8.2f")
    w("xCGfwd  =", parg[I.IGXCGFWD] * FT_M, "8.2f", "  ft ")
    w("xCGaft  =", parg[I.IGXCGAFT] * FT_M, "8.2f", "  ft ")
    w("xNP     =", parg[I.IGXNP] * FT_M, "8.2f", "  ft ")
    w("mac     =", parg[I.IGCMA] * FT_M, "8.3f", "  ft ")
    w("S.M.fwd =", (parg[I.IGXNP] - parg[I.IGXCGFWD]) / parg[I.IGCMA],
      "8.3f")
    w("S.M.aft =", (parg[I.IGXNP] - parg[I.IGXCGAFT]) / parg[I.IGCMA],
      "8.3f")
    out.append(_blank())
    w("deps/da =", parg[I.IGDEPSDA], "8.3f")
    w("dCLn/da =", parg[I.IGDCLNDA], "8.3f")
    out.append(_blank())
    w("dCLh/dCL=", parg[I.IGDCLHDCL], "8.3f")
    w("dCLn/dCL=", parg[I.IGDCLNDCL], "8.3f")
    out.append(_blank())
    w("deltap  =", parg[I.IGDELTAP] * PSI_PA, "8.3f", "  psi")
    for label, idx in (("tskin   =", I.IGTSKIN), ("tfweb   =", I.IGTFWEB),
                       ("tcone   =", I.IGTCONE), ("tfloor  =", I.IGTFLOOR)):
        w(label, parg[idx] * IN_M, "8.3f", "  in  ")
    w("hfloor  =", parg[I.IGHFLOOR] * IN_M, "8.2f", "  in  ")
    for label, idx in (("Rfuse   =", I.IGRFUSE), ("dRfuse  =", I.IGDRFUSE),
                       ("wfb     =", I.IGWFB)):
        w(label, parg[idx] * IN_M, "8.3f", "  in  ")
    w("cabVol  =", parg[I.IGCABVOL] * FT_M ** 3, "8.1f", "  ft^3")
    out.append(_blank())
    w("AR      =", parg[I.IGAR], "8.4f")
    w("sweep   =", parg[I.IGSWEEP], "8.3f")
    w("lambdas =", parg[I.IGLAMBDAS], "8.4f")
    w("lambdat =", parg[I.IGLAMBDAT], "8.4f")
    for label, v in (("co      =", co), ("cs      =", cs), ("ct      =", ct),
                     ("bo      =", parg[I.IGBO]), ("bs      =", parg[I.IGBS]),
                     ("b       =", parg[I.IGB])):
        w(label, v * FT_M, "8.3f", "  ft  ")
    w("S       =", parg[I.IGS] * FT_M ** 2, "8.2f", "  ft^2")
    out.append(_blank())
    w("cstrut  =", parg[I.IGCSTRUT] * FT_M, "8.3f", "  ft  ")
    out.append(_blank())
    w("ARh     =", parg[I.IGARH], "8.4f")
    w("lambdah =", parg[I.IGLAMBDAH], "8.4f")
    w("coh     =", parg[I.IGCOH] * FT_M, "8.2f", "  ft  ")
    w("bh      =", parg[I.IGBH] * FT_M, "8.2f", "  ft  ")
    w("Sh      =", parg[I.IGSH] * FT_M ** 2, "8.2f", "  ft^2")
    w("Sh/S    =", parg[I.IGSH] / parg[I.IGS], "8.3f")
    w("Vh      =", parg[I.IGVH], "8.3f")
    w("CLhCGfwd=", parg[I.IGCLHCGFWD], "8.3f")
    out.append(_blank())
    w("ARv     =", parg[I.IGARV], "8.4f")
    w("lambdav =", parg[I.IGLAMBDAV], "8.4f")
    w("cov     =", parg[I.IGCOV] * FT_M, "8.2f", "  ft  ")
    w("bv      =", parg[I.IGBV] * FT_M, "8.2f", "  ft  ")
    w("Sv      =", parg[I.IGSV] * FT_M ** 2, "8.2f", "  ft^2")
    w("Sv/S    =", parg[I.IGSV] / parg[I.IGS], "8.3f")
    w("Vv      =", parg[I.IGVV], "8.3f")
    w("CLveout =", parg[I.IGCLVEOUT], "8.3f")
    out.append(_blank())
    w("xwing   =", parg[I.IGXWING] * FT_M, "8.2f", "  ft ")
    out.append(_blank())
    return out


def tofwrt(parg, parm, para, pare) -> list:
    """``tofwrt`` -- takeoff performance, and the noise stub."""
    out = []
    out.append(f" F_TO ={parm[I.IMFTO] * LB_N:8.1f} lb")
    out.append(f" V_1  ={parm[I.IMV1] * NMI_M / HR_S:8.2f} kt")
    out.append(f" V_2  ={parm[I.IMV2] * NMI_M / HR_S:8.2f} kt")
    out.append(f" CL2  ={para[I.IACL, I.IPCLIMB1]:8.4f}")
    out.append(f" l_1  ={parm[I.IML1] * FT_M:8.1f} ft")
    out.append(f" l_BF ={parm[I.IMLBF] * FT_M:8.1f} ft")
    out.append(f" l_TO ={parm[I.IMLTO] * FT_M:8.1f} ft")
    for label, idx in ((" tan(gam)_BF =", I.IMGAMVBF),
                       (" tan(gam)_TO =", I.IMGAMVTO),
                       (" tan(gam)_CB =", I.IMGAMVCB)):
        out.append(f"{label}{math.tan(parm[idx]) * 100.0:7.2f} %")
    out.append(_blank())
    # The literal in the source starts with a blank of its own, and
    # list-directed output adds another, so this line has two.
    out.append(_say(RULE3))
    out.append(_say("Noise..."))
    out.append(_blank())
    out.append(_say("   x [m]   z [m]     dB"))
    # The noise numbers are whatever noise.f left in parm; it is not on the
    # sizing path and this port does not run it.
    for x, z, dB, what in (
            (6500.0 - parm[I.IMLTO], 0.0, parm[I.IMDBSL], "  sideline"),
            (parm[I.IMXCB], parm[I.IMZCB], parm[I.IMDBCB], "  cutback"),
            (parm[I.IMXFO], parm[I.IMZFO], parm[I.IMDBFO], "  flyover")):
        out.append(f" {x:8.1f}{z:8.1f}{dB:9.3f}{what}")
    return out


def prfwrt(parg, parm, para, pare) -> list:
    """``prfwrt`` -- one line per mission point."""
    out = []
    out.append("           R        h        t       VTAS "
               "    Mach    CL     L/D'    W/WMTO   W/WTO "
               "    Tt4     M2     mdotf    TSFC'     gamma")
    out.append("         [nmi]     [ft]     [hr]     [kt] "
               "                                         "
               "     [K]            [kg/s]  [1/hr]     [deg]")
    rWTO = parg[I.IGWMTO] / parm[I.IMWTO]
    for ip in range(I.IPSTATIC, I.IPDESCENTN + 1):
        if ip == I.IPROTATE:
            continue
        CD = para[I.IACD, ip]
        LoD = 0.0 if CD == 0.0 else para[I.IACL, ip] / CD
        out.append(
            f" {I.CPLAB[ip - 1]}:  "
            f"{para[I.IARANGE, ip] * NMI_M:8.2f}"
            f"{para[I.IAALT, ip] * FT_M:9.1f}"
            f"{para[I.IATIME, ip] * HR_S:9.4f}"
            f"{pare[I.IEU0, ip] * NMI_M / HR_S:9.2f}"
            f"{para[I.IAMACH, ip]:9.5f}"
            f"{para[I.IACL, ip]:8.4f}"
            f"{LoD:8.3f}"
            f"{para[I.IAFRACW, ip]:9.5f}"
            f"{para[I.IAFRACW, ip] * rWTO:9.5f}"
            f"{pare[I.IETT4, ip]:8.1f}"
            f"{pare[I.IEM2, ip]:8.4f}"
            f"{pare[I.IEMCORE, ip] * pare[I.IEFF, ip] * parg[I.IGNENG]:9.4f}"
            f"{pare[I.IETSFC, ip] / HR_S:9.5f}"
            f"{para[I.IAGAMV, ip] * 180.0 / math.pi:9.3f}")
    return out


def blfwrt(bl, Mach: float, Reunit: float) -> list:
    """``blfwrt`` -- the fuselage BL station table."""
    out = []
    out.append(" " + "     s       x       r       b  "
               + "  uinv/V    ue/V     Me      Hk    Rtheta  "
               + "    Cf/2      CD    "
               + " delta*   theta "
               + "  D/rV    P/rV2   K/rV3   Q/rV   Phi/rV3   b CD    b Cf/2")
    out.append(" " + "    [ft]    [ft]    [ft]    [ft]"
               + "                                                       "
               + "  [ft]    [ft]  "
               + " [ft^2]  [ft^2]  [ft^2]  [ft^2]  [ft^2]    [ft]     [ft] ")

    gmi = GAMSL - 1.0
    lunit = FT_M
    for i in range(bl.nbl):
        ue = bl.ue[i]
        asq = 1.0 + 0.5 * gmi * Mach ** 2 * (1.0 - ue ** 2)
        Me = ue * Mach / math.sqrt(asq)
        rhbl = asq ** (1.0 / gmi)
        mubl = asq / Reunit
        Rtheta = rhbl * ue * bl.th[i] / mubl
        beff = bl.b[i] + 2.0 * math.pi * bl.ds[i]

        dsa = bl.ds[i] * beff * rhbl * ue
        tha = bl.th[i] * beff * rhbl * ue ** 2
        tsa = bl.ts[i] * beff * rhbl * ue ** 3 * 0.5
        dca = bl.dc[i] * beff * rhbl * ue
        tau = 0.5 * rhbl * ue ** 2 * bl.cf[i] * beff
        dis = rhbl * ue ** 3 * bl.cd[i] * beff

        out.append(
            " "
            + f"{bl.s[i] * lunit:8.2f}{bl.x[i] * lunit:8.2f}"
            + f"{bl.z[i] * lunit:8.2f}{beff * lunit:8.2f}"
            + f"{bl.uinv[i]:8.4f}{ue:8.4f}{Me:8.4f}{bl.hk[i]:7.3f}"
            + _gfmt(Rtheta, 11, 3)
            + f"{0.5 * bl.cf[i]:10.6f}{bl.cd[i]:10.6f}"
            + f"{bl.ds[i] * lunit:8.4f}{bl.th[i] * lunit:8.4f}"
            + f"{dsa * lunit ** 2:8.4f}{tha * lunit ** 2:8.4f}"
            + f"{tsa * lunit ** 2:8.4f}{dca * lunit ** 2:8.4f}"
            + f"{bl.ph[i] * lunit ** 2:8.4f}"
            + f"{dis * lunit:10.6f}{tau * lunit:10.6f}")
        if i == bl.iblte - 1:
            out.append(_blank())
    return out


@dataclass
class MapPoint:
    """Where the three compressors sit on their maps, at one mission point.

    Corrected speed and mass flow are each given as a *fraction of the design
    value*, which is the form the maps are written in and the form you plot.
    ``mapwrt`` writes exactly these numbers.
    """
    #: Fan: corrected mass flow / design, pressure ratio, corrected speed /
    #: design, polytropic efficiency.
    fan: tuple = ()
    lpc: tuple = ()
    hpc: tuple = ()
    OPR: float = 0.0
    #: Specific thrust, and TSFC in 1/hr -- the units ``mapwrt`` prints.
    Fsp: float = 0.0
    TSFC: float = 0.0


def map_point(pare) -> MapPoint:
    """The compressor map coordinates at one mission point.

    ``pare`` is a single column. The corrected quantities are formed the way
    ``mapwrt`` forms them, which is *not* quite the way ``tfoper`` does: the
    fan's corrected mass flow is scaled by the bypass ratio and the turbines'
    by ``1 + ff``, so each number is the flow through that component rather
    than the core flow.
    """
    Tt2, Tt19, Tt25 = pare[I.IETT2], pare[I.IETT19], pare[I.IETT25]
    pt2, pt19, pt25 = pare[I.IEPT2], pare[I.IEPT19], pare[I.IEPT25]
    BPR = pare[I.IEBPR]
    mcore = pare[I.IEMCORE]

    Nbf = pare[I.IENF] / math.sqrt(Tt2 / _TREF)
    Nblc = pare[I.IEN1] / math.sqrt(Tt19 / _TREF)
    Nbhc = pare[I.IEN2] / math.sqrt(Tt25 / _TREF)
    mbf = mcore * math.sqrt(Tt2 / _TREF) / (pt2 / _PREF) * BPR
    mblc = mcore * math.sqrt(Tt19 / _TREF) / (pt19 / _PREF)
    mbhc = mcore * math.sqrt(Tt25 / _TREF) / (pt25 / _PREF)

    pif, pilc, pihc = pare[I.IEPIF], pare[I.IEPILC], pare[I.IEPIHC]
    return MapPoint(
        fan=(mbf / pare[I.IEMBFD], pif, Nbf / pare[I.IENBFD], pare[I.IEEPF]),
        lpc=(mblc / pare[I.IEMBLCD], pilc, Nblc / pare[I.IENBLCD],
             pare[I.IEEPLC]),
        hpc=(mbhc / pare[I.IEMBHCD], pihc, Nbhc / pare[I.IENBHCD],
             pare[I.IEEPHC]),
        OPR=pilc * pihc,
        Fsp=pare[I.IEFSP],
        TSFC=pare[I.IETSFC] / HR_S)


def mapwrt(pare, ilabel: int = 0) -> list:
    """``mapwrt`` -- one row of compressor map coordinates.

    With ``ilabel`` non-zero the two-line comment header comes first, as it
    does for the first row of the file. ``chp`` is an argument in the Fortran
    but is never written, so this does not take it.
    """
    m = map_point(pare)
    out = []
    if ilabel != 0:
        out.append("#")
        out.append("#   mf/mfD    pif     Nf/NfD   epolf    "
                   "  mlc/mlcD   pilc   Nlc/NlcD  epollc   "
                   "  mhc/mhcD   pihc   Nhc/NhcD  epolhc   "
                   "    OPR      Fsp      TSFC  ")
    row = " "
    for group in (m.fan, m.lpc, m.hpc):
        row += "".join(f"{v:9.5f}" for v in group) + "   "
    row += "".join(f"{v:9.5f}" for v in (m.OPR, m.Fsp, m.TSFC))
    out.append(row)
    return out


def tfwrt(parg, para, pare) -> list:
    """The whole ``tfan_MMM.dat`` file -- the engine's track over the mission.

    Static, rotation, takeoff and cutback first, then a blank line before each
    of climb, cruise and descent, which is what separates the segments for a
    plotting program.
    """
    out = mapwrt(pare.column(I.IPSTATIC), 1)
    for ip in (I.IPROTATE, I.IPTAKEOFF, I.IPCUTBACK):
        out += mapwrt(pare.column(ip))
    for first, last in ((I.IPCLIMB1, I.IPCLIMBN),
                        (I.IPCRUISE1, I.IPCRUISEN),
                        (I.IPDESCENT1, I.IPDESCENTN)):
        out.append(_blank())
        for ip in range(first, last + 1):
            out += mapwrt(pare.column(ip))
    return out


def airwrt(chp: str, para, parg) -> list:
    """``airwrt`` -- the aerodynamic state at one mission point."""
    out = [_blank()]

    def w(label, value, d, unit=""):
        out.append(f" {chp}: {label}{value:8.{d}f}{unit}")

    cosL = math.cos(parg[I.IGSWEEP] * math.pi / 180.0)
    CD = para[I.IACD]
    LoD = 0.0 if CD == 0.0 else para[I.IACL] / CD

    S, Sh = parg[I.IGS], parg[I.IGSH]
    co, coh, cma = parg[I.IGCO], parg[I.IGCOH], parg[I.IGCMA]
    CL, CLh = para[I.IACL], para[I.IACLH]
    CLhtail = CLh * Sh / S

    a = atmos(para[I.IAALT] / 1000.0)
    u0 = a.a * para[I.IAMACH]
    q0 = 0.5 * a.rho * u0 ** 2

    # Lift split between wing, fuselage carryover and tail.
    lambdat, lambdas = parg[I.IGLAMBDAT], parg[I.IGLAMBDAS]
    gammat = lambdat * para[I.IARCLT]
    gammas = lambdas * para[I.IARCLS]
    etao, etas = parg[I.IGBO] / parg[I.IGB], parg[I.IGBS] / parg[I.IGB]
    fLo, fLt, AR = parg[I.IGFLO], parg[I.IGFLT], parg[I.IGAR]
    Kc = (etao + 0.5 * (1.0 + lambdas) * (etas - etao)
          + 0.5 * (lambdas + lambdat) * (1.0 - etas))
    Ko = 1.0 / (AR * Kc)
    Kp = (etao + 0.5 * (1.0 + gammas) * (etas - etao)
          + 0.5 * (gammas + gammat) * (1.0 - etas)
          + fLo * etao + 2.0 * fLt * Ko * gammat * lambdat)
    Kf = etao + fLo * etao
    if CL == 0.0:
        Lwingf = Lfusef = Ltailf = 0.0
    else:
        Lwingf = (1.0 - CLhtail / CL) * (1.0 - Kf / Kp)
        Lfusef = (1.0 - CLhtail / CL) * Kf / Kp
        Ltailf = CLhtail / CL

    CMwing = ((co * para[I.IACMW0]
               + (co * para[I.IACMW1] - parg[I.IGXWBOX])
               * (CL - CLh * Sh / S)) / cma)
    CMtail = ((coh * para[I.IACMH0] * Sh / S
               + (coh * para[I.IACMH1] - parg[I.IGXHBOX]) * CLh * Sh / S)
              / cma)
    CMfuse = parg[I.IGCMVF1] * (CL - parg[I.IGCLMF0]) / (S * cma)
    SM = (para[I.IAXNP] - para[I.IAXCG]) / cma

    out.append(f" {chp}: alt.   ={_ffmt(para[I.IAALT] * FT_M, 8, 0)} ft")
    w("dyn.pr.=", q0 * LB_N / FT_M ** 2, 2, " psf")
    w("VTAS   =", u0 * FT_M, 2, " ft/s")
    w("W/WMTO =", para[I.IAFRACW], 5)
    w("Mach   =", para[I.IAMACH], 4)
    w("L/D    =", LoD, 3)
    w("gamV   =", para[I.IAGAMV] * 180.0 / math.pi, 5, " deg")
    w("e      =", para[I.IASPANEFF], 4)
    w("CL     =", CL, 4)
    for label, idx in (("CD     =", I.IACD), ("CDi    =", I.IACDI),
                       ("CDfuse =", I.IACDFUSE), ("CDwing =", I.IACDWING),
                       ("CDover =", I.IACDOVER), ("CDhtail=", I.IACDHTAIL),
                       ("CDvtail=", I.IACDVTAIL), ("CDnace =", I.IACDNACE),
                       ("CDstrut=", I.IACDSTRUT)):
        w(label, para[idx], 5)
    w("Mperp  =", para[I.IAMACH] * cosL, 4)
    for label, idx in (("clpo   =", I.IACLPO), ("clps   =", I.IACLPS),
                       ("clpt   =", I.IACLPT), ("fduo   =", I.IAFDUO),
                       ("fdus   =", I.IAFDUS), ("fdut   =", I.IAFDUT)):
        w(label, para[idx], 4)
    w("cdpw   =", para[I.IACDPW], 5)
    w("cdfw   =", para[I.IACDFW], 5)
    w("cdw    =", para[I.IACDPW] + para[I.IACDFW], 5)
    w("Clh    =", CLh, 4)
    w("ClhSh/S=", CLhtail, 4)
    w("CMwing =", CMwing, 4)
    w("CMfuse =", CMfuse, 4)
    w("CMtail =", CMtail, 4)
    w("cmpo   =", para[I.IACMPO], 4)
    w("cmps   =", para[I.IACMPS], 4)
    w("cmpt   =", para[I.IACMPT], 4)
    w("dh/dR  =", math.tan(para[I.IAGAMV]), 5)
    w("xCG    =", para[I.IAXCG] * FT_M, 2, " ft")
    w("xNP    =", para[I.IAXNP] * FT_M, 2, " ft")
    w("S.M.   =", SM, 3)
    w("Wbuoy  =", para[I.IAWBUOY] * LB_N, 1, " lb")
    w("Lwing  =", Lwingf * 100.0, 2, " %")
    w("Lfuse  =", Lfusef * 100.0, 2, " %")
    w("Ltail  =", Ltailf * 100.0, 2, " %")
    return out


def engwrt(chp: str, pare) -> list:
    """``engwrt`` -- the engine state at one mission point.

    Note the station table prints ``Rt5`` in the ``R`` column for stations
    18, 19, 21 and 25 rather than each station's own ``Rt``. That is what the
    source does -- four copy-paste slips in a row -- and it is reproduced.
    """
    out = []

    out.append(_blank())

    def g(label, *vals):
        """``format(1x,a,': ',a,6g13.5)``."""
        out.append(f" {chp}: {label}" + "".join(_gfmt(v, 13, 5)
                                                for v in vals))

    def gu(label, v, unit=""):
        """``format(1x,a,': ',a,g13.5,a)``."""
        out.append(f" {chp}: {label}" + _gfmt(v, 13, 5) + unit)

    ncrowx = I.NCROWX
    epsrow = [pare[I.IEEPSC1 + k] for k in range(ncrowx)]
    Tmrow = [pare[I.IETMET1 + k] for k in range(ncrowx)]
    ncrow = 0
    for k in range(ncrowx):
        if epsrow[k] > 0.0:
            ncrow = k + 1

    def t(stn, what):
        return pare[getattr(I, f"IE{what}{stn}")]

    tot = {n: {w: t(n, w) for w in ("TT", "HT", "PT", "CPT", "RT")}
           for n in ("0", "18", "19", "2", "21", "25", "3", "4", "41",
                     "45", "49", "5", "7")}
    gam = {n: v["CPT"] / (v["CPT"] - v["RT"]) for n, v in tot.items()}

    p0, T0, u0 = pare[I.IEP0], pare[I.IET0], pare[I.IEU0]
    M0, a0, rho0 = pare[I.IEM0], pare[I.IEA0], pare[I.IERHO0]

    st = {}
    for n in ("2", "25", "5", "6", "7", "8"):
        st[n] = {w: pare[getattr(I, f"IE{w}{n}")]
                 for w in ("P", "T", "U", "R", "A", "CP")}
    u9, A9 = pare[I.IEU9], pare[I.IEA9]

    def mach(n):
        s = st[n]
        return s["U"] / math.sqrt(s["T"] * s["R"] * s["CP"]
                                  / (s["CP"] - s["R"]))

    M2, M25 = pare[I.IEM2], pare[I.IEM25]
    M5, M6, M7, M8 = (mach(n) for n in ("5", "6", "7", "8"))

    BPR = pare[I.IEBPR]
    pif, pilc, pihc = pare[I.IEPIF], pare[I.IEPILC], pare[I.IEPIHC]
    N1, N2 = pare[I.IEN1], pare[I.IEN2]
    ff = pare[I.IEFF]
    fo = pare[I.IEMOFFT] / pare[I.IEMCORE]
    fc = pare[I.IEFC]
    mcore = pare[I.IEMCORE]

    Tref, pref = _TREF, _PREF
    Nbf = pare[I.IENF] / math.sqrt(tot["2"]["TT"] / Tref)
    Nblc = N1 / math.sqrt(tot["19"]["TT"] / Tref)
    Nbhc = N2 / math.sqrt(tot["25"]["TT"] / Tref)
    Nbht = N2 / math.sqrt(tot["41"]["TT"] / Tref)
    Nblt = N1 / math.sqrt(tot["45"]["TT"] / Tref)

    def mb(n, fac):
        return (mcore * math.sqrt(tot[n]["TT"] / Tref)
                / (tot[n]["PT"] / pref) * fac)

    mbf = mb("2", BPR)
    mblc = mb("19", 1.0)
    mbhc = mb("25", 1.0 - fo)
    mbht = mb("41", 1.0 - fo + ff)
    mblt = mb("45", 1.0 - fo + ff)

    g("BPR    =", BPR)
    g("FPR    =", pif)
    g("OPR    =", pilc * pihc)
    g("pilc   =", pilc)
    g("pihc   =", pihc)
    g("1/piht =", tot["41"]["PT"] / tot["45"]["PT"])
    g("1/pilt =", tot["45"]["PT"] / tot["49"]["PT"])
    g("pid    =", pare[I.IEPID])
    g("pib    =", pare[I.IEPIB])
    g("pifn   =", pare[I.IEPIFN])
    g("pitn   =", pare[I.IEPITN])
    g("N1     =", N1)
    g("N2     =", N2)
    g("Nlc_c% =", Nblc / pare[I.IENBLCD] * 100.0)
    g("Nhc_c% =", Nbhc / pare[I.IENBHCD] * 100.0)
    g("Nht_c% =", Nbht / pare[I.IENBHTD] * 100.0)
    g("Nlt_c% =", Nblt / pare[I.IENBLTD] * 100.0)
    g("mlc_c% =", mblc / pare[I.IEMBLCD] * 100.0)
    g("mhc_c% =", mbhc / pare[I.IEMBHCD] * 100.0)
    g("mht_c% =", mbht / pare[I.IEMBHTD] * 100.0)
    g("mlt_c% =", mblt / pare[I.IEMBLTD] * 100.0)
    g("mf_c%  =", mbf / pare[I.IEMBFD] * 100.0)
    for label, idx in (("epf    =", I.IEEPF), ("eplc   =", I.IEEPLC),
                       ("ephc   =", I.IEEPHC), ("epht   =", I.IEEPHT),
                       ("eplt   =", I.IEEPLT), ("etab   =", I.IEETAB)):
        g(label, pare[idx])
    g("fo     =", fo)
    g("ffbar  =", ff / (1.0 - fo - fc))
    g("ff     =", ff)
    g("fc     =", fc)
    g("epsrow =", *epsrow[:ncrow])
    g("Tmrow  =", *Tmrow[:ncrow])

    u5, u6, u7, u8 = (st[n]["U"] for n in ("5", "6", "7", "8"))
    effp6 = 2.0 * u0 / (u0 + u6)
    effp8 = 2.0 * u0 / (u0 + u8)
    effp9 = 0.0 if u0 + u9 == 0.0 else 2.0 * u0 / (u0 + u9)

    Phiinl, Kinl = pare[I.IEPHIINL], pare[I.IEKINL]
    mdot = ff * mcore
    PK = (0.5 * ((1.0 - fo + ff) * (u6 ** 2 - u0 ** 2)
                 + BPR * (u8 ** 2 - u0 ** 2)
                 + fo * (u9 ** 2 - u0 ** 2)) * mcore + Phiinl)
    Phij = 0.5 * ((1.0 - fo + ff) * (u6 - u0) ** 2
                  + BPR * (u8 - u0) ** 2
                  + fo * (u9 - u0) ** 2) * mcore
    Pprop = PK - Phij + mdot * u0 ** 2
    etap = 0.0 if PK == 0.0 else Pprop / PK

    F6sp = ((1.0 - fo + ff) * u6 - u0) / ((1.0 + BPR) * a0)
    F8sp = BPR * (u8 - u0) / ((1.0 + BPR) * a0)
    F9sp = (fo * u9) / ((1.0 + BPR) * a0)
    Fsp = F6sp + F8sp + F9sp

    # Kerrebrock's thrust definition, as an alternative.
    FA = (((1.0 - fo + ff) * u5 - u0) * mcore
          + (st["5"]["P"] - p0) * st["5"]["A"]
          + BPR * (u7 - u0) * mcore + (st["7"]["P"] - p0) * st["7"]["A"]
          + (fo * u9) * mcore)
    Acap = 99.9999 if u0 == 0.0 else mcore * (1.0 + BPR) / (rho0 * u0)

    out.append(_blank())
    gu("Fsp    =", Fsp)
    gu("Feng   =", pare[I.IEFE] / 1.0e3, " kN")
    gu("FengA  =", FA / 1.0e3, " kN")
    gu("mcore  =", mcore, " kg/s")
    gu("mcool  =", mcore * fc, " kg/s")
    gu("mfuel  =", mcore * ff, " kg/s")
    gu("mofft  =", pare[I.IEMOFFT], " kg/s")
    gu("Pofft  =", pare[I.IEPOFFT] / 1000.0, " kW")
    gu("PK     =", PK / 1000.0, " kW")
    gu("Phijet =", Phij / 1000.0, " kW")
    gu("Phiinl =", Phiinl / 1000.0, " kW")
    gu("Kinl   =", Kinl / 1000.0, " kW")
    gu("etap   =", etap)
    gu("TSFC   =", pare[I.IETSFC] * 3600.0, " 1/hr")
    gu("hfuel  =", pare[I.IEHFUEL] / 1.0e6, " MJ/kg")

    # --- total-state table ------------------------------------------------
    out.append(_blank())
    out.append(f" {chp}:  loc     Tt      pt       cpt       R"
               "    cpt/(cpt-R)")
    out.append("   " + "           K       kPa     J/kg K   J/kg K")

    def trow(stn, label, Rcol=None):
        v = tot[stn]
        R = v["RT"] if Rcol is None else tot[Rcol]["RT"]
        out.append("      " + label
                   + f"{v['TT']:9.1f}{v['PT'] / 1000.0:9.2f}"
                   + f"{v['CPT']:9.1f}{R:9.1f}{gam[stn]:9.4f}")

    trow("0", " 0 ")
    trow("2", " 2 ")
    trow("21", " 21", Rcol="5")       # Rt5, not Rt21 -- see the docstring
    trow("7", " 7 ")
    out.append(_blank())
    trow("0", " 0 ")
    trow("18", " 18", Rcol="5")
    trow("19", " 19", Rcol="5")
    trow("25", " 25", Rcol="5")
    trow("3", " 3 ")
    trow("4", " 4 ")
    trow("41", " 41")
    trow("45", " 45")
    trow("49", " 49")
    trow("5", " 5 ")

    # --- component powers -------------------------------------------------
    def ht(n):
        return tot[n]["HT"]

    def Tt(n):
        return tot[n]["TT"]

    mdot2 = mcore * BPR
    Pfan = mdot2 * (ht("2") - ht("21"))
    Plpc = mcore * (ht("19") - ht("25"))
    Phpc = mcore * (1.0 - fo) * (ht("25") - ht("3"))
    Phpt = mcore * (1.0 - fo + ff) * (ht("41") - ht("45"))
    Plpt = mcore * (1.0 - fo + ff) * (ht("45") - ht("49"))
    cpf = (ht("21") - ht("2")) / (Tt("21") - Tt("2"))
    cplc = (ht("25") - ht("19")) / (Tt("25") - Tt("19"))
    cphc = (ht("3") - ht("25")) / (Tt("3") - Tt("25"))
    cpht = (ht("45") - ht("41")) / (Tt("45") - Tt("41"))
    cplt = (ht("49") - ht("45")) / (Tt("49") - Tt("45"))

    out.append(_blank())
    out.append(f" {chp}:  device     power    dh/dT     epol     eta")
    out.append("   " + "               kW      J/kg K                ")
    for label, P, cp, ep, eta in (
            ("fan 2 -21", Pfan, cpf, pare[I.IEEPF], pare[I.IEETAF]),
            ("LPC 19-25", Plpc, cplc, pare[I.IEEPLC], pare[I.IEETALC]),
            ("HPC 25-3 ", Phpc, cphc, pare[I.IEEPHC], pare[I.IEETAHC]),
            ("HPT 41-45", Phpt, cpht, pare[I.IEEPHT], pare[I.IEETAHT]),
            ("LPT 45-49", Plpt, cplt, pare[I.IEEPLT], pare[I.IEETALT])):
        out.append("     " + label
                   + f"{P / 1000.0:9.1f}{cp:9.1f}{ep:9.4f}{eta:9.4f}")

    # --- static-state table -----------------------------------------------
    out.append(_blank())
    out.append(f" {chp}:  loc   T       p       M       u        A "
               "    Fsp    eta_p")
    out.append("    " + "        K      kPa             m/s      m^2")

    def srow(n, label, M, extra=()):
        s = st[n]
        line = ("     " + label
                + f"{s['T']:8.1f}{s['P'] / 1000.0:8.2f}{M:8.4f}"
                + f"{s['U']:8.2f}{s['A']:8.4f}")
        for v in extra:
            line += f"{v:8.4f}"
        out.append(line)

    srow("0", " 0 ", M0) if False else out.append(
        "     " + " 0 " + f"{T0:8.1f}{p0 / 1000.0:8.2f}{M0:8.4f}"
        + f"{u0:8.2f}{Acap:8.4f}{Fsp:8.4f}")
    srow("2", " 2 ", M2)
    out.append(_blank())
    srow("25", " 25", M25)
    out.append(_blank())
    srow("5", " 5 ", M5)
    srow("6", " 6 ", M6, (F6sp, effp6))
    out.append(_blank())
    srow("7", " 7 ", M7)
    srow("8", " 8 ", M8, (F8sp, effp8))
    out.append(_blank())
    # Station 9 has no T, p or M -- the format skips those three fields.
    out.append("     " + " 9 " + " " * 24
               + f"{u9:8.2f}{A9:8.4f}{F9sp:8.4f}{effp9:8.4f}")
    return out


def _point_names():
    """The mission points the report walks, and what it calls each."""
    # The first three names are written as literals and so are not padded;
    # the rest go through a character*20 variable and are.
    pts = [(I.IPROTATE, "Rotation..."), (I.IPTAKEOFF, "Takeoff..."),
           (I.IPCUTBACK, "Cutback...")]
    for k, ip in enumerate(range(I.IPCLIMB1, I.IPCLIMBN + 1), start=1):
        pts.append((ip, f"Climb{k}...".ljust(20)))
    for k, ip in enumerate(range(I.IPCRUISE1, I.IPCRUISEN + 1), start=1):
        pts.append((ip, f"Cruise{k}...".ljust(20)))
    for k, ip in enumerate(range(I.IPDESCENT1, I.IPDESCENTN + 1), start=1):
        pts.append((ip, f"Descent{k}...".ljust(20)))
    return pts


def report(case, result, *, Lfblwrite: bool = None) -> str:
    """The whole ``.out`` report for a completed run.

    ``case`` is a :class:`~tasopt_py.tasfile.TasCase` and ``result`` a
    :class:`~tasopt_py.run.RunResult`. Returns the text, newline-terminated
    per line, as the Fortran writes it.
    """
    pari, parg = case.pari, case.parg
    if Lfblwrite is None:
        # The case file's own flag. The 737 sets it, the D8 does not, and
        # getting this from a default rather than the case is what made the
        # D8's report come out 54 lines too long.
        Lfblwrite = case.settings.Lfblwrite
    out = []

    out.append(_blank())
    out.append(RULE1)
    out.append(f" TASOPT v{VERSION:5.2f}")
    out.append(_blank())
    out.append(RULE2)
    out.append(" Config:  " + _c64(case.configname))
    out.append(_blank())
    out.append(" Case:  " + _c64(case.casename[0]))
    out.append("     :  " + _c64(case.casename[1]))
    out.append(_blank())
    # getparm.f reads one airfoil file and then duplicates it into a second
    # slot, so the reference prints the same name twice.
    name = case.airfoil_name + " "
    out.append(" Airfoil database used:  " + name)
    out.append(" Airfoil database used:  " + name)

    # outwrt is where the *fleet* PFEI gets stored, and everything else that
    # wants it reads parg afterwards -- notably pltwrt, whose PFEI column is
    # populated only because the report happens to be written first. With
    # Loutwrite off, that column would carry the "unset" fill value.
    PFEI = sum(m.parm[I.IMWOPT] * m.parm[I.IMPFEI] for m in case.missions)
    parg[I.IGPFEI] = PFEI
    out.append(_blank())
    out.append(RULE2)
    out.append(f" Fleet PFEI ={PFEI:8.4f} KJ/kg-km")

    out.append(_blank())
    out.append(RULE2)
    out.append(_say("Airframe parameters..."))
    out.append(_blank())
    out += geowrt(pari, parg)

    if Lfblwrite and result.fuselage_bl is not None:
        out.append(_blank())
        out.append(RULE2)
        out.append(_say("Fuselage BL+Wake development..."))
        out.append(_blank())
        m = case.missions[0]
        out += blfwrt(result.fuselage_bl,
                      m.para[I.IAMACH, I.IPCRUISE1],
                      m.para[I.IAREUNIT, I.IPCRUISE1])

    for km, m in enumerate(case.missions, start=1):
        out.append(_blank())
        out.append(RULE1)
        out.append(f" Fleet mission{km:4d}")
        out.append(_blank())
        out.append(f" wOpt ={m.parm[I.IMWOPT]:8.5f}")

        out.append(_blank())
        out.append(RULE2)
        ip1, ipn = I.IPCRUISE1, I.IPCRUISEN
        Wfburn = m.parm[I.IMWFUEL] / (1.0 + parg[I.IGFRESERVE])
        ffuelb = Wfburn / m.parm[I.IMWTO]
        LoD = m.para[I.IACL, ip1] / m.para[I.IACD, ip1]
        out.append(_say("Cruise performance..."))
        out.append(_blank())
        out.append(" Range =" + _ffmt(m.parm[I.IMRANGE] * NMI_M, 8, 0)
                   + " nmi")
        out.append(f" M_CR  ={m.para[I.IAMACH, ip1]:8.4f}")
        out.append(f" h_CR1 ={m.para[I.IAALT, ip1] * KFT_M:8.3f} kft")
        out.append(f" h_CR2 ={m.para[I.IAALT, ipn] * KFT_M:8.3f} kft")
        out.append(f" Wf/W  ={ffuelb:9.5f}")
        out.append(f" L/D   ={LoD:8.3f}")
        out.append(f" PFEI  ={m.parm[I.IMPFEI]:8.4f} KJ/kg-km")

        out.append(_blank())
        out.append(RULE2)
        out.append(_say("Takeoff performance..."))
        out.append(_blank())
        out += tofwrt(parg, m.parm, m.para, m.pare)

        out.append(_blank())
        out.append(RULE2)
        out.append(_say("Mission profile summary..."))
        out.append(_blank())
        out += prfwrt(parg, m.parm, m.para, m.pare)

        out.append(_blank())
        out.append(RULE2)
        out.append(_say("Aero, Engine parameters..."))
        out.append(_blank())
        out.append(RULE3)
        for ip, name in _point_names():
            out.append(f" {I.CPLAB[ip - 1]}:  {name}")
            out += airwrt(I.CPLAB[ip - 1], m.para.column(ip), parg)
            out += engwrt(I.CPLAB[ip - 1], m.pare.column(ip))
            out.append(_blank())
            out.append(RULE3)

    return "".join(line + "\n" for line in out)
