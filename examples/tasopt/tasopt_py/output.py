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
===================================  ==========================

Not covered: the ``Aero, Engine parameters...`` section, which is ``airwrt``
and ``engwrt`` writing about 124 lines for each of the 17 mission points --
some 2100 of ``737.out``'s 4565 lines. It is a flat dump of ``para``/``pare``
entries and would be mechanical to add; nothing else needs it.
:func:`report` therefore produces a *prefix* of the reference file, and
``tests/test_output.py`` diffs it against the corresponding lines.

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

from .model import indices as I

__all__ = ["report", "geowrt", "tofwrt", "prfwrt", "blfwrt", "VERSION",
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
    n = 1 if v == 0.0 else math.floor(math.log10(abs(v))) + 1
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


def report(case, result, *, Lfblwrite: bool = True) -> str:
    """The whole ``.out`` report for a completed run.

    ``case`` is a :class:`~tasopt_py.tasfile.TasCase` and ``result`` a
    :class:`~tasopt_py.run.RunResult`. Returns the text, newline-terminated
    per line, as the Fortran writes it.
    """
    pari, parg = case.pari, case.parg
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

    PFEI = sum(m.parm[I.IMWOPT] * m.parm[I.IMPFEI] for m in case.missions)
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

    return "".join(line + "\n" for line in out)
