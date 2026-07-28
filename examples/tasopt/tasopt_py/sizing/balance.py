"""Pitch trim, centre of gravity, and tail sizing -- a port of ``balance.f``.

Three routines:

``cglpay``
    The forward and aft CG limits, found by asking how much payload can be
    loaded from each end of the cabin before the CG stops moving that way.
``balance``
    Trim the aircraft at one flight condition, by one of three means, and
    report CG, centre of pressure and neutral point.
``htsize``
    Size the horizontal tail and place the wing, simultaneously, against a
    forward-CG trim requirement and a stability or cruise-trim requirement.

The trim residual
-----------------
All three share one moment equation. Writing ``cCM`` for the total pitching
moment about the origin times the reference chord,

    ``cCM = co CMw0 + (co CMw1 - xwbox)(CL - CLh Sh/S)``
    ``    + coh CMh0 Sh/S + (coh CMh1 - xhbox) CLh Sh/S``
    ``    + CMVf1 (CL - CLMf0)/S``

the aircraft is trimmed when the centre of pressure and the centre of gravity
coincide:

    ``Res = cCM/CL + xW/W = 0``

since ``xCP = -cCM/CL`` and ``xCG = xW/W``. The fuselage contributes through
``CMVf1``, a volume-derived moment slope, rather than through a lift.

``balance``'s ``itrim``
-----------------------
0 changes nothing and just reports; 1 adjusts the tail lift coefficient; 2
adjusts tail area; 3 moves the wing box. Each is a single Newton step on the
same residual, using the analytic derivative for that variable -- the moment
equation is linear in each of them, so one step is exact, and ``xCP`` and
``xCG`` come out equal.

``htsize``'s 2x2
----------------
The tail area and the wing position are solved together because each affects
both requirements. Row 1 is either a fixed tail volume coefficient
(``iHTsize == 1``) or trim at the forward CG limit with the tail at its
maximum download. Row 2 is, by ``ixwmove``: 0 keep the wing where it is,
1 place it so cruise needs exactly ``CLhspec``, 2 place it for a stability
margin ``SMmin`` at the aft CG limit.

Tail weight is rescaled with area inside the loop (``Whtail`` proportional to
``Sh``), so growing the tail pays its own weight penalty as the iteration runs
rather than only on the next outer pass.

A note on the forward/aft limits
--------------------------------
``cglpay`` assumes **zero fuel** is the worst case for both limits, stated as
a bare assumption in the source. It then solves a quadratic in payload
fraction for each end of the cabin, taking the ``-`` root at the front and the
``+`` root at the back. The quadratic arises because moving payload changes
both the moment and the weight, so CG is a ratio of two linear functions of
the payload fraction.

Verified against the compiled Fortran; see ``tests/test_balance.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..model import indices as I

__all__ = ["CGLimits", "BalanceResult", "balance", "cglpay", "htsize",
           "HTSizeError"]

ITMAX = 10
TOLER = 1.0e-7


class HTSizeError(RuntimeError):
    """The tail-area / wing-position solve did not converge."""


@dataclass(frozen=True)
class CGLimits:
    rfuelF: float    # fuel fraction at the forward limit (always 0)
    rpayF: float     # payload fraction at the forward limit
    xcgF: float      # forward CG
    rfuelB: float
    rpayB: float
    xcgB: float      # aft CG


@dataclass(frozen=True)
class BalanceResult:
    xCG: float
    xCP: float
    xNP: float


def _fixed_weights(parg):
    """The weights that do not depend on payload or fuel fraction."""
    WMTO = parg[I.IGWMTO]
    return dict(
        Wpay=parg[I.IGWPAY], Wfuel=parg[I.IGWFUEL], Wfuse=parg[I.IGWFUSE],
        Wwing=parg[I.IGWWING], Wstrut=parg[I.IGWSTRUT],
        Whtail=parg[I.IGWHTAIL], Wvtail=parg[I.IGWVTAIL],
        Weng=parg[I.IGWENG],
        Whpesys=WMTO * parg[I.IGFHPESYS],
        Wlgnose=WMTO * parg[I.IGFLGNOSE],
        Wlgmain=WMTO * parg[I.IGFLGMAIN])


def _cabin(parg):
    xcabin = 0.5 * (parg[I.IGXSHELL1] + parg[I.IGXSHELL2])
    lcabin = parg[I.IGXSHELL2] - parg[I.IGXSHELL1]
    return xcabin, lcabin


def _cCM(co, coh, CMw0, CMw1, CMh0, CMh1, CMVf1, CLMf0,
         xwbox, xhbox, S, Sh, CL, CLh):
    """The moment group, and its derivatives w.r.t. xwbox, Sh and CLh."""
    cCM = (co * CMw0 + (co * CMw1 - xwbox) * (CL - CLh * Sh / S)
           + coh * CMh0 * Sh / S + (coh * CMh1 - xhbox) * CLh * Sh / S
           + CMVf1 * (CL - CLMf0) / S)
    cCM_xwbox = -(CL - CLh * Sh / S)
    cCM_Sh = ((co * CMw1 - xwbox) * (-CLh / S)
              + coh * CMh0 / S + (coh * CMh1 - xhbox) * CLh / S)
    cCM_CLh = ((co * CMw1 - xwbox) * (-Sh / S)
               + (coh * CMh1 - xhbox) * Sh / S)
    return cCM, cCM_xwbox, cCM_Sh, cCM_CLh


def _xNP(parg, co, coh, CMw1, CMh1, CMVf1, xwbox, xhbox, S, Sh):
    """Neutral point, with the engine-inlet normal-force contribution."""
    dCLhdCL = parg[I.IGDCLHDCL]
    dCLndCL = parg[I.IGDCLNDCL]
    neng = parg[I.IGNENG]
    dfan = parg[I.IGDFAN]
    Afan = 0.25 * math.pi * dfan ** 2
    # The inlet's normal force acts a quarter fan diameter ahead of the face.
    xengcp = parg[I.IGXENG] - 0.25 * dfan

    xNP = ((co * CMw1 - xwbox) * (dCLhdCL * Sh / S - 1.0)
           - (coh * CMh1 - xhbox) * dCLhdCL * Sh / S
           - CMVf1 / S
           - neng * xengcp * dCLndCL * Afan / S)
    xNP_xwbox = -(dCLhdCL * Sh / S - 1.0)
    xNP_Sh = ((co * CMw1 - xwbox) * dCLhdCL / S
              - (coh * CMh1 - xhbox) * dCLhdCL / S)
    return xNP, xNP_xwbox, xNP_Sh


def cglpay(parg) -> CGLimits:
    """Forward and aft CG limits, from the worst-case payload arrangements.

    Zero fuel is assumed worst-case for both, as in the source.
    """
    w = _fixed_weights(parg)
    xcabin, lcabin = _cabin(parg)
    delxw = parg[I.IGXWING] - parg[I.IGXWBOX]

    rfuel = 0.0
    We = (rfuel * w["Wfuel"] + w["Wfuse"] + w["Wwing"] + w["Wstrut"]
          + w["Whtail"] + w["Wvtail"] + w["Weng"] + w["Whpesys"]
          + w["Wlgnose"] + w["Wlgmain"])
    xWe = (rfuel * (w["Wfuel"] * parg[I.IGXWBOX] + parg[I.IGDXWFUEL])
           + parg[I.IGXWFUSE]
           + w["Wwing"] * parg[I.IGXWBOX] + parg[I.IGDXWWING]
           + w["Wstrut"] * parg[I.IGXWBOX] + parg[I.IGDXWSTRUT]
           + w["Whtail"] * parg[I.IGXHBOX] + parg[I.IGDXWHTAIL]
           + w["Wvtail"] * parg[I.IGXVBOX] + parg[I.IGDXWVTAIL]
           + w["Weng"] * parg[I.IGXENG]
           + w["Whpesys"] * parg[I.IGXHPESYS]
           + w["Wlgnose"] * parg[I.IGXLGNOSE]
           + w["Wlgmain"] * (parg[I.IGXWBOX] + delxw + parg[I.IGDXLGMAIN]))

    Wpay = w["Wpay"]
    rpay = [0.0, 0.0]
    xcg = [0.0, 0.0]
    for i, (xi, sgn) in enumerate(((0.0, -1.0), (1.0, 1.0))):
        # CG is (a0 + a1 r + a2 r^2)/(b0 + b1 r); setting it stationary in r
        # gives a quadratic, whose two roots are the two limits.
        a0 = xWe
        a1 = (xcabin + (xi - 0.5) * lcabin) * Wpay
        a2 = -(xi - 0.5) * lcabin * Wpay
        b0 = We
        b1 = Wpay
        AA = a2 * b1
        BB = 2.0 * a2 * b0
        CC = a1 * b0 - a0 * b1
        rpay[i] = (-BB - sgn * math.sqrt(BB ** 2 - 4.0 * AA * CC)) * 0.5 / AA
        xpay = xcabin + (xi - 0.5) * lcabin * (1.0 - rpay[i])
        W = Wpay * rpay[i] + We
        xW = xpay * Wpay * rpay[i] + xWe
        xcg[i] = xW / W

    return CGLimits(rfuelF=0.0, rpayF=rpay[0], xcgF=xcg[0],
                    rfuelB=0.0, rpayB=rpay[1], xcgB=xcg[1])


def balance(pari, parg, para, rfuel: float, rpay: float, xipay: float,
            itrim: int) -> BalanceResult:
    """Trim at one condition; writes ``xCG``, ``xCP``, ``xNP`` into ``para``.

    ``xipay`` places a partial payload: 0 all the way forward in the cabin,
    0.5 centred, 1 all the way aft.
    """
    w = _fixed_weights(parg)
    xcabin, lcabin = _cabin(parg)
    xpay = xcabin + (xipay - 0.5) * lcabin * (1.0 - rpay)
    xwbox = parg[I.IGXWBOX]

    lim = cglpay(parg)
    dxwing = parg[I.IGXWING] - parg[I.IGXWBOX]
    # The main gear sits a fixed distance behind the aft CG limit.
    dxlg = lim.xcgB + parg[I.IGDXLGMAIN] - parg[I.IGXWBOX]

    S, Sh = parg[I.IGS], parg[I.IGSH]
    co, coh = parg[I.IGCO], parg[I.IGCOH]
    xhbox = parg[I.IGXHBOX]

    # Tail weight is taken proportional to area about the current value, so
    # the Newton step on Sh carries its own weight change.
    Sh1 = Sh

    W = (rpay * w["Wpay"] + rfuel * w["Wfuel"] + w["Wfuse"] + w["Wwing"]
         + w["Wstrut"] + w["Whtail"] * Sh / Sh1 + w["Wvtail"] + w["Weng"]
         + w["Whpesys"] + w["Wlgnose"] + w["Wlgmain"])
    W_Sh = w["Whtail"] / Sh1

    xW = (rpay * w["Wpay"] * xpay
          + rfuel * (w["Wfuel"] * xwbox + parg[I.IGDXWFUEL])
          + parg[I.IGXWFUSE]
          + w["Wwing"] * xwbox + parg[I.IGDXWWING]
          + w["Wstrut"] * xwbox + parg[I.IGDXWSTRUT]
          + (w["Whtail"] * xhbox + parg[I.IGDXWHTAIL]) * Sh / Sh1
          + w["Wvtail"] * parg[I.IGXVBOX] + parg[I.IGDXWVTAIL]
          + w["Weng"] * parg[I.IGXENG]
          + w["Whpesys"] * parg[I.IGXHPESYS]
          + w["Wlgnose"] * parg[I.IGXLGNOSE]
          + w["Wlgmain"] * (xwbox + dxlg))
    xW_xwbox = rfuel * w["Wfuel"] + w["Wwing"] + w["Wstrut"] + w["Wlgmain"]
    xW_Sh = (w["Whtail"] * xhbox + parg[I.IGDXWHTAIL]) / Sh1

    CMw0, CMw1 = para[I.IACMW0], para[I.IACMW1]
    CMh0, CMh1 = para[I.IACMH0], para[I.IACMH1]
    CL, CLh = para[I.IACL], para[I.IACLH]
    CMVf1, CLMf0 = parg[I.IGCMVF1], parg[I.IGCLMF0]

    cCM, cCM_xwbox, cCM_Sh, cCM_CLh = _cCM(
        co, coh, CMw0, CMw1, CMh0, CMh1, CMVf1, CLMf0,
        xwbox, xhbox, S, Sh, CL, CLh)

    Res = cCM / CL + xW / W
    Res_CLh = cCM_CLh / CL
    Res_Sh = cCM_Sh / CL + xW_Sh / W - (xW / W ** 2) * W_Sh
    Res_xwbox = cCM_xwbox / CL + xW_xwbox / W

    # One Newton step -- exact, since the residual is linear in each variable.
    if itrim == 1:
        delCLh = -Res / Res_CLh
        CLh += delCLh
        cCM += cCM_CLh * delCLh
        para[I.IACLH] = CLh
    elif itrim == 2:
        delSh = -Res / Res_Sh
        Sh += delSh
        cCM += cCM_Sh * delSh
        xW += xW_Sh * delSh
        # NOTE: the source updates the weight *moment* for the tail area
        # change but not the weight itself -- W is left at its pre-step
        # value even though W_Sh was computed. So with itrim = 2 the
        # reported xCG is xW_new/W_old and does not exactly equal xCP,
        # unlike the other two trim modes. Reproduced deliberately.
        parg[I.IGSH] = Sh
    elif itrim == 3:
        delxwbox = -Res / Res_xwbox
        xwbox += delxwbox
        cCM += cCM_xwbox * delxwbox
        xW += xW_xwbox * delxwbox
        parg[I.IGXWBOX] = xwbox
        parg[I.IGXWING] = xwbox + dxwing

    xNP, _, _ = _xNP(parg, co, coh, CMw1, CMh1, CMVf1, xwbox, xhbox, S, Sh)

    xCP = -cCM / CL
    xCG = xW / W
    para[I.IAXCG] = xCG
    para[I.IAXCP] = xCP
    para[I.IAXNP] = xNP
    return BalanceResult(xCG=xCG, xCP=xCP, xNP=xNP)


def htsize(pari, parg, paraF, paraB, paraC) -> None:
    """Size the horizontal tail and place the wing together.

    ``paraF``/``paraB``/``paraC`` are the aero states at the forward-CG, aft-CG
    and cruise conditions. Writes ``Sh``, ``xwbox``, ``xwing``, the CG limits,
    and -- for fixed tail volume -- the ``CLh`` that forward-CG trim needs.
    """
    cosL = math.cos(parg[I.IGSWEEP] * math.pi / 180.0)

    lim = cglpay(parg)
    parg[I.IGXCGFWD] = lim.xcgF
    parg[I.IGXCGAFT] = lim.xcgB
    rpayC = 1.0

    CLF = paraF[I.IACLPMAX] * cosL ** 2
    CLhF = parg[I.IGCLHCGFWD]
    CLC = paraF[I.IACL]
    CLhC = parg[I.IGCLHSPEC]
    SM = parg[I.IGSMMIN]
    cma = parg[I.IGCMA]

    iHTsize = pari[I.IIHTSIZE]
    ixwmove = pari[I.IIXWMOVE]

    w = _fixed_weights(parg)
    xcabin, lcabin = _cabin(parg)
    xpayF = xcabin + (0.0 - 0.5) * lcabin * (1.0 - lim.rpayF)
    xpayB = xcabin + (1.0 - 0.5) * lcabin * (1.0 - lim.rpayB)
    xpayC = xcabin

    xwbox = parg[I.IGXWBOX]
    dxwing = parg[I.IGXWING] - parg[I.IGXWBOX]
    dxlg = parg[I.IGXCGAFT] + parg[I.IGDXLGMAIN] - parg[I.IGXWBOX]

    S, Sh = parg[I.IGS], parg[I.IGSH]
    co, coh = parg[I.IGCO], parg[I.IGCOH]
    xhbox, xvbox = parg[I.IGXHBOX], parg[I.IGXVBOX]
    CMVf1, CLMf0 = parg[I.IGCMVF1], parg[I.IGCLMF0]

    # Cruise fuel is whatever is left over from the cruise weight fraction.
    WfuelC = (paraC[I.IAFRACW] * parg[I.IGWMTO] - rpayC * w["Wpay"]
              - w["Wfuse"] - w["Wwing"] - w["Wstrut"] - w["Whtail"]
              - w["Wvtail"] - w["Weng"] - w["Whpesys"] - w["Wlgnose"]
              - w["Wlgmain"])
    rfuelC = WfuelC / w["Wfuel"]

    Sh_o = Sh
    Whtail_o = w["Whtail"]
    dxWhtail_o = parg[I.IGDXWHTAIL]

    dSh = dxw = 0.0
    converged = False
    for _ in range(ITMAX):
        Whtail = (Whtail_o / Sh_o) * Sh
        Whtail_Sh = Whtail_o / Sh_o
        dxWhtail = (dxWhtail_o / Sh_o) * Sh
        dxWhtail_Sh = dxWhtail_o / Sh_o

        We = (w["Wfuse"] + w["Wwing"] + w["Wstrut"] + Whtail + w["Wvtail"]
              + w["Weng"] + w["Whpesys"] + w["Wlgnose"] + w["Wlgmain"])
        We_Sh = Whtail_Sh
        xWe = (parg[I.IGXWFUSE]
               + w["Wwing"] * xwbox + parg[I.IGDXWWING]
               + w["Wstrut"] * xwbox + parg[I.IGDXWSTRUT]
               + Whtail * xhbox + dxWhtail
               + w["Wvtail"] * xvbox + parg[I.IGDXWVTAIL]
               + w["Weng"] * parg[I.IGXENG]
               + w["Whpesys"] * parg[I.IGXHPESYS]
               + w["Wlgnose"] * parg[I.IGXLGNOSE]
               + w["Wlgmain"] * (xwbox + dxlg))
        xWe_Sh = Whtail_Sh * xhbox + dxWhtail_Sh
        xWe_xw = w["Wwing"] + w["Wstrut"] + w["Wlgmain"]

        Wfuel, Wpay = w["Wfuel"], w["Wpay"]
        WF = We + lim.rfuelF * Wfuel + lim.rpayF * Wpay
        WB = We + lim.rfuelB * Wfuel + lim.rpayB * Wpay
        WC = We + rfuelC * Wfuel + rpayC * Wpay
        WF_Sh = WB_Sh = WC_Sh = We_Sh

        xWF = xWe + lim.rpayF * Wpay * xpayF + lim.rfuelF * (
            Wfuel * xwbox + parg[I.IGDXWFUEL])
        xWB = xWe + lim.rpayB * Wpay * xpayB + lim.rfuelB * (
            Wfuel * xwbox + parg[I.IGDXWFUEL])
        xWC = xWe + rpayC * Wpay * xpayC + rfuelC * (
            Wfuel * xwbox + parg[I.IGDXWFUEL])
        xWF_xw = xWe_xw + lim.rfuelF * Wfuel
        xWB_xw = xWe_xw + lim.rfuelB * Wfuel
        xWC_xw = xWe_xw + rfuelC * Wfuel
        xWF_Sh = xWB_Sh = xWC_Sh = xWe_Sh

        a = [[0.0, 0.0], [0.0, 0.0]]
        r = [0.0, 0.0]

        if iHTsize == 1:
            # Fixed tail volume coefficient.
            lhtail = parg[I.IGXHTAIL] - (xwbox + dxwing)
            lhtail_xw = -1.0
            Vh = parg[I.IGVH]
            Sh = Vh * S * cma / lhtail
            r[0] = Sh * lhtail - Vh * S * cma
            a[0][0] = lhtail
            a[0][1] = Sh * lhtail_xw
        else:
            # Trim at the forward CG limit with the tail at max download.
            cCM, cCM_xw, cCM_Sh, _ = _cCM(
                co, coh, paraF[I.IACMW0], paraF[I.IACMW1], paraF[I.IACMH0],
                paraF[I.IACMH1], CMVf1, CLMf0, xwbox, xhbox, S, Sh, CLF, CLhF)
            r[0] = cCM / CLF + xWF / WF
            a[0][0] = cCM_Sh / CLF + xWF_Sh / WF - (xWF / WF ** 2) * WF_Sh
            a[0][1] = cCM_xw / CLF + xWF_xw / WF

        if ixwmove == 0:
            # Wing stays put.
            r[1] = 0.0
            a[1][0] = 0.0
            a[1][1] = 1.0
        elif ixwmove == 1:
            # Place the wing so cruise trims at exactly CLhspec.
            cCM, cCM_xw, cCM_Sh, _ = _cCM(
                co, coh, paraC[I.IACMW0], paraC[I.IACMW1], paraC[I.IACMH0],
                paraC[I.IACMH1], CMVf1, CLMf0, xwbox, xhbox, S, Sh, CLC, CLhC)
            r[1] = cCM / CLC + xWC / WC
            a[1][0] = cCM_Sh / CLC + xWC_Sh / WC - (xWC / WC ** 2) * WC_Sh
            a[1][1] = cCM_xw / CLC + xWC_xw / WC
        elif ixwmove == 2:
            # Place the wing for a stability margin at the aft CG limit.
            xNP, xNP_xw, xNP_Sh = _xNP(
                parg, co, coh, paraB[I.IACMW1], paraB[I.IACMH1], CMVf1,
                xwbox, xhbox, S, Sh)
            r[1] = xWB / WB - xNP + SM * cma
            a[1][0] = xWB_Sh / WB - xNP_Sh - (xWB / WB ** 2) * WB_Sh
            a[1][1] = xWB_xw / WB - xNP_xw

        det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
        d1 = (r[0] * a[1][1] - a[0][1] * r[1]) / det
        d2 = (a[0][0] * r[1] - r[0] * a[1][0]) / det
        dSh, dxw = -d1, -d2
        Sh += dSh
        xwbox += dxw

        # Both changes are measured against a natural scale: the mean chord
        # for position, a fifth of the wing area for the tail.
        if max(abs(dxw) / cma, abs(dSh) / (0.2 * S)) < TOLER:
            converged = True
            break

    if not converged:
        raise HTSizeError(
            f"htsize: pitch not converged. dxwbox = {dxw!r}, dSh = {dSh!r}")

    parg[I.IGSH] = Sh
    parg[I.IGXWBOX] = xwbox
    parg[I.IGXWING] = xwbox + dxwing

    if iHTsize == 1:
        # With the area fixed by volume, back out the CLh forward-CG trim
        # needs. Note WF and xWF are NOT recomputed here: the source reuses
        # whatever the last loop iteration left in them, so they correspond
        # to the tail area from *before* the final Newton update. The moment
        # group below, by contrast, is recomputed with the converged Sh and
        # xwbox. Reproduced as written -- recomputing both moves CLhCGfwd by
        # about 1e-8, which is the loop's own tolerance.
        cCM, _, _, cCM_CLh = _cCM(
            co, coh, paraF[I.IACMW0], paraF[I.IACMW1], paraF[I.IACMH0],
            paraF[I.IACMH1], CMVf1, CLMf0, xwbox, xhbox, S, Sh, CLF, CLhF)
        res = cCM / CLF + xWF / WF
        parg[I.IGCLHCGFWD] = CLhF - res / (cCM_CLh / CLF)
