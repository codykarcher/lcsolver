"""Fuselage sizing and weight — port of TASOPT ``src/fusew.f``.

Sizes a pressurized fuselage of circular (or multi-bubble) cross-section and
returns its weight breakdown and bending stiffnesses.

The physics, in the order the routine works:

1. **Skin** thickness from hoop stress under cabin differential pressure,
   ``tskin = deltap*Rfuse/sigskin``.
2. **Shell** weight = skin plus stringers, frames and additional fittings,
   applied as the multiplier ``(1 + fstring + fframe + ffadd)``.
3. **Floor** beams sized for a landing-load case, with the bending moment
   depending on whether there is a centre support (a multi-bubble fuselage
   has one, a single tube does not).
4. **Tailcone** sized for vertical tail *torsion*, not bending.
5. **Bending material** added where the shell alone is insufficient. This is
   the subtle part — see below.

Bending material and the quadratic
----------------------------------
The added horizontal-bending material starts at station ``xhbend`` and runs
aft. That station is where the required bending area falls to zero, i.e.
where the shell's own inertia is just sufficient. Writing the required area
as a quadratic in x,

    Abar2 x^2 - Abar1 x + Abar0 = 0

``xhbend`` is its smaller root. The discriminant is floored at zero
(``desc = max(0, ...)``) so that a fuselage whose shell is everywhere
adequate does not produce a complex root; the port keeps that floor.

Multi-bubble cross-sections
---------------------------
``nfweb`` internal webs at half-width ``wfb`` give a multi-bubble section.
``nfweb = 0, wfb = 0`` is the ordinary single tube. The ``ksum`` loop
accumulates the web offsets for the vertical-axis inertia and is the only
loop in the routine.

Units are SI throughout (N, m, kg, Pa); ``gee`` is gravitational
acceleration, so the returned quantities are weights (N), not masses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

PI = 3.1415926535897932384626


@dataclass(frozen=True)
class FuselageResult:
    # thicknesses
    tskin: float
    tcone: float
    tfweb: float
    tfloor: float
    # bending-material start stations
    xhbend: float
    xvbend: float
    # stiffnesses
    EIhshell: float
    EIhbend: float
    EIvshell: float
    EIvbend: float
    GJshell: float
    GJcone: float
    # weights
    Wshell: float
    Wcone: float
    Wwindow: float
    Winsul: float
    Wfloor: float
    Whbend: float
    Wvbend: float
    Wfuse: float
    xWfuse: float
    cabVol: float


def fusew(*, gee, Nland, Wfix, Wpay, Wpadd, Wseat, Wapu, Weng,
          fstring, fframe, ffadd, deltap,
          Wpwindow, Wppinsul, Wppfloor,
          Whtail, Wvtail, rMh, rMv, Lhmax, Lvmax,
          bv, lambdav, nvtail,
          Rfuse, dRfuse, wfb, nfweb, lambdac,
          xnose, xshell1, xshell2, xconend,
          xhtail, xvtail,
          xwing, xwbox, cbox,
          xfix, xapu, xeng,
          hfloor,
          sigskin, sigbend, rhoskin, rhobend,
          Eskin, Ebend, Gskin) -> FuselageResult:
    """Size the fuselage. Arguments and meanings follow ``fusew.f`` exactly."""

    # cone and floor materials are taken as the skin/stringer materials
    taucone = sigskin
    rhocone = rhoskin
    sigfloor = sigbend
    taufloor = sigbend
    rhofloor = rhobend

    rE = Ebend / Eskin

    lnose = xshell1 - xnose
    lshell = xshell2 - xshell1
    lfloor = xshell2 - xshell1 + 2.0 * Rfuse
    xshell = 0.5 * (xshell1 + xshell2)  # noqa: F841  (kept: mirrors the Fortran)

    # ---- cross-section geometry -----------------------------------------
    wfblim = max(min(wfb, Rfuse), 0.0)
    thetafb = math.asin(wfblim / Rfuse)
    # NOTE: the Fortran uses the *unlimited* wfb here, not wfblim. For
    # wfb > Rfuse that is a sqrt of a negative number. Preserved as-is.
    hfb = math.sqrt(Rfuse**2 - wfb**2)
    sin2t = 2.0 * hfb * wfb / Rfuse**2
    cost = hfb / Rfuse

    perim = (2.0 * PI + 4.0 * thetafb) * Rfuse + 2.0 * dRfuse

    # ---- pressure-driven thicknesses ------------------------------------
    tskin = deltap * Rfuse / sigskin
    tfweb = 2.0 * deltap * wfb / sigskin

    Askin = ((2.0 * PI + 4.0 * nfweb * thetafb) * Rfuse * tskin
             + 2.0 * dRfuse * tskin)
    # Afweb here is the *shell web* area. The Fortran later reassigns the same
    # name to the floor web area; the two are unrelated, so they are kept
    # distinct here (Afweb_shell / Afweb_floor) to avoid the shadowing.
    Afweb_shell = nfweb * (2.0 * hfb + dRfuse) * tfweb
    Afuse = ((PI + nfweb * (2.0 * thetafb + sin2t)) * Rfuse**2
             + 2.0 * Rfuse * dRfuse)

    Sbulk = (2.0 * PI + 4.0 * nfweb * thetafb) * Rfuse**2
    Snose = ((2.0 * PI + 4.0 * nfweb * thetafb) * Rfuse**2
             * (0.333 + 0.667 * (lnose / Rfuse)**1.6)**0.625)

    # ---- volumes and volume moments --------------------------------------
    Vcyl = Askin * lshell
    Vnose = Snose * tskin
    Vbulk = Sbulk * tskin
    Vfweb = Afweb_shell * lshell

    xVcyl = Vcyl * 0.5 * (xshell1 + xshell2)
    xVnose = Vnose * 0.5 * (xnose + xshell1)
    xVbulk = Vbulk * (xshell2 + 0.5 * Rfuse)
    xVfweb = Vfweb * 0.5 * (xshell1 + xshell2)

    Wskin = rhoskin * gee * (Vcyl + Vnose + Vbulk)
    Wfweb = rhoskin * gee * Vfweb
    xWskin = rhoskin * gee * (xVcyl + xVnose + xVbulk)
    xWfweb = rhoskin * gee * xVfweb

    Wshell = Wskin * (1.0 + fstring + fframe + ffadd) + Wfweb
    xWshell = xWskin * (1.0 + fstring + fframe + ffadd) + xWfweb

    # ---- windows, insulation ---------------------------------------------
    Wwindow = Wpwindow * lshell
    xWwindow = Wwindow * 0.5 * (xshell1 + xshell2)

    Winsul = Wppinsul * ((1.1 * PI + 2.0 * thetafb) * Rfuse * lshell
                         + 0.55 * (Snose + Sbulk))
    xWinsul = Winsul * 0.5 * (xshell1 + xshell2)

    xWfix = Wfix * xfix
    xWapu = Wapu * xapu
    xWseat = Wseat * 0.5 * (xshell1 + xshell2)
    xWpadd = Wpadd * 0.5 * (xshell1 + xshell2)

    # ---- floor beams ------------------------------------------------------
    P = (Wpay + Wseat) * Nland
    wfloor1 = wfb + Rfuse
    if wfb == 0.0:
        # full-width floor, simply supported at the walls
        Smax = 0.50 * P
        Mmax = 0.25 * P * wfloor1
    else:
        # floor with a centre support
        Smax = (5.0 / 16.0) * P
        Mmax = (9.0 / 256.0) * P * wfloor1

    Afweb_floor = 1.5 * Smax / taufloor
    Afcap = 2.0 * Mmax / (sigfloor * hfloor)

    Vfloor = (Afcap + Afweb_floor) * 2.0 * wfloor1
    Wfloor = rhofloor * gee * Vfloor + 2.0 * wfloor1 * lfloor * Wppfloor
    xWfloor = Wfloor * 0.5 * (xshell1 + xshell2)
    tfloor = 0.5 * Afcap / lfloor

    # ---- tailcone, sized by vertical-tail torsion -------------------------
    Qv = (nvtail * Lvmax * bv / 3.0) * (1.0 + 2.0 * lambdav) / (1.0 + lambdav)
    Vcone = ((Qv / taucone)
             * (PI + nfweb * 2.0 * thetafb)
             / (PI + nfweb * (2.0 * thetafb + sin2t))
             * (xconend - xshell2) / Rfuse
             * 2.0 / (1.0 + lambdac))
    Wcone = rhocone * gee * Vcone * (1.0 + fstring + fframe + ffadd)
    xWcone = Wcone * 0.5 * (xshell2 + xconend)
    tcone = Qv / (2.0 * taucone * Afuse)

    GJshell = Gskin * 4.0 * Afuse**2 * tskin / perim
    GJcone = Gskin * 4.0 * Afuse**2 * tcone / perim

    # ---- lumped tail ------------------------------------------------------
    Wtail = Whtail + Wvtail + Wcone + Wapu + Weng
    xtail = (xhtail * Whtail + xvtail * Wvtail + xWcone
             + xapu * Wapu + xeng * Weng) / Wtail

    # ---- shell bending inertia -------------------------------------------
    tshell = tskin * (1.0 + rE * fstring * rhoskin / rhobend)
    # allowable bending stress is reduced by the pressure-induced hoop stress
    sigMh = sigbend - rE * 0.5 * deltap * Rfuse / tshell
    sigMv = sigbend - rE * 0.5 * deltap * Rfuse / tshell
    xbulk = xshell2

    Ihshell = (((PI + nfweb * (2.0 * thetafb + sin2t)) * Rfuse**2
                + 8.0 * nfweb * cost * 0.5 * dRfuse * Rfuse
                + (2.0 * PI + 4.0 * nfweb * thetafb) * (0.5 * dRfuse)**2)
               * Rfuse * tshell
               + 0.66667 * nfweb * (hfb + 0.5 * dRfuse)**3 * tfweb)

    hfuse = Rfuse + 0.5 * dRfuse
    A2 = (1.0 / (hfuse * sigMh)
          * Nland * (Wpay + Wpadd + Wshell + Wwindow + Winsul + Wfloor + Wseat)
          * 0.5 / lshell)
    A1 = 1.0 / (hfuse * sigMh) * (Nland * Wtail + rMh * Lhmax)
    A0 = -Ihshell / (rE * hfuse**2)
    Abar2 = A2
    Abar1 = 2.0 * A2 * xbulk + A1
    Abar0 = A2 * xbulk**2 + A1 * xtail + A0
    desc = max(0.0, Abar1**2 - 4.0 * Abar0 * Abar2)
    xhbend = (Abar1 - math.sqrt(desc)) * 0.5 / Abar2

    dxwing = xwing - xwbox
    xf = xwing + dxwing + 0.5 * cbox
    xb = xwing - dxwing + 0.5 * cbox

    Ahbendf = Abar2 * xf**2 - Abar1 * xf + Abar0
    Ahbendb = Abar2 * xb**2 - Abar1 * xb + Abar0

    Vhbendf = (A2 * ((xbulk - xf)**3 - (xbulk - xhbend)**3) / 3.0
               + A1 * ((xtail - xf)**2 - (xtail - xhbend)**2) / 2.0
               + A0 * (xhbend - xf))
    Vhbendb = (A2 * ((xbulk - xb)**3 - (xbulk - xhbend)**3) / 3.0
               + A1 * ((xtail - xb)**2 - (xtail - xhbend)**2) / 2.0
               + A0 * (xhbend - xb))
    Vhbendc = 0.5 * (Ahbendf + Ahbendb) * cbox

    Whbend = rhobend * gee * (Vhbendf + Vhbendb + Vhbendc)
    xWhbend = Whbend * xwing

    EIhshell = Eskin * Ihshell
    EIhbend = Ebend * 0.5 * (Ahbendf + Ahbendb) * 2.0 * hfuse**2

    # ---- vertical-axis bending -------------------------------------------
    nk = int(nfweb / 2.0 + 0.001)
    ik = (int(nfweb + 0.001) + 1) % 2
    ksum = 0.0
    for k in range(1, nk + 1):
        ksum += float(2 * k - ik)**2

    Ivshell = (((PI + nfweb * (2.0 * thetafb - sin2t)) * Rfuse**2
                + 8.0 * cost * nfweb * wfb * Rfuse
                + (2.0 * PI + 4.0 * thetafb) * (nfweb * wfb)**2
                + 4.0 * thetafb * wfb**2 * ksum) * Rfuse * tshell)

    widf = Rfuse + nfweb * wfb
    B1 = 1.0 / (widf * sigMv) * (rMv * Lvmax * nvtail)
    B0 = -Ivshell / (rE * widf**2)
    xvbend = xvtail + B0 / B1

    Avbendb = B1 * (xtail - xb) + B0
    Vvbendb = (B1 * ((xtail - xb)**2 - (xtail - xvbend)**2) / 2.0
               + B0 * (xvbend - xb))
    Vvbendc = 0.5 * Avbendb * cbox
    Wvbend = rhobend * gee * (Vvbendb + Vvbendc)
    xWvbend = Wvbend * (2.0 * xwing + xvbend) / 3.0

    EIvshell = Eskin * Ivshell
    EIvbend = Ebend * Avbendb * 2.0 * widf**2

    # ---- totals -----------------------------------------------------------
    Wfuse = (Wfix + Wapu + Wpadd + Wseat
             + Wshell + Wcone + Wwindow + Winsul + Wfloor
             + Whbend + Wvbend)
    xWfuse = (xWfix + xWapu + xWpadd + xWseat
              + xWshell + xWcone + xWwindow + xWinsul + xWfloor
              + xWhbend + xWvbend)

    cabVol = Afuse * (lshell + 0.67 * lnose + 0.67 * Rfuse)

    return FuselageResult(
        tskin=tskin, tcone=tcone, tfweb=tfweb, tfloor=tfloor,
        xhbend=xhbend, xvbend=xvbend,
        EIhshell=EIhshell, EIhbend=EIhbend,
        EIvshell=EIvshell, EIvbend=EIvbend,
        GJshell=GJshell, GJcone=GJcone,
        Wshell=Wshell, Wcone=Wcone, Wwindow=Wwindow, Winsul=Winsul,
        Wfloor=Wfloor, Whbend=Whbend, Wvbend=Wvbend,
        Wfuse=Wfuse, xWfuse=xWfuse, cabVol=cabVol)


__all__ = ["fusew", "FuselageResult"]
