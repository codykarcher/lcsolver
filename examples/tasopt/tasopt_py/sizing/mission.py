"""Fly the aircraft through its mission -- a port of ``mission.f``.

Marches the 17 mission points, integrating range, time and weight fraction,
and returns the fuel burnt. This is the routine `wsize` calls once per outer
iteration, and the one that closes the weight-fuel loop.

Structure
---------
The mission is not integrated in one sweep. It is built in three pieces, in
this order, because each needs the one before it:

1. **Climb**, from the takeoff point to the start of cruise, over
   ``ipclimb1..ipclimbn`` at equal altitude intervals. Each point solves for
   its own flight-path angle, and the trajectory is integrated with a
   predictor-corrector: forward Euler ahead to set the next point's speed,
   then trapezoidal back once that point's integrands are known.

2. **Cruise**, as a single Breguet-style interval between two points, with a
   *cruise-climb* angle derived from the fuel burn rate and the atmospheric
   pressure gradient::

       gamVcr = (D/L) p0 TSFC / (rho0 g V - p0 TSFC)

   The aircraft drifts up as it lightens. The end-of-cruise altitude is not
   an input -- it falls out of splitting the total range between the climb,
   the cruise slope and the descent slope.

3. **Descent**, at a linearly varying slope between two specified angles,
   over equal *distance* intervals (climb uses equal altitude intervals).

The flight-path angle iteration
-------------------------------
At each climb point, speed depends on the angle (through ``cos(gamma)``) and
the angle depends on thrust and drag, which depend on speed. The source
iterates to ``1e-12`` in the angle, up to 10 times, solving::

    sin(gamma) = (phi - (D/L) sqrt(1 - phi^2 + (D/L)^2)) / (1 + (D/L)^2)

with ``phi = F/W``. This is the exact solution of the force balance along and
normal to the path, not a small-angle form.

If the angle does not converge the source prints a warning and continues with
whatever it has. That is reproduced as a flag on the result.

Buoyancy
--------
Every point carries ``W_buoy``, the weight of the pressurised cabin air over
the ambient air it displaces. It is added to the weight in every force
balance (``BW = W + W_buoy``) and it is *not* small at altitude. Takeoff and
end-of-descent points set it to zero.

Reserves
--------
``Wfuel = WMTO * (fracW_climb1 - fracW_descentn) * (1 + freserve)`` -- the
reserve is a fraction of the burn, applied at the end, not flown.

Verified against the compiled Fortran; see ``tests/test_mission.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..aero.cdsum import cdsum
from ..atmosphere import atmos
from ..engine.tfcalc import tfcalc
from ..model import indices as I
from .balance import balance

__all__ = ["MissionResult", "mission", "ITERGMAX", "GAMV_TOL"]

ITERGMAX = 10
GAMV_TOL = 1.0e-12
GEE = 9.81
# Sea-level reference, as constants.inc fills them from atmos(0).
_T_SL, _P_SL, _RHO_SL = None, None, None


def _sea_level():
    global _T_SL, _P_SL, _RHO_SL
    if _T_SL is None:
        a = atmos(0.0)
        _T_SL, _P_SL, _RHO_SL = a.T, a.p, a.rho
    return _T_SL, _P_SL, _RHO_SL


@dataclass
class MissionResult:
    WTO: float          # takeoff weight for this mission
    Wfuel: float        # fuel loaded, including reserve
    PFEI: float         # payload-fuel energy intensity
    gamV_converged: bool = True


def _set_atmosphere(pare, para, parg, ip, alt, Mach=None, ground_T=None):
    """Fill the ambient state at one point, and its buoyancy weight."""
    a = atmos(alt / 1000.0)
    if ground_T is not None:
        # Takeoff points use the specified ground temperature, with the
        # standard pressure and density/viscosity scaled to it.
        T0 = ground_T
        p0 = a.p
        rho0 = a.rho * (a.T / T0)
        a0 = a.a * math.sqrt(T0 / a.T)
        mu0 = a.mu * (T0 / a.T) ** 0.8
    else:
        T0, p0, rho0, a0, mu0 = a.T, a.p, a.rho, a.a, a.mu

    pare[I.IEP0, ip] = p0
    pare[I.IET0, ip] = T0
    pare[I.IEA0, ip] = a0
    pare[I.IERHO0, ip] = rho0
    pare[I.IEMU0, ip] = mu0
    if Mach is not None:
        pare[I.IEM0, ip] = Mach
        pare[I.IEU0, ip] = Mach * a0
        para[I.IAREUNIT, ip] = Mach * a0 * rho0 / mu0
    return p0, rho0


def _buoyancy(parg, p0):
    """Weight of cabin air over the ambient air it displaces.

    ``RSL`` is not 287. ``tasopt.f`` fills ``constants.inc`` with
    ``RSL = pSL/(rhoSL*TSL)``, and TASOPT's atmosphere sets
    ``rho = gam p/((gam-1) cp T)``, so it comes out at
    ``(gam-1) cp/gam = 286.857``. Using 287 shifts the cabin density, and
    hence the buoyancy weight, by 5e-4.
    """
    T_sl, p_sl, rho_sl = _sea_level()
    R_sl = p_sl / (rho_sl * T_sl)
    rhocab = max(parg[I.IGPCABIN], p0) / (R_sl * T_sl)
    return rhocab, R_sl


def mission(pari, parg, parm, para, pare, table, initeng: int = 0,
            ipc1: int = 0) -> MissionResult:
    """Fly the mission; writes every point into ``para``/``pare``.

    ``initeng`` 0 initialises the engine state at each point, 1 assumes it is
    already there. ``ipc1`` 1 skips recomputing the start-of-cruise point.

    Climb flight-path angles are read from ``para[IAGAMV, ipclimb1:ipclimbn]``
    as starting guesses and written back; zeros are an acceptable guess.
    """
    S = parg[I.IGS]
    WMTO = parg[I.IGWMTO]
    Rangetot = parm[I.IMRANGE]
    para[I.IARANGE, I.IPDESCENTN] = Rangetot

    rpay = parm[I.IMWPAY] / parg[I.IGWPAY]
    xipay = 0.0
    Wzero = WMTO - parg[I.IGWFUEL] - parg[I.IGWPAY] + parm[I.IMWPAY]
    WTO = parm[I.IMWTO]

    T0TO = parm[I.IMT0TO]
    gamV_ok = True

    # --- known operating conditions --------------------------------------
    for ip in (I.IPSTATIC, I.IPROTATE, I.IPTAKEOFF, I.IPCUTBACK, I.IPCLIMB1):
        _set_atmosphere(pare, para, parg, ip, para[I.IAALT, I.IPSTATIC],
                        ground_T=T0TO)
        para[I.IAWBUOY, ip] = 0.0
    ip = I.IPSTATIC
    pare[I.IEM0, ip] = 0.0
    pare[I.IEU0, ip] = 0.0
    para[I.IAMACH, ip] = 0.0
    para[I.IACL, ip] = 0.0
    para[I.IAREUNIT, ip] = 0.0

    _set_atmosphere(pare, para, parg, I.IPCRUISE1,
                    para[I.IAALT, I.IPCRUISE1],
                    Mach=para[I.IAMACH, I.IPCRUISE1])
    _set_atmosphere(pare, para, parg, I.IPDESCENTN,
                    para[I.IAALT, I.IPDESCENTN], ground_T=T0TO)
    para[I.IAWBUOY, I.IPDESCENTN] = 0.0

    # --- interpolate CL over the climb ------------------------------------
    # Between the specified values at ipclimb1+1 and *ipcruise1*, quadratic
    # in the fraction so the profile leaves the first climb point flat.
    CLa = para[I.IACL, I.IPCLIMB1 + 1]
    CLb = para[I.IACL, I.IPCRUISE1]
    for ip in range(I.IPCLIMB1 + 1, I.IPCLIMBN + 1):
        frac = (float(ip - (I.IPCLIMB1 + 1))
                / float(I.IPCLIMBN - (I.IPCLIMB1 + 1)))
        para[I.IACL, ip] = CLa * (1.0 - frac ** 2) + CLb * frac ** 2

    # --- interpolate CL over the descent ----------------------------------
    # Not from the specified descent values: the source multiplies the
    # end-of-cruise CL by 0.96 at the top of descent and 0.50 at the bottom,
    # and interpolates quadratically in the *remaining* fraction. Two
    # commented-out lines above it set both ends to the cruise CL, i.e. no
    # variation, which is not what ships. Note the loop runs to
    # ipdescentn - 1: the last descent point keeps its own approach CL.
    CLd = para[I.IACL, I.IPCRUISEN] * 0.96
    CLe = para[I.IACL, I.IPCRUISEN] * 0.50
    for ip in range(I.IPDESCENT1, I.IPDESCENTN):
        frac = (float(ip - I.IPDESCENT1)
                / float((I.IPDESCENTN - 1) - I.IPDESCENT1))
        fb = 1.0 - frac
        para[I.IACL, ip] = CLd * fb ** 2 + CLe * (1.0 - fb ** 2)

    # --- takeoff speeds ---------------------------------------------------
    ip = I.IPROTATE
    cosL = math.cos(parg[I.IGSWEEP] * math.pi / 180.0)
    CLmax = para[I.IACLPMAX, ip] * cosL ** 2
    Vstall = math.sqrt(2.0 * WTO / (pare[I.IERHO0, ip] * S * CLmax))
    Mstall = Vstall / pare[I.IEA0, ip]
    pare[I.IEU0, ip] = Vstall
    pare[I.IEM0, ip] = Mstall
    para[I.IAMACH, ip] = Mstall
    para[I.IAREUNIT, ip] = Vstall * pare[I.IERHO0, ip] / pare[I.IEMU0, ip]

    # FAR-25 takeoff safety speed.
    V2, M2, CL2 = Vstall * 1.2, Mstall * 1.2, CLmax / 1.2 ** 2
    for ip in (I.IPTAKEOFF, I.IPCUTBACK, I.IPCLIMB1):
        pare[I.IEU0, ip] = V2
        pare[I.IEM0, ip] = M2
        para[I.IAMACH, ip] = M2
        para[I.IAREUNIT, ip] = V2 * pare[I.IERHO0, ip] / pare[I.IEMU0, ip]
        para[I.IACL, ip] = CL2

    # Trim at takeoff, then copy the CG/CP/NP forward to the ground points.
    ip = I.IPTAKEOFF
    rfuel = (WTO - Wzero) / parg[I.IGWFUEL]
    col = para.column(ip)
    balance(pari, parg, col, rfuel, rpay, xipay, 1)
    para.set_column(ip, col)
    CLh2 = para[I.IACLH, ip]
    xCG2, xCP2, xNP2 = (para[I.IAXCG, ip], para[I.IAXCP, ip],
                        para[I.IAXNP, ip])
    for ip in range(1, I.IPROTATE + 1):
        para[I.IAXCG, ip] = xCG2
        para[I.IAXCP, ip] = xCP2
        para[I.IAXNP, ip] = xNP2
        para[I.IACLH, ip] = 0.0
    for ip in (I.IPCUTBACK, I.IPCLIMB1):
        para[I.IAXCG, ip] = xCG2
        para[I.IAXCP, ip] = xCP2
        para[I.IAXNP, ip] = xNP2
        para[I.IACLH, ip] = CLh2

    # End of descent flies V2 corrected for the lighter weight.
    ip = I.IPDESCENTN
    Vrat = math.sqrt(para[I.IAFRACW, ip] / para[I.IAFRACW, I.IPCLIMB1])
    pare[I.IEU0, ip] = V2 * Vrat
    pare[I.IEM0, ip] = M2 * Vrat
    para[I.IAMACH, ip] = M2 * Vrat
    para[I.IAREUNIT, ip] = (V2 * Vrat * pare[I.IERHO0, ip]
                            / pare[I.IEMU0, ip])
    para[I.IACL, ip] = CL2

    # --- climb points: equal altitude intervals ---------------------------
    altb = para[I.IAALT, I.IPTAKEOFF]
    altc = para[I.IAALT, I.IPCRUISE1]
    for ip in range(I.IPCLIMB1 + 1, I.IPCLIMBN + 1):
        frac = float(ip - I.IPCLIMB1) / float(I.IPCLIMBN - I.IPCLIMB1)
        alt = altb * (1.0 - frac) + altc * frac
        para[I.IAALT, ip] = alt
        p0, rho0 = _set_atmosphere(pare, para, parg, ip, alt)
        rhocab, _ = _buoyancy(parg, p0)
        para[I.IAWBUOY, ip] = (rhocab - rho0) * GEE * parg[I.IGCABVOL]

    # Burner temperature ramps between the takeoff and cruise values.
    fT1, fTn = parg[I.IGFTT4CL1], parg[I.IGFTT4CLN]
    Tt4TO = pare[I.IETT4, I.IPTAKEOFF]
    Tt4CR = pare[I.IETT4, I.IPCRUISE1]
    for ip in range(I.IPCLIMB1, I.IPCLIMBN + 1):
        frac = float(ip - I.IPCLIMB1) / float(I.IPCLIMBN - I.IPCLIMB1)
        Tfrac = fT1 * (1.0 - frac) + fTn * frac
        pare[I.IETT4, ip] = Tt4TO * (1.0 - Tfrac) + Tt4CR * Tfrac

    para[I.IARANGE, I.IPCLIMB1] = 0.0
    para[I.IATIME, I.IPCLIMB1] = 0.0
    para[I.IAFRACW, I.IPCLIMB1] = WTO / WMTO

    FoW = {}
    FFC = {}
    Vgi = {}

    # --- climb integration -------------------------------------------------
    for ip in range(I.IPCLIMB1, I.IPCLIMBN + 1):
        W = para[I.IAFRACW, ip] * WMTO
        BW = W + para[I.IAWBUOY, ip]
        CL = para[I.IACL, ip]
        rho = pare[I.IERHO0, ip]

        for _ in range(ITERGMAX):
            cosg = math.cos(para[I.IAGAMV, ip])
            V = math.sqrt(2.0 * BW * cosg / (rho * S * CL))
            Mach = V / pare[I.IEA0, ip]
            para[I.IAMACH, ip] = Mach
            para[I.IAREUNIT, ip] = V * rho / pare[I.IEMU0, ip]
            pare[I.IEU0, ip] = V
            pare[I.IEM0, ip] = Mach

            acol = para.column(ip)
            balance(pari, parg, acol, (W - Wzero) / parg[I.IGWFUEL],
                    rpay, xipay, 1)
            ecol = pare.column(ip)
            icdfun = 0 if ip == I.IPCLIMB1 else 1
            cdsum(pari, parg, acol, ecol, icdfun, table)
            tfcalc(pari, parg, acol, ecol, ip, 1, 1, initeng)
            para.set_column(ip, acol)
            pare.set_column(ip, ecol)

            F = pare[I.IEFE, ip] * parg[I.IGNENG]
            TSFC = pare[I.IETSFC, ip]
            DoL = para[I.IACD, ip] / para[I.IACL, ip]

            # Exact solution of the force balance along and normal to the
            # flight path -- not a small-angle form.
            phi = F / BW
            sing = ((phi - DoL * math.sqrt(1.0 - phi ** 2 + DoL ** 2))
                    / (1.0 + DoL ** 2))
            cosg = math.sqrt(1.0 - sing ** 2)
            gamV = math.atan2(sing, cosg)
            dgamV = gamV - para[I.IAGAMV, ip]
            para[I.IAGAMV, ip] = gamV
            if abs(dgamV) < GAMV_TOL:
                break
        else:
            gamV_ok = False        # source prints and carries on

        cosg = math.cos(para[I.IAGAMV, ip])
        FoW[ip] = F / (BW * cosg) - DoL
        FFC[ip] = F / (W * V * cosg) * TSFC
        Vgi[ip] = 1.0 / (V * cosg)

        if ip > I.IPCLIMB1:
            # Corrector: trapezoidal, now that this point's integrands exist.
            dh = para[I.IAALT, ip] - para[I.IAALT, ip - 1]
            dVsq = pare[I.IEU0, ip] ** 2 - pare[I.IEU0, ip - 1] ** 2
            FoWavg = 0.5 * (FoW[ip] + FoW[ip - 1])
            FFCavg = 0.5 * (FFC[ip] + FFC[ip - 1])
            Vgiavg = 0.5 * (Vgi[ip] + Vgi[ip - 1])
            dR = (dh + 0.5 * dVsq / GEE) / FoWavg
            para[I.IARANGE, ip] = para[I.IARANGE, ip - 1] + dR
            para[I.IATIME, ip] = para[I.IATIME, ip - 1] + dR * Vgiavg
            para[I.IAFRACW, ip] = (para[I.IAFRACW, ip - 1]
                                   * math.exp(-dR * FFCavg))

        if ip < I.IPCLIMBN:
            # Predictor: forward Euler, to give the next point a speed.
            if para[I.IAGAMV, ip + 1] <= 0.0:
                para[I.IAGAMV, ip + 1] = para[I.IAGAMV, ip]
            Wn = para[I.IAFRACW, ip + 1] * WMTO
            BWn = Wn + para[I.IAWBUOY, ip + 1]
            cosgn = math.cos(para[I.IAGAMV, ip + 1])
            Vn = math.sqrt(2.0 * BWn * cosgn
                           / (pare[I.IERHO0, ip + 1] * S
                              * para[I.IACL, ip + 1]))
            pare[I.IEU0, ip + 1] = Vn
            dh = para[I.IAALT, ip + 1] - para[I.IAALT, ip]
            dVsq = Vn ** 2 - pare[I.IEU0, ip] ** 2
            dR = (dh + 0.5 * dVsq / GEE) / FoW[ip]
            para[I.IARANGE, ip + 1] = para[I.IARANGE, ip] + dR
            para[I.IATIME, ip + 1] = para[I.IATIME, ip] + dR * Vgi[ip]
            para[I.IAFRACW, ip + 1] = (para[I.IAFRACW, ip]
                                       * math.exp(-dR * FFC[ip]))

    # --- cruise ------------------------------------------------------------
    for idx in (I.IARANGE, I.IATIME, I.IAFRACW, I.IAWBUOY):
        para[idx, I.IPCRUISE1] = para[idx, I.IPCLIMBN]

    ip = I.IPCRUISE1
    acol = para.column(ip)
    balance(pari, parg, acol,
            (para[I.IAFRACW, ip] * WMTO - Wzero) / parg[I.IGWFUEL],
            rpay, xipay, 1)
    para.set_column(ip, acol)
    if ipc1 == 0:
        acol = para.column(ip)
        ecol = pare.column(ip)
        cdsum(pari, parg, acol, ecol, 1, table)
        DoL = acol[I.IACD] / acol[I.IACL]
        W = acol[I.IAFRACW] * WMTO
        BW = W + acol[I.IAWBUOY]
        ecol[I.IEFE] = BW * (DoL + acol[I.IAGAMV]) / parg[I.IGNENG]
        tfcalc(pari, parg, acol, ecol, ip, 2, 1, initeng)
        para.set_column(ip, acol)
        pare.set_column(ip, ecol)

    # Cruise-climb: the aircraft drifts up as it burns fuel, at the angle
    # that keeps it on the same pressure surface.
    TSFC = pare[I.IETSFC, ip]
    V = pare[I.IEU0, ip]
    p0 = pare[I.IEP0, ip]
    rho0 = pare[I.IERHO0, ip]
    DoL = para[I.IACD, ip] / para[I.IACL, ip]
    gamVcr1 = DoL * p0 * TSFC / (rho0 * GEE * V - p0 * TSFC)
    F = pare[I.IEFE, ip] * parg[I.IGNENG]
    W = para[I.IAFRACW, ip] * WMTO
    BW = W + para[I.IAWBUOY, ip]
    cosg = math.cos(gamVcr1)
    FoW[ip] = F / (BW * cosg) - DoL
    FFC[ip] = F / (W * V * cosg) * TSFC
    Vgi[ip] = 1.0 / (V * cosg)
    para[I.IAGAMV, ip] = gamVcr1

    # Split the remaining range between the cruise and descent slopes; the
    # end-of-cruise altitude falls out of that, it is not an input.
    gamVde1 = parm[I.IMGAMVDE1]
    gamVden = parm[I.IMGAMVDEN]
    gamVdeb = 0.5 * (gamVde1 + gamVden)
    alte = para[I.IAALT, I.IPDESCENTN]
    dRclimb = para[I.IARANGE, I.IPCLIMBN] - para[I.IARANGE, I.IPCLIMB1]
    dRcruise = ((alte - altc - gamVdeb * (Rangetot - dRclimb))
                / (gamVcr1 - gamVdeb))
    altd = altc + gamVcr1 * dRcruise

    ip = I.IPCRUISEN
    para[I.IAALT, ip] = altd
    p0, rho0 = _set_atmosphere(pare, para, parg, ip, altd,
                               Mach=para[I.IAMACH, ip])
    rhocab, _ = _buoyancy(parg, p0)
    para[I.IAWBUOY, ip] = (rhocab - rho0) * GEE * parg[I.IGCABVOL]

    acol = para.column(ip)
    balance(pari, parg, acol,
            (para[I.IAFRACW, ip] * WMTO - Wzero) / parg[I.IGWFUEL],
            rpay, xipay, 1)
    ecol = pare.column(ip)
    cdsum(pari, parg, acol, ecol, 1, table)

    DoL = acol[I.IACD] / acol[I.IACL]
    W = para[I.IAFRACW, ip] * WMTO
    BW = W + para[I.IAWBUOY, ip]
    # Thrust to hold the drift-up angle carried in from the previous pass.
    F = BW * (DoL + acol[I.IAGAMV])
    ecol[I.IEFE] = F / parg[I.IGNENG]
    tfcalc(pari, parg, acol, ecol, ip, 2, 1, initeng)
    para.set_column(ip, acol)
    pare.set_column(ip, ecol)

    TSFC = pare[I.IETSFC, ip]
    V = pare[I.IEU0, ip]
    p0 = pare[I.IEP0, ip]
    rho0 = pare[I.IERHO0, ip]
    gamVcr2 = DoL * p0 * TSFC / (rho0 * GEE * V - p0 * TSFC)
    cosg = math.cos(gamVcr2)
    FoW[ip] = F / (BW * cosg) - DoL
    FFC[ip] = F / (W * V * cosg) * TSFC
    Vgi[ip] = 1.0 / (V * cosg)
    para[I.IAGAMV, ip] = gamVcr2

    # One interval, constant integrand -- the same assumption as Breguet.
    ip1, ipn = I.IPCRUISE1, I.IPCRUISEN
    FoWavg = 0.5 * (FoW[ipn] + FoW[ip1])
    FFCavg = 0.5 * (FFC[ipn] + FFC[ip1])
    Vgiavg = 0.5 * (Vgi[ipn] + Vgi[ip1])
    para[I.IARANGE, ipn] = para[I.IARANGE, ip1] + dRcruise
    para[I.IATIME, ipn] = para[I.IATIME, ip1] + dRcruise * Vgiavg
    para[I.IAFRACW, ipn] = (para[I.IAFRACW, ip1]
                            * math.exp(-dRcruise * FFCavg))

    for ip in range(I.IPCRUISE1 + 1, I.IPCRUISEN):
        frac = float(ip - I.IPCRUISE1) / float(I.IPCRUISEN - I.IPCRUISE1)
        para[I.IAALT, ip] = altc * (1.0 - frac) + altd * frac
        p0, rho0 = _set_atmosphere(pare, para, parg, ip, para[I.IAALT, ip],
                                   Mach=para[I.IAMACH, ip])
        rhocab, _ = _buoyancy(parg, p0)
        para[I.IAWBUOY, ip] = (rhocab - rho0) * GEE * parg[I.IGCABVOL]

    # --- descent -----------------------------------------------------------
    ip = I.IPDESCENT1
    for idx in (I.IEP0, I.IET0, I.IEA0, I.IERHO0, I.IEMU0, I.IEM0, I.IEU0):
        pare[idx, ip] = pare[idx, I.IPCRUISEN]
    for idx in (I.IAMACH, I.IAREUNIT, I.IAALT, I.IARANGE, I.IATIME,
                I.IAFRACW, I.IAWBUOY):
        para[idx, ip] = para[idx, I.IPCRUISEN]

    Rd = para[I.IARANGE, I.IPDESCENT1]
    Re = para[I.IARANGE, I.IPDESCENTN]
    altd_ = para[I.IAALT, I.IPDESCENT1]
    para[I.IAGAMV, I.IPDESCENT1] = gamVde1
    para[I.IAGAMV, I.IPDESCENTN] = gamVden
    for ip in range(I.IPDESCENT1 + 1, I.IPDESCENTN):
        frac = float(ip - I.IPDESCENT1) / float(I.IPDESCENTN - I.IPDESCENT1)
        R = Rd * (1.0 - frac) + Re * frac
        # Integral of a linearly varying slope over the distance.
        alt = altd_ + (Re - Rd) * (gamVde1 * (frac - 0.5 * frac ** 2)
                                   + gamVden * 0.5 * frac ** 2)
        para[I.IAGAMV, ip] = gamVde1 * (1.0 - frac) + gamVden * frac
        para[I.IARANGE, ip] = R
        para[I.IAALT, ip] = alt
        p0, rho0 = _set_atmosphere(pare, para, parg, ip, alt)
        rhocab, _ = _buoyancy(parg, p0)
        para[I.IAWBUOY, ip] = (rhocab - rho0) * GEE * parg[I.IGCABVOL]
    para[I.IAWBUOY, I.IPDESCENTN] = 0.0

    for ip in range(I.IPDESCENT1, I.IPDESCENTN + 1):
        gamVde = para[I.IAGAMV, ip]
        cosg = math.cos(gamVde)
        W = para[I.IAFRACW, ip] * WMTO
        BW = W + para[I.IAWBUOY, ip]
        rho = pare[I.IERHO0, ip]
        V = math.sqrt(2.0 * BW * cosg / (rho * S * para[I.IACL, ip]))
        para[I.IAMACH, ip] = V / pare[I.IEA0, ip]
        para[I.IAREUNIT, ip] = V * rho / pare[I.IEMU0, ip]
        pare[I.IEU0, ip] = V
        pare[I.IEM0, ip] = para[I.IAMACH, ip]

        acol = para.column(ip)
        balance(pari, parg, acol, (W - Wzero) / parg[I.IGWFUEL],
                rpay, xipay, 1)
        ecol = pare.column(ip)
        icdfun = 0 if ip == I.IPDESCENTN else 1
        cdsum(pari, parg, acol, ecol, icdfun, table)

        DoL = acol[I.IACD] / acol[I.IACL]
        Fspec = BW * (math.sin(gamVde) + cosg * DoL)
        ecol[I.IEFE] = Fspec / parg[I.IGNENG]

        if initeng == 0 and ip > I.IPDESCENT1:
            # Seed the engine from the previous point, and estimate the new
            # burner temperature and turbine exit pressure from the ambient
            # change -- a cold start here is much less likely to converge.
            for idx in (I.IEMBF, I.IEMBLC, I.IEMBHC, I.IEPIF, I.IEPILC,
                        I.IEPIHC):
                ecol[idx] = pare[idx, ip - 1]
            dTburn = pare[I.IETT4, ip - 1] - pare[I.IETT3, ip - 1]
            OTR = pare[I.IETT3, ip - 1] / pare[I.IETT2, ip - 1]
            ecol[I.IETT4] = pare[I.IET0, ip] * OTR + dTburn + 50.0
            ecol[I.IEPT5] = (pare[I.IEPT5, ip - 1] * pare[I.IEP0, ip]
                             / pare[I.IEP0, ip - 1])
            inite = 1
        else:
            inite = initeng

        tfcalc(pari, parg, acol, ecol, ip, 2, 1, inite)
        para.set_column(ip, acol)
        pare.set_column(ip, ecol)

        TSFC = pare[I.IETSFC, ip]
        F = pare[I.IEFE, ip] * parg[I.IGNENG]
        FoW[ip] = F / (BW * cosg) - DoL
        FFC[ip] = F / (W * V * cosg) * TSFC
        Vgi[ip] = 1.0 / (V * cosg)

        # ...and then FFC is immediately overwritten from the fuel mass flow
        # directly. The source's comment reads "if F < 0, then TSFC is not
        # valid", but there is no `if` -- the override is unconditional. It
        # matters here because descent thrust genuinely can go negative, at
        # which point TSFC = g*mdot/F changes sign and the TSFC form of FFC is
        # meaningless. The direct form has no such problem.
        mfuel = pare[I.IEFF, ip] * pare[I.IEMCORE, ip] * parg[I.IGNENG]
        FFC[ip] = GEE * mfuel / (W * cosg * V)

        if ip > I.IPDESCENT1:
            # Corrector: trapezoidal over the interval just flown.
            FFCavg = 0.5 * (FFC[ip] + FFC[ip - 1])
            Vgiavg = 0.5 * (Vgi[ip] + Vgi[ip - 1])
            dR = para[I.IARANGE, ip] - para[I.IARANGE, ip - 1]
            para[I.IATIME, ip] = para[I.IATIME, ip - 1] + dR * Vgiavg
            para[I.IAFRACW, ip] = (para[I.IAFRACW, ip - 1]
                                   * math.exp(-dR * FFCavg))

        if ip < I.IPDESCENTN:
            # Predictor: forward Euler, using the interval BEHIND this point
            # as the estimate for the one ahead. At the first descent point
            # that interval is zero (it shares a range with the last cruise
            # point), so the first predicted step is a no-op -- the corrector
            # on the next pass supplies the real value.
            gamVn = para[I.IAGAMV, ip + 1]
            cosgn = math.cos(gamVn)
            Wn = para[I.IAFRACW, ip + 1] * WMTO
            BWn = Wn + para[I.IAWBUOY, ip + 1]
            Vn = math.sqrt(2.0 * BWn * cosgn
                           / (pare[I.IERHO0, ip + 1] * S
                              * para[I.IACL, ip + 1]))
            pare[I.IEU0, ip + 1] = Vn
            dR = para[I.IARANGE, ip] - para[I.IARANGE, ip - 1]
            para[I.IATIME, ip + 1] = para[I.IATIME, ip] + dR * Vgi[ip]
            para[I.IAFRACW, ip + 1] = (para[I.IAFRACW, ip]
                                       * math.exp(-dR * FFC[ip]))

    # --- fuel and PFEI -----------------------------------------------------
    fburn = para[I.IAFRACW, I.IPCLIMB1] - para[I.IAFRACW, I.IPDESCENTN]
    ffuel = fburn * (1.0 + parg[I.IGFRESERVE])
    Wfuel = WMTO * ffuel
    WTO = Wzero + Wfuel
    parm[I.IMWTO] = WTO
    parm[I.IMWFUEL] = Wfuel
    PFEI = (WMTO * fburn * pare[I.IEHFUEL, I.IPCRUISE1]
            / (parm[I.IMWPAY] * parm[I.IMRANGE]))
    parm[I.IMPFEI] = PFEI

    return MissionResult(WTO=WTO, Wfuel=Wfuel, PFEI=PFEI,
                         gamV_converged=gamV_ok)
