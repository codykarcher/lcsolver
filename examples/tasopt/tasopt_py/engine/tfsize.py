"""On-design turbofan sizing — a port of ``tfsize.f``.

Given a required thrust and the cycle parameters (pressure ratios, bypass
ratio, turbine inlet temperature, component efficiencies), this sizes the
engine: core mass flow, all station states, and the flow areas at the fan
face, HPC face and both nozzles.

The calculation follows Kerrebrock, except that the constant-cp gas relations
are replaced by calls into the variable-cp gas routines
(:mod:`tasopt_py.gas.mixture`), and a turbine cooling model
(:mod:`tasopt_py.engine.cooling`) is layered on top.

Station numbering
-----------------
``0`` freestream, ``18`` diffuser exit, ``19`` core-side fan-face,
``2`` fan face, ``21`` fan exit, ``25`` LPC exit, ``3`` HPC exit,
``4`` burner exit, ``4a`` start of cooling mix, ``41`` turbine inlet after
mixing, ``45`` HPT exit, ``49`` LPT exit, ``5`` core nozzle, ``6`` core
plume, ``7`` fan nozzle, ``8`` fan plume, ``9`` offtake plume.

The outer loop
--------------
Core mass flow and thrust are coupled: thrust needs a mass flow, and the
offtake fraction ``fo = mofft/mcore`` needs the mass flow to be known. With
no inlet distortion and no offtakes, one pass suffices. Otherwise the routine
iterates up to 60 times, under-relaxing the offtake fraction at 0.8 and
converging on ``1 - mcold/mcore`` below 1e-12.

Boundary layer ingestion enters as a mass-averaged entropy defect ``Kinl``,
applied as a total-pressure loss ``pt = pt18 * exp(-sb)`` at the fan face
(``iBLIc = 0``) or at both fan and core faces (``iBLIc = 1``). The boundary
layer is taken as adiabatic, so only pressure changes -- temperature,
enthalpy and entropy at station 2 are those at 18.

Two source behaviours reproduced
--------------------------------
* ``sbcore2 = sbfan`` in the ``iBLIc = 1`` branch uses the *previous* pass's
  ``sbfan``, not the ``sbfan2`` computed two lines above. Whether that is
  intended lagging or a slip is not clear from the source; it converges
  either way, and changing it would change the answer.
* Excessive cooling flow, a negative fan plume velocity and a negative core
  plume velocity all ``stop`` the Fortran outright. Here they raise
  :class:`EngineSizingError`, which carries the same diagnostic values.

Verified against the compiled Fortran; see ``tests/test_tfsize.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..gas.mixture import (gas_burn, gas_delh, gas_mach, gas_prat, gas_tset,
                           gasfuel, gassum)
from .cooling import mcool, tmcalc
from .maps import CMAPF, CMAPHC, CMAPLC, ecmap

__all__ = ["EngineSizingError", "Station", "TFSizeResult", "tfsize"]

# Air mass fractions, from airfrac.inc: N2, O2, CO2, H2O, Ar, fuel.
# The ACTIVE line -- airfrac.inc carries three, the first two commented out,
# and the live one is last. Taking the first (0.781/0.209/0.0004/0/0.0096)
# shifts sized core mass flow by ~0.14%, which is small enough to look like a
# tolerance issue rather than the wrong constants. Same trap as tfmap.inc.
AIRFRAC = [0.7532, 0.2315, 0.0006, 0.0020, 0.0127, 0.0]
NGAS = 6
NAIR = NGAS - 1

TOLER = 1.0e-12     # fractional core mass flow convergence
RLXFO = 0.8         # offtake fraction under-relaxation
RLXS = 0.85         # BLI entropy under-relaxation


class EngineSizingError(RuntimeError):
    """Raised where the Fortran would ``stop``: choked cooling or a plume
    that cannot expand to ambient."""


@dataclass
class Station:
    """Total and static state at one station. Unset entries stay None."""
    Tt: float = None
    ht: float = None
    pt: float = None
    cpt: float = None
    Rt: float = None
    T: float = None
    u: float = None
    p: float = None
    cp: float = None
    R: float = None
    A: float = None


@dataclass
class TFSizeResult:
    TSFC: float
    Fsp: float
    hfuel: float
    ff: float          # mdot_fuel / mdot_core
    mcore: float
    stations: dict = field(default_factory=dict)
    epsrow: list = field(default_factory=list)
    Tmrow: list = field(default_factory=list)
    ncrow: int = 0
    # polytropic efficiencies actually used, and isentropic ones (informative)
    epf: float = None
    eplc: float = None
    ephc: float = None
    epht: float = None
    eplt: float = None
    etaf: float = None
    etalc: float = None
    etahc: float = None
    etaht: float = None
    etalt: float = None
    converged: bool = True


def tfsize(gee, M0, T0, p0, a0, M2, M25,
           Feng, Phiinl, Kinl, iBLIc,
           BPR, pif, pilc, pihc,
           pid, pib, pifn, pitn,
           Tt4, Ttf, ifuel, etab,
           epf0, eplc0, ephc0, epht0, eplt0,
           pifK, epfK,
           mofft, Pofft,
           Tt9, pt9,
           epsl, epsh,
           icool,
           Mtexit, dTstrk, StA, efilm, tfilm,
           M4a, ruc,
           ncrowx, epsrow=None, Tmrow=None) -> TFSizeResult:
    """Size a turbofan for a required thrust ``Feng``.

    ``Tt4`` is the burner exit total temperature and is an *input* -- the
    cycle is sized to hit it, and the fuel fraction follows.

    ``icool`` selects the cooling treatment: 0 none, 1 ``epsrow`` supplied and
    ``Tmrow`` computed from it, 2 ``Tmrow`` supplied and ``epsrow`` computed.
    """
    alpha = list(AIRFRAC)
    epsrow = list(epsrow) if epsrow is not None else [0.0] * ncrowx
    Tmrow = list(Tmrow) if Tmrow is not None else [0.0] * ncrowx

    mcore = 0.0
    fo = 0.0
    Pom = 0.0

    # Combustion-change mass fractions for this fuel, scaled by burner
    # efficiency; the shortfall goes into the fuel slot.
    gamma = gasfuel(ifuel, NGAS)
    gamma = [etab * g for g in gamma[:NAIR]] + [1.0 - etab]
    beta = [0.0] * NAIR + [1.0]

    # --- freestream ------------------------------------------------------
    st0 = gassum(alpha, NAIR, T0)
    h0, s0, cp0, R0 = st0.h, st0.s, st0.cp, st0.r
    gam0 = cp0 / (cp0 - R0)
    u0 = M0 * a0

    hspec = h0 + 0.5 * u0 ** 2
    Tguess = T0 * (1.0 + 0.5 * (gam0 - 1.0) * M0 ** 2)
    Tt0 = gas_tset(alpha, NAIR, hspec, Tguess)
    s = gassum(alpha, NAIR, Tt0)
    st0_, ht0, cpt0, Rt0 = s.s, s.h, s.cp, s.r
    pt0 = p0 * math.exp((st0_ - s0) / Rt0)
    at0 = math.sqrt(Tt0 * Rt0 * cpt0 / (cpt0 - Rt0))

    # --- offtake plume 9 ---------------------------------------------------
    Trat = (p0 / pt9) ** (Rt0 / cpt0)
    if Trat < 1.0:
        u9 = math.sqrt(2.0 * cpt0 * Tt9 * (1.0 - Trat))
        rho9 = p0 / (Rt0 * Tt0 * Trat)
    else:
        u9 = 0.0
        rho9 = p0 / (Rt0 * Tt0)

    # --- diffuser 0-18 -----------------------------------------------------
    Tt18, st18, ht18, cpt18, Rt18 = Tt0, st0_, ht0, cpt0, Rt0
    pt18 = pt0 * pid

    epht, eplt = epht0, eplt0
    pt2 = pt19 = pt18
    Tt2 = Tt19 = Tt18
    sbfan = sbcore = 0.0

    npass = 1 if (Kinl == 0.0 and mofft == 0.0 and Pofft == 0.0) else 60

    S = {}
    dmfrac = 0.0
    converged = True

    for ipass in range(1, npass + 1):
        # --- fan inlet, corrected for BLI ---------------------------------
        if ipass == 1:
            sbfan = sbcore = 0.0
        else:
            a2sq = at0 ** 2 / (1.0 + 0.5 * (gam0 - 1.0) * M2 ** 2)
            if iBLIc == 0:
                mmix = BPR * mcore * math.sqrt(Tt2 / Tt0) * pt0 / pt2
                sbfan2 = Kinl * gam0 / (mmix * a2sq)
                sbcore2 = 0.0
            else:
                mmix = (BPR * mcore * math.sqrt(Tt2 / Tt0) * pt0 / pt2
                        + mcore * math.sqrt(Tt19 / Tt0) * pt0 / pt19)
                sbfan2 = Kinl * gam0 / (mmix * a2sq)
                # NOTE: the source uses sbfan, the previous pass's value,
                # not sbfan2 computed just above. Reproduced deliberately.
                sbcore2 = sbfan
            sbfan += RLXS * (sbfan2 - sbfan)
            sbcore += RLXS * (sbcore2 - sbcore)

        # The BL is adiabatic, so only total pressure changes.
        Tt2, ht2, st2, cpt2, Rt2 = Tt18, ht18, st18, cpt18, Rt18
        pt2 = pt18 * math.exp(-sbfan)
        Tt19, ht19, st19, cpt19, Rt19 = Tt18, ht18, st18, cpt18, Rt18
        pt19 = pt18 * math.exp(-sbcore)

        # --- fan 2-21, and the fan duct nozzle ----------------------------
        epf = ecmap(pif, 1.0, pif, 1.0, CMAPF, epf0, pifK, epfK).eff
        pt21, Tt21, ht21, st21, cpt21, Rt21 = gas_prat(
            alpha, NAIR, pt2, Tt2, ht2, st2, cpt2, Rt2, pif, epf)
        pt7, Tt7, ht7, st7, cpt7, Rt7 = (pt21 * pifn, Tt21, ht21, st21,
                                         cpt21, Rt21)

        # --- LP compressor 19-25 -------------------------------------------
        eplc = ecmap(pilc, 1.0, pilc, 1.0, CMAPLC, eplc0, 1.0, 0.0).eff
        pt25, Tt25, ht25, st25, cpt25, Rt25 = gas_prat(
            alpha, NAIR, pt19, Tt19, ht19, st19, cpt19, Rt19, pilc, eplc)

        # --- HP compressor 25-3 ---------------------------------------------
        ephc = ecmap(pihc, 1.0, pihc, 1.0, CMAPHC, ephc0, 1.0, 0.0).eff
        pt3, Tt3, ht3, st3, cpt3, Rt3 = gas_prat(
            alpha, NAIR, pt25, Tt25, ht25, st25, cpt25, Rt25, pihc, ephc)

        # --- combustor 3-4 ---------------------------------------------------
        # ffb is mdot_fuel/mdot_burner; lam the product composition.
        ffb, lam = gas_burn(alpha, beta, gamma, NGAS, ifuel, Tt3, Ttf, Tt4)
        s4 = gassum(lam, NAIR, Tt4)
        st4, ht4, cpt4, Rt4 = s4.s, s4.h, s4.cp, s4.r
        pt4 = pt3 * pib

        # --- cooling mix 4 -> 41 ---------------------------------------------
        if icool == 0:
            ff = ffb * (1.0 - fo)
            pt41, Tt41, ht41, st41, cpt41, Rt41 = (pt4, Tt4, ht4, st4,
                                                   cpt4, Rt4)
            lambdap = list(lam)
            ncrow = 0
        else:
            gmi4 = Rt4 / (cpt4 - Rt4)
            Trrat = 1.0 / (1.0 + 0.5 * gmi4 * Mtexit ** 2)
            if icool == 1:
                ncrow = len([e for e in epsrow if e > 0.0]) or ncrowx
                Tmrow = tmcalc(ncrow, epsrow, Tt3, Tt4, dTstrk, Trrat,
                               efilm, tfilm, StA)
            else:
                flow = mcool(Tmrow[:ncrowx], Tt3, Tt4, dTstrk, Trrat,
                             efilm, tfilm, StA)
                ncrow, epsrow = flow.ncrow, flow.epsrow

            fc = sum((1.0 - fo) * epsrow[i] for i in range(ncrow))
            if fc >= 0.99:
                raise EngineSizingError(
                    f"excessive cooling flow: mcool/mcore = {fc}, "
                    f"Tt3 = {Tt3}, Tt4 = {Tt4}, Tmetal = {Tmrow[0]}")

            ff = (1.0 - fo - fc) * ffb

            # Speed at the start of mixing, station 4a.
            p4a, T4a, h4a, s4a, cp4a, R4a = gas_mach(
                lam, NAIR, pt4, Tt4, ht4, st4, cpt4, Rt4, 0.0, M4a, 1.0)
            u4a = math.sqrt(max(2.0 * (ht4 - h4a), 0.0))
            uc = ruc * u4a

            frac4 = (1.0 - fo - fc + ff) / (1.0 - fo + ff)
            fracm = fc / (1.0 - fo + ff)
            lambdap = [frac4 * lam[i] + fracm * alpha[i] for i in range(NAIR)]
            lambdap += [0.0] * (NGAS - NAIR)

            ht41 = frac4 * ht4 + fracm * ht3
            Tt41 = gas_tset(lambdap, NAIR, ht41, frac4 * Tt4 + fracm * Tt3)
            s41t = gassum(lambdap, NAIR, Tt41)
            st41, ht41, cpt41, Rt41 = s41t.s, s41t.h, s41t.cp, s41t.r

            # Mixed velocity from momentum at constant static pressure.
            p41 = p4a
            u41 = frac4 * u4a + fracm * uc
            h41 = ht41 - 0.5 * u41 ** 2
            T41 = gas_tset(lambdap, NAIR, h41, T4a + (h41 - h4a) / cp4a)
            s41 = gassum(lambdap, NAIR, T41)
            pt41, Tt41, ht41, st41, cpt41, Rt41 = gas_delh(
                lambdap, NAIR, p41, T41, s41.h, s41.s, s41.cp, s41.r,
                ht41 - s41.h, 1.0)

        # --- turbine work ------------------------------------------------------
        dhfac = -(1.0 - fo) / (1.0 - fo + ff) / (1.0 - epsh)
        dlfac = -1.0 / (1.0 - fo + ff) / (1.0 - epsl)
        dhht = (ht3 - ht25) * dhfac
        dhlt = (ht25 - ht19 + BPR * (ht21 - ht2) + Pom) * dlfac

        pt45, Tt45, ht45, st45, cpt45, Rt45 = gas_delh(
            lambdap, NAIR, pt41, Tt41, ht41, st41, cpt41, Rt41,
            dhht, 1.0 / epht)
        pt49, Tt49, ht49, st49, cpt49, Rt49 = gas_delh(
            lambdap, NAIR, pt45, Tt45, ht45, st45, cpt45, Rt45,
            dhlt, 1.0 / eplt)
        pt5, Tt5, ht5, st5, cpt5, Rt5 = (pt49 * pitn, Tt49, ht49, st49,
                                         cpt49, Rt49)

        # --- fan plume 7-8 -------------------------------------------------------
        pt8, ht8, Tt8, st8, cpt8, Rt8 = pt7, ht7, Tt7, st7, cpt7, Rt7
        p8, T8, h8, s8, cp8, R8 = gas_prat(
            alpha, NAIR, pt8, Tt8, ht8, st8, cpt8, Rt8, p0 / pt8, 1.0)
        if h8 >= ht8:
            raise EngineSizingError(
                f"negative fan plume velocity: pt8 = {pt8}, Tt8 = {Tt8}, "
                f"p8 = {p8}, T8 = {T8}, pif = {pif}, BPR = {BPR}")
        u8 = math.sqrt(2.0 * (ht8 - h8))
        rho8 = p8 / (R8 * T8)

        # --- core plume 5-6 ------------------------------------------------------
        pt6, ht6, Tt6, st6, cpt6, Rt6 = pt5, ht5, Tt5, st5, cpt5, Rt5
        p6, T6, h6, s6, cp6, R6 = gas_prat(
            lambdap, NAIR, pt6, Tt6, ht6, st6, cpt6, Rt6, p0 / pt6, 1.0)
        if h6 >= ht6:
            raise EngineSizingError(
                f"negative core plume velocity: pt6 = {pt6}, Tt6 = {Tt6}, "
                f"p6 = {p6}, T6 = {T6}, pif = {pif}, BPR = {BPR}")
        u6 = math.sqrt(2.0 * (ht6 - h6))
        rho6 = p6 / (R6 * T6)

        # Effective fuel heating value (informative).
        cpa = 0.5 * (cpt3 + cpt4)
        hfuel = cpa * (Tt4 - Tt3 + ffb * (Tt4 - Ttf)) / (etab * ffb)

        # --- size the core mass flow ----------------------------------------------
        mcold = mcore
        foold = fo
        Finl = 0.0 if u0 == 0.0 else Phiinl / u0
        mcore = ((Feng - Finl)
                 / ((1.0 - fo + ff) * u6 - u0 + BPR * (u8 - u0) + fo * u9))

        dfo = mofft / mcore - foold
        fo += RLXFO * dfo
        mcore *= min(2.0, 1.0 / (1.0 - dfo))
        Pom = Pofft / mcore

        Fsp = Feng / (u0 * mcore * (1.0 + BPR))
        TSFC = 0.0 if Feng <= 0.0 else (gee * ff * mcore) / Feng

        # --- nozzles: choke if the plume is supersonic -----------------------------
        M8 = u8 / math.sqrt(cp8 * R8 / (cp8 - R8) * T8)
        if M8 < 1.0:
            p7, T7, h7, s7, cp7, R7, u7 = p8, T8, h8, s8, cp8, R8, u8
        else:
            p7, T7, h7, s7, cp7, R7 = gas_mach(
                alpha, NAIR, pt7, Tt7, ht7, st7, cpt7, Rt7, 0.0, 1.0, 1.0)
            u7 = math.sqrt(2.0 * (ht7 - h7))
        rho7 = p7 / (R7 * T7)
        A7 = BPR * mcore / (rho7 * u7)
        A8 = BPR * mcore / (rho8 * u8)

        M6 = u6 / math.sqrt(cp6 * R6 / (cp6 - R6) * T6)
        if M6 < 1.0:
            p5, T5, h5, s5, cp5, R5, u5 = p6, T6, h6, s6, cp6, R6, u6
        else:
            p5, T5, h5, s5, cp5, R5 = gas_mach(
                lambdap, NAIR, pt5, Tt5, ht5, st5, cpt5, Rt5, 0.0, 1.0, 1.0)
            u5 = math.sqrt(2.0 * (ht5 - h5))
        rho5 = p5 / (R5 * T5)
        A5 = (1.0 - fo + ff) * mcore / (rho5 * u5)
        A6 = (1.0 - fo + ff) * mcore / (rho6 * u6)
        A9 = 0.0 if u9 == 0.0 else fo * mcore / (rho9 * u9)

        # --- fan and compressor face areas -------------------------------------------
        p2, T2, h2, s2_, cp2, R2 = gas_mach(
            alpha, NAIR, pt2, Tt2, ht2, st2, cpt2, Rt2, 0.0, M2, 1.0)
        u2 = math.sqrt(2.0 * (ht2 - h2))
        rho2 = p2 / (R2 * T2)
        p19, T19, h19, s19, cp19, R19 = gas_mach(
            alpha, NAIR, pt19, Tt19, ht19, st19, cpt19, Rt19, 0.0, M2, 1.0)
        u19 = math.sqrt(2.0 * (ht19 - h19))
        rho19 = p19 / (R19 * T19)
        A2 = BPR * mcore / (rho2 * u2) + mcore / (rho19 * u19)

        p25, T25, h25, s25, cp25, R25 = gas_mach(
            alpha, NAIR, pt25, Tt25, ht25, st25, cpt25, Rt25, 0.0, M25, 1.0)
        u25 = math.sqrt(2.0 * (ht25 - h25))
        rho25 = p25 / (R25 * T25)
        A25 = (1.0 - fo) * mcore / (rho25 * u25)

        if ipass >= 2:
            dmfrac = 1.0 - mcold / mcore
            if abs(dmfrac) < TOLER:
                break
    else:
        if npass > 1:
            converged = False

    # --- isentropic component efficiencies (informative only) --------------------
    _, _, ht21i, *_ = gas_prat(alpha, NAIR, pt2, Tt2, ht2, st2, cpt2, Rt2,
                               pif, 1.0)[:4]
    etaf = (ht21i - ht2) / (ht21 - ht2)
    _, _, ht25i, *_ = gas_prat(alpha, NAIR, pt19, Tt19, ht19, st19, cpt19,
                               Rt19, pilc, 1.0)[:4]
    etalc = (ht25i - ht19) / (ht25 - ht19)
    _, _, ht3i, *_ = gas_prat(alpha, NAIR, pt25, Tt25, ht25, st25, cpt25,
                              Rt25, pihc, 1.0)[:4]
    etahc = (ht3i - ht25) / (ht3 - ht25)
    _, _, ht45i, *_ = gas_prat(lambdap, NAIR, pt41, Tt41, ht41, st41, cpt41,
                               Rt41, pt45 / pt41, 1.0)[:4]
    etaht = (ht45 - ht41) / (ht45i - ht41)
    _, _, ht49i, *_ = gas_prat(lambdap, NAIR, pt45, Tt45, ht45, st45, cpt45,
                               Rt45, pt49 / pt45, 1.0)[:4]
    etalt = (ht49 - ht45) / (ht49i - ht45)

    S = {
        0: Station(Tt=Tt0, ht=ht0, pt=pt0, cpt=cpt0, Rt=Rt0, u=u0),
        # 18 is behind the inlet, 19 is the core stream after any BLI
        # entropy rise. With Kinl = 0 all three of 18, 19 and 2 coincide.
        18: Station(Tt=Tt18, ht=ht18, pt=pt18, cpt=cpt18, Rt=Rt18),
        19: Station(Tt=Tt19, ht=ht19, pt=pt19, cpt=cpt19, Rt=Rt19),
        2: Station(Tt=Tt2, ht=ht2, pt=pt2, cpt=cpt2, Rt=Rt2,
                   T=T2, u=u2, p=p2, cp=cp2, R=R2, A=A2),
        21: Station(Tt=Tt21, ht=ht21, pt=pt21, cpt=cpt21, Rt=Rt21),
        25: Station(Tt=Tt25, ht=ht25, pt=pt25, cpt=cpt25, Rt=Rt25,
                    T=T25, u=u25, p=p25, cp=cp25, R=R25, A=A25),
        3: Station(Tt=Tt3, ht=ht3, pt=pt3, cpt=cpt3, Rt=Rt3),
        4: Station(Tt=Tt4, ht=ht4, pt=pt4, cpt=cpt4, Rt=Rt4),
        41: Station(Tt=Tt41, ht=ht41, pt=pt41, cpt=cpt41, Rt=Rt41),
        45: Station(Tt=Tt45, ht=ht45, pt=pt45, cpt=cpt45, Rt=Rt45),
        49: Station(Tt=Tt49, ht=ht49, pt=pt49, cpt=cpt49, Rt=Rt49),
        5: Station(Tt=Tt5, ht=ht5, pt=pt5, cpt=cpt5, Rt=Rt5,
                   T=T5, u=u5, p=p5, cp=cp5, R=R5, A=A5),
        6: Station(T=T6, u=u6, p=p6, cp=cp6, R=R6, A=A6),
        7: Station(Tt=Tt7, ht=ht7, pt=pt7, cpt=cpt7, Rt=Rt7,
                   T=T7, u=u7, p=p7, cp=cp7, R=R7, A=A7),
        8: Station(T=T8, u=u8, p=p8, cp=cp8, R=R8, A=A8),
        9: Station(u=u9, A=A9),
    }

    return TFSizeResult(TSFC=TSFC, Fsp=Fsp, hfuel=hfuel, ff=ff, mcore=mcore,
                        stations=S, epsrow=epsrow, Tmrow=Tmrow, ncrow=ncrow,
                        epf=epf, eplc=eplc, ephc=ephc, epht=epht, eplt=eplt,
                        etaf=etaf, etalc=etalc, etahc=etahc, etaht=etaht,
                        etalt=etalt, converged=converged)
