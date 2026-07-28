"""Off-design turbofan operation -- a port of ``tfoper.f``.

Where :func:`tasopt_py.engine.tfsize.tfsize` picks the flow areas to deliver a
required thrust, this routine takes those areas as fixed hardware and asks
what the engine actually *does* at some other flight condition. That inverts
the problem: the pressure ratios and mass flows are no longer inputs, they are
what has to be solved for.

The nine unknowns
-----------------
=========  ===================================================
``pf``     fan pressure ratio
``pl``     LP compressor pressure ratio
``ph``     HP compressor pressure ratio
``mf``     fan corrected mass flow
``ml``     LP compressor corrected mass flow
``mh``     HP compressor corrected mass flow
``Tb``     burner exit total temperature
``Pc``     turbine exit total pressure (station 5)
``Mi``     fan-face Mach number
=========  ===================================================

and the nine constraints that close them:

1. fan and LPC turn at the same speed through the gearbox
2. LPT inlet corrected mass flow is fixed (choked IGV, vertical-line map)
3. HPT inlet corrected mass flow is fixed
4. fan nozzle passes the fan flow through ``A7``
5. core nozzle passes the core flow through ``A5``
6. LPC flow equals HPC flow plus the mass offtake
7. either ``Tt4`` is the specified value, or thrust is (see ``iTFspec``)
8. LPT exit pressure from the pressure path equals the one from the work path
9. fan and core streams together fill the fan-face area ``A2``

The Jacobian
------------
The Fortran carries a hand-differentiated 9x9 Jacobian, and roughly two
thirds of its 3400 lines are that bookkeeping -- every ``x_y`` variable in the
source is one entry of one chain rule. This port computes the same Jacobian by
central differences instead.

That is a deliberate departure, and it is worth being precise about what it
costs. Newton's method converges to the same root regardless of how the
Jacobian is obtained; the Jacobian sets only the *rate*. So the converged
answer is unaffected in principle, and in practice this port agrees with the
Fortran to ~1e-10 rather than the ~1e-15 the closed-form modules reach. The
gap is not error in the physics -- it is that the two runs stop at slightly
different points inside the same convergence ball, since the Fortran's
termination test is on the size of the Newton step, and the steps differ.

What is gained is that the physics is stated once, in about 300 lines, where
the source states it interleaved with its own derivatives across 3400. Every
constant, clamp and branch below is still the source's.

Convergence and clamps
----------------------
The step limits (``dTb <= 0.5(Tb - Tt3)``, ``dPc <= (Pc - p0)``, and so on)
are not cosmetic: several of the gas-path quantities are undefined outside
them -- a burner temperature below compressor exit, a turbine exit pressure
below ambient -- and an unlimited Newton step reaches them easily from a cold
start. Efficiencies are floored (fan 0.60, compressors 0.70, turbines 0.80)
because the maps are quadratics that go negative far enough off design.

``Mi`` is capped at 0.98. If a Newton step would push the fan face past it,
the source discards row 9 of the system, replaces it with ``Mi = Mimax``, and
re-solves -- treating choking as a constraint that becomes active rather than
letting the iteration walk through it. That is reproduced here.

Verified against the compiled Fortran; see ``tests/test_tfoper.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..gas.mixture import (gas_burn, gas_delh, gas_mach, gas_mass, gas_prat,
                           gas_tset, gasfuel, gassum)
from .cooling import mcool, tmcalc
from .maps import (CMAPF, CMAPHC, CMAPLC, TMAPH, TMAPL, Ncmap, ecmap, etmap)
from .tfsize import AIRFRAC, NAIR, NGAS, Station

__all__ = ["TFOperResult", "TFOperError", "tfoper"]

TOLER = 1.0e-10      # max relative Newton change, as in the source
ITMAX = 50
EPFMIN = 0.60        # floor on fan polytropic efficiency
MIMAX = 0.98         # fan-face Mach above which it is held artificially

# Central-difference step, relative to each variable. Large enough that the
# inner Newton solves' own 1e-6 K tolerance does not dominate the difference,
# small enough that truncation stays far below it.
FD_STEP = 1.0e-6


class TFOperError(RuntimeError):
    """The Newton iteration failed, or the gas path became unphysical."""


@dataclass
class TFOperResult:
    TSFC: float
    Fsp: float
    hfuel: float
    ff: float            # mdot_fuel / mdot_core
    Feng: float
    mcore: float
    BPR: float
    pif: float
    pilc: float
    pihc: float
    mbf: float
    mblc: float
    mbhc: float
    Nbf: float
    Nblc: float
    Nbhc: float
    stations: dict = field(default_factory=dict)
    epsrow: list = field(default_factory=list)
    Tmrow: list = field(default_factory=list)
    ncrow: int = 0
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
    iterations: int = 0
    converged: bool = True


@dataclass
class _Cycle:
    """Everything one pass through the gas path produces."""
    res: list
    st: dict
    scalars: dict
    lambdap: list


def _solve(a, b):
    """Gaussian elimination with partial pivoting -- stands in for ``gaussn``.

    Returns the solution in place of *b*. The pivoting differs from the
    Fortran's, which matters only for round-off, not for the root.
    """
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(m[i][k]))
        if abs(m[p][k]) < 1e-300:
            raise TFOperError("singular Newton system in tfoper")
        m[k], m[p] = m[p], m[k]
        for i in range(k + 1, n):
            f = m[i][k] / m[k][k]
            if f != 0.0:
                for j in range(k, n + 1):
                    m[i][j] -= f * m[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / m[i][i]
    return x


def _nozzle(comp, pt, Tt, ht, st, cpt, Rt, p0):
    """Static state in a nozzle throat: expand to ambient, choke if it wants to.

    Returns ``(u, rho, p, T, cp, R, M)``. If a full expansion to ``p0`` would
    give supersonic flow the throat is sonic instead, and the state is
    re-solved at M = 1 -- the convergent nozzle cannot do better.
    """
    p, T, h, s, cp, R = gas_prat(comp, NAIR, pt, Tt, ht, st, cpt, Rt,
                                 p0 / pt, 1.0)
    u = math.sqrt(2.0 * max(ht - h, 0.0))
    M = u / math.sqrt(T * cp * R / (cp - R))
    if M >= 1.0:
        M = 1.0
        p, T, h, s, cp, R = gas_mach(comp, NAIR, pt, Tt, ht, st, cpt, Rt,
                                     0.0, 1.0, 1.0)
    u = math.sqrt(2.0 * (ht - h)) if ht > h else 0.0
    return u, p / (R * T), p, T, cp, R, M


def _cycle_pass(x, c, allow_pc_init=False):
    """One pass down the gas path from the nine unknowns; returns residuals.

    ``c`` carries every fixed input -- flight condition, hardware areas,
    design-point map anchors and component settings. Nothing here mutates it.
    """
    pf, pl, ph, mf, ml, mh, Tb, Pc, Mi = x
    alpha, beta, gamma = c["alpha"], c["beta"], c["gamma"]
    Tref, pref, p0, u0 = c["Tref"], c["pref"], c["p0"], c["u0"]
    st = {}

    # --- inlet, with the boundary-layer defect ---------------------------
    # The BL is mixed in as a mass-averaged entropy rise. Tt2/pt2 are
    # approximated by Tt0/pt0 inside mmix to avoid a circular definition.
    Tt0, pt0, gam0, at0 = c["Tt0"], c["pt0"], c["gam0"], c["at0"]
    if u0 == 0.0:
        sbfan = sbcore = 0.0
    else:
        a2sq = at0 ** 2 / (1.0 + 0.5 * (gam0 - 1.0) * Mi ** 2)
        if c["iBLIc"] == 0:
            mmix = mf * math.sqrt(Tref / Tt0) * pt0 / pref
            sbfan = c["Kinl"] * gam0 / (mmix * a2sq)
            sbcore = 0.0
        else:
            mmix = (mf * math.sqrt(Tref / Tt0) * pt0 / pref
                    + ml * math.sqrt(Tref / Tt0) * pt0 / pref)
            sbfan = c["Kinl"] * gam0 / (mmix * a2sq)
            sbcore = sbfan

    # The BL is adiabatic, so only pressure changes.
    Tt18, ht18, st18, cpt18, Rt18 = (c["Tt18"], c["ht18"], c["st18"],
                                     c["cpt18"], c["Rt18"])
    pt18 = c["pt18"]
    Tt2, ht2, st2, cpt2, Rt2 = Tt18, ht18, st18, cpt18, Rt18
    pt2 = pt18 * math.exp(-sbfan)
    Tt19, ht19, st19, cpt19, Rt19 = Tt18, ht18, st18, cpt18, Rt18
    pt19 = pt18 * math.exp(-sbcore)

    p2, T2, h2, s2, cp2, R2 = gas_mach(alpha, NAIR, pt2, Tt2, ht2, st2,
                                       cpt2, Rt2, 0.0, Mi, 1.0)
    u2 = math.sqrt(2.0 * (ht2 - h2))
    rho2 = p2 / (R2 * T2)

    p19, T19, h19, s19, cp19, R19 = gas_mach(alpha, NAIR, pt19, Tt19, ht19,
                                             st19, cpt19, Rt19, 0.0, Mi, 1.0)
    u19 = math.sqrt(2.0 * (ht19 - h19))
    rho19 = p19 / (R19 * T19)

    # Offtakes, as fractions of the core flow the LPC is passing.
    fo = c["mofft"] / ml * math.sqrt(Tt19 / Tref) * pref / pt19
    Pom = c["Pofft"] / ml * math.sqrt(Tt19 / Tref) * pref / pt19

    # --- fan 2-21, and the fan duct to 7 ---------------------------------
    epf = ecmap(pf, mf, c["pifD"], c["mbfD"], CMAPF, c["epf0"],
                c["pifK"], c["epfK"]).eff
    epf = max(epf, EPFMIN)
    if pf < 1.0:
        epf = 1.0 / epf          # a fan running as a turbine
    pt21, Tt21, ht21, st21, cpt21, Rt21 = gas_prat(
        alpha, NAIR, pt2, Tt2, ht2, st2, cpt2, Rt2, pf, epf)
    pt7, Tt7, ht7, st7, cpt7, Rt7 = (pt21 * c["pifn"], Tt21, ht21, st21,
                                     cpt21, Rt21)

    # --- LP compressor 19-25 ---------------------------------------------
    eplc = max(ecmap(pl, ml, c["pilcD"], c["mblcD"], CMAPLC, c["eplc0"],
                     1.0, 0.0).eff, 0.70)
    pt25, Tt25, ht25, st25, cpt25, Rt25 = gas_prat(
        alpha, NAIR, pt19, Tt19, ht19, st19, cpt19, Rt19, pl, eplc)

    # --- HP compressor 25-3 ----------------------------------------------
    ephc = max(ecmap(ph, mh, c["pihcD"], c["mbhcD"], CMAPHC, c["ephc0"],
                     1.0, 0.0).eff, 0.70)
    pt3, Tt3, ht3, st3, cpt3, Rt3 = gas_prat(
        alpha, NAIR, pt25, Tt25, ht25, st25, cpt25, Rt25, ph, ephc)

    # --- burner 3-4 -------------------------------------------------------
    ffb, lam = gas_burn(alpha, beta, gamma, NGAS, c["ifuel"], Tt3, c["Ttf"],
                        Tb)
    Tt4 = Tb
    s4 = gassum(lam, NAIR, Tt4)
    st4, ht4, cpt4, Rt4 = s4.s, s4.h, s4.cp, s4.r
    pt4 = c["pib"] * pt3

    icool = c["icool"]
    epsrow = list(c["epsrow"])
    Tmrow = list(c["Tmrow"])
    ncrow = c["ncrow"]

    if icool == 0:
        pt41, Tt41, ht41, st41, cpt41, Rt41 = (pt4, Tt4, ht4, st4, cpt4, Rt4)
        lambdap = list(lam)
        fc = 0.0
        ff = (1.0 - fo) * ffb
    else:
        gmi4 = Rt4 / (cpt4 - Rt4)
        Trrat = 1.0 / (1.0 + 0.5 * gmi4 * c["Mtexit"] ** 2)
        if icool == 1:
            Tmrow = tmcalc(ncrow, epsrow, Tt3, Tt4, c["dTstrk"], Trrat,
                           c["efilm"], c["tfilm"], c["StA"])
        else:
            flow = mcool(c["Tmrow"][:c["ncrowx"]], Tt3, Tt4, c["dTstrk"],
                         Trrat, c["efilm"], c["tfilm"], c["StA"])
            ncrow, epsrow = flow.ncrow, flow.epsrow
        fc = sum((1.0 - fo) * epsrow[i] for i in range(ncrow))
        ff = (1.0 - fo - fc) * ffb

        # Coolant is injected at station 4a and mixes to 41.
        p4a, T4a, h4a, s4a, cp4a, R4a = gas_mach(
            lam, NAIR, pt4, Tt4, ht4, st4, cpt4, Rt4, 0.0, c["M4a"], 1.0)
        u4a = math.sqrt(2.0 * (ht4 - h4a)) if ht4 > h4a else 0.0
        uc = c["ruc"] * u4a

        frac4 = (1.0 - fo - fc + ff) / (1.0 - fo + ff)
        fracm = fc / (1.0 - fo + ff)
        lambdap = [frac4 * lam[i] + fracm * alpha[i] for i in range(NAIR)]
        lambdap += [0.0] * (NGAS - NAIR)

        ht41 = frac4 * ht4 + fracm * ht3
        Tt41 = gas_tset(lambdap, NAIR, ht41, frac4 * Tt4 + fracm * Tt3)
        s41t = gassum(lambdap, NAIR, Tt41)
        st41, ht41, cpt41, Rt41 = s41t.s, s41t.h, s41t.cp, s41t.r

        # Mixing is at constant static pressure, so the momentum average
        # of the velocities sets the loss.
        p41 = p4a
        u41 = frac4 * u4a + fracm * uc
        h41 = ht41 - 0.5 * u41 ** 2
        T41 = gas_tset(lambdap, NAIR, h41, T4a + (h41 - h4a) / cp4a)
        s41 = gassum(lambdap, NAIR, T41)
        pt41, Tt41, ht41, st41, cpt41, Rt41 = gas_delh(
            lambdap, NAIR, p41, T41, s41.h, s41.s, s41.cp, s41.r,
            ht41 - s41.h, 1.0)

    # --- spool speeds ------------------------------------------------------
    Nf = Ncmap(pf, mf, c["pifD"], c["mbfD"], c["NbfD"], CMAPF).Nb
    Nl = Ncmap(pl, ml, c["pilcD"], c["mblcD"], c["NblcD"], CMAPLC).Nb
    Nh = Ncmap(ph, mh, c["pihcD"], c["mbhcD"], c["NbhcD"], CMAPHC).Nb

    BPR = mf / ml * math.sqrt(Tt19 / Tt2) * pt2 / pt19

    # --- turbine work ------------------------------------------------------
    # The HPT drives the HPC; the LPT drives the LPC, the fan (scaled by BPR)
    # and the power offtake.
    dhfac = -(1.0 - fo) / (1.0 - fo + ff) / (1.0 - c["epsh"])
    dlfac = -1.0 / (1.0 - fo + ff) / (1.0 - c["epsl"])
    dhht = (ht3 - ht25) * dhfac
    dhlt = (ht25 - ht19 + BPR * (ht21 - ht2) + Pom) * dlfac

    mbht = ml * (1.0 - fo + ff) * math.sqrt(Tt41 / Tt19) * pt19 / pt41
    Nbht = Nh * math.sqrt(Tt25 / Tt41)
    epht = max(etmap(dhht, mbht, Nbht, c["pihtD"], c["mbhtD"], c["NbhtD"],
                     c["epht0"], TMAPH, Tt41, cpt41, Rt41), 0.80)
    pt45, Tt45, ht45, st45, cpt45, Rt45 = gas_delh(
        lambdap, NAIR, pt41, Tt41, ht41, st41, cpt41, Rt41, dhht, 1.0 / epht)

    mblt = ml * (1.0 - fo + ff) * math.sqrt(Tt45 / Tt19) * pt19 / pt45
    Nblt = Nl * math.sqrt(Tt19 / Tt45)
    eplt = max(etmap(dhlt, mblt, Nblt, c["piltD"], c["mbltD"], c["NbltD"],
                     c["eplt0"], TMAPL, Tt45, cpt45, Rt45), 0.80)

    if allow_pc_init and Pc == 0.0:
        _, _, _, _, _, _ = (0,) * 6
        pt49i = gas_delh(lambdap, NAIR, pt45, Tt45, ht45, st45, cpt45, Rt45,
                         dhlt, 1.0 / eplt)[0]
        Pc = max(pt49i * c["pitn"], p0 * (1.0 + 0.2 * c["M0"] ** 2) ** 3.5)
    Pc = max(Pc, 1.000001 * p0)

    # Pressure path to station 5: the turbine is asked for the pressure ratio
    # that lands on Pc, and residual 8 then makes that agree with the work.
    pilt = Pc / pt45 / c["pitn"]
    pt49, Tt49, ht49, st49, cpt49, Rt49 = gas_prat(
        lambdap, NAIR, pt45, Tt45, ht45, st45, cpt45, Rt45, pilt, 1.0 / eplt)
    pt5, Tt5, ht5, st5, cpt5, Rt5 = (pt49 * c["pitn"], Tt49, ht49, st49,
                                     cpt49, Rt49)

    # --- nozzle throats ----------------------------------------------------
    u7, rho7, p7, T7, cp7, R7, M7 = _nozzle(alpha, pt7, Tt7, ht7, st7, cpt7,
                                            Rt7, p0)
    u5, rho5, p5, T5, cp5, R5, M5 = _nozzle(lambdap, pt5, Tt5, ht5, st5, cpt5,
                                            Rt5, p0)

    # --- residuals ---------------------------------------------------------
    res = [0.0] * 9
    rrel = [0.0] * 9

    trf = math.sqrt(Tt2 / Tref)
    trl = math.sqrt(Tt19 / Tref)
    res[0] = c["Gearf"] * trf * Nf - trl * Nl
    rrel[0] = res[0] / Nl

    res[1] = mblt - c["mbltD"]
    rrel[1] = res[1] / c["mbltD"]

    res[2] = mbht - c["mbhtD"]
    rrel[2] = res[2] / c["mbhtD"]

    mdotf = mf * math.sqrt(Tref / Tt2) * pt2 / pref
    res[3] = mdotf - rho7 * u7 * c["A7"]
    rrel[3] = res[3] / mdotf

    mdotc = (1.0 - fo + ff) * ml * math.sqrt(Tref / Tt19) * pt19 / pref
    res[4] = mdotc - rho5 * u5 * c["A5"]
    rrel[4] = res[4] / mdotc

    mdotl = ml * math.sqrt(Tref / Tt19) * pt19 / pref
    mdoth = mh * math.sqrt(Tref / Tt25) * pt25 / pref
    res[5] = mdoth - mdotl + c["mofft"]
    rrel[5] = res[5] / mdoth

    u8 = u6 = None
    F = None
    if c["iTFspec"] == 1:
        res[6] = Tt4 - c["Tt4spec"]
        rrel[6] = res[6] / c["Tt4spec"]
    else:
        p8, T8, h8, s8, cp8, R8 = gas_prat(alpha, NAIR, pt7, Tt7, ht7, st7,
                                           cpt7, Rt7, p0 / pt7, 1.0)
        u8 = math.sqrt(2.0 * (ht7 - h8)) if ht7 > h8 else 0.0
        p6, T6, h6, s6, cp6, R6 = gas_prat(lambdap, NAIR, pt5, Tt5, ht5, st5,
                                           cpt5, Rt5, p0 / pt5, 1.0)
        u6 = math.sqrt(2.0 * (ht5 - h6)) if ht5 > h6 else 0.0
        Finl = 0.0 if u0 == 0.0 else c["Phiinl"] / u0
        F = (((1.0 - fo + ff) * u6 - u0 + BPR * (u8 - u0) + fo * c["u9"])
             * mdotl + Finl)
        res[6] = F - c["Fspec"]
        rrel[6] = res[6] / max(c["Fspec"], F, 1e-6)

    # Work path to station 5, for comparison with the pressure path above.
    pt5h = gas_delh(lambdap, NAIR, pt45, Tt45, ht45, st45, cpt45, Rt45,
                    dhlt, 1.0 / eplt)[0]
    res[7] = pt5 - c["pitn"] * pt5h
    rrel[7] = res[7] / pt5

    mfA = mf * math.sqrt(Tref / Tt2) * pt2 / pref / (rho2 * u2)
    mlA = ml * math.sqrt(Tref / Tt19) * pt19 / pref / (rho19 * u19)
    res[8] = mfA + mlA - c["A2"]
    rrel[8] = res[8]

    for name, vals in (
            ("0", (Tt0, c["ht0"], pt0, c["cpt0"], c["Rt0"])),
            ("18", (Tt18, ht18, pt18, cpt18, Rt18)),
            ("19", (Tt19, ht19, pt19, cpt19, Rt19)),
            ("2", (Tt2, ht2, pt2, cpt2, Rt2)),
            ("21", (Tt21, ht21, pt21, cpt21, Rt21)),
            ("25", (Tt25, ht25, pt25, cpt25, Rt25)),
            ("3", (Tt3, ht3, pt3, cpt3, Rt3)),
            ("4", (Tt4, ht4, pt4, cpt4, Rt4)),
            ("41", (Tt41, ht41, pt41, cpt41, Rt41)),
            ("45", (Tt45, ht45, pt45, cpt45, Rt45)),
            ("49", (Tt49, ht49, pt49, cpt49, Rt49)),
            ("5", (Tt5, ht5, pt5, cpt5, Rt5)),
            ("7", (Tt7, ht7, pt7, cpt7, Rt7))):
        s = Station()
        s.Tt, s.ht, s.pt, s.cpt, s.Rt = vals
        st[int(name)] = s
    st[2].T, st[2].u, st[2].p, st[2].cp, st[2].R = T2, u2, p2, cp2, R2
    st[19].T, st[19].u, st[19].p = T19, u19, p19
    st[5].T, st[5].u, st[5].p, st[5].cp, st[5].R = T5, u5, p5, cp5, R5
    st[7].T, st[7].u, st[7].p, st[7].cp, st[7].R = T7, u7, p7, cp7, R7

    scalars = dict(ffb=ffb, ff=ff, fo=fo, fc=fc, Pom=Pom, BPR=BPR,
                   Nf=Nf, Nl=Nl, Nh=Nh, Nbht=Nbht, Nblt=Nblt,
                   mbht=mbht, mblt=mblt, mdotl=mdotl, mdotc=mdotc,
                   mdoth=mdoth, mdotf=mdotf, dhht=dhht, dhlt=dhlt,
                   epf=epf, eplc=eplc, ephc=ephc, epht=epht, eplt=eplt,
                   M5=M5, M7=M7, u5=u5, u7=u7, u6=u6, u8=u8, F=F,
                   Pc=Pc, ncrow=ncrow, epsrow=epsrow, Tmrow=Tmrow,
                   rrel=rrel, rho2=rho2, rho19=rho19, u19=u19)
    return _Cycle(res=res, st=st, scalars=scalars, lambdap=lambdap)


def tfoper(gee, M0, T0, p0, a0, Tref, pref,
           Phiinl, Kinl, iBLIc,
           pid, pib, pifn, pitn,
           Gearf,
           pifD, pilcD, pihcD, pihtD, piltD,
           mbfD, mblcD, mbhcD, mbhtD, mbltD,
           NbfD, NblcD, NbhcD, NbhtD, NbltD,
           A2, A25, A5, A7,
           iTFspec,
           Ttf, ifuel, etab,
           epf0, eplc0, ephc0, epht0, eplt0,
           pifK, epfK,
           mofft, Pofft,
           Tt9, pt9,
           epsl, epsh,
           icool,
           Mtexit, dTstrk, StA, efilm, tfilm,
           M4a, ruc,
           ncrowx, ncrow=0, epsrow=None, Tmrow=None,
           *,
           Tt4=0.0, Feng=0.0, pt5=0.0,
           pif=0.0, pilc=0.0, pihc=0.0,
           mbf=0.0, mblc=0.0, mbhc=0.0,
           M2=0.0, M25=0.6, mcore=0.0) -> TFOperResult:
    """Run a sized turbofan off design.

    ``iTFspec`` picks what is being held: 1 specifies ``Tt4`` and thrust comes
    out, 2 specifies ``Feng`` and the burner temperature comes out.

    The starting guesses ``pif``/``pilc``/``pihc``/``mbf``/``mblc``/``mbhc``/
    ``pt5``/``M2`` default to zero, which the source reads as "use the design
    point"; passing the previous flight point's answer instead is what makes a
    mission sweep cheap.

    ``icool`` matches :func:`tasopt_py.engine.tfsize.tfsize`: 0 none, 1
    ``epsrow`` given and metal temperature computed, 2 ``Tmrow`` given and the
    coolant computed.
    """
    alpha = list(AIRFRAC)
    beta = [0.0] * NAIR + [1.0]
    gamma = gasfuel(ifuel, NGAS)
    gamma = [etab * g for g in gamma[:NAIR]] + [1.0 - etab]

    # --- freestream, fixed for the whole solve ---------------------------
    s0 = gassum(alpha, NAIR, T0)
    h0, sr0, cp0, R0 = s0.h, s0.s, s0.cp, s0.r
    gam0 = cp0 / (cp0 - R0)
    u0 = M0 * a0
    Tt0 = gas_tset(alpha, NAIR, h0 + 0.5 * u0 ** 2,
                   T0 * (1.0 + 0.5 * (gam0 - 1.0) * M0 ** 2))
    st0 = gassum(alpha, NAIR, Tt0)
    pt0 = p0 * math.exp((st0.s - sr0) / st0.r)
    at0 = math.sqrt(Tt0 * st0.r * st0.cp / (st0.cp - st0.r))

    # Offtake plume, station 9.
    Trat = (p0 / pt9) ** (st0.r / st0.cp)
    if Trat < 1.0:
        u9 = math.sqrt(2.0 * st0.cp * Tt9 * (1.0 - Trat))
        rho9 = p0 / (st0.r * Tt0 * Trat)
    else:
        u9 = 0.0
        rho9 = p0 / (st0.r * Tt0)

    c = dict(alpha=alpha, beta=beta, gamma=gamma, Tref=Tref, pref=pref,
             p0=p0, u0=u0, M0=M0, Tt0=Tt0, pt0=pt0, gam0=gam0, at0=at0,
             ht0=st0.h, cpt0=st0.cp, Rt0=st0.r,
             Tt18=Tt0, ht18=st0.h, st18=st0.s, cpt18=st0.cp, Rt18=st0.r,
             pt18=pt0 * pid,
             Phiinl=Phiinl, Kinl=Kinl, iBLIc=iBLIc,
             pib=pib, pifn=pifn, pitn=pitn, Gearf=Gearf,
             pifD=pifD, pilcD=pilcD, pihcD=pihcD, pihtD=pihtD, piltD=piltD,
             mbfD=mbfD, mblcD=mblcD, mbhcD=mbhcD, mbhtD=mbhtD, mbltD=mbltD,
             NbfD=NbfD, NblcD=NblcD, NbhcD=NbhcD, NbhtD=NbhtD, NbltD=NbltD,
             A2=A2, A25=A25, A5=A5, A7=A7, iTFspec=iTFspec,
             Ttf=Ttf, ifuel=ifuel, etab=etab,
             epf0=epf0, eplc0=eplc0, ephc0=ephc0, epht0=epht0, eplt0=eplt0,
             pifK=pifK, epfK=epfK, mofft=mofft, Pofft=Pofft, u9=u9,
             epsl=epsl, epsh=epsh, icool=icool, Mtexit=Mtexit,
             dTstrk=dTstrk, StA=StA, efilm=efilm, tfilm=tfilm,
             M4a=M4a, ruc=ruc, ncrowx=ncrowx, ncrow=ncrow,
             epsrow=list(epsrow) if epsrow is not None else [0.0] * ncrowx,
             Tmrow=list(Tmrow) if Tmrow is not None else [0.0] * ncrowx,
             Tt4spec=Tt4, Fspec=Feng)

    # --- starting guesses; zero means "use the design point" -------------
    x = [pif or pifD, pilc or pilcD, pihc or pihcD,
         mbf or mbfD, mblc or mblcD, mbhc or mbhcD,
         Tt4, pt5, M2]
    if x[8] in (0.0, 1.0):
        x[8] = 0.6

    # Prime Pc once, exactly where the source does. After this it is nonzero,
    # so the branch never reopens and the residual is a pure function of x --
    # which it has to be for the finite differences below to mean anything.
    if x[7] == 0.0:
        x[7] = _cycle_pass(x, c, allow_pc_init=True).scalars["Pc"]

    converged = False
    cyc = None
    for it in range(1, ITMAX + 1):
        x[7] = max(x[7], 1.000001 * p0)
        cyc = _cycle_pass(x, c)
        res = cyc.res

        # Jacobian by central differences.
        a = [[0.0] * 9 for _ in range(9)]
        for j in range(9):
            h = FD_STEP * max(abs(x[j]), 1.0e-6)
            xp, xm = list(x), list(x)
            xp[j] += h
            xm[j] -= h
            rp = _cycle_pass(xp, c).res
            rm = _cycle_pass(xm, c).res
            for i in range(9):
                a[i][j] = (rp[i] - rm[i]) / (2.0 * h)

        d = [-v for v in _solve(a, res)]

        # If the fan face is being pushed past its cap, drop the area
        # constraint and pin Mi instead -- choking as an active constraint.
        if x[8] >= MIMAX and d[8] > 0.0:
            a2 = [row[:] for row in a]
            r2 = list(res)
            a2[8] = [0.0] * 9
            a2[8][8] = 1.0
            r2[8] = x[8] - MIMAX
            d = [-v for v in _solve(a2, r2)]

        dmax = max(abs(d[k]) / abs(x[k]) for k in range(9))

        # Step limits. Each one keeps a variable inside the range where the
        # gas path is defined at all.
        Tt3 = cyc.st[3].Tt
        hi = [0.30 * (x[0] - 0.8), 0.25 * (x[1] - 0.8), 0.25 * (x[2] - 1.0),
              0.20 * x[3], 0.20 * x[4], 0.20 * x[5],
              0.50 * (x[6] - Tt3), 1.00 * (x[7] - p0), 1.001 * (MIMAX - x[8])]
        lo = [-0.30 * (x[0] - 0.8), -0.25 * (x[1] - 0.8), -0.25 * (x[2] - 1.0),
              -0.20 * x[3], -0.20 * x[4], -0.20 * x[5],
              -0.30 * (x[6] - Tt3), -0.50 * (x[7] - p0), -0.20 * x[8]]
        rlx = 1.0
        for k in range(9):
            if rlx * d[k] > hi[k]:
                rlx = hi[k] / d[k]
        for k in range(9):
            if rlx * d[k] < lo[k]:
                rlx = lo[k] / d[k]

        if dmax < TOLER:
            converged = True
            break

        for k in range(9):
            x[k] += rlx * d[k]
        x[8] = min(x[8], MIMAX)
    else:
        raise TFOperError(
            f"tfoper: convergence failed after {ITMAX} iterations, "
            f"iTFspec={iTFspec}, dmax={dmax!r}")

    # === converged; assemble the outputs =================================
    sc = cyc.scalars
    st = cyc.st
    lambdap = cyc.lambdap
    fo, ff, ffb, BPR = sc["fo"], sc["ff"], sc["ffb"], sc["BPR"]

    # Fully expanded plumes, 8 and 6.
    s7 = st[7]
    p8, T8, h8, s8, cp8, R8 = gas_prat(alpha, NAIR, s7.pt, s7.Tt, s7.ht,
                                       gassum(alpha, NAIR, s7.Tt).s,
                                       s7.cpt, s7.Rt, p0 / s7.pt, 1.0)
    u8 = math.sqrt(2.0 * (s7.ht - h8)) if s7.ht > h8 else 0.0
    rho8 = p8 / (R8 * T8)

    s5 = st[5]
    p6, T6, h6, s6, cp6, R6 = gas_prat(lambdap, NAIR, s5.pt, s5.Tt, s5.ht,
                                       gassum(lambdap, NAIR, s5.Tt).s,
                                       s5.cpt, s5.Rt, p0 / s5.pt, 1.0)
    u6 = math.sqrt(2.0 * (s5.ht - h6)) if s5.ht > h6 else 0.0
    rho6 = p6 / (R6 * T6)

    mcore = x[4] * math.sqrt(Tref / st[19].Tt) * st[19].pt / pref
    fo = mofft / mcore

    cpa = 0.5 * (st[3].cpt + st[4].cpt)
    hfuel = cpa * (st[4].Tt - st[3].Tt + ffb * (st[4].Tt - Ttf)) / (etab * ffb)

    Finl = 0.0 if u0 == 0.0 else Phiinl / u0
    Feng = ((1.0 - fo + ff) * u6 - u0 + BPR * (u8 - u0) + fo * u9) * mcore \
        + Finl
    Fsp = 0.0 if u0 == 0.0 else Feng / (u0 * mcore * (1.0 + BPR))
    TSFC = 0.0 if Feng == 0.0 else (gee * ff * mcore) / Feng

    for stn, (T, u, p, cp, R, rho) in (
            (8, (T8, u8, p8, cp8, R8, rho8)),
            (6, (T6, u6, p6, cp6, R6, rho6))):
        s = Station()
        s.T, s.u, s.p, s.cp, s.R = T, u, p, cp, R
        src = st[7] if stn == 8 else st[5]
        s.Tt, s.ht, s.pt, s.cpt, s.Rt = src.Tt, src.ht, src.pt, src.cpt, src.Rt
        st[stn] = s
    st[8].A = BPR * mcore / (rho8 * u8)
    st[6].A = (1.0 - fo + ff) * mcore / (rho6 * u6)
    s9 = Station()
    s9.u = u9
    s9.A = 0.0 if u9 == 0.0 else fo * mcore / (rho9 * u9)
    st[9] = s9
    M8 = u8 / math.sqrt(T8 * R8 * cp8 / (cp8 - R8))
    M6 = u6 / math.sqrt(T6 * R6 * cp6 / (cp6 - R6))

    # Isentropic efficiencies -- informative only, nothing above uses them.
    def _isen(comp, s_in, pi):
        s = gassum(comp, NAIR, s_in.Tt)
        return gas_prat(comp, NAIR, s_in.pt, s_in.Tt, s_in.ht, s.s,
                        s_in.cpt, s_in.Rt, pi, 1.0)[2]
    ht21i = _isen(alpha, st[2], x[0])
    etaf = (ht21i - st[2].ht) / (st[21].ht - st[2].ht)
    ht25i = _isen(alpha, st[19], x[1])
    etalc = (ht25i - st[19].ht) / (st[25].ht - st[19].ht)
    ht3i = _isen(alpha, st[25], x[2])
    etahc = (ht3i - st[25].ht) / (st[3].ht - st[25].ht)
    ht45i = _isen(lambdap, st[41], st[45].pt / st[41].pt)
    etaht = (st[45].ht - st[41].ht) / (ht45i - st[41].ht)
    ht49i = _isen(lambdap, st[45], st[49].pt / st[45].pt)
    etalt = (st[5].ht - st[45].ht) / (ht49i - st[45].ht)

    # Station 25 static state: sonic-ish unless the flow is too small for it.
    s25 = st[25]
    ss25 = gassum(alpha, NAIR, s25.Tt)
    M25s = 0.99
    p25s, T25s, h25s, _, cp25s, R25s = gas_mach(
        alpha, NAIR, s25.pt, s25.Tt, s25.ht, ss25.s, s25.cpt, s25.Rt,
        0.0, M25s, 1.0)
    u25s = math.sqrt(2.0 * (s25.ht - h25s))
    mdot25s = p25s / (R25s * T25s) * u25s * A25
    if (1.0 - fo) * mcore >= mdot25s:
        u25, p25, T25, cp25, R25, M25 = u25s, p25s, T25s, cp25s, R25s, M25s
    else:
        p25, T25, h25, _, cp25, R25 = gas_mass(
            alpha, NAIR, s25.pt, s25.Tt, s25.ht, ss25.s, s25.cpt, s25.Rt,
            (1.0 - fo) * mcore / A25, min(M25, 0.90))
        u25 = math.sqrt(2.0 * (s25.ht - h25))
        M25 = u25 / math.sqrt(T25 * R25 * cp25 / (cp25 - R25))
    s25.T, s25.u, s25.p, s25.cp, s25.R = T25, u25, p25, cp25, R25
    st[0].u = u0

    return TFOperResult(
        TSFC=TSFC, Fsp=Fsp, hfuel=hfuel, ff=ff, Feng=Feng, mcore=mcore,
        BPR=BPR, pif=x[0], pilc=x[1], pihc=x[2],
        mbf=x[3], mblc=x[4], mbhc=x[5],
        Nbf=sc["Nl"] / Gearf * math.sqrt(st[19].Tt / st[2].Tt),
        Nblc=sc["Nl"], Nbhc=sc["Nh"],
        stations=st, epsrow=sc["epsrow"], Tmrow=sc["Tmrow"],
        ncrow=sc["ncrow"],
        epf=sc["epf"], eplc=sc["eplc"], ephc=sc["ephc"],
        epht=sc["epht"], eplt=sc["eplt"],
        etaf=etaf, etalc=etalc, etahc=etahc, etaht=etaht, etalt=etalt,
        iterations=it, converged=converged)
