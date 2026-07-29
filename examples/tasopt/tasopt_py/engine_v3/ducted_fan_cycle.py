"""Ducted fan gas path -- ``ductedfansize.jl`` and ``ductedfanoper.jl``.

A turbofan with the core removed. The fan is driven by a shaft from
somewhere else -- a fuel cell stack, a battery, a turboshaft -- so the cycle
has no burner, no turbine and no fuel flow, and the thing that closes it is
**shaft power in** rather than fuel energy.

TASOPT 2.16 has no equivalent. Its ``tfsize``/``tfoper`` always burn fuel;
you cannot ask them for an electrically driven fan.

Station numbering (the reference's, kept)::

     0  freestream          2  fan face
    18  fan face outside    21 fan exit
     7  fan nozzle          8  fan plume

What actually closes each direction
-----------------------------------
:func:`ducted_fan_size` is given **thrust** and finds the mass flow that
delivers it, then the areas that pass that mass flow. It iterates because
the boundary-layer-ingestion entropy loss ``sbfan`` depends on the mass flow
it is solving for.

:func:`ducted_fan_operate` is given fixed **areas** and either a thrust or a
shaft power, and solves for the fan pressure ratio and mass flow that fit
them. Two unknowns, two residuals, one Newton -- and unlike 2.16's
``tfoper`` it is only two, because there is no core to balance.

Verified against TASOPT.jl; see ``tests/test_ducted_fan_cycle.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..gas.mixture import gassum, gas_tset, gas_prat, gas_mach
from .maps import FAN_MAP, compressor_speed_and_efficiency

__all__ = ["AIR_ALPHA", "DuctedFanState", "ducted_fan_size",
           "ducted_fan_operate", "RLXS", "V3_CMAPF"]

#: N2, O2, CO2, H2O, Ar -- with a sixth slot the reference carries and never
#: fills, since there is no combustion to make anything.
AIR_ALPHA = [0.7532, 0.2315, 0.0006, 0.0020, 0.0127, 0.0]
NAIR = 5

#: Under-relaxation on the BLI entropy loss.
RLXS = 0.85
TOLER = 1.0e-12
NPASS = 60


@dataclass
class DuctedFanState:
    """Everything the two routines return, named rather than positional.

    The reference returns a 50-element tuple; unpacking it correctly is its
    own hazard.
    """
    TSEC: float = 0.0        # thrust-specific energy consumption, J/N
    Fsp: float = 0.0         # specific thrust
    Pfan: float = 0.0        # shaft power into the fan, W
    mfan: float = 0.0        # fan mass flow, kg/s
    Feng: float = 0.0        # net thrust, N
    pif: float = 0.0         # fan pressure ratio
    Nf: float = 0.0          # fan speed
    Nbf: float = 0.0         # corrected fan speed
    epf: float = 0.0         # fan polytropic efficiency
    etaf: float = 0.0        # fan overall (isentropic) efficiency
    u0: float = 0.0
    # stagnation states, keyed by station
    Tt: dict = None
    ht: dict = None
    pt: dict = None
    cpt: dict = None
    Rt: dict = None
    # static states, keyed by station
    T: dict = None
    u: dict = None
    p: dict = None
    cp: dict = None
    R: dict = None
    A: dict = None
    M: dict = None
    mbf: float = 0.0

    def __post_init__(self):
        for f in ("Tt", "ht", "pt", "cpt", "Rt", "T", "u", "p", "cp", "R",
                  "A"):
            if getattr(self, f) is None:
                setattr(self, f, {})


def _freestream(M0, T0, p0, a0):
    """Stagnation state at station 0."""
    s0 = gassum(AIR_ALPHA, NAIR, T0)
    gam0 = s0.cp / (s0.cp - s0.r)
    u0 = M0 * a0
    Tguess = T0 * (1.0 + 0.5 * (gam0 - 1.0) * M0 ** 2)
    Tt0 = gas_tset(AIR_ALPHA, NAIR, s0.h + 0.5 * u0 ** 2, Tguess)
    st0 = gassum(AIR_ALPHA, NAIR, Tt0)
    pt0 = p0 * math.exp((st0.s - s0.s) / st0.r)
    at0 = math.sqrt(Tt0 * st0.r * st0.cp / (st0.cp - st0.r))
    return u0, gam0, Tt0, st0, pt0, at0


def _nozzle(pt7, Tt7, ht7, st7, cpt7, Rt7, p0):
    """Expand the fan stream to ambient. ``(station 7 state, station 8)``.

    The nozzle is choked or not, and which decides where the thrust comes
    from. If the plume Mach number is below one, stations 7 and 8 are the
    same point and all the thrust is momentum. If it is above one, station 7
    is held at ``M = 1`` and the leftover pressure at the throat contributes
    a pressure thrust term as well.
    """
    p8, T8, h8, s8, cp8, R8 = gas_prat(AIR_ALPHA, NAIR, pt7, Tt7, ht7, st7,
                                       cpt7, Rt7, p0 / pt7, 1.0)
    if h8 >= ht7:
        raise ValueError(
            f"negative fan plume velocity: the nozzle total enthalpy "
            f"{ht7:.6g} J/kg is not above the static {h8:.6g} J/kg at "
            f"pt8 = {pt7:.6g} Pa, p8 = {p8:.6g} Pa. The fan is not raising "
            "the pressure enough to expand to ambient.")
    u8 = math.sqrt(2.0 * (ht7 - h8))
    rho8 = p8 / (R8 * T8)
    M8 = u8 / math.sqrt(cp8 * R8 / (cp8 - R8) * T8)

    if M8 < 1.0:
        st = (p8, T8, h8, s8, cp8, R8, u8)
    else:
        p7, T7, h7, s7, cp7, R7 = gas_mach(AIR_ALPHA, NAIR, pt7, Tt7, ht7,
                                           st7, cpt7, Rt7, 0.0, 1.0, 1.0)
        st = (p7, T7, h7, s7, cp7, R7, math.sqrt(2.0 * (ht7 - h7)))
    return st, (p8, T8, h8, s8, cp8, R8, u8, rho8, M8)


def ducted_fan_size(gee, M0, T0, p0, a0, M2, Feng, Phiinl, Kinl, iBLIc,
                    pif, pid, pifn, epf0, dh_radiator=0.0,
                    dp_radiator=0.0) -> DuctedFanState:
    """Size a ducted fan for a required net thrust.

    Returns a :class:`DuctedFanState`. ``pif`` is the design fan pressure
    ratio, ``pid`` and ``pifn`` the diffuser and nozzle pressure ratios, and
    ``epf0`` the fan's peak polytropic efficiency.

    The loop is over the **boundary-layer-ingestion entropy rise**
    ``sbfan``: ingesting the fuselage wake costs stagnation pressure at the
    fan face, the cost depends on the mass flow, and the mass flow depends
    on the cost. Under-relaxed at :data:`RLXS`. With ``Kinl = 0`` the loop
    is trivial -- ``sbfan`` stays zero -- and it converges on the second
    pass.

    Note the fan efficiency comes from the map at the map's *own* design
    point, which is exactly a grid knot. The value is well defined there
    (only the derivative is not; see ``DISCREPANCIES.md`` §78), and sizing
    does not use the derivative.
    """
    u0, gam0, Tt0, st0, pt0, at0 = _freestream(M0, T0, p0, a0)
    pt18 = pt0 * pid
    Tt18, ht18, s18, cpt18, Rt18 = Tt0, st0.h, st0.s, st0.cp, st0.r

    mfan = 0.0
    sbfan = 0.0
    pt2 = pt18
    Tt2 = Tt18
    out = DuctedFanState()

    for ipass in range(1, NPASS + 1):
        if ipass > 1:
            a2sq = at0 ** 2 / (1.0 + 0.5 * (gam0 - 1.0) * M2 ** 2)
            mmix = mfan * math.sqrt(Tt2 / Tt0) * pt0 / pt2
            sbfan += RLXS * (Kinl * gam0 / (mmix * a2sq) - sbfan)

        Tt2, ht2, st2, cpt2, Rt2 = Tt18, ht18, s18, cpt18, Rt18
        pt2 = pt18 * math.exp(-sbfan)

        # At the design point the map is asked for its own design pressure
        # ratio and mass flow, so the inverse lands on the map's knot.
        _, epf = compressor_speed_and_efficiency(FAN_MAP, pif, 1.0, pif,
                                                 1.0, 1.0, epf0)[:2]

        pt21, Tt21, ht21, st21, cpt21, Rt21 = gas_prat(
            AIR_ALPHA, NAIR, pt2, Tt2, ht2, st2, cpt2, Rt2, pif, epf)

        pt7 = pt21 * pifn - dp_radiator
        ht7 = ht21 + dh_radiator
        Tt7 = gas_tset(AIR_ALPHA, NAIR, ht7, Tt21)
        s7 = gassum(AIR_ALPHA, NAIR, Tt7)
        st7, cpt7, Rt7, ht7 = s7.s, s7.cp, s7.r, s7.h

        (p7, T7, h7, _, cp7, R7, u7), (p8, T8, h8, _, cp8, R8, u8, rho8,
                                       M8) = _nozzle(
            pt7, Tt7, ht7, st7, cpt7, Rt7, p0)
        rho7 = p7 / (R7 * T7)

        mfold = mfan
        Finl = 0.0 if u0 == 0.0 else Phiinl / u0
        mfan = (Feng - Finl) / (u7 + (p7 - p0) / (rho7 * u7) - u0)

        Pfan = mfan * (ht21 - ht2)
        A7 = mfan / (rho7 * u7)
        A8 = mfan / (rho8 * u8)

        p2, T2, h2, _, cp2, R2 = gas_mach(AIR_ALPHA, NAIR, pt2, Tt2, ht2,
                                          st2, cpt2, Rt2, 0.0, M2, 1.0)
        u2 = math.sqrt(2.0 * (ht2 - h2))
        A2 = mfan / ((p2 / (R2 * T2)) * u2)

        if ipass >= 2 and abs(1.0 - mfold / mfan) < TOLER:
            _, _, ht21i, _, _, _ = gas_prat(AIR_ALPHA, NAIR, pt2, Tt2, ht2,
                                            st2, cpt2, Rt2, pif, 1.0)
            out.TSEC = Pfan / Feng
            out.Fsp = Feng / (u0 * mfan) if u0 != 0.0 else math.inf
            out.Pfan, out.mfan, out.Feng, out.pif = Pfan, mfan, Feng, pif
            out.epf, out.etaf = epf, (ht21i - ht2) / (ht21 - ht2)
            out.u0 = u0
            out.Tt = {0: Tt0, 18: Tt18, 2: Tt2, 21: Tt21, 7: Tt7}
            out.ht = {0: st0.h, 18: ht18, 2: ht2, 21: ht21, 7: ht7}
            out.pt = {0: pt0, 18: pt18, 2: pt2, 21: pt21, 7: pt7}
            out.cpt = {0: st0.cp, 18: cpt18, 2: cpt2, 21: cpt21, 7: cpt7}
            out.Rt = {0: st0.r, 18: Rt18, 2: Rt2, 21: Rt21, 7: Rt7}
            out.T = {2: T2, 7: T7, 8: T8}
            out.u = {2: u2, 7: u7, 8: u8}
            out.p = {2: p2, 7: p7, 8: p8}
            out.cp = {2: cp2, 7: cp7, 8: cp8}
            out.R = {2: R2, 7: R7, 8: R8}
            out.A = {2: A2, 7: A7, 8: A8}
            return out

    raise ValueError(
        f"ducted fan sizing did not converge in {NPASS} passes: the "
        f"boundary-layer-ingestion entropy loop is still moving the mass "
        f"flow by {abs(1.0 - mfold / mfan):.3g} per pass")


# --------------------------------------------------------------------------
# Off design
# --------------------------------------------------------------------------

#: v3's re-fitted fan-speed map constants. **Not** 2.16's shipped set, which
#: is ``(3.50, 0.80, 0.03, 0.95, -0.50, 3.0, 6.0, 0.0, 0.0)``.
#:
#: Worth knowing where these are used: the fan *efficiency* comes from the
#: tabulated pyCycle map, but the fan *speed* still comes from 2.16's
#: analytic ``Ncmap`` with these constants. So one routine reads two
#: different fan models, and they were not fitted to each other.
V3_CMAPF = (3.31140687, 0.77839352, 0.03086818, 0.57042461, -0.81725615,
            6.24179886, 15.42860808, 2.95705214, 0.61792148)

#: Below this the efficiency returned by the map is inverted rather than
#: used -- see :func:`ducted_fan_operate`.
EPF_MIN = 0.60


def _off_design_residual(x, ctx, iPspec, store=False):
    """The three residuals, and optionally the full state.

    ``x = (pif, mbf, M2)``. The residuals are all relative:

    1. mass flow through the nozzle throat matches the fan mass flow
    2. mass flow through the fan face matches it too
    3. the requested thrust *or* shaft power is delivered

    Residual 3 is the only one that changes with ``iPspec``, which is why an
    electrically driven fan can be run either way round: give it a thrust
    and ask what power it needs, or give it a power and ask what thrust it
    makes.
    """
    (M0, T0, p0, a0, Tref, pref, Phiinl, Kinl, iBLIc, pid, pifn, pifD,
     mbfD, NbfD, A2, A7, epf0, Feng, Peng, dh_rad, dp_rad) = ctx
    pf, mf, Mi = x

    u0, gam0, Tt0, st0, pt0, at0 = _freestream(M0, T0, p0, a0)
    pt18 = pt0 * pid
    Tt18, ht18, s18, cpt18, Rt18 = Tt0, st0.h, st0.s, st0.cp, st0.r

    if u0 == 0.0:
        sbfan = 0.0
    else:
        a2sq = at0 ** 2 / (1.0 + 0.5 * (gam0 - 1.0) * Mi ** 2)
        mmix = mf * math.sqrt(Tref / Tt0) * pt0 / pref
        sbfan = Kinl * gam0 / (mmix * a2sq)

    Tt2, ht2, st2, cpt2, Rt2 = Tt18, ht18, s18, cpt18, Rt18
    pt2 = pt18 * math.exp(-sbfan)

    p2, T2, h2, _, cp2, R2 = gas_mach(AIR_ALPHA, NAIR, pt2, Tt2, ht2, st2,
                                      cpt2, Rt2, 0.0, Mi, 1.0)
    u2 = math.sqrt(2.0 * (ht2 - h2))
    rho2 = p2 / (R2 * T2)

    _, epf = compressor_speed_and_efficiency(FAN_MAP, pf, mf, pifD, mbfD,
                                             1.0, epf0)[:2]
    if pf < 1.0:
        # A "fan" with a pressure ratio below one is a turbine, and the
        # polytropic exponent flips. The reference inverts the efficiency;
        # a solver only reaches here while wandering.
        epf = 1.0 / epf

    pt21, Tt21, ht21, st21, cpt21, Rt21 = gas_prat(
        AIR_ALPHA, NAIR, pt2, Tt2, ht2, st2, cpt2, Rt2, pf, epf)

    pt7 = pt21 * pifn - dp_rad
    ht7 = ht21 + dh_rad
    Tt7 = gas_tset(AIR_ALPHA, NAIR, ht7, Tt21)
    s7 = gassum(AIR_ALPHA, NAIR, Tt7)
    st7, cpt7, Rt7, ht7 = s7.s, s7.cp, s7.r, s7.h

    pfn = p0 / pt7
    p7, T7, h7, s7v, cp7, R7 = gas_prat(AIR_ALPHA, NAIR, pt7, Tt7, ht7, st7,
                                        cpt7, Rt7, pfn, 1.0)
    u7 = math.sqrt(2.0 * max(ht7 - h7, 0.0))
    M7 = u7 / math.sqrt(T7 * cp7 * R7 / (cp7 - R7))
    if M7 > 1.0:
        M7 = 1.0
        p7, T7, h7, s7v, cp7, R7 = gas_mach(AIR_ALPHA, NAIR, pt7, Tt7, ht7,
                                            st7, cpt7, Rt7, 0.0, M7, 1.0)
    u7 = math.sqrt(2.0 * (ht7 - h7)) if ht7 > h7 else 0.0
    rho7 = p7 / (R7 * T7)

    # Station 8 expands isentropically to ambient regardless of choking, so
    # a choked nozzle has u8 > u7 and the plume area is not the throat area.
    p8, T8, h8, _, cp8, R8 = gas_prat(AIR_ALPHA, NAIR, pt7, Tt7, ht7, st7,
                                      cpt7, Rt7, pfn, 1.0)
    u8 = math.sqrt(2.0 * (ht7 - h8)) if ht7 > h8 else 0.0
    M8 = u8 / math.sqrt(T8 * R8 * cp8 / (cp8 - R8))
    rho8 = p8 / (R8 * T8)

    Finl = 0.0 if u0 == 0.0 else Phiinl / u0
    mfan = mf * math.sqrt(Tref / Tt2) * pt2 / pref
    F = mfan * (u7 - u0) + A7 * (p7 - p0) + Finl
    P = mfan * (ht21 - ht2)

    res = [(mfan - rho7 * A7 * u7) / mfan,
           (mfan - rho2 * A2 * u2) / mfan,
           (P - Peng) / Peng if iPspec else (F - Feng) / Feng]
    if not store:
        return res

    from ..engine.maps import Ncmap
    _, _, ht21i, _, _, _ = gas_prat(AIR_ALPHA, NAIR, pt2, Tt2, ht2, st2,
                                    cpt2, Rt2, pf, 1.0)
    out = DuctedFanState()
    out.TSEC = P / F
    out.Fsp = 0.0 if u0 == 0.0 else F / (u0 * mfan)
    out.Feng, out.Pfan, out.mfan, out.pif = F, P, mfan, pf
    out.Nbf = Ncmap(pf, mf, pifD, mbfD, NbfD, V3_CMAPF).Nb
    out.epf, out.etaf = epf, (ht21i - ht2) / (ht21 - ht2)
    out.u0 = u0
    out.Tt = {0: Tt0, 18: Tt18, 2: Tt2, 21: Tt21, 7: Tt7}
    out.ht = {0: st0.h, 18: ht18, 2: ht2, 21: ht21, 7: ht7}
    out.pt = {0: pt0, 18: pt18, 2: pt2, 21: pt21, 7: pt7}
    out.cpt = {0: st0.cp, 18: cpt18, 2: cpt2, 21: cpt21, 7: cpt7}
    out.Rt = {0: st0.r, 18: Rt18, 2: Rt2, 21: Rt21, 7: Rt7}
    out.T = {2: T2, 7: T7, 8: T8}
    out.u = {2: u2, 7: u7, 8: u8}
    out.p = {2: p2, 7: p7, 8: p8}
    out.cp = {2: cp2, 7: cp7, 8: cp8}
    out.R = {2: R2, 7: R7, 8: R8}
    out.A = {2: A2, 7: A7, 8: mfan / (rho8 * u8)}
    out.M = {2: Mi, 7: M7, 8: M8}
    out.mbf = mf
    return res, out


def ducted_fan_operate(M0, T0, p0, a0, Tref, pref, Phiinl, Kinl, iBLIc,
                       pid, pifn, pifD, mbfD, NbfD, A2, A7, epf0,
                       Feng=0.0, Peng=0.0, M2=0.0, pif=0.0, mbf=0.0,
                       dh_radiator=0.0, dp_radiator=0.0,
                       iPspec=False) -> DuctedFanState:
    """Run a sized ducted fan off design.

    Three unknowns -- fan pressure ratio, corrected mass flow, fan-face Mach
    number -- against three residuals. Set ``iPspec`` to specify shaft power
    instead of thrust.

    Half the size of 2.16's ``tfoper``, which balances a core as well. Here
    the fan is fed by a shaft, so there is no turbine work to match and no
    fuel flow to find.

    Zero arguments mean "use the design value": ``pif``, ``mbf`` and ``M2``
    all fall back rather than being taken literally, and ``M2 == 1.0`` also
    falls back -- the reference treats a sonic fan face as a placeholder
    rather than a request.
    """
    from scipy.optimize import root

    pf = pif if pif != 0.0 else pifD
    mf = mbf if mbf != 0.0 else mbfD
    Mi = M2 if M2 not in (0.0, 1.0) else 0.6

    ctx = (M0, T0, p0, a0, Tref, pref, Phiinl, Kinl, iBLIc, pid, pifn, pifD,
           mbfD, NbfD, A2, A7, epf0, Feng, Peng, dh_radiator, dp_radiator)

    sol = root(lambda x: _off_design_residual(x, ctx, iPspec),
               [pf, mf, Mi], method="hybr", tol=1e-14)
    res, out = _off_design_residual(sol.x, ctx, iPspec, store=True)
    if max(abs(r) for r in res) > 1.0e-10:
        raise ValueError(
            f"ducted fan off-design solve did not converge (iPspec="
            f"{iPspec}): residuals {[f'{r:.3g}' for r in res]} at "
            f"pif={sol.x[0]:.6g}, mbf={sol.x[1]:.6g}, M2={sol.x[2]:.6g}")
    return out
