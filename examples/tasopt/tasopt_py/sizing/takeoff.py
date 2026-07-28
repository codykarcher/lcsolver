"""Takeoff run and balanced field length -- a port of ``takeoff.f``.

Computes the normal takeoff distance and time, the initial climb angle, and
the balanced field length: the runway length at which aborting and continuing
after an engine failure cost exactly the same distance.

The velocity model
------------------
Thrust falls off with speed as ``F = F0 - KV V^2/2``, fitted through the
static and rotation points -- so a two-point fit, not a map lookup. With
rolling friction and drag, the ground-roll equation integrates in closed form
to

    ``V^2 = Vlim^2 (1 - exp(-k l))``

where ``Vlim`` is the speed at which thrust, friction and drag balance and
``k`` is a decay constant. Every segment below is one of these, with its own
``k`` and ``Vlim``:

===  ==================================================================
A    all engines, ground roll
B    one engine out, ground roll continuing
C    braking, all engines idle, spoilers out, max braking friction
===  ==================================================================

``VClimsq`` is negative -- braking has no equilibrium speed -- which is
correct: the same closed form runs the deceleration backwards.

The balanced field
------------------
Two unknowns, the decision distance ``l1`` and the field length ``lBF``, from
two conditions: continuing on the remaining engines reaches ``V2`` by ``lBF``,
and braking from the same point stops by ``lBF``. Solved by a 2x2 Newton in 15
iterations; if it fails the source *prints a warning and carries on with
whatever it has*, which is reproduced here as a flag on the result rather than
an exception, since the caller is a sizing loop that would otherwise abort.

Fallbacks when takeoff is impossible
------------------------------------
If ``VAlim <= V2`` the aircraft cannot reach V2 at all, and the source
substitutes ``lTO = 10/kA`` -- a made-up number that is simply large. If
``VBlim <= V2`` the engine-out case is impossible and it substitutes
``lBF = 10*lTO`` and ``V1 = V2 = Vstall``. Both are there so an optimiser
walking through an infeasible region gets a large finite penalty instead of a
crash, and both are flagged on the result.

Verified against the compiled Fortran; see ``tests/test_takeoff.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..aero.cdsum import cdsum
from ..model import indices as I

__all__ = ["TakeoffResult", "takeoff", "CD_IVERT"]

TOLER = 1.0e-7
ITMAX = 15
GEE = 9.81
#: Drag of the deflected rudder needed to hold heading with one engine out.
#: A bare constant in the source, not derived and not settable from input.
CD_IVERT = 0.002


@dataclass(frozen=True)
class TakeoffResult:
    V1: float          # decision speed
    V2: float          # takeoff safety speed
    lTO: float         # normal takeoff distance
    l1: float          # distance to the decision point
    lBF: float         # balanced field length
    tTO: float         # normal takeoff time
    FTO: float         # max static thrust, all engines
    gamVTO: float      # normal climbout angle
    gamVBF: float      # engine-out climbout angle
    WfTO: float        # fuel burned during the run
    normal_takeoff_possible: bool = True
    engine_out_takeoff_possible: bool = True
    bfl_converged: bool = True


def takeoff(pari, parg, parm, para, pare, table=None) -> TakeoffResult:
    """Takeoff performance. ``para``/``pare`` are the full mission matrices.

    The ``ipstatic`` and ``iprotate`` points must already hold engine states;
    this routine does not run the engine, it reads the thrust already there.
    """
    W = parm[I.IMWTO]
    S = parg[I.IGS]
    dfan = parg[I.IGDFAN]
    HTRf = parg[I.IGHTRF]
    neng = parg[I.IGNENG]
    muroll = parg[I.IGMUROLL]
    mubrake = parg[I.IGMUBRAKE]

    Vstall = pare[I.IEU0, I.IPROTATE]
    V2 = pare[I.IEU0, I.IPTAKEOFF]

    CDgear = parg[I.IGCDGEAR]
    CDeng = parg[I.IGCDEFAN] * (0.25 * math.pi * dfan ** 2) / S
    CDspoiler = parg[I.IGCDSPOIL]

    # Two-point thrust fit through the static and rotation points.
    Fmax = pare[I.IEFE, I.IPSTATIC]
    Fref = pare[I.IEFE, I.IPROTATE]
    F01 = (Fmax + Fref) / 2.0
    KV1 = (Fmax - Fref) / Vstall ** 2
    FTO = Fmax * neng

    ip = I.IPROTATE
    rho0 = pare[I.IERHO0, ip]
    para[I.IACL, ip] = 0.0        # wings unloaded during the roll
    para[I.IACLH, ip] = 0.0

    para_roll = para.column(ip)
    pare_roll = pare.column(ip)
    cdsum(pari, parg, para_roll, pare_roll, 0, table)
    para.set_column(ip, para_roll)
    CDroll = para[I.IACD, ip] + CDgear

    # --- segment A: all engines, ground roll ------------------------------
    KV = KV1 * neng
    F0 = F01 * neng
    kA = (KV + rho0 * S * CDroll) / (W / GEE)
    VAlimsq = 2.0 * (F0 - W * muroll) / (KV + rho0 * S * CDroll)
    VAlim = math.sqrt(VAlimsq)

    normal_ok = VAlim > V2
    if not normal_ok:
        # Cannot reach V2 at all; substitute a large finite distance.
        lTO = 10.0 / kA
        tTO = 2.0 * lTO / V2
    else:
        Vrat = V2 / VAlim
        lTO = -math.log(1.0 - Vrat ** 2) / kA
        tTO = math.log((1.0 + Vrat) / (1.0 - Vrat)) / (kA * VAlim)

    CDclimb = para[I.IACD, I.IPCLIMB1]
    F2 = F0 - KV * 0.5 * V2 ** 2
    singTO = (F2 - 0.5 * V2 ** 2 * rho0 * S * CDclimb) / W
    singTO = max(0.01, min(0.99, singTO))

    # --- segments B and C: engine out ------------------------------------
    CDb = CDroll + CDeng + CD_IVERT
    CDc = CDroll + CDeng * neng + CDspoiler
    KV = KV1 * (neng - 1.0)
    F0 = F01 * (neng - 1.0)
    kB = (KV + rho0 * S * CDb) / (W / GEE)
    kC = (rho0 * S * CDc) / (W / GEE)
    VBlimsq = 2.0 * (F0 - W * muroll) / (KV + rho0 * S * CDb)
    # Negative: braking has no equilibrium speed, the same closed form just
    # runs the deceleration the other way.
    VClimsq = 2.0 * (-W * mubrake) / (rho0 * S * CDc)
    VBlim = math.sqrt(VBlimsq) if VBlimsq > 0.0 else 0.0

    engine_out_ok = VBlim > V2
    converged = True
    if not engine_out_ok:
        l1 = 0.7 * lTO
        lBF = 10.0 * lTO
        V1 = Vstall
        V2 = Vstall
        singBF = 0.0
    else:
        l1 = 0.8 * lTO
        lBF = 1.3 * lTO
        V2sq = V2 ** 2
        converged = False
        dl1 = dlBF = 0.0
        for _ in range(ITMAX):
            exA = math.exp(kA * (-l1))
            exB = math.exp(kB * (lBF - l1))
            exC = math.exp(kC * (lBF - l1))
            exA_l1, exB_l1, exC_l1 = -kA * exA, -kB * exB, -kC * exC
            exA_lBF, exB_lBF, exC_lBF = 0.0, kB * exB, kC * exC

            # Continue on the remaining engines and reach V2 by lBF.
            r1 = (VAlimsq * (1.0 - exA) + (VBlimsq - V2sq) * exB - VBlimsq)
            a11 = VAlimsq * (-exA_l1) + (VBlimsq - V2sq) * exB_l1
            a12 = VAlimsq * (-exA_lBF) + (VBlimsq - V2sq) * exB_lBF
            # Or brake from the same point and stop by lBF.
            r2 = VAlimsq * (1.0 - exA) - VClimsq * (1.0 - exC)
            a21 = VAlimsq * (-exA_l1) - VClimsq * (-exC_l1)
            a22 = VAlimsq * (-exA_lBF) - VClimsq * (-exC_lBF)

            det = a11 * a22 - a12 * a21
            dl1 = -(r1 * a22 - a12 * r2) / det
            dlBF = -(a11 * r2 - r1 * a21) / det
            l1 += dl1
            lBF += dlBF
            if max(abs(dl1), abs(dlBF)) < TOLER * lTO:
                converged = True
                break
        # The source prints a warning and continues; so do we, via the flag.

        V1 = VAlim * math.sqrt(1.0 - math.exp(-kA * l1))
        F2 = F0 - KV * 0.5 * V2 ** 2
        CD = CDclimb + CDgear + CDeng
        singBF = (F2 - 0.5 * V2 ** 2 * rho0 * S * CD) / W
        singBF = max(0.01, min(0.99, singBF))

    # --- fuel burned during the run --------------------------------------
    mdotf1 = pare[I.IEMCORE, I.IPSTATIC] * pare[I.IEFF, I.IPSTATIC] * neng
    mdotf2 = pare[I.IEMCORE, I.IPROTATE] * pare[I.IEFF, I.IPROTATE] * neng
    WfTO = 0.5 * (mdotf1 + mdotf2) * tTO * GEE

    para[I.IAFRACW, I.IPSTATIC] = (para[I.IAFRACW, I.IPTAKEOFF]
                                   + WfTO / parg[I.IGWMTO])
    para[I.IATIME, I.IPSTATIC] = para[I.IATIME, I.IPTAKEOFF] - tTO
    para[I.IARANGE, I.IPSTATIC] = para[I.IARANGE, I.IPTAKEOFF] - lTO

    parm[I.IMV1] = V1
    parm[I.IMV2] = V2
    parm[I.IMLTO] = lTO
    parm[I.IML1] = l1
    parm[I.IMLBF] = lBF
    parm[I.IMTTO] = tTO
    parm[I.IMFTO] = FTO
    parm[I.IMGAMVTO] = math.asin(singTO)
    parm[I.IMGAMVBF] = math.asin(singBF)

    return TakeoffResult(V1=V1, V2=V2, lTO=lTO, l1=l1, lBF=lBF, tTO=tTO,
                         FTO=FTO, gamVTO=math.asin(singTO),
                         gamVBF=math.asin(singBF), WfTO=WfTO,
                         normal_takeoff_possible=normal_ok,
                         engine_out_takeoff_possible=engine_out_ok,
                         bfl_converged=converged)
