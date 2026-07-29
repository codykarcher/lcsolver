"""Off-design mission performance -- ``woper.f``.

Flies an aircraft that has *already been sized* on a different mission: a
different range, a different payload, or both. The geometry, the structure and
the engine are fixed; what is solved for is the fuel, and hence the takeoff
weight, that closes the new mission.

Where ``wsize`` iterates on weight *and* resizes everything each pass, this
iterates on weight alone. One fixed point, three or four passes:

1. Rescale the start-of-cruise altitude so the pressure follows the weight
   (``pralt``), since a lighter aircraft cruises higher.
2. Fly it (``mission``), which returns the fuel burnt.
3. Test the takeoff weight against the last two iterations.

Then the static and rotation engine points and ``takeoff``, as ``wsize`` does.

The design mission comes in as ``parad``/``pared`` and is used three ways: as
the initial state for every point, as the reference the weight fractions are
scaled from, and -- through ``pralt`` -- as the pressure the new cruise
altitude is scaled against. The starting fuel guess is Breguet with the range
ratio, ``R ~ ln(1 + f)``, taken from the design mission's own fuel fraction
rather than from an assumed L/D.

Things in the source worth knowing
----------------------------------
* ``rlx`` is computed at the top of the loop -- 1.0, dropping to 0.5 for the
  last five iterations -- and then never used. There is no under-relaxation.
* Two blocks are dead. ``pare`` is copied from ``pared`` at the top for every
  point, and then copied again inside ``if(initeng.eq.0)``, which can only
  produce the same array. And ``para(iaCfnace)`` is copied from the design
  mission over ``ipstatic..ipdescentn``, then unconditionally overwritten with
  a flat 0.003 at every point a few lines later.
* Like ``wsize``, a non-converged run is not an error: the ``return`` after
  the warning is commented out.
* The fuselage BL is solved once, before the loop, exactly as in ``wsize``.

Verified against the compiled Fortran; see ``tests/test_woper.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..aero.fusebl import fusebl
from ..aero.moment import surfcm
from ..engine.tfcalc import tfcalc
from ..model import indices as I
from .mission import MissionResult, mission
from .takeoff import TakeoffResult, takeoff
from .wsize import FT_M, GEE, LB_N, TOLERW, pralt

__all__ = ["WOperResult", "woper", "NACELLE_CF_GUESS"]

#: Flat nacelle wetted-area Cf the loop starts from, overwriting whatever the
#: design mission had.
NACELLE_CF_GUESS = 0.003


@dataclass
class WOperResult:
    converged: bool
    iterations: int
    errw: float
    mission: MissionResult = None
    takeoff: TakeoffResult = None
    history: list = field(default_factory=list)


def woper(pari, parg, parm, para, pare, parad, pared, *,
          iterfmax: int, initeng: int, table=None,
          Litprint: bool = False) -> WOperResult:
    """Converge one off-design mission on an already-sized aircraft.

    ``parad``/``pared`` are the *design* mission's aero and engine matrices,
    read only. ``parm`` carries this mission's range and payload on the way in
    and its takeoff weight and fuel on the way out.
    """
    # Start every point from the design mission.
    for ip in range(1, I.IPTOTAL + 1):
        for ia in range(1, I.IATOTAL + 1):
            para[ia, ip] = parad[ia, ip]
        for ie in range(1, I.IETOTAL + 1):
            pare[ie, ip] = pared[ie, ip]
        # The mission-varying excrescence factors are commented out here and
        # in getparm.f, so they stay at the design mission's values.

    ip = I.IPCRUISE1
    col = para.column(ip)
    fusebl(pari, parg, col)
    para.set_column(ip, col)

    # Assume the K.E., dissipation and drag areas hold at every point.
    KAfTE = para[I.IAKAFTE, ip]
    DAfsurf = para[I.IADAFSURF, ip]
    DAfwake = para[I.IADAFWAKE, ip]
    PAfinf = para[I.IAPAFINF, ip]
    for jp in range(1, I.IPTOTAL + 1):
        para[I.IAKAFTE, jp] = KAfTE
        para[I.IADAFSURF, jp] = DAfsurf
        para[I.IADAFWAKE, jp] = DAfwake
        para[I.IAPAFINF, jp] = PAfinf

    Rangemax = parg[I.IGRANGE]
    Rangetot = parm[I.IMRANGE]
    WMTO = parg[I.IGWMTO]

    # Zero-fuel weight for *this* mission's payload.
    Wzero = WMTO - parg[I.IGWFUEL] - parg[I.IGWPAY] + parm[I.IMWPAY]

    # Breguet with R ~ ln(1 + f), scaled off the design mission's own fuel
    # fraction rather than an assumed L/D.
    gmax = math.log(1.0 + parg[I.IGWFUEL] / Wzero)
    gmaxp = gmax * Rangetot / Rangemax
    Wfuel = (math.exp(gmaxp) - 1.0) * Wzero
    WTO = Wzero + Wfuel
    parm[I.IMWFUEL] = Wfuel
    parm[I.IMWTO] = WTO

    # Scale the weight fractions by the takeoff and descent weight ratios.
    rTO = WTO / WMTO
    rDE = Wzero / (WMTO - parg[I.IGWFUEL])
    for jp in (I.IPSTATIC, I.IPROTATE, I.IPTAKEOFF, I.IPCUTBACK):
        para[I.IAFRACW, jp] = parad[I.IAFRACW, jp] * rTO
    for jp in range(I.IPCLIMB1, I.IPCLIMBN + 1):
        para[I.IAFRACW, jp] = parad[I.IAFRACW, jp] * rTO
    for jp in range(I.IPCRUISE1, I.IPCRUISEN + 1):
        frac = float(jp - I.IPCRUISE1) / float(I.IPCRUISEN - I.IPCRUISE1)
        rCR = rTO * (1.0 - frac) + rDE * frac
        para[I.IAFRACW, jp] = parad[I.IAFRACW, jp] * rCR
    for jp in range(I.IPDESCENT1, I.IPDESCENTN + 1):
        para[I.IAFRACW, jp] = parad[I.IAFRACW, jp] * rDE

    for jp in range(1, I.IPTOTAL + 1):
        para[I.IAGAMV, jp] = parad[I.IAGAMV, jp]

    # Takeoff speed and V, Re over climb and descent -- needed to start the
    # trajectory integration.
    ip = I.IPTAKEOFF
    VTO = (pared[I.IEU0, ip]
           * math.sqrt(pared[I.IERHO0, ip] / pare[I.IERHO0, ip]))
    ReTO = VTO * pare[I.IERHO0, ip] / pare[I.IEMU0, ip]

    ip = I.IPCRUISE1
    VCR = pared[I.IEU0, ip]
    ReCR = parad[I.IAREUNIT, ip]

    for jp in range(I.IPROTATE, I.IPCLIMB1 + 1):
        pare[I.IEU0, jp] = VTO
        para[I.IAREUNIT, jp] = ReTO
    for jp in range(I.IPCLIMB1 + 1, I.IPCLIMBN + 1):
        frac = float(jp - I.IPCLIMB1) / float(I.IPCLIMBN - I.IPCLIMB1)
        pare[I.IEU0, jp] = VTO * (1.0 - frac) + VCR * frac
        para[I.IAREUNIT, jp] = ReTO * (1.0 - frac) + ReCR * frac
    for jp in range(I.IPDESCENT1, I.IPDESCENTN + 1):
        frac = float(jp - I.IPDESCENT1) / float(I.IPDESCENTN - I.IPDESCENT1)
        pare[I.IEU0, jp] = VTO * frac + VCR * (1.0 - frac)
        para[I.IAREUNIT, jp] = ReTO * frac + ReCR * (1.0 - frac)

    # `if(initeng.eq.0)` re-copies pared into pare here. pare already *is*
    # pared, from the block at the top, so the branch cannot change anything.

    # Likewise this copy of the design Cfnace is overwritten below.
    for jp in range(I.IPSTATIC, I.IPDESCENTN + 1):
        para[I.IACFNACE, jp] = parad[I.IACFNACE, jp]

    # ---- wing and tail pitching moment constants -------------------------
    b = parg[I.IGB]
    bs = parg[I.IGBS]
    bo = parg[I.IGBO]
    sweep = parg[I.IGSWEEP]
    Xaxis = parg[I.IGXAXIS]
    lambdas = parg[I.IGLAMBDAS]
    lambdat = parg[I.IGLAMBDAT]
    AR = parg[I.IGAR]
    fLo = parg[I.IGFLO]
    fLt = parg[I.IGFLT]

    for ip_src, lo, hi in ((I.IPTAKEOFF, I.IPSTATIC, I.IPCLIMB1),
                           (I.IPCRUISE1, I.IPCLIMB1 + 1, I.IPDESCENTN - 1),
                           (I.IPDESCENTN, I.IPDESCENTN, I.IPDESCENTN)):
        gammat = lambdat * para[I.IARCLT, ip_src]
        gammas = lambdas * para[I.IARCLS, ip_src]
        m = surfcm(b=b, bs=bs, bo=bo, sweep=sweep, Xaxis=Xaxis,
                   lambdat=lambdat, lambdas=lambdas,
                   gammat=gammat, gammas=gammas,
                   AR=AR, fLo=fLo, fLt=fLt,
                   cmpo=para[I.IACMPO, ip_src],
                   cmps=para[I.IACMPS, ip_src],
                   cmpt=para[I.IACMPT, ip_src])
        for jp in range(lo, hi + 1):
            para[I.IACMW0, jp] = m.CM0
            para[I.IACMW1, jp] = m.CM1

    bh = parg[I.IGBH]
    boh = parg[I.IGBOH]
    mh = surfcm(b=bh, bs=boh, bo=boh, sweep=parg[I.IGSWEEPH], Xaxis=Xaxis,
                lambdat=parg[I.IGLAMBDAH], lambdas=1.0,
                gammat=parg[I.IGLAMBDAH], gammas=1.0,
                AR=parg[I.IGARH], fLo=0.0, fLt=fLt,
                cmpo=0.0, cmps=0.0, cmpt=0.0)
    for jp in range(I.IPSTATIC, I.IPDESCENTN + 1):
        para[I.IACMH0, jp] = mh.CM0
        para[I.IACMH1, jp] = mh.CM1

    # Flat initial guess for the nacelle wetted-area Cf, at every point --
    # this is what makes the copy from the design mission above dead.
    for jp in range(1, I.IPTOTAL + 1):
        para[I.IACFNACE, jp] = NACELLE_CF_GUESS

    WTO1 = WTO2 = 0.0
    Lconv = False
    errw = 1.0
    history = []
    m_result = None

    iterw = 0
    for iterw in range(1, iterfmax + 1):
        # rlx is set here in the source -- 1.0, and 0.5 for the last five
        # iterations -- and never read. There is no under-relaxation.

        # Rescale the start-of-cruise altitude so pressure follows weight.
        ip = I.IPCRUISE1
        altkm = para[I.IAALT, ip] / 1000.0
        p0new = (pared[I.IEP0, ip] * para[I.IAFRACW, ip]
                 / parad[I.IAFRACW, ip])
        altkm, T0, p0, rho0, a0, mu0 = pralt(p0new, GEE, altkm)
        para[I.IAALT, ip] = altkm * 1000.0
        Mach = para[I.IAMACH, ip]
        para[I.IAREUNIT, ip] = Mach * a0 * rho0 / mu0

        ipc1 = 0        # the cruise1 point still has to be computed
        m_result = mission(pari, parg, parm, para, pare, table, initeng,
                           ipc1)

        # CDwing has to be defined at rotation in case there is wing BLI.
        ip = I.IPROTATE
        cdfw = para[I.IACDFW, ip] * para[I.IAFEXCDW, ip]
        cdpw = para[I.IACDPW, ip] * para[I.IAFEXCDW, ip]
        cosL = math.cos(parg[I.IGSWEEP] * math.pi / 180.0)
        para[I.IACDWING, ip] = cdfw + cdpw * cosL ** 3

        # Max error over the last two iterations, so one near-zero change
        # cannot fake convergence.
        WTO = parm[I.IMWTO]
        errw1 = (WTO - WTO1) / WTO
        errw2 = (WTO - WTO2) / WTO
        errw = max(abs(errw1), abs(errw2))

        row = (iterw, errw2,
               parm[I.IMWTO] * LB_N, parm[I.IMWFUEL] * LB_N,
               para[I.IAALT, I.IPCRUISE1] * FT_M,
               para[I.IAALT, I.IPCRUISEN] * FT_M,
               para[I.IAGAMV, I.IPCLIMB1] * 180.0 / math.pi,
               para[I.IAGAMV, I.IPCLIMBN] * 180.0 / math.pi)
        history.append(row)
        if Litprint:
            if iterw == 1:
                print()
                print("  iterw     errW     "
                      "      WTO        Wfuel   "
                      "    h_CR1     h_CR2    "
                      "   gam_BOC   gam_TOC   ")
            print("%6d%14.10f%13.4f%13.4f%11.2f%11.2f%11.5f%11.5f" % row)

        if errw < TOLERW:
            Lconv = True
            break

        WTO1 = WTO2
        WTO2 = parm[I.IMWTO]

    if not Lconv:
        # Printed, then execution carries on -- the `return` is commented out.
        print(f"WOPER: Weight iteration not converged.  dWrel = {errw}")

    for ip in (I.IPSTATIC, I.IPROTATE):
        acol = para.column(ip)
        ecol = pare.column(ip)
        tfcalc(pari, parg, acol, ecol, ip, 1, 1, initeng)
        para.set_column(ip, acol)
        pare.set_column(ip, ecol)

    t_result = takeoff(pari, parg, parm, para, pare, table)

    return WOperResult(converged=Lconv, iterations=iterw, errw=errw,
                       mission=m_result, takeoff=t_result, history=history)
