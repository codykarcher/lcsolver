"""The outer sizing loop -- ``wsize.f``.

Sizes an aircraft to fly its design mission. This is the routine every other
ported module exists to serve: it fixes the geometry, weighs the structure,
sizes and weighs the engine, flies the mission, and iterates until the maximum
takeoff weight stops moving.

The loop
--------
The fixed point is on **weight**. Each pass, in order:

1. **Fuselage** -- ``fusew`` sizes the shell for the cabin overpressure at the
   end of cruise-climb and the tailcone for the maneuver tail loads.
2. **Weight update** -- ``Wupdate0`` re-solves ``WMTO`` from the weight
   *fractions*, under-relaxed.
3. **Convergence test** -- on the largest relative ``WMTO`` change over the
   last three iterations, not just the last one, so a single near-zero step
   cannot fake convergence.
4. **Wing** -- ``wingsc`` sets the area and chords to carry the start-of-cruise
   weight, then ``surfw`` sizes the box and weighs it.
5. **Tails** -- horizontal tail from a volume coefficient for the first two
   passes, then from ``htsize``, which sizes it and locates the wing at the
   same time; vertical tail from a volume coefficient or from the engine-out
   yaw moment.
6. **Engine** -- trim, ``cdsum``, then ``tfcalc`` sizes the engine for the
   cruise thrust and ``tfweight`` weighs it.
7. **Mission** -- ``mission`` flies it and returns the fuel burnt.
8. **Cooling** -- ``tfcalc`` again at takeoff rotation, to set the turbine
   cooling flows the whole mission then uses.
9. **Weight update** again, and round.

After the loop: static and rotation engine points, ``takeoff``, the CG limits
from ``cglpay``, and the cruise neutral point.

The fuselage boundary layer is **not** in the loop
--------------------------------------------------
``fusebl`` is called once, before it, and its four areas are copied to every
mission point. The fuselage geometry is an input and does not change over the
sizing, so there is exactly one BL solve per sizing -- confirmed by
instrumenting the shipped program. It is also the single most expensive call
in the routine, so this matters.

Things in the source worth knowing
----------------------------------
* ``Wupdate0`` hardwires ``fsum = 0.0`` and never updates it, so the
  ``if(fsum .ge. 1.0) go to 110`` guard after it can never fire. Only the
  ``Wupdate`` at the bottom of the loop can detect a diverging weight.
* Non-convergence is **not** an error. The ``return`` after the "not
  converged" message is commented out, so the routine prints and carries on
  into the takeoff and balance calculations regardless. ``converged`` on the
  result says which happened.
* ``ichoke5``/``ichoke7``, the nozzle-choked flags, are outputs of ``wsize``
  in the original but nothing reads them numerically -- the only use is a
  commented-out print. They are not carried here.
* ``pralt`` and ``muair`` are defined in ``wsize.f`` but not called from it.
  ``pralt`` is used by ``woper.f`` and is ported below; ``muair`` is used by
  nothing, and would disagree slightly with the atmosphere if it were --
  it takes its reference temperature as 288.0 K where ``atmos`` uses 288.2.
* ``Wupdate1`` is an alternative weight update, kept and ported, but its call
  site in ``Wupdate`` is commented out.

Verified against the compiled Fortran; see ``tests/test_wsize.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..aero.cdsum import cdsum
from ..aero.fusebl import fusebl
from ..aero.loading import wingpo
from ..aero.moment import surfcm
from ..atmosphere import atmos
from ..engine.cooling import mcool
from ..engine.tfcalc import tfcalc
from ..engine.weight import tfweight
from ..model import indices as I
from ..structures.fuselage import fusew
from ..structures.planform import surfdx, tailpo, wingsc
from ..structures.surface import surfw
from .balance import balance, cglpay, htsize
from .mission import MissionResult, mission
from .takeoff import TakeoffResult, takeoff

__all__ = ["WSizeResult", "wsize", "Wupdate", "Wupdate0", "Wupdate1",
           "pralt", "TOLERW", "GEE", "ExcessiveCoolingFlow"]

#: Convergence tolerance on the fractional weight change between iterations.
#: Two tighter values, 1e-10 and 1e-14, are commented out beside it.
TOLERW = 1.0e-9
GEE = 9.81
CPSL = 1004.0
GAMSL = 1.4
LB_N = 1.0 / 4.44822
FT_M = 1.0 / 0.3048


class ExcessiveCoolingFlow(Exception):
    """Raised where ``wsize.f`` prints and calls ``stop``."""


@dataclass
class WSizeResult:
    converged: bool
    iterations: int
    errw: float                       # largest relative WMTO change
    fsum: float                       # weight fractions summed
    mission: MissionResult = None
    takeoff: TakeoffResult = None
    history: list = field(default_factory=list)   # per-iteration print rows
    #: The one fuselage BL solve, kept for the report. Nothing numerical
    #: reads it; see tasopt_py.output.blfwrt.
    fuselage_bl: object = None


# --------------------------------------------------------------------------
# Weight updates
# --------------------------------------------------------------------------

def Wupdate(parg, rlx: float) -> float:
    """Re-solve ``WMTO`` from the weight fractions. Returns ``fsum``.

    Everything but the payload and fuselage is held as a *fraction* of
    ``WMTO``, so the update is a single division by ``1 - fsum``, under-relaxed
    against the previous value. If ``fsum >= 1`` the weight has exploded and
    nothing is written.
    """
    WMTO = parg[I.IGWMTO]

    fwing = parg[I.IGWWING] / WMTO
    fstrut = parg[I.IGWSTRUT] / WMTO
    fhtail = parg[I.IGWHTAIL] / WMTO
    fvtail = parg[I.IGWVTAIL] / WMTO
    feng = parg[I.IGWENG] / WMTO
    ffuel = parg[I.IGWFUEL] / WMTO
    fhpesys = parg[I.IGFHPESYS]
    flgnose = parg[I.IGFLGNOSE]
    flgmain = parg[I.IGFLGMAIN]

    Wpay = parg[I.IGWPAY]
    Wfuse = parg[I.IGWFUSE]

    fsum = (fwing + fstrut + fhtail + fvtail + feng + ffuel
            + fhpesys + flgnose + flgmain)
    if fsum >= 1.0:
        return fsum

    WMTO = rlx * (Wpay + Wfuse) / (1.0 - fsum) + (1.0 - rlx) * WMTO

    parg[I.IGWMTO] = WMTO
    parg[I.IGWWING] = WMTO * fwing
    parg[I.IGWSTRUT] = WMTO * fstrut
    parg[I.IGWHTAIL] = WMTO * fhtail
    parg[I.IGWVTAIL] = WMTO * fvtail
    parg[I.IGWENG] = WMTO * feng
    parg[I.IGWFUEL] = WMTO * ffuel
    return fsum


def Wupdate0(parg, rlx: float) -> float:
    """Sum the weights directly rather than through fractions.

    Always returns ``fsum = 0``: the source sets it to zero and never touches
    it again, so the caller's explosion guard on it is dead. Only the landing
    gear and systems fractions are treated as fractions here.
    """
    WMTO = parg[I.IGWMTO]

    ftotadd = (parg[I.IGFHPESYS] + parg[I.IGFLGNOSE] + parg[I.IGFLGMAIN])

    fsum = 0.0        # set here and never updated; see the docstring

    Wsum = (parg[I.IGWPAY] + parg[I.IGWFUSE] + parg[I.IGWWING]
            + parg[I.IGWSTRUT] + parg[I.IGWHTAIL] + parg[I.IGWVTAIL]
            + parg[I.IGWENG] + parg[I.IGWFUEL])

    parg[I.IGWMTO] = rlx * Wsum / (1.0 - ftotadd) + (1.0 - rlx) * WMTO
    return fsum


def Wupdate1(parg, rlx: float) -> float:
    """An alternative update that breaks the fuselage into its pieces.

    Not used: the call to it at the top of ``Wupdate`` is commented out. It
    differs in holding the tailcone and bending material as fractions and
    rebuilding ``Wfuse`` from its parts afterwards.
    """
    WMTO = parg[I.IGWMTO]

    fwing = parg[I.IGWWING] / WMTO
    fstrut = parg[I.IGWSTRUT] / WMTO
    fhbend = parg[I.IGWHBEND] / WMTO
    fvbend = parg[I.IGWVBEND] / WMTO
    fcone = parg[I.IGWCONE] / WMTO
    fhtail = parg[I.IGWHTAIL] / WMTO
    fvtail = parg[I.IGWVTAIL] / WMTO
    feng = parg[I.IGWENG] / WMTO
    ffuel = parg[I.IGWFUEL] / WMTO

    ftotadd = (parg[I.IGFHPESYS] + parg[I.IGFLGNOSE] + parg[I.IGFLGMAIN])

    Wfix = parg[I.IGWFIX]
    Wpay = parg[I.IGWPAY]
    Wapu = parg[I.IGWPAY] * parg[I.IGFAPU]
    Wpadd = parg[I.IGWPAY] * parg[I.IGFPADD]
    Wseat = parg[I.IGWPAY] * parg[I.IGFSEAT]
    Wshell = parg[I.IGWSHELL]
    Wwindow = parg[I.IGWWINDOW]
    Winsul = parg[I.IGWINSUL]
    Wfloor = parg[I.IGWFLOOR]

    fsum = (fwing + fstrut + fhbend + fvbend + fcone + fhtail + fvtail
            + feng + ffuel + ftotadd)
    if fsum >= 1.0:
        return fsum

    WMTO = (rlx * (Wfix + Wpay + Wpadd + Wapu + Wshell
                   + Wwindow + Winsul + Wfloor + Wseat) / (1.0 - fsum)
            + (1.0 - rlx) * WMTO)

    parg[I.IGWMTO] = WMTO
    parg[I.IGWWING] = WMTO * fwing
    parg[I.IGWSTRUT] = WMTO * fstrut
    parg[I.IGWHBEND] = WMTO * fhbend
    parg[I.IGWVBEND] = WMTO * fvbend
    parg[I.IGWCONE] = WMTO * fcone
    parg[I.IGWHTAIL] = WMTO * fhtail
    parg[I.IGWVTAIL] = WMTO * fvtail
    parg[I.IGWENG] = WMTO * feng
    parg[I.IGWFUEL] = WMTO * ffuel

    parg[I.IGWFUSE] = (Wfix + Wapu + Wpadd + Wseat + Wshell
                       + parg[I.IGWCONE] + Wwindow + Winsul + Wfloor
                       + parg[I.IGWHBEND] + parg[I.IGWVBEND])
    return fsum


def pralt(p0spec: float, gee: float, altkm: float):
    """Standard altitude and atmospheric state at a specified pressure.

    Newton on altitude with the hydrostatic slope. Defined in ``wsize.f`` but
    called only from ``woper.f``. Returns ``(altkm, T0, p0, rho0, a0, mu0)``;
    the source prints and returns on failure rather than raising, and so does
    this.
    """
    a = atmos(altkm)
    for _ in range(15):
        a = atmos(altkm)
        delp = a.p - p0spec
        dpdh = -a.rho * gee
        dh = -delp / dpdh
        if abs(dh) < 0.001:
            return altkm, a.T, a.p, a.rho, a.a, a.mu
        altkm = altkm + dh / 1000.0
    return altkm, a.T, a.p, a.rho, a.a, a.mu


# --------------------------------------------------------------------------

def _atmosphere_at(para, pare, ip, Mach=None):
    """Fill the ambient state at one mission point from its altitude."""
    a = atmos(para[I.IAALT, ip] / 1000.0)
    pare[I.IEP0, ip] = a.p
    pare[I.IET0, ip] = a.T
    pare[I.IEA0, ip] = a.a
    pare[I.IERHO0, ip] = a.rho
    pare[I.IEMU0, ip] = a.mu
    if Mach is None:
        Mach = para[I.IAMACH, ip]
    pare[I.IEM0, ip] = Mach
    pare[I.IEU0, ip] = Mach * a.a
    para[I.IAREUNIT, ip] = Mach * a.a * a.rho / a.mu


def wsize(pari, parg, parm, para, pare, *,
          iterwmax: int, wrlx1: float, wrlx2: float, wrlx3: float,
          initwgt: int, initeng: int, table=None,
          Litprint: bool = False) -> WSizeResult:
    """Size the aircraft for its design mission.

    ``initwgt`` 0 starts from crude internal guesses, 1 from the weights
    already in ``parg`` (a previous converged solution). ``initeng`` 0
    initialises the engine state at each point, 1 reuses what is there.
    ``wrlx1``/``wrlx2``/``wrlx3`` are the under-relaxation factors for the
    first few iterations, the middle, and the last quarter respectively --
    the last is there to quash a limit cycle.

    ``table`` is the airfoil database, injected in place of ``iairf``.

    Everything is written back into ``parg``/``parm``/``para``/``pare``;
    the return value carries only what the Fortran returns through its
    argument list plus the sub-results a caller would otherwise have to dig
    out of the arrays.
    """
    sl = atmos(0.0)
    TSL, pSL, rhoSL = sl.T, sl.p, sl.rho
    RSL = pSL / (rhoSL * TSL)     # constants.inc derives it, it is not 287
    Tref, pref = TSL, pSL

    errw = 1.0
    fsum = 0.0

    iwplan = pari[I.IIWPLAN]
    iengloc = pari[I.IIENGLOC]
    iengwgt = pari[I.IIENGWGT]
    iVTsize = pari[I.IIVTSIZE]

    # ---- fuselage BL, once, at start of cruise -------------------------
    ip = I.IPCRUISE1
    col = para.column(ip)
    fuselage_bl = fusebl(pari, parg, col)
    para.set_column(ip, col)

    # Assume the K.E., dissipation and drag areas are the same at every point.
    KAfTE = para[I.IAKAFTE, ip]
    DAfsurf = para[I.IADAFSURF, ip]
    DAfwake = para[I.IADAFWAKE, ip]
    PAfinf = para[I.IAPAFINF, ip]
    for jp in range(1, I.IPTOTAL + 1):
        para[I.IAKAFTE, jp] = KAfTE
        para[I.IADAFSURF, jp] = DAfsurf
        para[I.IADAFWAKE, jp] = DAfwake
        para[I.IAPAFINF, jp] = PAfinf

    # ---- quantities fixed over the weight iteration ---------------------
    Rangetot = parm[I.IMRANGE]
    Wpay = parm[I.IMWPAY]
    parg[I.IGRANGE] = Rangetot
    parg[I.IGWPAY] = Wpay

    Wfix = parg[I.IGWFIX]
    xfix = parg[I.IGXFIX]

    fapu = parg[I.IGFAPU]
    fpadd = parg[I.IGFPADD]
    fseat = parg[I.IGFSEAT]
    feadd = parg[I.IGFEADD]
    fhadd = parg[I.IGFHADD]
    fvadd = parg[I.IGFVADD]
    fwadd = (parg[I.IGFFLAP] + parg[I.IGFSLAT] + parg[I.IGFAILE]
             + parg[I.IGFLETE] + parg[I.IGFRIBS] + parg[I.IGFSPOI]
             + parg[I.IGFWATT])

    fstring = parg[I.IGFSTRING]
    fframe = parg[I.IGFFRAME]
    ffadd = parg[I.IGFFADD]
    fpylon = parg[I.IGFPYLON]

    fhpesys = parg[I.IGFHPESYS]
    flgnose = parg[I.IGFLGNOSE]
    flgmain = parg[I.IGFLGMAIN]
    freserve = parg[I.IGFRESERVE]

    fLo = parg[I.IGFLO]
    fLt = parg[I.IGFLT]

    Rfuse = parg[I.IGRFUSE]
    dRfuse = parg[I.IGDRFUSE]
    wfb = parg[I.IGWFB]
    nfweb = parg[I.IGNFWEB]
    hfloor = parg[I.IGHFLOOR]
    xnose = parg[I.IGXNOSE]
    xshell1 = parg[I.IGXSHELL1]
    xshell2 = parg[I.IGXSHELL2]
    xconend = parg[I.IGXCONEND]
    xwbox = parg[I.IGXWBOX]
    xhbox = parg[I.IGXHBOX]
    xvbox = parg[I.IGXVBOX]
    xapu = parg[I.IGXAPU]
    xeng = parg[I.IGXENG]

    Wapu = Wpay * fapu
    Wpadd = Wpay * fpadd
    Wseat = Wpay * fseat

    Wpwindow = parg[I.IGWPWINDOW]
    Wppinsul = parg[I.IGWPPINSUL]
    Wppfloor = parg[I.IGWPPFLOOR]

    rMh = parg[I.IGRMH]
    rMv = parg[I.IGRMV]
    CLhmax = parg[I.IGCLHMAX]
    CLvmax = parg[I.IGCLVMAX]

    lambdas = parg[I.IGLAMBDAS]
    lambdat = parg[I.IGLAMBDAT]
    lambdahs = 1.0                      # tails have no inner panel
    lambdah = parg[I.IGLAMBDAH]
    lambdavs = 1.0
    lambdav = parg[I.IGLAMBDAV]
    lambdac = parg[I.IGLAMBDAC]

    sweep = parg[I.IGSWEEP]
    wbox = parg[I.IGWBOX]
    hboxo = parg[I.IGHBOXO]
    hboxs = parg[I.IGHBOXS]
    rh = parg[I.IGRH]
    AR = parg[I.IGAR]
    bo = parg[I.IGBO]
    etas = parg[I.IGETAS]
    Xaxis = parg[I.IGXAXIS]

    sweeph = parg[I.IGSWEEPH]
    wboxh = parg[I.IGWBOXH]
    hboxh = parg[I.IGHBOXH]
    rhh = parg[I.IGRHH]
    ARh = parg[I.IGARH]
    boh = parg[I.IGBOH]

    sweepv = parg[I.IGSWEEPV]
    wboxv = parg[I.IGWBOXV]
    hboxv = parg[I.IGHBOXV]
    rhv = parg[I.IGRHV]
    ARv = parg[I.IGARV]
    bov = parg[I.IGBOV]

    nvtail = parg[I.IGNVTAIL]

    zs = parg[I.IGZS]
    hstrut = parg[I.IGHSTRUT]
    tohstrut = 0.05
    zsh = 0.0                           # no struts on tails
    zsv = 0.0

    Nlift = parg[I.IGNLIFT]
    Nland = parg[I.IGNLAND]

    Vne = parg[I.IGVNE]
    qne = 0.5 * rhoSL * Vne ** 2

    sigfac = parg[I.IGSIGFAC]
    sigcap = parg[I.IGSIGCAP] * sigfac
    tauweb = parg[I.IGTAUWEB] * sigfac
    rhoweb = parg[I.IGRHOWEB]
    rhocap = parg[I.IGRHOCAP]

    sigskin = parg[I.IGSIGSKIN] * sigfac
    sigbend = parg[I.IGSIGBEND] * sigfac
    rhoskin = parg[I.IGRHOSKIN]
    rhobend = parg[I.IGRHOBEND]
    rEshell = parg[I.IGRESHELL]

    sigstrut = parg[I.IGSIGSTRUT] * sigfac
    rhostrut = parg[I.IGRHOSTRUT]

    # Tail stresses and densities are taken as the wing's, "keeps it simpler".
    sigcaph, tauwebh, rhowebh, rhocaph = sigcap, tauweb, rhoweb, rhocap
    sigcapv, tauwebv, rhowebv, rhocapv = sigcap, tauweb, rhoweb, rhocap

    neng = parg[I.IGNENG]
    yeng = parg[I.IGYENG]
    rSnace = parg[I.IGRSNACE]

    _atmosphere_at(para, pare, I.IPCRUISE1)
    _atmosphere_at(para, pare, I.IPROTATE, Mach=0.25)

    # =====================================================================
    # Initial guesses. None of these affect the converged answer.
    # =====================================================================
    cbox = 0.0
    bv = 0.0
    if initwgt == 0:
        Whtail = 0.05 * Wpay / sigfac
        Wvtail = 0.05 * Wpay / sigfac
        Wwing = 0.5 * Wpay / sigfac
        Wstrut = 0.0 * Wpay / sigfac
        Weng = 0.3 * Wpay
        feng = 0.08

        ip = I.IPCRUISE1
        W = 5.0 * Wpay
        S = W / (0.5 * pare[I.IERHO0, ip] * pare[I.IEU0, ip] ** 2
                 * para[I.IACL, ip])
        b = math.sqrt(S * AR)
        bs = b * etas
        Winn = 0.15 * Wpay / sigfac
        Wout = 0.05 * Wpay / sigfac
        dyWinn = Winn * 0.30 * (0.5 * (bs - bo))
        dyWout = Wout * 0.25 * (0.5 * (b - bs))

        parg[I.IGWHTAIL] = Whtail
        parg[I.IGWVTAIL] = Wvtail
        parg[I.IGWWING] = Wwing
        parg[I.IGWSTRUT] = Wstrut
        parg[I.IGWENG] = Weng
        parg[I.IGWINN] = Winn
        parg[I.IGWOUT] = Wout
        parg[I.IGDXWHTAIL] = 0.0
        parg[I.IGDXWVTAIL] = 0.0
        parg[I.IGDYWINN] = dyWinn
        parg[I.IGDYWOUT] = dyWout

        c = surfdx(b, bs, bo, lambdat, lambdas, sweep)
        xwing = xwbox + c.dx
        parg[I.IGXWING] = xwing

        # No sweep offset assumed for the tails initially.
        parg[I.IGXHTAIL] = xhbox
        parg[I.IGXVTAIL] = xvbox

        cbox = 0.0
        parg[I.IGLNACE] = 0.5 * S / b
        fSnace = 0.2
        parg[I.IGFSNACE] = fSnace

        # Fuel fraction from Breguet, with a flat L/D and TSFC.
        DoL = 1.0 / 18.0
        TSFC = 1.0 / 7000.0
        V = pare[I.IEU0, I.IPCRUISE1]
        ffburn = 1.0 - math.exp(-Rangetot * DoL * TSFC / V)
        ffburn = min(ffburn, 0.8 / (1.0 + freserve))

        ffuelb = ffburn * (1.0 + freserve)      # start of climb
        ffuelc = ffburn * (0.90 + freserve)     # start of cruise
        ffueld = ffburn * (0.02 + freserve)     # start of descent
        ffuele = ffburn * (0.0 + freserve)      # landing
        ffuel = ffuelb                          # max is at start of climb

        # Clear the climb angles, to force initial guesses in mission.
        for jp in range(1, I.IPTOTAL + 1):
            para[I.IAGAMV, jp] = 0.0

        for jp in (I.IPSTATIC, I.IPROTATE, I.IPTAKEOFF, I.IPCUTBACK):
            para[I.IAFRACW, jp] = 1.0

        for jp in range(I.IPCLIMB1, I.IPCLIMBN + 1):
            frac = float(jp - I.IPCLIMB1) / float(I.IPCLIMBN - I.IPCLIMB1)
            ffp = ffuelb * (1.0 - frac) + ffuelc * frac
            para[I.IAFRACW, jp] = 1.0 - ffuel + ffp

        for jp in range(I.IPCRUISE1, I.IPCRUISEN + 1):
            frac = float(jp - I.IPCRUISE1) / float(I.IPCRUISEN - I.IPCRUISE1)
            ffp = ffuelc * (1.0 - frac) + ffueld * frac
            para[I.IAFRACW, jp] = 1.0 - ffuel + ffp

        for jp in range(I.IPDESCENT1, I.IPDESCENTN + 1):
            frac = (float(jp - I.IPDESCENT1)
                    / float(I.IPDESCENTN - I.IPDESCENT1))
            ffp = ffueld * (1.0 - frac) + ffuele * frac
            para[I.IAFRACW, jp] = 1.0 - ffuel + ffp

        # Tail areas, needed for the initial fuselage bending material.
        Sh = (2.0 * Wpay) / (qne * CLhmax)
        Sv = (2.0 * Wpay) / (qne * CLvmax)
        bv = math.sqrt(Sv * ARv)
        parg[I.IGSH] = Sh
        parg[I.IGSV] = Sv

        for jp in range(1, I.IPTOTAL + 1):
            para[I.IACMW0, jp] = 0.0
            para[I.IACMW1, jp] = 0.0
            para[I.IACMH0, jp] = 0.0
            para[I.IACMH1, jp] = 0.0
            para[I.IACLH, jp] = 0.0

        # A cruise-climb angle is needed to guess the end-of-cruise altitude,
        # which sets the cabin deltap that sizes the shell.
        LoD = 18.0
        gamVcr = 0.0002
        para[I.IACD, I.IPCRUISE1] = para[I.IACL, I.IPCRUISE1] / LoD
        para[I.IAGAMV, I.IPCRUISE1] = gamVcr

        p0c = pare[I.IEP0, I.IPCRUISE1]
        # End-of-cruise pressure scales with weight.
        p0d = p0c * (1.0 - ffuel + ffueld) / (1.0 - ffuel + ffuelc)
        pare[I.IEP0, I.IPCRUISEN] = p0d

        # Single-engine thrust, speed and fan area, for engine-out VT sizing.
        pare[I.IEFE, I.IPROTATE] = 2.0 * Wpay / neng
        pare[I.IEU0, I.IPROTATE] = 70.0
        Afan = 3.0e-5 * Wpay / neng
        parg[I.IGDFAN] = math.sqrt(Afan * 4.0 / math.pi)

        # Fan-face Mach numbers, for the nacelle CD.
        M2des = pare[I.IEM2, I.IPCRUISE1]
        for jp in range(I.IPSTATIC, I.IPCRUISEN + 1):
            pare[I.IEM2, jp] = M2des
        for jp in range(I.IPDESCENT1, I.IPDESCENTN + 1):
            pare[I.IEM2, jp] = 0.8 * M2des

        # ---- initial turbine cooling mass flow ratios --------------------
        ip = I.IPROTATE
        cpc = 1080.0        # average compressor cp; 1025.0 is commented out
        cp4 = 1340.0
        Rgc = 288.0         # average compressor R
        Rg4 = 288.0
        M0to = pare[I.IEU0, ip] / pare[I.IEA0, ip]
        T0to = pare[I.IET0, ip]
        epolhc = pare[I.IEEPOLHC, ip]
        OPRto = (pare[I.IEPILC, I.IPCRUISE1]
                 * pare[I.IEPIHC, I.IPCRUISE1])
        Tt4to = pare[I.IETT4, ip]
        dTstrk = pare[I.IEDTSTRK, ip]
        Mtexit = pare[I.IEMTEXIT, ip]
        efilm = pare[I.IEEFILM, ip]
        tfilm = pare[I.IETFILM, ip]
        StA = pare[I.IESTA, ip]
        Tmrow = [parg[I.IGTMETAL]] * I.NCROWX

        Tt2to = T0to * (1.0 + 0.5 * (GAMSL - 1.0) * M0to ** 2)
        Tt3to = Tt2to * OPRto ** (Rgc / (epolhc * cpc))
        Trrat = 1.0 / (1.0 + 0.5 * Rg4 / (cp4 - Rg4) * Mtexit ** 2)
        cf = mcool(Tmrow, Tt3to, Tt4to, dTstrk, Trrat, efilm, tfilm, StA)
        epstot = sum(cf.epsrow[:cf.ncrow])
        fo = pare[I.IEMOFFT, ip] / pare[I.IEMCORE, ip]
        fc = (1.0 - fo) * epstot

        if fc >= 0.99:
            raise ExcessiveCoolingFlow(
                f"WSIZE: Excessive cooling flow. mcool/mcore = {fc}, "
                f"Tt3 Tt4 Tmetal = {Tt3to} {Tt4to} {parg[I.IGTMETAL]}")

        for jp in range(1, I.IPTOTAL + 1):
            pare[I.IEFC, jp] = fc
            for icrow in range(I.NCROWX):
                pare[I.IEEPSC1 + icrow, jp] = cf.epsrow[icrow]
                pare[I.IETMET1 + icrow, jp] = Tmrow[icrow]

    else:
        # Start from the weights already in parg.
        S = parg[I.IGS]
        b = parg[I.IGB]
        bs = parg[I.IGBS]
        bo = parg[I.IGBO]
        bv = parg[I.IGBV]
        cbox = parg[I.IGCO] * parg[I.IGWBOX]

        Wwing = parg[I.IGWWING]
        Wstrut = parg[I.IGWSTRUT]
        Weng = parg[I.IGWENG]

        WMTO = parg[I.IGWMTO]
        feng = parg[I.IGWENG] / WMTO
        ffuel = parg[I.IGWFUEL] / WMTO

        xwing = parg[I.IGXWING]
        fSnace = parg[I.IGFSNACE]
        Sh = parg[I.IGSH]
        Sv = parg[I.IGSV]
        ARh = parg[I.IGARH]
        ARv = parg[I.IGARV]

    # =====================================================================
    WMTO1 = WMTO2 = WMTO3 = 0.0     # previous iterations, for convergence
    Lconv = False
    exploded = False
    inite1 = 0
    history = []
    m_result = None

    # These start at zero purely for the first-iteration printout.
    parg[I.IGB] = 0.0
    parg[I.IGS] = 0.0

    iterw = 0
    for iterw in range(1, iterwmax + 1):
        # Be cautious for longer when starting from crude guesses.
        itrlx = 5 if initwgt == 0 else 2
        if iterw <= itrlx:
            rlx = wrlx1
        elif iterw >= (3 * iterwmax) // 4:
            # Under-relax near the iteration limit, to quash a limit cycle.
            rlx = wrlx3
        else:
            rlx = wrlx2

        # ---- fuselage sizing --------------------------------------------
        Lhmax = qne * Sh * CLhmax
        Lvmax = qne * Sv * CLvmax / nvtail

        # Cabin overpressure at end of cruise-climb, assuming p ~ W/S.
        wcd = para[I.IAFRACW, I.IPCRUISEN] / para[I.IAFRACW, I.IPCRUISE1]
        deltap = parg[I.IGPCABIN] - pare[I.IEP0, I.IPCRUISE1] * wcd
        parg[I.IGDELTAP] = deltap

        Wengtail = 0.0 if iengloc == 1 else parg[I.IGWENG]

        Whtail = parg[I.IGWHTAIL]
        Wvtail = parg[I.IGWVTAIL]
        xhtail = parg[I.IGXHTAIL]
        xvtail = parg[I.IGXVTAIL]
        xwbox = parg[I.IGXWBOX]
        xwing = parg[I.IGXWING]

        Eskin = parg[I.IGECAP]
        Ebend = Eskin * rEshell
        Gskin = Eskin * 0.5 / (1.0 + 0.3)
        fw = fusew(gee=GEE, Nland=Nland, Wfix=Wfix, Wpay=Wpay, Wpadd=Wpadd,
                   Wseat=Wseat, Wapu=Wapu, Weng=Wengtail,
                   fstring=fstring, fframe=fframe, ffadd=ffadd,
                   deltap=deltap,
                   Wpwindow=Wpwindow, Wppinsul=Wppinsul, Wppfloor=Wppfloor,
                   Whtail=Whtail, Wvtail=Wvtail, rMh=rMh, rMv=rMv,
                   Lhmax=Lhmax, Lvmax=Lvmax,
                   bv=bv, lambdav=lambdav, nvtail=nvtail,
                   Rfuse=Rfuse, dRfuse=dRfuse, wfb=wfb, nfweb=nfweb,
                   lambdac=lambdac,
                   xnose=xnose, xshell1=xshell1, xshell2=xshell2,
                   xconend=xconend, xhtail=xhtail, xvtail=xvtail,
                   xwing=xwing, xwbox=xwbox, cbox=cbox,
                   xfix=xfix, xapu=xapu, xeng=xeng, hfloor=hfloor,
                   sigskin=sigskin, sigbend=sigbend,
                   rhoskin=rhoskin, rhobend=rhobend,
                   Eskin=Eskin, Ebend=Ebend, Gskin=Gskin)

        parg[I.IGTSKIN] = fw.tskin
        parg[I.IGTCONE] = fw.tcone
        parg[I.IGTFWEB] = fw.tfweb
        parg[I.IGTFLOOR] = fw.tfloor
        parg[I.IGXHBEND] = fw.xhbend
        parg[I.IGXVBEND] = fw.xvbend
        parg[I.IGEIHSHELL] = fw.EIhshell
        parg[I.IGEIHBEND] = fw.EIhbend
        parg[I.IGEIVSHELL] = fw.EIvshell
        parg[I.IGEIVBEND] = fw.EIvbend
        parg[I.IGGJSHELL] = fw.GJshell
        parg[I.IGGJCONE] = fw.GJcone
        parg[I.IGWSHELL] = fw.Wshell
        parg[I.IGWCONE] = fw.Wcone
        parg[I.IGWWINDOW] = fw.Wwindow
        parg[I.IGWINSUL] = fw.Winsul
        parg[I.IGWFLOOR] = fw.Wfloor
        parg[I.IGWHBEND] = fw.Whbend
        parg[I.IGWVBEND] = fw.Wvbend
        parg[I.IGWFUSE] = fw.Wfuse
        parg[I.IGXWFUSE] = fw.xWfuse
        parg[I.IGCABVOL] = fw.cabVol

        # Buoyancy, needed to size the wing and engine at cruise.
        ip = I.IPCRUISE1
        rhocab = max(parg[I.IGPCABIN], pare[I.IEP0, ip]) / (RSL * TSL)
        WbuoyCR = (rhocab - pare[I.IERHO0, ip]) * GEE * fw.cabVol

        # ---- total max takeoff weight ------------------------------------
        if iterw == 1 and initwgt == 0:
            fsum = feng + ffuel + fhpesys + flgnose + flgmain
            WMTO = ((Wpay + fw.Wfuse + Wwing + Wstrut + Whtail + Wvtail)
                    / (1.0 - fsum))
            parg[I.IGWMTO] = WMTO
            parg[I.IGWENG] = WMTO * feng
            parg[I.IGWFUEL] = WMTO * ffuel
        else:
            fsum = Wupdate0(parg, rlx)
            # This guard is dead -- Wupdate0 always reports fsum = 0.
            if fsum >= 1.0:
                exploded = True
                break
            parm[I.IMWTO] = parg[I.IGWMTO]
            parm[I.IMWFUEL] = parg[I.IGWFUEL]

        parm[I.IMWTO] = parg[I.IGWMTO]

        # ---- convergence tests --------------------------------------------
        # Max error over the last three iterations, so one "lucky" near-zero
        # change cannot fake convergence.
        WMTO = parg[I.IGWMTO]
        errw1 = (WMTO - WMTO1) / WMTO
        errw2 = (WMTO - WMTO2) / WMTO
        errw3 = (WMTO - WMTO3) / WMTO
        errw = max(abs(errw1), abs(errw2), abs(errw3))

        row = (iterw, errw1,
               parm[I.IMWTO] * LB_N, parg[I.IGWFUEL] * LB_N,
               parg[I.IGWFUSE] * LB_N, parg[I.IGWWING] * LB_N,
               parg[I.IGWENG] * LB_N,
               parg[I.IGB] * FT_M, parg[I.IGS] * FT_M ** 2,
               parg[I.IGSH] * FT_M ** 2, parg[I.IGXWBOX] * FT_M)
        history.append(row)
        if Litprint:
            if iterw == 1:
                print()
                print("  iterw     errW    "
                      "      WMTO        Wfuel        Wfuse        Wwing"
                      "         Weng"
                      "        span     area     HTarea   xwbox")
            print("%6d%14.10f%13.4f%13.4f%13.4f%13.4f%13.4f%9.3f%10.3f"
                  "%10.3f%12.5f" % row)

        if errw < TOLERW:
            Lconv = True
            break

        # ---- wing sizing --------------------------------------------------
        WMTO = parg[I.IGWMTO]

        ip = I.IPCRUISE1
        W = WMTO * para[I.IAFRACW, ip]
        CL = para[I.IACL, ip]
        qinf = 0.5 * pare[I.IERHO0, ip] * pare[I.IEU0, ip] ** 2
        BW = W + WbuoyCR

        wp = wingsc(BW, CL, qinf, AR, etas, bo, lambdat, lambdas)
        S, b, bs, co = wp.S, wp.b, wp.bs, wp.co
        parg[I.IGS] = S
        parg[I.IGB] = b
        parg[I.IGBS] = bs
        parg[I.IGCO] = co

        # Center wing box chord, for fusew on the next cycle.
        cbox = co * wbox

        c = surfdx(b, bs, bo, lambdat, lambdas, sweep)
        dxwing, macco = c.dx, c.macco
        xwing = xwbox + dxwing
        cma = macco * co
        parg[I.IGXWING] = xwing
        parg[I.IGCMA] = cma

        # ---- wing pitching moment constants, in three blocks -------------
        for ip_src, lo, hi in ((I.IPTAKEOFF, I.IPSTATIC, I.IPCLIMB1),
                               (I.IPCRUISE1, I.IPCLIMB1 + 1,
                                I.IPDESCENTN - 1),
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

        # ---- wing center load, from the cruise spanload -------------------
        ip = I.IPCRUISE1
        gammat = lambdat * para[I.IARCLT, ip]
        gammas = lambdas * para[I.IARCLS, ip]
        Lhtail = WMTO * parg[I.IGCLHNRAT] * parg[I.IGSH] / parg[I.IGS]
        po = wingpo(b, bs, bo, lambdat, lambdas, gammat, gammas,
                    AR, Nlift, WMTO, Lhtail, fLo, fLt)

        # Engines on the wing sit at ys = bs/2 and relieve the bending.
        Weng1 = parg[I.IGWENG] / neng if iwplan == 1 else 0.0

        Winn = parg[I.IGWINN]
        Wout = parg[I.IGWOUT]
        dyWinn = parg[I.IGDYWINN]
        dyWout = parg[I.IGDYWOUT]
        rhofuel = parg[I.IGRHOFUEL]
        Ecap = parg[I.IGECAP]
        Eweb = Ecap
        Gcap = Ecap * 0.5 / (1.0 + 0.3)
        Gweb = Ecap * 0.5 / (1.0 + 0.3)
        sw = surfw(gee=GEE, po=po, b=b, bs=bs, bo=bo, co=co, zs=zs,
                   lambdat=lambdat, lambdas=lambdas,
                   gammat=gammat, gammas=gammas,
                   Nload=Nlift, iwplan=iwplan, We=Weng1,
                   Winn=Winn, Wout=Wout, dyWinn=dyWinn, dyWout=dyWout,
                   sweep=sweep, wbox=wbox, hboxo=hboxo, hboxs=hboxs,
                   rh=rh, fLt=fLt,
                   tauweb=tauweb, sigcap=sigcap, sigstrut=sigstrut,
                   Ecap=Ecap, Eweb=Eweb, Gcap=Gcap, Gweb=Gweb,
                   rhoweb=rhoweb, rhocap=rhocap, rhostrut=rhostrut,
                   rhofuel=rhofuel)

        Wwing = 2.0 * (sw.Wscen + sw.Wsinn + sw.Wsout) * (1.0 + fwadd)
        dxWwing = 2.0 * (sw.dxWsinn + sw.dxWsout) * (1.0 + fwadd)

        if pari[I.IIFWCEN] == 0:
            Wfmax = 2.0 * (sw.Wfinn + sw.Wfout)
        else:
            Wfmax = 2.0 * (sw.Wfcen + sw.Wfinn + sw.Wfout)
        dxWfmax = 2.0 * (sw.dxWfinn + sw.dxWfout)

        rfmax = parg[I.IGWFUEL] / Wfmax
        parg[I.IGWWING] = Wwing * rlx + parg[I.IGWWING] * (1.0 - rlx)
        parg[I.IGWFMAX] = Wfmax
        parg[I.IGDXWWING] = dxWwing
        parg[I.IGDXWFUEL] = dxWfmax * rfmax

        parg[I.IGTBWEBS] = sw.tbwebs
        parg[I.IGTBCAPS] = sw.tbcaps
        parg[I.IGTBWEBO] = sw.tbwebo
        parg[I.IGTBCAPO] = sw.tbcapo
        parg[I.IGASTRUT] = sw.Astrut
        parg[I.IGCOSLS] = sw.cosLs
        parg[I.IGWWEB] = sw.Wweb
        parg[I.IGWCAP] = sw.Wcap
        parg[I.IGWSTRUT] = sw.Wstrut
        parg[I.IGSOMAX] = sw.So
        parg[I.IGMOMAX] = sw.Mo
        parg[I.IGSSMAX] = sw.Ss
        parg[I.IGMSMAX] = sw.Ms
        parg[I.IGEICO] = sw.EIco
        parg[I.IGEICS] = sw.EIcs
        parg[I.IGEINO] = sw.EIno
        parg[I.IGEINS] = sw.EIns
        parg[I.IGGJO] = sw.GJo
        parg[I.IGGJS] = sw.GJs
        parg[I.IGDXWSTRUT] = sw.dxWstrut

        cstrut = math.sqrt(0.5 * sw.Astrut / (tohstrut * hstrut))
        parg[I.IGCSTRUT] = cstrut
        parg[I.IGSSTRUT] = 2.0 * cstrut * sw.lsp

        # Individual panel weights, with the fuel in them.
        rfmax = parg[I.IGWFUEL] / Wfmax
        parg[I.IGWINN] = sw.Wsinn * (1.0 + fwadd) + rfmax * sw.Wfinn
        parg[I.IGWOUT] = sw.Wsout * (1.0 + fwadd) + rfmax * sw.Wfout
        parg[I.IGDYWINN] = sw.dyWsinn * (1.0 + fwadd) + rfmax * sw.dyWfinn
        parg[I.IGDYWOUT] = sw.dyWsout * (1.0 + fwadd) + rfmax * sw.dyWfout

        # ---- tail sizing ---------------------------------------------------
        depsda = parg[I.IGDEPSDA]
        tanL = math.tan(sweep * math.pi / 180.0)
        tanLh = math.tan(sweeph * math.pi / 180.0)
        Mach = para[I.IAMACH, I.IPCRUISE1]
        beta = math.sqrt(1.0 - Mach ** 2)
        dCLhdCL = ((beta + 2.0 / AR) / (beta + 2.0 / ARh)
                   * math.sqrt((beta ** 2 + tanL ** 2)
                               / (beta ** 2 + tanLh ** 2))
                   * (1.0 - depsda))
        parg[I.IGDCLHDCL] = dCLhdCL

        dCLnda = parg[I.IGDCLNDA]
        dCLndCL = (dCLnda * (beta + 2.0 / AR)
                   * math.sqrt(beta ** 2 + tanL ** 2)
                   / (2.0 * math.pi * (1.0 + 0.5 * hboxo)))
        parg[I.IGDCLNDCL] = dCLndCL

        xhtail = parg[I.IGXHTAIL]
        xvtail = parg[I.IGXVTAIL]
        if iterw <= 2 and initwgt == 0:
            # Size the HT to a specified tail volume coefficient.
            lhtail = xhtail - xwing
            Sh = parg[I.IGVH] * S * cma / lhtail
            parg[I.IGSH] = Sh
        else:
            # Size the HT and locate the wing simultaneously.
            paraF = para.column(I.IPDESCENTN)
            paraB = para.column(I.IPCRUISE1)
            paraC = para.column(I.IPCRUISE1)
            htsize(pari, parg, paraF, paraB, paraC)
            xwbox = parg[I.IGXWBOX]
            xwing = parg[I.IGXWING]
            lhtail = xhtail - xwing
            Sh = parg[I.IGSH]
            parg[I.IGVH] = Sh * lhtail / (S * cma)

        ip = I.IPROTATE
        qstall = 0.5 * pare[I.IERHO0, ip] * (pare[I.IEU0, ip] / 1.2) ** 2
        CDAe = parg[I.IGCDEFAN] * 0.25 * math.pi * parg[I.IGDFAN] ** 2
        De = qstall * CDAe
        Fe = pare[I.IEFE, ip]
        Me = (Fe + De) * yeng

        lvtail = xvtail - xwing
        if iVTsize == 1:
            Sv = parg[I.IGVV] * S * b / lvtail
            parg[I.IGSV] = Sv
            parg[I.IGCLVEOUT] = Me / (qstall * Sv * lvtail)
        else:
            # Size the VT to balance the engine-out yaw moment.
            Sv = Me / (qstall * parg[I.IGCLVEOUT] * lvtail)
            parg[I.IGSV] = Sv
            parg[I.IGVV] = Sv * lvtail / (S * b)

        th = tailpo(Sh, ARh, lambdah, qne, CLhmax)
        bh, coh, poh = th.b, th.co, th.po
        parg[I.IGBH] = bh
        parg[I.IGCOH] = coh

        # The VT is sized as a single tail plus its own bottom image.
        tv = tailpo(2.0 * Sv / nvtail, 2.0 * ARv, lambdav, qne, CLvmax)
        bv2, cov, pov = tv.b, tv.co, tv.po
        bv = 0.5 * bv2          # actual span of one VT, without its image
        parg[I.IGBV] = bv
        parg[I.IGCOV] = cov

        Ecap = parg[I.IGECAP]
        Eweb = Ecap
        Gcap = Ecap * 0.5 / (1.0 + 0.3)
        Gweb = Ecap * 0.5 / (1.0 + 0.3)
        sh = surfw(gee=GEE, po=poh, b=bh, bs=boh, bo=boh, co=coh, zs=zsh,
                   lambdat=lambdah, lambdas=lambdahs,
                   gammat=lambdah, gammas=lambdahs,
                   Nload=1.0, iwplan=0, We=0.0,
                   Winn=0.0, Wout=0.0, dyWinn=0.0, dyWout=0.0,
                   sweep=sweeph, wbox=wboxh, hboxo=hboxh, hboxs=hboxh,
                   rh=rhh, fLt=fLt,
                   tauweb=tauwebh, sigcap=sigcaph, sigstrut=sigstrut,
                   Ecap=Ecap, Eweb=Eweb, Gcap=Gcap, Gweb=Gweb,
                   rhoweb=rhowebh, rhocap=rhocaph, rhostrut=rhostrut,
                   rhofuel=rhofuel)

        Whtail = 2.0 * (sh.Wscen + sh.Wsinn + sh.Wsout) * (1.0 + fhadd)
        dxWhtail = 2.0 * (sh.dxWsinn + sh.dxWsout) * (1.0 + fhadd)
        parg[I.IGWHTAIL] = Whtail
        parg[I.IGDXWHTAIL] = dxWhtail
        parg[I.IGTBWEBH] = sh.tbwebo
        parg[I.IGTBCAPH] = sh.tbcapo
        parg[I.IGEICH] = sh.EIco
        parg[I.IGEINH] = sh.EIno
        parg[I.IGGJH] = sh.GJo

        parg[I.IGXHTAIL] = xhbox + surfdx(bh, boh, boh, lambdah, lambdahs,
                                          sweeph).dx

        mh = surfcm(b=bh, bs=boh, bo=boh, sweep=sweeph, Xaxis=Xaxis,
                    lambdat=lambdah, lambdas=1.0,
                    gammat=lambdah, gammas=1.0,
                    AR=ARh, fLo=0.0, fLt=fLt,
                    cmpo=0.0, cmps=0.0, cmpt=0.0)
        for jp in range(I.IPSTATIC, I.IPDESCENTN + 1):
            para[I.IACMH0, jp] = mh.CM0
            para[I.IACMH1, jp] = mh.CM1

        Ecap = parg[I.IGECAP]
        Eweb = Ecap
        Gcap = Ecap * 0.5 / (1.0 + 0.3)
        Gweb = Ecap * 0.5 / (1.0 + 0.3)
        sv = surfw(gee=GEE, po=pov, b=bv2, bs=bov, bo=bov, co=cov, zs=zsv,
                   lambdat=lambdav, lambdas=lambdavs,
                   gammat=lambdav, gammas=lambdavs,
                   Nload=1.0, iwplan=0, We=0.0,
                   Winn=0.0, Wout=0.0, dyWinn=0.0, dyWout=0.0,
                   sweep=sweepv, wbox=wboxv, hboxo=hboxv, hboxs=hboxv,
                   rh=rhv, fLt=fLt,
                   tauweb=tauwebv, sigcap=sigcapv, sigstrut=sigstrut,
                   Ecap=Ecap, Eweb=Eweb, Gcap=Gcap, Gweb=Gweb,
                   rhoweb=rhowebv, rhocap=rhocapv, rhostrut=rhostrut,
                   rhofuel=rhofuel)

        # Half of the tail-plus-image, times the number of tails.
        Wvtail = (sv.Wscen + sv.Wsinn + sv.Wsout) * (1.0 + fvadd) * nvtail
        dxWvtail = (sv.dxWsinn + sv.dxWsout) * (1.0 + fvadd) * nvtail
        parg[I.IGWVTAIL] = Wvtail
        parg[I.IGDXWVTAIL] = dxWvtail
        parg[I.IGTBWEBV] = sv.tbwebo
        parg[I.IGTBCAPV] = sv.tbcapo
        parg[I.IGEICV] = sv.EIco
        parg[I.IGEINV] = sv.EIno
        parg[I.IGGJV] = sv.GJo

        parg[I.IGXVTAIL] = xvbox + surfdx(bv2, bov, bov, lambdav, lambdavs,
                                          sweepv).dx

        # ---- engine and nacelle sizing and weight -------------------------
        WMTO = parg[I.IGWMTO]
        ip = I.IPCRUISE1

        # Trim by adjusting CLh, so cdsum sees the right wing cl.
        Wzero = WMTO - parg[I.IGWFUEL]
        Wf = para[I.IAFRACW, ip] * WMTO - Wzero
        rfuel = Wf / parg[I.IGWFUEL]
        acol = para.column(ip)
        ecol = pare.column(ip)
        balance(pari, parg, acol, rfuel, 1.0, 0.0, 1)

        icdfun = 1          # look the wing section drag up in the database
        cdsum(pari, parg, acol, ecol, icdfun, table)

        DoL = acol[I.IACD] / acol[I.IACL]
        gamV = acol[I.IAGAMV]
        W = acol[I.IAFRACW] * WMTO
        BW = W + WbuoyCR
        Fdes = BW * (DoL + gamV)
        ecol[I.IEFE] = Fdes / neng

        icall = 0           # size the engine
        icool = 1           # use the cooling flows already set
        inite1 = 0 if (iterw <= 1 or initeng == 0) else 1
        tfcalc(pari, parg, acol, ecol, ip, icall, icool, inite1)
        para.set_column(ip, acol)
        pare.set_column(ip, ecol)

        # Store the engine design point at every operating point.
        parg[I.IGA5] = pare[I.IEA5, ip] / pare[I.IEA5FAC, ip]
        parg[I.IGA7] = pare[I.IEA7, ip] / pare[I.IEA7FAC, ip]
        for jp in range(1, I.IPTOTAL + 1):
            pare[I.IEA2, jp] = pare[I.IEA2, ip]
            pare[I.IEA25, jp] = pare[I.IEA25, ip]
            pare[I.IEA5, jp] = parg[I.IGA5] * pare[I.IEA5FAC, jp]
            pare[I.IEA7, jp] = parg[I.IGA7] * pare[I.IEA7FAC, jp]
            for idx in (I.IENBFD, I.IENBLCD, I.IENBHCD, I.IENBHTD,
                        I.IENBLTD, I.IEMBFD, I.IEMBLCD, I.IEMBHCD,
                        I.IEMBHTD, I.IEMBLTD, I.IEPIFD, I.IEPILCD,
                        I.IEPIHCD, I.IEPIHTD, I.IEPILTD):
                pare[idx, jp] = pare[idx, ip]

        dfan = parg[I.IGDFAN]
        dlcomp = parg[I.IGDLCOMP]

        mdotc = pare[I.IEMBLCD, ip] * math.sqrt(Tref / TSL) * (pSL / pref)
        BPR = pare[I.IEBPR, ip]
        OPR = pare[I.IEPILC, ip] * pare[I.IEPIHC, ip]

        ew = tfweight(iengwgt, parg[I.IGGEARF], OPR, BPR, mdotc, dfan,
                      rSnace, dlcomp, neng, feadd, fpylon)
        parg[I.IGWENG] = ew.Weng
        parg[I.IGWEBARE] = ew.Webare
        parg[I.IGWNACE] = ew.Wnac

        Snace = ew.Snace1 * neng
        fSnace = Snace / S
        parg[I.IGFSNACE] = fSnace
        parg[I.IGLNACE] = parg[I.IGDFAN] * parg[I.IGRSNACE] * 0.15

        # ---- fly the mission ----------------------------------------------
        ipc1 = 1        # the cruise1 point is already computed, above
        m_result = mission(pari, parg, parm, para, pare, table, inite1, ipc1)
        parg[I.IGWFUEL] = parm[I.IMWFUEL]

        # ---- size the cooling flow at takeoff rotation, at Vstall ---------
        ip = I.IPROTATE
        acol = para.column(ip)
        ecol = pare.column(ip)
        # CDwing has to be defined here in case there is wing BLI.
        cdfw = acol[I.IACDFW] * acol[I.IAFEXCDW]
        cdpw = acol[I.IACDPW] * acol[I.IAFEXCDW]
        cosL = math.cos(parg[I.IGSWEEP] * math.pi / 180.0)
        acol[I.IACDWING] = cdfw + cdpw * cosL ** 3

        icall = 1       # fixed engine geometry, specified Tt4
        icool = 2       # set the turbine cooling mass flow
        tfcalc(pari, parg, acol, ecol, ip, icall, icool, inite1)
        para.set_column(ip, acol)
        pare.set_column(ip, ecol)

        # Tmetal was specified, so propagate the blade-row cooling ratios.
        for jp in range(1, I.IPTOTAL + 1):
            for icrow in range(I.NCROWX):
                pare[I.IEEPSC1 + icrow, jp] = pare[I.IEEPSC1 + icrow, ip]
            pare[I.IEFC, jp] = pare[I.IEFC, ip]

        # ---- recompute the max weight with the latest information --------
        fsum = Wupdate(parg, rlx)
        if fsum >= 1.0:
            exploded = True
            break
        parm[I.IMWTO] = parg[I.IGWMTO]
        parm[I.IMWFUEL] = parg[I.IGWFUEL]

        WMTO3 = WMTO2
        WMTO2 = WMTO1
        WMTO1 = parg[I.IGWMTO]

    if not Lconv:
        # The source prints this and carries on -- its `return` is commented
        # out -- so a non-converged sizing still produces takeoff numbers.
        print(f"WSIZE: Weight iteration not converged.  dWrel = {errw} "
              f"{fsum}")

    # =====================================================================
    # Takeoff, CG limits, neutral point.
    # =====================================================================
    for ip in (I.IPSTATIC, I.IPROTATE):
        acol = para.column(ip)
        ecol = pare.column(ip)
        tfcalc(pari, parg, acol, ecol, ip, 1, 1, inite1)
        para.set_column(ip, acol)
        pare.set_column(ip, ecol)

    t_result = takeoff(pari, parg, parm, para, pare, table)

    cg = cglpay(parg)
    parg[I.IGXCGFWD] = cg.xcgF
    parg[I.IGXCGAFT] = cg.xcgB
    parg[I.IGRPAYFWD] = cg.rpayF
    parg[I.IGRPAYAFT] = cg.rpayB

    # Neutral point at cruise.
    ip = I.IPCRUISE1
    WMTO = parg[I.IGWMTO]
    Wzero = WMTO - parg[I.IGWFUEL]
    Wf = para[I.IAFRACW, ip] * WMTO - Wzero
    rfuel = Wf / parg[I.IGWFUEL]
    acol = para.column(ip)
    balance(pari, parg, acol, rfuel, 1.0, 0.0, 0)
    para.set_column(ip, acol)
    parg[I.IGXNP] = acol[I.IAXNP]

    return WSizeResult(converged=Lconv, iterations=iterw, errw=errw,
                       fsum=fsum, mission=m_result, takeoff=t_result,
                       history=history, fuselage_bl=fuselage_bl)
