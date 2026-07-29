"""Certification-point noise -- ``noise.f``.

Two jobs, and they are worth separating because only one of them needs an
acoustic model:

1. **Set the takeoff and cutback engine operating points.** These two mission
   points are not touched by ``wsize`` or ``mission``; ``noise.f`` is the only
   routine that runs the engine there. It puts the takeoff point at full
   power and the cutback point at whatever thrust holds the prescribed cutback
   climb angle, and writes their flight-path angles back into ``para``.
   Everything downstream that reports on those points -- the ``.out`` file's
   ``TO:`` and ``CB:`` rows and their whole engine blocks -- depends on this.

2. **Work out the three certification observer positions and the dB at each**
   -- sideline, cutback and flyover. The geometry is plain trigonometry on the
   takeoff and cutback climb angles; the decibels come from ``tfnoise``, a
   full ESDU/Heidmann fan-and-jet model.

This module does 1 and the geometry half of 2 unconditionally. The decibels
need ``tfnoise``, which is **not ported**; pass one in as the ``tfnoise``
argument if you have it, and the three ``parm`` entries are filled. Without
it they are left exactly as they were, which in a fresh run means the
"unset" fill value -- deliberately, so a missing acoustic model reads as
missing rather than as silence.

The source's own header is worth repeating: "This routine is for noise
estimates only, and is not intended to replace ANOPP etc."

Things in the source worth knowing
----------------------------------
* The atmospheric state used for the *flyover* point is not the cutback
  point's. ``noise.f`` perturbs the takeoff point's ``p0``/``rho0``/``mu0``/
  ``c0`` up to the cutback altitude for the cutback-noise calculation, and
  then the flyover block reuses those perturbed values without recomputing
  them. Reproduced.
* ``c2`` and the fan ``RPM`` for the flyover are computed from the *takeoff*
  point's ``M0`` and ``M2``, because the lines that re-read them at the
  cutback point come afterwards. Reproduced; the ordering is load-bearing.
* ``Mt = 1.2``, ``Mtr = 1.38``, ``Mtrd = 1.30``, 20 rotor blades, 44 stator
  vanes, rotor-stator spacing ``0.2 dfan``: all hard-wired, and the source
  notes the fan geometry came from the SAX-40 design.
* **The angle of attack is passed in different units at different points.**
  ``alpha = 5.0`` for the sideline and ``alpha = 5.0*pi/180`` for the cutback
  and flyover, so the sideline calculation is handed 5 *radians* -- 286
  degrees -- and the other two 5 degrees. It reaches the jet-noise
  correlation as ``cos(alpha)`` in the effective-velocity and convective-Mach
  terms, and 5 radians gives 0.28 where 5 degrees gives 0.996 -- so the
  sideline jet is computed with a much weaker flight-velocity correction than
  the other two points. Reproduced: without it the cutback and flyover come
  out about 1 dB from the reference, with it the whole report matches.
* ``type = 0`` (unmixed exhaust) and ``method = 2`` (Heidmann, as in ANOPP)
  are the live settings; mixed exhaust and the ESDU and Allied Signal variants
  are commented out beside them.

Verified against the compiled Fortran; see ``tests/test_noise.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..aero.cdsum import cdsum
from ..engine.tfcalc import tfcalc
from ..model import indices as I

__all__ = ["noise", "NoiseGeometry", "FAN", "L_OBSERVER", "SIDELINE_Y",
           "EXHAUST_UNMIXED", "METHOD_HEIDMANN"]

GEE = 9.81
GAMSL = 1.4
#: Distance from brake release the certification observer sits at [m].
L_OBSERVER = 6500.0
#: Sideline microphone offset [m].
SIDELINE_Y = 450.0
EXHAUST_UNMIXED = 0
METHOD_HEIDMANN = 2


@dataclass
class FanNoiseGeometry:
    """Fan tip Mach numbers and blade counts, all hard-wired in the source."""
    Mt: float = 1.2        # tangential tip Mach number
    Mtr: float = 1.38      # relative tip Mach number
    Mtrd: float = 1.30     # relative tip Mach number at design
    B: float = 20.0        # rotor blades
    V: float = 44.0        # stator vanes
    #: Angle of attack passed to the acoustic model. The source writes
    #: ``alpha = 5.0`` for the sideline point and ``alpha = 5.0*pi/180`` for
    #: the other two, so the sideline is given 5 *radians* -- 286 degrees --
    #: where the others get 5 degrees. See the module docstring.
    alpha_sideline: float = 5.0
    alpha_other: float = 5.0 * math.pi / 180.0
    vector: float = 0.0


FAN = FanNoiseGeometry()


@dataclass
class NoiseGeometry:
    """Where each certification observer sits, relative to the aircraft."""
    lCB: float = 0.0          # ground distance to the cutback point
    dhTO: float = 0.0         # altitude gained by the cutback point
    xSL: float = 0.0
    zSL: float = 0.0
    xCB: float = 0.0
    zCB: float = 0.0
    xFO: float = 0.0
    zFO: float = 0.0
    dBSL: float = None
    dBCB: float = None
    dBFO: float = None


def noise(pari, parg, parm, para, pare, initeng: int, table=None,
          tfnoise=None) -> NoiseGeometry:
    """Set the takeoff and cutback engine points, and the observer geometry.

    ``tfnoise`` is the acoustic model, called as

        tfnoise(x, y, z, climb, alpha, vector, rho0, p0, T0, mu0, c0,
                A6, A8, u6, u8, T6, T8, M0, etaf, FPR, mdot, BPR,
                Mtrd, Mtr, Mt, RPM, rss, B, V, htr, neng, type, method)

    returning either the total A-weighted SPL or an object carrying it as
    ``.total`` -- :func:`tasopt_py.acoustics.tfnoise` returns the latter, with
    the six components broken out. Omit it and the decibels are left
    untouched.
    """
    W = parm[I.IMWTO]
    neng = parg[I.IGNENG]
    htr = parg[I.IGHTRF]
    dfan = parg[I.IGDFAN]
    rss = 0.2 * dfan * 100.0        # rotor-stator spacing [mm]

    g = NoiseGeometry()

    # ---- takeoff point, at full power ------------------------------------
    ip = I.IPTAKEOFF
    acol, ecol = para.column(ip), pare.column(ip)
    cdsum(pari, parg, acol, ecol, 0, table)

    # Start the engine solve from the rotation point's state.
    ipc = I.IPROTATE
    for idx in (I.IEMBF, I.IEMBLC, I.IEMBHC, I.IEPIF, I.IEPILC, I.IEPIHC,
                I.IEPT5):
        ecol[idx] = pare[idx, ipc]

    tfcalc(pari, parg, acol, ecol, ip, 1, 1, initeng)

    F = ecol[I.IEFE] * neng
    DoL = acol[I.IACD] / acol[I.IACL]
    phi = F / W
    sing = ((phi - DoL * math.sqrt(1.0 - phi ** 2 + DoL ** 2))
            / (1.0 + DoL ** 2))
    acol[I.IAGAMV] = math.atan2(sing, math.sqrt(1.0 - sing ** 2))
    para.set_column(ip, acol)
    pare.set_column(ip, ecol)

    # ---- cutback point, at the thrust that holds the prescribed angle ----
    ip = I.IPCUTBACK
    acol, ecol = para.column(ip), pare.column(ip)
    cdsum(pari, parg, acol, ecol, 0, table)

    DoL = acol[I.IACD] / acol[I.IACL]
    gamVCB = parm[I.IMGAMVCB]
    singCB, cosgCB = math.sin(gamVCB), math.cos(gamVCB)
    ecol[I.IEFE] = W * (singCB + DoL * cosgCB) / neng

    # Start from the first climb point's state, with a burner rise carried
    # across rather than a temperature.
    ipc = I.IPCLIMB1
    for idx in (I.IEMBF, I.IEMBLC, I.IEMBHC, I.IEPIF, I.IEPILC, I.IEPIHC,
                I.IEPT5):
        ecol[idx] = pare[idx, ipc]
    dTburn = pare[I.IETT4, ipc] - pare[I.IETT3, ipc]
    ecol[I.IETT4] = pare[I.IETT3, ipc] + dTburn

    tfcalc(pari, parg, acol, ecol, ip, 2, 1, 1)

    F = ecol[I.IEFE] * neng
    DoL = acol[I.IACD] / acol[I.IACL]
    phi = F / W
    sing = ((phi - DoL * math.sqrt(1.0 - phi ** 2 + DoL ** 2))
            / (1.0 + DoL ** 2))
    acol[I.IAGAMV] = math.atan2(sing, math.sqrt(1.0 - sing ** 2))
    para.set_column(ip, acol)
    pare.set_column(ip, ecol)

    # =====================================================================
    # Observer geometry, and the decibels if an acoustic model was given.
    # =====================================================================
    ip = I.IPTAKEOFF
    A2, M2 = pare[I.IEA2, ip], pare[I.IEM2, ip]
    src = dict(A6=pare[I.IEA6, ip], A8=pare[I.IEA8, ip],
               u6=pare[I.IEU6, ip], u8=pare[I.IEU8, ip],
               T6=pare[I.IET6, ip], T8=pare[I.IET8, ip],
               M0=pare[I.IEM0, ip], etaf=pare[I.IEETAF, ip],
               FPR=pare[I.IEPIF, ip], BPR=pare[I.IEBPR, ip],
               mdot=pare[I.IEMCORE, ip])
    M0 = src["M0"]

    rho0, p0 = pare[I.IERHO0, ip], pare[I.IEP0, ip]
    T0, mu0, c0 = pare[I.IET0, ip], pare[I.IEMU0, ip], pare[I.IEA0, ip]

    def rpm_at(c0_):
        c2 = c0_ * math.sqrt((1.0 + 0.5 * (GAMSL - 1.0) * M0 ** 2)
                             / (1.0 + 0.5 * (GAMSL - 1.0) * M2 ** 2))
        return FAN.Mt * c2 / (0.5 * dfan) * 30.0 / math.pi

    def call_tfnoise(x, y, z, climb, rho0_, p0_, T0_, mu0_, c0_, RPM, s,
                     alpha):
        if tfnoise is None:
            return None
        out = tfnoise(x, y, z, climb, alpha, FAN.vector,
                       rho0_, p0_, T0_, mu0_, c0_,
                       s["A6"], s["A8"], s["u6"], s["u8"], s["T6"], s["T8"],
                       s["M0"], s["etaf"], s["FPR"], s["mdot"], s["BPR"],
                       FAN.Mtrd, FAN.Mtr, FAN.Mt, RPM,
                      rss, FAN.B, FAN.V, htr, neng,
                      EXHAUST_UNMIXED, METHOD_HEIDMANN)
        return getattr(out, "total", out)

    # ---- sideline: 450 m abeam, at the takeoff point ---------------------
    # Note `climb` here is the raw angle, while the two below are asin(sin g)
    # -- the same number by a different route, as in the source.
    g.xSL, g.zSL = 0.0, 0.0
    dB = call_tfnoise(0.0, SIDELINE_Y, 0.0, parm[I.IMGAMVTO],
                      rho0, p0, T0, mu0, c0, rpm_at(c0), src,
                      FAN.alpha_sideline)
    if dB is not None:
        parm[I.IMDBSL] = dB
        g.dBSL = dB

    # ---- cutback: still at takeoff power and climb angle ------------------
    gamVTO = parm[I.IMGAMVTO]
    singTO, cosgTO = math.sin(gamVTO), math.cos(gamVTO)
    tangTO = singTO / cosgTO
    cott = math.tan(0.5 * math.pi - parm[I.IMTHCB])

    lTO = parm[I.IMLTO]
    dhTO = (L_OBSERVER - lTO) * tangTO / (1.0 + tangTO * cott)
    lCB = (L_OBSERVER - lTO) / (1.0 + tangTO * cott) + lTO
    g.dhTO, g.lCB = dhTO, lCB

    # The atmosphere is nudged to the cutback altitude by the hydrostatic
    # gradient rather than recomputed. These values are then reused, unchanged,
    # by the flyover block below.
    p0 = pare[I.IEP0, ip] - pare[I.IERHO0, ip] * GEE * dhTO
    T0 = pare[I.IET0, ip]
    prat = p0 / pare[I.IEP0, ip]
    Trat = T0 / pare[I.IET0, ip]
    rho0 = pare[I.IERHO0, ip] * prat / Trat
    mu0 = pare[I.IEMU0, ip] * Trat              # mu ~ T
    c0 = pare[I.IEA0, ip] * math.sqrt(Trat)

    x, z = L_OBSERVER - lCB, dhTO
    g.xCB, g.zCB = x, z
    parm[I.IMXCB], parm[I.IMZCB], parm[I.IMLCB] = x, z, lCB
    dB = call_tfnoise(x, 0.0, z, math.asin(singTO),
                      rho0, p0, T0, mu0, c0, rpm_at(c0), src,
                      FAN.alpha_other)
    if dB is not None:
        parm[I.IMDBCB] = dB
        g.dBCB = dB

    # ---- flyover: at cutback power, same V2 --------------------------------
    ip = I.IPCUTBACK
    # RPM still uses the *takeoff* M0 and M2 -- the cutback values are read
    # only after this line in the source.
    RPM = rpm_at(c0)
    climb = math.asin(singCB)

    A2, M2 = pare[I.IEA2, ip], pare[I.IEM2, ip]
    src = dict(A6=pare[I.IEA6, ip], A8=pare[I.IEA8, ip],
               u6=pare[I.IEU6, ip], u8=pare[I.IEU8, ip],
               T6=pare[I.IET6, ip], T8=pare[I.IET8, ip],
               M0=pare[I.IEM0, ip], etaf=pare[I.IEETAF, ip],
               FPR=pare[I.IEPIF, ip], BPR=pare[I.IEBPR, ip],
               mdot=pare[I.IEMCORE, ip])

    # Aircraft position at the cutback point relative to the observer, then
    # the point on the climb line directly overhead (a 90-degree sighting).
    rxCB, rzCB = -parm[I.IMXCB], parm[I.IMZCB]
    sxr = cosgCB * rzCB - singCB * rxCB
    rnx, rnz = -sxr * singCB, sxr * cosgCB

    theta = math.pi / 2.0                        # sighting elevation, 90 deg
    sint, cost = math.sin(math.pi - theta), math.cos(math.pi - theta)
    sxt = cosgCB * cost + singCB * sint
    tanp = sxt / math.sqrt(1.0 - sxt ** 2)
    xa = rnx + rnz * tanp
    za = rnz - rnx * tanp

    x, z = -xa, za
    g.xFO, g.zFO = x, z
    parm[I.IMXFO], parm[I.IMZFO] = x, z
    dB = call_tfnoise(x, 0.0, z, climb, rho0, p0, T0, mu0, c0, RPM, src,
                      FAN.alpha_other)
    if dB is not None:
        parm[I.IMDBFO] = dB
        g.dBFO = dB

    return g
