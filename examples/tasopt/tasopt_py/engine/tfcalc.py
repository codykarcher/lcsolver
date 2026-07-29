"""Engine wrapper over sizing and off-design -- a port of ``tfcalc.f``.

One entry point for one mission point. Unpacks the engine state out of
``pare``, calls :func:`~tasopt_py.engine.tfsize.tfsize` or
:func:`~tasopt_py.engine.tfoper.tfoper`, and packs the answer back.

``icall``
    0 size the engine (``tfsize``); 1 run it off design at specified ``Tt4``;
    2 run it off design at specified thrust.
``icool``
    0 no cooling; 1 coolant fractions given and metal temperatures computed;
    2 metal temperature given and coolant computed.
``initeng``
    0 start ``tfoper``'s Newton from the design point; 1 start from whatever
    is currently in ``pare``. Marching a mission means passing 1 -- and, given
    what §22 of DISCREPANCIES.md says about that iteration, it is not
    optional in practice.

What sizing leaves behind
-------------------------
The design call is what establishes the map anchors every later off-design
call reads: ``mbfD``..``mbltD``, ``NbfD``..``NbltD``, ``pifD``..``piltD`` and
the areas ``A2``, ``A25``, ``A5``, ``A7``. Corrected flows are formed at the
design point, and the spool speeds are unity there *by definition* -- the
design case defines what 100% speed means.

The turbine design pressure ratios are set **twice**: first as
``pihtD = pt41/pt45``, then immediately overwritten by the
temperature-derived ``Trh**gexh``, with the comment that this is "to be fully
consistent with TFOPER's turbine efficiency function". The two differ, and
only the second is what ``etmap`` expects. The first assignment is dead.

Boundary-layer ingestion
------------------------
``Phiinl`` and ``Kinl`` -- the dissipation and kinetic-energy defects handed
to the engine -- are built from the fuselage BL result and the wing drag,
each scaled by its ingested fraction. The wing's surface dissipation is taken
as 85% of the total: ``fDwake = 0.15`` here, matching the 15% wake fraction
``cdsum`` uses for its BLI credit. Both are the same bare constant written out
twice in two files; they have to stay in step or the ingestion is
double-counted or lost.

Static points ingest nothing: at ``M0 == 0`` both defects are zero.

Verified against the compiled Fortran; see ``tests/test_tfcalc.py``.
"""
from __future__ import annotations

import math

from ..model import indices as I
from .tfoper import tfoper
from .tfsize import tfsize

__all__ = ["tfcalc", "TFCalcError", "FD_WAKE"]

#: Fraction of wing dissipation assumed to be in the wake rather than on the
#: surface. Must match ``cdsum.WING_WAKE_FRACTION``.
FD_WAKE = 0.15

GEE = 9.81
TREF = 288.2
PREF = 101320.0


class TFCalcError(RuntimeError):
    """The engine calculation failed at this operating point."""


def _ingestion(parg, para, pare, neng):
    """``(Phiinl, Kinl)`` per engine, from the fuselage and wing wakes."""
    M0 = pare[I.IEM0]
    if M0 == 0.0:
        return 0.0, 0.0            # static: nothing to ingest
    rho0, u0 = pare[I.IERHO0], pare[I.IEU0]
    S = parg[I.IGS]

    DAfsurf = para[I.IADAFSURF]
    KAfTE = para[I.IAKAFTE]
    fBLIf = parg[I.IGFBLIF]

    CDAwing = para[I.IACDWING] * S
    DAwsurf = CDAwing * (1.0 - FD_WAKE)
    KAwTE = DAwsurf
    fBLIw = parg[I.IGFBLIW]

    q = 0.5 * rho0 * u0 ** 3
    Phiinl = q * (DAfsurf * fBLIf + DAwsurf * fBLIw) / neng
    Kinl = q * (KAfTE * fBLIf + KAwTE * fBLIw) / neng
    return Phiinl, Kinl


def tfcalc(pari, parg, para, pare, ip: int, icall: int, icool: int,
           initeng: int) -> None:
    """Run the engine at one mission point, in place on ``para``/``pare``."""
    ifuel = pari[I.IIFUEL]
    iBLIc = pari[I.IIBLIC]
    Gearf = parg[I.IGGEARF]
    neng = parg[I.IGNENG]

    # Offtakes are specified per unit payload and per unit MTOW, then split
    # between engines.
    mofft = (parg[I.IGMOFWPAY] * parg[I.IGWPAY]
             + parg[I.IGMOFWMTO] * parg[I.IGWMTO]) / neng
    Pofft = (parg[I.IGPOFWPAY] * parg[I.IGWPAY]
             + parg[I.IGPOFWMTO] * parg[I.IGWMTO]) / neng

    ncrowx = I.NCROWX
    epsrow = [pare[I.IEEPSC1 + k] for k in range(ncrowx)]
    Tmrow = [parg[I.IGTMETAL]] * ncrowx
    ncrow = ncrowx if icool in (1, 2) else 0

    Phiinl, Kinl = _ingestion(parg, para, pare, neng)

    common = dict(
        gee=GEE, M0=pare[I.IEM0], T0=pare[I.IET0], p0=pare[I.IEP0],
        a0=pare[I.IEA0], Phiinl=Phiinl, Kinl=Kinl, iBLIc=iBLIc,
        pid=pare[I.IEPID], pib=pare[I.IEPIB], pifn=pare[I.IEPIFN],
        pitn=pare[I.IEPITN], Ttf=pare[I.IETFUEL], ifuel=ifuel,
        etab=pare[I.IEETAB],
        epf0=pare[I.IEEPOLF], eplc0=pare[I.IEEPOLLC],
        ephc0=pare[I.IEEPOLHC], epht0=pare[I.IEEPOLHT],
        eplt0=pare[I.IEEPOLLT],
        pifK=pare[I.IEPIFK], epfK=pare[I.IEEPFK],
        mofft=mofft, Pofft=Pofft, Tt9=pare[I.IETT9], pt9=pare[I.IEPT9],
        epsl=pare[I.IEEPSL], epsh=pare[I.IEEPSH], icool=icool,
        Mtexit=pare[I.IEMTEXIT], dTstrk=pare[I.IEDTSTRK],
        StA=pare[I.IESTA], efilm=pare[I.IEEFILM], tfilm=pare[I.IETFILM],
        M4a=pare[I.IEM4A], ruc=pare[I.IERUC], ncrowx=ncrowx)

    if icall == 0:
        r = tfsize(M2=pare[I.IEM2], M25=pare[I.IEM25], Feng=pare[I.IEFE],
                   BPR=pare[I.IEBPR], pif=pare[I.IEPIF],
                   pilc=pare[I.IEPILC], pihc=pare[I.IEPIHC],
                   Tt4=pare[I.IETT4], epsrow=epsrow, Tmrow=Tmrow, **common)
        st = r.stations
        mcore, ff = r.mcore, r.ff
        fo = mofft / mcore
        BPR = pare[I.IEBPR]

        def cm(stn, extra=1.0):
            return (mcore * math.sqrt(st[stn].Tt / TREF)
                    / (st[stn].pt / PREF) * extra)

        mbf, mblc = cm(2, BPR), cm(19)
        mbhc = cm(25, 1.0 - fo)
        mbht = cm(41, 1.0 - fo + ff)
        mblt = cm(45, 1.0 - fo + ff)

        # At the design point the spool speeds are unity by definition.
        Nbf = (1.0 / Gearf) / math.sqrt(st[2].Tt / TREF)
        Nblc = 1.0 / math.sqrt(st[19].Tt / TREF)
        Nbhc = 1.0 / math.sqrt(st[25].Tt / TREF)
        Nbht = 1.0 / math.sqrt(st[41].Tt / TREF)
        Nblt = 1.0 / math.sqrt(st[45].Tt / TREF)

        # Turbine design pressure ratios: NOT pt41/pt45. The source assigns
        # that first and then overwrites it with the temperature-derived form
        # below, which is the one etmap is written against.
        Trh = st[41].Tt / (st[41].Tt + (st[45].ht - st[41].ht) / st[41].cpt)
        Trl = st[45].Tt / (st[45].Tt + (st[49].ht - st[45].ht) / st[45].cpt)
        pihtD = Trh ** (st[41].cpt / (st[41].Rt * pare[I.IEEPOLHT]))
        piltD = Trl ** (st[45].cpt / (st[45].Rt * pare[I.IEEPOLLT]))

        for idx, val in ((I.IEA2, st[2].A), (I.IEA25, st[25].A),
                         (I.IEA5, st[5].A), (I.IEA7, st[7].A),
                         (I.IENBFD, Nbf), (I.IENBLCD, Nblc),
                         (I.IENBHCD, Nbhc), (I.IENBHTD, Nbht),
                         (I.IENBLTD, Nblt),
                         (I.IEMBFD, mbf), (I.IEMBLCD, mblc),
                         (I.IEMBHCD, mbhc), (I.IEMBHTD, mbht),
                         (I.IEMBLTD, mblt),
                         (I.IEPIFD, pare[I.IEPIF]),
                         (I.IEPILCD, pare[I.IEPILC]),
                         (I.IEPIHCD, pare[I.IEPIHC]),
                         (I.IEPIHTD, pihtD), (I.IEPILTD, piltD)):
            pare[idx] = val

        pif, pilc, pihc = (pare[I.IEPIF], pare[I.IEPILC], pare[I.IEPIHC])
        M2, M25 = pare[I.IEM2], pare[I.IEM25]
        A2, A25 = st[2].A, st[25].A
        TSFC, Fsp, hfuel = r.TSFC, r.Fsp, r.hfuel
        epsrow, Tmrow, ncrow = r.epsrow, r.Tmrow, r.ncrow
        Fe = pare[I.IEFE]
        Tt4 = pare[I.IETT4]
        stations = st
    else:
        if initeng == 0:
            # Cold: zeros tell tfoper to fall back on the design point.
            guess = dict(pif=0.0, pilc=0.0, pihc=0.0, mbf=0.0, mblc=0.0,
                         mbhc=0.0, pt5=0.0, M2=1.0, M25=1.0)
        else:
            guess = dict(pif=pare[I.IEPIF], pilc=pare[I.IEPILC],
                         pihc=pare[I.IEPIHC], mbf=pare[I.IEMBF],
                         mblc=pare[I.IEMBLC], mbhc=pare[I.IEMBHC],
                         pt5=pare[I.IEPT5], M2=pare[I.IEM2],
                         M25=pare[I.IEM25])

        if icall == 1:
            Fe_in, Tt4_in, iTFspec = 0.0, pare[I.IETT4], 1
        else:
            # Tt4 is the starting guess when thrust is what is specified.
            Fe_in, Tt4_in, iTFspec = pare[I.IEFE], pare[I.IETT4], 2

        try:
            r = tfoper(Tref=TREF, pref=PREF, Gearf=Gearf,
                       pifD=pare[I.IEPIFD], pilcD=pare[I.IEPILCD],
                       pihcD=pare[I.IEPIHCD], pihtD=pare[I.IEPIHTD],
                       piltD=pare[I.IEPILTD],
                       mbfD=pare[I.IEMBFD], mblcD=pare[I.IEMBLCD],
                       mbhcD=pare[I.IEMBHCD], mbhtD=pare[I.IEMBHTD],
                       mbltD=pare[I.IEMBLTD],
                       NbfD=pare[I.IENBFD], NblcD=pare[I.IENBLCD],
                       NbhcD=pare[I.IENBHCD], NbhtD=pare[I.IENBHTD],
                       NbltD=pare[I.IENBLTD],
                       A2=pare[I.IEA2], A25=pare[I.IEA25],
                       A5=pare[I.IEA5], A7=pare[I.IEA7],
                       iTFspec=iTFspec, ncrow=ncrow, epsrow=epsrow,
                       Tmrow=Tmrow, Tt4=Tt4_in, Feng=Fe_in, **common,
                       **guess)
        except Exception as exc:
            raise TFCalcError(
                f"tfcalc: engine failed at operating point {ip} "
                f"(icall={icall}, iTFspec={iTFspec}): {exc}") from exc

        st = r.stations
        mcore, ff = r.mcore, r.ff
        fo = mofft / mcore
        mbf, mblc, mbhc = r.mbf, r.mblc, r.mbhc
        Nbf, Nblc, Nbhc = r.Nbf, r.Nblc, r.Nbhc
        pif, pilc, pihc = r.pif, r.pilc, r.pihc
        BPR = r.BPR
        A2, A25 = pare[I.IEA2], pare[I.IEA25]
        s2 = st[2]
        M2 = s2.u / math.sqrt(s2.T * s2.cp * s2.R / (s2.cp - s2.R))
        s25 = st[25]
        M25 = s25.u / math.sqrt(s25.T * s25.cp * s25.R / (s25.cp - s25.R))
        TSFC, Fsp, hfuel = r.TSFC, r.Fsp, r.hfuel
        epsrow, Tmrow, ncrow = r.epsrow, r.Tmrow, r.ncrow
        Fe, Tt4 = r.Feng, st[4].Tt
        stations = st

        pare[I.IEM2] = M2
        pare[I.IEM25] = M25
        if icall == 1:
            pare[I.IEFE] = Fe
        else:
            pare[I.IETT4] = Tt4
        pare[I.IEBPR] = BPR

    # --- cooling bookkeeping ---------------------------------------------
    if icool == 1:
        for k in range(ncrowx):
            pare[I.IETMET1 + k] = Tmrow[k]
    elif icool == 2:
        epstot = 0.0
        for k in range(ncrowx):
            epstot += epsrow[k]
            pare[I.IEEPSC1 + k] = epsrow[k]
        pare[I.IEFC] = (1.0 - fo) * epstot

    # --- component diameters from the flow areas -------------------------
    Alc = A2 / (1.0 + BPR)
    parg[I.IGDFAN] = math.sqrt(4.0 * A2 / (math.pi
                                           * (1.0 - parg[I.IGHTRF] ** 2)))
    parg[I.IGDLCOMP] = math.sqrt(4.0 * Alc / (math.pi
                                              * (1.0 - parg[I.IGHTRLC] ** 2)))
    parg[I.IGDHCOMP] = math.sqrt(4.0 * A25 / (math.pi
                                              * (1.0 - parg[I.IGHTRHC] ** 2)))

    # --- store the state back --------------------------------------------
    pare[I.IETSFC] = TSFC
    pare[I.IEFSP] = Fsp
    pare[I.IEHFUEL] = hfuel
    pare[I.IEFF] = ff
    pare[I.IEMCORE] = mcore
    pare[I.IEMOFFT] = mofft
    pare[I.IEPOFFT] = Pofft
    pare[I.IEPHIINL] = Phiinl
    pare[I.IEKINL] = Kinl

    pare[I.IENF] = Nbf * math.sqrt(stations[2].Tt / TREF)
    pare[I.IEN1] = Nblc * math.sqrt(stations[19].Tt / TREF)
    pare[I.IEN2] = Nbhc * math.sqrt(stations[25].Tt / TREF)
    pare[I.IENBF] = Nbf
    pare[I.IENBLC] = Nblc
    pare[I.IENBHC] = Nbhc
    pare[I.IEMBF] = mbf
    pare[I.IEMBLC] = mblc
    pare[I.IEMBHC] = mbhc
    pare[I.IEPIF] = pif
    pare[I.IEPILC] = pilc
    pare[I.IEPIHC] = pihc

    for stn, prefix in ((0, "0"), (18, "18"), (19, "19"), (2, "2"),
                        (21, "21"), (25, "25"), (3, "3"), (4, "4"),
                        (41, "41"), (45, "45"), (49, "49"), (5, "5"),
                        (7, "7")):
        s = stations.get(stn)
        if s is None:
            continue
        for attr, name in (("Tt", "TT"), ("ht", "HT"), ("pt", "PT"),
                           ("cpt", "CPT"), ("Rt", "RT")):
            idx = getattr(I, f"IE{name}{prefix}", None)
            val = getattr(s, attr)
            if idx is not None and val is not None:
                pare[idx] = val

    # Static state at the stations that have one, plus the nozzle areas and
    # jet speeds. Nothing on the sizing path reads these -- ``output.f``'s
    # ``engwrt`` prints them, and ``fobj`` reads ``u8`` for the jet-velocity
    # ratio constraint -- but they are part of the engine state the Fortran
    # leaves behind, so leaving them at the caller's fill value would hand a
    # later port of either routine a sentinel.
    for stn, prefix in ((2, "2"), (25, "25"), (5, "5"), (6, "6"),
                        (7, "7"), (8, "8")):
        s = stations.get(stn)
        if s is None:
            continue
        for attr, name in (("p", "P"), ("T", "T"), ("R", "R"),
                           ("cp", "CP"), ("u", "U")):
            idx = getattr(I, f"IE{name}{prefix}", None)
            val = getattr(s, attr)
            if idx is not None and val is not None:
                pare[idx] = val

    for stn, idx in ((6, I.IEA6), (8, I.IEA8), (9, I.IEA9)):
        s = stations.get(stn)
        if s is not None and s.A is not None:
            pare[idx] = s.A
    s9 = stations.get(9)
    if s9 is not None and s9.u is not None:
        pare[I.IEU9] = s9.u

    # Polytropic and isentropic component efficiencies, as achieved.
    for attr, idx in (("epf", I.IEEPF), ("eplc", I.IEEPLC),
                      ("ephc", I.IEEPHC), ("epht", I.IEEPHT),
                      ("eplt", I.IEEPLT), ("etaf", I.IEETAF),
                      ("etalc", I.IEETALC), ("etahc", I.IEETAHC),
                      ("etaht", I.IEETAHT), ("etalt", I.IEETALT)):
        val = getattr(r, attr, None)
        if val is not None:
            pare[idx] = val
