"""Wing / tail box sizing and weight — port of TASOPT ``src/surfw.f``.

Sizes the structural box of a lifting surface and returns its shear,
bending moment, skin/web gauges, section stiffnesses and weight breakdown.
The same routine serves the wing, the horizontal tail and the vertical tail —
they differ only in the loads and planform handed to it.

Planform and stations
---------------------
The surface is described by three spanwise stations:

    etao = bo/b     centreline / side-of-body
    etas = bs/b     strut attach (or the planform break)
    1.0             tip

with taper ratios ``lambdas`` (break) and ``lambdat`` (tip), and load
factors ``gammas``, ``gammat`` giving the local lift per unit span relative
to the centreline value ``po``.

Sizing logic
------------
The box is sized at two stations, ``etas`` and ``etao``, from the local shear
and bending moment. Cap thickness comes from a cubic:

    tbcap = 0.5 (hrms - (hrms^3 - con)^(1/3)),  con = 6 M h / (c^3 sigcap wbox cosL^4)

which is the exact inverse of the box bending-inertia expression, not an
approximation.

``rh`` is the ratio of web height to box height, and ``havg`` / ``hrms`` are
the area- and inertia-weighted mean box heights over the chordwise taper.

Strut branch
------------
``iwplan`` selects the configuration:

* ``0`` or ``1`` — cantilever wing, with (1) or without (0) an engine at
  ``etas``. The root station carries the full inboard load, and So/Mo are
  floored at Ss/Ms so a heavy outboard engine cannot produce a
  negatively-tapered structure (the Fortran comment notes this is "deemed not
  feasible for downloads").
* anything else — strut-braced. The inboard box is sized to the strut-attach
  loads only, and the strut carries tension ``Tstrut``.

Units are SI; ``gee`` is gravitational acceleration, so outputs are weights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

PI = 3.1415926535897932384626


@dataclass(frozen=True)
class SurfaceResult:
    # strut-attach station
    Ss: float
    Ms: float
    tbwebs: float
    tbcaps: float
    EIcs: float
    EIns: float
    GJs: float
    # root station
    So: float
    Mo: float
    tbwebo: float
    tbcapo: float
    EIco: float
    EIno: float
    GJo: float
    # strut
    Astrut: float
    lsp: float
    cosLs: float
    # structural weights and moments
    Wscen: float
    Wsinn: float
    Wsout: float
    dxWsinn: float
    dxWsout: float
    dyWsinn: float
    dyWsout: float
    # fuel volume weights
    Wfcen: float
    Wfinn: float
    Wfout: float
    dxWfinn: float
    dxWfout: float
    dyWfinn: float
    dyWfout: float
    # component totals
    Wweb: float
    Wcap: float
    Wstrut: float
    dxWweb: float
    dxWcap: float
    dxWstrut: float


def surfw(*, gee, po, b, bs, bo, co, zs,
          lambdat, lambdas, gammat, gammas,
          Nload, iwplan, We,
          Winn, Wout, dyWinn, dyWout,
          sweep, wbox, hboxo, hboxs, rh, fLt,
          tauweb, sigcap, sigstrut, Ecap, Eweb, Gcap, Gweb,
          rhoweb, rhocap, rhostrut, rhofuel) -> SurfaceResult:
    """Size a lifting-surface box. Arguments follow ``surfw.f`` exactly."""

    cosL = math.cos(sweep * PI / 180.0)
    sinL = math.sin(sweep * PI / 180.0)

    etao = bo / b
    etas = bs / b

    cop = co * cosL
    csp = co * cosL * lambdas

    # tip load correction
    dLt = fLt * po * co * gammat * lambdat
    dMt = dLt * 0.5 * b * (1.0 - etas)

    # mean and rms box heights over the chordwise height taper rh
    havgo = hboxo * (1.0 - (1.0 - rh) / 3.0)
    havgs = hboxs * (1.0 - (1.0 - rh) / 3.0)
    hrmso = hboxo * math.sqrt(1.0 - (1.0 - rh) / 1.5 + (1.0 - rh)**2 / 5.0)
    hrmss = hboxs * math.sqrt(1.0 - (1.0 - rh) / 1.5 + (1.0 - rh)**2 / 5.0)

    # ---- strut-attach shear and moment from the outer-wing loading -------
    Ss = ((po * b / 4.0) * (gammas + gammat) * (1.0 - etas)
          + dLt - Nload * Wout)
    Ms = ((po * b**2 / 24.0) * (gammas + 2.0 * gammat) * (1.0 - etas)**2
          + dMt - Nload * dyWout)

    # ---- size the strut-attach station -----------------------------------
    cs = co * lambdas
    tbwebs = Ss * 0.5 / (cs**2 * tauweb * rh * hboxs * cosL**2)
    con = Ms * 6.0 * hboxs / (cs**3 * sigcap * wbox * cosL**4)
    tbcaps = 0.5 * (hrmss - (hrmss**3 - con)**(1.0 / 3.0))
    Abcaps = 2.0 * tbcaps * wbox
    Abwebs = 2.0 * tbwebs * rh * hboxs

    EIcs = (Ecap * csp**4 * (hrmss**3 - (hrmss - 2.0 * tbcaps)**3) * wbox / 12.0
            + Eweb * csp**4 * tbwebs * (rh * hboxs)**3 / 6.0)
    EIns = (Ecap * csp**4 * tbcaps * wbox**3 / 6.0
            + Eweb * csp**4 * tbwebs * rh * hboxs * 0.5 * wbox**2)
    GJs = (csp**4 * 2.0 * ((wbox - tbwebs) * (havgs - tbcaps))**2
           / ((rh * hboxs - tbcaps) / (Gweb * tbwebs)
              + (wbox - tbwebs) / (Gcap * tbcaps)))

    if iwplan in (0, 1):
        # ---- cantilever: root carries the full inboard load ---------------
        Tstrutp = 0.0
        So = (Ss - Nload * We
              + 0.25 * po * b * (1.0 + gammas) * (etas - etao)
              - Nload * Winn)
        Mo = (Ms + (Ss - Nload * We) * 0.5 * b * (etas - etao)
              + (1.0 / 24.0) * po * b**2 * (1.0 + 2.0 * gammas) * (etas - etao)**2
              - Nload * dyWinn)
        # A heavy outboard engine could otherwise drive So/Mo below the
        # strut-attach values, implying a negatively-tapered structure.
        So = max(So, Ss)
        Mo = max(Mo, Ms)

        tbwebo = So * 0.5 / (co**2 * tauweb * rh * hboxo * cosL**2)
        con = Mo * 6.0 * hboxo / (co**3 * sigcap * wbox * cosL**4)
        tbcapo = 0.5 * (hrmso - (hrmso**3 - con)**(1.0 / 3.0))
        Abcapo = 2.0 * tbcapo * wbox
        Abwebo = 2.0 * tbwebo * rh * hboxo

        lsp = 0.0
        cosLs = 1.0
    else:
        # ---- strut-braced -------------------------------------------------
        ls = math.sqrt(zs**2 + (0.5 * b * (etas - etao))**2)
        Rstrut = (po * b / 12.0) * (etas - etao) * (1.0 + 2.0 * gammas) + Ss
        Tstrut = Rstrut * ls / zs

        # inboard box sized to the strut-attach loads only
        So = Ss
        Mo = Ms

        tbwebo = So * 0.5 / (co**2 * tauweb * rh * hboxo * cosL**2)
        con = Mo * 6.0 * hboxo / (co**3 * sigcap * wbox * cosL**4)
        tbcapo = 0.5 * (hrmso - (hrmso**3 - con)**(1.0 / 3.0))
        Abcapo = 2.0 * tbcapo * wbox
        Abwebo = 2.0 * tbwebo * rh * hboxo

        lsp = math.sqrt(zs**2 + (0.5 * b * (etas - etao) / cosL)**2)
        Tstrutp = Tstrut * lsp / ls
        cosLs = ls / lsp

    EIco = (Ecap * cop**4 * (hrmso**3 - (hrmso - 2.0 * tbcapo)**3) * wbox / 12.0
            + Eweb * cop**4 * tbwebo * (rh * hboxo)**3 / 6.0)
    EIno = (Ecap * cop**4 * tbcapo * wbox**3 / 6.0
            + Eweb * cop**4 * tbwebo * rh * hboxo * 0.5 * wbox**2)
    GJo = (cop**4 * 2.0 * ((wbox - tbwebo) * (havgo - tbcapo))**2
           / ((rh * hboxo - tbcapo) / (Gweb * tbwebo)
              + (wbox - tbwebo) / (Gcap * tbcapo)))

    Abfuels = (wbox - 2.0 * tbwebs) * (havgs - 2.0 * tbcaps)
    Abfuelo = (wbox - 2.0 * tbwebo) * (havgo - 2.0 * tbcapo)

    Astrut = Tstrutp / sigstrut

    # ---- planform volumes and their moments -------------------------------
    Vcen = co**2 * b * etao / 2.0
    Vinn = (co**2 * b * (etas - etao)
            * (1.0 + lambdas + lambdas**2) / 6.0 * cosL)
    Vout = (co**2 * b * (1.0 - etas)
            * (lambdas**2 + lambdas * lambdat + lambdat**2) / 6.0 * cosL)

    dxVinn = (co**2 * b**2 * (etas - etao)**2
              * (1.0 + 2.0 * lambdas + 3.0 * lambdas**2) / 48.0 * sinL)
    dxVout = (co**2 * b**2 * (1.0 - etas)**2
              * (lambdas**2 + 2.0 * lambdas * lambdat + 3.0 * lambdat**2) / 48.0
              * sinL
              + co**2 * b**2 * (etas - etao) * (1.0 - etas)
              * (lambdas**2 + lambdas * lambdat + lambdat**2) / 12.0 * sinL)

    dyVinn = (co**2 * b**2 * (etas - etao)**2
              * (1.0 + 2.0 * lambdas + 3.0 * lambdas**2) / 48.0 * cosL)
    dyVout = (co**2 * b**2 * (1.0 - etas)**2
              * (lambdas**2 + 2.0 * lambdas * lambdat + 3.0 * lambdat**2) / 48.0
              * cosL)

    # chord^2-weighted mean areas for the inner panel
    Abcapi = (Abcapo + Abcaps * lambdas**2) / (1.0 + lambdas**2)
    Abwebi = (Abwebo + Abwebs * lambdas**2) / (1.0 + lambdas**2)

    Wscen = (rhocap * Abcapo + rhoweb * Abwebo) * gee * Vcen
    Wsinn = (rhocap * Abcapi + rhoweb * Abwebi) * gee * Vinn
    Wsout = (rhocap * Abcaps + rhoweb * Abwebs) * gee * Vout

    dxWsinn = (rhocap * Abcapi + rhoweb * Abwebi) * gee * dxVinn
    dxWsout = (rhocap * Abcaps + rhoweb * Abwebs) * gee * dxVout
    dyWsinn = (rhocap * Abcapi + rhoweb * Abwebi) * gee * dyVinn
    dyWsout = (rhocap * Abcaps + rhoweb * Abwebs) * gee * dyVout

    Abfueli = (Abfuelo + Abfuels * lambdas**2) / (1.0 + lambdas**2)

    Wfcen = rhofuel * Abfuelo * gee * Vcen
    Wfinn = rhofuel * Abfueli * gee * Vinn
    Wfout = rhofuel * Abfuels * gee * Vout
    dxWfinn = rhofuel * Abfueli * gee * dxVinn
    dxWfout = rhofuel * Abfuels * gee * dxVout
    dyWfinn = rhofuel * Abfueli * gee * dyVinn
    dyWfout = rhofuel * Abfuels * gee * dyVout

    Wcap = 2.0 * rhocap * gee * (Abcapo * Vcen + Abcapi * Vinn + Abcaps * Vout)
    Wweb = 2.0 * rhoweb * gee * (Abwebo * Vcen + Abwebi * Vinn + Abwebs * Vout)
    dxWcap = 2.0 * rhocap * gee * (Abcapi * dxVinn + Abcaps * dxVout)
    dxWweb = 2.0 * rhoweb * gee * (Abwebi * dxVinn + Abwebs * dxVout)

    Wstrut = 2.0 * rhostrut * gee * Astrut * lsp
    dxWstrut = Wstrut * 0.25 * b * (etas - etao) * sinL / cosL

    # NOTE: surfw.f recomputes Vout here identically to above before
    # returning. Dead code; not reproduced.

    return SurfaceResult(
        Ss=Ss, Ms=Ms, tbwebs=tbwebs, tbcaps=tbcaps,
        EIcs=EIcs, EIns=EIns, GJs=GJs,
        So=So, Mo=Mo, tbwebo=tbwebo, tbcapo=tbcapo,
        EIco=EIco, EIno=EIno, GJo=GJo,
        Astrut=Astrut, lsp=lsp, cosLs=cosLs,
        Wscen=Wscen, Wsinn=Wsinn, Wsout=Wsout,
        dxWsinn=dxWsinn, dxWsout=dxWsout,
        dyWsinn=dyWsinn, dyWsout=dyWsout,
        Wfcen=Wfcen, Wfinn=Wfinn, Wfout=Wfout,
        dxWfinn=dxWfinn, dxWfout=dxWfout,
        dyWfinn=dyWfinn, dyWfout=dyWfout,
        Wweb=Wweb, Wcap=Wcap, Wstrut=Wstrut,
        dxWweb=dxWweb, dxWcap=dxWcap, dxWstrut=dxWstrut)


__all__ = ["surfw", "SurfaceResult"]
