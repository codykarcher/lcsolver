"""Aircraft drag buildup -- a port of ``cdsum`` and ``cditrp`` from ``cdsum.f``.

Sums every drag contribution at one mission point and writes the result back
into the ``para`` array:

======================  ===========================================
``CDi``                 induced, from the Trefftz plane (``cditrp``)
``CDwing``, ``CDover``  wing profile, and the fuselage carryover
``CDhtail``, ``CDvtail``  tail profile
``CDfuse``              fuselage, from the BL calculation's ``PAfinf``
``CDnace``              nacelle skin friction
``CDstrut``             strut profile
``dCDBLIf``, ``dCDBLIw``  *negative* deltas crediting boundary-layer ingestion
======================  ===========================================

Two things about this routine are worth stating plainly.

The fuselage drag is not computed here
--------------------------------------
``CDfuse = PAfinf/S`` -- it is read straight out of ``para``, having been put
there by the axisymmetric boundary-layer solve in ``fusebl``. The wetted-area
route through ``bodycd`` is still in the source but commented out, so
``bodycd`` is dead code. That means a drag buildup done without first running
the BL solve reports zero fuselage drag rather than failing.

BLI is credited, not modelled
-----------------------------
``dCDBLIf`` and ``dCDBLIw`` are negative drag increments proportional to the
fraction of each wake ingested. The wing term uses

    ``CDWwake = CDwing * 0.15``

with the comment "assume 15% of the wing dissipation is in wake" -- a flat
constant, not derived from anything. It sets how much wing drag BLI can
recover, so it matters whenever ``fBLIw`` is nonzero.

``icdfun``
----------
With ``icdfun == 1`` the wing section drag is looked up from the airfoil
database by spanwise quadrature (``surfcd2``). Otherwise the stored
``cdfw``/``cdpw`` in ``para`` are reused and only scaled -- which is what the
inner iterations do, to avoid re-interpolating the tables every pass.

Verified against the compiled Fortran; see ``tests/test_cdsum.py``.
"""
from __future__ import annotations

import math

from ..model import indices as I
from .airfoil import airfun
from .drag import surfcd, surfcd2
from .loading import wingcl
from .trefftz import trefftz1

__all__ = ["cdsum", "cditrp", "cfturb", "WING_WAKE_FRACTION"]

# Fraction of wing dissipation assumed to end up in the wake, and so available
# to be recovered by ingestion. A bare constant in the source.
WING_WAKE_FRACTION = 0.15

# Trefftz-plane panelling, fixed in cditrp.
NPOUT_WING, NPINN_WING, NPIMG_WING = 20, 6, 3
NPOUT_TAIL, NPINN_TAIL, NPIMG_TAIL = 10, 0, 2
KTIP = 16
#: The wing root streamline contracts to this fraction of bo in the wake.
BOP_FRACTION = 0.2


def cfturb(re: float) -> float:
    """Turbulent flat-plate skin friction against Re_l -- White's fit.

    ``wsize.f`` carries two other correlations above this one, both commented
    out: Hoerner's ``0.427/(log10(Re) - 0.407)^2.64`` and a modified version
    with a weaker Re dependence. The live one is White's.
    """
    return 0.523 / math.log(0.06 * re) ** 2


def cditrp(pari, parg, para) -> None:
    """Induced drag from the Trefftz plane; writes ``CDi`` and ``spaneff``.

    Wing and horizontal tail are solved together, so the induced drag includes
    their mutual interference. Both loadings are rescaled to the lift each
    surface is actually carrying (``Lspec`` true), so only the *shape* of the
    assumed loading matters.
    """
    CL = para[I.IACL]
    if CL == 0.0:
        # No lift, no induced drag -- and a span efficiency of 1 by convention,
        # since the ratio that defines it is 0/0.
        para[I.IACDI] = 0.0
        para[I.IASPANEFF] = 1.0
        return

    CLhtail = para[I.IACLH] * parg[I.IGSH] / parg[I.IGS]

    b = [parg[I.IGB], parg[I.IGBH]]
    bs = [parg[I.IGBS], parg[I.IGBOH]]
    bo = [parg[I.IGBO], parg[I.IGBOH]]
    bop = [parg[I.IGBO] * BOP_FRACTION, parg[I.IGBOH]]
    zcent = [parg[I.IGZWING], parg[I.IGZHTAIL]]
    gammas = [parg[I.IGLAMBDAS] * para[I.IARCLS], 1.0]
    gammat = [parg[I.IGLAMBDAT] * para[I.IARCLT], parg[I.IGLAMBDAH]]
    po = [1.0, 1.0]
    CLsurfsp = [CL - CLhtail, CLhtail]

    npout = [NPOUT_WING, NPOUT_TAIL]
    npinn = [NPINN_WING, NPINN_TAIL]
    # A T-tail has no fuselage to put an image in.
    npimg = [NPIMG_WING, 0 if bo[1] == 0.0 else NPIMG_TAIL]

    r = trefftz1(2, npout, npinn, npimg, parg[I.IGS], parg[I.IGB],
                 b, bs, bo, bop, zcent, po, gammat, gammas,
                 parg[I.IGFLO], KTIP, True, CLsurfsp)

    para[I.IACDI] = r.CD
    para[I.IASPANEFF] = r.spanef


def cdsum(pari, parg, para, pare, icdfun: int, table=None) -> float:
    """Total aircraft CD at one mission point. Returns it and stores it.

    ``table`` is an :class:`~tasopt_py.aero.airfoil.AirfoilTable`; it is only
    needed when ``icdfun == 1``.
    """
    S = parg[I.IGS]
    b, bs, bo = parg[I.IGB], parg[I.IGBS], parg[I.IGBO]
    co = parg[I.IGCO]
    sweep = parg[I.IGSWEEP]
    lambdat, lambdas = parg[I.IGLAMBDAT], parg[I.IGLAMBDAS]
    gammat = parg[I.IGLAMBDAT] * para[I.IARCLT]
    gammas = parg[I.IGLAMBDAS] * para[I.IARCLS]
    hboxo, hboxs = parg[I.IGHBOXO], parg[I.IGHBOXS]
    hboxt = hboxs
    fLo, fLt = parg[I.IGFLO], parg[I.IGFLT]

    CL, Mach = para[I.IACL], para[I.IAMACH]
    fduo, fdus, fdut = para[I.IAFDUO], para[I.IAFDUS], para[I.IAFDUT]
    fexcdw = para[I.IAFEXCDW]
    CLhtail = para[I.IACLH] * parg[I.IGSH] / S

    Reunit = para[I.IAREUNIT]
    Rerefw, Rereft = para[I.IAREREFW], para[I.IAREREFT]
    aRexp = para[I.IAAREXP]

    # Root shock-unsweep constant. The wing gets 0.5; both tails get zero,
    # so their sections are treated as fully swept all the way to the root.
    rkSunsw = 0.5
    rkSunsh = rkSunsv = 0.0
    # Only the horizontal tail gets a nonzero carryover factor.
    fCDwcen = fCDvcen = 0.0
    fCDhcen = parg[I.IGFCDHCEN]

    Reco = Reunit * co

    if icdfun == 1:
        if table is None:
            raise ValueError("cdsum: icdfun=1 needs an airfoil table")

        def airfoil(cl, toc, Mperp):
            g = airfun(cl, toc, Mperp, table)
            return g.cdf, g.cdp, g.cdw, g.cm

        w = surfcd2(S, b, bs, bo, lambdat, lambdas, gammat, gammas,
                    hboxo, hboxs, hboxt, Mach, sweep, co, CL, CLhtail,
                    fLo, fLt, Reco, aRexp, rkSunsw, fexcdw,
                    table.ARe, airfoil, fduo, fdus, fdut)
        CDwing, CDover = w.CDwing, w.CDover
        cdfw, cdpw = w.CDfwing, w.CDpwing
        clpo, clps, clpt = w.clpo, w.clps, w.clpt
        para[I.IACDFW] = cdfw
        para[I.IACDPW] = cdpw
    else:
        # Reuse the stored section drag; only the excrescence factor is new.
        cdfw = para[I.IACDFW] * fexcdw
        cdpw = para[I.IACDPW] * fexcdw
        w = surfcd(S, b, bs, bo, lambdat, lambdas, sweep, co,
                   cdfw, cdpw, Reco, Rerefw, aRexp, rkSunsw, fCDwcen)
        CDwing, CDover = w.CDsurf, w.CDover
        cl = wingcl(b, bs, bo, lambdat, lambdas, gammat, gammas,
                    sweep, parg[I.IGAR], CL, CLhtail, fLo, fLt,
                    fduo, fdus, fdut)
        clpo, clps, clpt = cl.clo, cl.cls, cl.clt

    para[I.IACDWING] = CDwing
    para[I.IACDOVER] = CDover
    para[I.IACLPO] = clpo
    para[I.IACLPS] = clps
    para[I.IACLPT] = clpt

    # --- tails ------------------------------------------------------------
    # Both tails use the same section drag, scaled by one excrescence factor.
    cdft = para[I.IACDFT] * para[I.IAFEXCDT]
    cdpt = para[I.IACDPT] * para[I.IAFEXCDT]

    h = surfcd(S, parg[I.IGBH], parg[I.IGBOH], parg[I.IGBOH],
               parg[I.IGLAMBDAH], 1.0, parg[I.IGSWEEPH], parg[I.IGCOH],
               cdft, cdpt, Reunit * parg[I.IGCOH], Rereft, aRexp,
               rkSunsh, fCDhcen)
    CDhtail = h.CDsurf

    v = surfcd(S, parg[I.IGBV], parg[I.IGBOV], parg[I.IGBOV],
               parg[I.IGLAMBDAV], 1.0, parg[I.IGSWEEPV], parg[I.IGCOV],
               cdft, cdpt, Reunit * parg[I.IGCOV], Rereft, aRexp,
               rkSunsv, fCDvcen)
    CDvtail = v.CDsurf * parg[I.IGNVTAIL]

    para[I.IACDHTAIL] = CDhtail
    para[I.IACDVTAIL] = CDvtail

    # --- fuselage: taken from the BL solve, not computed here -------------
    CDfuse = para[I.IAPAFINF] / S
    para[I.IACDFUSE] = CDfuse

    # --- nacelle ----------------------------------------------------------
    lnace = parg[I.IGLNACE]
    if Reunit == 0.0:
        Cfnace = 0.0
    else:
        Cfnace = cfturb(Reunit * lnace) * para[I.IAFEXCDF]
    para[I.IACFNACE] = Cfnace

    # Surface speed from a vortex-sheet model of the nacelle: the mean of the
    # leading-edge and mid-nacelle overspeeds satisfies
    # (rVnLE + rVnace)/2 = Mach*rVnace / Mach, hence the rearrangement.
    rVnace = parg[I.IGRVNACE]
    rVnLE = max(2.0 * rVnace - pare[I.IEM2] / max(Mach, 0.001), 0.0)
    rVnsurf3 = 0.25 * (rVnLE + rVnace) * (rVnLE ** 2 + rVnace ** 2)
    CDnace = parg[I.IGFSNACE] * Cfnace * rVnsurf3
    para[I.IACDNACE] = CDnace

    # --- strut ------------------------------------------------------------
    cosLs = parg[I.IGCOSLS]
    CDstrut = ((parg[I.IGSSTRUT] / S)
               * (para[I.IACDFS] + para[I.IACDPS] * cosLs ** 3)
               * parg[I.IGRVSTRUT] ** 3)
    para[I.IACDSTRUT] = CDstrut

    # --- induced ----------------------------------------------------------
    cditrp(pari, parg, para)
    CDi = para[I.IACDI]

    # --- boundary-layer ingestion credits, both negative ------------------
    dCDBLIf = -parg[I.IGFBLIF] * para[I.IADAFWAKE] / S
    CDWwake = CDwing * WING_WAKE_FRACTION
    dCDBLIw = -parg[I.IGFBLIW] * CDWwake

    CD = (CDi + CDfuse + CDwing + CDover + CDhtail + CDvtail + CDstrut
          + CDnace + dCDBLIf + dCDBLIw)
    para[I.IACD] = CD
    return CD
