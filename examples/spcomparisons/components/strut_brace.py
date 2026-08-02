"""The wing strut, ported from ``surfw.f``'s ``iwplan=2`` branch.

What a strut changes, in TASOPT's model
---------------------------------------
Three things, all small in code and one large in consequence:

1. THE ROOT STOPS ACCUMULATING MOMENT. surfw.f:129-130 sets ``So = Ss`` and
   ``Mo = Ms``: the inner panel is sized by the STRUT-ATTACH station's loads
   instead of by the cantilever build-up from break to root. That is the whole
   structural payoff, and it lives in ``wingbox_tasopt.py`` (``strut=True``),
   not here -- the box owns its stations.

2. A TENSION MEMBER APPEARS. The strut carries the entire inner-panel lift
   plus the outboard shear as a reaction (surfw.f:119)

       Rstrut = (po*b/12)*(etas - etao)*(1 + 2*gammas) + Ss

   resolved along the strut axis, whose length follows from the vertical base
   ``zs`` (deck: 154 in) and the attach half-span. TASOPT sizes it in pure
   tension against ``sigstrut`` -- no buckling case in the sizing loop (the
   nonlinear beam-column study in ``strut.f`` is a standalone driver that
   never gets called) -- and this port keeps that basis.

3. THE STRUT HAS DRAG. wsize.f:994-998 derives a chord from the tension area
   through a fixed section (t/c ``hstrut`` = 0.15, shell t/h = 0.05,
   wsize.f:239) and cdsum.f:295-300 charges

       CDstrut = (Sstrut/S) * (cdfs + cdps*cosLs^3) * rVstrut^3

   on wing area. Deck constants cdfs = 0.0085, cdps = 0.0035, rVstrut = 1.0.

What TASOPT does NOT model here, so neither do we: strut buckling (tension
sizing only, which is why the strut is light), strut-wing interference drag
beyond the cd constants, and the high-wing fuselage rearrangement a real
strut-braced layout implies. The attach station is the PLANFORM BREAK
``eta_s`` -- TASOPT overloads one deck input for both -- so moving the strut
outboard (a SUGAR-style 55% attach) means moving the break with it.

No reference case exists: none of the shipped decks runs iwplan=2, so unlike
the cantilever box (validated to 0.1% against the 737 printout) this is a port
of the equations without a numbers-level check against their author's output.

Convexity
---------
Every row is GP-clean except the strut-length equality:

* ``ls`` (vertical-plane length) appears only in ``cosLs = ls/lsp``, and drag
  grows with cosLs, so the optimiser pushes ls DOWN -- the one-sided
  ``ls^2 >= zs^2 + (b*deta/2)^2`` floor is exactly what it lands on.
* ``lsp`` (true length) has MIXED pressure -- weight and tension want it
  short, the cosLs denominator wants it long -- so it must be an equality.
"""
from __future__ import annotations

#: Deck values, common to every TASOPT 2.16 run deck (737.tas etc.).
Z_S = 154.0 * 0.0254        # m, strut vertical base ("zs")
H_STRUT = 0.15              # strut section t/c ("hstrut")
TOH_STRUT = 0.05            # shell thickness / section height (wsize.f:239)
CDF_STRUT = 0.0085          # strut friction cd ("cdfs")
CDP_STRUT = 0.0035          # strut pressure cd ("cdps")
RV_STRUT = 1.0              # local/freestream velocity ratio ("rVstrut")
from .wingbox_tasopt import RCLS


def add_strut_brace(wing, *, po, b, deta, lam_s, Ss, S, cosL,
                    material=None, z_s=Z_S, prefix="Strut_"):
    """Add the strut. Returns ``(group, constraints)``.

    ``po``, ``deta``, ``Ss`` are the BOX's load intensity, inner-panel span
    fraction and break shear (wingbox_tasopt exposes them); ``lam_s`` the
    break taper, ``S``/``b``/``cosL`` the wing's. The caller charges
    ``W_strut`` into the wing weight row and ``C_D_strut`` into the wing's
    drag coefficient -- this module computes, the wing pays.
    """
    from .wingbox import ALUMINIUM
    material = material or ALUMINIUM
    sb = wing.group("strut", prefix=prefix)
    V, C = sb.Variable, sb.Constant

    Rstrut = V("R_strut", 6e5, "N", "strut reaction at the attach station")
    Tstrutp = V("T_strut", 9e5, "N", "strut axial tension, true length")
    Astrut = V("A_strut", 4e-3, "m^2", "strut cross-section area")
    ls = V("l_s", 5.0, "m", "strut length in the vertical plane")
    lsp = V("l_sp", 5.2, "m", "strut true length (swept)")
    cosLs = V("cos_Lambda_s", 0.97, "-", "cosine of the strut sweep")
    cstrut = V("c_strut", 0.55, "m", "strut chord")
    Sstrut = V("S_strut", 5.5, "m^2", "strut planform area, both sides")
    Wstrut = V("W_strut", 1.5e3, "N", "strut weight, both sides")
    CDstrut = V("C_D_strut", 5e-4, "-", "strut profile drag on wing area")

    zs = C("z_s", z_s, "m", "strut vertical base (deck zs)")
    sigstrut = C("sigma_strut", material["sigma"], "Pa",
                 "strut allowable tensile stress")
    rhostrut = C("rho_strut", material["rho"], "kg/m^3", "strut density")
    hstrut = C("h_strut", H_STRUT, "-", "strut section t/c")
    toh = C("t_o_h_strut", TOH_STRUT, "-", "strut shell thickness/height")
    cdfs = C("cd_f_strut", CDF_STRUT, "-", "strut friction cd")
    cdps = C("cd_p_strut", CDP_STRUT, "-", "strut pressure cd")
    rV = C("rV_strut", RV_STRUT, "-", "strut local velocity ratio")
    g = C("g", 9.81, "m/s^2", "gravitational acceleration")

    gam_s = lam_s * RCLS
    cons = [
        # surfw.f:119. One-sided: tension, area, weight and chord (hence
        # drag) all grow with Rstrut, so the pressure is downward onto it.
        Rstrut >= (po * b / 12.0) * deta * (1.0 + 2.0 * gam_s) + Ss,
        # Strut geometry. ls's only consumer is the cosLs numerator and drag
        # grows with cosLs, so the floor binds (see module docstring); lsp
        # carries mixed pressure and is pinned.
        ls ** 2 >= zs ** 2 + (0.5 * b * deta) ** 2,
        lsp ** 2 == zs ** 2 + (0.5 * b * deta / cosL) ** 2,    # [SP] SigEq
        cosLs * lsp == ls,
        # surfw.f:120,136,152: T = R*ls/zs along the vertical-plane strut,
        # stretched to the true length -- the two ls factors cancel, so the
        # true-axis tension is R*lsp/zs directly.
        Tstrutp == Rstrut * lsp / zs,
        Astrut == Tstrutp / sigstrut,
        # surfw.f:214: both struts.
        Wstrut == 2.0 * rhostrut * g * Astrut * lsp,
        # wsize.f:995-996: chord from the tension area through the fixed
        # section, planform area over both struts.
        cstrut ** 2 == 0.5 * Astrut / (toh * hstrut),
        Sstrut == 2.0 * cstrut * lsp,
        # cdsum.f:299, one-sided: drag pressure pushes CDstrut down onto it.
        CDstrut * S >= Sstrut * rV ** 3 * (cdfs + cdps * cosLs ** 3),
    ]
    return sb, cons
