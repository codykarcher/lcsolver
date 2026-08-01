"""TASOPT's station-based wing box, ported from ``surfw.f``.

An alternative to the closed-form box in ``wingbox.py`` (Hoburg 2014, which is
what SPaircraft uses). Both are kept: pass ``box_model="hoburg"`` or
``"tasopt"`` and nothing else in the model changes.

Why this exists
---------------
Audited against TASOPT's own 737 at matched inputs -- same sigma_cap (206.9
MPa), N_lift 3.0, N_land 6.0, V_ne, wing area to 1.1% and span to 1.4% -- the
closed-form box comes out

    W_cap   16,905 lb   against TASOPT's 13,820    x1.22
    W_web      885 lb   against            642     x1.38
    W_wing  29,176 lb   against         23,717     x1.23

and our box is 10% THICKER (tau 0.14 against hboxo 0.1268), which should make
it lighter, so like-for-like the gap is nearer 35%. The non-dimensional cap
thickness tells the same story directly: 0.00569 here against TASOPT's
0.00342.

The two are not the same equations with different constants. Hoburg's is a
closed-form span integration over a SINGLE-taper wing; TASOPT sizes discrete
stations on a CRANKED planform (root, break, tip) and integrates real box
volumes. That difference is the 22%, and it matters for this study because
sweep is priced through the box: TASOPT reaches 26 degrees WITH its structural
penalty active, while the closed form needs the penalty off to get there.

Validation
----------
Fed TASOPT's own 737 loads and geometry, this reproduces its published station
values:

    tcapo/c  0.0034238  vs 0.00342     twebo/c  0.00089258 vs 0.00089
    tcaps/c  0.0064584  vs 0.00647     twebs/c  0.0015278  vs 0.00153
    W_cap    13,801 lb  vs 13,819.8    W_web    641.95 lb  vs 642.0

Residuals are rounding in TASOPT's 4-figure output.

The one non-convexity, and how it is handled
--------------------------------------------
``surfw.f`` sizes the cap as

    tbcap = 0.5*(hrms - (hrms**3 - con)**(1/3))

which is a difference inside a cube root -- not a posynomial. Substituting
``u = hrms - 2*tbcap`` and cubing gives, exactly,

    con + 12*hrms*tcap**2  ==  6*hrms**2*tcap + 8*tcap**3

with every term positive: a signomial equality, which is the form this model
already uses throughout. No approximation is involved; it is algebraically
identical to the Fortran.

What is NOT ported yet
----------------------
The load-relief terms. ``surfw.f`` subtracts ``Nload*Winn``, ``Nload*Wout``
and ``Nload*We`` -- the inertial relief of the wing's own structure and of a
wing-mounted engine -- from the shear and moment. TASOPT converges these by
outer iteration; in an SP they would be variables in a fixed point, which the
solver can do but which is a second piece of work.

Omitting relief makes this box CONSERVATIVE: loads are higher than TASOPT's,
so the wing comes out heavier than it should. On the 737 the engine term alone
is ``Nload*We`` ~ 18,400 lb against a break shear of 123,500, so this is not a
small omission and the port should not be read as final until it is in.
``W_relief_inn``, ``W_relief_out`` and ``W_engine`` are accepted here so the
wiring is a caller change rather than an edit to this file.
"""
from __future__ import annotations

import numpy as np

#: TASOPT 737 deck values, used as defaults. These describe a transport
#: planform, not this particular aeroplane, but they are the calibrated set
#: and a class that wants its own should pass them.
ETA_O = 0.1016      # bo/b, centrebody fraction (11.833 ft / 116.427 ft)
ETA_S = 0.285       # bs/b, planform break
LAMBDA_S = 0.700    # break/root chord ratio
LAMBDA_T = 0.250    # tip/root chord ratio
#: Spanwise cl ratios from the deck's STRUCTURAL sizing block ("clean climb,
#: cruise, descent, also for wing structure sizing") -- NOT the takeoff block,
#: which carries 1.1/0.6 and would size the box on the wrong case.
RCLS = 1.238
RCLT = 0.90
F_LO = -0.3         # fuselage lift carryover loss
F_LT = -0.05        # tip lift rolloff
R_H = 0.75          # web height / box height
W_BOX = 0.50        # box width / chord


def planform(eta_o=ETA_O, eta_s=ETA_S, lam_s=LAMBDA_S, lam_t=LAMBDA_T):
    """``(Kc, Kp)`` -- the chord and load distribution integrals, wingpo.f."""
    gam_s, gam_t = lam_s * RCLS, lam_t * RCLT
    Kc = (eta_o + 0.5 * (1.0 + lam_s) * (eta_s - eta_o)
          + 0.5 * (lam_s + lam_t) * (1.0 - eta_s))
    Ko = 1.0 / Kc          # AR factored out; caller divides by AR
    Kp = (eta_o + 0.5 * (1.0 + gam_s) * (eta_s - eta_o)
          + 0.5 * (gam_s + gam_t) * (1.0 - eta_s)
          + F_LO * eta_o)   # the fLt term needs AR; added by the caller
    return Kc, Ko, Kp, gam_s, gam_t


def h_rms(hbox, rh=R_H):
    """Root-mean-square box height, surfw.f:45."""
    return hbox * np.sqrt(1.0 - (1.0 - rh) / 1.5 + (1.0 - rh) ** 2 / 5.0)


def h_avg(hbox, rh=R_H):
    """Mean box height, surfw.f:42."""
    return hbox * (1.0 - (1.0 - rh) / 3.0)


def volumes(co, b, cosL, eta_o=ETA_O, eta_s=ETA_S,
            lam_s=LAMBDA_S, lam_t=LAMBDA_T):
    """``(Vcen, Vinn, Vout)`` box volumes, surfw.f:154-161.

    Note ``Vcen`` carries NO ``cosL``: the centre section runs through the
    fuselage and is not swept. Taking the thickness terms' ``1/cos^2`` and
    ``1/cos^4`` without this compensating ``cosL`` over-charges sweep by a
    full power in each row.
    """
    Vcen = co ** 2 * b * eta_o / 2.0
    Vinn = (co ** 2 * b * (eta_s - eta_o)
            * (1.0 + lam_s + lam_s ** 2) / 6.0 * cosL)
    Vout = (co ** 2 * b * (1.0 - eta_s)
            * (lam_s ** 2 + lam_s * lam_t + lam_t ** 2) / 6.0 * cosL)
    return Vcen, Vinn, Vout


def add_wingbox_tasopt(surfacetype, *, AR, b, S, tau, Lmax, group,
                       N_lift=3.0, cosL=None, material=None,
                       eta_o=ETA_O, eta_s=ETA_S,
                       lam_s=LAMBDA_S, lam_t=LAMBDA_T,
                       W_relief_inn=None, W_relief_out=None, W_engine=None):
    """Add the station-based box. Returns ``(vars, constraints)``.

    ``Lmax`` is the maximum net load the box carries (``N*W - L_htail`` in
    TASOPT's terms); ``N_lift`` is applied by the caller in forming it, as in
    the closed-form box, so it is not applied again here.
    """
    from .wingbox import ALUMINIUM
    material = material or ALUMINIUM
    V, C = group.Variable, group.Constant

    _cosL = cosL if cosL is not None else 1.0
    Kc, Ko, Kp0, gam_s, gam_t = planform(eta_o, eta_s, lam_s, lam_t)
    # the fLt term in Kp carries 1/AR, which is a variable here
    hrms_f = h_rms(1.0)          # multiplies tau below
    rh, wbox = R_H, W_BOX

    # ---- variables --------------------------------------------------------
    co = V("c_o", 5.0, "m", "root chord (TASOPT box station)")
    po = V("p_o_box", 1e5, "N/m", "root load intensity")
    So = V("S_o", 6e5, "N", "root shear")
    Ss = V("S_s", 5e5, "N", "break shear")
    Mo = V("M_o", 4e6, "N*m", "root bending moment")
    Ms = V("M_s", 2e6, "N*m", "break bending moment")
    tcapo = V("t_cap_o", 0.0034, "-", "non-dim cap thickness, root")
    tcaps = V("t_cap_s", 0.0065, "-", "non-dim cap thickness, break")
    twebo = V("t_web_o", 0.0009, "-", "non-dim web thickness, root")
    twebs = V("t_web_s", 0.0015, "-", "non-dim web thickness, break")
    Wcap = V("W_cap", 6e4, "N", "weight of spar caps")
    Wweb = V("W_web", 3e3, "N", "weight of shear webs")
    Wstruct = V("W_struct", 7e4, "N", "structural weight")

    g = C("g", 9.81, "m/s^2", "gravitational acceleration")
    rhocap = C("rho_cap", material["rho"], "kg/m^3", "spar cap density")
    rhoweb = C("rho_web", material["rho"], "kg/m^3", "shear web density")
    sigcap = C("sigma_max", material["sigma"], "Pa", "allowable tensile stress")
    tauwebC = C("sigma_max_shear", material["tau"], "Pa",
                "allowable shear stress")
    wwb = C("r_w_c", wbox, "-", "wingbox width-to-chord ratio")

    out = dict(r_w_c=wwb, W_cap=Wcap, W_web=Wweb, W_struct=Wstruct,
               t_cap=tcapo, t_web=twebo, c_o=co, M_r=Mo,
               S_o=So, M_o=Mo, S_s=Ss, M_s=Ms)

    hbox = tau                      # box height / chord == airfoil t/c
    hrms = hrms_f * hbox
    cs = co * lam_s
    Vcen, Vinn, Vout = volumes(co, b, _cosL, eta_o, eta_s, lam_s, lam_t)

    # relief terms; omitted by default -- see the module docstring
    _zero = 0.0 * Lmax
    Rin = W_relief_inn if W_relief_inn is not None else _zero
    Rout = W_relief_out if W_relief_out is not None else _zero
    Reng = W_engine if W_engine is not None else _zero

    cons = [
        AR == b ** 2 / S,
        # chord distribution, wingpo.f: S = co*b*Kc
        S == co * b * Kc,
        # root load intensity. The fLt tip-rolloff term carries 1/AR and is
        # negative, so it moves to the left as an all-positive equality.
        po * Kp0 * b + (-2.0 * F_LT * Ko * gam_t * lam_t / AR) * po * b
            == Lmax,                                          # [SP] SigEq

        # ---- outer wing, at the break (surfw.f:50-51) ---------------------
        # dLt = fLt*po*co*gammat*lambdat is NEGATIVE (fLt = -0.05), so it and
        # the relief both move left to keep every coefficient positive.
        Ss + (-F_LT) * po * co * gam_t * lam_t + N_lift * Rout
            == (po * b / 4.0) * (gam_s + gam_t) * (1.0 - eta_s),  # [SP] SigEq
        Ms + (-F_LT) * po * co * gam_t * lam_t * 0.5 * b * (1.0 - eta_s)
            == (po * b ** 2 / 24.0) * (gam_s + 2.0 * gam_t)
               * (1.0 - eta_s) ** 2,                              # [SP] SigEq

        # ---- root station (surfw.f:88-96) ---------------------------------
        So + N_lift * Reng + N_lift * Rin
            == Ss + 0.25 * po * b * (1.0 + gam_s) * (eta_s - eta_o),
        Mo == Ms + Ss * 0.5 * b * (eta_s - eta_o)
              + (1.0 / 24.0) * po * b ** 2 * (1.0 + 2.0 * gam_s)
                * (eta_s - eta_o) ** 2,                           # [SP] SigEq
        # surfw.f:102-103 limits So,Mo to at least the break values
        So >= Ss,
        Mo >= Ms,

        # ---- station sizing, surfw.f:54-57 and 106-109 --------------------
        twebo == So * 0.5 / (co ** 2 * tauwebC * rh * hbox * _cosL ** 2),
        twebs == Ss * 0.5 / (cs ** 2 * tauwebC * rh * hbox * _cosL ** 2),
        # The cap cubic, exactly: con + 12*hrms*t^2 == 6*hrms^2*t + 8*t^3
        (Mo * 6.0 * hbox / (co ** 3 * sigcap * wwb * _cosL ** 4)
         + 12.0 * hrms * tcapo ** 2
         == 6.0 * hrms ** 2 * tcapo + 8.0 * tcapo ** 3),          # [SP] SigEq
        (Ms * 6.0 * hbox / (cs ** 3 * sigcap * wwb * _cosL ** 4)
         + 12.0 * hrms * tcaps ** 2
         == 6.0 * hrms ** 2 * tcaps + 8.0 * tcaps ** 3),          # [SP] SigEq

        # ---- weights, surfw.f:181-209 ------------------------------------
        # Abcap = 2*tcap*wbox, Abweb = 2*tweb*rh*hbox, and the inner panel is
        # the lambda^2-weighted blend of root and break (surfw.f:181-182).
        Wcap >= 2.0 * rhocap * g * (
            2.0 * tcapo * wwb * Vcen
            + (2.0 * tcapo * wwb + 2.0 * tcaps * wwb * lam_s ** 2)
              / (1.0 + lam_s ** 2) * Vinn
            + 2.0 * tcaps * wwb * Vout),
        Wweb >= 2.0 * rhoweb * g * (
            2.0 * twebo * rh * hbox * Vcen
            + (2.0 * twebo * rh * hbox + 2.0 * twebs * rh * hbox * lam_s ** 2)
              / (1.0 + lam_s ** 2) * Vinn
            + 2.0 * twebs * rh * hbox * Vout),
        Wstruct >= Wcap + Wweb,
    ]
    return out, cons
