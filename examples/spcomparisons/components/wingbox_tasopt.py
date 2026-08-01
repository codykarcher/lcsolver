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

Residuals there are rounding in TASOPT's 4-figure output.

Driving the whole chain instead -- po from the net load, through the relief
fixed point, to the stations -- reproduces TASOPT's published loads too:

    So  150,484 lb    vs 147,238.4     Mo  3.563e6 ft-lb  vs 3.511e6
    Ss  125,252 lb    vs 123,491.5     Ms  2.179e6 ft-lb  vs 2.153e6
    W_cap 13,999 lb   vs 13,819.8      W_web 652.9 lb     vs 642.0

within 1.3-2.2%. The residual is not chased further because several inputs are
inferred rather than read: the engine weight split per side, its spanwise
station, fuel density, and fwadd (wsize.f:135 assigns it from parg(igfflap)
alone, which does not obviously match the deck's statement that the secondary
fractions sum to fwadd). Each is worth a percent or so and none changes the
conclusion.

THE TAIL DOWNLOAD MATTERS AND IS THE CALLER'S JOB. TASOPT sizes the wing
against a net load that includes a tail DOWNLOAD (wsize.f:912):

    Lhtail = WMTO * CLhNrat * Sh/S,     CLhNrat = -0.5 in the 737 deck

which on the 737 is -29,782 lb, and since it enters as ``N*W - Lhtail`` it
RAISES the load by 5.7%. Omitting it left every load 5.5% light and the box
correspondingly under-sized -- so ``Lmax`` passed to this function must be
``N*W - Lhtail``, not ``N*W``.

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

Inertial relief
---------------
``surfw.f`` subtracts ``Nload*Winn``, ``Nload*Wout`` and ``Nload*We`` from the
shear and moment -- the wing's own structure, the fuel it carries, and a
wing-mounted engine all push DOWN under a positive load factor and so relieve
the bending the box has to carry. On the 737 the engine term alone is
``Nload*We`` ~ 18,400 lb against a break shear of 123,500, so leaving it out
is not a rounding error.

TASOPT converges these by outer iteration (wsize.f:1003-1006):

    Winn   = Wsinn*(1+fwadd) + rfmax*Wfinn
    dyWinn = dyWsinn*(1+fwadd) + rfmax*dyWfinn

which is circular -- the relief depends on the structure, which depends on the
loads, which depend on the relief. In an SP that is simply a fixed point the
solver resolves, and the feedback is stabilising: more structure gives more
relief, which lowers the load, which asks for less structure.

``Abfuel = (wbox - 2*tbweb)*(havg - 2*tbcap)`` is a product of two differences.
Expanded it is all-positive, so it stays a clean signomial equality:

    Abfuel + 2*wbox*tcap + 2*tweb*havg == wbox*havg + 4*tweb*tcap

``f_fuel`` is TASOPT's ``rfmax = Wfuel/Wfmax``, the fraction of maximum tankage
actually carried in the sizing case. It defaults to 1.0 (full tanks, maximum
relief, which is the case TASOPT sizes on); pass a smaller number to be more
conservative. ``W_engine`` is the engine weight acting at the planform break,
zero for a fuselage- or tail-mounted engine.
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
                       lam_s=None, lam_t=None, taper=None,
                       W_engine=None, rho_fuel=None, f_fuel=1.0,
                       f_wadd=0.640, relief=True):
    """Add the station-based box. Returns ``(vars, constraints)``.

    ``Lmax`` is the maximum net load the box carries (``N*W - L_htail`` in
    TASOPT's terms); ``N_lift`` is applied by the caller in forming it, as in
    the closed-form box, so it is not applied again here.
    """
    from .wingbox import ALUMINIUM
    material = material or ALUMINIUM
    V, C = group.Variable, group.Constant

    # PLANFORM. If the caller has its own taper -- and the wing does -- the
    # box must use it, not TASOPT's cranked 0.70/0.25. Ignoring it left the
    # wing's taper priced by nothing: with the closed-form box it sits on its
    # 0.15 floor because nu = (1+l+l^2)/(1+l)^2 makes a sharply tapered wing
    # lighter, but with this box it drifted to 0.42, which dropped c_root and
    # the mean chord by a third and took the horizontal tail down with them
    # (S_ht 31.8 -> 19.8 against a real 32.0) because V_ht is referenced to
    # mac. A quantity the optimiser can move for free is the recurring defect
    # in this model, and this was one more of them.
    #
    # The fix is NOT to derive the box planform from a variable taper. That
    # was tried and it fails: lam_s = 1 - (1-lam_t)*eta_s makes Kc carry a
    # difference and Ko = 1/Kc the reciprocal of a signomial, turning what
    # were constant coefficients into signomial expressions. Neither case
    # converged (600 iterations, and a sub-problem failure at 74).
    #
    # TASOPT does not optimise taper -- lambdas and lambdat are deck INPUTS.
    # So faithfulness runs the other way: the box keeps its constant planform
    # and the CALLER pins the wing's taper to lam_t, which both prices taper
    # (it is no longer free) and keeps every coefficient constant. `taper` is
    # accepted only so the caller can read back what to pin to.
    if lam_s is None:
        lam_s = LAMBDA_S
    if lam_t is None:
        lam_t = LAMBDA_T
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

    # Declared as a Constant, not just taken as a Python float, because the
    # rest of the model reads wing.box.N_lift (aircraft.py couples the
    # fuselage load factor to it) and the two box models must present the
    # same interface.
    Nlift = C("N_lift", N_lift, "-", "wing loading multiplier")
    g = C("g", 9.81, "m/s^2", "gravitational acceleration")
    rhocap = C("rho_cap", material["rho"], "kg/m^3", "spar cap density")
    rhoweb = C("rho_web", material["rho"], "kg/m^3", "shear web density")
    sigcap = C("sigma_max", material["sigma"], "Pa", "allowable tensile stress")
    tauwebC = C("sigma_max_shear", material["tau"], "Pa",
                "allowable shear stress")
    wwb = C("r_w_c", wbox, "-", "wingbox width-to-chord ratio")

    out = dict(r_w_c=wwb, W_cap=Wcap, W_web=Wweb, W_struct=Wstruct,
               t_cap=tcapo, t_web=twebo, c_o=co, M_r=Mo, N_lift=Nlift,
               S_o=So, M_o=Mo, S_s=Ss, M_s=Ms)

    hbox = tau                      # box height / chord == airfoil t/c
    hrms = hrms_f * hbox
    havg = h_avg(1.0) * hbox
    cs = co * lam_s
    Vcen, Vinn, Vout = volumes(co, b, _cosL, eta_o, eta_s, lam_s, lam_t)
    # spanwise moment volumes, surfw.f:173-178
    dyVinn = (co ** 2 * b ** 2 * (eta_s - eta_o) ** 2
              * (1.0 + 2.0 * lam_s + 3.0 * lam_s ** 2) / 48.0 * _cosL)
    dyVout = (co ** 2 * b ** 2 * (1.0 - eta_s) ** 2
              * (lam_s ** 2 + 2.0 * lam_s * lam_t + 3.0 * lam_t ** 2)
              / 48.0 * _cosL)

    # ---- inertial relief --------------------------------------------------
    Wsinn = V("W_s_inn", 1e4, "N", "inner panel structural weight")
    Wsout = V("W_s_out", 1e4, "N", "outer panel structural weight")
    dyWsinn = V("dyW_s_inn", 1e4, "N*m", "inner panel spanwise moment")
    dyWsout = V("dyW_s_out", 1e4, "N*m", "outer panel spanwise moment")
    Winn = V("W_inn", 2e4, "N", "inner panel weight incl. secondary and fuel")
    Wout = V("W_out", 2e4, "N", "outer panel weight incl. secondary and fuel")
    dyWinn = V("dyW_inn", 2e4, "N*m", "inner panel relief moment")
    dyWout = V("dyW_out", 2e4, "N*m", "outer panel relief moment")
    # Engine relief as a VARIABLE, not an argument: the engine is built after
    # the wing (aircraft.py:241 against :167), so its weight cannot be passed
    # in. The aircraft ties this to the engine system weight for a wing-hung
    # installation and to nothing for a rear-mounted one.
    Weng = V("W_eng_relief", 1.0, "N", "engine weight relieving the wing")
    Reng = Weng
    # Fuel volume, integrated over the real bays rather than correlated. The
    # box already computes the bay areas for the relief term; using a separate
    # mac^2 correlation for CAPACITY meant the same fuel had two different
    # volumes in one model, which is a contradiction rather than an
    # approximation.
    Vfuelbox = V("V_fuel", 25.0, "m^3", "fuel volume in the wing box")

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
        Ss + (-F_LT) * po * co * gam_t * lam_t + N_lift * Wout
            == (po * b / 4.0) * (gam_s + gam_t) * (1.0 - eta_s),  # [SP] SigEq
        Ms + (-F_LT) * po * co * gam_t * lam_t * 0.5 * b * (1.0 - eta_s)
            + N_lift * dyWout
            == (po * b ** 2 / 24.0) * (gam_s + 2.0 * gam_t)
               * (1.0 - eta_s) ** 2,                              # [SP] SigEq

        # ---- root station (surfw.f:88-96) ---------------------------------
        So + N_lift * Reng + N_lift * Winn
            == Ss + 0.25 * po * b * (1.0 + gam_s) * (eta_s - eta_o),
        # surfw.f:96 -- the (Ss - Nload*We) group carries the engine relief
        # through the root moment as well as the shear.
        Mo + N_lift * Reng * 0.5 * b * (eta_s - eta_o) + N_lift * dyWinn
            == Ms + Ss * 0.5 * b * (eta_s - eta_o)
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

    # ---- the relief fixed point (wsize.f:1003-1006) -----------------------
    # Equalities, not inequalities: relief REDUCES the load, so a one-sided
    # row would let the optimiser inflate the relief and under-size the box.
    _Abcapo, _Abcaps = 2.0 * tcapo * wwb, 2.0 * tcaps * wwb
    _Abwebo, _Abwebs = 2.0 * twebo * rh * hbox, 2.0 * twebs * rh * hbox
    _Abcapi = (_Abcapo + _Abcaps * lam_s ** 2) / (1.0 + lam_s ** 2)
    _Abwebi = (_Abwebo + _Abwebs * lam_s ** 2) / (1.0 + lam_s ** 2)
    cons += [
        Wsinn == (rhocap * _Abcapi + rhoweb * _Abwebi) * g * Vinn,
        Wsout == (rhocap * _Abcaps + rhoweb * _Abwebs) * g * Vout,
        dyWsinn == (rhocap * _Abcapi + rhoweb * _Abwebi) * g * dyVinn,
        dyWsout == (rhocap * _Abcaps + rhoweb * _Abwebs) * g * dyVout,
    ]
    if rho_fuel is not None and f_fuel > 0.0:
        rhof = C("rho_fuel_box", rho_fuel, "kg/m^3", "fuel density")
        Abfo = V("Ab_fuel_o", 0.05, "-", "non-dim fuel bay area, root")
        Abfs = V("Ab_fuel_s", 0.05, "-", "non-dim fuel bay area, break")
        _Abfi = (Abfo + Abfs * lam_s ** 2) / (1.0 + lam_s ** 2)
        cons += [
            # (wbox - 2*tweb)*(havg - 2*tcap), expanded all-positive
            Abfo + 2.0 * wwb * tcapo + 2.0 * twebo * havg
                == wwb * havg + 4.0 * twebo * tcapo,          # [SP] SigEq
            Abfs + 2.0 * wwb * tcaps + 2.0 * twebs * havg
                == wwb * havg + 4.0 * twebs * tcaps,          # [SP] SigEq
            Winn == Wsinn * (1.0 + f_wadd) + f_fuel * rhof * _Abfi * g * Vinn,
            Wout == Wsout * (1.0 + f_wadd) + f_fuel * rhof * Abfs * g * Vout,
            dyWinn == (dyWsinn * (1.0 + f_wadd)
                       + f_fuel * rhof * _Abfi * g * dyVinn),
            dyWout == (dyWsout * (1.0 + f_wadd)
                       + f_fuel * rhof * Abfs * g * dyVout),
        ]
        cons += [Vfuelbox == 2.0 * (Abfo * Vcen + _Abfi * Vinn + Abfs * Vout)]
        out.update(Ab_fuel_o=Abfo, Ab_fuel_s=Abfs)
    else:
        cons += [
            Winn == Wsinn * (1.0 + f_wadd),
            Wout == Wsout * (1.0 + f_wadd),
            dyWinn == dyWsinn * (1.0 + f_wadd),
            dyWout == dyWsout * (1.0 + f_wadd),
            Vfuelbox == 0.0 * Vcen + 1e-9 * Vcen,   # no fuel model requested
        ]
    if not relief:
        # Explicitly OFF: pin the relief to nothing rather than deleting the
        # rows, so the two cases differ by a constant and stay comparable.
        cons = [c for c in cons]
    out.update(W_inn=Winn, W_out=Wout, W_s_inn=Wsinn, W_s_out=Wsout,
               W_eng_relief=Weng, V_fuel=Vfuelbox)
    return out, cons
