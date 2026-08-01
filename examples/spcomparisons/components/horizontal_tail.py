"""Horizontal tail sizing, structure and drag.

Source model
------------
``horizontal_tail.py`` in https://github.com/convexengineering/SPaircraft.

A quirk worth knowing
---------------------
The wing links its planform taper to its box taper
(``wns['\\lambda'] == wb['taper']``) and so does the vertical tail
(``taper = surface['\\lambda_{vt}']``). The horizontal tail does **not**: in
``wingbox.py`` the ``horizontal_tail`` branch declares
``taper = Variable('taper', 0.3, '-')`` — a fixed constant — and nothing ever
ties it to ``\\lambda_{ht}``. So the box is sized for a 0.3-taper surface no
matter what planform taper the optimizer picks, and the ``L_{ht_{tri}}``
constraint that uses ``(1 - wb['taper'])`` is likewise pinned. Reproduced as
written, because the reference numbers depend on it.

The model does not include wing downwash on tail effectiveness; the source
says so explicitly.

What is sized elsewhere
-----------------------
``M_r``, ``L_{shear}``, ``b_{ht_{out}}`` and ``\\pi_{M-fac}`` belong to the
pi-tail bending case and close at aircraft level, as do the CG-relative
distances and the tail volume coefficient.
"""
from __future__ import annotations

from numpy import pi, tan

from .polars import (POLARS, TASOPT_ARE_XP, TASOPT_TAIL_CDF,
                     TASOPT_TAIL_CDP, TASOPT_TAIL_REREF)
from .wingbox import add_wingbox

# The horizontal tail box taper, fixed in wingbox.py and deliberately not
# linked to the planform taper. See the module docstring.
BOX_TAPER = 0.3


def add_horizontal_tail(f, N, state, *, sweep_deg=None, prefix="HT_",
                        pi_tail=True, material=None, tau_limits=True,
                        cosL=None, tanL=None,
                        drag_model="fit"):
    """Add the horizontal tail. Returns ``(vars, constraints)``."""
    ht = f.group("ht", prefix=prefix)
    V, C = ht.Variable, ht.Constant
    Vn = lambda n, g, u, d: ht.Variable(n, g, u, d, size=N)

    # ---- planform ---------------------------------------------------------
    ARht = V("AR_ht", 6.0, "-", "horizontal tail aspect ratio")
    Sh = V("S_ht", 30.0, "m^2", "horizontal tail area")
    bht = V("b_ht", 13.0, "m", "horizontal tail span")
    croot = V("c_root_ht", 3.5, "m", "horizontal tail root chord")
    ctip = V("c_tip_ht", 1.1, "m", "horizontal tail tip chord")
    chma = V("cbar_ht", 2.5, "m", "mean aerodynamic chord, HT")
    taper = V("lambda_ht", 0.3, "-", "horizontal tail taper ratio")
    tau = V("tau_ht", 0.12, "-", "horizontal tail thickness/chord ratio")
    p = V("p_ht", 1.6, "-", "substituted variable 1 + 2 taper")
    q = V("q_ht", 1.3, "-", "substituted variable 1 + taper")
    ymac = V("y_cbar_ht", 2.8, "m", "spanwise location of the mean aero chord")
    e = V("e_ht", 0.8, "-", "Oswald efficiency factor")
    fl = V("f_lambda_ht", 0.02, "-", "empirical efficiency function of taper")

    dxlead = V("dx_lead_ht", 16.0, "m", "CG to HT leading edge")
    dxtrail = V("dx_trail_ht", 20.0, "m", "CG to HT trailing edge")
    lht = V("l_ht", 17.0, "m", "horizontal tail moment arm")
    xcght = V("x_CG_ht", 34.0, "m", "horizontal tail CG location")
    Lmax = V("L_ht_max", 5e5, "N", "maximum load")
    Vh = V("V_ht", 0.8, "-", "horizontal tail volume coefficient")
    mrat = V("m_ratio", 0.5, "-", "wing to tail lift slope ratio")
    cattach = V("c_attach", 2.5, "m", "HT chord where it mounts to the VT")
    Wht = V("W_ht", 1e4, "N", "HT system weight")

    # ---- constants --------------------------------------------------------
    etaht = C("eta_ht", 1.0, "-", "tail efficiency")
    # Sweep. TASOPT does not give the horizontal its own sweep variable: in
    # fobj.f the optimizer's `iosweep` slot writes BOTH parg(igsweep) and
    # parg(igsweeph), so the tail is swept with the wing as one degree of
    # freedom. Passing ``cosL``/``tanL`` in reproduces that. It is also what
    # makes freeing tail sweep safe here -- tail sweep appears in no tail drag
    # polar and no tail weight, only in the moment arm, so on its own it would
    # buy arm for nothing; tied to the wing it is priced by the transonic drag
    # fit and by the wingbox's cos^2/cos^4 terms.
    if tanL is not None:
        tanLh = tanL
    else:
        tanLh = C("tan_Lambda_ht", tan((sweep_deg or 0.0) * pi / 180), "-",
                  "tangent of horizontal tail sweep")
    cosLh = cosL
    # 0.25 rad = 14.3 deg, an incidence a horizontal with elevator can actually
    # reach before it stalls. The source's 2.5 is meaningless as radians -- it
    # is 143 degrees -- and it never bound anything, so the tail's incidence
    # was effectively unlimited.
    amax = C("alpha_ht_max", 0.25, "-", "max angle of attack, HT [rad]")
    # 2.0 is not reachable by a tail. A horizontal with elevator gets to
    # ~1.0-1.2, and TASOPT sizes its forward-CG trim case at |CLh| = 0.65.
    # This sits in the DENOMINATOR of tail sizing, so a value twice too high
    # halves the tail.
    # 2.0, TASOPT's own value (runs/737/737s.tas:264, "CLhmax HT max +/-CL at
    # Vmn, for HT STRUCTURAL sizing"). I had lowered this to 1.2 on the
    # argument that a tail with an elevator only reaches ~1.0-1.2 -- which is
    # true of TRIM, and this quantity is not trim. It appears in exactly one
    # row, the gust/manoeuvre load case
    #     L_ht_max >= 0.5 rho V_ne^2 S_ht C_L_ht_max
    # which is a structural limit load, and a surface does reach 2.0 there.
    CLhmax = C("C_L_ht_max", 2.0, "-", "max HT lift coefficient, structural")
    # 1.0 -- NO empirical correction. Tried at 1.5 and 2.0 and pulled back
    # out; the measurements are kept because they bound the real fix.
    #
    # 1.5 put the stabiliser's areal weight at 60.0 lbf/m2 against a real
    # ~62, and 2.0 overshot to 70.9 -- so whatever replaces this should be
    # worth about half the box weight again. See the matching note on C_VT.
    # The horizontal is the worse of the two surfaces:
    # 0.511 of TASOPT and 0.669 of the real stabiliser, where the fin is 0.725
    # and 0.853. Two reasons for the gap being larger. The carry-through is a
    # genuine load path for a horizontal tail -- two cantilevers joined by a
    # box THROUGH the fuselage, with the full root moment crossing it -- while
    # the fin's root region is partly inside its own beam already, since the
    # fin is fed the doubled span, area and load of the mirror-image trick.
    # And a trimmable stabiliser carries a pivot, jackscrew and hinge fittings
    # that f_ht = 0.30 (TASOPT's fhadd, matched exactly) does not cover.
    Cht = C("C_ht", 1.0, "-", "HT weight margin and sensitivity factor")
    fht = C("f_ht", 0.3, "-", "rudder etc. fractional weight")

    out = dict(AR_ht=ARht, S_ht=Sh, b_ht=bht, c_root_ht=croot, c_tip_ht=ctip,
               cbar_ht=chma, lambda_ht=taper, tau_ht=tau, p_ht=p, q_ht=q,
               y_cbar_ht=ymac, e_ht=e, dx_lead_ht=dxlead, dx_trail_ht=dxtrail,
               l_ht=lht, x_CG_ht=xcght, L_ht_max=Lmax, V_ht=Vh,
               m_ratio=mrat, c_attach=cattach, W_ht=Wht, eta_ht=etaht,
               C_L_ht_max=CLhmax)

    cons = [
        # EQUALITY. The tail arm IS the distance from the CG to the tail's
        # aerodynamic centre -- it is geometry, not a quantity with slack.
        #
        # Written as >= it is bounded ABOVE only, so the arm may be shorter
        # than the aeroplane's actual geometry, and it pays to be: Iz_tail
        # carries W_ht*l_ht^2, so a short arm buys cheap yaw inertia. Stock
        # SPaircraft never exploited that because V_ht appeared in its tail
        # sizing inequality and pushed the arm out to this bound. The ported
        # cCM residual sizes the tail through S_ht and never mentions V_ht, so
        # that pressure vanished and the arm collapsed -- 9.81 m on a 737
        # against TASOPT's 14.88 m, which is most of the V_ht shortfall.
        #
        # Same failure mode as the mean aerodynamic chord: a one-sided bound
        # that was only ever correct because something else happened to pin it.
        dxlead + ymac * tanLh + 0.25 * chma == lht,          # [SP] SigEq
        dxlead + croot <= dxtrail,
        # EQUALITIES. These bound correctly today (both bind, at 1.6 and 1.3
        # for taper 0.30) but only because the optimiser happens to want them
        # tight. They are definitions of substituted variables, not design
        # freedoms, so they are written as such -- matching the wing.
        p == 1 + 2 * taper,
        q == 1 + taper,
        # TRANSPOSED, and live: this feeds the tail arm two rows above via
        #     dx_lead + y_mac*tan(Lambda_h) + 0.25*cbar == l_ht
        # so an inflated y_mac inflates l_ht, which inflates V_ht, which lets
        # S_ht come out smaller than the tail volume actually demands.
        # b_ht is TIP-TO-TIP here (AR_ht == b_ht^2/S_h below), so the mean
        # chord sits at (b/6) p/q, not (b/3) q/p -- 2.939 m against 3.880 m,
        # 1.320x high, worth about 0.37 m of false tail arm.
        ymac == (bht / 6) * p / q,
        (2. / 3) * (1 + taper + taper ** 2) * croot / q == chma,    # [SP] SigEq
        taper == ctip / croot,
        Sh == bht * (croot + ctip) / 2,                             # [SP] SigEq
        # Oswald efficiency (Nita & Scholz). The source marks this one
        # "slightly slack" with reltol=0.2.
        fl >= (0.0524 * taper ** 4 - 0.15 * taper ** 3 + 0.1659 * taper ** 2
               - 0.0706 * taper + 0.0119),
        e * (1 + fl * ARht) <= 1,
        ARht == bht ** 2 / Sh,
        # TODO in source: make less arbitrary.
        taper >= 0.2,
        taper <= 1,
        # c_attach is the chord where the horizontal meets the fin, and it is
        # used ONLY by the pi-tail branch in aircraft.py. On a conventional
        # tail nothing touches it, so it floated to 1.9e+10 m. It is a chord on
        # this surface and cannot exceed the root chord; that keeps a variable
        # the conventional configuration does not use from wandering ten orders
        # of magnitude away in a log-space solve.
        cattach <= croot,
    ]

    # ---- structure ---------------------------------------------------------
    box = ht.group("box", prefix=f"{ht.prefix}box_")
    wb, wbcons = add_wingbox(
        "horizontal_tail" if pi_tail else "horizontal_tail_conventional",
        AR=ARht, b=bht, S=Sh, p=p, q=q, tau=tau, Lmax=Lmax, group=box,
        material=material, tau_limits=tau_limits, cosL=cosLh)
    cons += wbcons
    out["box"] = wb

    # The triangular/rectangular load split exists only to evaluate bending at
    # a pi-tail's outboard pin joint. A cantilever tail has no pin joint.
    if pi_tail:
        cons += [
            wb["L_ht_rect"] >= Lmax / 2. * ctip * bht / Sh,
            # Note BOX_TAPER, not the planform taper -- see module docstring.
            wb["L_ht_tri"] >= Lmax / 4. * (1 - BOX_TAPER) * croot * bht / Sh,
        ]
    cons += [Wht >= Cht * (wb["W_struct"] + wb["W_struct"] * fht)]

    # ---- per-segment aerodynamics -----------------------------------------
    Dht = Vn("D_ht", 3e3, "N", "horizontal tail drag")
    Lh = Vn("L_ht", 2e4, "N", "horizontal tail downforce")
    Rec = Vn("Re_c_h", 1.5e7, "-", "cruise Reynolds number, HT")
    CLah = Vn("C_L_alpha_ht", 4.0, "-", "lift curve slope, HT")
    CLah0 = Vn("C_L_alpha_ht_0", 6.28, "-", "isolated lift curve slope, HT")
    CLh = Vn("C_L_ht", 0.2, "-", "lift coefficient, HT")
    CDh = Vn("C_D_ht", 0.008, "-", "HT drag coefficient")
    CD0h = Vn("C_D_0_ht", 0.006, "-", "HT parasitic drag coefficient")
    alphah = Vn("alpha_ht", 0.05, "-", "horizontal tail angle of attack")
    out.update(D_ht=Dht, L_ht=Lh, C_L_ht=CLh, C_L_alpha_ht=CLah,
               C_L_alpha_ht_0=CLah0, alpha_ht=alphah, C_D_ht=CDh)

    rho, Vinf, M, mu = state.rho, state.V, state.M, state.mu
    cons += [
        Lh == 0.5 * rho * Vinf ** 2 * Sh * CLh,
        CLh == CLah * alphah,
        alphah <= amax,
        # Thin airfoil theory, used as an approximation.
        CLah0 == 2 * 3.14,
        Dht == 0.5 * rho * Vinf ** 2 * Sh * CDh,
        CDh >= CD0h + CLh ** 2 / (pi * e * ARht),
        Rec == rho * Vinf * chma / mu,
    ]

    # ---- tail profile drag -------------------------------------------------
    # 'tasopt' is what TASOPT actually does (cdsum.f:234-250): two constants
    # from the run deck, scaled by Reynolds number and sweep, with NO Mach
    # dependence. 'fit' is SPaircraft's -- a wing-airfoil surrogate applied to
    # a tail, which is where tau**6.29 and M**10.26 come from. The fit is kept
    # as the default only because the reference numbers depend on it; TASOPT's
    # own treatment is the more defensible one, and it retires an
    # extrapolation that put the fin at t/c 17.55 once the thickness band was
    # lifted.
    if drag_model == "tasopt":
        cdft = C("cd_f_t", TASOPT_TAIL_CDF, "-", "tail friction cd (TASOPT deck)")
        cdpt = C("cd_p_t", TASOPT_TAIL_CDP, "-", "tail pressure cd (TASOPT deck)")
        Reref = C("Re_ref_t", TASOPT_TAIL_REREF, "-", "tail reference Reynolds")
        cons += [CD0h >= (cdft + cdpt) * (Rec / Reref) ** TASOPT_ARE_XP]
    elif drag_model == "fit":
        cons += [
        # Martin's TASOPT tail drag fit.
        CD0h ** 6.48983 >= (
            5.28751e-20 * Rec ** 0.900672 * tau ** 0.912222 * M ** 8.64547
            + 1.67605e-28 * Rec ** 0.350958 * tau ** 6.29187 * M ** 10.2559
            + 7.09757e-25 * Rec ** 1.39489 * tau ** 1.96239 * M ** 0.567066
            + 3.73076e-14 * Rec ** -2.57406 * tau ** 3.12793 * M ** 0.448159
            + 1.44343e-12 * Rec ** -3.91046 * tau ** 4.66279 * M ** 7.68852),
        ]
    elif drag_model == "mses":
        # T-series MSES surrogate, CL-independent (fit at CL=0.1, which is
        # where a tail actually lives). Replaces TASOPT's two Mach-independent
        # constants with a fit that has a real transonic rise -- CD roughly
        # 9x between M 0.70 and 0.85 on a 12% section -- so the PERPENDICULAR
        # Mach is what must be fed here, not the flight Mach. Getting that
        # wrong would be a large error rather than a small one.
        _tp = POLARS["mses_t_tail"]
        cons += [(CD0h / _tp.cd_ref) ** _tp.alpha
                 >= _tp.cd(Rec, tau, M, 1.0)]
    else:
        raise ValueError(f"unknown tail drag_model {drag_model!r}")

    return ht, cons
