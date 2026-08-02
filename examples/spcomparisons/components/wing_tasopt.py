"""The TASOPT wing: a cranked planform with two independent taper ratios.

STATUS: FUTURE WORK. NOT USED. `wing_model` defaults to "hoburg" and every
result in the study comes from wing.py. This module is kept because the
geometry and the ported structure are correct and the debugging is expensive
to redo, not because it works: it does not converge.

Where it stands, so the next attempt starts from the evidence rather than
repeating it:

  * Eight defects were found and fixed. Seven are listed at the points they
    occur; the eighth, `Ko = 1.0/Kc` in wingbox_tasopt.planform(), is the one
    that mattered -- a reciprocal of a posynomial in the load intensity row,
    upstream of every load in the box. Fixing it moved the failure from
    "sub-problem infeasible at iteration 32" to "600 iterations without
    converging", which is the only change of failure MODE any fix produced.

  * A warm start from a CONVERGED hoburg aeroplane -- 1200 values copied, the
    cranked planform derived from that solution, residuals 0.00e+00 on both
    signomial equalities -- still failed at iteration 32 (before the Ko fix).
    That rules out initialisation: the problem is the formulation.

  * At 200 iterations post-fix: stationarity 4.45, max_violation 1.6e-2. A
    converging solve here reaches ~1e-6 and ~1e-9. That is stuck, not slow.

  * Two defects remain candidates and were fixed but never confirmed to help:
    the tip-rolloff sign (fLt is negative, so the term belongs on the right of
    the po equality, worth ~0.05% of Kp0) and TWO different eta_o -- the wing
    carried it as a variable at ~0.122 while the box defaulted to the deck
    constant 0.1016, so area was integrated over one planform and volume over
    another.

The lesson, which is the reason to read this before continuing: planform() and
volumes() were written and VALIDATED with float arguments -- they reproduce
TASOPT's own 737 to 0.1% that way -- and then called with variables. `Ko` and
the eta_o mismatch are both that same trap, and a standalone test passing at
0.1% gave false confidence about behaviour inside the optimisation. Build the
next version with variables from the start and validate it INSIDE the solve.

The original text follows.


An alternative to ``wing.py`` (SPaircraft/Hoburg), selected with
``wing_model="tasopt"``. Both build the same group -- ``f.wing`` -- and expose
the same names, so the rest of the aircraft does not know which it has.

What actually differs
---------------------
``wing.py`` is a SINGLE-taper wing: one ratio ``lambda = c_tip/c_root``, area
``S = b(c_root+c_tip)/2``, and a mean chord ``(2/3)(1+l+l^2) c_root/q``. That
is the geometry Hoburg's closed-form box is derived over and it is internally
consistent, so it is left exactly as it was.

A real transport wing is CRANKED: the leading edge is one straight swept line
and the trailing edge kinks at a planform break, which gives two independent
taper ratios -- root-to-break and break-to-tip. TASOPT models it that way
(``lambdas``, ``lambdat`` in the run deck) and ``surfw.f`` sizes three
stations on it. Trying to drive that from one taper is not an approximation of
it, it is a different wing.

Both tapers are DESIGN VARIABLES here. That is not just for freedom, it is
also what keeps the model tractable: deriving the break taper from the tip
taper as ``lam_s = 1 - (1-lam_t)*eta_s`` -- which is what a straight-tapered
wing would give -- puts a difference inside ``Kc`` and makes ``Ko = 1/Kc`` the
reciprocal of a signomial. That was tried and neither case converged (600
iterations, and a sub-problem failure at 74). With ``lam_s`` independent every
planform coefficient stays posynomial:

    Kc   == eta_o + (1+lam_s)(eta_s-eta_o)/2 + (lam_s+lam_t)(1-eta_s)/2
    Kmac == eta_o + (eta_s-eta_o)(1+lam_s+lam_s^2)/3
                  + (1-eta_s)(lam_s^2+lam_s*lam_t+lam_t^2)/3
    Ko*Kc == 1

Independent tapers are EASIER for the solver, not harder.

Geometry
--------
Constant leading-edge sweep, trailing edge following the taper. With the LE a
straight line at ``tan_Lambda_LE`` and the chord piecewise linear:

    dc/dy|inner = c_o(lam_s - 1) / (b/2 (eta_s - eta_o))
    dc/dy|outer = c_o(lam_t - lam_s) / (b/2 (1 - eta_s))
    tan_Lambda_TE = tan_Lambda_LE + dc/dy      (different on each panel)

Both derivatives are negative for a tapered wing, so the TE sweeps aft of the
LE by an amount that changes at the break -- which is the crank.

Mean aerodynamic chord
----------------------
``mac = (1/S) int c^2 dy`` over the cranked planform, which works out to

    mac * S == b * c_o^2 * Kmac

and is exactly twice the volume bracket the box already integrates. The
single-taper form in wing.py is wrong for this planform, and it is not a small
error: it also normalises the static margin and the tail volume coefficient,
so getting it wrong mis-sizes the horizontal tail without touching the tail.

Trailing-edge sweep is NOT imposed, deliberately. It follows from the chord
slopes already here -- ``tan(L_TE) = tan(L_LE) - D`` per panel -- but as a GP
row it would force ``tan(L_TE) > 0``, and on a 737-like planform the inner
panel sits within a degree of unswept (D = 0.539 against tan(L_LE) = 0.54), so
a real wing can and does sweep its inner trailing edge FORWARD. Imposing
positivity there would be a constraint nobody wrote, and this model has had
enough of those. ``dc_dy_inn`` and ``dc_dy_out`` are exposed so the spar and
gear geometry can form it where the sign is understood.

There are no ``p`` and ``q`` here. They are the single-taper substitutions
``1+2*lambda`` and ``1+lambda``; on a cranked wing they mean nothing.
"""
from __future__ import annotations

from math import cos, pi, tan

from .polars import POLARS, YORK_C
from .wingbox_tasopt import ETA_S, RCLS, RCLT, add_wingbox_tasopt

#: Planform break station, as a fraction of semi-span. A deck input in TASOPT
#: (0.285 for the 737) rather than an optimisation variable, and kept that way:
#: it is where the trailing-edge kink and the main gear go, which is a layout
#: decision rather than something to be swept.
ETA_BREAK = ETA_S


def add_wing_tasopt(f, N, state, *, sweep_deg=None, prefix="Wing_",
                    rho_fuel=817.0, material=None, sweep_pricing=False,
                    polar=YORK_C, W_engine=None, eta_break=ETA_BREAK,
                    tau_max=0.14, lam_s_pin=None, lam_t_pin=None,
                    f_L_total=1.0, f_slat=0.1, e_model="nita"):
    """Add the cranked-planform wing. Returns ``(group, constraints)``.

    Same signature and same exposed names as ``wing.add_wing``, so
    ``aircraft.py`` can select either without knowing the difference.
    """
    wing = f.group("wing", prefix=prefix)
    V, C = wing.Variable, wing.Constant
    Vn = lambda n, g, u, d: f.Variable(name=f"{prefix}{n}", guess=g, units=u,
                                       description=d, size=N)
    out = {}

    # ---- planform ---------------------------------------------------------
    AR = V("AR", 11.0, "-", "wing aspect ratio")
    S = V("S", 125.0, "m^2", "wing area")
    b = V("b", 37.0, "m", "wing span")
    co = V("c_root", 5.9, "m", "centreline (root) chord")
    cs = V("c_break", 4.1, "m", "chord at the planform break")
    ct = V("c_tip", 1.5, "m", "tip chord")
    mac = V("mac", 3.8, "m", "mean aerodynamic chord")
    lam_s = V("lambda_s", 0.70, "-", "break/root taper ratio")
    lam_t = V("lambda", 0.25, "-", "tip/root taper ratio")
    Kc = V("K_c", 0.60, "-", "chord distribution integral")
    Ko = V("K_o", 1.67, "-", "1/K_c")
    Kmac = V("K_mac", 0.42, "-", "mean-chord integral")
    tanLE = V("tan_Lambda_LE", 0.60, "-", "tangent of LEADING EDGE sweep")
    tanQCi = V("tan_Lambda_qc_inn", 0.49, "-", "tan c/4 sweep, inner panel")
    Dinn = V("dc_dy_inn", 0.54, "-", "chord taper rate, inner panel (-dc/dy)")
    Dout = V("dc_dy_out", 0.21, "-", "chord taper rate, outer panel (-dc/dy)")
    tau = V("tau", 0.12, "-", "wing thickness/chord ratio")
    e = V("e", 0.85, "-", "Oswald efficiency factor")
    fl = V("f_lambda_w", 0.02, "-", "empirical efficiency function of taper")
    xw = V("x_w", 18.0, "m", "position of wing aerodynamic centre")
    dxACwing = V("dx_AC_wing", 1.0, "m", "wing aerodynamic centre shift")
    Lmax = V("L_max", 3e6, "N", "maximum load")
    Vfuel = V("V_fuel_max", 20.0, "m^3", "available fuel volume")
    WfuelWing = V("W_fuel_wing", 1e5, "N", "fuel weight carried in the wing")
    Wwing = V("W_wing", 1e5, "N", "wing system weight")

    g = C("g", 9.81, "m/s^2", "gravitational acceleration")
    eta = C("eta", 0.97, "-", "lift efficiency (sectional vs actual)")
    etas = C("eta_s", eta_break, "-", "planform break station, fraction b/2")
    taumax = C("tau_max", tau_max, "-", "max thickness/chord")
    rhofuel = C("rho_fuel", rho_fuel, "kg/m^3", "fuel density")
    Cwing = C("C_wing", 1.0, "-", "wing structural weight margin")

    # ---- sweep ------------------------------------------------------------
    if sweep_deg is None:
        cosL = V("cos_Lambda", 0.94, "-", "cosine of quarter-chord sweep")
        tanL = V("tan_Lambda", 0.36, "-", "tangent of quarter-chord sweep")
        sweep_cons = [tanL ** 2 * cosL ** 2 + cosL ** 2 == 1.0,  # [SP] SigEq
                      cosL <= 0.9999, cosL >= 0.766]
    else:
        cosL = C("cos_Lambda", cos(sweep_deg * pi / 180), "-",
                 "cosine of quarter-chord sweep")
        tanL = C("tan_Lambda", tan(sweep_deg * pi / 180), "-",
                 "tangent of quarter-chord sweep")
        sweep_cons = []

    out.update(AR=AR, S=S, b=b, c_root=co, c_break=cs, c_tip=ct, mac=mac,
               lambda_=lam_t, lambda_s=lam_s, tau=tau, e=e, x_w=xw,
               cos_Lambda=cosL, tan_Lambda=tanL, dx_AC_wing=dxACwing,
               L_max=Lmax, V_fuel_max=Vfuel, W_fuel_wing=WfuelWing,
               W_wing=Wwing, eta_s=etas, tan_Lambda_LE=tanLE,
               tan_Lambda_qc_inn=tanQCi, dc_dy_inn=Dinn, dc_dy_out=Dout)

    # ---- per-segment aero variables ---------------------------------------
    alpha = Vn("alpha_w", 0.08, "-", "wing angle of attack")
    CLaw = Vn("C_L_alpha_w", 5.0, "-", "lift curve slope, wing")
    Re = Vn("Re_w", 2e7, "-", "Reynolds number, wing")
    CDp = Vn("C_D_p_w", 0.005, "-", "wing parasitic drag coefficient")
    CDi = Vn("C_D_i_w", 0.006, "-", "wing induced drag coefficient")
    CDw = Vn("C_d_w", 0.011, "-", "wing drag coefficient")
    CLw = Vn("C_L", 0.5, "-", "wing lift coefficient")
    Dwing = Vn("D_wing", 2e4, "N", "wing drag")
    Lw = Vn("L_w", 6e5, "N", "wing lift")
    dLo = Vn("dL_o", 1e4, "N", "centre wing lift loss")
    # 0.10, not wing.py's 0.9. eta_o is the centrebody fraction w_fuse/(b/2),
    # about 0.10 on a 737. The 0.9 guess is harmless there because eta_o only
    # enters dLo, but here it is structural -- it sits inside Kc and Kmac --
    # and starting at 0.9 makes (eta_s - eta_o) NEGATIVE, i.e. a planform
    # whose crank is inboard of the fuselage side. Both cases failed in the
    # SIA sub-problem (iterations 38 and 23) for that reason alone.
    etao = Vn("eta_o", 0.10, "-", "centre wing span coefficient")
    po = Vn("p_o", 2e4, "N/m", "centre section theoretical wing loading")
    dLt = Vn("dL_t", 1e3, "N", "wing tip lift loss")
    cmw = Vn("c_m_w", 0.1, "-", "wing pitching moment coefficient")
    fLo = C("f_L_o", 0.5, "-", "centre wing lift reduction coefficient")
    fLt = C("f_L_t", 0.05, "-", "wing tip lift reduction coefficient")
    amax = C("alpha_max_w", 0.1, "-", "max angle of attack")
    TipReduct = C("TipReduct", 1.0, "-", "tip device induced drag reduction")
    out.update(alpha_w=alpha, C_L=CLw, C_d_w=CDw, D_wing=Dwing, L_w=Lw,
               C_L_alpha_w=CLaw, c_m_w=cmw, eta_o=etao, p_o=po)

    rho, Vinf, M, mu = state.rho, state.V, state.M, state.mu
    eo = etao[0] if N > 1 else etao

    cons = sweep_cons + [
        # AR == b^2/S is NOT written here: add_wingbox_tasopt writes exactly
        # that row on exactly these objects (wingbox_tasopt.py, first row of
        # its cons), and a duplicated equality is a LICQ failure -- the pair's
        # gradients are linearly dependent, the multiplier becomes a ray, and
        # the solver runs it to the tau ceiling where complementarity can
        # never close. That is the "stationarity 4.45, stuck not slow" failure
        # recorded in this module's docstring: the same defect was found and
        # measured on the fin (three-row combination) and the horizontal tail
        # (exact pair) via SVD of the equality Jacobian, and removing those
        # plus this took the free-sweep D8 from non-convergent to a clean KKT
        # point. One owner per identity: the box.
        # ---- cranked planform, all posynomial ----------------------------
        cs == lam_s * co,
        ct == lam_t * co,
        Kc == eo + (1 + lam_s) * (etas - eo) / 2
              + (lam_s + lam_t) * (1 - etas) / 2,             # [SP] SigEq
        S == co * b * Kc,                                      # [SP] SigEq
        Ko * Kc == 1,
        Kmac == eo + (etas - eo) * (1 + lam_s + lam_s ** 2) / 3
                + (1 - etas) * (lam_s ** 2 + lam_s * lam_t
                                + lam_t ** 2) / 3,             # [SP] SigEq
        # mac = (1/S) int c^2 dy over the cranked planform. An EQUALITY for
        # the same reason wing.py's is: nothing stops an inflated mac
        # otherwise, and Re = rho V mac/mu makes a long chord buy cheap drag,
        # while mac also normalises the static margin and V_ht.
        mac * S == b * co ** 2 * Kmac,                         # [SP] SigEq
        # sensible planform bounds: monotonically decreasing chord, and no
        # degenerate triangles at either panel.
        # The break must lie outboard of the fuselage side. Implicit in every
        # formula here -- (eta_s - eta_o) appears in Kc, Kmac and both box
        # volume integrals -- and imposing it stops the solver wandering into
        # a planform that has no geometric meaning.
        eo <= etas,
        lam_t <= lam_s, lam_s <= 1.0,
        lam_t >= 0.10, lam_s >= 0.25,
        # TAPER PINS. Left free, lam_t runs to its 0.10 floor -- a 0.54 m tip
        # chord on a 737 -- because nothing here prices tip stall or aileron
        # authority. TASOPT's own optimizer has lambdat ACTIVE in the 737 deck
        # and still lands on exactly the deck's 0.25, so whatever holds it
        # there is not in our port either; until it is, the deck values are
        # the defensible basis, same as BPR and FPR on the engine.
        *([lam_s == lam_s_pin] if lam_s_pin is not None else []),
        *([lam_t == lam_t_pin] if lam_t_pin is not None else []),
        # Oswald efficiency (Nita & Scholz). Taken on the TIP taper, which is
        # the ratio the correlation was fitted against. Without this pair `e`
        # is a free variable and a higher one is pure profit -- induced drag
        # goes as 1/e -- which is the same defect that let q, taper and x_n
        # drift elsewhere in this model.
        # e_model="nita": the Nita-Scholz taper correlation, as wing.py.
        # e_model="trefftz": e is left FREE HERE and closed by the
        # aircraft-level Trefftz-plane surrogate rows (they need the tail
        # and fuselage, which do not exist yet when the wing builds). The
        # flag is a contract: selecting "trefftz" and not adding those rows
        # leaves e dangling.
        *([fl >= (0.0524 * lam_t ** 4 - 0.15 * lam_t ** 3
                  + 0.1659 * lam_t ** 2 - 0.0706 * lam_t + 0.0119),
           e * (1 + fl * AR) <= 1] if e_model == "nita" else []),
        # ---- chord slopes, and the leading edge --------------------------
        # D is defined as -dc/dy, positive for a chord that decreases
        # outboard, so every coefficient below stays positive.
        Dinn * (b / 2) * (etas - eo) + cs == co,               # [SP] SigEq
        Dout * (b / 2) * (1 - etas) + ct == cs,                # [SP] SigEq
        # ONE straight leading edge; the QUARTER-CHORD sweep is what varies
        # between panels. x_qc = x_LE + c/4 gives tan(L_LE) = tan(L_qc) +
        # 0.25*D, so a common LE forces different c/4 sweeps wherever the
        # chord slopes differ -- which is exactly what the crank is.
        #
        # tan_Lambda is the OUTER panel's quarter chord: that panel is 72% of
        # the semi-span and carries most of the area, so it is the right
        # single reference for the drag polar's perpendicular Mach and for the
        # box. On a 737 planform holding c/4 constant instead would give an LE
        # of 31.9 deg inboard against 28.4 outboard -- a kinked leading edge,
        # which is not the aeroplane. TASOPT carries one `sweep` and quietly
        # accepts that; this model does not.
        tanLE == tanL + 0.25 * Dout,                           # [SP] SigEq
        tanQCi + 0.25 * Dinn == tanLE,                         # [SP] SigEq
        tau <= taumax,
        WfuelWing <= rhofuel * Vfuel * g,
        dxACwing <= 1. / 24. * (co + 5. * ct) / S * b ** 2 * tanL,
    ]

    # ---- structure: the station-based box ---------------------------------
    box = wing.group("box", prefix=f"{wing.prefix}box_")
    wb, wbcons = add_wingbox_tasopt(
        "wing", AR=AR, b=b, S=S, tau=tau, Lmax=Lmax, group=box,
        cosL=cosL if sweep_pricing else None, material=material,
        eta_o=eo, eta_s=eta_break, lam_s=lam_s, lam_t=lam_t, Kc=Kc, Ko=Ko,
        W_engine=W_engine, rho_fuel=rho_fuel)
    cons += wbcons
    out["box"] = wb
    # Capacity comes from the box's own bay integration, not a mac^2
    # correlation. Same fuel, one volume.
    cons += [Vfuel == wb["V_fuel"]]

    # Secondary structure fractions. The 737 deck's set sums to 0.640; the
    # D8 deck's to 0.540 -- because the D8 HAS NO SLATS (d82.tas fslat =
    # 0.000, and its printed Wslat is 0.0). Hardwiring the 737's 0.10 slat
    # fraction onto a slatless aeroplane was ~450 lbf of phantom hardware,
    # masking part of the spar-cap undercount. f_slat comes from the caller.
    fnames = [("f_flap", 0.2), ("f_slat", f_slat), ("f_aileron", 0.04),
              ("f_lete", 0.1), ("f_ribs", 0.15), ("f_spoiler", 0.02),
              ("f_watt", 0.03)]
    fracs = [C(n, v, "-", f"{n} fractional weight") for n, v in fnames]
    cons += [Wwing >= Cwing * wb["W_struct"] + wb["W_struct"] * sum(fracs)]

    # ---- aerodynamics -----------------------------------------------------
    cons += [
        0.5 * rho * Vinf ** 2 * S * CLw >= Lw + dLo + 2. * dLt,
        dLo == etao * fLo * b / 2 * po,
        # TASOPT's tip rolloff uses the TIP taper and its cl ratio:
        # dLt = fLt*po*co*gammat*lambdat with gammat = lambdat*rclt.
        dLt == fLt * po * co * (lam_t * RCLT) * lam_t,
        (AR / eta) ** 2 * (1 + tanL ** 2 - M ** 2) + 8 * pi * AR / CLaw
            == (2 * pi * AR / CLaw) ** 2,                       # [SP] SigEq
        CLw == CLaw * alpha,
        alpha <= amax,
        Dwing == 0.5 * rho * Vinf ** 2 * S * CDw,
        CDw >= CDp + CDi,
        # Induced drag on TOTAL lift, not wing lift. TASOPT's Trefftz-plane
        # CDi charges the aircraft CL -- its printed cruise numbers close
        # exactly as CL_tot^2/(pi*AR*e) -- because the fuselage carryover
        # lift still trails vorticity. Charging only C_L_wing handed the
        # D8's Ltow = 1.195 carryover 19.5% of the lift induced-drag-free,
        # a 1.43x discount on CDi. f_L_total is the caller's Ltow; the
        # conventional tube's 1.02 makes this a 4% correction there.
        CDi >= TipReduct * (f_L_total * CLw) ** 2 / (pi * e * AR),
        Re == rho * Vinf * mac / mu,
    ]

    _polar = POLARS[polar] if isinstance(polar, str) else polar
    _cl = CLw / cosL ** 2 if _polar.perp_cl else CLw
    cons += [(CDp / _polar.cd_ref) ** _polar.alpha
             >= _polar.cd(Re, tau, cosL * M, _cl)
             * _polar.re_factor(Re) ** _polar.alpha]
    if _polar.m_perp_max is not None:
        cons += [cosL * M <= _polar.m_perp_max]

    return wing, cons
