"""Wing geometry, structure and aerodynamics.

Source model
------------
``wing.py`` in https://github.com/convexengineering/SPaircraft (Philippe
Kirschen's thesis wing model), plus ``wingbox.py`` for the structure.

Signomial content
-----------------
Three constraints here are not GP-compatible, and each is signomial for a
different reason:

* ``S == b*(c_root + c_tip)/2`` — a ``SignomialEquality``: written as an
  equality because area must be exact, and posynomial on both sides.
* the DATCOM swept-wing lift-curve-slope relation, also a
  ``SignomialEquality``, whose ``1 + tan(Lambda)^2 - M^2`` term subtracts the
  Mach number. This is the constraint that makes the *aerodynamics* signomial
  rather than merely the geometry.
* ``0.5 rho V^2 S C_L >= L_w + dLo + 2 dLt`` — the centre-section and tip lift
  losses appear as a sum on the greater side.

The source has a fourth, ``A_tri >= 0.5*(1-taper)*c_root*b``, a subtraction
inside the bound. It is deleted here: nothing read ``A_tri``, and nothing
bounded it above, so the constraint could always be met by raising it and
restricted nothing. See the note at the declaration below.

Drag
----
The parasitic drag polar is Martin York's fit to the TASOPT C-series
transonic airfoils, in (Re, tau, cos(Lambda)*M, C_L). The source also carries
Philippe's earlier fit, commented out; the two are not interchangeable, and
the older one has exponents up to 157 on ``cos(Lambda)*M``.
"""
from __future__ import annotations

from numpy import cos, pi, tan

from .polars import POLARS, YORK_C
from .wingbox import add_wingbox

#: Empirical mass correction on the Hoburg wing box, applied to the cap and
#: the web and therefore to W_struct and every secondary fraction taken off it.
#: 0.95 means the wing comes out at 95% of what the closed-form box predicts.
#:
#: WHY IT IS HERE. With sweep priced the box over-predicts against TASOPT's
#: station-based calculation: measured on the 737 the wing lands at 1.134 with
#: caps 1.104 and web 1.766. The documented single-taper (uncranked) penalty is
#: W_cap 1.241 / W_web 1.423 / W_wing 1.249, so most of that overshoot has a
#: named cause and a real fix -- the cranked box in wingbox_tasopt.py.
#:
#: This factor is NOT that fix. It is a flat empirical scale with no physics in
#: it, kept as a single named constant so it is trivial to find and delete when
#: the cranked box lands. WING ONLY: the tails come out LIGHT (H_tail 0.839 of
#: the real stabiliser), so scaling them down would widen a gap, not close one.
#: The box divides by this internally (`_wc * W >= RHS`), hence the reciprocal.
WING_MASS_FACTOR = 0.95


#: NO WING WEIGHT CORRECTION. Two lived here and both are gone.
#:
#: F_CRANKED = 1.2 credited the box for a cranked planform carrying root
#: bending more efficiently than the single taper line the closed form
#: integrates. F_TASOPT_WING = 0.8 was added on top, calibrated because
#: the wing read 31,558 lbf against TASOPT's 23,717 -- a ratio of 1.331
#: that looked like a wing-model error.
#:
#: It was not. The aircraft carrying that wing had DOUBLE the engine
#: weight it should have (the weight fit returns a set total; see
#: turbofan/model.py), so it was some 7,800 lbf heavy plus snowball and
#: the wing was correctly sized for that heavier aeroplane. The 1.331
#: was measuring the engine error.
#:
#: With the engine corrected both credits are removed and the box stands
#: on its own arithmetic. Calibrating a constant against a symptom hides
#: the cause and then has to be undone.


def add_wing(f, N, state, *, sweep_deg=None, prefix="Wing_",
             rho_fuel=817.0, material=None, sweep_pricing=False, polar=YORK_C,
             W_engine=None):
    """Add the wing (geometry + structure + per-segment aero) to ``f``.

    ``state`` supplies the per-segment freestream: rho, V, M, mu.
    ``sweep_deg`` is the quarter-chord sweep; the model uses its sine and
    tangent as constants rather than carrying the angle itself, which is what
    keeps the geometry monomial.
    Returns ``(vars, constraints)``.
    """
    wing = f.group("wing", prefix=prefix)
    V, C = wing.Variable, wing.Constant
    Vn = lambda n, g, u, d: wing.Variable(n, g, u, d, size=N)

    # ---- planform ---------------------------------------------------------
    AR = V("AR", 11.0, "-", "wing aspect ratio")
    S = V("S", 125.0, "m^2", "wing area")
    b = V("b", 37.0, "m", "wing span")
    croot = V("c_root", 5.5, "m", "wing root chord")
    ctip = V("c_tip", 1.5, "m", "wing tip chord")
    mac = V("mac", 3.8, "m", "mean aerodynamic chord")
    ymac = V("y_mac", 7.5, "m", "spanwise location of the mean aero chord")
    taper = V("lambda", 0.25, "-", "wing taper ratio")
    p = V("p", 1.5, "-", "substituted variable 1 + 2 taper")
    q = V("q", 1.25, "-", "substituted variable 1 + taper")
    tau = V("tau", 0.12, "-", "wing thickness/chord ratio")
    e = V("e", 0.85, "-", "Oswald efficiency factor")
    fl = V("f_lambda_w", 0.02, "-", "empirical efficiency function of taper")
    # A_tri, the triangular half of the source model's area decomposition, is
    # deleted here. Nothing consumed it -- it was defined by one constraint and
    # read by none -- and because nothing bounded it above, that constraint
    # (`A_tri >= 0.5*(1-taper)*c_root*b`) could always be satisfied by raising
    # A_tri, so it restricted taper, c_root and b not at all. Removing the pair
    # is therefore exact.
    #
    # It was not harmless. It was one of the wing's four signomial constraints,
    # it was degenerate at every solution, and it was the only variable in the
    # model with no upper bound -- which is why the source carries
    # `Atri <= 1e10*units('m**2')` on the next line purely to quiet gpkit's
    # bounded check, and why gpkit reports it sitting at 1e+30. See
    # docs/PRESOLVE.md.
    Arect = V("A_rect", 55.0, "m^2", "rectangular wing area")
    xw = V("x_w", 18.0, "m", "position of wing aerodynamic centre")
    dxACwing = V("dx_AC_wing", 1.0, "m", "wing aerodynamic centre shift")
    Lmax = V("L_max", 3e6, "N", "maximum load")
    Vfuel = V("V_fuel_max", 20.0, "m^3", "available fuel volume")
    WfuelWing = V("W_fuel_wing", 1e5, "N", "fuel weight carried in the wing")
    Wwing = V("W_wing", 1e5, "N", "wing system weight")

    # ---- planform constants ----------------------------------------------
    g = C("g", 9.81, "m/s^2", "gravitational acceleration")
    eta = C("eta", 0.97, "-", "lift efficiency (sectional vs actual)")
    # 14%, not TASOPT's 0.14733, because the aero model is York's fit to the
    # C-series table -- and that table's thickness grid STOPS at 0.145
    # (air/C.air: 0.09 .. 0.145). A 0.14733 cap lets the wing sit outside the
    # data the polar was built from, in the one direction where thickness is
    # cheapest and the drag penalty is least trustworthy. Restore TASOPT's
    # value alongside a polar refitted over a wider thickness range.
    tau_max = C("tau_max_w", 0.14, "-", "max allowed wing thickness")
    # Follows the fuel. 817 is kerosene; liquid hydrogen is 70, and a
    # hydrogen wing carrying kerosene density would size its tank volume
    # nearly 12x too small. Harmless today only because the hydrogen
    # architectures fly a dry wing -- which is exactly the kind of
    # "harmless" that stops being harmless when someone tries a
    # partially wet hydrogen wing.
    rhofuel = C("rho_fuel", rho_fuel, "kg/m^3", "density of fuel")
    Cwing = C("C_wing", 1.0, "-", "wing weight margin and sensitivity factor")

    # Quarter-chord sweep. With ``sweep_deg=None`` it is a DESIGN VARIABLE,
    # carried as cos(Lambda) rather than Lambda so that every appearance stays
    # monomial -- the trig-of-a-variable obstruction never arises. tan(Lambda)
    # rides alongside, tied by tan^2 cos^2 + cos^2 = 1, which is a signomial
    # equality and therefore fine in an SP.
    #
    # This matters more than it looks. Cruise Mach is free here, and the
    # transonic drag fit prices compressibility as (cos(Lambda)*M)^19.25. A
    # wing pinned at the D8.2's deliberately low 13.237 degrees is a wing
    # designed NOT to fly fast, so leaving it fixed while freeing Mach asks
    # the optimiser to go fast on the wrong wing -- and it will decline,
    # which reads as "the fuel-optimal Mach is 0.60" when it is really "0.60
    # is all this sweep can do".
    if sweep_deg is None:
        cosL = V("cos_Lambda", 0.94, "-", "cosine of quarter-chord sweep")
        tanL = V("tan_Lambda", 0.36, "-", "tangent of quarter-chord sweep")
    else:
        cosL = C("cos_Lambda", cos(sweep_deg * pi / 180), "-",
                 "cosine of quarter-chord sweep")
        tanL = C("tan_Lambda", tan(sweep_deg * pi / 180), "-",
                 "tangent of quarter-chord sweep")

    out = dict(cos_Lambda=cosL, tan_Lambda=tanL,
               AR=AR, S=S, b=b, c_root=croot, c_tip=ctip, mac=mac, y_mac=ymac,
               taper=taper, p=p, q=q, tau=tau, e=e, x_w=xw,
               dx_AC_wing=dxACwing, L_max=Lmax, V_fuel_max=Vfuel,
               W_fuel_wing=WfuelWing, W_wing=Wwing, A_rect=Arect)

    cons = [
        Arect == ctip * b,
        # p and q are DEFINED below as equalities. The one-sided pair that
        # used to sit here (p >= 1 + 2*taper, 2*q >= 1 + p) was left in place
        # when those were added, and redundant active rows are not free: the
        # equality implies the inequality, so both bind at the same point with
        # parallel gradients, and the multipliers are then indeterminate. That
        # is what the wing's 1e4 duals were -- an order of magnitude larger
        # than anything else in the model and read at the time as the price of
        # the taper pin -- and it is why the wing-position dual scan came back
        # with 1e11 multipliers and no usable information.
        # Spanwise station of the mean aerodynamic chord. Two errors, which
        # partly cancelled: p/q was TRANSPOSED, and the factor was b/3 where
        # a full-span b needs b/6. The standard result is
        #     y_mac = (b/6)(1 + 2*lambda)/(1 + lambda) = (b/6) p/q
        # since y_mac/(b/2) = (1/3) p/q and b here is TIP-TO-TIP (AR = b^2/S).
        # The old form gave 9.758 m against 7.026 m, 1.389x high.
        #
        # Inert in the wing today -- nothing reads y_mac, which is its own
        # smell -- but the identical expression is live in both tails, where
        # it sets the moment arms. Fixed in all three.
        ymac == (b / 6) * p / q,
        # EQUALITY, not a one-sided bound.
        #
        # SPaircraft writes this as mac >= (2/3)(1+lam+lam^2) c_root/q, which
        # only stops the chord being too SHORT. Nothing stops it being too
        # long, and it pays to be: Re == rho V mac/mu, and York's profile drag
        # fit carries Re^-0.55, so inflating mac buys a fictitious Reynolds
        # number and cheaper drag. On the E175 it ran to 12.75 m against a
        # true 2.74 m -- a chord four times the wing's actual mean chord.
        #
        # It also silently wrecks everything mac normalises: the static margin
        # (xAC - xCG)/mac, and the tail volume coefficient Sh*l_ht/(S*mac),
        # which read 0.148 against TASOPT's 0.575 for that reason alone and
        # not because the tail was mis-sized.
        #
        # The mean aerodynamic chord is determined by the planform. It is an
        # equality.
        # p and q are DEFINED here. They are described at their declaration as
        # "1 + 2 taper" and "1 + taper" and were never constrained to be so --
        # inherited that way from SPaircraft, where the closed-form wing box
        # consumes both (nu**3.94 >= 0.86 p**-2.38 + ..., 12 >= AR Lmax q**2/...)
        # and pins them implicitly. That is not a definition, it is a
        # coincidence of which model happens to read them, and it broke the
        # moment the TASOPT box -- which uses neither -- was selected: q
        # inflated from 1.15 to 1.56, and since mac = (2/3)(1+l+l^2) c_root/q
        # that shrank the mean chord 20%, which took the horizontal tail from
        # 31.8 to 20.1 m2 because V_ht is referenced to mac. Free, because
        # nothing else read q.
        p == 1 + 2 * taper,
        q == 1 + taper,
        (2. / 3) * (1 + taper + taper ** 2) * croot / q == mac,   # [SP] SigEq
        taper == ctip / croot,
        S == b * (croot + ctip) / 2,                                # [SP] SigEq
        # Oswald efficiency: Nita & Scholz, "Estimating the Oswald factor from
        # basic aircraft geometrical parameters".
        fl >= (0.0524 * taper ** 4 - 0.15 * taper ** 3 + 0.1659 * taper ** 2
               - 0.0706 * taper + 0.0119),
        e * (1 + fl * AR) <= 1,
        # Taper and thickness are PINNED, not bounded. Left free they sat on
        # their artificial floors -- taper on 0.15 and tau on the 0.14 fit
        # cap -- against TASOPT's 0.250/0.127 and a real 737's 0.24/0.125.
        # A variable resting on a bound nobody meant as physics is not a
        # design freedom, and both of these were set by the bound rather than
        # by any trade. Pinning them to the real aircraft's values makes the
        # comparison honest and removes two quantities the optimiser was
        # exploiting.
        taper == 0.25,
        tau == 0.13,
        # Fuel volume, GP approximation of the TASOPT signomial constraint.
        Vfuel <= 0.3026 * mac ** 2 * b * tau,
        WfuelWing <= rhofuel * Vfuel * g,
    ]

    if sweep_deg is None:
        cons += [
            # tan^2 L cos^2 L + cos^2 L == 1, the Pythagorean identity written
            # so both sides are posynomial.
            #
            # It MUST be the equality. The GP-legal half (<= 1) was tried when
            # this row's linearization was implicated in the free-sweep D8's
            # subproblem failures, with the argument that the pressure on
            # cos_Lambda is upward (the wingbox charges 1/cos^3, the FAR CLmax
            # correction pays (cos/cos_ref)^2) so the row would bind. On the D8
            # it does -- slack 3e-10. On the 737 it does NOT: the solution held
            # cos at 24.1 deg of sweep while claiming tan of 18.3 deg, slack
            # 7.6e-2, taking the low-tan trim benefits (dx_AC moved 65%)
            # against the high-sweep drag relief and pocketing 200 lbf of
            # fuel. Which side binds is configuration-dependent, so the
            # relaxation is not safe. The D8's convergence problem was solved
            # elsewhere: duplicate AR equalities (LICQ failure) and the inner
            # ipopt tolerance -- see solve options in the runners.
            tanL ** 2 * cosL ** 2 + cosL ** 2 == 1.0,        # [SP] SigEq
            # Hard bounds: 0 to 40 degrees of quarter-chord sweep.
            #
            # These are a stand-in, and deliberately blunt, because with the
            # York fit sweep is not being decided by physics. The fit is valid
            # only to M_perp ~ 0.74, so at cruise Mach the LOWER edge of usable
            # sweep is set by the validity fence below rather than by drag --
            # a 737 at M 0.785 lands on 19.49 deg because acos(0.74/0.785) is
            # 19.49, not because anything aerodynamic said so.
            #
            # The fence is at least Mach-ADAPTIVE, which is the physically
            # right shape: a slow aeroplane is required to sweep less, a fast
            # one more. So the pair behaves sensibly across the matrix even
            # though neither is a drag model. 40 deg is above anything in the
            # study (787: 32.2, 737: 25, A320: 25) and below where tip stall
            # and high-lift effectiveness -- neither of which this model has --
            # would start to dominate.
            #
            # Replace both with a polar refitted through the drag rise; then
            # sweep is priced rather than bounded. See components/polars.py.
            cosL <= 0.9999, cosL >= 0.766,
        ]

    # ---- structure --------------------------------------------------------
    box = wing.group("box", prefix=f"{wing.prefix}box_")
    # The closed-form box, over this single-taper planform. The station-based
    # TASOPT box lives with the cranked planform it was derived for, in
    # wing_tasopt.py -- pairing it with this geometry left the wing's taper
    # priced by nothing and drove it to 0.42.
    wb, wbcons = add_wingbox("wing", AR=AR, b=b, S=S, p=p, q=q, tau=tau,
                             Lmax=Lmax, tau_max=tau_max, group=box,
                             cosL=cosL if sweep_pricing else None,
                             weight_credit=1.0 / WING_MASS_FACTOR,
                             material=material)
    cons += wbcons
    out["box"] = wb

    # ---- secondary structure fractions ------------------------------------
    # TASOPT's 737 deck, runs/737/737.tas lines 219-225. f_slat was 0.001
    # here, which is 100x low and made slats weightless; every other fraction
    # already matched. It is not a technology parameter -- a slat weighs what a
    # slat weighs -- so it is corrected in place rather than made swappable.
    fnames = [("f_flap", 0.2), ("f_slat", 0.1), ("f_aileron", 0.04),
              ("f_lete", 0.1), ("f_ribs", 0.15), ("f_spoiler", 0.02),
              ("f_watt", 0.03)]
    fracs = [C(n, v, "-", f"{n} fractional weight") for n, v in fnames]

    cons += [
        # No correction here: it is applied inside the box, to W_cap and
        # W_web, so the COMPONENTS match TASOPT and not merely their sum.
        Wwing >= Cwing * wb["W_struct"] + wb["W_struct"] * sum(fracs),
        # REVERTED TO `<=` PENDING THE WING INVESTIGATION.
        #
        # As an equality this is the correct statement -- the offset from the
        # box station to the wing's area centroid is pure planform geometry
        # and should not be purchasable -- and the optimiser really was
        # exploiting the one-sided form, driving it to EXACTLY ZERO because
        # dx_AC_wing appears in the empty-CG moment sum and zeroing it holds
        # the CG forward for free.
        #
        # It is reverted anyway, for two measured reasons. It does NOT do what
        # it was changed for: it moves the wing box 0.07 m (20.161 -> 20.229)
        # against the 3.80 m offset from TASOPT it was meant to explain. And
        # it costs the locked-Mach case its convergence -- 2000 iterations and
        # 899 s leave stationarity at 6.8e-03, no better than at 400, so it is
        # a genuine conflict rather than a budget. Restore the equality once
        # the wing position is understood; the defect it fixes is real.
        # ORIGINAL NOTE: this is the chordwise offset from the wing box station to
        # the wing's aerodynamic/area centroid, and for a swept wing it is
        # pure planform geometry -- the outboard panels sit aft of the root,
        # so the centroid does too. As a `<=` it was a quantity the optimiser
        # could move for free, and it drove it to EXACTLY ZERO: dx_AC_wing
        # appears in the empty-CG moment sum (aircraft.py:1971) as
        # W_wing*(x_wing + dx_AC_wing), so zeroing it holds the empty CG
        # forward and buys static margin at no cost.
        #
        # The consequence was the largest geometric error in the model. With
        # the wing's AC pinned to the box station, the only way to put the AC
        # where trim and stability want it is to move the whole BOX aft -- so
        # the 737's wingbox sat at 20.16 m against TASOPT's 16.36 m, +3.80 m,
        # and dragged the rear spar and the main gear aft with it (x_m 22.64 m
        # against a real 20.5 m, pinned exactly to the rear spar). Comparing
        # centroid to centroid rather than box to box, the same two aircraft
        # differ by only +1.22 m: we were placing the BOX where TASOPT places
        # the CENTROID.
        #
        # The formula itself was never wrong -- it evaluates to 2.068 m here
        # against TASOPT's dxwing of 2.578 m -- only its sense.
        dxACwing <= 1. / 24. * (croot + 5. * ctip) / S * b ** 2 * tanL,
    ]

    # ---- per-segment aerodynamics -----------------------------------------
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
    etao = Vn("eta_o", 0.9, "-", "centre wing span coefficient")
    po = Vn("p_o", 2e4, "N/m", "centre section theoretical wing loading")
    dLt = Vn("dL_t", 1e3, "N", "wing tip lift loss")
    cmw = Vn("c_m_w", 0.1, "-", "wing pitching moment coefficient")

    fLo = C("f_L_o", 0.5, "-", "centre wing lift reduction coefficient")
    fLt = C("f_L_t", 0.05, "-", "wing tip lift reduction coefficient")
    amax = C("alpha_max_w", 0.1, "-", "max angle of attack")
    TipReduct = C("TipReduct", 1.0, "-",
                  "induced drag reduction from wing tip devices")

    out.update(alpha_w=alpha, C_L=CLw, C_d_w=CDw, D_wing=Dwing, L_w=Lw,
               C_L_alpha_w=CLaw, c_m_w=cmw, eta_o=etao, p_o=po)

    rho, Vinf, M, mu = state.rho, state.V, state.M, state.mu

    cons += [
        0.5 * rho * Vinf ** 2 * S * CLw >= Lw + dLo + 2. * dLt,
        dLo == etao * fLo * b / 2 * po,
        # TODO in source: c_root ~ c_o and taper ~ gamma_t.
        dLt == fLt * po * croot * taper ** 2,
        # DATCOM swept-wing lift curve slope. The `- M^2` is what makes this
        # signomial, and it is written as an equality on purpose.
        (AR / eta) ** 2 * (1 + tanL ** 2 - M ** 2) + 8 * pi * AR / CLaw
            == (2 * pi * AR / CLaw) ** 2,                           # [SP] SigEq
        CLw == CLaw * alpha,
        alpha <= amax,
        Dwing == 0.5 * rho * Vinf ** 2 * S * CDw,
        CDw >= CDp + CDi,
        CDi >= TipReduct * CLw ** 2 / (pi * e * AR),
        Re == rho * Vinf * mac / mu,
    ]

    # ---- profile drag, from a swappable polar ------------------------------
    # See components/polars.py. The polar owns its own coefficients, its
    # Reynolds convention, whether its lift argument is the perpendicular
    # section c_l, and the perpendicular Mach it stops being valid at.
    _polar = POLARS[polar] if isinstance(polar, str) else polar
    _cl = CLw / cosL ** 2 if _polar.perp_cl else CLw
    # Normalised polars carry a cd_ref (1.0 for the un-normalised ones), so
    # the same row serves both. See Polar.cd and polars.from_yaml.
    cons += [(CDp / _polar.cd_ref) ** _polar.alpha
             >= _polar.cd(Re, tau, cosL * M, _cl) * _polar.re_factor(Re)
             ** _polar.alpha]
    if _polar.m_perp_max is not None:
        # A statement about the FIT, not the aeroplane: it is slack at any real
        # design point (a 737 at M 0.785 with 25 deg sweep sits at M_perp =
        # 0.711) and binds only where the surrogate stops modelling anything.
        cons += [cosL * M <= _polar.m_perp_max]

    return wing, cons
