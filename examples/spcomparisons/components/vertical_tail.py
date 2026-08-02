"""Vertical tail sizing, structure and drag.

Source model
------------
``vertical_tail.py`` in https://github.com/convexengineering/SPaircraft.

The doubled-span trick
----------------------
The box beam in ``wingbox.py`` is written for a full-span surface, but a
vertical tail is a half-span one. The source reconciles this by handing the
box *doubled* span, area and maximum load, giving it a separate aspect ratio
variable ``AR_{vt}`` (explicitly "double span") rather than the tail's own
``A_{vt}``, and then halving the resulting structural weight. Get any one of
those four wrong and the tail is silently mis-sized, so they are kept together
here.

What is sized elsewhere
-----------------------
This model declares but does not constrain the engine-out case (``D_{wm}``,
``T_e``, ``y_{eng}``, ``L_{vt_{max}}``) or the yaw-rate landing case
(``V_{land}``, ``I_{z,max}``, ``\\dot{r}_{req}``). Those close at aircraft
level, because they need the fuselage and engine positions. The same is true
of the moment arm ``l_{vt}`` and the CG-relative distances.
"""
from __future__ import annotations

from numpy import cos, pi, tan

from .polars import (POLARS, TASOPT_ARE_XP, TASOPT_TAIL_CDF,
                     TASOPT_TAIL_CDP, TASOPT_TAIL_REREF)
from .wingbox import add_wingbox


#: Minimum vertical tail volume coefficient, per CONFIGURATION.
#:
#: Engine-out is the ONLY fin sizing case in this model, and for a
#: centreline-engine layout that is a weak requirement: the D8's fin collapsed
#: to 6.0 m2 with its spar-cap inertia resting on the solver's 1e-9 positivity
#: floor, which stops the KKT certificate closing (complementarity 1.0 against
#: a 1e-6 tolerance) even though the objective itself has converged.
#:
#: A real fin also has to handle crosswind landing, weathercock stability and
#: spin recovery, none of which this model contains, so a floor is standing in
#: for them. The values are TASOPT's own prescribed volume coefficients --
#: runs/737/737.tas `Vv 0.10` and runs/D8/d82.tas `Vv 0.03` -- rather than
#: numbers picked to make the solve behave.
#:
#: CONVENTIONAL stays at the old 0.001 -- deliberately inactive. A podded
#: underwing layout has a large engine-out moment, so engine-out sizes the fin
#: properly there: the 737 lands at V_vt 0.083 against a real 0.089 with the
#: floor nowhere near. Raising it to TASOPT's prescribed 0.10 would OVERRIDE a
#: working physical result with a worse number, so it is left alone.
V_VT_MIN_CONVENTIONAL = 0.001
#: DOUBLE-BUBBLE stays inactive too, and for a better reason than the
#: conventional case: the fin does not need it.
#:
#: It looked like it did. The D8 used to put S_vt on the floor at 6.0 m2 with
#: the box's I_cap resting on the solver's 1e-9 positivity bound, which reads
#: as a fin with no sizing case. It was not: a floor never cured it -- 0.03
#: went infeasible, 0.024 ran 200 iterations without converging -- because the
#: collapse was a SYMPTOM of a degenerate solve, not a missing constraint. Two
#: causes, both now fixed:
#:
#:   1. AR_vt was constrained twice over (see the note by A_vt below), so the
#:      fin's dual was a ray and complementarity could never close.
#:   2. With sweep left free the wing carries the signomial identity
#:      tan^2 cos^2 + cos^2 == 1, an EQUALITY with no interior. It sits at the
#:      top of the solver's blocking list and the elastic phase reports zero
#:      rows with positive slack (sum -5.7e-09) -- not infeasible, critically
#:      constrained. Pinning the D8's sweep converges at 15, 20 and 25 deg.
#:
#: With both addressed the fin sizes ITSELF: V_vt comes out 0.0285-0.0289
#: against TASOPT's prescribed 0.03, with this floor inactive by a factor of
#: ~29. So it is left off, and the number above is a disabled mechanism rather
#: than a calibration.
V_VT_MIN_DOUBLE_BUBBLE = 0.001


def add_vertical_tail(f, N, state, *, v_vt_min=V_VT_MIN_CONVENTIONAL, sweep_deg, prefix="VT_",
                      material=None, tau_limits=True, sweep_pricing=False,
                      drag_model="fit"):
    """Add the vertical tail. Returns ``(vars, constraints)``."""
    vt = f.group("vt", prefix=prefix)
    V, C = vt.Variable, vt.Constant
    Vn = lambda n, g, u, d: vt.Variable(n, g, u, d, size=N)

    # ---- planform ---------------------------------------------------------
    Avt = V("A_vt", 1.7, "-", "vertical tail aspect ratio")
    ARvt = V("AR_vt", 3.4, "-", "VT aspect ratio on doubled span")
    Svt = V("S_vt", 25.0, "m^2", "vertical tail reference area")
    bvt = V("b_vt", 6.5, "m", "vertical tail span")
    croot = V("c_root_vt", 5.0, "m", "vertical tail root chord")
    ctip = V("c_tip_vt", 1.5, "m", "vertical tail tip chord")
    cma = V("cbar_vt", 3.6, "m", "vertical tail mean aerodynamic chord")
    taper = V("lambda_vt", 0.3, "-", "vertical tail taper ratio")
    tau = V("tau_vt", 0.12, "-", "vertical tail thickness/chord ratio")
    p = V("p_vt", 1.6, "-", "substituted variable 1 + 2 taper")
    q = V("q_vt", 1.3, "-", "substituted variable 1 + taper")
    ymac = V("y_cbar_vt", 2.6, "m", "spanwise location of the mean aero chord")
    zmac = V("z_cbar_vt", 2.6, "m", "vertical location of the mean aero chord")
    e = V("e_vt", 0.8, "-", "span efficiency of the vertical tail")

    dxlead = V("dx_lead_vt", 16.0, "m", "CG to VT leading edge")
    dxtrail = V("dx_trail_vt", 21.0, "m", "CG to VT trailing edge")
    lvt = V("l_vt", 18.0, "m", "vertical tail moment arm")
    xCGvt = V("x_CG_vt", 35.0, "m", "x-location of the tail CG")

    Lvmax = V("L_vt_max", 5e5, "N", "maximum load for structural sizing")
    LvtEO = V("L_vt_EO", 1e5, "N", "vertical tail lift, engine out")
    CLvtEO = V("C_L_vt_EO", 0.5, "-", "VT lift coefficient, engine out")
    clvtEO = V("c_l_vt_EO", 0.7, "-", "sectional lift coefficient, engine out")
    CLvyaw = V("C_L_vt_yaw", 0.8, "-", "VT lift coefficient at rotation")
    Dwm = V("D_wm", 5e3, "N", "engine-out windmill drag")
    V1 = V("V_1", 65.0, "m/s", "minimum takeoff velocity")
    Te = V("T_e", 1e5, "N", "thrust per engine at takeoff")
    Iz = V("I_z_max", 1e7, "kg*m^2", "aircraft z-axis moment of inertia")
    Vvt = V("V_vt", 0.1, "-", "vertical tail volume coefficient")
    Wvt = V("W_vt", 1e4, "N", "total VT system weight")

    # ---- constants --------------------------------------------------------
    # 1.0 -- NO empirical correction. Tried at 1.25 and pulled back out; what
    # it revealed is recorded below because it says where the real fix is.
    #
    # A factor here stands in for the CARRY-THROUGH
    # structure -- the centre section that carries the root moment across the
    # fuselage -- which TASOPT builds explicitly for every surface
    # (Wvtail = (Wscen + Wsinn + Wsout)*(1 + fvadd)*nvtail, wsize.py:1028) and
    # this box does not build at all. The box here is W_struct = W_web + W_cap
    # for the exposed panels only, which is why the fin came out at 0.725 of
    # TASOPT and 0.853 of the real aircraft while its added-weight fraction
    # f_VT = 0.4 already matches TASOPT's fvadd exactly.
    #
    # It is NOT a gauge effect -- measured, the fin's cap runs 2.9 mm and its
    # web 3.6 mm, two to three times any minimum gauge, so that constraint
    # would never bind.
    #
    # And a multiplier does not work HERE even as a stopgap, which is the
    # useful finding. At 1.25 the fin did not get denser, it got BIGGER:
    # S_vt 30.0 -> 32.3 m2 (1.22 of the real fin) while areal weight stayed
    # under at 50.8 lbf/m2 against a real 57. A heavier fin raises MTOW,
    # raises thrust, raises the engine-out moment, and the extra root chord
    # runs FORWARD against the pinned trailing edge, shortening l_vt and
    # demanding more area again. The fin is sized by engine-out, so weight
    # feeds straight back into area.
    #
    # The physical replacement is a monomial and needs no new calibration:
    #     W_cen = rho*g*(t_cap*r_w_c + t_web*r_h*tau) * c_root^2 * b_o
    # with b_o the fuselage width at the tail station. Retire this factor when
    # that is built.
    CVT = C("C_VT", 1.0, "-", "VT weight margin and sensitivity")
    mu0 = C("mu_0", 1.8e-5, "N*s/m^2", "dynamic viscosity at sea level")
    tanL = C("tan_Lambda_vt", tan(sweep_deg * pi / 180), "-",
             "tangent of leading edge sweep")
    # The fin's sweep stays an INPUT, as it is in TASOPT -- getparm.f reads
    # igsweepv separately and fobj.f never assigns it, so TASOPT optimizes the
    # wing and horizontal together and leaves the vertical fixed. It is priced
    # structurally now even so: cos(Lambda_vt) goes to the box.
    cosLv = C("cos_Lambda_vt", cos(sweep_deg * pi / 180), "-",
              "cosine of vertical tail sweep")

    out = dict(A_vt=Avt, S_vt=Svt, b_vt=bvt, c_root_vt=croot, c_tip_vt=ctip,
               cbar_vt=cma, lambda_vt=taper, tau_vt=tau, p_vt=p, q_vt=q,
               y_cbar_vt=ymac, z_cbar_vt=zmac, dx_lead_vt=dxlead,
               dx_trail_vt=dxtrail, l_vt=lvt, x_CG_vt=xCGvt, L_vt_max=Lvmax,
               L_vt_EO=LvtEO, C_L_vt_EO=CLvtEO, D_wm=Dwm, V_1=V1, T_e=Te,
               I_z_max=Iz, V_vt=Vvt, W_vt=Wvt, C_L_vt_yaw=CLvyaw, e_vt=e,
               c_l_vt_EO=clvtEO, AR_vt=ARvt)

    rho0 = V("rho_TO", 1.225, "kg/m^3", "air density at sea level")
    # A fin with rudder reaches ~1.2-1.5, not 2.6.
    # 2.6 is TASOPT's own number for this exact quantity (runs/737/737s.tas:265,
    # "CLvmax  VT max +/-CL at Vmn, for VT structural sizing"). It was 1.4 here,
    # and the consequence was almost the entire vertical-bending shortfall: this
    # coefficient sets L_vt_max, which sets B_1v, which sets the tailcone's
    # vertical bending material, so W_vbend came out at 0.513 of TASOPT against
    # a coefficient ratio of 1.4/2.6 = 0.538. The horizontal's counterpart was
    # ported correctly -- C_L_ht_max = 2.0 matches TASOPT's CLhmax = 2.0
    # exactly -- which is what makes this look like a slip rather than a choice.
    #
    # Note this is a STRUCTURAL design coefficient at V_ne, not a control one:
    # it says what load the fin must survive, not what the rudder can deliver.
    # c_l_vt_EO is the control-side number and stays separate.
    CLvmax = C("C_L_vt_max", 2.6, "-", "max VT lift coefficient")
    Vvtmin = C("V_vt_min", v_vt_min, "-", "minimum VT volume coefficient")
    out.update(rho_TO=rho0, C_L_vt_max=CLvmax, V_vt_min=Vvtmin)

    cons = [
        CLvyaw == 0.85 * CLvmax,
        LvtEO == 0.5 * rho0 * V1 ** 2 * Svt * CLvtEO,
        CLvtEO * (1 + clvtEO / (pi * e * Avt)) <= clvtEO,
        Avt == bvt ** 2 / Svt,
        # AR_vt, the STRUCTURAL aspect ratio, is NOT written here. It used to
        # be, as `ARvt == 2.0*Avt`, and that row was one equation too many.
        #
        # The bug it was added to fix was real: AR_vt was declared free with no
        # relation to A_vt anywhere, so the optimizer could take the
        # aerodynamic benefit of a high-aspect-ratio fin (CL_vt_EO rises with
        # A_vt, so the fin can be smaller) while handing the box a low aspect
        # ratio and dodging the structural penalty entirely.
        #
        # But add_wingbox already closes it. It is handed span 2*b_vt and area
        # 2*S_vt by the doubled-span convention and imposes AR == b^2/S, i.e.
        # AR_vt == (2 b_vt)^2/(2 S_vt) == 2 b_vt^2/S_vt -- which is exactly
        # this row composed with the one above. Three equalities, two
        # independent facts, and CONSISTENT, so the iterate was always right
        # and only the dual was sick: the three gradients are linearly
        # dependent (-r1 + r2 + r3 == 0 identically), LICQ fails, the
        # multiplier is a ray rather than a point, and SIA drove it along that
        # ray to the tau ceiling of 1e12. Complementarity is |lambda * log g|,
        # so 1e12 times a machine-zero residual is O(1) forever -- the test
        # cannot pass no matter how converged the point is.
        #
        # That is what stopped the D8: not the fin, which is why a volume floor
        # never helped. AR_vt is still fully determined, just derived from
        # b_vt and S_vt in one place instead of two.
        # EQUALITY: a trapezoid's area IS span x mean chord. The wing and the
        # horizontal tail both write this one as ==; the fin was left <= in
        # the source, which lets the optimizer claim a larger planform than
        # the area it pays structural weight for.
        Svt == bvt * (croot + ctip) / 2,                            # [SP] SigEq
        # EQUALITIES, as for the horizontal tail: definitions of substituted
        # variables rather than design freedoms.
        p == 1 + 2 * taper,
        q == 1 + taper,
        # TRANSPOSED. Unlike the wing and the horizontal tail the FACTOR is
        # right -- b_vt is the fin HEIGHT, one panel root-to-tip rather than
        # tip-to-tip (note ARvt == 2.0*Avt above), so the centroid genuinely
        # sits at b/3 -- but p/q was inverted. Correcting it RAISES the mean
        # chord, 2.127 m -> 3.222 m, and with it the fin arm through
        #     dx_lead + z_mac*tan(Lambda) + 0.25*cbar == l_vt
        # which is the opposite direction to the horizontal tail's correction.
        ymac == (bvt / 3) * p / q,
        zmac == (bvt / 3) * p / q,
        (2. / 3) * (1 + taper + taper ** 2) * croot / q == cma,     # [SP] SigEq
        taper == ctip / croot,
        dxlead + croot <= dxtrail,
        # EQUALITY, for the same reason as the horizontal tail arm: this is
        # geometry. Bounded above only, the fin arm shrinks to buy cheap yaw
        # inertia (Iz_tail carries W_vt*l_vt^2) with nothing holding it out.
        dxlead + ymac * tanL + 0.25 * cma == lvt,            # [SP] SigEq
        # TODO in source: constrain taper by tip Reynolds number instead.
        taper >= 0.25,
        Vvt >= Vvtmin,
        # Fin aspect ratio, capped at TASOPT's own value (runs/737/737s.tas:253,
        # "2.0 ! ARv VT aspect ratio"); real transport fins sit near 1.9.
        #
        # The cap DOES decide it -- the model's own optimum runs higher -- and
        # that is understood rather than accidental. A taller fin puts its mean
        # aerodynamic chord higher, and through
        #     ymac == (b_vt/3) q/p ;  dx_lead + ymac tan(Lambda) + c/4 == l_vt
        # a 25 degree sweep converts that height into moment ARM. So height
        # buys effectiveness, and the only things that stop a real designer
        # exploiting it -- fin/rudder flutter and rudder effectiveness at large
        # deflection -- are not in this model. Until they are, the bound stands
        # in for them, at the reference implementation's value rather than at
        # one I picked.
        Avt >= 0.8, Avt <= 2.0,
        # Engine-out sectional lift coefficient. A finned surface with a rudder
        # reaches ~1.0-1.2 before it stalls; 0.5, the pinned value, is what a
        # fin does at about half deflection.
        clvtEO <= 1.0,
    ]

    # ---- structure, on the doubled span -----------------------------------
    box = vt.group("box", prefix=f"{vt.prefix}box_")
    # `material` was not being passed, so the fin box silently fell back to
    # add_wingbox's default aluminium (250 MPa) while the wing and horizontal
    # tail took theirs from the technology level. Under TECH=cfm56_era that
    # left the fin sized on 2020s allowables and therefore too light -- the
    # kind of gap that shows up as agreement, because a lighter fin makes the
    # aeroplane closer to a reference it should not have matched that way.
    wb, wbcons = add_wingbox("vertical_tail", AR=ARvt, b=2. * bvt,
                             S=2. * Svt, p=p, q=q, tau=tau, Lmax=2. * Lvmax,
                             group=box, material=material,
                             tau_limits=tau_limits,
                             cosL=cosLv if sweep_pricing else None)
    cons += wbcons
    out["box"] = wb

    fVT = C("f_VT", 0.4, "-", "VT fractional weight")
    numspar = C("N_spar", 1.0, "-",
                "spars per VT carrying stress in the 1-in-20 case")
    cons += [Wvt >= numspar * CVT * (wb["W_struct"] + wb["W_struct"] * fVT)]

    # ---- per-segment drag --------------------------------------------------
    Dvt = Vn("D_vt", 3e3, "N", "vertical tail viscous drag in cruise")
    Rec = Vn("Re_vt", 2e7, "-", "vertical tail Reynolds number in cruise")
    CDvis = Vn("C_D_vis_vt", 0.005, "-", "viscous drag coefficient")
    out.update(D_vt=Dvt, C_D_vis_vt=CDvis)

    rho, Vinf, M, mu = state.rho, state.V, state.M, state.mu
    cons += [
        Dvt >= 0.5 * rho * Vinf ** 2 * Svt * CDvis,
        Rec == rho * Vinf * cma / mu,
    ]
    if drag_model == "tasopt":
        # TASOPT's own tail treatment: constants, Re-scaled. See polars.py.
        cdft = C("cd_f_t", TASOPT_TAIL_CDF, "-", "tail friction cd (TASOPT deck)")
        cdpt = C("cd_p_t", TASOPT_TAIL_CDP, "-", "tail pressure cd (TASOPT deck)")
        Reref = C("Re_ref_t", TASOPT_TAIL_REREF, "-", "tail reference Reynolds")
        cons += [CDvis >= (cdft + cdpt) * (Rec / Reref) ** TASOPT_ARE_XP]
    elif drag_model == "fit":
        cons += [
        # Martin's TASOPT tail drag fit. The exponents on the first and last
        # terms (tau^133.8 M^1022.7, M^-114.6) are not typos: they are
        # near-vertical barriers that fence the fit's valid region.
        CDvis ** 1.18909 >= (
            2.43701e-77 * Rec ** -0.52841 * tau ** 133.796 * M ** 1022.7
            + 0.00304307 * Rec ** -0.409988 * tau ** 1.22062 * M ** 1.55119
            + 0.000196709 * Rec ** 0.214479 * tau ** -0.0383195 * M ** -0.137561
            + 6.59349e-50 * Rec ** -0.498092 * tau ** 1.55922 * M ** -114.577),
        ]
    elif drag_model == "mses":
        # T-series MSES surrogate, CL-independent (fit at CL=0.1, which is
        # where a tail actually lives). Replaces TASOPT's two Mach-independent
        # constants with a fit that has a real transonic rise -- CD roughly
        # 9x between M 0.70 and 0.85 on a 12% section -- so the PERPENDICULAR
        # Mach is what must be fed here, not the flight Mach. Getting that
        # wrong would be a large error rather than a small one.
        _tp = POLARS["mses_t_tail"]
        cons += [(CDvis / _tp.cd_ref) ** _tp.alpha
                 >= _tp.cd(Rec, tau, cosLv * M, 1.0)]
    else:
        raise ValueError(f"unknown tail drag_model {drag_model!r}")

    return vt, cons
