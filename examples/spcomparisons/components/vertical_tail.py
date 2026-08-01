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


def add_vertical_tail(f, N, state, *, sweep_deg, prefix="VT_",
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
    CLvmax = C("C_L_vt_max", 1.4, "-", "max VT lift coefficient")
    Vvtmin = C("V_vt_min", 0.001, "-", "minimum VT volume coefficient")
    out.update(rho_TO=rho0, C_L_vt_max=CLvmax, V_vt_min=Vvtmin)

    cons = [
        CLvyaw == 0.85 * CLvmax,
        LvtEO == 0.5 * rho0 * V1 ** 2 * Svt * CLvtEO,
        CLvtEO * (1 + clvtEO / (pi * e * Avt)) <= clvtEO,
        Avt == bvt ** 2 / Svt,
        # The STRUCTURAL aspect ratio, tied to the planform one.
        #
        # AR_vt was declared free and never constrained -- three references in
        # this file, all of them declaration, export or the wingbox call, and
        # no relation to A_vt anywhere. That let the optimizer take the
        # aerodynamic benefit of a high-aspect-ratio fin (CL_vt_EO rises with
        # A_vt, so the fin can be smaller) while handing the box a low aspect
        # ratio and dodging the structural penalty entirely. A_vt duly sat on
        # whatever cap it was given.
        #
        # The box is handed span 2*b_vt and area 2*S_vt by the doubled-span
        # convention, so its aspect ratio is (2b)^2/(2S) = 2 A_vt. Not a
        # modelling choice -- arithmetic.
        ARvt == 2.0 * Avt,
        # EQUALITY: a trapezoid's area IS span x mean chord. The wing and the
        # horizontal tail both write this one as ==; the fin was left <= in
        # the source, which lets the optimizer claim a larger planform than
        # the area it pays structural weight for.
        Svt == bvt * (croot + ctip) / 2,                            # [SP] SigEq
        p >= 1 + 2 * taper,
        2 * q >= 1 + p,
        ymac == (bvt / 3) * q / p,
        zmac == (bvt / 3) * q / p,
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
