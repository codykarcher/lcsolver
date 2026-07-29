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

from .wingbox import add_wingbox


def add_vertical_tail(f, N, state, *, sweep_deg, prefix="VT_"):
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

    out = dict(A_vt=Avt, S_vt=Svt, b_vt=bvt, c_root_vt=croot, c_tip_vt=ctip,
               cbar_vt=cma, lambda_vt=taper, tau_vt=tau, p_vt=p, q_vt=q,
               y_cbar_vt=ymac, z_cbar_vt=zmac, dx_lead_vt=dxlead,
               dx_trail_vt=dxtrail, l_vt=lvt, x_CG_vt=xCGvt, L_vt_max=Lvmax,
               L_vt_EO=LvtEO, C_L_vt_EO=CLvtEO, D_wm=Dwm, V_1=V1, T_e=Te,
               I_z_max=Iz, V_vt=Vvt, W_vt=Wvt, C_L_vt_yaw=CLvyaw, e_vt=e,
               c_l_vt_EO=clvtEO, AR_vt=ARvt)

    rho0 = V("rho_TO", 1.225, "kg/m^3", "air density at sea level")
    CLvmax = C("C_L_vt_max", 2.6, "-", "max VT lift coefficient")
    Vvtmin = C("V_vt_min", 0.001, "-", "minimum VT volume coefficient")
    out.update(rho_TO=rho0, C_L_vt_max=CLvmax, V_vt_min=Vvtmin)

    cons = [
        CLvyaw == 0.85 * CLvmax,
        LvtEO == 0.5 * rho0 * V1 ** 2 * Svt * CLvtEO,
        CLvtEO * (1 + clvtEO / (pi * e * Avt)) <= clvtEO,
        Avt == bvt ** 2 / Svt,
        Svt <= bvt * (croot + ctip) / 2,                            # [SP]
        p >= 1 + 2 * taper,
        2 * q >= 1 + p,
        ymac == (bvt / 3) * q / p,
        zmac == (bvt / 3) * q / p,
        (2. / 3) * (1 + taper + taper ** 2) * croot / q == cma,     # [SP] SigEq
        taper == ctip / croot,
        dxlead + croot <= dxtrail,
        dxlead + ymac * tanL + 0.25 * cma >= lvt,                   # [SP]
        # TODO in source: constrain taper by tip Reynolds number instead.
        taper >= 0.25,
        Vvt >= Vvtmin,
    ]

    # ---- structure, on the doubled span -----------------------------------
    box = vt.group("box", prefix=f"{vt.prefix}box_")
    wb, wbcons = add_wingbox("vertical_tail", AR=ARvt, b=2. * bvt,
                             S=2. * Svt, p=p, q=q, tau=tau, Lmax=2. * Lvmax,
                             group=box)
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
        # Martin's TASOPT tail drag fit. The exponents on the first and last
        # terms (tau^133.8 M^1022.7, M^-114.6) are not typos: they are
        # near-vertical barriers that fence the fit's valid region.
        CDvis ** 1.18909 >= (
            2.43701e-77 * Rec ** -0.52841 * tau ** 133.796 * M ** 1022.7
            + 0.00304307 * Rec ** -0.409988 * tau ** 1.22062 * M ** 1.55119
            + 0.000196709 * Rec ** 0.214479 * tau ** -0.0383195 * M ** -0.137561
            + 6.59349e-50 * Rec ** -0.498092 * tau ** 1.55922 * M ** -114.577),
    ]

    return vt, cons
