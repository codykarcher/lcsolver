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

from .wingbox import add_wingbox


def add_wing(f, N, state, *, sweep_deg, prefix="Wing_"):
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
    tau_max = C("tau_max_w", 0.14733, "-", "max allowed wing thickness")
    rhofuel = C("rho_fuel", 817.0, "kg/m^3", "density of fuel")
    Cwing = C("C_wing", 1.0, "-", "wing weight margin and sensitivity factor")
    cosL = C("cos_Lambda", cos(sweep_deg * pi / 180), "-",
             "cosine of quarter-chord sweep")
    tanL = C("tan_Lambda", tan(sweep_deg * pi / 180), "-",
             "tangent of quarter-chord sweep")

    out = dict(AR=AR, S=S, b=b, c_root=croot, c_tip=ctip, mac=mac, y_mac=ymac,
               taper=taper, p=p, q=q, tau=tau, e=e, x_w=xw,
               dx_AC_wing=dxACwing, L_max=Lmax, V_fuel_max=Vfuel,
               W_fuel_wing=WfuelWing, W_wing=Wwing, A_rect=Arect)

    cons = [
        Arect == ctip * b,
        p >= 1 + 2 * taper,
        2 * q >= 1 + p,
        ymac == (b / 3) * q / p,
        (2. / 3) * (1 + taper + taper ** 2) * croot / q <= mac,
        taper == ctip / croot,
        S == b * (croot + ctip) / 2,                                # [SP] SigEq
        # Oswald efficiency: Nita & Scholz, "Estimating the Oswald factor from
        # basic aircraft geometrical parameters".
        fl >= (0.0524 * taper ** 4 - 0.15 * taper ** 3 + 0.1659 * taper ** 2
               - 0.0706 * taper + 0.0119),
        e * (1 + fl * AR) <= 1,
        taper >= 0.15,
        # Fuel volume, GP approximation of the TASOPT signomial constraint.
        Vfuel <= 0.3026 * mac ** 2 * b * tau,
        WfuelWing <= rhofuel * Vfuel * g,
    ]

    # ---- structure --------------------------------------------------------
    box = wing.group("box", prefix=f"{wing.prefix}box_")
    wb, wbcons = add_wingbox("wing", AR=AR, b=b, S=S, p=p, q=q, tau=tau,
                             Lmax=Lmax, tau_max=tau_max, group=box)
    cons += wbcons
    out["box"] = wb

    # ---- secondary structure fractions ------------------------------------
    fnames = [("f_flap", 0.2), ("f_slat", 0.001), ("f_aileron", 0.04),
              ("f_lete", 0.1), ("f_ribs", 0.15), ("f_spoiler", 0.02),
              ("f_watt", 0.03)]
    fracs = [C(n, v, "-", f"{n} fractional weight") for n, v in fnames]

    cons += [
        Wwing >= Cwing * wb["W_struct"] + wb["W_struct"] * sum(fracs),
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
        # Martin York's fit to the TASOPT C-series transonic airfoils.
        CDp ** 1.6515 >= (
            1.61418 * (Re / 1000) ** -0.550434 * tau ** 1.29151
                * (cosL * M) ** 3.03609 * CLw ** 1.77743
            + 0.0466407 * (Re / 1000) ** -0.389048 * tau ** 0.784123
                * (cosL * M) ** -0.340157 * CLw ** 0.950763
            + 190.811 * (Re / 1000) ** -0.218621 * tau ** 3.94654
                * (cosL * M) ** 19.2524 * CLw ** 1.15233
            + 2.82283e-12 * (Re / 1000) ** 1.18147 * tau ** -1.75664
                * (cosL * M) ** 0.10563 * CLw ** -1.44114),
    ]

    return wing, cons
