"""Landing gear sizing: geometry, tip-over, struts, wheels and weight.

Source model
------------
``landing_gear.py`` in https://github.com/convexengineering/SPaircraft.
Sources cited there: Torenbeek p360, Raymer p233/p244, Currey p145/p264,
and Mason's machining-constraint notes.

Two source defects reproduced here
----------------------------------
Both are reproduced rather than corrected, because the reference solution
depends on them. See DISCREPANCIES.md §17-18.

1. **The main-gear weight factor is computed in newtons, the nose-gear one in
   pounds-force.** The source writes::

       Fwm == Lwm*dtm/(1000*Lwm.units*dtm.units)     # Lwm.units is N
       Fwn == Lwn*dtn/(1000*units.lbf*units.inches)

   Currey's correlation ``F_w = L_w[lbf] * d[in] / 1000`` wants lbf. The nose
   line converts; the main line divides by its own declared unit, which is
   newtons, so ``F_{w_m}`` comes out 4.448x too large and the wheel assembly
   weight ``1.2 F_w^0.609`` about 2.4x too large. The tire-diameter constraint
   two lines below *does* convert (``Lwm/(4.44*units.N)``), which is what makes
   this look like an oversight rather than a choice.

2. **The nose-gear machining constraint uses the main-gear radius.** The
   source has::

       2*r_m/t_m <= 40
       2*r_m/t_n <= 40      # r_m, not r_n

   so the nose strut wall thickness is bounded against the *main* strut
   radius. The nose gear's own slenderness is never constrained.
"""
from __future__ import annotations

import math

import numpy as np
from pyomo.environ import units


def add_landing_gear(f, *, prefix="LG_"):
    """Add the landing gear. Returns ``(vars, constraints)``."""
    lg = f.group("lg", prefix=prefix)
    V, C = lg.Variable, lg.Constant

    # ---- geometry ---------------------------------------------------------
    B = V("B", 15.0, "m", "landing gear base")
    T = V("T", 6.0, "m", "main landing gear track")
    x_m = V("x_m", 20.0, "m", "x-location of main gear")
    x_n = V("x_n", 5.0, "m", "x-location of nose gear")
    y_m = V("y_m", 3.0, "m", "y-location of main gear (symmetric)")
    l_m = V("l_m", 2.0, "m", "length of main gear")
    l_n = V("l_n", 2.0, "m", "length of nose gear")
    dxm = V("dx_m", 1.0, "m", "distance between main gear and CG")
    dxn = V("dx_n", 14.0, "m", "distance between nose gear and CG")
    xcglg = V("x_CG_lg", 18.0, "m", "landing gear CG")
    z_CG_0 = V("z_CG", 2.0, "m", "CG height above the bottom of the fuselage")
    zwing = V("z_wing", 1.0, "m", "wing height above the fuselage base")
    hhold = V("h_hold", 1.0, "m", "hold height")
    x_upswp = V("x_up", 28.0, "m", "fuselage upsweep point")
    d_nac = V("d_nacelle", 2.0, "m", "nacelle diameter")
    t_nac = V("t_nacelle", 0.15, "m", "nacelle thickness")

    tan_phi = V("tan_phi", math.tan(math.radians(15.0)), "-", "angle between main gear and CG")
    tan_psi = V("tan_psi", math.tan(math.radians(63.0)), "-", "tip over angle")

    # ---- loads -------------------------------------------------------------
    L_m = V("L_m", 7e5, "N", "max static load through main gear")
    L_n = V("L_n", 1e5, "N", "min static load through nose gear")
    L_n_dyn = V("L_n_dyn", 8e4, "N", "dynamic braking load, nose gear")
    Lwm = V("L_w_m", 1.7e5, "N", "static load per wheel (main)")
    Lwn = V("L_w_n", 5e4, "N", "static load per wheel (nose)")
    Eland = V("E_land", 5e5, "J", "max KE to be absorbed in landing")

    # ---- struts and wheels -------------------------------------------------
    S_sa = V("S_sa", 0.3, "m", "stroke of the shock absorber")
    l_oleo = V("l_oleo", 0.75, "m", "length of the oleo shock absorber")
    d_oleo = V("d_oleo", 0.2, "m", "diameter of the oleo shock absorber")
    r_m = V("r_m", 0.08, "m", "radius of main gear struts")
    r_n = V("r_n", 0.05, "m", "radius of nose gear struts")
    t_m = V("t_m", 0.01, "m", "thickness of main gear strut wall")
    t_n = V("t_n", 0.008, "m", "thickness of nose gear strut wall")
    I_m = V("I_m", 1e-5, "m^4", "area moment of inertia (main strut)")
    I_n = V("I_n", 1e-6, "m^4", "area moment of inertia (nose strut)")
    dtm = V("d_t_m", 44.0, "in", "diameter of main gear tires")
    dtn = V("d_t_n", 35.0, "in", "diameter of nose gear tires")
    wtm = V("w_t_m", 0.4, "m", "width of main tires")
    wtn = V("w_t_n", 0.3, "m", "width of nose tires")
    Fwm = V("F_w_m", 5e3, "-", "weight factor (main)")
    Fwn = V("F_w_n", 500.0, "-", "weight factor (nose)")
    WAWm = V("W_wa_m", 200.0, "lbf", "wheel assembly weight, one main wheel")
    WAWn = V("W_wa_n", 60.0, "lbf", "wheel assembly weight, one nose wheel")

    # ---- weights -----------------------------------------------------------
    W_lg = V("W_lg", 4e4, "N", "weight of landing gear")
    W_mg = V("W_mg", 3e4, "N", "weight of main gear")
    W_ms = V("W_ms", 5e3, "N", "weight of main struts")
    W_mw = V("W_mw", 8e3, "N", "weight of main wheels (per strut)")
    W_ng = V("W_ng", 5e3, "N", "weight of nose gear")
    W_ns = V("W_ns", 1e3, "N", "weight of nose strut")
    W_nw = V("W_nw", 2e3, "N", "weight of nose wheels (total)")

    # ---- constants ---------------------------------------------------------
    g = C("g", 9.81, "m/s^2", "gravitational acceleration")
    E = C("E", 205.0, "GPa", "modulus of elasticity, 4340 steel")
    K = C("K", 2.0, "-", "column effective length factor")
    N_s = C("N_s", 2.0, "-", "factor of safety")
    eta_s = C("eta_s", 0.8, "-", "shock absorber efficiency")
    # Everything a landing gear is that this model does not compute: the oleo
    # (piston, seals, fluid, metering), trunnion, side and drag braces,
    # retraction actuation, doors and bay structure. What IS computed is a
    # thin-walled steel tube against yield and Euler buckling, plus wheels,
    # tyres and brakes -- a small fraction of a real leg.
    #
    # 1.5 left the gear at 2.0% of MTOW where transport gear runs 4-5%. The
    # empirical floors that used to paper over that are gone, because a flat
    # fraction of MTOW is length-blind and, binding, it overrode the structural
    # model's own length sensitivity. Calibrating here instead keeps that
    # sensitivity -- the strut is linear in length -- while putting the
    # magnitude where a real gear sits.
    faddm = C("f_add_m", 4.3, "-", "proportional added weight, main")
    faddn = C("f_add_n", 4.3, "-", "proportional added weight, nose")
    h_nac = C("h_nacelle", 0.5, "m", "minimum nacelle clearance")
    lam = C("lambda_LG", 2.5, "-", "ratio of max to static load")
    n_mg = C("n_mg", 2.0, "-", "number of main gear struts")
    nwps = C("n_wps", 2.0, "-", "number of wheels per strut")
    p_oleo = C("p_oleo", 1800.0, "lbf/in^2", "oleo pressure")
    rho_st = C("rho_st", 7850.0, "kg/m^3", "density of 4340 steel")
    sig_y_c = C("sigma_y_c", 470e6, "Pa",
                "compressive yield strength, 4340 steel")
    tan_15 = C("tan_phi_min", math.tan(math.radians(15.0)), "-",
                "lower bound on phi (15 deg)")
    tan_63 = C("tan_psi_max", math.tan(math.radians(63.0)), "-",
                "upper bound on psi (63 deg)")
    tan_gam = C("tan_gamma", math.tan(math.radians(5.0)), "-", "dihedral angle")
    # 12 degrees, the geometric tail-strike limit for a transport of this
    # class -- a 737-800 has a tail skid and strikes at about 11. Checking the
    # real aircraft against the upswept-tail form above gives tan(theta) <=
    # (2.0 + 1.5*1.88)/(39.5-17.5) = 0.219, i.e. 12.4 degrees, so 12 is
    # consistent with the aeroplane rather than fitted to our answer. The 15
    # that was here is a rotation angle, not a strike angle.
    tan_th = C("tan_theta_max", math.tan(math.radians(12.0)), "-",
                "max rotation angle")
    w_ult = C("w_ult", 10.0, "ft/s", "ultimate velocity of descent")
    Clg = C("C_lg", 1.0, "-", "landing gear weight margin/sensitivity factor")

    out = dict(B=B, T=T, x_m=x_m, x_n=x_n, y_m=y_m, l_m=l_m, l_n=l_n,
               dx_m=dxm, dx_n=dxn, x_CG_lg=xcglg, z_CG=z_CG_0, z_wing=zwing,
               h_hold=hhold, x_up=x_upswp, L_m=L_m, L_n=L_n, L_n_dyn=L_n_dyn,
               E_land=Eland, W_lg=W_lg, W_mg=W_mg, W_ng=W_ng, d_t_m=dtm,
               d_nacelle=d_nac, t_nacelle=t_nac, h_nacelle=h_nac,
               w_ult=w_ult, lambda_LG=lam, tan_phi=tan_phi, tan_psi=tan_psi,
               W_ms=W_ms, W_mw=W_mw, W_ns=W_ns, W_nw=W_nw, l_oleo=l_oleo,
               d_oleo=d_oleo, S_sa=S_sa, r_m=r_m, r_n=r_n, t_m=t_m, t_n=t_n,
               d_t_n=dtn, w_t_m=wtm, w_t_n=wtn, F_w_m=Fwm, F_w_n=Fwn,
               W_wa_m=WAWm, W_wa_n=WAWn, L_w_m=Lwm, L_w_n=Lwn, g=g)

    N, inch, lbf, m = units.N, units.inch, units.lbf, units.m

    cons = [
        # Track and base geometry
        l_n + zwing + y_m * tan_gam >= l_m,                          # [SP]
        T == 2 * y_m,
        # EQUALITY: B is the wheelbase, which IS the distance from the nose gear
        # to the main gear. Written as <= it may be shorter than the gear
        # actually is, and that pays -- the nose gear load is L_n = W*dx_m/B,
        # so a short B manufactures nose load. With the forward gear margin
        # imposed, the optimizer shrank B and parked the nose gear at the tip
        # of the nose (x_n = 1.3e-09 m).
        x_n + B == x_m,                                              # [SP] SigEq
        # The hard `x_n >= 5 m` that used to sit here is gone. It was a D8.2
        # dimension applied to every aircraft in the matrix, and through
        # `x_n <= l_nose` at aircraft level it forced a 5 m nose section onto a
        # Citation as well as a 787. What actually places a nose gear is the
        # forward gear margin -- the share of weight it carries -- and that is
        # imposed at aircraft level where the CG lives.
        # Longitudinal tip over (static)
        tan_phi == tan_15,
        # Lateral tip over in a turn (dynamic); stricter with forward CG.
        1 >= (z_CG_0 + l_m) ** 2 * (y_m ** 2 + B ** 2) / (dxn * y_m * tan_psi) ** 2,
        tan_psi <= tan_63,
        # Tail strike: longitudinal ground clearance at rotation.
        x_upswp - x_m <= l_m / tan_th,                               # [SP]
        # Volume for retraction
        y_m >= l_m,
        # Hard landing, Torenbeek (10-28)
        S_sa == (1 / eta_s) * (Eland / (L_m * lam)),
        l_oleo == 2.5 * S_sa,                                        # Raymer 244
        d_oleo == 1.3 * (4 * lam * L_m / n_mg / (np.pi * p_oleo)) ** 0.5,
        l_m >= l_oleo + dtm / 2,

        # Wheel weights (Currey p145). NOTE the asymmetry between these two
        # lines -- the main one divides by newtons, the nose one by lbf. See
        # the module docstring; reproduced as written.
        Fwm == Lwm * dtm / (1000 * N * inch),
        WAWm == 1.2 * Fwm ** 0.609 * lbf,
        Fwn == Lwn * dtn / (1000 * lbf * inch),
        WAWn == 1.2 * Fwn ** 0.609 * lbf,
        Lwm == L_m / (n_mg * nwps),
        Lwn == L_n / nwps,

        # Main wheel diameter/width (Raymer p233). The 4.44 is the source's
        # own slightly-off newton-to-lbf factor.
        dtm == 1.63 * (Lwm / (4.44 * N)) ** 0.315 * inch,
        wtm == 0.1043 * (Lwm / (4.44 * N)) ** 0.48 * inch,
        dtn == 0.8 * dtm,
        wtn == 0.8 * wtm,

        # Strut weights and compressive yield in the hard landing case
        W_ms >= 2 * np.pi * r_m * t_m * l_m * rho_st * g,
        N_s * lam * L_m / n_mg <= sig_y_c * (2 * np.pi * r_m * t_m),
        W_mw == nwps * WAWm,
        W_ns >= 2 * np.pi * r_n * t_n * l_n * rho_st * g,
        N_s * (L_n + L_n_dyn) <= sig_y_c * (2 * np.pi * r_n * t_n),
        W_nw >= nwps * WAWn,

        # Buckling
        L_m <= np.pi ** 2 * E * I_m / (K * l_m) ** 2,
        I_m == np.pi * r_m ** 3 * t_m,
        L_n <= np.pi ** 2 * E * I_n / (K * l_n) ** 2,
        I_n == np.pi * r_n ** 3 * t_n,

        # Machining constraint (Mason p89). The second line uses r_m against
        # t_n in the source -- see the module docstring.
        2 * r_m / t_m <= 40,
        2 * r_m / t_n <= 40,

        # Retraction constraints on strut diameter
        2 * wtm + 2 * r_m <= hhold,
        2 * wtn + 2 * r_n <= 0.8 * m,

        # Weight accounting (Currey p264)
        W_mg >= n_mg * (W_ms + W_mw * (1 + faddm)),
        W_ng >= W_ns + W_nw * (1 + faddn),
        W_lg >= Clg * (W_mg + W_ng),
        W_lg * xcglg >= W_ng * x_n + W_mg * x_m,
        x_m >= xcglg,
    ]

    return lg, cons
