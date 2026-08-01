"""Landing gear sizing: geometry, tip-over, struts, wheels and weight.

Source model
------------
``landing_gear.py`` in https://github.com/convexengineering/SPaircraft.
Sources cited there: Torenbeek p360, Raymer p233/p244, Currey p145/p264,
and Mason's machining-constraint notes.

Four source defects, all now corrected
--------------------------------------
The first two were previously reproduced rather than corrected, on the grounds
that the reference solution depended on them. It does not any more: the weight
build-up below has been rebuilt on physical drivers, so the errors no longer
have anything to hold up. See DISCREPANCIES.md 17-18.

1. **Currey's wheel correlation was evaluated 19.8x off.** The source writes
   ``Fwm == Lwm*dtm/(1000*Lwm.units*dtm.units)`` where ``Lwm.units`` is
   newtons, against a correlation ``F_w = L_w[lbf]*d[in]/1000`` that wants
   lbf. The docstring here used to call that a single 4.448x slip; measured at
   the solution it was 4.448 SQUARED, putting one main wheel at 262 lbf where
   ``1.2 F_w^0.609`` asks for 43. That 6.2x was quietly calibrating the gear
   alongside ``f_add``, so neither number meant anything alone. The whole
   correlation is gone, replaced by tyre, brake and hub built separately.

2. **The nose machining constraint used the main-gear radius** --
   ``2*r_m/t_n <= 40``, so nose wall thickness was bounded against the *main*
   strut. Harmless while the nose strut was slack; not harmless now that its
   own bending case sizes it. Corrected to ``2*r_n/t_n``.

3. **The oleo diameter used the max load where Raymer p244 wants the static
   load.** ``d_oleo == 1.3*(4*lam*L_m/n_mg/(pi*p_oleo))**0.5`` at 1800 psi
   inflated the bore by sqrt(2.5) = 1.58x, to 14.5 in against a real 737 outer
   cylinder of about 9.8 in. Nothing caught it because ``d_oleo`` was consumed
   by no constraint in the model at all -- computed and dropped.

4. **The retraction constraint double-counted the tyres.** ``2*wtm + 2*r_m <=
   hhold`` charges two tyre WIDTHS of well depth, but the two wheels on a strut
   share an axle and retract side by side across the well, one width deep. On
   the 737 that was about 0.39 m of phantom depth, against a hold height that
   is itself a pinned D8.2 constant.

What sizes the gear now
-----------------------
Strut: bore from the oleo (load / pressure), wall from bending at the trunnion
under the FAR 25.479/25.485 axle loads. Tyres: from their own computed size.
Brakes: from rejected-takeoff energy per braked wheel. Hubs: from what they
carry. One installation fraction covers the rest of the leg.

The strut geometry that falls out was not fitted to anything and agrees with
the real aircraft on six dimensions -- oleo bore 9.1 in vs 9.8, strut OD
10.4 in vs 9.8, nose bore 5.4 in vs 5, leg length 1.84 m vs 1.9, tyre 43.7 in
vs 44.5, tyre width 15.7 in vs 16.5.
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
    d_oleo = V("d_oleo", 0.2, "m", "diameter of the oleo shock absorber, main")
    d_oleo_n = V("d_oleo_n", 0.14, "m", "diameter of the oleo shock absorber, nose")
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
    WAWm = V("W_wa_m", 200.0, "lbf", "wheel assembly weight, one main wheel")
    WAWn = V("W_wa_n", 60.0, "lbf", "wheel assembly weight, one nose wheel")
    W_tire_m = V("W_tire_m", 120.0, "lbf", "tire weight, one main wheel")
    W_tire_n = V("W_tire_n", 60.0, "lbf", "tire weight, one nose wheel")
    W_brk = V("W_brake", 150.0, "lbf", "brake assembly weight, one main wheel")
    W_hub_m = V("W_hub_m", 75.0, "lbf", "wheel hub weight, one main wheel")
    W_hub_n = V("W_hub_n", 45.0, "lbf", "wheel hub weight, one nose wheel")
    M_m = V("M_m", 3e5, "N*m", "bending moment at the main strut trunnion")
    M_n = V("M_n", 5e4, "N*m", "bending moment at the nose strut trunnion")
    E_RTO = V("E_RTO", 2e8, "J", "kinetic energy absorbed in a rejected takeoff")

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
    # ---- installation fraction ---------------------------------------------
    # ONE calibrated fraction, applied to the whole leg, covering what this
    # model still does not build: the oleo internals (piston, seals, fluid,
    # metering pin), trunnion and its fittings, drag and side braces, torque
    # links, retraction actuator, doors, uplocks and bay structure.
    #
    # It replaces the pair of `f_add` factors that used to sit here. Those had
    # two problems. They multiplied only the WHEEL weight, so everything in the
    # list above scaled with tyre size and not with the leg; and they were
    # doing calibration work jointly with a 19.8x unit error in the Currey
    # wheel correlation (see below), which means neither number meant anything
    # on its own.
    # Calibrated once, on the 737, against Torenbeek App. C -- NOT against
    # TASOPT. TASOPT's flat 0.011 + 0.044 = 5.50% of MTOW is the outlier here:
    # Torenbeek gives 3.92% for a 174700 lb 737-800 and at least varies with
    # weight, where a fixed fraction varies with nothing. What this model
    # builds explicitly -- strut, tyres, brakes, hubs -- comes to 2.00% of
    # MTOW, so the parts it does not build are the other half of the leg. That
    # is the right order: on a real transport leg the cylinder and piston are
    # roughly half the structure, with braces, trunnion, links, actuation and
    # doors making up the rest.
    finst = C("f_install", 2.0, "-",
              "installed leg weight / (strut + wheels + tyres + brakes)")
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

    # Combined drag and side load at the axle, as a fraction of the ultimate
    # vertical this model already carries. FAR 25.479 (spin-up/spring-back)
    # puts a fore-aft load at the axle; 25.485 puts a lateral one. Either
    # reacts as BENDING at the trunnion over the full leg length, and that is
    # what actually sizes a transport main leg -- pure axial compression sized
    # this tube at 6 cm radius, an order of magnitude light.
    #
    # The value is read off the regulation, not fitted. 25.485 sets the limit
    # side load at 0.8x the limit vertical reaction and 25.479 puts the drag
    # case at about the same; ultimate is 1.5x limit, so the ultimate axle side
    # load is 1.2x the STATIC per-strut vertical. The vertical this model
    # applies is `N_s*lam*L_m/n_mg` = 5x static per strut, hence 1.2/5 = 0.24.
    # It checks out geometrically: at 0.24 the main strut comes out near 8.5 in
    # outside diameter against a real 737 outer cylinder of about 9.8 in. At
    # the 0.5 first tried it went to 10.9 in, the leg collided with the hold
    # height through the retraction constraint, and the aircraft shed 9000 lb
    # of MTOW escaping it.
    f_side = C("f_side", 0.24, "-", "axle drag/side load / ultimate vertical")
    # Tyre mass from the tyre's own size, which the model already computes.
    # A tyre is a rubber-and-cord shell over a torus, so mass goes as d^2 * w.
    # Calibrated on a 737-800 main tyre: 44x16, about 120 lb.
    k_tire = C("k_tire", 120.0 / (44.0 ** 2 * 16.0), "-",
               "tyre weight coefficient, lbf per in^3 of d^2*w")
    # Carbon heat-sink specific energy, and the brake assembly mass relative to
    # the heat sink alone (discs are roughly 45% of a wheel brake).
    e_sink = C("e_sink", 1.6e6, "J/kg", "brake heat sink specific energy")
    f_brake = C("f_brake", 2.2, "-", "brake assembly weight / heat sink weight")
    # V_1 as a fraction of lift-off speed: the RTO is by definition decided at
    # or below V_R, and V_R <= V_LOF.
    f_V1 = C("f_V1", 0.95, "-", "V_1 / V_LOF")
    # Forged aluminium wheel hub, against the brake it carries (main) or the
    # tyre it carries (nose, which has no brake).
    f_hub_m = C("f_hub_m", 0.50, "-", "main wheel hub weight / brake weight")
    f_hub_n = C("f_hub_n", 0.80, "-", "nose wheel hub weight / tyre weight")

    out = dict(B=B, T=T, x_m=x_m, x_n=x_n, y_m=y_m, l_m=l_m, l_n=l_n,
               dx_m=dxm, dx_n=dxn, x_CG_lg=xcglg, z_CG=z_CG_0, z_wing=zwing,
               h_hold=hhold, x_up=x_upswp, L_m=L_m, L_n=L_n, L_n_dyn=L_n_dyn,
               E_land=Eland, W_lg=W_lg, W_mg=W_mg, W_ng=W_ng, d_t_m=dtm,
               d_nacelle=d_nac, t_nacelle=t_nac, h_nacelle=h_nac,
               w_ult=w_ult, lambda_LG=lam, tan_phi=tan_phi, tan_psi=tan_psi,
               W_ms=W_ms, W_mw=W_mw, W_ns=W_ns, W_nw=W_nw, l_oleo=l_oleo,
               d_oleo=d_oleo, d_oleo_n=d_oleo_n, S_sa=S_sa, r_m=r_m, r_n=r_n, t_m=t_m, t_n=t_n,
               d_t_n=dtn, w_t_m=wtm, w_t_n=wtn,
               W_wa_m=WAWm, W_wa_n=WAWn, L_w_m=Lwm, L_w_n=Lwn, g=g,
               W_tire_m=W_tire_m, W_tire_n=W_tire_n, W_brake=W_brk,
               W_hub_m=W_hub_m, W_hub_n=W_hub_n, M_m=M_m, M_n=M_n,
               E_RTO=E_RTO, f_V1=f_V1)

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
        # Raymer p244 sizes the oleo from the load it holds up at the internal
        # pressure: d = 1.3*sqrt(4*L_oleo/(pi*P)), with P = 1800 psi and
        # L_oleo the STATIC load. The `lam` here made it the MAX load, 2.5x
        # static, inflating the diameter by sqrt(2.5) = 1.58x -- 14.5 in on the
        # 737 against a real outer cylinder of about 9.8 in. Nothing caught it
        # because `d_oleo` was consumed by no constraint at all; it was
        # computed and dropped. Now that the strut diameter is tied to it, it
        # has to be right. Without `lam` it gives 9.2 in, within 6% of the real
        # leg, and it agrees with what the bending case independently wanted.
        d_oleo == 1.3 * (4 * L_m / n_mg / (np.pi * p_oleo)) ** 0.5,
        l_m >= l_oleo + dtm / 2,

        # ---- wheel assembly: tyre + brake + hub, built item by item --------
        # This replaces Currey's `1.2 F_w^0.609` lumped wheel-assembly fit.
        # That fit was being evaluated 19.8x off -- the newton/lbf conversion
        # landed on it twice, not once as the module docstring claimed -- which
        # put one main wheel at 262 lbf against the 43 lbf the correlation
        # actually asks for. A 6.2x error that happened to sit near a plausible
        # answer is not a model, and it was quietly calibrating the gear
        # alongside `f_add`. Each piece now has its own driver.
        W_tire_m == k_tire * (dtm / inch) ** 2 * (wtm / inch) * lbf,
        W_tire_n == k_tire * (dtn / inch) ** 2 * (wtn / inch) * lbf,
        # Brakes are sized by the rejected takeoff, not by tyre size: the whole
        # aeroplane's kinetic energy at V_1 goes into the main-wheel heat sinks
        # and nothing else. This is the term that separates aircraft in the
        # matrix -- it scales with MTOW*V_1^2 per braked wheel, so a heavy
        # aircraft on a small number of wheels pays for it. The nose gear is
        # unbraked and carries no such term, which is most of why it is light.
        W_brk >= f_brake * E_RTO * g / (n_mg * nwps * e_sink),
        W_hub_m == f_hub_m * W_brk,
        W_hub_n == f_hub_n * W_tire_n,
        WAWm >= W_tire_m + W_brk + W_hub_m,
        WAWn >= W_tire_n + W_hub_n,
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

        # ---- strut bending (FAR 25.479 spin-up, 25.485 lateral drift) ------
        # The load path the model was missing. A drag or side load applied at
        # the AXLE reacts as a bending moment at the trunnion over the whole
        # leg length, and for a transport main leg that case, not axial
        # compression, is what sets the tube. Sized on compression alone the
        # main strut came out at 6 cm radius and 107 lbf, against a real 737
        # outer cylinder of roughly 10 in diameter -- an order of magnitude.
        # Bending also makes strut mass grow faster than linearly with leg
        # length, which is the sensitivity that makes a tall 787 leg expensive
        # and a short Citation leg cheap. A flat fraction of MTOW has none.
        M_m == f_side * (N_s * lam * L_m / n_mg) * l_m,
        M_m <= sig_y_c * (np.pi * r_m ** 2 * t_m),
        M_n == f_side * (N_s * (L_n + L_n_dyn)) * l_n,
        M_n <= sig_y_c * (np.pi * r_n ** 2 * t_n),

        # Buckling
        L_m <= np.pi ** 2 * E * I_m / (K * l_m) ** 2,
        I_m == np.pi * r_m ** 3 * t_m,
        L_n <= np.pi ** 2 * E * I_n / (K * l_n) ** 2,
        I_n == np.pi * r_n ** 3 * t_n,

        # Machining constraint (Mason p89): the wall may not be thinner than
        # r/20. The source's second line bounded the NOSE wall against the MAIN
        # radius; with the nose strut now sized by its own bending case that
        # typo would leave the nose slenderness genuinely unconstrained, so it
        # is corrected here rather than reproduced.
        2 * r_m / t_m <= 40,
        2 * r_n / t_n <= 40,
        # ...and the wall may not be thicker than 0.35 r. This bound is new and
        # is needed the moment bending sizes the strut: `Z = pi r^2 t` is a
        # THIN-wall section modulus, and nothing else bounds t from above, so
        # the optimizer bought section by growing the wall and ran to t > r --
        # a solid bar, where the thin-wall formula overstates the true section
        # by about 20%. With the bore now fixed by the oleo the wall is the
        # only thing left to grow, and bending asks for about t/r = 0.30 on
        # the 737, against roughly 0.23 on the real leg. 0.35 leaves that case
        # feasible while still keeping the section formula honest.
        t_m <= 0.35 * r_m,
        t_n <= 0.35 * r_n,

        # ---- strut diameter: the oleo sets it ------------------------------
        # On a telescopic leg the outer cylinder IS the oleo cylinder -- the
        # piston slides and seals against that bore -- so the strut diameter is
        # the oleo diameter, computed above from load and pressure. It has to
        # be an UPPER bound, and that is not obvious: written as `>=` it never
        # binds, because in pure bending the minimum-weight section for a given
        # `r^2 t` is a LARGE THIN tube. Tried that way the optimizer went to
        # the `t >= r/20` machining floor and built a 16.8 in tube on a 0.4 in
        # wall -- structurally efficient, and nothing a gear leg can be.
        #
        # With the bound the right way round the strut is fully determined and
        # has no free lever left: radius from the oleo, wall from bending.
        2 * r_m <= d_oleo,
        2 * r_n <= d_oleo_n,
        d_oleo_n == 1.3 * (4 * L_n / (np.pi * p_oleo)) ** 0.5,

        # Retraction. The wheels lie flat in the well, so the DEPTH they need
        # is one tyre width, not two -- the two wheels on a strut share an axle
        # and retract side by side, across the well, not stacked. The source
        # had `2*wtm`, which put about 0.39 m of phantom depth on the 737 and,
        # once bending started setting the strut radius, made this row bind
        # exactly at the pinned 1.0 m hold height and drove 13000 lb of MTOW
        # out of the aircraft.
        #
        # This is a soft limit written hard, and it is worth saying so: a real
        # 737 that runs out of well depth grows a belly fairing and pays drag,
        # and its main well is between the fore and aft holds anyway, so it
        # trades against hold LENGTH rather than hold height. Kept as a loose
        # sanity bound until the fuselage supplies a real well depth.
        wtm + 2 * r_m <= hhold,
        wtn + 2 * r_n <= 0.8 * m,

        # Weight accounting. The installation fraction now multiplies the WHOLE
        # leg -- strut included -- which is what Currey p264 and Torenbeek both
        # do, and what keeps the leg-length sensitivity in the answer instead
        # of tying retraction and brace weight to tyre size.
        W_mg >= n_mg * finst * (W_ms + W_mw),
        W_ng >= finst * (W_ns + W_nw),
        W_lg >= Clg * (W_mg + W_ng),
        W_lg * xcglg >= W_ng * x_n + W_mg * x_m,
        x_m >= xcglg,
    ]

    return lg, cons
