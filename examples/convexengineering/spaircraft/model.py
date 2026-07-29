"""SPaircraft: a signomial-programming transonic commercial aircraft.

Source model
------------
``aircraft.py`` in https://github.com/convexengineering/SPaircraft, with the
component models in ``wing.py``, ``fuselage.py``, ``horizontal_tail.py``,
``vertical_tail.py``, ``landing_gear.py`` and ``wingbox.py``, and the engine
from https://github.com/convexengineering/turbofan.

Paper
-----
    M. York, B. Öztürk, E. Burnell and W. Hoburg, "Efficient Aircraft
    Multidisciplinary Design Optimization and Sensitivity Analysis via
    Signomial Programming", AIAA Journal, DOI 10.2514/1.J057020

Configuration
-------------
Only ``optimalD8`` -- the paper's D8.2 -- is built here, because it is the
only configuration that converges and the only one the source's CI exercises
(see ``reference.py`` and DISCREPANCIES.md §14). Its geometry flags, from
``geometryFlags.py``, are: rear engines, boundary layer ingestion, pi-tail,
double-bubble fuselage, engine 3 (the TASOPT D8.2 turbofan).

Mission
-------
Three climb segments and two cruise, one mission, 3000 nm with 180
passengers. Climb is modelled with an excess-power formulation; cruise range
comes from the segment flight times. Segment weights step down by the fuel
burnt, which is what closes the sizing loop: fuel burn depends on weight,
weight depends on fuel carried.

Objective: minimize total fuel weight ``W_{f_{total}}``.

Verification
------------
``reference.json`` holds the gpkit solution at a *converged* tolerance --
not the shipped ``reltol=0.01``, which does not converge and does not
reproduce. See ``reference.py``.
"""
from __future__ import annotations

import numpy as np
from numpy import cos, pi, tan
from pyomo.environ import units

from edi import Formulation

from .flight_state import add_flight_state
from .fuselage import add_fuselage
from .horizontal_tail import add_horizontal_tail
from .landing_gear import add_landing_gear
from .vertical_tail import add_vertical_tail
from .wing import add_wing

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turbofan.model import add_engine  # noqa: E402

# Geometry, from subs/optimalD8.py.
SWEEP_W, SWEEP_VT, SWEEP_HT = 13.237, 25.0, 8.0
NCLIMB, NCRUISE = 3, 2


def build(Nclimb: int = NCLIMB, Ncruise: int = NCRUISE,
          pi_tail_supports: str = "pinned",
          seed: str | None = None) -> Formulation:
    """Build the D8.2. Returns an EDI ``Formulation``.

    ``seed="reference"`` initialises every variable from ``reference.json``
    instead of the hand-written guesses. That is a statement about the
    *solver*, not the model: the constraints are verified independently by
    ``crosscheck``, which shows the gpkit optimum satisfies all 3713 of them
    to 1e-7. Seeding only asks whether EDI's PCCP loop can hold and reproduce
    that point, which the naive all-guesses start cannot reach.
    """
    N = Nclimb + Ncruise
    f = Formulation()
    V = lambda n, g, u, d: f.Variable(name=n, guess=g, units=u, description=d)
    Vn = lambda n, g, u, d: f.Variable(name=n, guess=g, units=u,
                                       description=d, size=N)
    C = lambda n, v, u, d: f.Constant(name=n, value=v, units=u, description=d)

    st, cons = add_flight_state(f, N)
    wing, c = add_wing(f, N, st, sweep_deg=SWEEP_W); cons += c
    vt, c = add_vertical_tail(f, N, st, sweep_deg=SWEEP_VT); cons += c
    ht, c = add_horizontal_tail(f, N, st, sweep_deg=SWEEP_HT); cons += c
    lg, c = add_landing_gear(f); cons += c
    fu, c = add_fuselage(f); cons += c
    eng, c = add_engine(f, N, st, engine="D82_SPaircraft", BLI=True,
                        prefix="Eng_"); cons += c

    # ---- aircraft-level scalars -------------------------------------------
    W_total = V("W_total", 1.4e5, "lbf", "total aircraft weight")
    W_totalmax = V("W_total_max", 1.4e5, "lbf", "maximum total weight")
    W_dry = V("W_dry", 7.4e4, "lbf", "zero-fuel aircraft weight")
    W_ftotal = V("W_f_total", 2.1e4, "lbf", "total fuel weight")
    W_fprimary = V("W_f_primary", 1.75e4, "lbf", "fuel less reserves")
    W_fclimb = V("W_f_climb", 5e3, "lbf", "fuel burned in climb")
    W_fcruise = V("W_f_cruise", 1.25e4, "lbf", "fuel burned in cruise")
    Wmisc = V("W_misc", 1e4, "lbf", "sum of miscellaneous weights")
    Whpesys = V("W_hpesys", 1.4e3, "lbf", "power systems weight")
    xmisc = V("x_misc", 6.0, "m", "misc weight centroid")
    xhpesys = V("x_hpesys", 9.7, "m", "power systems x-location")
    Iz = V("I_z", 1e7, "kg*m^2", "aircraft z-axis moment of inertia")
    Izwing = V("I_z_wing", 5e6, "kg*m^2", "wing moment of inertia")
    Iztail = V("I_z_tail", 2e6, "kg*m^2", "tail moment of inertia")
    Izfuse = V("I_z_fuse", 3e6, "kg*m^2", "fuselage moment of inertia")

    # engine installation
    Snace = V("S_nacelle", 8.0, "m^2", "nacelle surface area")
    lnace = V("l_nacelle", 1.6, "m", "nacelle length")
    fSnace = V("f_S_nacelle", 0.06, "-", "non-dimensional nacelle area")
    Ainlet = V("A_inlet", 3.2, "m^2", "inlet area")
    Afancowl = V("A_fancowl", 1.6, "m^2", "fan cowling area")
    Aexh = V("A_exh", 3.2, "m^2", "exhaust area")
    Acorecowl = V("A_corecowl", 6.0, "m^2", "core cowling area")
    Wnace = V("W_nacelle", 2e3, "lbf", "nacelle weight")
    Wpylon = V("W_pylon", 6e2, "lbf", "engine pylon weight")
    Weadd = V("W_eadd", 8e2, "lbf", "additional engine system weight")
    Wengsys = V("W_engsys", 1e4, "lbf", "total engine system weight")
    xeng = V("x_eng", 32.0, "m", "engine x-location")
    y_eng = V("y_eng", 2.0, "m", "engine moment arm")

    # ---- aircraft-level constants ------------------------------------------
    g = C("g", 9.81, "m/s^2", "gravitational acceleration")
    numeng = C("n_eng", 2.0, "-", "number of engines")
    numVT = C("n_vt", 2.0, "-", "number of vertical tails")
    numaisle = C("n_aisle", 2.0, "-", "number of aisles")
    Vne = C("V_ne", 143.92, "m/s", "never-exceed speed")
    rhoTO = C("rho_TO_ac", 1.225, "kg/m^3", "air density at takeoff")
    ReserveFraction = C("f_fuel_res", 0.20, "-", "fuel reserve fraction")
    f_wingfuel = C("f_wingfuel", 1.0, "-", "fraction of fuel in wing tanks")
    FuelFrac = C("FuelFrac", 0.9, "-", "usable fraction of max fuel volume")
    SMmin = C("SM_min", 0.05, "-", "minimum static margin")
    dxCG = C("dx_CG", 6.0, "ft", "max CG travel range")
    Mmin = C("M_min", 0.72, "-", "minimum cruise Mach number")
    Ltow = C("f_L_total_wing", 1.195, "-", "total lift as a fraction of wing lift")
    fhpesys = C("f_hpesys", 0.01, "-", "power systems weight fraction")
    rSnace = C("r_S_nacelle", 6.0, "-", "nacelle and pylon wetted area factor")
    rvnace = C("r_v_nacelle", 0.925, "-", "incoming nacelle velocity ratio")
    fpylon = C("f_pylon", 0.05, "-", "pylon weight fraction")
    feadd = C("f_eadd", 0.1, "-", "additional engine weight fraction")
    Ceng = C("C_engsys", 1.0, "-", "engine system weight margin")
    Dreduct = C("D_reduct", 0.98416, "-", "BLI drag reduction factor")
    bmax = C("b_max", 140.0 * 0.3048, "m", "max wing span")
    CLwmax = C("C_L_w_max", 2.15 / cos(SWEEP_W * pi / 180) ** 2, "-",
               "max wing lift coefficient")
    Fsafetyfac = C("Fsafetyfac", 1.0, "-", "safety factor on initial climb thrust")
    MinCruiseAlt = C("MinCruiseAlt", 38478.0, "ft", "minimum cruise altitude")
    maxclimbtime = C("MaxClimbTime", 16.0, "min", "max time in climb")
    CDwm = C("C_D_wm", 0.5, "-", "windmill drag coefficient")
    Vland = C("V_land", 72.0, "m/s", "aircraft landing speed")
    rreq = C("rdot_req", 0.1475, "s^-2", "required yaw rate at landing")
    ReqRng = C("R_req", 3000.0, "nmi", "required cruise range")
    Vmn = C("V_mn", 133.94, "m/s", "manoeuvring speed")
    w_seat = C("w_seat", 0.5, "m", "seat width")
    w_aisle = C("w_aisle", 0.51, "m", "aisle width")
    w_sys = C("w_sys", 0.1, "m", "width between cabin and skin for systems")

    # ---- per-segment aircraft performance -----------------------------------
    D = Vn("D", 6e4, "N", "total aircraft drag")
    C_D = Vn("C_D", 0.03, "-", "total aircraft drag coefficient")
    LoD = Vn("LoD", 18.0, "-", "lift-to-drag ratio")
    W_avg = Vn("W_avg", 1.3e5, "lbf", "geometric mean segment weight")
    W_start = Vn("W_start", 1.4e5, "lbf", "segment start weight")
    W_end = Vn("W_end", 1.3e5, "lbf", "segment end weight")
    W_burn = Vn("W_burn", 4e3, "lbf", "segment fuel burn")
    WLoad = Vn("W_Load", 5e3, "N/m^2", "wing loading")
    tmin = Vn("tmin", 60.0, "min", "segment flight time in minutes")
    thr = Vn("thr", 1.0, "hr", "segment flight time in hours")
    xAC = Vn("x_AC", 19.0, "m", "aerodynamic centre of the aircraft")
    xCG = Vn("x_CG", 18.0, "m", "centre of gravity of the aircraft")
    xNP = Vn("x_NP", 19.0, "m", "neutral point of the aircraft")
    SM = Vn("SM", 0.1, "-", "stability margin")
    PCFuel = Vn("F_fuel", 0.5, "-", "fraction of fuel remaining")
    W_buoy = Vn("W_buoy", 1e3, "lbf", "buoyancy weight")
    rhocabin = Vn("rho_cabin", 0.88, "kg/m^3", "cabin air density")
    Ltotal = Vn("L_total", 6e5, "N", "total lift")
    Dfuse = Vn("D_fuse", 1.5e4, "N", "fuselage drag")
    Lfuse = Vn("L_fuse", 1e5, "N", "fuselage lift")
    Vnace = Vn("V_nacelle", 210.0, "m/s", "incoming nacelle flow velocity")
    V2 = Vn("V_2", 140.0, "m/s", "interior nacelle flow velocity")
    Vnacrat = Vn("V_nacelle_ratio", 1.2, "-", "nacelle velocity ratio")
    rvnsurf = Vn("r_v_nsurf", 1.0, "-", "intermediate nacelle drag parameter")
    Cfnace = Vn("C_f_nacelle", 0.003, "-", "nacelle skin friction coefficient")
    Renace = Vn("Re_nacelle", 2e7, "-", "nacelle Reynolds number")
    Cdnace = Vn("C_d_nacelle", 2e-4, "-", "nacelle drag coefficient")
    Dnace = Vn("D_nacelle", 1e3, "N", "drag on one nacelle")
    theta = Vn("theta", 0.02, "-", "aircraft climb angle")
    excessP = Vn("P_excess", 1e6, "W", "excess power during climb")
    RC = Vn("RC", 1500.0, "feet/min", "rate of climb")
    dhft = Vn("dhft", 12000.0, "feet", "altitude change per segment")
    Rseg = Vn("R_segment", 600.0, "nmi", "down range covered in each segment")

    Vstall = C("V_stall", 120.0, "knots", "aircraft stall speed")
    WLoadmax = C("W_Load_max", 6664.0, "N/m^2", "max wing loading")
    Pcabin = C("P_cabin", 75000.0, "Pa", "cabin air pressure")
    Tcabin = C("T_cabin", 297.0, "K", "cabin air temperature")
    minRC = C("RC_min", 500.0, "feet/min", "minimum rate of climb")
    CDfuse = C("C_D_fuse", 0.018081, "-", "fuselage drag coefficient")
    cmw = C("c_m_w_val", 1.9, "-", "wing pitching moment coefficient")

    f.Objective(W_ftotal)

    # ---- variable linking ---------------------------------------------------
    cons += [
        wing.c_root == fu.c_0,
        wing.x_w == fu.x_wing,
        fu.N_lift == wing.box.N_lift,
        Ltow * wing.L_max >= wing.box.N_lift * W_totalmax + ht.L_ht_max,
        wing.b <= bmax,

        # ---- weight build-up -------------------------------------------------
        fu.W_fuse + numeng * Wengsys + fu.W_tail + wing.W_wing + Wmisc <= W_dry,
        W_ftotal + W_dry + fu.W_payload <= W_total,
        W_ftotal >= W_fprimary + ReserveFraction * W_fprimary,
        W_fprimary >= W_fclimb + W_fcruise,
        W_totalmax >= W_total,
        wing.W_fuel_wing >= f_wingfuel * W_ftotal / FuelFrac,

        # ---- landing gear and power systems ----------------------------------
        Wmisc >= lg.W_lg + Whpesys,
        Whpesys == fhpesys * W_totalmax,
        lg.x_n <= fu.l_nose,
        lg.x_m >= fu.x_wing,
        lg.x_m <= wing.dx_AC_wing + fu.x_wing,
        xhpesys == 1.1 * fu.l_nose,
        xmisc * Wmisc >= xhpesys * Whpesys,
        lg.d_nacelle >= eng.d_f + 2 * lg.t_nacelle,
        lg.d_nacelle <= wing.b,
        # Hard landing, Torenbeek (10-26): 10 ft/s sink at max landing weight.
        lg.E_land >= W_totalmax / (2 * g) * lg.w_ult ** 2,
        lg.x_up == fu.x_shell2,
        lg.L_n == W_totalmax * lg.dx_m / lg.B,
        lg.L_m == W_totalmax * lg.dx_n / lg.B,
        lg.L_n_dyn >= 0.31 * ((lg.z_CG + lg.l_m) / lg.B) * W_totalmax,
        y_eng >= lg.y_m,

        # ---- fuselage --------------------------------------------------------
        # Tail cone sizing, driven by the VT root moment.
        3. * (numVT * vt.box.M_r) * vt.c_root_vt * (fu.p_lambda_vt - 1.)
            >= numVT * vt.L_vt_max * vt.b_vt * fu.p_lambda_vt,
        fu.V_cone * (1. + fu.lambda_cone) * (pi + 4. * fu.theta_db)
            >= (numVT * vt.box.M_r * vt.c_root_vt / fu.tau_cone
                * (pi + 2. * fu.theta_db) * (fu.l_cone / fu.R_fuse)),
        fu.W_tail >= numVT * vt.W_vt + ht.W_ht,
        2. * fu.w_fuse >= (fu.SPR * w_seat + numaisle * w_aisle
                              + 2. * w_sys + fu.t_db),
        fu.B_1v == fu.r_M_v * numVT * vt.L_vt_max / (fu.w_fuse * fu.sigma_M_v),

        # ---- horizontal tail --------------------------------------------------
        ht.m_ratio * (1 + 2 / wing.AR) == 1 + 2 / ht.AR_ht,     # [SP] SigEq
        ht.x_CG_ht <= fu.l_fuse,
        ht.V_ht == ht.S_ht * ht.l_ht / (wing.S * wing.mac),
        ht.L_ht_max >= 0.5 * rhoTO * Vne ** 2 * ht.S_ht * ht.C_L_ht_max,

        # ---- vertical tail ----------------------------------------------------
        vt.L_vt_max >= 0.5 * rhoTO * Vne ** 2 * vt.S_vt * vt.C_L_vt_max,
        vt.x_CG_vt <= fu.l_fuse,
        vt.V_vt == numVT * vt.S_vt * vt.l_vt / (wing.S * wing.b),
        # Yaw rate at flare
        numVT * .5 * vt.rho_TO * Vland ** 2 * vt.S_vt * vt.l_vt
            * vt.C_L_vt_yaw >= rreq * vt.I_z_max,
        # One-engine-out moment balance (TASOPT 2.0 p45)
        numVT * vt.L_vt_EO * vt.l_vt >= vt.T_e * y_eng + vt.D_wm * y_eng,
        vt.D_wm >= 0.5 * vt.rho_TO * vt.V_1 ** 2. * eng.A_2 * CDwm,

        # ---- moment of inertia -------------------------------------------------
        Iz >= Izwing + Iztail + Izfuse,
        vt.I_z_max >= Iz,

        # ---- engine installation ------------------------------------------------
        Snace == rSnace * np.pi * 0.25 * eng.d_f ** 2,
        lnace == 0.15 * eng.d_f * rSnace,
        fSnace == Snace * wing.S ** -1,
        Ainlet == 0.4 * Snace,
        Afancowl == 0.2 * Snace,
        Aexh == 0.4 * Snace,
        Acorecowl == 3. * np.pi * eng.d_LPC ** 2,
        Wnace >= ((2.5 + 0.238 * eng.d_f / units.inch) * Ainlet + 1.9 * Afancowl
                  + (2.5 + 0.0363 * eng.d_f / units.inch) * Aexh + 1.9 * Acorecowl
                  ) * units.lbf / units.ft ** 2,
        Weadd == feadd * eng.W_engine,
        Wpylon >= (Wnace + Weadd + eng.W_engine) * fpylon,
        Wengsys >= Ceng * (Wpylon + Wnace + Weadd + eng.W_engine),
    ]

    # ---- rear-engine + BLI configuration -------------------------------------
    cons += [
        # Engine-out moment arm for a rear-mounted, BLI engine.
        y_eng == 0.5 * fu.w_fuse,
        # Wing root moment, with wing weight and fuel load relief.
        wing.box.M_r * wing.c_root >= (
            (wing.L_max - wing.box.N_lift * (wing.W_wing + f_wingfuel * W_ftotal))
            * (wing.b ** 2 / (12 * wing.S) * (wing.c_root + 2 * wing.c_tip))),
        fu.A_1h_Land >= (fu.N_land * (fu.W_tail + numeng * Wengsys + fu.W_apu))
                           / (fu.h_fuse * fu.sigma_bend),
        fu.A_1h_MLF >= (fu.N_lift * (fu.W_tail + numeng * Wengsys + fu.W_apu)
                           + fu.r_M_h * ht.L_ht_max) / (fu.h_fuse * fu.sigma_M_h),
        Izwing >= ((wing.W_fuel_wing + wing.W_wing) / (wing.S * g)
                   * wing.c_root * wing.b ** 3. * (1. / 12. - (1. - wing.lambda_) / 16.)),
        Iztail >= ((fu.W_apu + vt.W_vt + numeng * Wengsys) * vt.l_vt ** 2. / g
                   + ht.W_ht * ht.l_ht ** 2. / g),
        # x_wing and l_vt stand in for CG-relative distances so I_z stays scalar.
        Izfuse >= ((fu.W_fuse + fu.W_payload_max) / fu.l_fuse
                   * (fu.x_wing ** 3. + vt.l_vt ** 3.) / (3. * g)),
        xeng <= fu.x_shell2 + 1.00 * fu.l_cone,
        xeng >= fu.x_shell2 + 0.75 * fu.l_cone,

        # ---- double-bubble floor loading -------------------------------------
        fu.S_floor == (5. / 16.) * fu.P_floor,
        fu.M_floor == 9. / 256. * fu.P_floor * fu.w_floor,
        fu.dR_fuse == fu.R_fuse * 0.43 / 1.75,
    ]

    # ---- pi-tail horizontal tail ---------------------------------------------
    hb = ht.box
    Mrout = V("M_r_out", 1e5, "N", "HT moment at the VT attachment")
    cons += [
        hb["b_ht_out"] == 0.5 * ht.b_ht - fu.w_fuse,               # [SP] SigEq
        Mrout * ht.c_attach >= (hb["L_ht_rect_out"] * (0.5 * hb["b_ht_out"])
                                   + hb["L_ht_tri_out"] * (1. / 3. * hb["b_ht_out"])),
        hb["L_shear"] >= hb["L_ht_rect_out"] + hb["L_ht_tri_out"],
        ht.c_tip_ht + (1. - ht.lambda_ht) * 2. * hb["b_ht_out"] / ht.b_ht
            * ht.c_root_ht == ht.c_attach,                         # [SP] SigEq
    ]

    if pi_tail_supports == "pinned":
        # SOURCE BEHAVIOUR. The verticals are treated as pin joints carrying
        # no moment, so the inboard span is simply supported and the
        # centreline moment is the applied moment MINUS the support reaction.
        # That subtraction is what makes M_r degenerate: the two terms can
        # very nearly cancel, the constraint stops binding, and M_r collapses
        # onto the 1e-30 box floor along with I_cap and t_cap. Reproduced
        # because the gpkit reference depends on it -- see DISCREPANCIES.md.
        cons += [
            ht.b_ht / 4. * hb["L_ht_rect"] + ht.b_ht / 3. * hb["L_ht_tri"]
                == hb["b_ht_out"] * ht.L_ht_max / 2.,                 # [SP] SigEq
            hb["M_r"] * ht.c_root_ht >= (hb["L_ht_rect"] * (ht.b_ht / 4.)
                                            + hb["L_ht_tri"] * (ht.b_ht / 6.)
                                            - fu.w_fuse * ht.L_ht_max / 2.),
            hb["pi_M_fac"] >= ((0.5 * (Mrout * ht.c_attach
                                       + hb["M_r"] * ht.c_root_ht)
                                * fu.w_fuse
                                / (0.5 * Mrout * ht.c_attach * hb["b_ht_out"])
                                + 1.0) * hb["b_ht_out"] / (0.5 * ht.b_ht)),
        ]
    else:
        # FIXED SUPPORTS. A pi-tail horizontal joins two verticals rigidly, so
        # the inboard span is a beam BUILT IN at both ends, not pin-jointed.
        # For span L under load W the standard results are
        #
        #     hogging at each support   W*L/12
        #     sagging at midspan        W*L/24
        #
        # against W*L/8 at midspan and zero at the supports if pinned. Two
        # consequences, and they are the point of the change:
        #
        # 1. The sizing station moves to the ATTACHMENT, where the fixed-end
        #    moment adds to the overhang moment. A root moment never sizes a
        #    pi-tail horizontal -- there is no root, only two supports.
        # 2. Every moment is now a SUM of positive terms. Nothing can cancel,
        #    so M_r cannot collapse, and the constraint is posynomial rather
        #    than signomial -- strictly easier for the solver as well as more
        #    physical.
        #
        # The verticals sit at +/- w_fuse, so the built-in span is 2*w_fuse.
        Lin = V("L_ht_in", 1e5, "N", "HT load inboard of the VT attachments")
        Mfe = V("M_fe", 1e4, "N", "fixed-end moment per attachment chord")
        cons += [
            # Load inboard of the attachments. The section is untapered over
            # this span, so its share of the load is its share of the span.
            Lin >= ht.L_ht_max * (2. * fu.w_fuse) / ht.b_ht,
            # Fixed-end (hogging) moment at each support, W*L/12.
            Mfe * ht.c_attach >= Lin * (2. * fu.w_fuse) / 12.,
            # The attachment carries the overhang AND the fixed-end moment;
            # both hog the beam over the support, so they add.
            Mrout * ht.c_attach >= (hb["L_ht_rect_out"] * (0.5 * hb["b_ht_out"])
                                       + hb["L_ht_tri_out"] * (1. / 3. * hb["b_ht_out"])
                                       + Mfe * ht.c_attach),
            # Sagging at the centreline, W*L/24 -- half the fixed-end value
            # and a third of what a pinned span would carry.
            hb["M_r"] * ht.c_root_ht >= Lin * (2. * fu.w_fuse) / 24.,
            # Load split, unchanged in form but now with no cancellation.
            ht.b_ht / 4. * hb["L_ht_rect"] + ht.b_ht / 3. * hb["L_ht_tri"]
                == hb["b_ht_out"] * ht.L_ht_max / 2.,                 # [SP] SigEq
            # The cap must carry the larger of the two stations.
            hb["pi_M_fac"] >= 1.0,
            hb["pi_M_fac"] >= Mrout * ht.c_attach
                              / (hb["M_r"] * ht.c_root_ht),
        ]

    # ---- per-segment performance -----------------------------------------------
    cons += [
        rhocabin == Pcabin / (st.R * Tcabin),
        st.V >= Vstall,
        W_avg >= (W_start * W_end) ** .5 + W_buoy,
        tmin == thr,
        W_buoy >= rhocabin * g * fu.V_cabin,
        # Fuselage lift, as a fraction of wing lift.
        Lfuse == (Ltow - 1.) * wing.L_w,                    # [SP] SigEq
        Ltotal == Ltow * wing.L_w,
        Ltotal >= W_avg + ht.L_ht,

        # ---- drag ------------------------------------------------------
        Dfuse == (0.5 * st.rho * st.V ** 2 * CDfuse
                     * fu.l_fuse * fu.R_fuse
                     * (st.M ** 2 / fu.M_fuseD ** 2)),
        D >= Dreduct * (wing.D_wing + Dfuse + numVT * vt.D_vt
                           + ht.D_ht + numeng * Dnace),
        C_D == D / (.5 * st.rho * st.V ** 2 * wing.S),
        LoD == W_avg / D,

        # ---- wing loading and lift losses -------------------------------
        WLoad <= WLoadmax,
        WLoad == (.5 * wing.C_L * st.rho * st.V ** 2),
        wing.p_o >= wing.L_w * wing.c_root / wing.S,
        wing.eta_o == fu.w_fuse / (wing.b / 2),

        # ---- stability ---------------------------------------------------
        xAC <= fu.x_wing + 0.25 * wing.dx_AC_wing + xNP,
        wing.c_m_w == cmw,
        # Neutral point approximation, from Unified's aircraft design rules.
        (xNP / wing.mac / ht.V_ht * (wing.AR + 2.)
         * (1. + 2. / ht.AR_ht)
         == (1. + 2. / wing.AR) * (wing.AR - 2.)),             # [SP] SigEq
        xCG + vt.dx_trail_vt <= fu.l_fuse,
        vt.x_CG_vt >= xCG + 0.5 * (vt.dx_lead_vt + vt.dx_trail_vt),
        ht.x_CG_ht >= xCG + 0.5 * (ht.dx_lead_ht + ht.dx_trail_ht),
        ht.C_L_alpha_ht + (2 * wing.C_L_alpha_w / (pi * wing.AR))
            * ht.eta_ht * ht.C_L_alpha_ht_0
            <= ht.C_L_alpha_ht_0 * ht.eta_ht,
        ht.C_L_ht >= 0.01,
        SM <= (xAC - xCG) / wing.mac,
        SM >= SMmin,
        xAC / wing.mac <= (xCG / wing.mac + cmw / wing.C_L
                                 + ht.V_ht * (ht.C_L_ht / wing.C_L)),

        # ---- nacelle drag --------------------------------------------------
        Renace == st.rho * st.V * lnace / st.mu,
        Cfnace == 0.94 * 4. * 0.0743 / (Renace ** 0.2),
        Vnace == rvnace * st.V,
        Vnacrat >= 2. * Vnace / st.V - V2 / st.V,
        rvnsurf ** 3. >= 0.25 * (Vnacrat + rvnace) * (Vnacrat ** 2. + rvnace ** 2.),
        Cdnace == fSnace * Cfnace[0] * rvnsurf ** 3.,
        Dnace == Cdnace * 0.5 * st.rho * st.V ** 2. * wing.S,
        V2 == eng.M_2 * st.a,

        # ---- pi-tail trailing edge ------------------------------------------

        # ---- climb -----------------------------------------------------------
        excessP + st.V * D <= st.V * numeng * eng.F,
        RC == excessP / W_avg,
        theta * st.V == RC,
        dhft == tmin * RC,
        Rseg == thr * st.V,
        numeng * eng.F >= D + W_avg * theta,

        # ---- CG ----------------------------------------------------------------
        xCG * W_avg >= (
            xmisc * Wmisc + lg.x_CG_lg * lg.W_lg
            + 0.5 * (fu.W_fuse + fu.W_payload) * fu.l_fuse
            + ht.W_ht * ht.x_CG_ht + vt.W_vt * vt.x_CG_vt
            + numeng * Wengsys * xeng
            + wing.W_wing * (fu.x_wing + wing.dx_AC_wing)
            + (PCFuel + ReserveFraction) * W_fprimary
            * (fu.x_wing + wing.dx_AC_wing * PCFuel)),

        # ---- fuel burn -----------------------------------------------------------
        W_burn == numeng * eng.TSFC * thr * eng.F,
        W_start >= W_end + W_burn,
        PCFuel <= 1.0000001,

        # ---- engine operating point -----------------------------------------------
        eng.M_2 == st.M,
        eng.M_25 == 0.6,
        eng.hold_2 == 1. + .5 * (1.398 - 1.) * 0.6 ** 2,
        eng.hold_25 == 1. + .5 * (1.354 - 1.) * 0.6 ** 2,
        eng.c1 == 1. + 0.5 * .401 * st.M ** 2.,          # [SP] SigEq
    ]


    # Aircraft-level geometry and stability limits. These mention no segment,
    # so the source's per-segment loop stated each of them N times over; they
    # are stated once here. Identical rows constrain nothing extra -- presolve
    # would drop the copies anyway -- so this removes 3*(N-1) rows and leaves
    # the feasible set alone.
    cons += [
        ht.AR_ht >= 4.,
        ht.dx_trail_ht <= (vt.dx_lead_vt + vt.b_vt / tan(SWEEP_VT * pi / 180)
                           + fu.w_fuse / tan(SWEEP_HT * pi / 180)
                           + ht.c_root_ht),
        SMmin + dxCG / wing.mac + cmw / CLwmax
            <= ht.V_ht * ht.m_ratio + ht.V_ht * ht.C_L_ht_max / CLwmax,
    ]

    # ---- mission stitching ------------------------------------------------------
    cons += [
        W_start[0] == W_total,
        st.hft[0] == dhft[0],
        st.hft[Nclimb - 1] >= MinCruiseAlt,
        RC[0] >= 2500. * units.ft / units.min,
        theta[Nclimb - 1] >= 0.015,
        vt.T_e == Fsafetyfac * eng.F[0],
        W_dry + fu.W_payload + ReserveFraction * W_fprimary <= W_end[N - 1],
        W_fclimb >= f.sum(W_burn[:Nclimb]),
        W_fcruise >= f.sum(W_burn[Nclimb:]),
        f.sum(Rseg) >= ReqRng,
        f.sum(thr[:Nclimb]) <= maxclimbtime,
        lg.dx_n + lg.x_n >= xCG[Nclimb],
        lg.dx_m + xCG[Nclimb] >= lg.x_m,
        lg.x_m >= lg.tan_phi * (lg.z_CG + lg.l_m) + xCG[Nclimb],
        # Mission caps bypass ratio rather than fixing it; subs/optimalD8.py
        # supplies neither alpha_max nor alpha_OD, so both are free.
        eng.alpha_max <= 100.0,
    ]
    # Segment-to-segment stitching: each of these relates a segment to the one
    # before it, which is what the offset slices say.
    cons += [
        W_start[1:] == W_end[:-1],
        st.hft[1:] == st.hft[:-1] + dhft[1:],                      # [SP] SigEq
        dhft[1:Nclimb] == dhft[:Nclimb - 1],
        RC[1:Nclimb] >= minRC,
        st.M[Nclimb:] >= Mmin,
    ]
    # Keep Mach inside the transonic band. Not in the source, but needed here:
    # the VT drag fit carries M**1022.7 and M**-114.577, and in the log-space
    # GP those become exponents of ~1023*log(M). A line search that steps M
    # even slightly above 1 overflows exp() and IPOPT reports "Error in an
    # AMPL evaluation". gpkit avoids this because MOSEK's exponential-cone
    # form never forms exp() explicitly. 0.1-0.95 is far outside any
    # physically meaningful excursion for this aircraft.
    cons += [st.M <= 0.95, st.M >= 0.1]
    cons += [Rseg[Nclimb:N - 1] == Rseg[Nclimb + 1:N]]
    # Fuel still to burn after each segment. The slice bound moves with the
    # segment, so this one keeps its loop.
    for i in range(N):
        rest = f.sum(W_burn[i + 1:])
        cons += [PCFuel[i] >= (rest + 0.0000001 * W_fprimary) / W_fprimary]
    # Wing max angle of attack differs between climb and cruise.
    cons += [wing.alpha_w[:Nclimb] <= 0.18,
             wing.alpha_w[Nclimb:] <= 0.10]

    # ---- substitutions -------------------------------------------------------
    # subs/optimalD8.py *fixes* these; leaving any of them free lets the
    # optimizer choose it. n_pass is the one that matters most: unpinned, the
    # payload collapses to 15 lbf and the whole aircraft shrinks with it,
    # landing at 409 lbf of fuel instead of 20860.
    for handle, name, value, unit in [
        (fu, "n_pass", 180.0, None),
        (fu, "W_cargo", 0.1, units.N),
        (fu, "l_nose", 29.0, units.ft),
        (fu, "h_floor", 5.12, units.inch),
        (fu, "w_db", 0.93, units.m),
        (fu, "lambda_cone", 0.3, None),
        (vt, "A_vt", 2.2, None),
        (vt, "V_1", 70.0, units.m / units.s),
        (vt, "c_l_vt_EO", 0.5, None),
        (vt, "e_vt", 0.8, None),
        (vt, "lambda_vt", 0.3, None),
        # 1.225, not the 1.23 that subs/optimalD8.py specifies: vertical_tail.py
        # and wing.py declare rho with 1.225 baked in, and the solved reference
        # carries 1.225 for \rho_{TO}, \rho_0 and \rho_{T/O} alike. The
        # substitution does not take. Matching the reference, not the subs dict.
        (vt, "rho_TO", 1.225, units.kg / units.m ** 3),
        (ht, "lambda_ht", 0.3, None),
        (ht, "C_L_ht_fCG", 0.85, None),
        (lg, "z_CG", 2.0, units.m),
        (lg, "z_wing", 0.5, units.m),
        (lg, "h_hold", 1.0, units.m),
        (lg, "t_nacelle", 0.15, units.m),
    ]:
        cons.append(handle[name] == (value * unit if unit is not None else value))

    cons += _bound_constraints(f)
    f.ConstraintList(cons)
    _bound_variables(f)
    if seed == "reference":
        _seed_from_reference(f)
    return f


def _seed_from_reference(f):
    """Initialise every variable from the recorded gpkit optimum."""
    from pyomo.core.base.var import IndexedVar
    from .crosscheck import reference_point
    ref = reference_point(Path(__file__).with_name("reference.json"))
    n = 0
    for v in f.get_variables():
        items = ([(f"{v.name}[{i}]", v[i]) for i in v.index_set()]
                 if isinstance(v, IndexedVar) else [(v.name, v)])
        for nm, vd in items:
            base = nm.split("[")[0]
            idx = int(nm.split("[")[1].rstrip("]")) if "[" in nm else None
            val = ref.get(base)
            if val is None:
                continue
            if isinstance(val, list):
                val = val[idx] if idx is not None and idx < len(val) else val[0]
            vd.set_value(float(val), skip_validation=True)
            n += 1
    return n


ABS_LO, ABS_HI = 1e-30, 1e30


def _bound_constraints(f):
    """The same box as ``_bound_variables``, but expressed as constraints.

    Both forms are needed and they are not redundant. Pyomo variable bounds
    reach the raw-NLP path only: EDI's log-space GP backend extracts the
    model into coefficient/exponent rows and builds a *fresh* Pyomo model
    over its own variable vector, so declared bounds never reach the PCCP
    subproblems. Only constraints survive that translation.
    """
    import pyomo.environ as pyo
    from pyomo.core.base.var import IndexedVar
    out = []
    for v in f.get_variables():
        items = (v[i] for i in v.index_set()) if isinstance(v, IndexedVar) else (v,)
        for vd in items:
            u = pyo.units.get_units(vd)
            out += [vd <= ABS_HI * u, vd >= ABS_LO * u]
    return out


def _bound_variables(f):
    """Bound every free variable strictly positive, as gpkit's Bounded does.

    ``SPaircraft.py`` never solves this model bare -- it wraps it in
    ``Bounded(m)``, which brackets every variable in 1e-30..1e30 so that a
    diverging sequential-GP run hits a bound instead of running to an
    infinitely low cost. The same is needed here: without it the PCCP loop
    runs 301 subproblems and then dies with a non-finite objective gradient.

    The lower bound matters for a second reason beyond divergence. EDI
    declares variables over ``Reals``, so nothing stops a solver iterate from
    going negative -- and this model is full of fractional and negative
    powers (the wing drag polar alone has ``C_L**-1.44114``). One negative
    iterate and IPOPT reports "Invalid number in NLP function or derivative".
    Every quantity in a geometric program is strictly positive by
    construction, so a positive lower bound is not a modelling choice here,
    it is the domain.

    The box has to be the *absolute* 1e-30..1e30 that gpkit uses, not a
    relative one around each guess. A relative box looks better conditioned,
    but it excludes the reference solution: in the converged gpkit answer the
    horizontal tail's box collapses, with ``I_{cap}`` sitting at 1.0e-30 --
    exactly on gpkit's artificial floor -- ``M_r`` at 1.3e-20 and ``W_{cap}``
    at 0.14 N. A box even six decades around a physically-scaled guess cuts
    that point off and makes the model infeasible. See DISCREPANCIES.md §19:
    the degeneracy is a property of the reference, not of this rebuild.
    """
    from pyomo.core.base.var import IndexedVar
    n = 0
    for v in f.get_variables():
        items = (v[i] for i in v.index_set()) if isinstance(v, IndexedVar) else (v,)
        for vd in items:
            vd.setlb(ABS_LO)
            vd.setub(ABS_HI)
            n += 1
    return n


# gpkit reference name -> (rebuilt name, scale). Scale converts to the unit
# the reference reports in.
CHECKS = [
    ("fuel_lbf", "W_f_total", 1.0),
    ("takeoff_weight_lbf", "W_total", 1.0),
    ("dry_weight_lbf", "W_dry", 1.0),
    ("span_ft", "Wing_b", 3.28084),
]


# EDI's PCCP loop defaults to 50 iterations, which is not enough here: the
# model has 1174 variables and the sequential-GP sequence is still moving at
# 50. It settles by ~200, and 500 gives the same answer to seven figures.
MAX_ITER = 200


def verify(seed=None, rtol=0.02, max_iter=MAX_ITER):
    """Solve and diff the headline quantities against the gpkit reference."""
    from harness import solve_edi, feasibility, load_reference, solution_dict

    fm = build(seed=seed)
    solve_edi(fm, solver="ipopt-convex", max_iter=max_iter)
    sol = solution_dict(fm)
    ref = load_reference(Path(__file__).with_name("reference.json"))
    chk = ref["optimalD8"]["checked"]
    paper = ref["optimalD8"].get("paper_sp", {})

    rows = []
    for key, name, scale in CHECKS:
        got = sol[name] * scale
        exp = chk[key]
        rows.append((key, got, exp, abs(got - exp) / abs(exp),
                     paper.get(key)))
    nv, worst, where = feasibility(fm)
    return rows, (nv, worst, where)


if __name__ == "__main__":
    import sys
    seed = "reference" if "--seed" in sys.argv else None
    rows, (nv, worst, where) = verify(seed=seed)
    print(f"\nSPaircraft D8.2 ({'seeded' if seed else 'cold start'})")
    print(f"{'quantity':22} {'rebuilt':>12} {'gpkit':>12} {'rel':>9} "
          f"{'paper':>12}")
    worst_rel = 0.0
    for key, got, exp, rel, pap in rows:
        worst_rel = max(worst_rel, rel)
        p = "--" if pap is None else f"{pap:12.6g}"
        print(f"{key:22} {got:12.6g} {exp:12.6g} {rel:9.2e} {p:>12}")
    print(f"worst relative difference vs gpkit: {worst_rel:.2e}")
    print(f"feasibility: {nv} violated, worst rel {worst:.2e}"
          + (f" at {where}" if where else ""))
