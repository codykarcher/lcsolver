"""Double-bubble fuselage: cross-section, pressure shell, floor, cone, bending.

Source model
------------
``fuselage.py`` in https://github.com/convexengineering/SPaircraft. The
bending-material model is ported from TASOPT.

The double bubble
-----------------
A conventional tube is the special case ``w_{db} = 0``, ``\\theta_{db} = 0``,
``\\Delta R_{fuse} = 0`` — which is how the 737 and 777 substitution sets
configure it, except that they use 1e-4 rather than 0, because a true zero
leaves several of these monomials undefined in log space. The D8 sets
``w_{db} = 0.93 m``, joining two circular lobes along a vertical web of
half-height ``h_{db}``, and it is that web which carries the extra pressure
load.

The joining angle uses a first-order Taylor expansion,
``\\theta_{db} == w_{db}/R_{fuse}`` (the source's own comment reads "first
order Taylor works..."), and several downstream constraints carry the matching
cosine expansion ``(1 - \\theta_{db}^2/2)``.

Bending material
----------------
The horizontal bending model is evaluated for *two* load cases — emergency
landing (``N_{land} = 6``) and maximum wing aero load (``N_{lift}``) — and
whichever is more constraining wins, because ``V_{hbend_f}`` and
``V_{hbend_b}`` are each bounded below by both. Each case carries its own
zero-bending station ``x_{hbend}``, located by a signomial equality. The
vertical bending model has one case, driven by the vertical tail load through
``B_{1v}``, which is defined at aircraft level.

This is the most signomial-dense component in the aircraft: eight
``SignomialEquality`` constraints and a dozen more inequalities with
subtractions on the bounding side.

Sized elsewhere
---------------
``S_{floor}``, ``M_{floor}``, ``V_{cone}``, ``A_{1h}`` (both cases),
``B_{1v}``, ``x_{wing}``, ``c_0``, ``h_{floor}``, ``W_{tail}`` and
``N_{lift}`` close at aircraft level.
"""
from __future__ import annotations

from numpy import pi


def add_fuselage(f, *, prefix="Fuse_", l_tank=None, SPR=8.0,
                 cabin_aux_m=1.61):
    """Add the fuselage. Returns ``(vars, constraints)``."""
    fuse = f.group("fuse", prefix=prefix)
    V, C = fuse.Variable, fuse.Constant
    out = {}

    def var(n, g, u, d):
        out[n] = V(n, g, u, d)
        return out[n]

    def con(n, v, u, d):
        out[n] = C(n, v, u, d)
        return out[n]

    # ---- passengers and payload -------------------------------------------
    npass = var("n_pass", 180.0, "-", "number of passengers")
    nseat = var("n_seat", 180.0, "-", "number of seats")
    nrows = var("n_rows", 30.0, "-", "number of rows")
    Wpay = var("W_payload", 38700.0, "lbf", "payload weight")
    Wpaymax = var("W_payload_max", 38700.0, "lbf", "maximum payload weight")
    Wpass = var("W_pass", 32400.0, "lbf", "passenger weight")
    Wlugg = var("W_lugg", 6000.0, "lbf", "passenger luggage weight")
    Wcargo = var("W_cargo", 0.0225, "lbf", "cargo weight")

    # ---- cross-section ------------------------------------------------------
    Adb = var("A_db", 0.01, "m^2", "web cross-sectional area")
    Afloor = var("A_floor", 0.05, "m^2", "floor beam cross-sectional area")
    Afuse = var("A_fuse", 12.0, "m^2", "fuselage cross-sectional area")
    Askin = var("A_skin", 0.02, "m^2", "skin cross-sectional area")
    hdb = var("h_db", 1.5, "m", "web half-height")
    hfloor = var("h_floor", 0.1, "m", "floor beam height")
    hfuse = var("h_fuse", 1.8, "m", "fuselage height")
    dRfuse = var("dR_fuse", 0.3, "m", "fuselage extension height")
    Rfuse = var("R_fuse", 1.8, "m", "fuselage radius")
    tdb = var("t_db", 0.002, "m", "web thickness")
    thetadb = var("theta_db", 0.4, "-", "double-bubble joining angle")
    tshell = var("t_shell", 0.002, "m", "shell thickness")
    tskin = var("t_skin", 0.001, "m", "skin thickness")
    wdb = var("w_db", 0.5, "m", "double-bubble added half-width")
    wfloor = var("w_floor", 2.0, "m", "floor half-width")
    wfuse = var("w_fuse", 2.0, "m", "fuselage half-width")

    # ---- lengths, cone, areas ------------------------------------------------
    lamcone = var("lambda_cone", 0.3, "-", "tailcone radius taper ratio")
    lcone = var("l_cone", 6.0, "m", "cone length")
    c0 = var("c_0", 5.5, "m", "root chord of the wing")
    lfuse = var("l_fuse", 38.0, "m", "fuselage length")
    lnose = var("l_nose", 5.0, "m", "nose length")
    lshell = var("l_shell", 25.0, "m", "shell length")
    lfloor = var("l_floor", 28.0, "m", "floor length")
    Sbulk = var("S_bulk", 25.0, "m^2", "bulkhead surface area")
    Snose = var("S_nose", 50.0, "m^2", "nose surface area")

    # ---- volumes -------------------------------------------------------------
    Vbulk = var("V_bulk", 0.03, "m^3", "bulkhead skin volume")
    Vcabin = var("V_cabin", 350.0, "m^3", "cabin volume")
    Vcone = var("V_cone", 0.05, "m^3", "cone skin volume")
    Vcyl = var("V_cyl", 0.4, "m^3", "cylinder skin volume")
    Vdb = var("V_db", 0.2, "m^3", "web volume")
    Vfloor = var("V_floor", 0.2, "m^3", "floor volume")
    Vnose = var("V_nose", 0.05, "m^3", "nose skin volume")

    # ---- stresses and floor loads --------------------------------------------
    sigth = var("sigma_theta", 1e8, "N/m^2", "skin hoop stress")
    sigx = var("sigma_x", 5e7, "N/m^2", "axial stress in skin")
    Mfloor = var("M_floor", 5e5, "N*m", "max bending moment in floor beams")
    Pfloor = var("P_floor", 1e6, "N", "distributed floor load")
    Sfloor = var("S_floor", 5e5, "N", "maximum shear in floor beams")
    taucone = var("tau_cone", 1e8, "N/m^2", "shear stress in cone")

    # ---- bending model --------------------------------------------------------
    A0h = var("A_0h", 0.02, "m^2", "horizontal bending area constant A0h")
    A1hLand = var("A_1h_Land", 0.001, "m", "A1h, landing case")
    A1hMLF = var("A_1h_MLF", 0.001, "m", "A1h, max aero load case")
    A2hLand = var("A_2h_Land", 1e-4, "-", "A2h, landing case")
    A2hMLF = var("A_2h_MLF", 1e-4, "-", "A2h, max aero load case")
    AhbendbLand = var("A_hbendb_Land", 0.005, "m^2", "bending area, rear box, landing")
    AhbendbMLF = var("A_hbendb_MLF", 0.005, "m^2", "bending area, rear box, max aero")
    AhbendfLand = var("A_hbendf_Land", 0.002, "m^2", "bending area, front box, landing")
    AhbendfMLF = var("A_hbendf_MLF", 0.003, "m^2", "bending area, front box, max aero")
    Avbendb = var("A_vbend_b", 0.005, "m^2", "vertical bending area at rear box")
    B0v = var("B_0v", 0.02, "m^2", "vertical bending area constant B0")
    B1v = var("B_1v", 0.001, "m", "vertical bending area constant B1")
    Ihshell = var("I_h_shell", 0.05, "m^4", "shell horizontal bending inertia")
    Ivshell = var("I_v_shell", 0.05, "m^4", "shell vertical bending inertia")
    sigMh = var("sigma_M_h", 1e8, "N/m^2", "horizontal bending material stress")
    sigMv = var("sigma_M_v", 1e8, "N/m^2", "vertical bending material stress")
    Vhbend = var("V_hbend", 0.05, "m^3", "horizontal bending material volume")
    Vhbendb = var("V_hbend_b", 0.03, "m^3", "horizontal bending volume, back")
    Vhbendc = var("V_hbend_c", 0.02, "m^3", "horizontal bending volume, centre")
    Vhbendf = var("V_hbend_f", 0.003, "m^3", "horizontal bending volume, front")
    Vvbend = var("V_vbend", 0.002, "m^3", "vertical bending material volume")
    Vvbendb = var("V_vbend_b", 0.001, "m^3", "vertical bending volume, back")
    Vvbendc = var("V_vbend_c", 0.001, "m^3", "vertical bending volume, centre")
    Whbend = var("W_hbend", 300.0, "lbf", "horizontal bending material weight")
    Wvbend = var("W_vbend", 50.0, "lbf", "vertical bending material weight")
    xhbendLand = var("x_hbend_Land", 60.0, "ft", "horizontal zero-bending station, landing")
    xhbendMLF = var("x_hbend_MLF", 60.0, "ft", "horizontal zero-bending station, max aero")
    xvbend = var("x_vbend", 60.0, "ft", "vertical zero-bending station")

    # ---- weights ---------------------------------------------------------------
    Wapu = var("W_apu", 1400.0, "lbf", "APU weight")
    Wcone = var("W_cone", 1000.0, "lbf", "cone weight")
    Wdb = var("W_db", 500.0, "lbf", "web weight")
    Wfloor = var("W_floor", 3000.0, "lbf", "floor weight")
    Wfuse = var("W_fuse", 25000.0, "lbf", "fuselage weight")
    Winsul = var("W_insul", 1000.0, "lbf", "insulation material weight")
    Wpadd = var("W_padd", 13500.0, "lbf", "misc weight (galley, toilets, doors)")
    Wseat = var("W_seat", 2500.0, "lbf", "seating weight")
    Wshell = var("W_shell", 8000.0, "lbf", "shell weight")
    Wskin = var("W_skin", 5000.0, "lbf", "skin weight")
    Wtail = var("W_tail", 5000.0, "lbf", "total tail weight")
    Wwindow = var("W_window", 2000.0, "lbf", "window weight")

    # ---- x-locations -------------------------------------------------------------
    xshell1 = var("x_shell1", 5.0, "m", "start of cylinder section")
    xshell2 = var("x_shell2", 30.0, "m", "end of cylinder section")
    xtail = var("x_tail", 33.0, "m", "x-location of tail")
    xwing = var("x_wing", 18.0, "m", "x-location of wing quarter chord")
    xf = var("x_f", 19.5, "m", "x-location of front of wingbox")
    xb = var("x_b", 16.5, "m", "x-location of back of wingbox")
    Nlift = var("N_lift", 3.0, "-", "wing maximum load factor")

    # ---- constants ------------------------------------------------------------
    g = con("g", 9.81, "m/s^2", "acceleration due to gravity")
    Nland = con("N_land", 6.0, "-", "emergency landing load factor")
    rE = con("r_E", 1.0, "-", "ratio of stringer to skin moduli")
    fapu = con("f_apu", 0.035, "-", "APU weight as a fraction of payload")
    fpadd = con("f_padd", 0.35, "-", "misc weight as a fraction of payload")
    Wavgpasstot = con("W_avg_pass_total", 215.0, "lbf",
                      "average passenger weight including payload")
    w = con("r_w_c", 0.5, "-", "wingbox width-to-chord ratio")
    Cfuse = con("C_fuse", 1.0, "-", "fuselage weight margin and sensitivity")
    rMh = con("r_M_h", 0.4, "-", "horizontal inertial relief factor")
    rMv = con("r_M_v", 0.7, "-", "vertical inertial relief factor")
    plamv = con("p_lambda_vt", 1.6, "-", "1 + 2 * VT taper ratio")
    # These carry per-configuration values; the defaults are the D8's.
    dPover = con("dP_over", 8.382, "psi", "cabin overpressure")
    SPR = con("SPR", SPR, "-", "number of seats per row")
    pitch = con("p_s", 81.0, "cm", "seat pitch")
    l_aux = con("l_cabin_aux", cabin_aux_m, "m",
                "cabin length for galleys, lavatories, doors and exit rows, "
                "as a fraction of seat-row length")
    sigskin = con("sigma_skin", 15000.0 / 0.000145, "Pa",
                  "max allowable skin stress")
    sigbend = con("sigma_bend", 30000.0 / 0.000145, "Pa",
                  "bending material stress")
    sigfloor = con("sigma_floor", 30000.0 / 0.000145, "Pa",
                   "max allowable floor stress")
    taufloor = con("tau_floor", 30000.0 / 0.000145, "Pa",
                   "max allowable shear web stress")
    rhoskin = con("rho_skin", 2700.0, "kg/m^3", "skin density")
    rhobend = con("rho_bend", 2700.0, "kg/m^3", "stringer density")
    rhofloor = con("rho_floor", 2700.0, "kg/m^3", "floor material density")
    rhocone = con("rho_cone", 2700.0, "kg/m^3", "cone material density")
    Wppfloor = con("Wpp_floor", 60.0, "N/m^2", "floor weight per unit area")
    Wppinsul = con("Wpp_insul", 22.0, "N/m^2", "insulation weight per area")
    Wpseat = con("Wp_seat", 1.0, "N", "weight per seat")
    Wpwindow = con("Wp_window", 435.0, "N/m", "window weight per unit length")
    ffadd = con("f_fadd", 0.2, "-", "fractional added weight, reinforcements")
    fframe = con("f_frame", 0.25, "-", "fractional frame weight")
    fstring = con("f_string", 0.35, "-", "fractional stringer weight")
    fseat = con("f_seat", 0.1, "-", "fractional seat weight")
    flugg1 = con("f_lugg_1", 0.4, "-", "proportion of passengers, one suitcase")
    flugg2 = con("f_lugg_2", 0.1, "-", "proportion of passengers, two suitcases")
    Wavgpass = con("W_avg_pass", 180.0, "lbf", "average passenger weight")
    Wcarryon = con("W_carry_on", 15.0, "lbf", "average carry-on weight")
    Wchecked = con("W_checked", 40.0, "lbf", "average checked bag weight")
    Wfix = con("W_fix", 3000.0, "lbf", "fixed weight (pilots, seats, navcom)")
    MfuseD = con("M_fuseD", 0.72, "-", "fuselage drag reference Mach")

    cons = [
        # ---- passengers -------------------------------------------------------
        Wlugg >= flugg2 * npass * 2 * Wchecked + flugg1 * npass * Wchecked + Wcarryon,
        Wpass >= npass * Wavgpass,
        Wpay >= Wpass + Wlugg + Wcargo,
        Wpay >= npass * Wavgpasstot,
        Wpaymax >= Wpay,
        # EQUALITY. As `nseat >= npass` the seat count was a design variable
        # bounded only from below: the optimiser could add empty rows and
        # stretch the cabin. Nothing paid for that until the CG envelope was
        # put on TASOPT's cglpay footing, at which point a longer cabin moved
        # the forward CG limit aft (x_CG_fwd = x_shell1 + l_shell*r_F) and so
        # bought tail-trim relief -- and the 180-seat 737 promptly grew to
        # 35.24 rows, a 4.24 m fuselage stretch carrying no passengers.
        #
        # A latent fault, not one the CG change introduced: under the old
        # envelope n_rows sat at exactly 30.0000 only because nothing gave it
        # a reason to move. The seat count is an INPUT in TASOPT, and a cabin
        # is sized for the seats it has.
        nseat == npass,
        nrows == nseat / SPR,
        # The cylindrical shell holds the cabin AND, on a hydrogen aircraft,
        # the LH2 tank behind it. Everything downstream of l_shell -- skin,
        # insulation, floor, cone station, both bending distributions -- then
        # follows automatically, which is why this one row is the whole
        # structural hook for putting a tank in the fuselage.
        # Cabin = seat rows PLUS the galleys, lavatories, doors and exit rows
        # that a cabin actually contains. Seat pitch times rows alone gives
        # 24.30 m where TASOPT's 737 pressure shell is 25.91 m.
        #
        # I declined to add this earlier on the grounds that total fuselage
        # length already matched TASOPT to 1.6%. That was wrong: the total
        # matched while BOTH parts were off, an overlong tailcone (8.43 m
        # against 6.10) compensating for a short cabin. It showed up when the
        # horizontal tail turned out to be capped by having to fit on the
        # fuselage -- x_CG + dx_trail_ht <= l_fuse binding with zero slack,
        # with the tailcone taper already at its limit. The tail was
        # length-starved because the cabin was short.
        #
        # 6.6% of seat-row length, which is what reconciles the two.
        # ADDITIVE, not 6.6% of seat-row length. See SizeClass.cabin_aux_m:
        # galleys, lavatories and doors are fixed objects and do not shrink
        # with row count, and as a fraction they left a Citation 0.21 m for
        # all of them. The 737 is unchanged -- 6.6% of its 24.30 m of seat
        # rows IS 1.61 m, which is the calibration point.
        lshell == nrows * pitch + l_aux
                  + (l_tank if l_tank is not None else 0.0 * Rfuse),

        # ---- fuselage joint angle ---------------------------------------------
        thetadb == wdb / Rfuse,
        hdb >= Rfuse * (1.0 - .5 * thetadb ** 2),                    # [SP]

        # ---- cross-section -----------------------------------------------------
        Adb >= (2 * hdb + dRfuse) * tdb,
        Afuse >= ((pi + 2 * thetadb + 2 * thetadb * (1 - thetadb ** 2 / 2))
                  * Rfuse ** 2 + 2 * dRfuse * Rfuse),                # [SP]
        Askin >= (2 * pi + 4 * thetadb) * Rfuse * tskin + 2 * dRfuse * tskin,
        wfloor == wfuse,
        wfuse <= Rfuse + wdb,
        hfuse == Rfuse + 0.5 * dRfuse,                               # [SP] SigEq
        tshell <= tskin * (1. + rE * fstring * rhoskin / rhobend),   # [SP]

        # ---- surface areas ------------------------------------------------------
        Snose ** (8. / 5.) >= Sbulk ** (8. / 5.) * (1. / 3. + 2. / 3. * (lnose / Rfuse) ** (8. / 5.)),
        Sbulk >= (2 * pi + 4 * thetadb) * Rfuse ** 2,

        # ---- lengths -------------------------------------------------------------
        lfuse == lnose + lshell + lcone,                             # [SP] SigEq
        lcone == Rfuse / lamcone,
        # The tailcone taper is a free variable and nothing in this model
        # resists a stubby one: a short cone is simply lighter. It floated to
        # 0.418, giving 4.44 m of cone, where TASOPT's 737 deck runs its cone
        # from xblend2 = 97 ft to xconend = 117 ft -- 6.10 m on a 1.88 m radius,
        # i.e. a taper of 0.308.
        #
        # What actually sets aft-body length on a real aeroplane is the upsweep
        # angle needed to rotate to takeoff attitude without a tail strike,
        # which depends on gear position and rotation angle. This model has the
        # rotation angle (tan_theta_max in the landing gear) but never connects
        # it to the aft body. Until it does, the taper is bounded at TASOPT's
        # value rather than left to shrink for free.
        lamcone <= 0.32,
        lamcone >= 0.22,
        xshell1 == lnose,
        xshell2 >= lnose + lshell,

        # ---- pressure shell loading ----------------------------------------------
        tskin == dPover * Rfuse / sigskin,
        tdb == 2 * dPover * wdb / sigskin,
        sigx == dPover * Rfuse / (2 * tshell),
        sigth == dPover * Rfuse / tskin,

        # ---- floor loading --------------------------------------------------------
        lfloor >= lshell + 2 * Rfuse,
        Pfloor >= Nland * (Wpaymax + Wseat),
        Afloor >= 2. * Mfloor / (sigfloor * hfloor) + 1.5 * Sfloor / taufloor,
        Vfloor == 2 * wfloor * Afloor,
        Wfloor >= rhofloor * g * Vfloor + 2 * wfloor * lfloor * Wppfloor,

        # ---- tail cone -------------------------------------------------------------
        taucone == sigskin,
        Wcone >= rhocone * g * Vcone * (1 + fstring + fframe),
        xtail >= lnose + lshell + .5 * lcone,

        # ---- shell bending inertias -------------------------------------------------
        Ihshell <= (((pi + 4 * thetadb) * Rfuse ** 2
                     + 8. * (1 - thetadb ** 2 / 2) * (dRfuse / 2.) * Rfuse
                     + (2 * pi + 4 * thetadb) * (dRfuse / 2) ** 2) * Rfuse * tshell
                    + 2. / 3 * (hdb + dRfuse / 2.) ** 3 * tdb),      # [SP]
        Ivshell <= ((pi * Rfuse ** 2 + 8 * wdb * Rfuse
                     + (2 * pi + 4 * thetadb) * wdb ** 2) * Rfuse * tshell),  # [SP]

        # ---- horizontal bending material --------------------------------------------
        xhbendLand >= xwing, xhbendLand <= lfuse,
        xhbendMLF >= xwing, xhbendMLF <= lfuse,
        A0h == A2hLand * (xshell2 - xhbendLand) ** 2 + A1hLand * (xtail - xhbendLand),
        A0h == A2hMLF * (xshell2 - xhbendMLF) ** 2 + A1hMLF * (xtail - xhbendMLF),
        A2hLand >= (Nland * (Wpaymax + Wpadd + Wshell + Wwindow + Winsul + Wfloor + Wseat)
                    / (2 * lshell * hfuse * sigMh)),
        A2hMLF >= (Nlift * (Wpaymax + Wpadd + Wshell + Wwindow + Winsul + Wfloor + Wseat)
                   / (2 * lshell * hfuse * sigMh)),
        A0h == Ihshell / (rE * hfuse ** 2),                          # [SP]

        AhbendfLand >= A2hLand * (xshell2 - xf) ** 2 + A1hLand * (xtail - xf) - A0h,
        AhbendfMLF >= A2hMLF * (xshell2 - xf) ** 2 + A1hMLF * (xtail - xf) - A0h,
        AhbendbLand >= A2hLand * (xshell2 - xb) ** 2 + A1hLand * (xtail - xb) - A0h,
        AhbendbMLF >= A2hMLF * (xshell2 - xb) ** 2 + A1hMLF * (xtail - xb) - A0h,

        Vhbendf >= (A2hLand / 3 * ((xshell2 - xf) ** 3 - (xshell2 - xhbendLand) ** 3)
                    + A1hLand / 2 * ((xtail - xf) ** 2 - (xtail - xhbendLand) ** 2)
                    - A0h * (xhbendLand - xf)),
        Vhbendf >= (A2hMLF / 3 * ((xshell2 - xf) ** 3 - (xshell2 - xhbendMLF) ** 3)
                    + A1hMLF / 2 * ((xtail - xf) ** 2 - (xtail - xhbendMLF) ** 2)
                    - A0h * (xhbendMLF - xf)),
        Vhbendb >= (A2hLand / 3 * ((xshell2 - xb) ** 3 - (xshell2 - xhbendLand) ** 3)
                    + A1hLand / 2 * ((xtail - xb) ** 2 - (xtail - xhbendLand) ** 2)
                    - A0h * (xhbendLand - xb)),
        Vhbendb >= (A2hMLF / 3 * ((xshell2 - xb) ** 3 - (xshell2 - xhbendMLF) ** 3)
                    + A1hMLF / 2 * ((xtail - xb) ** 2 - (xtail - xhbendMLF) ** 2)
                    - A0h * (xhbendMLF - xb)),
        Vhbendc >= .5 * (AhbendfLand + AhbendbLand) * c0 * w,
        Vhbendc >= .5 * (AhbendfMLF + AhbendbMLF) * c0 * w,
        Vhbend >= Vhbendc + Vhbendf + Vhbendb,
        Whbend >= g * rhobend * Vhbend,

        # ---- vertical bending material -----------------------------------------------
        xvbend >= xwing, xvbend <= lfuse,
        B0v == B1v * (xtail - xvbend),                               # [SP] SigEq
        B0v == Ivshell / (rE * wfuse ** 2),
        Avbendb >= B1v * (xtail - xb) - B0v,
        Vvbendb >= 0.5 * B1v * ((xtail - xb) ** 2 - (xtail - xvbend) ** 2) - B0v * (xvbend - xb),
        Vvbendc >= 0.5 * Avbendb * c0 * w,
        Vvbend >= Vvbendb + Vvbendc,
        Wvbend >= rhobend * g * Vvbend,

        # ---- wingbox stations -----------------------------------------------------------
        xf == xwing + .5 * c0 * w,                                   # [SP] SigEq
        xb == xwing - .5 * c0 * w,                                   # [SP] SigEq
        sigMh <= sigbend - rE * dPover / 2 * Rfuse / tshell,
        sigMv <= sigbend - rE * dPover / 2 * Rfuse / tshell,

        # ---- volumes ---------------------------------------------------------------------
        Vcyl == Askin * lshell,
        Vnose == Snose * tskin,
        Vbulk == Sbulk * tskin,
        Vdb == Adb * lshell,
        Vcabin >= Afuse * (lshell + 0.67 * lnose + 0.67 * Rfuse),

        # ---- weights ------------------------------------------------------------------------
        Wapu == Wpaymax * fapu,
        Wdb == rhoskin * g * Vdb,
        Winsul >= Wppinsul * ((1.1 * pi + 2 * thetadb) * Rfuse * lshell
                              + 0.55 * (Snose + Sbulk)),
        # Windows are a cabin item: the tank bay has none, so this must
        # scale with the seated length rather than the whole shell.
        Wwindow >= Wpwindow * nrows * pitch,
        Wpadd == Wpaymax * fpadd,
        # Two independent estimates of seat weight; both must hold.
        Wseat >= Wpseat * nseat,
        Wseat >= fseat * Wpaymax,
        Wskin >= rhoskin * g * (Vcyl + Vnose + Vbulk),
        Wshell >= Wskin * (1 + fstring + ffadd + fframe) + Wdb,
        Wfuse >= Cfuse * (Wshell + Wfloor + Winsul + Wapu + Wfix + Wwindow
                          + Wpadd + Wseat + Whbend + Wvbend + Wcone),
    ]

    return fuse, cons
