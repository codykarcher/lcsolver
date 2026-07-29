"""Replicating TASOPT.jl's LH2-burning turbofan aircraft as an SP.

The purpose
-----------
The fuel-cell model's comparison against TASOPT.jl's `cryo_input` aircraft
left a 63,000 lb MTOW gap, most of it airframe scope rather than physics.
This variant asks the sharp question: with the SAME propulsion architecture
(H2-burning turbofan) and the calibrations READ FROM THEIR SOLVED DESIGN --
never tuned to their MTOW -- how close does the SP land, and what exactly
carries the residual?

Every calibration and its provenance:

* ``k_fuse = 260 N/m^2``  -- validated: their Wfuse over their hull area is
  259. The earlier fuselage gap was geometry (no nose/tailcone, small tank),
  not the constant.
* hull: R = 2.54 m, nose+tailcone = 48.06 - 25.15 - 9.51 = 13.4 m, their
  cabin 25.15 m; l_fuse = 38.55 m + tank length.
* ``f_nonstruct = 1.64`` -- their buildup's own secondary/box ratio
  (flaps+slats+ribs+ailerons+spoilers+wiring over caps+webs).
* ``tau_max = 0.126``    -- TASOPT's airfoil thickness, not our 0.15.
* tails = 0.112 x wing   -- their (Whtail+Wvtail)/Wwing.
* ``W_add = 0.065 W_MTO`` -- decomposes exactly as TASOPT's default
  fractions: 0.044 main gear + 0.011 nose gear + 0.010 hpesys.
* engines: ``W_eng = 0.2963 F_TO``, ``F_TO = 0.294 W_MTO`` -- their sized
  engine weight over their takeoff thrust, and their thrust-to-weight.
* ``TSFC = 0.23527 / h`` at cruise -- their engine's own solved value
  (hydrogen: ~1/2.8 of kerosene, as the LHV ratio demands).
* reserves 20% (their ``freserve``), not the FC model's 10%.
* cruise at their point: V = 237.5 m/s, rho = 0.374 (their CL of 0.57 at
  their wing loading), sweep effects absent from both wingbox and drag.

The tank model is untouched -- it is the verified part.

Known differences left in, deliberately: the Hoburg wing box is not
TASOPT's beam model; the mission is cruise-only against their full climb/
descent profile (their own numbers say that costs ~4% on fuel); and there
is no trim/CG chain.
"""
from __future__ import annotations

import math

from edi import Formulation

from .cryo_tank import add_cryo_tank
from .wing_h2 import add_wing_h2

N_CRUISE = 4
PI = 3.141592653589793


def build(N: int = N_CRUISE, *, mode: str = "free") -> Formulation:
    """``mode="free"``: wing geometry -- S, AR, CL, tau AND sweep -- is
    optimised, with York's transonic airfoil fit supplying the physics that
    makes each trade real (wave drag prices CL, thickness and sweep; the
    structural span factor 1/cos^2(Lambda) prices sweep the other way;
    Nita-Scholz prices AR through the Oswald factor). Sweep is carried as
    ``cos(Lambda)`` itself, which keeps every appearance monomial -- the
    trig-of-a-variable obstruction never arises.

    ``mode="point"``: the original replication -- AR, CL and sweep pinned to
    TASOPT's values -- which validates the weight accounting at their design
    point."""
    f = Formulation()
    Vb = lambda n, g, u, d, bd: f.Variable(name=n, guess=g, units=u,
                                           description=d, bounds=bd)
    Vnb = lambda n, g, u, d, bd: f.Variable(name=n, guess=g, units=u,
                                            description=d, size=N, bounds=bd)
    C = lambda n, v, u, d: f.Constant(name=n, value=v, units=u, description=d)

    # Their planform: taper 0.25 (p = 1.5, q = 1.25), AR pinned to their
    # 10.1 below -- replication means their design point, not a free
    # optimum. The Hoburg box carries no sweep, so the ultimate load gets
    # the structural-span factor 1/cos^2(26 deg); TASOPT's own cap sizing
    # goes as 1/cos^4, so this is a conservative UNDER-correction and the
    # wing is expected to land ~16% light -- the honest Hoburg-vs-beam gap.
    point = (mode == "point")
    wing, cons = add_wing_h2(f, f_nonstruct=1.64,
                             tau_max=0.126 if point else 0.15,
                             p_taper=1.5, q_taper=1.25)
    # Their tank pressure policy (pressure_venting = 2 atm) and their
    # structural heat-leak factor (heat_leak_factor = 1.3), both from the
    # cryo TOML rather than assumed.
    tank, c = add_cryo_tank(f, pvent=2.0265e5, qfac=1.3); cons += c

    W_MTO = Vb("W_MTO", 7.4e5, "N", "maximum take-off weight", (1e5, 3e6))
    W_dry = Vb("W_dry", 6.5e5, "N", "zero-fuel weight", (1e5, 3e6))
    W_fuse = Vb("W_fuse", 2.0e5, "N", "fuselage weight", (1e4, 1e6))
    W_tail = Vb("W_tail", 1.4e4, "N", "empennage weight", (5e2, 2e5))
    W_eng = Vb("W_eng", 6.5e4, "N", "engines incl. nacelles and pylons",
               (1e3, 5e5))
    W_add = Vb("W_add", 4.9e4, "N", "landing gear + power systems", (1e3, 5e5))
    F_TO = Vb("F_TO", 2.2e5, "N", "takeoff thrust, both engines", (1e4, 1e6))
    l_fuse = Vb("l_fuse", 48.0, "m", "fuselage length", (10.0, 80.0))
    S_wet = Vb("S_wet", 1030.0, "m^2", "total wetted area", (100.0, 4000.0))

    W = Vnb("W", 7.4e5, "N", "weight at segment start", (1e5, 3e6))
    D = Vnb("D", 5.0e4, "N", "drag", (2e3, 5e5))
    # In "free" mode CL is priced by the transonic fit, not capped.
    C_L = Vnb("C_L", 0.57, "-", "lift coefficient",
              (0.10, 0.57 if point else 0.85))
    cosL = Vb("cosL", 0.9, "-", "cosine of quarter-chord sweep",
              (0.72, 0.999))
    CDp = Vnb("CDp", 0.008, "-", "wing profile+wave drag coeff", (1e-4, 0.1))
    CDi = Vnb("CDi", 0.012, "-", "induced drag coeff", (1e-4, 0.1))
    e_osw_v = Vb("e_osw", 0.975, "-", "Oswald (span) efficiency", (0.5, 1.0))
    mac = Vb("mac", 3.9, "m", "mean aerodynamic chord", (0.5, 12.0))
    Re = Vb("Re", 2.4e7, "-", "chord Reynolds number", (1e6, 2e8))
    C_D = Vnb("C_D", 0.0388, "-", "drag coefficient", (0.008, 0.30))
    t_seg = Vnb("t_seg", 5.85e3, "s", "segment duration", (100.0, 5e4))
    mdot_f = Vnb("mdot_f", 0.33, "kg/s", "fuel flow", (0.005, 5.0))

    g = C("g", 9.81, "m/s^2", "gravitational acceleration")
    rho = C("rho", 0.374, "kg/m^3", "cruise density, their operating point")
    V_inf = C("V_inf", 237.5, "m/s", "their cruise speed")
    M_crz = C("M_crz", 0.80, "-", "cruise Mach")
    mu_air = C("mu_air", 1.45e-5, "kg/(m*s)", "viscosity at cruise")
    k_mac = C("k_mac", 1.12, "-", "mac over S/b at taper 0.25")
    f_lam = C("f_lam", 0.00248, "-", "Nita-Scholz taper function, lam=0.25")
    C_f = C("C_f", 0.0032, "-", "equivalent skin friction coefficient")
    W_pay = C("W_pay", 172146.2, "N", "payload, 180 pax x 215 lbf")
    R_req = C("R_req", 5.556e6, "m", "3000 nmi")
    R_fuse = C("R_fuse", 2.54, "m", "their fuselage radius")
    l_fixed = C("l_fixed", 38.55, "m", "their nose + cabin + tailcone")
    S_nace = C("S_nace", 24.3, "m^2", "nacelle wetted area, their fan size")
    k_fuse = C("k_fuse", 260.0, "N/m^2", "validated against their hull: 259")
    N_ult = C("N_ult", 3.0, "-", "ultimate load factor, same as theirs")
    f_res = C("f_res", 1.20, "-", "their freserve = 0.20")
    f_tail = C("f_tail", 0.112, "-", "their (Wht+Wvt)/Wwing")
    f_add = C("f_add", 0.065, "-", "their gear + hpesys fractions")
    k_eng = C("k_eng", 0.2963, "-", "their Weng / F_TO")
    TW = C("TW", 0.294, "-", "their takeoff thrust / MTOW")
    TSFC = C("TSFC", 0.23527 / 3600.0, "1/s", "their cruise TSFC, hydrogen")
    AR_fix = C("AR_fix", 10.1, "-", "their aspect ratio")
    k_sweep = C("k_sweep", 1.0 / math.cos(math.radians(26.0)) ** 2, "-",
                "structural-span factor for their 26 deg sweep")
    clear = C("clearance", 0.1, "m", "their tank-to-fuselage clearance")
    # Hoburg box -> TASOPT beam model, calibrated ONCE at their exact
    # planform and load (13,758 lb vs their 16,456 lb box): the declared
    # theory-difference factor, not a tuning knob.
    k_beam = C("k_beam", 1.196, "-", "TASOPT beam over Hoburg box, at their point")
    # Their insulation policy: sized so cruise boil-off is 0.4 %/hour of the
    # fuel load (sizes_insulation = true in their TOML).
    r_boil = C("r_boil", 0.004 / 3600.0, "1/s", "their boil-off rate limit")

    if point:
        cons += [wing["AR"] == AR_fix, cosL == 0.8988]
    cons += [
        # Structural span: load normal to the swept axis over the longer
        # structural span. This is what prices sweep against the wave drag
        # that rewards it.
        wing["L_max"] * cosL ** 2 >= N_ult * W_MTO,
        l_fuse >= l_fixed + tank["l_tank"],
        S_wet >= 2.0 * wing["S"] + 2.0 * PI * R_fuse * l_fuse + S_nace,
        R_fuse >= tank["R_o"] + tank["t_insul"] + clear,
        W_fuse >= k_fuse * 2.0 * PI * R_fuse * l_fuse,
        W_tail >= f_tail * k_beam * wing["W_wing"],
        F_TO >= TW * W_MTO,
        W_eng >= k_eng * F_TO,
        W_add >= f_add * W_MTO,
        tank["m_boil"] * g <= r_boil * tank["W_fuel"],
        W_dry >= (k_beam * wing["W_wing"] + W_fuse + W_tail + W_pay
                  + tank["W_tank"] + W_eng + W_add),
        W_MTO >= W_dry + tank["W_fuel"],
        W[0] >= W_MTO,
    ]
    for i in range(N):
        cons += [
            W[i] <= 0.5 * rho * V_inf ** 2 * wing["S"] * C_L[i],
            # Fuselage + nacelle skin friction only -- the wing's profile
            # and wave drag come from the fit, so the wetted-area lump must
            # not double-count it.
            C_D[i] >= (C_f * (2.0 * PI * R_fuse * l_fuse + S_nace)
                       / wing["S"] + CDp[i] + CDi[i]),
            CDi[i] >= C_L[i] ** 2 / (PI * wing["AR"] * e_osw_v),
            # Nita-Scholz: span efficiency falls with AR.
            e_osw_v + f_lam * e_osw_v * wing["AR"] <= 1.0,
            mac == k_mac * wing["S"] / wing["b"],
            Re == rho * V_inf * mac / mu_air,
            # Martin York's fit to the TASOPT C-series transonic airfoils --
            # the physics that lets CL, tau and sweep be free variables.
            CDp[i] ** 1.6515 >= (
                1.61418 * (Re / 1000.0) ** -0.550434 * wing["tau"] ** 1.29151
                    * (cosL * M_crz) ** 3.03609 * C_L[i] ** 1.77743
                + 0.0466407 * (Re / 1000.0) ** -0.389048
                    * wing["tau"] ** 0.784123
                    * (cosL * M_crz) ** -0.340157 * C_L[i] ** 0.950763
                + 190.811 * (Re / 1000.0) ** -0.218621
                    * wing["tau"] ** 3.94654
                    * (cosL * M_crz) ** 19.2524 * C_L[i] ** 1.15233
                + 2.82283e-12 * (Re / 1000.0) ** 1.18147
                    * wing["tau"] ** -1.75664
                    * (cosL * M_crz) ** 0.10563 * C_L[i] ** -1.44114),
            D[i] >= 0.5 * rho * V_inf ** 2 * wing["S"] * C_D[i],
            # Their engine at their cruise TSFC; thrust = drag in cruise.
            mdot_f[i] * g >= TSFC * D[i],
            V_inf * t_seg[i] >= R_req / N,
        ]
    for i in range(N - 1):
        cons += [W[i] <= W[i + 1]
                 + g * (mdot_f[i] + tank["m_boil"]) * t_seg[i]]
    cons += [tank["W_fuel"] >= f_res * sum(
        g * (mdot_f[i] + tank["m_boil"]) * t_seg[i] for i in range(N))]

    f.Objective(W_MTO)
    f.ConstraintList(cons)
    return f
