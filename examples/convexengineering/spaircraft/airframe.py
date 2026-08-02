"""The part of SPaircraft that every configuration shares.

``model.py`` builds the D8.2 as a single 500-line ``build()``. Most of what is
in there is not about the D8 at all -- the weight build-up, the mission, the
stability relations, the nacelle drag, the CG sum -- and rebuilding it once
per configuration would mean maintaining three copies of the same aircraft.
This module holds that common part so a configuration is what it should be: a
short file that states its numbers and its structural layout.

What a "deck" is
----------------
A deck is a build file for one aircraft: ``conventional.py`` (the 737-800),
``d8_no_bli.py``, and ``model.py`` (the D8.2, which predates this split). A
deck does four things:

1. calls :func:`add_components` with its sweeps and its engine;
2. calls :func:`add_constants` with the numbers from its ``subs/`` dict;
3. calls :func:`add_shared` for the constraints common to all aircraft;
4. adds its own **structural** constraints, and pins its own substitutions.

Step 4 is the configuration. It is deliberately not a flag here. Where the
source writes ``if rearengine: ... if wingengine: ...`` this module writes
nothing at all, and each deck states the physics it actually has -- so a wing
engine's root-moment relief lives in the file for the aircraft that has wing
engines, next to the ground-clearance constraint that only makes sense there.
Nothing in this module branches on configuration, and nothing in it needs to
be read to understand one.

The five structural choices, and where each is stated
-----------------------------------------------------
======================  ==================================================
engine location         wing root moment relief, ``A_1h``, ``I_z``,
                        ``x_eng``, ground clearance, ``y_eng``
fuselage cross-section  floor shear/moment coefficients, ``dR_fuse``
horizontal tail mount   root/attachment moments, ``pi_M_fac``
tail trailing edge      what limits ``dx_trail_ht``
inlet                   ``BLI`` and the engine name, ``D_reduct``
======================  ==================================================

Every one of those is a deck's business. Everything else is here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy import pi
from pyomo.environ import units

from .flight_state import add_flight_state
from .fuselage import add_fuselage
from .horizontal_tail import add_horizontal_tail
from .landing_gear import add_landing_gear
from .vertical_tail import add_vertical_tail
from .wing import add_wing

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turbofan.model import add_engine  # noqa: E402

__all__ = ["Parts", "add_components", "add_constants", "add_variables",
           "add_shared", "pin", "set_constant", "SHARED", "PER_DECK"]


class _NS:
    """Attribute and item access over a flat dict, like the group handles."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getitem__(self, name):
        return self.__dict__[name]

    def __repr__(self):                                  # pragma: no cover
        return f"_NS({', '.join(sorted(self.__dict__))})"


Parts = _NS


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def add_components(f, N, *, sweep_w, sweep_vt, sweep_ht, engine, BLI):
    """Add the six airframe components and the engine.

    Returns ``(parts, constraints)``. ``parts`` carries the group handles as
    ``st, wing, vt, ht, lg, fu, eng``.

    The component models take no configuration of their own -- a wing is a
    wing -- so the only thing that varies here is the three sweeps and which
    engine hangs off it.
    """
    st, cons = add_flight_state(f, N)
    wing, c = add_wing(f, N, st, sweep_deg=sweep_w); cons += c
    vt, c = add_vertical_tail(f, N, st, sweep_deg=sweep_vt); cons += c
    ht, c = add_horizontal_tail(f, N, st, sweep_deg=sweep_ht); cons += c
    lg, c = add_landing_gear(f); cons += c
    fu, c = add_fuselage(f); cons += c
    eng, c = add_engine(f, N, st, engine=engine, BLI=BLI, prefix="Eng_")
    cons += c
    return _NS(st=st, wing=wing, vt=vt, ht=ht, lg=lg, fu=fu, eng=eng), cons


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Aircraft-level constants that are identical in every ``subs/`` deck --
#: airframe-independent physics, the common mission, and cabin dimensions.
#: ``name: (value, units, description)``.
SHARED = {
    "g":            (9.81, "m/s^2", "gravitational acceleration"),
    "n_eng":        (2.0, "-", "number of engines"),
    "V_ne":         (143.92, "m/s", "never-exceed speed"),
    # subs/optimalD8.py and subs/optimal737.py both say 133.76, and 133.76 is
    # what the recorded gpkit solution carries. model.py has 133.94, which is
    # a transcription slip; the difference is 0.13% on a constraint that does
    # not bind at the optimum.
    "V_mn":         (133.76, "m/s", "manoeuvring speed"),
    "rho_TO_ac":    (1.225, "kg/m^3", "air density at takeoff"),
    "f_fuel_res":   (0.20, "-", "fuel reserve fraction"),
    "FuelFrac":     (0.9, "-", "usable fraction of max fuel volume"),
    "f_hpesys":     (0.01, "-", "power systems weight fraction"),
    "f_eadd":       (0.1, "-", "additional engine weight fraction"),
    "C_engsys":     (1.0, "-", "engine system weight margin"),
    "C_D_wm":       (0.5, "-", "windmill drag coefficient"),
    "V_land":       (72.0, "m/s", "aircraft landing speed"),
    "R_req":        (3000.0, "nmi", "required cruise range"),
    "w_seat":       (0.5, "m", "seat width"),
    "w_aisle":      (0.51, "m", "aisle width"),
    "w_sys":        (0.1, "m", "width between cabin and skin for systems"),
    "V_stall":      (120.0, "knots", "aircraft stall speed"),
    "W_Load_max":   (6664.0, "N/m^2", "max wing loading"),
    "P_cabin":      (75000.0, "Pa", "cabin air pressure"),
    "T_cabin":      (297.0, "K", "cabin air temperature"),
    "RC_min":       (500.0, "feet/min", "minimum rate of climb"),
    "c_m_w_val":    (1.9, "-", "wing pitching moment coefficient"),
}

#: Constants whose value *is* the configuration. Every deck must supply all of
#: them; there is no default, because a default would be one aircraft's number
#: silently standing in for another's. ``name: (units, description)``.
PER_DECK = {
    "n_vt":              ("-", "number of vertical tails"),
    "n_aisle":           ("-", "number of aisles"),
    "SM_min":            ("-", "minimum static margin"),
    "dx_CG":             ("ft", "max CG travel range"),
    "M_min":             ("-", "minimum cruise Mach number"),
    "f_L_total_wing":    ("-", "total lift as a fraction of wing lift"),
    "r_S_nacelle":       ("-", "nacelle and pylon wetted area factor"),
    "r_v_nacelle":       ("-", "incoming nacelle velocity ratio"),
    "f_pylon":           ("-", "pylon weight fraction"),
    "f_wingfuel":        ("-", "fraction of fuel in wing tanks"),
    "D_reduct":          ("-", "BLI drag reduction factor"),
    "b_max":             ("m", "max wing span"),
    "C_L_w_max":         ("-", "max wing lift coefficient"),
    "Fsafetyfac":        ("-", "safety factor on initial climb thrust"),
    "MinCruiseAlt":      ("ft", "minimum cruise altitude"),
    "MaxClimbTime":      ("min", "max time in climb"),
    "rdot_req":          ("s^-2", "required yaw rate at landing"),
    "C_D_fuse":          ("-", "fuselage drag coefficient"),
}


def add_constants(f, **values):
    """Create the aircraft-level constants. Returns a namespace.

    Every key in :data:`PER_DECK` must be given; keys in :data:`SHARED` may be
    overridden but rarely should be. An unknown key is an error rather than a
    silent no-op -- a misspelt constant would otherwise leave the intended one
    at its default and change the aeroplane without saying so.
    """
    unknown = set(values) - set(SHARED) - set(PER_DECK)
    if unknown:
        raise KeyError(f"unknown constant(s): {sorted(unknown)}")
    missing = set(PER_DECK) - set(values)
    if missing:
        raise KeyError(f"deck must supply: {sorted(missing)}")

    spec = {n: (values.get(n, v), u, d) for n, (v, u, d) in SHARED.items()}
    spec.update({n: (values[n], u, d) for n, (u, d) in PER_DECK.items()})

    ns = _NS()
    for name, (value, unit, desc) in spec.items():
        setattr(ns, name, f.Constant(name=name, value=value, units=unit,
                                     description=desc))
    return ns


def set_constant(handle, name, value):
    """Override a component's ``Constant`` in place.

    EDI constants are mutable Pyomo ``Param``s, so a deck can retune one a
    component declared with a different aircraft in mind -- ``SPR`` and
    ``M_fuseD`` are both hard-coded to D8 values in ``fuselage.py`` -- without
    touching the component model or adding a constraint to undo it.
    """
    handle[name].set_value(value)


def pin(handle, name, value, unit=None):
    """A constraint fixing a component *variable* to a substituted value.

    The source's ``subs/`` dicts fix these; leaving one free lets the
    optimizer choose it, which is not always benign. ``n_pass`` is the extreme
    case: unpinned, the payload collapses to 15 lbf and the aeroplane shrinks
    around it.
    """
    return handle[name] == (value * unit if unit is not None else value)


# ---------------------------------------------------------------------------
# Aircraft-level variables
# ---------------------------------------------------------------------------

def add_variables(f, N):
    """Every aircraft-level free variable. Returns a namespace.

    These are the same for all configurations even when the constraints that
    size them are not -- a rear-engined aeroplane and a wing-engined one both
    have an ``x_eng``, they just put it in different places.
    """
    V = lambda n, g, u, d: f.Variable(name=n, guess=g, units=u, description=d)
    Vn = lambda n, g, u, d: f.Variable(name=n, guess=g, units=u,
                                       description=d, size=N)
    v = _NS()

    # ---- scalars ----------------------------------------------------------
    v.W_total     = V("W_total", 1.4e5, "lbf", "total aircraft weight")
    v.W_totalmax  = V("W_total_max", 1.4e5, "lbf", "maximum total weight")
    v.W_dry       = V("W_dry", 7.4e4, "lbf", "zero-fuel aircraft weight")
    v.W_ftotal    = V("W_f_total", 2.1e4, "lbf", "total fuel weight")
    v.W_fprimary  = V("W_f_primary", 1.75e4, "lbf", "fuel less reserves")
    v.W_fclimb    = V("W_f_climb", 5e3, "lbf", "fuel burned in climb")
    v.W_fcruise   = V("W_f_cruise", 1.25e4, "lbf", "fuel burned in cruise")
    v.Wmisc       = V("W_misc", 1e4, "lbf", "sum of miscellaneous weights")
    v.Whpesys     = V("W_hpesys", 1.4e3, "lbf", "power systems weight")
    v.xmisc       = V("x_misc", 6.0, "m", "misc weight centroid")
    v.xhpesys     = V("x_hpesys", 9.7, "m", "power systems x-location")
    v.Iz          = V("I_z", 1e7, "kg*m^2", "aircraft z-axis moment of inertia")
    v.Izwing      = V("I_z_wing", 5e6, "kg*m^2", "wing moment of inertia")
    v.Iztail      = V("I_z_tail", 2e6, "kg*m^2", "tail moment of inertia")
    v.Izfuse      = V("I_z_fuse", 3e6, "kg*m^2", "fuselage moment of inertia")

    # ---- engine installation ----------------------------------------------
    v.Snace       = V("S_nacelle", 8.0, "m^2", "nacelle surface area")
    v.lnace       = V("l_nacelle", 1.6, "m", "nacelle length")
    v.fSnace      = V("f_S_nacelle", 0.06, "-", "non-dimensional nacelle area")
    v.Ainlet      = V("A_inlet", 3.2, "m^2", "inlet area")
    v.Afancowl    = V("A_fancowl", 1.6, "m^2", "fan cowling area")
    v.Aexh        = V("A_exh", 3.2, "m^2", "exhaust area")
    v.Acorecowl   = V("A_corecowl", 6.0, "m^2", "core cowling area")
    v.Wnace       = V("W_nacelle", 2e3, "lbf", "nacelle weight")
    v.Wpylon      = V("W_pylon", 6e2, "lbf", "engine pylon weight")
    v.Weadd       = V("W_eadd", 8e2, "lbf", "additional engine system weight")
    v.Wengsys     = V("W_engsys", 1e4, "lbf", "total engine system weight")
    v.xeng        = V("x_eng", 32.0, "m", "engine x-location")
    v.y_eng       = V("y_eng", 2.0, "m", "engine moment arm")

    # ---- per-segment ------------------------------------------------------
    v.D        = Vn("D", 6e4, "N", "total aircraft drag")
    v.C_D      = Vn("C_D", 0.03, "-", "total aircraft drag coefficient")
    v.LoD      = Vn("LoD", 18.0, "-", "lift-to-drag ratio")
    v.W_avg    = Vn("W_avg", 1.3e5, "lbf", "geometric mean segment weight")
    v.W_start  = Vn("W_start", 1.4e5, "lbf", "segment start weight")
    v.W_end    = Vn("W_end", 1.3e5, "lbf", "segment end weight")
    v.W_burn   = Vn("W_burn", 4e3, "lbf", "segment fuel burn")
    v.WLoad    = Vn("W_Load", 5e3, "N/m^2", "wing loading")
    v.tmin     = Vn("tmin", 60.0, "min", "segment flight time in minutes")
    v.thr      = Vn("thr", 1.0, "hr", "segment flight time in hours")
    v.xAC      = Vn("x_AC", 19.0, "m", "aerodynamic centre of the aircraft")
    v.xCG      = Vn("x_CG", 18.0, "m", "centre of gravity of the aircraft")
    v.xNP      = Vn("x_NP", 19.0, "m", "neutral point of the aircraft")
    v.SM       = Vn("SM", 0.1, "-", "stability margin")
    v.PCFuel   = Vn("F_fuel", 0.5, "-", "fraction of fuel remaining")
    v.W_buoy   = Vn("W_buoy", 1e3, "lbf", "buoyancy weight")
    v.rhocabin = Vn("rho_cabin", 0.88, "kg/m^3", "cabin air density")
    v.Ltotal   = Vn("L_total", 6e5, "N", "total lift")
    v.Dfuse    = Vn("D_fuse", 1.5e4, "N", "fuselage drag")
    v.Lfuse    = Vn("L_fuse", 1e5, "N", "fuselage lift")
    v.Vnace    = Vn("V_nacelle", 210.0, "m/s", "incoming nacelle flow velocity")
    v.V2       = Vn("V_2", 140.0, "m/s", "interior nacelle flow velocity")
    v.Vnacrat  = Vn("V_nacelle_ratio", 1.2, "-", "nacelle velocity ratio")
    v.rvnsurf  = Vn("r_v_nsurf", 1.0, "-", "intermediate nacelle drag parameter")
    v.Cfnace   = Vn("C_f_nacelle", 0.003, "-", "nacelle skin friction coefficient")
    v.Renace   = Vn("Re_nacelle", 2e7, "-", "nacelle Reynolds number")
    v.Cdnace   = Vn("C_d_nacelle", 2e-4, "-", "nacelle drag coefficient")
    v.Dnace    = Vn("D_nacelle", 1e3, "N", "drag on one nacelle")
    v.theta    = Vn("theta", 0.02, "-", "aircraft climb angle")
    v.excessP  = Vn("P_excess", 1e6, "W", "excess power during climb")
    v.RC       = Vn("RC", 1500.0, "feet/min", "rate of climb")
    v.dhft     = Vn("dhft", 12000.0, "feet", "altitude change per segment")
    v.Rseg     = Vn("R_segment", 600.0, "nmi", "down range covered in each segment")
    return v


# ---------------------------------------------------------------------------
# Shared constraints
# ---------------------------------------------------------------------------

def add_shared(f, N, Nclimb, p, v, K):
    """Every constraint that does not depend on the configuration.

    ``p`` is from :func:`add_components`, ``v`` from :func:`add_variables`,
    ``K`` from :func:`add_constants`. Returns a list of constraints; the deck
    appends its own and calls ``f.ConstraintList`` once.
    """
    wing, vt, ht, lg, fu, eng, st = p.wing, p.vt, p.ht, p.lg, p.fu, p.eng, p.st

    cons = [
        # ---- linking -------------------------------------------------------
        wing.c_root == fu.c_0,
        wing.x_w == fu.x_wing,
        fu.N_lift == wing.box.N_lift,
        K.f_L_total_wing * wing.L_max >= wing.box.N_lift * v.W_totalmax + ht.L_ht_max,
        wing.b <= K.b_max,

        # ---- weight build-up ------------------------------------------------
        fu.W_fuse + K.n_eng * v.Wengsys + fu.W_tail + wing.W_wing + v.Wmisc <= v.W_dry,
        v.W_ftotal + v.W_dry + fu.W_payload <= v.W_total,
        v.W_ftotal >= v.W_fprimary + K.f_fuel_res * v.W_fprimary,
        v.W_fprimary >= v.W_fclimb + v.W_fcruise,
        v.W_totalmax >= v.W_total,
        wing.W_fuel_wing >= K.f_wingfuel * v.W_ftotal / K.FuelFrac,

        # ---- landing gear and power systems ----------------------------------
        v.Wmisc >= lg.W_lg + v.Whpesys,
        v.Whpesys == K.f_hpesys * v.W_totalmax,
        lg.x_n <= fu.l_nose,
        lg.x_m >= fu.x_wing,
        lg.x_m <= wing.dx_AC_wing + fu.x_wing,
        v.xhpesys == 1.1 * fu.l_nose,
        v.xmisc * v.Wmisc >= v.xhpesys * v.Whpesys,
        lg.d_nacelle >= eng.d_f + 2 * lg.t_nacelle,
        lg.d_nacelle <= wing.b,
        # Hard landing, Torenbeek (10-26): 10 ft/s sink at max landing weight.
        lg.E_land >= v.W_totalmax / (2 * K.g) * lg.w_ult ** 2,
        lg.x_up == fu.x_shell2,
        lg.L_n == v.W_totalmax * lg.dx_m / lg.B,
        lg.L_m == v.W_totalmax * lg.dx_n / lg.B,
        lg.L_n_dyn >= 0.31 * ((lg.z_CG + lg.l_m) / lg.B) * v.W_totalmax,
        v.y_eng >= lg.y_m,

        # ---- fuselage --------------------------------------------------------
        # Tail cone sizing, driven by the VT root moment.
        3. * (K.n_vt * vt.box.M_r) * vt.c_root_vt * (fu.p_lambda_vt - 1.)
            >= K.n_vt * vt.L_vt_max * vt.b_vt * fu.p_lambda_vt,
        fu.V_cone * (1. + fu.lambda_cone) * (pi + 4. * fu.theta_db)
            >= (K.n_vt * vt.box.M_r * vt.c_root_vt / fu.tau_cone
                * (pi + 2. * fu.theta_db) * (fu.l_cone / fu.R_fuse)),
        fu.W_tail >= K.n_vt * vt.W_vt + ht.W_ht,
        2. * fu.w_fuse >= (fu.SPR * K.w_seat + K.n_aisle * K.w_aisle
                           + 2. * K.w_sys + fu.t_db),
        fu.B_1v == fu.r_M_v * K.n_vt * vt.L_vt_max / (fu.w_fuse * fu.sigma_M_v),

        # ---- horizontal tail --------------------------------------------------
        ht.m_ratio * (1 + 2 / wing.AR) == 1 + 2 / ht.AR_ht,     # [SP] SigEq
        ht.x_CG_ht <= fu.l_fuse,
        ht.V_ht == ht.S_ht * ht.l_ht / (wing.S * wing.mac),
        ht.L_ht_max >= 0.5 * K.rho_TO_ac * K.V_ne ** 2 * ht.S_ht * ht.C_L_ht_max,

        # ---- vertical tail ----------------------------------------------------
        vt.L_vt_max >= 0.5 * K.rho_TO_ac * K.V_ne ** 2 * vt.S_vt * vt.C_L_vt_max,
        vt.x_CG_vt <= fu.l_fuse,
        vt.V_vt == K.n_vt * vt.S_vt * vt.l_vt / (wing.S * wing.b),
        # Yaw rate at flare
        K.n_vt * .5 * vt.rho_TO * K.V_land ** 2 * vt.S_vt * vt.l_vt
            * vt.C_L_vt_yaw >= K.rdot_req * vt.I_z_max,
        # One-engine-out moment balance (TASOPT 2.0 p45)
        K.n_vt * vt.L_vt_EO * vt.l_vt >= vt.T_e * v.y_eng + vt.D_wm * v.y_eng,
        vt.D_wm >= 0.5 * vt.rho_TO * vt.V_1 ** 2. * eng.A_2 * K.C_D_wm,

        # ---- moment of inertia -------------------------------------------------
        v.Iz >= v.Izwing + v.Iztail + v.Izfuse,
        vt.I_z_max >= v.Iz,

        # ---- engine installation ------------------------------------------------
        v.Snace == K.r_S_nacelle * np.pi * 0.25 * eng.d_f ** 2,
        v.lnace == 0.15 * eng.d_f * K.r_S_nacelle,
        v.fSnace == v.Snace * wing.S ** -1,
        v.Ainlet == 0.4 * v.Snace,
        v.Afancowl == 0.2 * v.Snace,
        v.Aexh == 0.4 * v.Snace,
        v.Acorecowl == 3. * np.pi * eng.d_LPC ** 2,
        v.Wnace >= ((2.5 + 0.238 * eng.d_f / units.inch) * v.Ainlet
                    + 1.9 * v.Afancowl
                    + (2.5 + 0.0363 * eng.d_f / units.inch) * v.Aexh
                    + 1.9 * v.Acorecowl) * units.lbf / units.ft ** 2,
        v.Weadd == K.f_eadd * eng.W_engine,
        v.Wpylon >= (v.Wnace + v.Weadd + eng.W_engine) * K.f_pylon,
        v.Wengsys >= K.C_engsys * (v.Wpylon + v.Wnace + v.Weadd + eng.W_engine),
    ]

    # ---- per-segment performance ------------------------------------------
    cons += [
        v.rhocabin == K.P_cabin / (st.R * K.T_cabin),
        st.V >= K.V_stall,
        v.W_avg >= (v.W_start * v.W_end) ** .5 + v.W_buoy,
        v.tmin == v.thr,
        v.W_buoy >= v.rhocabin * K.g * fu.V_cabin,
        # Fuselage lift, as a fraction of wing lift.
        v.Lfuse == (K.f_L_total_wing - 1.) * wing.L_w,          # [SP] SigEq
        v.Ltotal == K.f_L_total_wing * wing.L_w,
        v.Ltotal >= v.W_avg + ht.L_ht,

        # ---- drag ------------------------------------------------------
        v.Dfuse == (0.5 * st.rho * st.V ** 2 * K.C_D_fuse
                    * fu.l_fuse * fu.R_fuse
                    * (st.M ** 2 / fu.M_fuseD ** 2)),
        v.D >= K.D_reduct * (wing.D_wing + v.Dfuse + K.n_vt * vt.D_vt
                             + ht.D_ht + K.n_eng * v.Dnace),
        v.C_D == v.D / (.5 * st.rho * st.V ** 2 * wing.S),
        v.LoD == v.W_avg / v.D,

        # ---- wing loading and lift losses -------------------------------
        v.WLoad <= K.W_Load_max,
        v.WLoad == (.5 * wing.C_L * st.rho * st.V ** 2),
        wing.p_o >= wing.L_w * wing.c_root / wing.S,
        wing.eta_o == fu.w_fuse / (wing.b / 2),

        # ---- stability ---------------------------------------------------
        v.xAC <= fu.x_wing + 0.25 * wing.dx_AC_wing + v.xNP,
        wing.c_m_w == K.c_m_w_val,
        # Neutral point approximation, from Unified's aircraft design rules.
        (v.xNP / wing.mac / ht.V_ht * (wing.AR + 2.)
         * (1. + 2. / ht.AR_ht)
         == (1. + 2. / wing.AR) * (wing.AR - 2.)),              # [SP] SigEq
        v.xCG + vt.dx_trail_vt <= fu.l_fuse,
        vt.x_CG_vt >= v.xCG + 0.5 * (vt.dx_lead_vt + vt.dx_trail_vt),
        ht.x_CG_ht >= v.xCG + 0.5 * (ht.dx_lead_ht + ht.dx_trail_ht),
        ht.C_L_alpha_ht + (2 * wing.C_L_alpha_w / (pi * wing.AR))
            * ht.eta_ht * ht.C_L_alpha_ht_0
            <= ht.C_L_alpha_ht_0 * ht.eta_ht,
        ht.C_L_ht >= 0.01,
        v.SM <= (v.xAC - v.xCG) / wing.mac,
        v.SM >= K.SM_min,
        v.xAC / wing.mac <= (v.xCG / wing.mac + K.c_m_w_val / wing.C_L
                             + ht.V_ht * (ht.C_L_ht / wing.C_L)),

        # ---- nacelle drag --------------------------------------------------
        v.Renace == st.rho * st.V * v.lnace / st.mu,
        v.Cfnace == 0.94 * 4. * 0.0743 / (v.Renace ** 0.2),
        v.Vnace == K.r_v_nacelle * st.V,
        v.Vnacrat >= 2. * v.Vnace / st.V - v.V2 / st.V,
        v.rvnsurf ** 3. >= 0.25 * (v.Vnacrat + K.r_v_nacelle)
                           * (v.Vnacrat ** 2. + K.r_v_nacelle ** 2.),
        v.Cdnace == v.fSnace * v.Cfnace[0] * v.rvnsurf ** 3.,
        v.Dnace == v.Cdnace * 0.5 * st.rho * st.V ** 2. * wing.S,
        v.V2 == eng.M_2 * st.a,

        # ---- climb -----------------------------------------------------------
        v.excessP + st.V * v.D <= st.V * K.n_eng * eng.F,
        v.RC == v.excessP / v.W_avg,
        v.theta * st.V == v.RC,
        v.dhft == v.tmin * v.RC,
        v.Rseg == v.thr * st.V,
        K.n_eng * eng.F >= v.D + v.W_avg * v.theta,

        # ---- CG ----------------------------------------------------------------
        # Identical for wing- and rear-engined aircraft: the source states it
        # twice under the two flags, with the same terms in a different order.
        v.xCG * v.W_avg >= (
            v.xmisc * v.Wmisc + lg.x_CG_lg * lg.W_lg
            + 0.5 * (fu.W_fuse + fu.W_payload) * fu.l_fuse
            + ht.W_ht * ht.x_CG_ht + vt.W_vt * vt.x_CG_vt
            + K.n_eng * v.Wengsys * v.xeng
            + wing.W_wing * (fu.x_wing + wing.dx_AC_wing)
            + (v.PCFuel + K.f_fuel_res) * v.W_fprimary
            * (fu.x_wing + wing.dx_AC_wing * v.PCFuel)),

        # ---- fuel burn -----------------------------------------------------------
        v.W_burn == K.n_eng * eng.TSFC * v.thr * eng.F,
        v.W_start >= v.W_end + v.W_burn,
        v.PCFuel <= 1.0000001,

        # ---- engine operating point -----------------------------------------------
        eng.M_2 == st.M,
        eng.M_25 == 0.6,
        eng.hold_2 == 1. + .5 * (1.398 - 1.) * 0.6 ** 2,
        eng.hold_25 == 1. + .5 * (1.354 - 1.) * 0.6 ** 2,
        eng.c1 == 1. + 0.5 * .401 * st.M ** 2.,                 # [SP] SigEq
    ]

    # Aircraft-level geometry and stability limits. These mention no segment,
    # so the source's per-segment loop stated each of them N times over; they
    # are stated once here. Identical rows constrain nothing extra -- presolve
    # would drop the copies anyway -- so this removes 2*(N-1) rows and leaves
    # the feasible set alone.
    cons += [
        ht.AR_ht >= 4.,
        K.SM_min + K.dx_CG / wing.mac + K.c_m_w_val / K.C_L_w_max
            <= ht.V_ht * ht.m_ratio + ht.V_ht * ht.C_L_ht_max / K.C_L_w_max,
    ]

    # ---- mission stitching ------------------------------------------------------
    cons += [
        v.W_start[0] == v.W_total,
        st.hft[0] == v.dhft[0],
        st.hft[Nclimb - 1] >= K.MinCruiseAlt,
        v.RC[0] >= 2500. * units.ft / units.min,
        v.theta[Nclimb - 1] >= 0.015,
        vt.T_e == K.Fsafetyfac * eng.F[0],
        v.W_dry + fu.W_payload + K.f_fuel_res * v.W_fprimary <= v.W_end[N - 1],
        v.W_fclimb >= f.sum(v.W_burn[:Nclimb]),
        v.W_fcruise >= f.sum(v.W_burn[Nclimb:]),
        f.sum(v.Rseg) >= K.R_req,
        f.sum(v.thr[:Nclimb]) <= K.MaxClimbTime,
        lg.dx_n + lg.x_n >= v.xCG[Nclimb],
        lg.dx_m + v.xCG[Nclimb] >= lg.x_m,
        lg.x_m >= lg.tan_phi * (lg.z_CG + lg.l_m) + v.xCG[Nclimb],
        # The mission caps bypass ratio rather than fixing it; the subs decks
        # supply neither alpha_max nor alpha_OD, so both are free.
        eng.alpha_max <= 100.0,
    ]
    # Segment-to-segment stitching: each of these relates a segment to the one
    # before it, which is what the offset slices say.
    cons += [
        v.W_start[1:] == v.W_end[:-1],
        st.hft[1:] == st.hft[:-1] + v.dhft[1:],                 # [SP] SigEq
        v.dhft[1:Nclimb] == v.dhft[:Nclimb - 1],
        v.RC[1:Nclimb] >= K.RC_min,
        st.M[Nclimb:] >= K.M_min,
    ]
    # Keep Mach inside the transonic band. Not in the source, but needed here:
    # the VT drag fit carries M**1022.7 and M**-114.577, and in the log-space
    # GP those become exponents of ~1023*log(M). A line search that steps M
    # even slightly above 1 overflows exp() and IPOPT reports "Error in an
    # AMPL evaluation". gpkit avoids this because MOSEK's exponential-cone
    # form never forms exp() explicitly. 0.1-0.95 is far outside any
    # physically meaningful excursion for these aircraft.
    cons += [st.M <= 0.95, st.M >= 0.1]
    cons += [v.Rseg[Nclimb:N - 1] == v.Rseg[Nclimb + 1:N]]
    # Fuel still to burn after each segment. The slice bound moves with the
    # segment, so this one keeps its loop.
    for i in range(N):
        rest = f.sum(v.W_burn[i + 1:])
        cons += [v.PCFuel[i] >= (rest + 0.0000001 * v.W_fprimary) / v.W_fprimary]
    # Wing max angle of attack differs between climb and cruise.
    cons += [wing.alpha_w[:Nclimb] <= 0.18,
             wing.alpha_w[Nclimb:] <= 0.10]
    return cons
