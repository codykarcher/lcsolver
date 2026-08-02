"""Certification performance constraints: takeoff, climb gradients, landing.

Why this exists
---------------
The model had NO takeoff sizing of any kind. Thrust was set by the cruise-climb
rate requirement alone (``RC[0] >= 2500 ft/min``), which produced 8,893 lbf per
engine on a 737-class aircraft against a CFM56-7B's ~27,300 lbf. That is not a
small error and it does not stay local: the engine-out yawing moment is
proportional to thrust, so the vertical tail came out at 11.6 m2 against the
real 26.4 m2, and ``V_vt`` at 0.037 against a transport norm near 0.09.

Real transport engines are sized by takeoff -- field length and the
one-engine-inoperative climb gradients -- not by cruise climb. TASOPT knows
this; its run decks carry ``lBFmax``, a balanced field length limit. This
module puts that back.

Rulesets
--------
The gradient requirements differ by certification basis, so they are data
rather than code. ``FAR25`` is the transport-category set that the E175, 737
and 787 certify under. ``FAR23_commuter`` is the commuter-category set, which
is the right basis for a small business jet or regional turboprop.

    CAUTION on the numbers. The FAR 25.121 gradients (2.4 / 2.7 / 3.0 % for
    two / three / four engines in the second segment, 1.2 / 1.5 / 1.7 %
    en route, 2.1 / 2.4 / 2.7 % approach, and 25.119's 3.2 % landing climb)
    are the standard transport values and I am confident in them. The Part 23
    commuter numbers below are my best recollection of the pre-Amendment-64
    text and SHOULD BE CHECKED against the current CFR before any result that
    depends on them is published. They are isolated here precisely so that
    checking them is a one-line edit.

What is approximated
--------------------
High lift. This model has no flap system: ``C_L_w_max`` is a single
cruise-referenced number and there is no flap geometry, no slat, no
deployment schedule. So takeoff and landing configurations are represented by
book increments -- a maximum lift coefficient and a drag increment per
configuration -- carried as named constants below rather than derived. They are
the least defensible numbers in this file, and they are the ones to replace
first if takeoff performance becomes a headline result rather than a sizing
constraint.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Ruleset:
    """Certification performance requirements, as data.

    Gradients are fractions, not percent. Each maps engine count -> required
    climb gradient with one engine inoperative, except ``landing_climb`` which
    is an all-engines-operating requirement.
    """
    name: str
    # 25.121(a): gear DOWN, takeoff flaps, OEI, at V_LOF
    first_segment: dict
    # 25.121(b): gear up, takeoff flaps, OEI, at V2 -- the one that sizes thrust
    second_segment: dict
    # 25.121(c): gear up, flaps up, OEI, en route
    enroute: dict
    # 25.121(d): approach flaps, OEI
    approach: dict
    # 25.119: all engines, landing flaps, balked landing
    landing_climb: float
    # 25.107: V2 >= v2_factor * V_stall(takeoff config)
    v2_factor: float
    # 25.125: V_ref >= vref_factor * V_stall(landing config)
    vref_factor: float
    # 25.109/25.125: demonstrated distance is factored up to field length
    takeoff_field_factor: float
    landing_field_factor: float
    source: str = ""


FAR25 = Ruleset(
    name="FAR25",
    first_segment={2: 0.000, 3: 0.003, 4: 0.005},
    second_segment={2: 0.024, 3: 0.027, 4: 0.030},
    enroute={2: 0.012, 3: 0.015, 4: 0.017},
    approach={2: 0.021, 3: 0.024, 4: 0.027},
    landing_climb=0.032,
    v2_factor=1.13,
    vref_factor=1.23,
    # 25.109: takeoff distance is the greater of the OEI distance and 1.15x
    # the all-engines distance. Applied as a single factor on the computed
    # ground roll plus air distance.
    takeoff_field_factor=1.15,
    # 25.125(b): landing distance is demonstrated distance / 0.6 for a dry
    # runway, i.e. a 1.667 factor.
    landing_field_factor=1.667,
    source="14 CFR 25.107, 25.109, 25.119, 25.121, 25.125",
)

FAR23_COMMUTER = Ruleset(
    name="FAR23_commuter",
    # Part 23 commuter category is less demanding than transport category and
    # does not break the requirement out by engine count the way 25.121 does.
    # VERIFY THESE AGAINST THE CURRENT CFR -- see the module docstring.
    first_segment={2: 0.000, 3: 0.000, 4: 0.000},
    second_segment={2: 0.020, 3: 0.020, 4: 0.020},
    enroute={2: 0.012, 3: 0.012, 4: 0.012},
    approach={2: 0.021, 3: 0.021, 4: 0.021},
    landing_climb=0.033,
    v2_factor=1.13,
    vref_factor=1.30,
    takeoff_field_factor=1.15,
    landing_field_factor=1.667,
    source="14 CFR 23.63/23.67/23.77 (commuter) -- CHECK BEFORE PUBLISHING",
)

RULESETS = {r.name: r for r in (FAR25, FAR23_COMMUTER)}


# --- high-lift stand-ins ----------------------------------------------------
# Book values, because there is no flap model. Raymer/Torenbeek ranges for a
# transport with slotted flaps and slats:
#   takeoff flaps   C_Lmax 1.8-2.2,  dCD ~ 0.015-0.025 flaps + 0.015-0.025 gear
#   landing flaps   C_Lmax 2.6-3.2,  dCD ~ 0.06-0.10
#: 2.2, not 2.0. A 737 at flaps 5 reaches roughly 2.2-2.4; 2.0 put our
#: takeoff stall speed at 140 kt against the real aircraft's ~129, and every
#: takeoff speed is referenced to it -- including the one the fin is sized at.
CLMAX_TO = 2.2
CLMAX_LAND = 2.8
DCD_FLAP_TO = 0.020
DCD_FLAP_LAND = 0.080
DCD_GEAR = 0.020
MU_ROLL = 0.025          # rolling friction, dry paved runway


def add_far(f, *, n_eng, ruleset="FAR25", prefix="FAR_",
            field_length_max_ft=None, landing_field_max_ft=None,
            k_mcg=0.88):
    """Add certification performance constraints. Returns ``(group, cons)``.

    The caller wires the aircraft-level quantities in afterwards via
    :func:`link`, because they live on other groups (weight, thrust, drag,
    wing area). Keeping the two apart means this module never reaches into the
    aircraft's namespace.

    ``field_length_max_ft`` is the straight runway limit -- pass a number and
    the takeoff field length is constrained to it, pass ``None`` and the field
    length is computed and reported but does not constrain the design.
    """
    rs = RULESETS[ruleset] if isinstance(ruleset, str) else ruleset
    ne = int(n_eng)
    if ne not in rs.second_segment:
        raise ValueError(f"{rs.name} has no gradient requirement for {ne} "
                         f"engines; known: {sorted(rs.second_segment)}")

    far = f.group("far", prefix=prefix)
    V, C = far.Variable, far.Constant

    # ---- speeds ------------------------------------------------------------
    Vs_to = V("V_s_TO", 60.0, "m/s", "stall speed, takeoff configuration")
    Vs_ld = V("V_s_land", 55.0, "m/s", "stall speed, landing configuration")
    V2 = V("V_2", 75.0, "m/s", "takeoff safety speed")
    Vlof = V("V_LOF", 70.0, "m/s", "lift-off speed")
    Vref = V("V_ref", 70.0, "m/s", "reference approach speed")

    Vmc = V("V_MC", 70.0, "m/s", "minimum control speed, air (FAR 25.149)")
    Vmcg = V("V_MCG", 62.0, "m/s", "minimum control speed, ground (FAR 25.149(e))")

    # ---- distances ---------------------------------------------------------
    s_g = V("s_ground", 1500.0, "m", "takeoff ground roll")
    s_air = V("s_air", 400.0, "m", "takeoff air distance to 35 ft")
    s_TO = V("s_TO", 2200.0, "m", "takeoff field length")
    s_land = V("s_land", 1800.0, "m", "landing field length")
    s_air_ld = V("s_air_land", 290.0, "m", "landing air distance from 50 ft")
    a_TO = V("a_TO", 2.0, "m/s^2", "mean takeoff acceleration")

    # ---- configuration aero ------------------------------------------------
    CLmaxTO = C("C_Lmax_TO", CLMAX_TO, "-", "max lift coefficient, takeoff flaps")
    CLmaxLD = C("C_Lmax_land", CLMAX_LAND, "-", "max lift coefficient, landing flaps")
    dCD_to = C("dC_D_flap_TO", DCD_FLAP_TO, "-", "takeoff flap drag increment")
    dCD_ld = C("dC_D_flap_land", DCD_FLAP_LAND, "-", "landing flap drag increment")
    dCD_lg = C("dC_D_gear", DCD_GEAR, "-", "landing gear drag increment")
    mu = C("mu_roll", MU_ROLL, "-", "rolling friction coefficient")
    # Rolling friction PLUS mean aerodynamic drag over the roll, as a single
    # coefficient. 0.04 is the standard dry-runway value for a jet transport
    # with takeoff flaps (Raymer, Torenbeek); the bare rolling term alone is
    # 0.02-0.03 and the aerodynamic part contributes the rest.
    mu_e = C("mu_eff", 0.04, "-", "effective roll resistance incl. drag")
    h_scr = C("h_screen", 10.668, "m", "35 ft screen height, FAR 25.109")
    h_scl = C("h_screen_land", 15.24, "m", "50 ft screen height, FAR 25.125")
    g = C("g", 9.81, "m/s^2", "gravitational acceleration")
    # ---- SECOND TAKEOFF CASE: sea level, standard day ---------------------
    # A transport has to meet its published field length at SEA LEVEL, and also
    # get out of a longer runway on a hot-and-high day. Those are two different
    # requirements and the model needs both: with only the hot-and-high one, at
    # the sea-level number, the 737 bought its way to 7700 ft with a 1.09 wing
    # AND 1.22 thrust, because it was being asked to do a sea-level job in
    # Denver air. The climb cases stay hot-and-high -- those are the ones that
    # genuinely bite at altitude -- and only the takeoff DISTANCE gets a second,
    # sea-level evaluation against the published number.
    Vs_sl = V("V_s_TO_sl", 60.0, "m/s", "stall speed, takeoff, sea level")
    Vlof_sl = V("V_LOF_sl", 70.0, "m/s", "lift-off speed, sea level")
    a_sl = V("a_TO_sl", 2.0, "m/s^2", "mean takeoff acceleration, sea level")
    sg_sl = V("s_ground_sl", 1500.0, "m", "ground roll, sea level")
    sair_sl = V("s_air_sl", 400.0, "m", "air distance to screen, sea level")
    sTO_sl = V("s_TO_sl", 2000.0, "m", "takeoff field length, sea level")
    rho_sl_c = C("rho_sl_TO", 1.225, "kg/m^3", "sea-level takeoff density")

    # ---- BALANCED FIELD LENGTH (FAR 25.109 accelerate-stop / -go) ----------
    # The all-engines roll above, factored by 1.15, is only HALF of 25.113:
    # the field length is the longer of that and the balanced field, and for
    # a twin the balanced field is what actually sizes the engine -- lose one
    # of two engines at V1 and the survivor has to drag the aircraft to V2 on
    # the runway that remains. Without this case the engine came out at 0.70
    # of TASOPT's on the D8: TASOPT constrains lBF (takeoff.f, LlBFcon in
    # every deck) and we were letting the all-engines case set thrust.
    #
    # Same three-phase structure as takeoff.f, reduced to mean accelerations
    # (theirs integrates F = F0 - KV*V^2 exactly and Newton-solves the
    # transcendental balance; a GP cannot hold an exp(), but the constant-a
    # reduction is the same physics and V_1 lands within a few knots):
    #   A: all engines, 0 -> V1        s_1 = V1^2 / 2a
    #   B: engine out, V1 -> V2        s_go = (V2^2 - V1^2) / 2a_OEI
    #   C: max braking, V1 -> 0        s_stop = V1^2 / 2(mu_brake g)
    # V_1 is FREE: raising it shortens the go branch and lengthens the stop,
    # so with both branches under the same runway limit the optimizer places
    # V_1 exactly where they balance -- which is the definition of V1.
    # takeoff.f's lBF ends at V2 on the ground (no air segment), and its
    # braking phase takes no credit for reversers, per FAR 25.109(f).
    V1 = V("V_1", 65.0, "m/s", "takeoff decision speed")
    a_oei = V("a_OEI", 1.0, "m/s^2", "mean OEI acceleration, V1 to V2")
    s_go = V("s_go_OEI", 600.0, "m", "OEI continue distance, V1 to V2")
    s_stp = V("s_stop", 600.0, "m", "accelerate-stop braking distance from V1")
    s_bf = V("s_BF", 2000.0, "m", "balanced field length")
    V1_sl = V("V_1_sl", 65.0, "m/s", "decision speed, sea level")
    aoei_sl = V("a_OEI_sl", 1.0, "m/s^2", "mean OEI acceleration, sea level")
    sgo_sl = V("s_go_OEI_sl", 600.0, "m", "OEI continue distance, sea level")
    sstp_sl = V("s_stop_sl", 600.0, "m", "braking distance, sea level")
    sbf_sl = V("s_BF_sl", 2000.0, "m", "balanced field length, sea level")
    mub = C("mu_brake", 0.35, "-",
            "max braking coefficient (TASOPT mubrake, both decks)")
    kmcg = C("k_MCG", k_mcg, "-",
             "V_MCG as a fraction of V_s_TO (calibrated; see SizeClass)")

    out = dict(_ruleset=rs, _n_eng=ne,
               V_s_TO=Vs_to, V_s_land=Vs_ld, V_2=V2, V_LOF=Vlof, V_ref=Vref,
               V_MC=Vmc, V_MCG=Vmcg,
               V_s_TO_sl=Vs_sl, V_LOF_sl=Vlof_sl, a_TO_sl=a_sl,
               s_ground_sl=sg_sl, s_air_sl_to=sair_sl, s_TO_sl=sTO_sl,
               rho_sl_TO=rho_sl_c,
               V_1=V1, a_OEI=a_oei, s_go_OEI=s_go, s_stop=s_stp, s_BF=s_bf,
               V_1_sl=V1_sl, a_OEI_sl=aoei_sl, s_go_OEI_sl=sgo_sl,
               s_stop_sl=sstp_sl, s_BF_sl=sbf_sl, mu_brake=mub, k_MCG=kmcg,
               s_ground=s_g, s_air=s_air, s_TO=s_TO, s_land=s_land, a_TO=a_TO,
               s_air_land=s_air_ld,
               C_Lmax_TO=CLmaxTO, C_Lmax_land=CLmaxLD, dC_D_flap_TO=dCD_to,
               dC_D_flap_land=dCD_ld, dC_D_gear=dCD_lg, mu_roll=mu, g=g,
               h_screen=h_scr, h_screen_land=h_scl, mu_eff=mu_e)

    cons = [
        # FAR 25.107 speeds, as EQUALITIES at their minimum compliant values.
        #
        # Written as >= they are only lower-bounded, and two of them are then
        # free to float upward: nothing in this model charges for a high V2, so
        # tying the engine-out control speed to it sent V2 to 98.8 m/s and the
        # fin shrank accordingly (fin force goes as V^2, so a faster control
        # speed buys a smaller fin for nothing). An operator does choose V2
        # above the minimum in practice, but that is a trade against field
        # length which this model does not represent, so the minimum compliant
        # speed is both the defensible and the conservative choice.
        V2 == rs.v2_factor * Vs_to,
        # FAR 25.149(c): V_MC may not exceed 1.13 V_SR. This is the speed at
        # which the fin has to hold the aircraft straight on one engine, and
        # it is the case that actually sizes a transport fin -- not the steady
        # engine-out balance at an arbitrary speed. Lower V_MC is harder (less
        # dynamic pressure over the fin), so the requirement is an upper bound
        # and the fin must cope at the worst permitted value.
        Vmc <= 1.13 * Vs_to,
        # FAR 25.149(e): V_MCG, the minimum control speed ON THE GROUND. This
        # is the case that actually sizes a transport fin, and it is harder
        # than the airborne one for two reasons: it happens at a LOWER speed,
        # so there is less dynamic pressure over the fin, and the aeroplane is
        # on its wheels, so the 5 degree bank that 25.149(b) allows in the air
        # -- which lets a sideslipping aircraft use its wing to help -- is not
        # available. The rudder does the whole job alone.
        #
        # Represented as a CALIBRATED FRACTION of the takeoff stall speed.
        # 25.149(e) defines V_MCG by a 30 ft lateral deviation with nosewheel
        # steering discounted -- a lateral-dynamics test rather than a flight
        # condition -- which needs a ground-handling model this study does not
        # have. It is also why no flight-phase speed reaches it: V_ref ~141,
        # V_1 ~145 and V_MC ~146 kt all sit far above a real V_MCG of ~113, so
        # deriving it from the balanced field would give a LARGER V_MCG and a
        # smaller fin, not a better one.
        #
        # This was previously `Vmcg == Vs_to`, a value with no aircraft behind
        # it. It sized the fin at 140 kt against a real 113, and since fin area
        # goes as 1/V^2 that under-sized it by about a third. k_MCG comes from
        # the SizeClass so it can differ by aircraft.
        Vmcg == kmcg * Vs_to,
        Vlof == 1.10 * Vs_to,
        Vref == rs.vref_factor * Vs_ld,
        # 25.109 / 25.125: demonstrated distance factored to field length.
        s_TO == rs.takeoff_field_factor * (s_g + s_air),      # [SP] SigEq
    ]

    if field_length_max_ft is not None:
        smax = C("s_TO_max", field_length_max_ft * 0.3048, "m",
                 "runway length available for takeoff")
        out["s_TO_max"] = smax
        cons += [s_TO <= smax]
    if landing_field_max_ft is not None:
        lmax = C("s_land_max", landing_field_max_ft * 0.3048, "m",
                 "runway length available for landing")
        out["s_land_max"] = lmax
        cons += [s_land <= lmax]

    return far, cons, out


def link(far, out, *, W_TO, W_land, S, rho_TO, T_TO, D_clean_TO, AR, e,
         n_eng, ruleset=None, cos_sweep=None, cos_sweep_ref=None,
         T_TO_sl=None, field_len_sl_ft=None):
    """Tie the certification model to the aircraft. Returns constraints.

    ``T_TO`` is TOTAL all-engine thrust at takeoff, ``D_clean_TO`` the clean
    drag at the takeoff condition; the configuration increments are added here.
    """
    rs = ruleset or out["_ruleset"]
    ne = int(n_eng)
    o = out
    Vs_to, Vs_ld = o["V_s_TO"], o["V_s_land"]
    V2, Vlof, Vref = o["V_2"], o["V_LOF"], o["V_ref"]
    s_g, s_air, s_TO, s_land, a = (o["s_ground"], o["s_air"], o["s_TO"],
                                   o["s_land"], o["a_TO"])
    CLmaxTO, CLmaxLD = o["C_Lmax_TO"], o["C_Lmax_land"]
    # SWEEP-CORRECTED MAXIMUM LIFT. A swept wing reaches a lower streamwise
    # C_Lmax than its section does, because only the velocity component normal
    # to the quarter chord does the lifting: C_Lmax ~ c_lmax * cos^2(Lambda).
    #
    # Without this, sweep is FREE in the model. The structural penalty is off
    # by default (sweep_pricing), and the one row that mentioned the effect --
    # `C_L_w_max * cos^2(Lambda) == 2.15` in aircraft.py -- is consumed by no
    # constraint at all, so nothing anywhere charged sweep for the stall speed
    # and field length it costs. Both takeoff and landing C_Lmax scale, since
    # both are flapped section values seen through the same sweep.
    #
    # Referenced to the sweep the constants were quoted at, so the calibration
    # is preserved: at cos_sweep == cos_sweep_ref the factor is exactly 1 and
    # C_Lmax_TO/land keep their 2.2/2.8. Only DEPARTURES from that sweep move.
    if cos_sweep is not None and cos_sweep_ref is not None:
        _swf = (cos_sweep / cos_sweep_ref) ** 2
        CLmaxTO = CLmaxTO * _swf
        CLmaxLD = CLmaxLD * _swf
    dCD_to, dCD_ld, dCD_lg, mu, g = (o["dC_D_flap_TO"], o["dC_D_flap_land"],
                                     o["dC_D_gear"], o["mu_roll"], o["g"])
    h_screen = o["h_screen"]
    s_air_land, h_screen_land = o["s_air_land"], o["h_screen_land"]
    mu_eff = o["mu_eff"]

    cons = [
        # Stall speeds in each configuration. EQUALITIES: a stall speed is a
        # defined quantity, V_s = sqrt(2W/(rho S C_Lmax)), not something with
        # slack.
        #
        # Written as >= it is bounded below only, and it pays to float upward:
        # V_MC = 1.13 V_s sets the dynamic pressure the fin works at, and fin
        # force goes as V^2, so a higher stall speed buys a smaller fin for
        # nothing. Measured, V_s_TO drifted to 77.1 m/s where the model's own
        # weight and area give 61.4 -- a 58% overstatement of the fin's
        # dynamic pressure.
        #
        # Same defect as `mac`, `l_ht`, `l_vt`, the tail-fits-on-fuselage row
        # and the 25.107 speeds: a quantity defined by an identity, written as
        # a one-sided bound, and promptly exploited.
        0.5 * rho_TO * Vs_to ** 2 * S * CLmaxTO == W_TO,      # [SP] SigEq
        0.5 * rho_TO * Vs_ld ** 2 * S * CLmaxLD == W_land,    # [SP] SigEq

        # ---- FAR 25.121(b) SECOND SEGMENT: the constraint that sizes thrust.
        # One engine inoperative, gear up, takeoff flaps, at V2. The remaining
        # engines must overcome drag AND climb at the required gradient.
        #
        # This is what was missing. With only a cruise-climb requirement the
        # engine came out three times too small, and everything sized off
        # thrust -- the vertical tail above all -- came out with it.
        (ne - 1) * T_TO / ne
            >= D_clean_TO + (dCD_to * 0.5 * rho_TO * V2 ** 2 * S)
             + W_TO * rs.second_segment[ne],

        # ---- FAR 25.121(a) FIRST SEGMENT: as above but gear DOWN.
        (ne - 1) * T_TO / ne
            >= D_clean_TO + ((dCD_to + dCD_lg) * 0.5 * rho_TO * Vlof ** 2 * S)
             + W_TO * rs.first_segment[ne],

        # ---- FAR 25.121(c) EN ROUTE: gear up, flaps up, OEI.
        (ne - 1) * T_TO / ne >= D_clean_TO + W_TO * rs.enroute[ne],

        # ---- FAR 25.119 LANDING CLIMB: all engines, landing flaps, at
        # landing weight. Sizes thrust on a light aircraft with big flaps.
        T_TO >= D_clean_TO + (dCD_ld * 0.5 * rho_TO * Vref ** 2 * S)
              + W_land * rs.landing_climb,

        # ---- FAR 25.121(d) APPROACH: OEI, approach flaps (half the landing
        # increment), at landing weight.
        (ne - 1) * T_TO / ne
            >= D_clean_TO + (0.5 * dCD_ld * 0.5 * rho_TO * Vref ** 2 * S)
             + W_land * rs.approach[ne],

        # ---- takeoff distance --------------------------------------------
        # Ground roll from a mean acceleration evaluated near 0.7*V_LOF, the
        # standard textbook reduction of the integral:
        #
        #     a = g/W * (T - D - mu(W - L))
        #     s_g = V_LOF^2 / (2 a)
        #
        # written as signomial equalities so both stay exact rather than being
        # relaxed to one side.
        # The classic textbook reduction: during the roll, rolling friction and
        # aerodynamic drag are lumped into one effective coefficient, so the
        # acceleration is g*(T/W - mu_eff). Using the mission's D[0] here --
        # which is CLIMB drag at altitude and speed -- was wrong and expensive:
        # it depressed mean acceleration to 1.3 m/s2 against a transport's
        # 2.0-2.5, which lengthened the ground roll, which drove the wing to
        # 165.9 m2 against a real 124.6 to make the field length.
        a * W_TO == g * (T_TO - mu_eff * W_TO),               # [SP] SigEq
        s_g * 2.0 * a == Vlof ** 2,                            # [SP] SigEq
        # Air distance from lift-off to the 35 ft screen, as a climb at the
        # second-segment gradient.
        s_air * rs.second_segment[ne] == h_screen,            # [SP] SigEq
        # Landing: the mirror image, decelerating from V_ref with braking.
        # Landing: air distance from the 50 ft screen plus a braked ground roll
        # at a mean 0.30 g, then factored by 1/0.6 per 25.125(b). The bare
        # ground-roll form (no air distance, 0.35 g) gave 3,278 ft against a
        # real 737-800's ~5,000 ft.
        s_land == rs.landing_field_factor
                  * (s_air_land + Vref ** 2 / (2.0 * 0.30 * g)),   # [SP] SigEq
        # Air distance from 50 ft at a 3 degree approach, flared.
        s_air_land * 0.0524 == h_screen_land,                      # [SP] SigEq
    ]

    # ---- BALANCED FIELD LENGTH, main takeoff condition ---------------------
    # See the declaration block in add_far for the model. Phase A's distance
    # V1^2/2a is a monomial in existing variables, so only the OEI and braking
    # phases need rows of their own. The braking row takes no reverser credit
    # (25.109(f)); the OEI roll reuses mu_eff, which slightly understates the
    # dead engine's windmill drag (TASOPT adds CDeng ~ 0.5*A_fan/S, worth
    # ~0.004 of effective mu -- a few percent of the go distance).
    V1, a_oei = o["V_1"], o["a_OEI"]
    s_go, s_stp, s_bf = o["s_go_OEI"], o["s_stop"], o["s_BF"]
    mub = o["mu_brake"]
    cons += [
        a_oei * W_TO == g * ((ne - 1) * T_TO / ne - mu_eff * W_TO),  # [SP] SigEq
        V2 ** 2 <= V1 ** 2 + 2.0 * a_oei * s_go,                     # [SP] SigIneq
        s_stp * 2.0 * mub * g == V1 ** 2,
        s_bf >= V1 ** 2 / (2.0 * a) + s_go,
        s_bf >= V1 ** 2 / (2.0 * a) + s_stp,
        # 25.107(a): V1 is bracketed by ground control below and lift-off
        # above. The lower tie is what finally makes V_MCG load-bearing.
        V1 >= o["V_MCG"],
        V1 <= Vlof,
    ]
    if "s_TO_max" in o:
        cons += [s_bf <= o["s_TO_max"]]
    # ---- the same takeoff, evaluated at SEA LEVEL --------------------------
    # Identical chain to the one above, at rho = 1.225 and with the engine's
    # UNLAPSED sea-level thrust. Only the takeoff distance is duplicated; the
    # climb gradients above stay at the hot-and-high condition, which is where
    # they are actually critical.
    if T_TO_sl is not None and field_len_sl_ft is not None:
        Vs_sl, Vlof_sl = o["V_s_TO_sl"], o["V_LOF_sl"]
        a_sl, sg_sl = o["a_TO_sl"], o["s_ground_sl"]
        sair_sl, sTO_sl = o["s_air_sl_to"], o["s_TO_sl"]
        rho_sl = o["rho_sl_TO"]
        smax_sl = far.Constant("s_TO_max_sl", field_len_sl_ft * 0.3048,
                               "m", "sea-level runway available")
        V1_sl, aoei_sl = o["V_1_sl"], o["a_OEI_sl"]
        sgo_sl, sstp_sl, sbf_sl = o["s_go_OEI_sl"], o["s_stop_sl"], o["s_BF_sl"]
        mub = o["mu_brake"]
        cons += [
            0.5 * rho_sl * Vs_sl ** 2 * S * CLmaxTO == W_TO,      # [SP] SigEq
            Vlof_sl == 1.10 * Vs_sl,
            a_sl * W_TO == g * (T_TO_sl - mu_eff * W_TO),         # [SP] SigEq
            sg_sl * 2.0 * a_sl == Vlof_sl ** 2,                   # [SP] SigEq
            sair_sl * rs.second_segment[ne] == h_screen,          # [SP] SigEq
            sTO_sl == rs.takeoff_field_factor * (sg_sl + sair_sl),  # [SP] SigEq
            sTO_sl <= smax_sl,
            # Balanced field at sea level, same three phases as the main case.
            # V2 and V_MCG at this density are monomial expressions of the
            # sea-level stall speed, so neither needs a variable of its own.
            aoei_sl * W_TO
                == g * ((ne - 1) * T_TO_sl / ne - mu_eff * W_TO),  # [SP] SigEq
            (rs.v2_factor * Vs_sl) ** 2
                <= V1_sl ** 2 + 2.0 * aoei_sl * sgo_sl,            # [SP] SigIneq
            sstp_sl * 2.0 * mub * g == V1_sl ** 2,
            sbf_sl >= V1_sl ** 2 / (2.0 * a_sl) + sgo_sl,
            sbf_sl >= V1_sl ** 2 / (2.0 * a_sl) + sstp_sl,
            V1_sl >= o["k_MCG"] * Vs_sl,
            V1_sl <= Vlof_sl,
            sbf_sl <= smax_sl,
        ]
    return cons
