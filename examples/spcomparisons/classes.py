"""The four size classes.

Payload is 215 lbf per passenger throughout -- SPaircraft's and TASOPT's
convention, which bundles the passenger with checked and carry-on baggage.
Using one convention across all four matters more than matching each type
certificate, because the whole point is comparing architectures at fixed
payload-range.

``R_fuse_guess`` is a *guess*, not a substitution. Fuselage radius is a free
variable in every case here -- see aircraft.py -- so these only seed the
solve. Seats abreast is a genuine input: it is an integer cabin-layout
decision that no continuous optimiser should be making.

Design Mach is NOT specified. Cruise Mach and altitude are free variables in
every case, bounded only by the transonic validity of the drag fits. The
Mach column below is the real aircraft's, recorded for comparison only.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SizeClass:
    key: str
    label: str
    n_pass: float
    seats_abreast: float
    range_nmi: float
    R_fuse_guess: float          # m, seed only
    ref_MTOW_lb: float           # the real aircraft, for comparison
    ref_OEW_lb: float
    ref_mach: float
    n_aisles: float
    ceiling_ft: float
    note: str
    # Engine deck and propulsor count, both of which have to scale with the
    # aircraft. The D8.2 cycle is a 180-passenger engine and the electric
    # powertrain defaulted to two 4.5 m fans, which is an 8-seater's
    # propulsion. Left fixed, the 787 and 737 columns were blocked by the
    # engine enthalpy rows and the actuator-disc momentum rows respectively --
    # both reported by the elastic Phase I, neither a physical result.
    engine: str = "D82_SPaircraft"
    n_fans: int = 2
    fan_d_max: float = 4.5
    #: Horizontal-tail volume floor, the real airframe's value -- TASOPT's own
    #: practice (its 737 deck PRESCRIBES Vh = 1.45; it derives no tail).
    #: None leaves the model's derived tail alone; set it for classes whose
    #: derived volume is known to under-shoot the airframe (the E175's
    #: certification/family margin puts its real Vh near 1.5 against a
    #: derived 1.12).
    v_ht_min: float | None = None
    #: MISSIONS: the payload-range points this class must fly, each one a
    #: LOWER BOUND on the shared design -- structure, tanks, engine and
    #: field performance all bind against their worst mission. None (the
    #: default) derives the single mission (n_pass*215 lbf, range_nmi),
    #: which is exactly the pre-multi-mission behaviour. State more corners
    #: as (payload_lbf, range_nmi) tuples; the 787 is the class that needs
    #: it (its full-cabin and long-range corners sit ~2,000 nmi apart).
    missions: tuple | None = None
    #: Cabin radius floor, m. The model's seat-pitch tube is a MINIMUM
    #: section; a class whose real aircraft is deliberately roomier (the
    #: E-jet double-dome is 3.0 m across against the 4-abreast minimum of
    #: 2.7) states that here.
    R_fuse_min: float | None = None
    # Engine-out decision speed, which sets the vertical tail. Pinned at the
    # D8.2's 70 m/s it gave a Citation and a 787 the same V1, which they
    # emphatically do not have.
    v1_mps: float = 70.0
    # ICAO aerodrome code gate box. 140 ft (42.7 m) was SPaircraft's single
    # number: too tight for a 787 (Code E, 65 m) and far too generous for a
    # business jet, so it silently sized the wing of both.
    span_max_m: float = 36.0
    # Certification basis. Transport-category aircraft certify under FAR 25;
    # a small business jet or commuter is a Part 23 aeroplane and its climb
    # gradient requirements are less demanding. See components/far.py.
    far_ruleset: str = "FAR25"
    # Runway available, feet. None leaves field length computed but not
    # limiting -- the OEI climb gradients still size thrust.
    field_length_ft: float | None = None
    # Minimum nose length: radome + avionics bay + flight deck. There is NO
    # regulatory minimum -- FAR 25 specifies no cockpit length -- so this is a
    # physical estimate: a two-crew transport flight deck is ~2.5 m of seated
    # space and the radome and avionics bay add ~1.5-2 m. 4.0 m matches the
    # real E175 and is not binding on the 737 (whose 1.2-calibre fineness rule
    # gives 4.45 m). It is a CLASS attribute because a light bizjet, single
    # pilot capable with a small radome, does not carry an airliner flight
    # deck, and a global 4.0 m would add ~2 m of nose to a 13 m aeroplane.
    # V_MCG as a fraction of the takeoff stall speed. FAR 25.149(e) defines
    # the ground minimum control speed by a 30 ft lateral deviation with
    # nosewheel steering discounted -- a lateral-dynamics test, not a flight
    # condition -- which is why it lands far BELOW every flight speed and why
    # no balanced-field or approach case reaches it. Representing it properly
    # needs a ground-handling model this study does not have.
    #
    # So it is a calibrated ratio, and it is honest about being one. 0.88 is
    # the 737-800: V_MCG ~113 kt against V_s (takeoff flaps) ~129. Transports
    # generally sit at 0.85-0.90. It was previously V_MCG == V_s_TO -- a value
    # with no aircraft behind it, which sized the fin at 140 kt instead of 113
    # and, since fin area goes as 1/V^2, under-sized it by roughly a third.
    #
    # Expected to differ by class: a light twin with a short moment arm and a
    # widebody with four engines do not share a rudder-authority margin.
    k_mcg: float = 0.88
    # MINIMUM NOSE-GEAR LOAD FRACTION, imposed at the AFT CG limit -- the
    # tip-back case, where the nose carries least. A CALIBRATION, not a rule,
    # and it needs to be per class because it selects the gear layout.
    #
    # 0.11 is chosen against the REAL 737: wheelbase 15.72 m against 15.3,
    # main gear 21.64 against 20.5, nose gear 5.92 against 5.2. The classic
    # design band is 8-15%, so it sits mid-band and is defensible on its own
    # terms too.
    #
    # RE-CALIBRATED. It was 0.10, chosen before cm_section was split into
    # cruise and landing values. That split raised the flaps-down wing
    # moment threefold, moved the CG envelope, and dropped the solution back
    # onto the long-wheelbase branch at 0.10 (nose gear 2.24 m, B 18.62).
    # The branch boundary now sits between 0.10 and 0.11.
    #
    # TASOPT is NOT the reference here, and deliberately so. Its own 737 puts
    # the main gear 1.00 ft behind its aft CG limit -- a 1.94% nose load, an
    # aeroplane that would sit on its tail -- because TASOPT has no nose-load
    # check anywhere, so nothing reconciles gear stations taken from the real
    # aircraft against a LOADABILITY-extreme aft CG the real aircraft is never
    # loaded to. Its gear stations cannot be a target for a model that does
    # check the load.
    #
    # Handle with care: the response is not smooth. Between 0.09 and 0.10 the
    # solution jumps between two branches -- a long-wheelbase layout with the
    # nose gear far forward (x_n 2.76, B 17.53) and a short one with it aft
    # (x_n 5.63, B 15.34) -- at almost equal cost, 1.045 against 1.049 MTOW.
    # Changing this constant selects a branch rather than nudging a trend, so
    # re-check the geometry after any change rather than interpolating.
    f_nose_load_min: float = 0.11
    # NOSE LENGTH, metres: nose tip to the front of the pressure shell. The
    # SECOND of TASOPT's two independent nose inputs -- xshell1 in the deck
    # (runs/737/737s.tas:294, 17.0 ft = 5.182 m), separate from and unrelated
    # to xlgnose below. TASOPT derives neither from the other, nor either from
    # fuselage radius.
    #
    # 4.0 was a placeholder default that only the Citation ever overrode, and
    # once the 1.2-calibre rule was removed it became the ONLY thing setting
    # nose length -- so the 737 was sizing its nose off a made-up number and
    # came out 1.18 m short.
    #
    # Only the 737 value is sourced; the rest await decks or drawings.
    l_nose_min_m: float = 5.182
    # NOSE GEAR STATION, metres aft of the nose tip. An INPUT, exactly as in
    # TASOPT, where xlgnose is a line in the deck (runs/737/737s.tas:340,
    # 14.0 ft = 4.267 m) and is derived from nothing.
    #
    # This replaces `lg.x_n >= fu.l_nose`, which required the nose gear to sit
    # at or behind the front of the pressure shell. TASOPT's own 737 breaks
    # that by 0.92 m -- xlgnose 4.267 against xshell1 5.182 -- because the
    # gear bay lives in the UNPRESSURISED nose section, ahead of the forward
    # bulkhead, which is where a nose gear actually goes. The constraint was
    # enforcing a relationship the reference aircraft does not satisfy, and
    # with the gear pinned to it the 15% nose-load rule dragged the whole nose
    # cone down onto whatever floor was underneath.
    #
    # ONLY THE 737 VALUE IS SOURCED. The others are placeholders pending decks
    # or drawings, and are flagged rather than silently scaled -- a nose gear
    # station is not a function of fuselage radius, which is the assumption
    # that produced the calibre rule this replaces.
    x_lg_nose_m: float = 4.267
    # Cabin length ABOVE the seat rows: galleys, lavatories, doors, exit rows.
    # Additive rather than a fraction of seat-row length, because these are
    # physically fixed objects -- a galley does not shrink when the cabin has
    # fewer rows. As 6.6% of seat rows it gave a Citation 0.21 m for its
    # galley, lavatory and doors, which is less than one lavatory.
    #
    # 1.61 m is TASOPT's 737: pressure shell 25.91 m against 24.30 m of seat
    # rows. That is ONE calibration point, and the true law is a fixed part
    # plus a part growing with passengers (lavatories run ~1 per 50 seats),
    # which two parameters cannot be fitted from one aircraft. So it is a
    # class attribute: 1.61 m is right for a 737, generous for a bizjet whose
    # galley is a cabinet, and probably light for a widebody.
    cabin_aux_m: float = 1.61
    # Reference speeds, all pinned at D8.2 values before: stall, landing and
    # the two structural design speeds. A Citation does not stall at the same
    # speed as a 787.
    v_stall_kt: float = 120.0
    v_land_mps: float = 72.0
    wing_load_max: float = 6664.0
    # "aluminium" or "composite". A 787's wing and empennage boxes are carbon;
    # everything else in this set is metal.
    box_material: str = "aluminium"


CLASSES = {
    "citation": SizeClass(
        key="citation", label="Cessna Citation X",
        n_pass=8, seats_abreast=2, range_nmi=3000,
        R_fuse_guess=0.86, ref_MTOW_lb=36_100, ref_OEW_lb=22_000,
        ref_mach=0.90,
        n_aisles=1, ceiling_ft=51000.0,
        span_max_m=24.0, field_length_ft=5300.0, v_stall_kt=95.0, v_land_mps=60.0, wing_load_max=4500.0,
        v1_mps=57.0, engine="CFM56", n_fans=2, fan_d_max=2.0,
        # Both ESTIMATES, not looked-up figures. A Citation X is 22 m long
        # with a long pointed radome and a two-seat flight deck, so ~3 m of
        # nose rather than an airliner's 4. Its "galley" is a forward
        # refreshment cabinet plus one aft lavatory, so ~1.2 m rather than a
        # 737's 1.61. Flagged because the alternative was letting a global
        # 4.0 m and 1.61 m size a 13-22 m aeroplane by default.
        l_nose_min_m=3.0, cabin_aux_m=1.2,
        note="Business jet. The hardest class for every alternative "
             "architecture: payload is tiny, so fixed masses (tank, stack, "
             "pack) are not amortised over anything."),
    "e175": SizeClass(
        key="e175", label="Embraer 175",
        # 88, not 76. The E175 airframe is certified for 88 seats single-class,
        # and every reference figure for it -- MTOW, length, wing area -- is
        # for that airframe. Sizing the model for 76 built a genuinely smaller
        # aeroplane and then compared it to the full-size one: fuselage came
        # out 17% short and MTOW 20% light, against the 737's 2% and 0.1%,
        # where the class was already configured near its real 189-seat max.
        # Moving to 88 closes most of it (MTOW 0.804 -> 0.910, l_fuse 0.832 ->
        # 0.914) and makes this class consistent with how b737 is set up.
        n_pass=88, seats_abreast=4, range_nmi=2200,
        R_fuse_guess=1.41, ref_MTOW_lb=85_517, ref_OEW_lb=48_100,
        # R_fuse_min IS class data (the E-jet double-dome is a published
        # 3.01 m section; the seat-pitch minimum tube is 2.71). v_ht_min is
        # deliberately NOT set: prescribing the real 26 m2 tail's volume is
        # not repeatable across classes -- see the tail-sizing note in
        # aircraft.py. The derived tail is the point-design answer; the real
        # E175's excess over it is E170/E175 COMMON-EMPENNAGE margin.
        R_fuse_min=1.50,
        ref_mach=0.78,
        n_aisles=1, ceiling_ft=41000.0,
        span_max_m=28.7, field_length_ft=6900.0, v_stall_kt=110.0, v_land_mps=66.0, wing_load_max=5500.0,
        v1_mps=67.0, engine="CFM56", n_fans=2, fan_d_max=3.0,
        note="Regional jet. Short enough that battery-electric is at least "
             "arguable, unlike the two larger classes."),
    "b737": SizeClass(
        key="b737", label="Boeing 737-800",
        n_pass=180, seats_abreast=6, range_nmi=3000,
        R_fuse_guess=1.88, ref_MTOW_lb=174_200, ref_OEW_lb=91_300,
        ref_mach=0.785,
        n_aisles=1, ceiling_ft=41000.0,
        span_max_m=36.0, field_length_ft=7700.0, v_stall_kt=120.0, v_land_mps=72.0, wing_load_max=6664.0,
        v1_mps=75.0, engine="TASOPT_737800", n_fans=4, fan_d_max=4.5,
        note="Single-aisle workhorse. Same 180 pax / 3000 nmi mission TASOPT "
             "and SPaircraft both use, so this class carries the verified "
             "reference point."),
    "b787": SizeClass(
        key="b787", label="Boeing 787-8",
        # 359 single-class (the exit-limit airframe), 9-abreast -- the same
        # convention correction the e175 class documents (76 -> 88): every
        # public reference figure is for the certified airframe, and sizing
        # the cabin for a two-class 242 built a fuselage 24% short and an
        # aeroplane 22% light. 8-abreast/242 was the pre-convention setting.
        # ... and the RANGE moves with the convention: 7,355 nmi is quoted
        # at 242 two-class -- a partial-payload corner of the payload-range
        # diagram -- and flying the full 359-seat cabin that far is not a
        # point the real airframe offers (the model dutifully sized a
        # 581,000 lb aircraft with more fuel than the real tanks hold).
        # 5,800 nmi is the ~full-cabin range off the published diagram, so
        # cabin and mission describe the same aeroplane. The structural fix
        # -- separate cabin count from mission payload so a class can state
        # BOTH corners -- is the right upgrade for the matrix, and matters
        # here more than anywhere because the 787's corners are 2,000 nmi
        # apart where the 737's nearly coincide.
        n_pass=359, seats_abreast=9, range_nmi=5800,
        # THE OTHER CORNER: 242 two-class seats (52,030 lbf at 215 lbf/seat)
        # over the quoted 7,355 nmi. As a secondary mission it is a lower
        # bound on the shared design -- it sizes the tanks and MTOW while
        # the full-cabin primary sizes the cabin and structure -- which is
        # the two-corner payload-range reality one mission cannot state.
        missions=((52_030.0, 7355.0),),
        R_fuse_guess=2.87, ref_MTOW_lb=502_500, ref_OEW_lb=264_500,
        ref_mach=0.85,
        n_aisles=2, ceiling_ft=43000.0,
        span_max_m=65.0, field_length_ft=10000.0, box_material="composite", v_stall_kt=135.0, v_land_mps=78.0, wing_load_max=7500.0,
        v1_mps=80.0, engine="GE90", n_fans=6, fan_d_max=5.5,
        note="Widebody, long haul. The range is what kills the alternatives: "
             "energy scales with it while the airframe does not."),
}

ORDER = ["citation", "e175", "b737", "b787"]
