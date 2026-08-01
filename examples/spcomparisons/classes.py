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
    l_nose_min_m: float = 4.0
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
        n_pass=242, seats_abreast=8, range_nmi=7355,
        R_fuse_guess=2.87, ref_MTOW_lb=502_500, ref_OEW_lb=264_500,
        ref_mach=0.85,
        n_aisles=2, ceiling_ft=43000.0,
        span_max_m=65.0, field_length_ft=10000.0, box_material="composite", v_stall_kt=135.0, v_land_mps=78.0, wing_load_max=7500.0,
        v1_mps=80.0, engine="GE90", n_fans=6, fan_d_max=5.5,
        note="Widebody, long haul. The range is what kills the alternatives: "
             "energy scales with it while the airframe does not."),
}

ORDER = ["citation", "e175", "b737", "b787"]
