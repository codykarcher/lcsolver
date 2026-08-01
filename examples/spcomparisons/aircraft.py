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

import os as _os
import numpy as np

#: Gear attachment station as a fraction of the wingbox at the gear's own
#: spanwise station: 0 = front spar, 1 = rear spar. Set GEAR_BOX_FRAC to
#: sweep it; see the note at the constraint itself.
_GEAR_BOX_FRAC = float(_os.environ.get("GEAR_BOX_FRAC", "0.0"))

#: Engine spanwise station, as a fraction of semi-span. Was 0.35, which is
#: about 17% too far outboard: TASOPT mounts the engine AT the planform break
#: (etas = 0.285 in its 737 deck, and surfw.f's inertial relief is written on
#: that assumption), and a real 737-800's engines sit at eta ~0.28-0.31.
#:
#: This is not cosmetic. y_eng sets the engine-out yawing moment, which sizes
#: the VERTICAL TAIL, so moving it inboard shortens the arm ~19% and shrinks
#: the fin. The fin currently agrees with the real aircraft to 1.8%, and that
#: agreement was reached with the engine in the wrong place -- so it has to be
#: re-earned here rather than assumed.
_ETA_ENG = 0.285

#: Let the engine station be a design variable instead of pinning it to
#: _ETA_ENG. Off by default: the pinned model is the one validated against the
#: 737, and freeing a variable is only safe once something resists it in both
#: directions -- here, the fin pulling inboard against the wing-bending relief
#: pulling outboard. Set FREE_Y_ENG=1 to try it.
_FREE_Y_ENG = bool(_os.environ.get("FREE_Y_ENG"))

from numpy import cos, pi, tan
from pyomo.environ import units

from edi import Formulation

from components.far import add_far, link as far_link
from components.flight_state import add_flight_state
from components.fuselage import add_fuselage
from components.horizontal_tail import add_horizontal_tail
from components.landing_gear import add_landing_gear
from components.vertical_tail import add_vertical_tail
from components.wing import add_wing
from components.wing_tasopt import add_wing_tasopt
from components.wingbox import ALUMINIUM, COMPOSITE
import components.technology as _technology
from components.technology import MODERN_COMPOSITE

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.turbofan.model import add_engine  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from components.cryo_tank import add_cryo_tank  # noqa: E402
from components.powertrain import add_powertrain  # noqa: E402
from components.fuel_cell import add_fuel_cell  # noqa: E402
from components.battery import add_battery  # noqa: E402
from components.electric import ElectricPropulsor  # noqa: E402

# Geometry, from subs/optimalD8.py.
SWEEP_W, SWEEP_VT, SWEEP_HT = 13.237, 25.0, 8.0
NCLIMB, NCRUISE = 3, 2


def build(size_class, arch, Nclimb: int = NCLIMB, Ncruise: int = NCRUISE,
          pi_tail_supports: str = "fixed", seed: str | None = None,
          tau_limits: bool = True, sweep_pricing: bool = False,
          # MSES refits are the default. York's fit was valid only to
          # M_perp ~0.74 and under-predicted drag 3-6x beyond it, so the model
          # carried a hard fence at that Mach -- and the FENCE, not
          # aerodynamics, was setting wing sweep (21 deg against TASOPT's 26.0
          # and a real ~25). mses_c is the like-for-like replacement, fitted to
          # the same C-series family and valid to M 0.86; mses tails replace
          # TASOPT's two Mach-independent constants with a fit that has a real
          # transonic rise. See components/polars.py.
          polar="mses_c", tail_drag: str = "mses",
          wing_model: str = "hoburg",
          sweep_deg: float | None = None):
    """Build the LH2 D8.2. Returns an EDI ``Formulation``.

    This is SPaircraft with the kerosene fuel system replaced by a liquid
    hydrogen one -- the same 1,172-variable airframe, trim chain, tails and
    landing gear, with four changes and nothing else:

    1. **The wing is dry.** ``f_wingfuel`` goes to zero, which removes the
       wing-tank volume constraint AND the fuel's bending relief in the root
       moment row. That relief is worth real wing weight, and losing it is a
       genuine structural penalty of hydrogen, not a modelling choice.
    2. **A cryogenic tank** (``SP_hydrogen_aircraft.cryo_tank``, derived from
       the TASOPT v3 port and verified against TASOPT.jl) sized by the fuel
       load, its radius fitted inside the fuselage.
    3. **The tank lengthens the fuselage.** ``l_shell`` grows by the tank
       length, so skin, insulation, floor, cone station and both bending
       distributions all follow through SPaircraft's own structure.
    4. **The engine burns hydrogen** -- the ``D82_LH2`` substitution set,
       identical to ``D82_SPaircraft`` except ``hf = 120`` MJ/kg.

    The fuel CG also moves: kerosene sat at the wing box and drained along
    ``dx_AC_wing``; LH2 sits in the fuselage tank aft of the cabin and does
    not shift the wing's aerodynamic centre.

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
    # Wing sweep is FREE -- see add_wing. The tails keep their D8.2 sweeps:
    # they are sized by volume coefficients and stability rather than by
    # cruise Mach, and their sweeps also appear in the pi-tail trailing-edge
    # geometry as plain trig constants, which would need the same treatment
    # for much less return.
    # Structural allowables come from the technology level, not from a
    # hard-coded material: validating a 1990s 737 against 2020s allowables is a
    # category error, and the 21% the caps were over-strength ran straight into
    # the MTOW gap. See components/technology.py.
    _tech = _technology.current()
    _mat = (MODERN_COMPOSITE.material()
            if size_class.box_material == "composite" else _tech.material())
    # ---- structural sweep pricing: OFF, and deliberately -------------------
    # TASOPT charges sweep in the box (surfw.f:106-107): web thickness as
    # 1/cos^2(L), cap sizing as 1/cos^4(L), against a box volume that carries a
    # compensating cos(L) (surfw.f:155-161). Net, cap weight rises about 34% at
    # 25 degrees. SPaircraft's box has none of it, so sweep is structurally
    # free here -- a real gap, and the port of it is written and switchable.
    #
    # It is off because the port over-corrects and I could not find out why
    # from the source alone. With it on, a 737 held at its design M 0.785 picks
    # a 3.9 degree wing; with only the drag-side fix it picks 15.7; the real
    # aircraft is about 25. The tell is that the objective barely cares --
    # 160,571 against 159,555, 0.6% -- so sweep is a nearly-flat direction and
    # its optimum moves a long way under small modelling errors. Settling it
    # needs TASOPT itself as the arbiter: run its 737 with iosweep active and
    # calibrate against what it picks. Until then, shipping a 3.9 degree
    # transport wing would be worse than the gap it closes.
    _cosL = (lambda w: w.cos_Lambda) if sweep_pricing else (lambda w: None)
    # sweep_deg=None leaves quarter-chord sweep a design variable (the default);
    # passing a number pins it, which is useful for like-for-like comparison
    # against TASOPT and the real aircraft while the polar refit is pending.
    # Two wing models, same group and same exposed names, so nothing
    # downstream knows which it has. "hoburg" is SPaircraft's single-taper
    # planform with the closed-form box; "tasopt" is the cranked planform --
    # two independent taper ratios, constant LE sweep, a trailing edge that
    # kinks at the break -- with the station-based box ported from surfw.f.
    # They are separate models rather than a switch inside one because the
    # geometry differs, not just the weight estimate: mac, the area integral
    # and the load distribution are all planform-dependent.
    _wing_add = {"hoburg": add_wing, "tasopt": add_wing_tasopt}[wing_model]
    wing, c = _wing_add(f, N, st, sweep_deg=sweep_deg, material=_mat,
                        sweep_pricing=sweep_pricing, polar=polar,
                        rho_fuel=70.0 if arch.fuel == "lh2" else 817.0)
    cons += c
    vt, c = add_vertical_tail(f, N, st, sweep_deg=SWEEP_VT, material=_mat,
                              tau_limits=tau_limits,
                              sweep_pricing=sweep_pricing,
                              drag_model=tail_drag); cons += c
    # HT sweep follows the wing, exactly as TASOPT's optimizer does it
    # (fobj.f writes igsweep and igsweeph from the same slot).
    ht, c = add_horizontal_tail(f, N, st,
                                cosL=_cosL(wing), tanL=wing.tan_Lambda,
                                pi_tail=arch.double_bubble,
                                material=_mat,
                                tau_limits=tau_limits,
                                drag_model=tail_drag); cons += c
    lg, c = add_landing_gear(f); cons += c
    # ---- certification performance ----------------------------------------
    # Takeoff sizing, which this model did not have at all. See
    # components/far.py for what it adds and what it approximates.
    _ne = int(size_class.n_fans) if arch.fuel == "electric" else 2
    far, c, farv = add_far(f, n_eng=_ne,
                           ruleset=getattr(size_class, "far_ruleset", "FAR25"),
                           field_length_max_ft=getattr(size_class,
                                                       "field_length_ft", None),
                           k_mcg=getattr(size_class, "k_mcg", 0.88))
    cons += c
    # The tank is sized first so the fuselage knows how much shell to add.
    # Its parameters are TASOPT's cryo case: 2 atm vent pressure and the 1.3
    # structural heat-leak factor.
    # ftankadd=0.35 and the 10 cm clearance are TASOPT's own parameters,
    # calibrated in add_cryo_tank's docstring against their documented LH2
    # turbofan. Without them this tank reads gravimetric 0.81 -- *better*
    # than TASOPT's 0.738 while holding a third of the fuel, which is
    # backwards, since a smaller tank has the worse square-cube ratio.
    # Written once with a units-correct zero stand-in rather than branching
    # the constraint list: an architecture without a tank contributes no tank
    # weight and no boil-off, which is exactly a zero term, not a missing row.
    tank = None
    if arch.cryo_tank:
        tank, c = add_cryo_tank(f, prefix="Tank_", pvent=2.0265e5, qfac=1.3,
                                ftankadd=0.35)
        cons += c
    # Gear dimensions scale against the 180-passenger reference (R_fuse 1.88 m)
    # that subs/optimalD8.py was written for.
    _gear_scale = size_class.R_fuse_guess / 1.88
    SPR_val = size_class.seats_abreast
    fu, c = add_fuselage(f, cabin_aux_m=size_class.cabin_aux_m,
                         l_tank=(tank.l_tank if tank else None),
                         SPR=SPR_val); cons += c
    electric = arch.fuel == "electric"
    fc = batt = pt = None
    if electric:
        # The propulsor is sized against the ACTUAL per-segment flight state,
        # not a fixed cruise point -- with altitude and Mach free, a disc
        # sized at one (u, rho) is sized for a condition nobody has to fly.
        pt, c = add_powertrain(f, N, prefix="PT_",
                               n_fans=size_class.n_fans, state=st,
                               d_max=size_class.fan_d_max)
        cons += c
        eng = ElectricPropulsor(f, N, pt, n_eng=float(size_class.n_fans))
        cons += eng.cons
        if arch.fuel_cell:
            fc, c = add_fuel_cell(f, N, prefix="FC_"); cons += c
        if arch.battery:
            batt, c = add_battery(f, prefix="Batt_"); cons += c
    else:
        # The D8 keeps its own cycle -- it is part of what makes it a D8.
        # Every other architecture takes the deck that suits the size class,
        # with the hydrogen twin when the fuel is hydrogen.
        if arch.double_bubble:
            eng_key = "D82_LH2" if arch.fuel == "lh2" else "D82_SPaircraft"
        else:
            eng_key = size_class.engine + ("_LH2" if arch.fuel == "lh2" else "")
        eng, c = add_engine(f, N, st, engine=eng_key, BLI=arch.BLI,
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
    # Propulsor count. A BUG until now for the electric architectures: the
    # shim was told n_eng = size_class.n_fans (4 on a 737, 6 on a 787) so it
    # divided the powertrain's total thrust by that, while the airframe here
    # still multiplied by 2. A four-fan 737 was therefore flying on half the
    # thrust it had paid for -- and the elastic Phase I duly reported the
    # actuator-disc momentum rows as the blockers, which I read as a fan-size
    # limit when the arithmetic was simply wrong.
    numeng = C("n_eng", float(size_class.n_fans) if electric else 2.0, "-",
               "number of propulsors")
    # TWO vertical tails is a D8 feature, not a universal one. Left at 2 for
    # every architecture -- as SPaircraft has it, since optimalD8 is all it
    # builds -- every "conventional" aeroplane here carried the D8's twin
    # fins, which is neither a 737 nor a 787. It changes fin sizing (one fin
    # doing all the work rather than two sharing it), tail weight, and the
    # cone bending that reacts the fin root moment.
    #
    # NOT fixed with it: the horizontal keeps the pi-tail structural model,
    # `b_ht_out == 0.5*b_ht - w_fuse`, which describes a horizontal spanning
    # BETWEEN two fins and is the wrong support condition on a single-fin
    # aircraft. Left deliberately -- the horizontal is about 1% of OEW, so
    # that error is far smaller than the fin-count one, and a
    # half-implemented cantilever tail would be worse than a stated limit.
    numVT = C("n_vt", 2.0 if arch.double_bubble else 1.0, "-",
              "number of vertical tails")
    # Class-driven. The existing cabin-width row -- seats, aisles, systems
    # width and the double-bubble web against 2*w_fuse -- already keeps the
    # radius honest while leaving it free; it only ever assumed the D8's two
    # aisles. A single-aisle 737 or Citation needs its own count.
    numaisle = C("n_aisle", size_class.n_aisles, "-", "number of aisles")
    Vne = C("V_ne", 143.92, "m/s", "never-exceed speed")
    rhoTO = C("rho_TO_ac", 1.225, "kg/m^3", "air density at takeoff")
    ReserveFraction = C("f_fuel_res", 0.20, "-", "fuel reserve fraction")
    # Zero: the hydrogen is in the fuselage. Small but nonzero keeps the
    # row well posed without giving the wing any meaningful relief.
    f_wingfuel = C("f_wingfuel", 0.5 if arch.wet_wing else 1e-6, "-",
                   "fraction of fuel in wing tanks")
    FuelFrac = C("FuelFrac", 0.9, "-", "usable fraction of max fuel volume")
    SMmin = C("SM_min", 0.01, "-", "minimum static margin")
    # CG travel, as a FRACTION OF MEAN AERODYNAMIC CHORD -- which is how CG
    # envelopes are actually quoted, and the only form in which one number can
    # be shared across a business jet and a widebody.
    #
    # SPaircraft pins the travel itself at 6 ft. On the D8.2's 3.8 m chord
    # that is ~48% MAC, already wide against the 15-30% transport convention;
    # carried across to a Citation's 2.2 m chord it becomes 83% MAC, which is
    # not an aircraft. And the term is not incidental: dx_CG/mac supplies
    # 34-83% of the horizontal tail sizing requirement, so pinning it in
    # METRES was quietly sizing every small aircraft's tail by an inherited
    # D8 dimension.
    #
    # It cannot be derived instead. The model tracks x_CG per segment, so the
    # FUEL-BURN excursion is computable -- but a certification CG envelope is
    # dominated by loading (empty vs full, forward vs aft cargo) and this
    # model carries a single payload at a fixed centroid. Deriving it would
    # replace the envelope with one small component of it.
    dxCG = V("dx_CG", 1.0, "m", "max CG travel range")
    # Cruise Mach is a DESIGN VARIABLE here, not a specification. SPaircraft
    # pins it near 0.72 because optimalD8 is a fixed design point; with the
    # airframe free, the Mach that minimises fuel is an answer, not an input.
    # The floor is only the lower edge of where the transonic drag fits were
    # trained -- below it the model is extrapolating, not optimising.
    Mmin = C("M_min", 0.45, "-", "lower edge of the drag fits validity")
    # Fuselage lift, as a fraction of wing lift. 1.195 says the hull carries
    # nearly 20% of the wing's lift, which is a DOUBLE-BUBBLE result: the D8's
    # wide flat-sided fuselage is a lifting body and SPaircraft's number
    # reflects that. A conventional circular tube does nothing of the kind, so
    # applying 1.195 to it hands every "conventional" aeroplane a fifth of its
    # lift for free. 1.02 is the residual a round fuselage actually carries.
    Ltow = C("f_L_total_wing", 1.195 if arch.double_bubble else 1.02, "-",
             "total lift as a fraction of wing lift")
    fhpesys = C("f_hpesys", 0.01, "-", "power systems weight fraction")
    rSnace = C("r_S_nacelle", 6.0, "-", "nacelle and pylon wetted area factor")
    rvnace = C("r_v_nacelle", 0.925, "-", "incoming nacelle velocity ratio")
    fpylon = C("f_pylon", 0.05, "-", "pylon weight fraction")
    feadd = C("f_eadd", 0.1, "-", "additional engine weight fraction")
    Ceng = C("C_engsys", 1.0, "-", "engine system weight margin")
    Dreduct = C("D_reduct", 0.98416 if arch.BLI else 1.0, "-",
                "BLI drag reduction factor")
    bmax = C("b_max", size_class.span_max_m, "m", "gate-box span limit")
    CLwmax = V("C_L_w_max", 2.15 / cos(SWEEP_W * pi / 180) ** 2, "-",
               "max wing lift coefficient")
    Fsafetyfac = C("Fsafetyfac", 1.0, "-", "safety factor on initial climb thrust")
    # SPaircraft fixes this at 38,478 ft, which is right for a 3,000 nmi
    # airliner and absurd for a 100 nmi one: at 2,500 ft/min the climb alone
    # would eat most of the mission, so a short-range aircraft would come out
    # infeasible for reasons that have nothing to do with its propulsion.
    # Short sectors genuinely cruise lower. The floor is eased below ~800 nmi
    # and left exactly as SPaircraft has it above, so no case in the main
    # matrix is affected -- this only opens up the battery range sweep.
    _min_crz = 38478.0 if size_class.range_nmi >= 800 else max(
        12000.0, 38478.0 * size_class.range_nmi / 800.0)
    MinCruiseAlt = C("MinCruiseAlt", _min_crz, "ft", "minimum cruise altitude")
    # MaxClimbTime is GONE. It was an arbitrary 16 minutes applied to every
    # class, and it is not what regulation asks for -- FAR 25 constrains climb
    # GRADIENT in the one-engine-inoperative configurations, not the clock.
    # The model already carries the physical requirements:
    #     RC[0]              >= 2500 ft/min   initial climb rate
    #     RC[1:Nclimb]       >= 500 ft/min    minimum through the climb
    #     theta[Nclimb-1]    >= 0.015         1.5% gradient at top of climb
    # The last of those is the FAR-shaped one (25.121(b) asks 2.4% OEI for a
    # twin at takeoff; 1.5% en-route is the cruise-ceiling analogue), and it
    # is what should bound the climb. A wall-clock cap on top of it just
    # penalised the classes whose optimum climbs slowly.
    CDwm = C("C_D_wm", 0.5, "-", "windmill drag coefficient")
    Vland = C("V_land", size_class.v_land_mps, "m/s", "aircraft landing speed")
    rreq = C("rdot_req", 0.1475, "s^-2", "required yaw rate at landing")
    ReqRng = C("R_req", size_class.range_nmi, "nmi", "required cruise range")
    # Structural design speeds, scaled on stall speed against the
    # 180-passenger reference. Pinned, they gave a Citation and a 787 the
    # same 260 kt manoeuvre and 280 kt never-exceed speeds, which then
    # size the vertical tail and the fuselage bending cases.
    _v_scale = size_class.v_stall_kt / 120.0
    Vmn = C("V_mn", 133.94 * _v_scale, "m/s", "manoeuvring speed")
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

    Vstall = C("V_stall", size_class.v_stall_kt, "knots", "stall speed")
    WLoadmax = C("W_Load_max", size_class.wing_load_max, "N/m^2", "max wing loading")
    Pcabin = C("P_cabin", 75000.0, "Pa", "cabin air pressure")
    Tcabin = C("T_cabin", 297.0, "K", "cabin air temperature")
    minRC = C("RC_min", 500.0, "feet/min", "minimum rate of climb")
    # Fuselage drag, CALIBRATED ON TASOPT'S BOUNDARY LAYER SOLUTION.
    #
    # Where 0.018081 came from: not TASOPT. TASOPT has no fuselage drag
    # coefficient at all -- fusebl.f solves the actual flow, a compressible
    # source-line potential model coupled to an axisymmetric integral boundary
    # layer (Drela, doc/axibl.pdf), strong enough to capture separation. The
    # 0.018081 is SPaircraft's own calibration and has no TASOPT provenance.
    #
    # So rather than pick between that constant and a teaching correlation,
    # this is fitted to the real thing: 19 fusebl solutions swept over
    # l = 27-52 m and d = 3.0-5.8 m at M 0.80, 35 kft, giving the profile
    # drag AREA directly,
    #
    #     DA_profile = 0.0077851 * l^0.9062 * d^1.1036   [m^2],  max err 2.2%
    #
    # which is a monomial and therefore exactly GP-representable.
    #
    # SPaircraft's form is DA ~ C_D_fuse * l * R, i.e. exponents (1.0, 1.0).
    # The BL says (0.906, 1.104): it over-weights length and under-weights
    # diameter. That is not academic here -- hydrogen stretches fuselages
    # ~30% at fixed diameter, and the two forms disagree on what that costs.
    k_fdrag = C("k_fuse_drag", 0.0077851, "m^-0.0098",
                "TASOPT fusebl fit coefficient")
    DAfuse = V("DA_fuse", 1.0, "m^2", "fuselage profile drag area")
    tankclear = C("tank_clearance", 0.10, "m",
                  "radial gap between tank insulation and fuselage wall")
    # Wing pitching moment, PORTED FROM TASOPT's surfcm.f rather than pinned.
    #
    # SPaircraft carries c_m_w = 1.9. TASOPT's own value is CMw0 = -0.09 (see
    # drv_balance.f), and it does not even specify that: surfcm.f COMPUTES the
    # moment from the planform, as CM = CM0 + CM1*(CL - CLh). 1.9 is 20-40x
    # the physical magnitude, and it was supplying ~74% of the horizontal tail
    # sizing requirement -- which is how two arbitrary numbers came to fix
    # every tail in this study.
    #
    # It is negative, which is why a GP cannot carry it directly and why 1.9
    # looks like a shifted variable (1.9 - 2.0 = -0.1) whose offset the
    # constraints then forgot to remove. The tail-sizing row wants the
    # MAGNITUDE, which is what is built here.
    #
    # For an unbroken wing (etas = etao, lambdas = 1) surfcm collapses to
    #     CM0 = (cos^4(L)/Kc) * C3/12,  Kc = etao + (1+lam)(1-etao)/2
    #     C3  = 4 * cm_section * (1 + lam + lam^2) * (1 - etao)
    # so the moment falls as cos^4 of sweep -- which now matters, because
    # sweep is a design variable here rather than a fixed 13.237 degrees.
    # CG envelope, derived rather than assumed -- TASOPT's cglpay.
    xCGe = V("x_CG_empty", 17.0, "m", "empty-aircraft centre of gravity")
    xCGfwd = V("x_CG_fwd", 16.0, "m", "most-forward CG, worst payload loading")
    xCGaft = V("x_CG_aft", 20.0, "m", "most-aft CG, worst payload loading")
    # The payload fractions that PRODUCE those two extremes. cglpay solves a
    # quadratic for each; here they are variables and the quadratic is carried
    # by the stationarity condition below, which is what makes it SP-legal.
    rpayF = V("r_pay_fwd", 0.45, "", "payload fraction giving the fwd CG limit")
    rpayB = V("r_pay_aft", 0.45, "", "payload fraction giving the aft CG limit")
    # TASOPT's cglpay, run on its own 737-class case, gives rpayF/rpayB =
    # 0.458/0.456 with the partial payload centred at xcabin -/+ 0.271*lcabin,
    # and a resulting CG travel of 2.22 m = 53% MAC.
    #
    # That last number matters: it VALIDATES SPaircraft's 6 ft (~48% MAC),
    # which I had cut to 25% on the grounds that transport CG envelopes run
    # 15-30% MAC. They do not, for this quantity, and halving it shrank every
    # tail in the study. Corrected to TASOPT's own loading geometry.
    r_pay_lim = C("r_pay_limit", 0.457, "-",
                  "payload fraction loaded at each CG limit (TASOPT cglpay)")
    x_pay_off = C("x_pay_offset", 0.271, "-",
                  "partial-payload centroid offset, fraction of cabin length")
    # Fuselage pitching moment -- TASOPT's CMVf1, ported.
    #
    # A fuselage is DESTABILISING in pitch (the slender-body/Munk moment), and
    # TASOPT carries it as a moment volume that shifts the neutral point
    # forward:  xNP = ... - CMVf1/S.  Its driver uses CMVf1 = 60 m3, which on
    # a 125 m2 wing is 0.48 m of forward shift -- about 12% of MAC.
    #
    # SPaircraft has no equivalent term at all, so every aircraft in it gets
    # pitch stability free from a hull that should be taking it away, and the
    # tail comes out smaller than it should. This is the most likely reason
    # the ported htsize still lands at V_ht ~0.44 against a real 737's ~1.1.
    #
    # Scaled on fuselage volume, since that is what the slender-body result
    # depends on: k_Mf calibrated so a 737-class hull (R 1.88 m, l 38 m)
    # reproduces TASOPT's 60 m3.
    k_Mf = C("k_M_fuse", 0.1422, "-",
             "fuselage moment volume / hull volume; set so a 737-class\n              hull (R 1.88 m, l 38 m) gives TASOPT's CMVf1 = 60 m3")
    CMVf1 = V("CM_V_f1", 60.0, "m^3", "fuselage pitching moment volume")
    dxNPf = V("dx_NP_fuse", 0.4, "m", "forward NP shift from the fuselage")
    # TASOPT's own trim-case constants (tests/test_balance.py GEOM/AERO,
    # which reproduce the compiled Fortran).
    # Tail lift coefficient available at rotation, with elevator deflected. A
    # horizontal with a full-span elevator reaches about 1.0-1.2 before it
    # stalls; 1.0 is the conservative end and leaves control margin.
    # Raymer's tail volume coefficient for a jet transport (Aircraft Design: A
    # Conceptual Approach, Table 6.4): V_HT = 1.00. The real 737-800 sits at
    # ~0.98, so Raymer and the aeroplane agree; TASOPT's deck uses 1.45, which
    # is conservative and, imposed here, distorts the layout -- it pushes the
    # wing aft and stretches the nose to 10.4 m to find the moment arm.
    V_HT_FLOOR = float(_os.environ.get("V_HT_FLOOR", 0.01))
    # Minimum nose-gear load fraction at the AFT CG. Per class, because it
    # selects the gear layout rather than nudging it. See
    # SizeClass.f_nose_load_min for the calibration and its branch behaviour.
    f_nose_min = getattr(size_class, "f_nose_load_min", 0.10)
    CLhrot = C("C_L_h_rotate", 1.25, "-",
               "HT lift coefficient available at takeoff rotation")
    # Wing C_L at the GROUND attitude with takeoff flaps -- the aeroplane is
    # still on its wheels at V_R, sitting at whatever incidence the gear gives
    # it, so this is far below C_Lmax_TO (2.0). It is what decides how much
    # weight the wing has already taken off the gear before the tail has to
    # lift the nose, and the old rotation row assumed it was zero.
    CLgnd = C("C_L_ground_TO", 0.9, "-",
              "wing C_L at the ground attitude, takeoff flaps")
    Wrot = V("W_rot", 8.0e4, "lbf", "weight still on the gear at rotation")
    CLhfwd = C("C_L_h_CGfwd", 0.65, "-",
               "|C_Lh| download available at the forward-CG trim case")
    CLpmax = C("C_L_p_max", 1.25, "-",
               "wing C_L for the forward-CG trim case (TASOPT CLpmax)")
    # |dCM/dCL| of the wing about its box axis. NEGATIVE in TASOPT; carried
    # here as a MAGNITUDE, with every appearance signed explicitly at its use.
    #
    # This was 0.015, a placeholder: surfcm.f was ported for CM0 (that is
    # c_m_w_val, with its eta_surfcm/Kc_surfcm rows) and CM1 was left as a
    # constant. But CM1 is the term that carries the box-to-aerodynamic-centre
    # offset, and it is dominated by SWEEP. Evaluating surfcm.f exactly on
    # TASOPT's own 737 deck (Xaxis 0.40, etao 0.1016, etas 0.285, lambdas 0.70,
    # lambdat 0.25, AR 10.1, sweep 26 deg, fLo -0.3, fLt -0.05, and the
    # forward-CG case rcls 1.1 / rclt 0.5):
    #
    #     etao*(1+fLo)*(Xaxis-0.25)      = +0.010672
    #     (Xaxis-0.25)*cosL^2*C1/3       = +0.037354
    #     -(tanL/Ko)*C2/12               = -0.235040     <- sweep, dominant
    #     tip-rolloff term               = +0.000669
    #     / Kp = 0.55287                 -> CM1 = -0.33705
    #
    # So the true value is 22x larger AND of the opposite sign. co*CM1 is
    # -1.98 m on TASOPT's 737 where our 0.015 gave +0.098 m, and since the
    # trim row solves for (x_wing - co*CMw1), that 2.08 m error went straight
    # into the wing station: the box sat at 19.94 m against TASOPT's 16.36.
    #
    # 0.337 is the 737 deck's value, so this is still a CONSTANT and still
    # only right for a 26-degree wing. Porting surfcm's CM1 properly -- it is
    # a function of etao, etas, the two tapers, the two lift-taper ratios and
    # sweep, all of which already exist as variables -- is the next step, and
    # is what the matrix will need. The cruise case gives -0.37782 on the same
    # deck, so the spread across mission points is about 12%.
    # Now a VARIABLE, computed by the surfcm CM1 rows further down rather than
    # pinned. The constant it replaces was 0.337, TASOPT's 737-deck value; our
    # own unbroken planform at 25 degrees evaluates to 0.3694, so even for the
    # aircraft it was calibrated on it was 9% off, and it would not have
    # travelled to the D8, the 787 or the Citation at all -- CM1's dominant
    # term is -(tanL/Ko)C2/12, so it scales with sweep and aspect ratio.
    CMw1 = V("C_M_w1", 0.37, "-", "|wing dCM/dCL| about the box axis "
             "(NEGATIVE; sign applied at each use), surfcm.f CM1")
    # surfcm CM1 workspace. gam_s = lambdas*rcls with lambdas = 1 for an
    # unbroken wing, so it is just rcls and stays a constant; gam_t follows
    # the taper and is therefore a variable.
    gam_t = V("gamma_t_cm", 0.125, "-", "surfcm tip lift-taper, lambda_t*rclt")
    C1_cm = V("C1_surfcm", 1.19, "-", "surfcm C1")
    C2_cm = V("C2_surfcm", 1.09, "-", "surfcm C2")
    Kp_cm = V("Kp_surfcm", 0.62, "-", "surfcm load-weighted span factor")
    # TASOPT 737 deck (runs/737/737s.tas): Xaxis 0.40 spar box axis x/c,
    # fLo -0.3 fuselage lift carryover loss, fLt -0.05 tip lift rolloff, and
    # the FORWARD-CG TAIL SIZING case rcls 1.1 / rclt 0.5 -- which is the case
    # this trim row is, so those are the right two of the three mission sets.
    Xax_m = C("X_axis_m_qc", 0.40 - 0.25, "-", "spar box axis aft of c/4, x/c")
    fLo_cm = C("f_L_o", -0.3, "-", "fuselage lift carryover loss factor")
    fLt_a = C("f_L_t_abs", 0.05, "-", "|tip lift rolloff factor|")
    gam_s = C("gamma_s_cm", 1.1, "-", "surfcm break lift-taper = rcls "
              "(lambdas = 1, unbroken wing)")
    rclt_cm = C("rclt", 0.5, "-", "tip/root cl ratio, forward-CG case")
    CMh0 = C("C_M_h0", 0.02, "-", "|tail zero-lift moment|")
    CMh1 = C("C_M_h1", 0.30, "-", "|tail dCM/dCL|")
    CLMf0 = C("C_L_Mf0", 0.185, "-", "C_L at which the fuselage moment is zero")
    dCLhdCL = C("dCLh_dCL", 0.55, "-", "tail lift response to aircraft C_L")
    xNPt = V("x_NP_tasopt", 15.0, "m", "neutral point, TASOPT balance.f form")
    # SECTION PITCHING MOMENT, |c_m|, BY MISSION CASE.
    #
    # TASOPT carries three (runs/737/737s.tas:186-203): -0.20 takeoff and
    # initial climb, -0.06 clean climb/cruise/descent, and -0.35 for the
    # landing, forward-CG tail sizing case. A wing with flaps down has a far
    # larger nose-down moment, and that is what the tail has to balance.
    #
    # We had ONE, 0.12, feeding both the per-segment cruise relation and the
    # forward-CG landing trim row. It cannot serve both: they are a factor of
    # six apart. 0.12 was a compromise, so the cruise case was twice too
    # strong and the landing case three times too weak.
    cm_sec = C("c_m_section", 0.12, "-",
               "section |c_m|, clean/cruise. Kept at 0.12 rather than "
               "TASOPT's 0.06: this is a supercritical section and the "
               "model's own cruise relation is calibrated against it.")
    cm_sec_land = C("c_m_section_land", 0.35, "-",
                    "section |c_m|, landing with flaps deployed -- TASOPT's "
                    "forward-CG tail sizing case, 737s.tas:201")
    eta_surf = V("eta_surfcm", 0.1, "-", "centrebody fraction of span, bo/b")
    Kc_cm = V("Kc_surfcm", 0.67, "-", "surfcm chord-weighted span factor")
    cmw = V("c_m_w_val", 0.05, "-", "wing pitching moment magnitude |CM0|")
    cmw_land = V("c_m_w_land", 0.15, "-",
                 "wing |CM0| with flaps down, forward-CG tail sizing case")

    # OBJECTIVE. Normally fuel burn; PUSH_XM swaps in 1/x_m so the solver
    # maximises the main gear station instead. That turns "how far aft can the
    # gear go, and what stops it" from a question about slacks -- whose duals
    # in an L1 feasibility solve are degenerate, all ~0.35 -- into an ordinary
    # optimisation whose duals rank the blockers exactly.
    # PUSH generalises that to any geometry variable, either direction:
    #
    #     PUSH=x_wing:fwd     how far FORWARD can the wing go, and what stops it
    #     PUSH=x_wing:aft     how far AFT
    #     PUSH=x_m:aft        the original gear question
    #
    # The answer is the active set at the optimum, ranked by multiplier. That
    # is a question about THIS model rather than about the physics, which is
    # the point: a limit nobody wrote is a limit worth finding.
    _PUSH = _os.environ.get("PUSH") or ("x_m:aft" if _os.environ.get("PUSH_XM")
                                        else "")
    if _PUSH:
        _tgt, _, _dir = _PUSH.partition(":")
        _pv = {"x_m": lg.x_m, "x_n": lg.x_n, "x_wing": fu.x_wing,
               "y_m": lg.y_m}[_tgt]
        # Minimising the variable drives it forward; minimising its reciprocal
        # drives it aft. Both are monomial, so neither disturbs the structure.
        f.Objective(_pv if _dir == "fwd" else 1.0 / _pv)
    else:
        f.Objective(W_ftotal)

    def _dry_buildup():
        """Everything that is not fuel and not payload."""
        tot = (fu.W_fuse + numeng * Wengsys + fu.W_tail + wing.W_wing
               + Wmisc)
        if tank is not None:
            tot = tot + tank.W_tank
        return _propulsion_extra(tot)

    def _propulsion_extra(base):
        """Stack or pack weight, ADDED to an existing term rather than started
        from zero. ``0.0 * W_dry`` is not a posynomial -- log(0) does not
        exist -- and building a sum that way makes the whole model fail
        structure detection with "neither a GP nor an SP". Terms that do not
        apply are omitted, not zeroed."""
        if fc is not None:
            base = base + fc.W_stack
        if batt is not None:
            base = base + batt.W_batt
        return base

    def _burn_row():
        """What the aircraft consumes per segment -- and it is not one thing.

        A turbofan burns fuel with THRUST. A fuel cell burns hydrogen with
        POWER. A battery burns nothing and does not get lighter, which is why
        it returns a zero-weight row rather than a small one: the weight
        decrement has to actually vanish, not merely shrink.
        """
        if not electric:
            row = numeng * eng.TSFC * thr * eng.F
            return row + g * tank.m_boil * thr if tank else row
        if fc is not None:
            row = g * fc.mdot_H2 * thr
            return row + g * tank.m_boil * thr if tank else row
        # Battery: nothing is consumed and the aircraft never gets lighter.
        # W_burn is left to its own lower bound and the decrement row
        # W_start >= W_end + W_burn goes slack, which is exactly right.
        return None

    W_tank_term = tank.W_tank if tank else None


    # ---- variable linking ---------------------------------------------------
    cons += [
        wing.c_root == fu.c_0,
        wing.x_w == fu.x_wing,
        fu.N_lift == wing.box.N_lift,
        Ltow * wing.L_max >= wing.box.N_lift * W_totalmax + ht.L_ht_max,
        wing.b <= bmax,

        # ---- weight build-up -------------------------------------------------
        # Assembled term by term so that an architecture without a tank, a
        # stack or a pack simply does not contribute one. Adding a zero
        # instead would be a zero monomial, which is not a posynomial, and
        # takes the whole model out of GP/SP form.
        _dry_buildup() <= W_dry,
        # The tank carries the whole fuel load, reserves included, and must
        # fit inside the fuselage with its insulation.

        # 10 cm of radial clearance between insulation and the fuselage
        # inner wall, for frames, mounts and the vapour line.

        W_ftotal + W_dry + fu.W_payload <= W_total,
        W_ftotal >= W_fprimary + ReserveFraction * W_fprimary,
        W_fprimary >= W_fclimb + W_fcruise,
        W_totalmax >= W_total,

        # ---- landing gear: empirical floor from TASOPT's own deck ----------
        # The structural model in landing_gear.py sizes a thin-walled steel
        # tube against axial yield AND Euler buckling -- both are there and
        # both are right -- but a bare tube is all it is. The oleo, trunnion,
        # side and drag braces, actuation, doors and bay structure are folded
        # into f_add = 1.5, and that is not enough: the gear came out at 3,123
        # lbf, 2.0% of MTOW, where transport gear runs 4-5%.
        #
        # TASOPT does not model gear structure at all; its 737 deck carries
        # flat fractions (runs/737/737s.tas:344-345):
        #
        #     0.011  ! flgnose   Wlgnose/WMTO
        #     0.044  ! flgmain   Wlgmain/WMTO
        #
        # Those go in as FLOORS rather than replacements, so the structural
        # model can still drive the gear heavier -- a tall gear on a low-wing
        # hydrogen aircraft, say -- but can no longer come in below what a real
        # aeroplane's gear weighs.
        # Scaled by gear LENGTH, not flat.
        #
        # A flat fraction is length-blind, and because the floor binds it
        # overrode the structural model's own length sensitivity (the strut is
        # 2*pi*r*t*l*rho*g, linear in l). Length then cost nothing while the
        # tail-strike row actively rewarded it -- longer legs buy cabin through
        # `l_nose + l_shell <= x_m + l_m/tan(theta)` -- so the gear grew to
        # 2.86 m and stood the fuselage centreline 4.72 m off the ground
        # against a real 737's ~2.9 m.
        #
        # Referencing to the fuselage radius makes it self-scaling across the
        # matrix and reproduces TASOPT's fraction exactly at l_m = R_fuse,
        # which is about where a transport's main gear sits.
        # REVERTED to flat fractions, and the reason is worth keeping.
        #
        # Scaling these by leg length is the right instinct -- the strut really
        # is 2*pi*r*t*l*rho*g, linear in length -- and it worked in the
        # direction intended: legs went 2.86 -> 1.48 m. But nothing in this
        # model sets a MINIMUM leg length, so the loop closed on itself: short
        # legs -> light gear -> nothing penalises short legs. It bottomed out
        # at l_m = 1.28 m with the gear at 3.3% of MTOW (real: 4-5%), and
        # dragged the wing aft to 61% of fuselage length chasing the tail
        # strike constraint.
        #
        # What sets minimum gear length on a low-wing twin is ENGINE GROUND
        # CLEARANCE, and this model has none: `h_nacelle` = 0.5 m is declared
        # in landing_gear.py, exported, and used in no constraint -- the fourth
        # declared-but-unwired variable found this session, after x_upswp,
        # AR_vt and C_L_ht_fCG. Restore the length scaling once the nacelle
        # clears the ground.
        # Scaled by leg LENGTH. The strut really is 2*pi*r*t*l*rho*g, linear in
        # length, and a flat fraction is length-blind -- binding, it overrode
        # the structural model's own sensitivity and left length free.
        #
        # This only became safe once the engine ground clearance above gave
        # length a floor. Without it the loop closed on itself: short legs ->
        # light gear -> nothing penalising short legs, bottoming out at 1.28 m
        # with the gear at 3.3% of MTOW and the wing dragged aft to 61% of
        # fuselage length. Referenced to the fuselage radius so it is
        # self-scaling across the matrix and reproduces TASOPT's fraction at
        # l_m = R_fuse.
        # Empirical weight floors REMOVED, so that gear weight is driven by gear
        # LENGTH through the structural model -- the strut is
        # 2*pi*r*t*l*rho*g, linear in length, and a flat fraction of MTOW is
        # length-blind. Binding, it overrode that sensitivity entirely.
        #
        # This only became safe once engine ground clearance gave length a
        # floor of its own; without it the loop closed on itself (short legs ->
        # light gear -> nothing penalising short legs).
        #     lg.W_mg >= 0.044 * W_totalmax,
        #     lg.W_ng >= 0.011 * W_totalmax,
    ]

    # ---- certification performance, linked to the aircraft -----------------
    # Takeoff is the first mission segment: maximum weight, sea level, all
    # engines. Landing weight is taken as the zero-fuel weight plus reserves,
    # approximated here by W_dry + payload, which is the conservative end.
    # The engine-out control speed IS a certification speed, not a per-class
    # constant. FAR 25.149 requires V_MC <= 1.13 V_SR, and the fin's engine-out
    # case should be evaluated there rather than at a number inherited from the
    # D8.2. On the 737 the two nearly coincide (73.9 m/s against the pinned
    # 75.0); on a Citation or a 787 they do not.
    # The fin's engine-out case is evaluated at V_MC, the minimum control speed,
    # which is where FAR 25.149 says directional control must still exist and
    # which is the WORST case for the fin -- lowest dynamic pressure with full
    # asymmetric thrust. Previously this ran at V_1, pinned per class at a D8.2
    # value with no connection to the aircraft's own stall speed.
    #
    # V_MC is bounded above by 1.13 V_SR and the fin must cope at that bound,
    # so it is imposed as an equality here: taking any lower V_MC would be
    # conservative but arbitrary, and taking a higher one is not permitted.
    # The fin's engine-out case runs at V_MCG, the GROUND minimum control
    # speed, not the airborne one: it is the lower speed and the one with no
    # bank available, so it is what sizes the surface. See components/far.py.
    cons += [vt.V_1 == farv["V_MCG"],
             farv["V_MC"] == 1.13 * farv["V_s_TO"]]

    # ---- sea-level takeoff thrust ------------------------------------------
    # Every takeoff event in this model -- engine-out at V_MC, the ground roll,
    # the FAR 25.121 gradients -- happens at sea level and about M 0.23. But
    # the mission's first segment sits at 12,379 ft and M 0.70, because this
    # model has no rotation point: its segments are [climb, climb, climb,
    # cruise, cruise]. TASOPT does have one and uses it exactly here --
    # wsize.f:1066 takes the engine-out thrust from `ip = iprotate`.
    #
    # Until a rotation segment exists, lapse the first segment's thrust back to
    # sea level rather than pretend the two conditions are the same:
    #
    #     F ~ delta * (1 - 0.49 sqrt(M))          (Mattingly, high bypass)
    #
    # The pressure ratio is carried exactly, as p_sl/p_atm[0]; the Mach factor
    # is a constant evaluated at the segment's solved M (~0.70) against M 0.23
    # at rotation, because writing it out would need sqrt of a variable inside
    # a difference. It is an approximation and it is labelled as one -- the
    # honest fix is the rotation segment.
    F_TO = f.Variable("F_takeoff", 2e5, "N", "sea-level takeoff thrust per engine")
    k_mach = f.Constant("k_mach_lapse", 1.294, "-",
                        "Mach lapse ratio, rotation vs first climb segment")
    cons += [F_TO == eng.F[0] * (st.p_sl / st.P_atm[0]) * k_mach]

    cons += far_link(
        far, farv,
        W_TO=W_totalmax,
        W_land=W_dry + fu.W_payload,
        S=wing.S,
        rho_TO=rhoTO,
        T_TO=numeng * F_TO,
        D_clean_TO=D[0],
        AR=wing.AR,
        e=wing.e,
        n_eng=_ne,
    )
    cons += [
        wing.W_fuel_wing >= f_wingfuel * W_ftotal / FuelFrac,

        # ---- landing gear and power systems ----------------------------------
        Wmisc >= lg.W_lg + Whpesys,
        Whpesys == fhpesys * W_totalmax,
        # REMOVED: `lg.x_n <= fu.l_nose`. It is in neither source and it points
        # the wrong way. TASOPT reads the nose gear station as a fixed input
        # (`parg(igxlgnose)`, getparm.f:683, 5.0 m in the decks) and uses it
        # only in the CG sum; SPaircraft writes `x_n >= 5*units.m  # nose gear
        # after nose` -- a LOWER bound, placing the leg AFT of the nose. Ours
        # confined it to the nose cone instead.
        #
        # That was the blocker behind a long chain: l_nose sits on its own
        # 1.2-calibre fineness floor because wetted area drives it down, so the
        # cap pinned x_n at 4.45 m, which forced the wheelbase long, which
        # starved L_n/W = dx_m/B onto its 8% floor, which is what stopped the
        # main gear moving aft to the rear spar.
        #
        # RESTORED WITH THE CORRECT SENSE. SPaircraft's row is
        #
        #     x_n >= 5*units.m,   # nose gear after nose
        #
        # and "after the nose" means AFT of the nose section, not inside it.
        # Ours had the same two quantities and the opposite inequality, which
        # is the whole defect. With the cap alone removed the leg ran forward
        # to x_n = 1.71 m on a 37.4 m aeroplane -- the same pathology, just
        # pointing the other way, because the load band constrains the RATIO
        # dx_m/B and is happy to buy it with a long wheelbase.
        #
        # NOSE GEAR IS FREE, and carries NO dependency on the nose cone.
        #
        # Two rows have stood here and both were wrong. `x_n >= l_nose`
        # required the gear at or behind the front of the pressure shell,
        # which TASOPT's own 737 violates by 0.92 m -- xlgnose 4.267 against
        # xshell1 5.182 -- because the bay sits in the unpressurised nose.
        # Pinning x_n to TASOPT's xlgnose instead fixed the geometry but
        # made the station an input where it can be a result.
        #
        # The cone length is now specified (l_nose == 5.182 for the 737),
        # so the two are decoupled: the cone is geometry, the gear station
        # is placed by the load rules -- the 15% maximum, which wants a long
        # wheelbase and so pulls the gear forward, against the aft-CG 8%
        # minimum and the weight of the leg itself.
        lg.x_m >= fu.x_wing,
        # The gear may not hang off the back of the wing: aft limit is the local
        # trailing edge. This REPLACES `x_m <= dx_AC_wing + x_wing`, which
        # capped the gear at the wing's aerodynamic centre and confined it to a
        # ~1 m window there. A real main gear attaches at the REAR SPAR, well
        # aft of the AC -- so that cap and the rear-spar rule below were in
        # direct contradiction, and it was the one infeasible row of 6301 that
        # the elastic-L1 diagnostic kept isolating.
        lg.x_m <= fu.x_f + 0.35 * wing.c_root + lg.y_m * wing.tan_Lambda,
        xhpesys == 1.1 * fu.l_nose,
        xmisc * Wmisc >= xhpesys * Whpesys,
        lg.d_nacelle >= eng.d_f + 2 * lg.t_nacelle,
        # ENGINE GROUND CLEARANCE -- the thing that sets minimum gear length on
        # a low-wing twin, and which this model did not have. `h_nacelle` was
        # declared in landing_gear.py, exported, and used in no constraint.
        #
        # The nacelle hangs under the wing, which sits z_wing above the
        # fuselage base, which sits l_m above the runway. Its lowest point has
        # to clear the ground.
        #
        # NOT a certification requirement, as far as I can establish. FAR
        # 25.925 sets a hard PROPELLER clearance (order of 7 inches for a
        # nosewheel aeroplane in the takeoff attitude), but I can find no
        # equivalent numeric nacelle dimension in Part 25 for turbofans --
        # there, clearance is driven by water and slush ingestion under
        # 25.1091, by FOD practice, and by roll-on-landing geometry. So 0.5 m
        # is a design value, not a regulation. It is well matched to reality:
        # a 737-800 runs about 0.46 m, low enough that the CFM56 nacelle is
        # famously flattened underneath to buy it back.
        lg.l_m + lg.z_wing >= lg.d_nacelle + lg.h_nacelle,
        lg.d_nacelle <= wing.b,
        # Hard landing, Torenbeek (10-26): 10 ft/s sink at max landing weight.
        lg.E_land >= W_totalmax / (2 * g) * lg.w_ult ** 2,
        lg.x_up == fu.x_shell2,
        lg.L_n == W_totalmax * lg.dx_m / lg.B,
        # FORWARD GEAR MARGIN. The nose gear carries 8-15% of the weight: below
        # that, nosewheel steering and braking lose authority and the aircraft
        # is prone to tipping back; above it, the gear is heavy and rotation
        # takes more tail. This -- not a fixed station -- is what places a nose
        # gear, and it replaces the hard 5 m floor that used to live in
        # landing_gear.py and was sizing the nose of every aircraft here.

        # TAIL STRIKE. The rotation-clearance row in landing_gear.py,
        #     x_up - x_m <= l_m / tan(theta_max)
        # was already there, but `x_up` was a free variable tied to nothing --
        # declared, used once, and never connected to the fuselage, so the
        # constraint could always be satisfied by moving an imaginary point.
        # The upsweep begins where the cylindrical section ends.
        # TAIL STRIKE, written directly rather than through `x_up`.
        #
        # landing_gear.py already carries the rotation-clearance row
        #     x_up - x_m <= l_m / tan(theta_max)
        # but `x_up` was a free variable connected to nothing -- declared, used
        # once, tied to no geometry -- so the constraint was always satisfiable
        # by moving an imaginary point and never bit.
        #
        # The critical point is the TAIL TIP, not the end of the cylindrical
        # section, because the aft fuselage is upswept: by the tail the belly
        # line has risen roughly 1.5 fuselage radii above where it sits under
        # the cabin. Using the cylinder end instead makes the constraint far
        # too strict -- strict enough that a real 737-800 fails it
        # ((30.0-17.5)*tan15 = 3.35 m of drop against ~2.0 m of gear) -- and
        # the model satisfied it by shoving the main gear to 62% of fuselage
        # length and the nose gear back to the nose tip.
        #
        # Written as this single inequality rather than as `x_up == l_nose +
        # l_shell`: the equality form couples the fuselage chain, gear length,
        # gear position and rotation angle simultaneously and cost the solve
        # its stationarity (1.9e-02 against a 1e-05 tolerance, feasible but
        # never settling). Same physics, one row, well behaved.
        (fu.l_fuse - lg.x_m) * lg.tan_theta_max
            <= lg.l_m + 1.5 * fu.R_fuse,
        # Forward gear margin: the nose gear carries 8-15% of the weight. This
        # is what places a nose gear, and it replaces the hard `x_n >= 5 m`
        # that used to sit in landing_gear.py sizing every aircraft's nose.
        # AT THE CG EXTREMES, not at the cruise CG.
        #
        # These were written against L_n == W*dx_m/B with dx_m measured from
        # xCG[Nclimb] -- the CRUISE CG -- so the band was being satisfied at a
        # loading condition that never governs. The nose carries LEAST at the
        # aft CG and MOST at the forward CG; checking the middle guarantees
        # neither. With the wing correctly placed the consequence was
        # immediate and severe: the gear sat 0.137 m aft of the aft CG limit,
        # 1.0% nose load against the 8% minimum, which is an aeroplane that
        # sits on its tail.
        #
        # x_m >= xCGaft + 0.08*B is posynomial <= monomial and so plain GP;
        # the forward one reverses and is signomial.
        # ONLY THE AFT ONE MOVES. Imposing BOTH ends over the full envelope is
        # arithmetically impossible here and it is worth recording why:
        #
        #     xCGaft + 0.08 B <= x_m <= xCGfwd + 0.15 B
        #  => travel <= 0.07 B  =>  B >= 1.9/0.07 = 27.1 m
        #
        # against a 13.25 m wheelbase. The sub-problem goes infeasible. This
        # is NOT specific to this model: TASOPT's own 737 carries 2.34 m of
        # travel on a 15.68 m wheelbase, travel/B = 0.149, and would fail the
        # same band -- which is why TASOPT does not impose one. Real aircraft
        # meet 8-15% because their CERTIFIED envelope is about 0.8 m, far
        # narrower than the LOADABILITY extreme cglpay computes; the gap is
        # operational loading rules, which this model does not have.
        #
        # So the aft end is imposed, because it is the one with a hard
        # physical consequence -- at the aft CG limit the nose was carrying
        # 1.0% and the aeroplane sits on its tail -- and the forward end stays
        # where it was until there is an operational envelope to hang it on.
        xCGaft + f_nose_min * lg.B <= lg.x_m,
        lg.L_n <= 0.15 * W_totalmax,
        lg.L_m == W_totalmax * lg.dx_n / lg.B,
        lg.L_n_dyn >= 0.31 * ((lg.z_CG + lg.l_m) / lg.B) * W_totalmax,
        y_eng >= lg.y_m,
        # MAIN GEAR TRACK -- set by retraction into the fuselage.
        #
        # The gear stows into the fuselage, so the axle has to sit within the
        # fuselage radius (a blister takes up any small excess). Inboard of
        # that is fine, down to where the tyres would meet on the centreline.
        #
        # This replaces a bound at 20% of semi-span, which was a stand-in for a
        # missing weight term -- track appears nowhere in the gear or wing
        # carry-through structure, so a wider track costs this model nothing
        # and y_m was degenerate.
        # Applied to the FOLDED position, which is what has to fit. Swinging the
        # leg inboard about its trunnion puts the retracted axle at y_m - l_m,
        # so:
        #
        #   tyre outer edge inside the fuselage:  (y_m - l_m) + w_t_m <= R
        #   tyres clear of the centreline:        (y_m - l_m)         >= w_t_m
        #
        # Rearranged below with everything on the positive side, so both stay
        # posynomial -- no subtraction, no signomial.
        #
        # Applying it to the EXTENDED axle instead is far too strong: a real
        # 737 sits at y_m = 2.86 m against a 1.88 m radius, well outside. Tried
        # that way the fuselage simply inflated to R = 2.54 m to swallow the
        # gear and the solve came apart. Folded, the real aircraft satisfies
        # both comfortably (0.86 + 0.4 <= 1.88, and 0.86 >= 0.4).
        #
        # Tyre width is a sized quantity, not a guess -- landing_gear.py
        # carries Raymer's correlation off the static wheel load,
        # w_t_m = 0.1043 (L_w_m/4.44)^0.48 in.
        lg.y_m + lg.w_t_m <= fu.R_fuse + lg.l_m,
        lg.y_m >= lg.l_m + lg.w_t_m,

        # GEAR ATTACHMENT AT THE REAR SPAR.
        #
        # The rear spar is ALREADY IN THE MODEL and I did not need to invent
        # one. fuselage.py:358-359 carries the wingbox stations
        #
        #     x_f == x_wing + 0.5*c_root*r_w_c      (aft  edge)
        #     x_b == x_wing - 0.5*c_root*r_w_c      (fwd  edge)
        #
        # so x_f IS the rear spar, and x_wing is the box CENTRE -- not the
        # quarter chord its description claims.
        #
        # What I wrote instead was a parallel definition, x_le + 0.8*c_local
        # with x_le taken as x_wing - 0.25*c_root. Two errors compounding: the
        # leading-edge reference was wrong because x_wing is the box centre,
        # and the fraction was invented. At the root it demanded
        # x_wing + 0.55*c_root against the box's actual aft edge at
        # x_wing + 0.25*c_root -- the gear placed 0.30*c_root, about 1.7 m,
        # aft of where the wing structure even extends. That is the ~1.2 m the
        # elastic-L1 diagnostic kept reporting, and the reason five separate
        # relaxations (engine station, nose-gear load, rotation authority,
        # tail area, and free ballast) all failed to help: nothing could,
        # because the requirement was off the back of the wingbox.
        #
        # GEAR ATTACHMENT STATION, as a fraction of the wingbox.
        #
        # `frac` interpolates along the box at the gear's own spanwise station:
        # 0 puts the leg on the FRONT spar, 1 on the REAR spar. Written as a
        # convex combination of x_b and x_f rather than `x_b + frac*(x_f-x_b)`
        # so every coefficient stays positive -- the nested subtraction is what
        # cost the earlier attempts their convergence.
        #
        # frac = 1 (the rear spar, which is where a real gear beam goes) is
        # NOT achievable in this model, and the reason is a genuine design
        # conflict rather than a bug. The nose-load rule caps the gear at
        #
        #     x_m <= (x_CG - f*x_n) / (1 - f)
        #
        # once dx_m and B are true identities, and the rear spar at the gear
        # station sits aft of that cap. GEAR_BOX_FRAC exists so the achievable
        # fraction is a measured number rather than an assumption; see the
        # sweep recorded below.
        lg.x_m >= ((1.0 - _GEAR_BOX_FRAC) * fu.x_b + _GEAR_BOX_FRAC * fu.x_f
                   + lg.y_m * wing.tan_Lambda),
        #
        # The leg hangs off wing structure and the rear spar sits at about 60%
        # of the local chord, so the gear station should be aft of it at the
        # gear's own spanwise station:
        #
        #   c_local + c_root*eta == c_root + c_tip*eta      (eta = 2 y_m / b)
        #   x_m >= x_wing - 0.25 c_root + y_m tan(L) + 0.60 c_local
        #
        # Both forms tried -- the direct c_root - (c_root - c_tip)*eta and the
        # all-positive equality above -- and both cost the solve its
        # convergence, landing x_wing at 66.9 m on a 37.2 m fuselage. The
        # hypothesis that the engine's forward bound (x_eng >= 0.8 x_wing) was
        # blocking the CG travel this needs was tested by relaxing it to 0.62
        # and made no difference at all: identical iterate to five figures.
        #
        # So the cause is elsewhere and I have not isolated it. Left here
        # because the constraint is right and the next person should not have
        # to rederive it.



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
        # Tie the box's inertial relief to the engine actually hung on the
        # wing. Declared inside the box because the engine is built after the
        # wing; a rear-mounted installation relieves nothing, so it is pinned
        # near zero rather than left free -- an unconstrained relief term
        # would let the optimiser invent load relief from nothing.
        *([wing.box.W_eng_relief == (1e-6 * Wengsys if arch.rear_engines
                                     else Wengsys)]
          if wing_model == "tasopt" else []),
    ]

    # ---- engine installation -------------------------------------------------
    # SPaircraft is optimalD8, so this whole block was written for REAR-mounted
    # engines and applied unconditionally. Left that way, every "conventional"
    # aircraft here is an MD-80: engines in the tailcone, their weight loading
    # the aft-fuselage bending, their inertia out at the tail, and no wing
    # bending relief at all. That is not a 737 or a 787, and the difference is
    # worth real wing weight.
    #
    # ``arch.rear_engines`` was declared and never read. It is read now.
    # Terms are built by OMISSION, never by multiplying by zero: a zero
    # monomial has no logarithm and takes the whole model out of GP/SP form.
    _rear = arch.rear_engines

    def _plus_eng(base):
        """Add the engine mass to a fuselage/tail term, if it is back there."""
        return base + numeng * Wengsys if _rear else base

    def _relief():
        """DISTRIBUTED relief only: wing structure and the fuel it carries.

        These are spread along the span roughly like the chord, so giving them
        the same moment arm as the lift is reasonable. A podded engine is a
        POINT mass and is not: it belongs at its own arm, y_eng, and is added
        separately in the root-moment row below.

        Lumping the engine in here -- which is what this did -- meant the
        relief did not depend on where the engine was. That is the whole
        reason airliners hang engines outboard, and the model could not see
        it: y_eng appeared only in the engine-out yaw moment, so the fin
        wanted it inboard and nothing wanted it out.
        """
        return wing.W_wing + f_wingfuel * W_ftotal

    def _iz_wing():
        base = ((wing.W_fuel_wing + wing.W_wing) / (wing.S * g)
                * wing.c_root * wing.b ** 3.
                * (1. / 12. - (1. - wing.lambda_) / 16.))
        if _rear:
            return base
        # y_eng, not the constant it happens to be pinned to. Those agreed
        # only because both were 0.285*b/2; two engine positions in one model
        # is the same defect as the two eta_o values in the cranked wing.
        return base + numeng * Wengsys * y_eng ** 2 / g
    cons += [
        # Engine-out yaw arm. A podded underwing engine sits far outboard, so
        # losing one is a much bigger yawing moment than losing a rear engine
        # tucked against the fuselage -- which the vertical tail has to size
        # for. This is a genuine penalty of the conventional layout and the
        # rear-engine D8 gets to avoid it.
        # Engine spanwise station. On the CRANKED wing it sits at the
        # planform break, which is both what most transports do (a 737's
        # engines are at eta ~0.28-0.31 against a 0.285 break) and what
        # TASOPT's surfw.f assumes -- its inertial relief is written with the
        # engine AT eta_s, propagating into the root moment through
        # (Ss - Nload*We)*0.5*b*(eta_s - eta_o). The single-taper model has no
        # break to sit on, so it keeps the 0.35 semi-span station it had.
        # Engine spanwise station. Rear-mounted is fixed against the fuselage.
        # Wing-mounted is either pinned to the planform break (the default,
        # and what TASOPT assumes) or FREE, if FREE_Y_ENG is set.
        #
        # Freeing it is only meaningful now that the root-moment row gives the
        # engine its own arm. Before that, y_eng appeared solely in the
        # engine-out yaw moment: the fin wanted it inboard, nothing wanted it
        # out, and it would have collapsed onto its lower bound. With the
        # relief span-dependent there is a real trade -- outboard relieves the
        # wing and grows the fin -- so both directions are now resisted.
        *([y_eng == 0.5 * fu.w_fuse] if _rear else
          ([y_eng >= 1.15 * (fu.w_fuse + 0.5 * lg.d_nacelle),
            y_eng <= 0.5 * (wing.b / 2.0)]
           if _FREE_Y_ENG else
           [y_eng == (wing.eta_s * wing.b / 2.0 if wing_model == "tasopt"
                      else _ETA_ENG * wing.b / 2.0)])),
        # Wing root moment, relieved by wing weight and fuel -- and, for an
        # underwing installation, by the engines themselves. That relief is
        # one of the reasons real airliners hang engines on the wing.
        #
        # ONLY for the closed-form box. The TASOPT box derives its own root
        # moment from the spanwise load distribution and carries its own
        # relief fixed point internally (see wingbox_tasopt), so imposing this
        # as well would define M_r twice with two different load models.
        # Written all-positive, with the engine at ITS OWN ARM. The
        # distributed relief keeps the lift's arm factor K; the engine gets
        # N_lift * n_eng * W_engsys * y_eng, which is what makes an outboard
        # engine buy a lighter wing. Rear-mounted engines relieve nothing.
        *([wing.box.M_r * wing.c_root
           + wing.box.N_lift * _relief()
             * (wing.b ** 2 / (12 * wing.S) * (wing.c_root + 2 * wing.c_tip))
           + (0.0 * Wengsys * y_eng if _rear
              else wing.box.N_lift * numeng * Wengsys * y_eng)
           >= wing.L_max
              * (wing.b ** 2 / (12 * wing.S)
                 * (wing.c_root + 2 * wing.c_tip))]
          if wing_model == "hoburg" else []),
        fu.A_1h_Land >= (fu.N_land * _plus_eng(fu.W_tail + fu.W_apu))
                           / (fu.h_fuse * fu.sigma_bend),
        fu.A_1h_MLF >= (fu.N_lift * _plus_eng(fu.W_tail + fu.W_apu)
                           + fu.r_M_h * ht.L_ht_max) / (fu.h_fuse * fu.sigma_M_h),
        Izwing >= _iz_wing(),
        Iztail >= (_plus_eng(fu.W_apu + vt.W_vt) * vt.l_vt ** 2. / g
                   + ht.W_ht * ht.l_ht ** 2. / g),
        # x_wing and l_vt stand in for CG-relative distances so I_z stays scalar.
        Izfuse >= ((fu.W_fuse + fu.W_payload_max) / fu.l_fuse
                   * (fu.x_wing ** 3. + vt.l_vt ** 3.) / (3. * g)),
        # Streamwise engine station: in the tailcone if rear-mounted, at the
        # wing box if podded underwing.
        *( [xeng <= fu.x_shell2 + 1.00 * fu.l_cone,
            xeng >= fu.x_shell2 + 0.75 * fu.l_cone]
           if _rear else
           [xeng <= fu.x_wing, xeng >= 0.8 * fu.x_wing] ),

        # ---- double-bubble floor loading -------------------------------------
        fu.S_floor == (5. / 16.) * fu.P_floor,
        fu.M_floor == 9. / 256. * fu.P_floor * fu.w_floor,
        fu.dR_fuse == fu.R_fuse * 0.43 / 1.75,
    ]

    # ---- horizontal tail attachment ------------------------------------------
    # A D8's horizontal spans BETWEEN two fins -- a beam on two supports, with
    # an outboard pin joint to react bending at. A conventional aircraft's is
    # a cantilever off the fuselage or fin root, and none of the pi-tail rows
    # below mean anything for it: there is no attachment chord, no outboard
    # span, no pin-joint shear. The cantilever case is sized inside the box
    # model instead (wingbox surfacetype "horizontal_tail_conventional").
    pi_tail = arch.double_bubble
    hb = ht.box
    if pi_tail:
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
        # Drag AREA straight from the BL fit -- no reference-area convention
        # to get wrong. The Mach factor is SPaircraft's compressibility
        # correction, retained because the fit was taken at a single M = 0.80.
        DAfuse >= k_fdrag * fu.l_fuse ** 0.9062 * (2.0 * fu.R_fuse) ** 1.1036,
        Dfuse == (0.5 * st.rho * st.V ** 2 * DAfuse
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
        # The fuselage moment volume, and the forward NP shift it produces.
        CMVf1 == k_Mf * pi * fu.R_fuse ** 2 * fu.l_fuse,
        dxNPf * wing.S == CMVf1,
        # xNP is the wing+tail neutral point; the hull pulls it FORWARD, which
        # is what makes the tail work harder.
        xAC + dxNPf <= fu.x_wing + 0.25 * wing.dx_AC_wing + xNP,
        wing.c_m_w == cmw,
        # Neutral point approximation, from Unified's aircraft design rules.
        (xNP / wing.mac / ht.V_ht * (wing.AR + 2.)
         * (1. + 2. / ht.AR_ht)
         == (1. + 2. / wing.AR) * (wing.AR - 2.)),             # [SP] SigEq
        xCG + vt.dx_trail_vt <= fu.l_fuse,
        # THE HORIZONTAL TAIL HAS TO FIT ON THE AEROPLANE TOO.
        #
        # The fin has had this row all along; the horizontal never did. Its
        # only placement rows are `x_CG_ht <= l_fuse`, which bounds the tail's
        # MIDPOINT, and the pi-tail geometry row above, which a conventional
        # configuration never builds. So on every conventional aircraft in this
        # study the horizontal tail was free to hang off the back of the
        # fuselage.
        #
        # And it paid to: tail area enters the trim residual as S_ht*l_ht, so a
        # longer arm buys a smaller surface, and nothing was charging for the
        # arm. The 737 came out with l_ht = 16.57 m on a 33 m fuselage -- an
        # arm half the length of the aircraft -- and a 22.8 m2 tail against the
        # real 32.8 m2.
        #
        # Same shape of defect as the one-sided `mac`, `l_ht` and `l_vt` rows:
        # a quantity the optimizer can inflate because nothing on the other
        # side of the model objects.
        xCG + ht.dx_trail_ht <= fu.l_fuse,
        vt.x_CG_vt >= xCG + 0.5 * (vt.dx_lead_vt + vt.dx_trail_vt),
        ht.x_CG_ht >= xCG + 0.5 * (ht.dx_lead_ht + ht.dx_trail_ht),
        # Horizontal tail lift-curve slope: downwash AND finite span.
        #
        # The source corrects only for wing downwash,
        #     CLah = CLah0 * eta * (1 - deps/dalpha),  deps/dalpha = 2 CLaw/(pi AR_w)
        # which references the WING's aspect ratio and leaves the tail's own
        # nowhere in the expression. So AR_ht had no aerodynamic consequence at
        # all: a tail of aspect ratio 2 was exactly as effective as one of 6.
        #
        # That is why AR_ht slid onto whatever floor existed and why V_ht fell
        # to 0.447 against TASOPT's 0.575 when the floor was removed -- every
        # structural row penalises aspect ratio and nothing rewarded it. The
        # induced-drag term C_L_ht**2/(pi e AR_ht) cannot do the job either,
        # because it is evaluated at the CRUISE tail load, and the cruise tail
        # is unloaded (C_L_ht = 0.01).
        #
        # Prandtl's finite-span correction supplies the missing pressure:
        #     CLah_3D = CLah0 / (1 + CLah0/(pi e_ht AR_ht))
        # so a stubby tail has a shallower lift-curve slope and must be BIGGER
        # to trim the same moment. Multiplying through by eta*CLah0 keeps the
        # row a posynomial <= monomial, i.e. still GP.
        ht.C_L_alpha_ht * (1 + ht.C_L_alpha_ht_0 / (pi * ht.e_ht * ht.AR_ht))
            + (2 * wing.C_L_alpha_w / (pi * wing.AR))
            * ht.eta_ht * ht.C_L_alpha_ht_0
            <= ht.C_L_alpha_ht_0 * ht.eta_ht,
        ht.C_L_ht >= 0.01,
        # HORIZONTAL TAIL VOLUME FLOOR -- now OFF by default.
        #
        # This existed because the trim and rotation cases only asked for
        # V_ht ~ 0.72 against a real 737's ~0.98, so Raymer's transport value
        # stood in for the cases the model does not contain (landing flare at
        # aft CG, stall recovery, deep-stall avoidance, gust loads).
        #
        # The reason it was needed has gone. The forward-CG trim case could
        # not size the tail while x_CG_fwd was a free unknown of its own trim
        # equality -- the optimiser simply co-solved the pair and put the
        # forward limit 2.3 m ahead of any loading the cabin can produce. With
        # the CG envelope on TASOPT's cglpay footing, trim asks for V_ht =
        # 0.922 unaided, and S_ht comes out at 1.049 of the real tail against
        # 1.325 with the floor imposed. A floor that binds ABOVE what the
        # physics wants is no longer standing in for a missing case; it is
        # just adding 9 m2 of tail and 4% of MTOW.
        #
        # Kept as a switch rather than deleted: set V_HT_FLOOR back to 1.00 to
        # restore the old behaviour.
        ht.V_ht >= V_HT_FLOOR,

        # THE TAIL MUST BE ABLE TO GENERATE THE LIFT IT IS SIZED FOR.
        #
        # Tail AREA comes from the forward-CG trim residual, which assumes the
        # tail delivers CLhfwd = 0.65 (TASOPT's value). Nothing anywhere said
        # the tail could actually reach 0.65. So the sizing row was satisfied
        # by a surface with no aerodynamic capability at all: with the finite-
        # span correction added above, the optimizer simply took AR_ht to 0.25
        # and the lift-curve slope to 0.079 per radian -- a tail that would
        # need 8 radians of incidence to make 0.65 -- and the objective did not
        # move by a single count.
        #
        # This closes it: at its maximum usable incidence the tail must reach
        # the coefficient its own sizing case assumes. Combined with the
        # finite-span and downwash terms this is what makes AR_ht mean
        # something, and it is the honest version of the arbitrary AR_ht >= 4
        # that used to sit here -- that floor was, near enough, this constraint
        # evaluated once and hard-coded.
        CLhfwd <= ht.C_L_alpha_ht * ht.alpha_ht_max,

        # ---- TAKEOFF ROTATION: the horizontal tail's ground case ----------
        # For a transport, the case that sizes the horizontal tail is usually
        # not cruise trim -- it is rotation. At V_R, with the CG fully forward
        # and the nosewheel still down, the tail has to pitch the aeroplane up
        # about its main gear. Cruise trim is a far gentler requirement, and
        # sizing on it alone is why V_ht came out at 0.707 against a real
        # 737's ~0.98 while every other tail quantity checked out: the static
        # margin matches TASOPT's 0.05 exactly, and the CG travel is 60% MAC
        # against TASOPT's ~53%, so neither of those was holding it back.
        #
        # This is the same shape of omission as the fin's, and it has the same
        # answer: the airborne case is not the critical one, the GROUND case
        # is. V_MCG did this for the vertical tail; rotation does it here.
        #
        # REMOVED, and the paragraph above is kept only as the record of why it
        # was ever here. The row was:
        #
        #     0.5*rhoTO*V_LOF**2 * S_ht * CLhrot * (xCG + l_ht - x_m)
        #         >= W_totalmax * (x_m - xCGfwd)
        #
        # TASOPT does not size the horizontal tail on rotation. `htsize`
        # (balance.f:236) says exactly what it sizes on, and it is two airborne
        # cases -- "1) Meet pitch trim requirement with forward CG, 2) Meet
        # stability requirement with aft CG" -- solved simultaneously for tail
        # area AND wing position. The rotation point (`iprotate`) exists in
        # wsize.f, but it sets engine cooling mass flow and takeoff thrust, not
        # tail area.
        #
        # Two things were wrong with it beyond provenance. It charged the tail
        # with the WHOLE aircraft weight at V_LOF, the speed at which the wing
        # is by definition carrying nearly all of that weight, so the residual
        # download on the gear is small and the row overstated it by roughly an
        # order of magnitude. And it is doubly steep in x_m: moving the gear aft
        # shrinks the tail's arm (xCG + l_ht - x_m) while growing the weight's
        # arm (x_m - xCGfwd), so the cost of an aft gear compounds.
        #
        # It was the single violated row when the gear is placed at the rear
        # spar -- a one-row IIS -- and the dominant sizer of S_ht everywhere
        # else (|dual| 10.4, an order of magnitude above any other tail row).
        # It drove S_ht to 63.9 m2 against a real 737's ~32.
        #
        # RESTORED, with the weight term corrected. Rotation happens at V_R,
        # NOT at lift-off: the aeroplane is still on its wheels at the ground
        # attitude, so the wing is flying at C_L_ground (~0.9) rather than at
        # C_Lmax_TO (2.0), and it has already taken part of the weight off the
        # gear. Roughly (V_LOF/V_s)^2 * C_L_ground/C_Lmax_TO ~ 1.21*0.45 ~ 0.55
        # of it, leaving ~45% for the tail to lift the nose against -- so the
        # old row overstated the load by about 2.2x, not by the order of
        # magnitude I first claimed.
        #
        # The residual is written as an all-positive identity so the moment row
        # carries one subtraction rather than a product of two.
        Wrot + 0.5 * rhoTO * farv["V_LOF"] ** 2 * wing.S * CLgnd
            == W_totalmax,                                    # [SP] SigEq
        0.5 * rhoTO * farv["V_LOF"] ** 2 * ht.S_ht * CLhrot
            * (xCG[Nclimb] + ht.l_ht - lg.x_m)
            >= Wrot * (lg.x_m - xCGfwd),
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
        V2 == (eng.M_2 * st.a if not electric else st.V),

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
        # Boil-off is charged against the fuel budget in every segment. It
        # has to be: nothing else in this model pays for insulation, so
        # without this row the optimiser deletes it -- t_insul ran to zero
        # and the tank got its weight and length for free.
        #
        # Charging it rather than imposing TASOPT's fixed 0.4 %/hour policy
        # makes insulation thickness a real trade here: foam weight and the
        # fuselage length it costs, against the hydrogen it boils away.
        #
        # A battery aircraft has no burn row at all: _burn_row() returns None
        # and the decrement below simply goes slack, which is the correct
        # statement that the aircraft does not get lighter.
        W_start >= W_end + W_burn,
        PCFuel <= 1.0000001,

    ]
    if not electric:
        # Turbofan cycle matching: fan face to LPC face to burner. An
        # electric fan has no core, no burner and no bypass ratio, so these
        # rows are absent rather than zeroed.
        cons += [
            eng.M_2 == st.M,
            eng.M_25 == 0.6,
            eng.hold_2 == 1. + .5 * (1.398 - 1.) * 0.6 ** 2,
            eng.hold_25 == 1. + .5 * (1.354 - 1.) * 0.6 ** 2,
            eng.c1 == 1. + 0.5 * .401 * st.M ** 2.,      # [SP] SigEq
        ]


    # Aircraft-level geometry and stability limits. These mention no segment,
    # so the source's per-segment loop stated each of them N times over; they
    # are stated once here. Identical rows constrain nothing extra -- presolve
    # would drop the copies anyway -- so this removes 3*(N-1) rows and leaves
    # the feasible set alone.
    # HT aspect-ratio floor removed. It was binding, and it was the only thing
    # setting AR_ht -- because the tail's induced drag, C_L_ht**2/(pi e AR_ht),
    # is evaluated at the CRUISE tail load, and the cruise tail is essentially
    # unloaded (C_L_ht = 0.01, on its own floor). That makes the induced term
    # ~1e-05 while every structural row penalises aspect ratio, so AR_ht slid
    # to whatever floor existed.
    #
    # The missing physics is that the tail's drag should be evaluated where the
    # tail is LOADED -- the forward-CG trim case at CLhfwd = 0.65, which is
    # what sizes its area. Until that is added, removing the floor lets AR_ht
    # find its structural optimum rather than sit on an arbitrary number.
    # cons += [ht.AR_ht >= 4.]
    if pi_tail:
        # The pi-tail's horizontal sits ON the fins, so its trailing edge is
        # placed by walking up the fin leading edge and back along the
        # horizontal's own sweep. A cantilever tail mounts at the fuselage and
        # is placed by the tail-arm and CG rows already above, so this
        # constraint has no counterpart for it.
        cons += [
            ht.dx_trail_ht <= (vt.dx_lead_vt + vt.b_vt / tan(SWEEP_VT * pi / 180)
                               + fu.w_fuse / tan(SWEEP_HT * pi / 180)
                               + ht.c_root_ht),
        ]
    cons += [
        # ---- tail sizing: TASOPT's htsize, not Kirschen's lumped row -------
        #
        # SPaircraft sizes the tail with ONE inequality,
        #     SM_min + dx_CG/mac + c_m_w/CL_max <= V_ht*(m_ratio + CLh_max/CL_max)
        # in which dx_CG (6 ft, pinned) and c_m_w (1.9, unexplained) between
        # them supply ~90% of the requirement. Replacing c_m_w with its
        # physical magnitude alone took V_ht from 0.61 to 0.20 -- a 10 m2
        # horizontal tail on a 737 -- so that row is calibrated around the
        # number rather than describing the physics.
        #
        # TASOPT (balance.f, htsize) uses TWO conditions instead, and solves
        # them together for tail area and wing position:
        #
        #   Row 1  trim at the FORWARD CG limit, wing at max lift and tail at
        #          max download. This is what sets tail AREA.
        #   Row 2  the AFT CG limit must sit SM_min*cma ahead of the neutral
        #          point. This is what sets wing POSITION.
        #
        # and it derives the CG limits from cabin loading (cglpay) rather
        # than assuming a travel.
        #
        # Row 1, forward-CG trim -- TASOPT's cCM residual, not a hand
        # simplification of it.
        #
        #   cCM = co*CMw0 + (co*CMw1 - xwbox)*(CL - CLh*Sh/S)
        #       + coh*CMh0*Sh/S + (coh*CMh1 - xhbox)*CLh*Sh/S
        #       + CMVf1*(CL - CLMf0)/S
        #   trimmed when  cCM/CL + xCG = 0,  i.e.  xCG = -cCM/CL
        #
        # written with CMw0 = -cmw and CLh = -CLhfwd since both are negative
        # (nose-down wing moment, tail download) and an SP carries magnitudes.
        # Every term is linear in Sh, so this is a signomial equality.
        #
        # My earlier hand version -- V_ht*CLh >= (xAC - xCG)/mac*CL + cmw --
        # gives V_ht = 1.38 on TASOPT's own numbers where TASOPT gives 0.575:
        # a 2.4x overestimate, because it drops the tail's own moment, the
        # fuselage term, and every Sh/S correction.
        # (x_AC - x_CG), not the other way round. At the forward CG the wing
        # AC lies BEHIND the CG, so wing lift makes a nose-down moment and the
        # tail balances it with DOWNLOAD. Writing the difference the wrong way
        # round makes the requirement negative for any normal forward CG, and
        # the tail simply vanishes -- which is exactly what happened: V_ht went
        # to 0.0000 and S_ht to 0.00 m2 while the model still "converged".
        # Grouped by DERIVING the expansion rather than eyeballing it. With
        # CMw0 = -cmw, CMh0 = -cmh0, CMh1 = -cmh1 and CLh = -clh (all four are
        # genuinely negative; an SP carries magnitudes), cCM = -CL*xCG
        # multiplied through by S separates into:
        #
        #   LHS  co*CMw1*CL*S + co*CMw1*clh*Sh + coh*cmh1*clh*Sh
        #        + xhbox*clh*Sh + CMVf1*CL + CL*xCG*S
        #   RHS  co*cmw*S + xwbox*CL*S + xwbox*clh*Sh + coh*cmh0*Sh
        #        + CMVf1*CLMf0
        #
        # My previous transcription had co*cmw*S on the LHS and xhbox*clh*Sh
        # on the RHS -- both on the wrong side, which is why the tail vanished
        # a second time.
        #
        # x_hbox is the horizontal tail BOX station, roughly the quarter chord
        # aft of its leading edge, not the tail's CG.
        # CMw1 IS NEGATIVE, and the two c_root*CMw1 terms have therefore moved
        # to the other side. This row solves for (x_wing - co*CMw1), which is
        # TASOPT's (xwbox - co*CMw1) -- the wing's effective AERODYNAMIC
        # reference, not the box station. With CMw1 held at +0.015 that
        # distinction collapsed: co*CMw1 was 0.098 m, so the row placed the
        # BOX where the AC belongs and the box slid aft to compensate.
        (xCGfwd * CLpmax * wing.S
         + ht.c_root_ht * CMh1 * CLhfwd * ht.S_ht
         + (xCG[Nclimb] + ht.dx_lead_ht + 0.25 * ht.c_root_ht)
           * CLhfwd * ht.S_ht
         + CMVf1 * CLpmax)
        == (wing.c_root * cmw_land * wing.S
            + fu.x_wing * CLpmax * wing.S
            + fu.x_wing * CLhfwd * ht.S_ht
            + wing.c_root * CMw1 * CLpmax * wing.S
            + wing.c_root * CMw1 * CLhfwd * ht.S_ht
            + ht.c_root_ht * CMh0 * ht.S_ht
            + CMVf1 * CLMf0),                                 # [SP] SigEq
        # Row 2, aft-CG stability -- with TASOPT's NEUTRAL POINT, not
        # SPaircraft's approximation.
        #
        # This was the last piece of htsize still un-ported, and it is what
        # sets WING POSITION. SPaircraft uses a Unified-course correlation,
        #     xNP/mac/V_ht*(AR+2)*(1+2/AR_ht) == (1+2/AR)*(AR-2)
        # which knows nothing about the fuselage or the engines. TASOPT builds
        # it from the moment slopes:
        #
        #   xNP = (co*CMw1 - xwbox)*(dCLhdCL*Sh/S - 1)
        #       - (coh*CMh1 - xhbox)*dCLhdCL*Sh/S
        #       - CMVf1/S - neng*xengcp*dCLndCL*Afan/S
        #
        # The consequence of the omission was measurable: the wing came out
        # 0.80 m aft of the forward CG against TASOPT's 2.11 m, and since that
        # arm is the dominant term in the trim requirement, the tail followed
        # it down -- V_ht 0.295 against 0.575, in almost exactly that ratio.
        #
        # Grouped from the expansion (CMh1 = -cmh1), times S:
        # Both c_root*CMw1 terms swap sides here too, for the same reason: the
        # expansion carries (co*CMw1 - xwbox) and CMw1 is negative, so each
        # appearance changes sign with it.
        (xNPt * wing.S + fu.x_wing * dCLhdCL * ht.S_ht
         + wing.c_root * CMw1 * dCLhdCL * ht.S_ht + CMVf1)
        == (wing.c_root * CMw1 * wing.S
            + fu.x_wing * wing.S
            + ht.c_root_ht * CMh1 * dCLhdCL * ht.S_ht
            + (xCG[Nclimb] + ht.dx_lead_ht + 0.25 * ht.c_root_ht)
              * dCLhdCL * ht.S_ht),                            # [SP] SigEq
        xCGaft + SMmin * wing.mac <= xNPt,
        # Reported for continuity; no longer sizes anything.
        # EQUALITY. As a one-sided bound this was a dead variable: nothing else
        # in the model reads dx_CG, so it floated to 3.0e+16 m. Harmless to the
        # design but not harmless to the solver -- SIA works in log space, and
        # a variable sixteen orders adrift is a large excursion for the
        # conditioning to carry -- and lethal to anything that reports it.
        dxCG == xCGaft - xCGfwd,                              # [SP] SigEq
    ]

    # ---- mission stitching ------------------------------------------------------
    cons += [
        W_start[0] == W_total,
        st.hft[0] == dhft[0],
        # Cruise altitude floor REMOVED. It was inherited from SPaircraft, it
        # was the second-hardest-binding row in the model (dual 0.39), and it
        # was pinning the one variable this study was asked to optimise.
        #
        # It could not simply be deleted before: the atmosphere was
        # troposphere-only, so letting the optimiser climb meant letting it
        # climb into air the model got wrong (and wrong in the flattering
        # direction -- extrapolated lapse gives falsely cold, dense, low-
        # viscosity air). With the two-layer ISA in flight_state.py the band is
        # modelled properly to 55,000 ft and the floor has nothing left to do.
        #     st.hft[Nclimb - 1] >= MinCruiseAlt,
        RC[0] >= 2500. * units.ft / units.min,
        theta[Nclimb - 1] >= 0.015,
        # Engine-out thrust at SEA LEVEL, not at 12,379 ft. This is the whole
        # reason the fin was half-size: the yawing moment is proportional to
        # the thrust of the failed engine, and it was being computed with
        # climb thrust at altitude for an event that happens on the runway.
        vt.T_e == Fsafetyfac * F_TO,
        W_dry + fu.W_payload + ReserveFraction * W_fprimary <= W_end[N - 1],
        W_fclimb >= f.sum(W_burn[:Nclimb]),
        W_fcruise >= f.sum(W_burn[Nclimb:]),
        f.sum(Rseg) >= ReqRng,
        # EQUALITIES. These are the distances from the nose and main gear to the
        # CG -- geometry, not quantities with slack.
        #
        # Written as >= they are bounded below only, and inflating them pays:
        # the nose gear load is L_n == W*dx_m/B, so a dx_m larger than the
        # actual gear-to-CG distance manufactures nose gear load out of
        # nothing. With the forward gear margin (L_n >= 0.08 W) imposed on top,
        # the optimizer satisfied it by stretching dx_m and then parked the
        # nose gear at x_n = 1.3e-09 -- the tip of the nose.
        #
        # Same defect as `mac`, `l_ht`, `l_vt`, the 25.107 speeds, the stall
        # speeds and the tail-strike row: an identity written as a one-sided
        # bound, harmless until something makes it pay.
        lg.dx_n + lg.x_n == xCG[Nclimb],                      # [SP] SigEq
        lg.dx_m + xCG[Nclimb] == lg.x_m,                      # [SP] SigEq
        lg.x_m >= lg.tan_phi * (lg.z_CG + lg.l_m) + xCG[Nclimb],
        # Mission caps bypass ratio rather than fixing it; subs/optimalD8.py
        # supplies neither alpha_max nor alpha_OD, so both are free.

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
    if getattr(arch, "lock_mach", False):
        # Cruise at the real aircraft's design Mach rather than the
        # fuel-optimal one. Climb stays free -- only the cruise segments are
        # a specification.
        cons += [st.M[i] == size_class.ref_mach for i in range(Nclimb, N)]

    # ---- energy source to shaft ------------------------------------------------
    # The propulsor and the thing feeding it were built independently, and
    # nothing links them by construction: add_powertrain asks for shaft power
    # and add_fuel_cell/add_battery supply electrical power, with the
    # drivetrain in between. Without these rows both models BUILD happily and
    # then size to their lower bounds, which looks like a converged answer and
    # is not one.
    if electric:
        eta_i, eta_m, eta_c = pt.eta_inv, pt.eta_mot, pt.eta_cbl
        for i in range(N):
            if fc is not None:
                # Stack electrical power, through inverter, motor and cables.
                cons += [fc.P_e[i] * eta_i * eta_m * eta_c
                         >= pt.P_shaft[i]]
            if batt is not None:
                # The pack is sized by BOTH: peak power (takeoff/climb) and
                # total energy (the whole mission). Which one binds is the
                # design outcome the battery model exists to expose.
                cons += [batt.P_peak * eta_i * eta_m * eta_c
                         >= pt.P_shaft[i]]
        if batt is not None:
            cons += [batt.E_req * eta_i * eta_m * eta_c
                     >= ReserveFraction * f.sum(
                         [pt.P_shaft[i] * thr[i] for i in range(N)])]

    # ---- ceiling ---------------------------------------------------------------
    # Two corrections, after two self-inflicted infeasibilities.
    #
    # 1. NO TROPOPAUSE CAP. flight_state.py is the standard TROPOSPHERE and
    #    keeps lapsing above 11 km, so it is too cold up there -- 9.7 K at
    #    41 kft. Capping at the tropopause looked like the honest fix, but
    #    SPaircraft REQUIRES cruise >= MinCruiseAlt = 38,478 ft = 11.73 km.
    #    The cap and that floor are a flat contradiction, and every small-class
    #    case went infeasible on it.
    #
    #    The extrapolation is inherited from SPaircraft, whose verified
    #    reference point cruises there with this same atmosphere. It biases
    #    every architecture identically -- air too cold, so too dense, so
    #    lift and drag at altitude are both optimistic -- which a comparison
    #    ACROSS architectures tolerates far better than an absolute number
    #    would. Fixing it needs a piecewise atmosphere: a monomial fits the
    #    isothermal layer to 4.7% over 9-15 km, but stretched across climb
    #    too (3-16 km) the same form is 39% out, so one power law cannot do
    #    both layers and the model has no way to pick a regime.
    #
    # 2. NO "CRUISE MUST NOT CLIMB" ROW EITHER, and this one was wrong on the
    #    physics as well as infeasible. Infeasible because altitude steps as
    #    hft[i] == hft[i-1] + dhft[i] and GP variables are STRICTLY POSITIVE,
    #    so dhft can never be zero and cruise altitude must strictly increase;
    #    forbidding that is a contradiction, not a constraint.
    #
    #    Wrong because cruise-climb is exactly right for a fuel burner. The
    #    ceiling RISES as the aircraft burns down, and step climb is the
    #    aircraft following it. The existing 1.5% gradient at top of climb --
    #    the HEAVIEST cruise point, and about 700 ft/min, far above the
    #    100 ft/min service ceiling and the 0 ft/min absolute ceiling -- is
    #    already the binding check, and everything after it is lighter.
    #
    #    A BATTERY aircraft is the exception, and it is a real architectural
    #    difference: it does not get lighter, so its ceiling does not rise
    #    and it has no business climbing through cruise.
    if arch.battery:
        for i in range(Nclimb, N - 1):
            cons += [st.hft[i + 1] <= st.hft[i] * 1.0001]

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
        (fu, "n_pass", size_class.n_pass, None),
        (fu, "W_cargo", 0.1, units.N),
        (fu, "w_db", 0.93 if arch.double_bubble else 1e-4, units.m),
        # A_vt freed: fin aspect ratio is a design variable, and the fin is the
        # surface this model sizes worst. It is not a free lunch -- the box in
        # wingbox.py is handed the DOUBLED span, so a higher aspect ratio costs
        # structural weight, which is the trade that should decide it.
        # V_1 is no longer pinned per class -- it is tied to the aircraft's own
        # takeoff stall speed below, which is what FAR 25.149 actually says.
        # c_l_vt_EO STAYS pinned at 0.5, and the experiment that says so is
        # worth recording. Freed with a 1.0 cap it went straight to the cap and
        # the fin collapsed -- 19.1 m2 to 11.1 m2, V_vt 0.066 to 0.034, against
        # a real 26.4 m2 and ~0.089.
        #
        # 0.5 is not really a lift coefficient here; it is a conservative
        # stand-in for the fin sizing cases this model does not contain. A
        # transport fin is sized by V_MC (FAR 25.149: directional control at
        # minimum control speed, with rudder deflection and a 5 degree bank
        # limit) and by crosswind landing. All this model has is a steady
        # engine-out moment balance and a landing yaw rate, neither of which
        # knows what a rudder is. Until V_MC is modelled properly, the low
        # coefficient is carrying that absence, and freeing it removes the
        # compensation without supplying the physics.
        # c_l_vt_EO STAYS pinned at 0.5 -- and 0.5 is TASOPT's own number for
        # this exact quantity (runs/737/737s.tas:237,
        # "CLveout  VT CL at engine-out trim").
        #
        # Tested twice, before and after V_MCG was added, and it behaves the
        # same both times: given a 1.0 cap it goes straight to the cap and the
        # fin halves (V_vt 0.091 -> 0.047, area 20.7 -> 11.8 m2). It is not
        # really a lift coefficient, it is a control-margin allowance standing
        # in for rudder authority, sideslip and handling requirements that
        # neither this model nor TASOPT represents. Freeing it removes the
        # allowance without supplying the physics.
        (vt, "c_l_vt_EO", 0.5, None),
        # e_vt STAYS pinned, and deliberately. It is a span efficiency, not a
        # design variable, and the obvious way to derive it -- the Nita-Scholz
        # taper fit the horizontal tail uses -- is wrong here: that correlation
        # is for a full-span wing, and applied to a fin it returns e = 0.996
        # because it knows nothing about the endplate effect of the fuselage
        # and tailplane. 0.8 is the standard book value for a fin including
        # that effect. Freeing it without the right correlation would just let
        # the optimizer buy a smaller fin at no cost.
        (vt, "e_vt", 0.8, None),
        (vt, "lambda_vt", 0.3, None),
        # 1.225, not the 1.23 that subs/optimalD8.py specifies: vertical_tail.py
        # and wing.py declare rho with 1.225 baked in, and the solved reference
        # carries 1.225 for \rho_{TO}, \rho_0 and \rho_{T/O} alike. The
        # substitution does not take. Matching the reference, not the subs dict.
        (vt, "rho_TO", 1.225, units.kg / units.m ** 3),
        (ht, "lambda_ht", 0.3, None),
        # C_L_ht_fCG removed: it was declared, pinned at 0.85, and used in
        # exactly zero constraints -- vestigial from SPaircraft's original tail
        # sizing, which the TASOPT htsize port replaced. The forward-CG tail
        # lift coefficient that actually does work is CLhfwd = 0.65, TASOPT's
        # value, and having a second pinned number claiming to be the same
        # quantity was purely a trap for the next reader.
        # Landing gear geometry, SCALED with the aircraft.
        #
        # These four are substitutions from subs/optimalD8.py and every one is
        # a 180-passenger D8.2 dimension. Left pinned they say a 787 has the
        # same 1.0 m cargo hold and the same 2.0 m CG height as an 8-seat
        # business jet, which is how the 787 column came to be blocked: the
        # elastic Phase I put `LG_h_hold` at the largest dual in the whole
        # report (0.678), i.e. the single row most responsible for the model
        # not closing. That is a hard-coded constant, not a physical limit.
        #
        # Scaled linearly on fuselage radius against the 180-passenger class,
        # which is the configuration the original substitutions describe.
        (lg, "z_CG", 2.0 * _gear_scale, units.m),
        (lg, "z_wing", 0.5 * _gear_scale, units.m),
        (lg, "h_hold", 1.0 * _gear_scale, units.m),
        # Nacelle wall thickness for gear clearance -- scales with the engine.
        (lg, "t_nacelle", 0.15 * _gear_scale, units.m),
    ]:
        cons.append(handle[name] == (value * unit if unit is not None else value))
        # START a pinned variable AT its pin.
        #
        # A substitution is a hard equality, so any other starting value is a
        # violation by construction -- and in log space a SMALL pin is a large
        # one. w_db is pinned to 1e-4 for a single-bubble hull while the
        # component guesses 0.5: ln(0.5/1e-4) = 8.5, which was the worst
        # violation left in the whole model once the stale tail was out of the
        # seed. Seeding the pin costs nothing and removes the class.
        try:
            handle[name].set_value(float(value), skip_validation=True)
        except Exception:
            pass

    _br = _burn_row()
    if _br is not None:
        cons += [W_burn >= _br]

    if tank:
        cons += [
            # The tank holds the whole fuel load -- there is no wing tank in
            # any hydrogen architecture here.
            tank.W_fuel >= W_ftotal,
            # Insulation and a clearance gap eat into the cabin radius.
            fu.R_fuse >= tank.R_o + tank.t_insul + tankclear,
        ]

    # surfcm, for an unbroken wing. Both rows carry a (1 - eta) difference,
    # so both are signomial equalities -- which is what an SP is for.
    cons += [
        eta_surf * wing.b == 2.0 * fu.w_fuse,
        Kc_cm + 0.5 * (1.0 + wing.lambda_) * eta_surf
            == eta_surf + 0.5 * (1.0 + wing.lambda_),          # [SP] SigEq
        3.0 * cmw * Kc_cm + cm_sec * (1.0 + wing.lambda_
                                      + wing.lambda_ ** 2)
            * eta_surf * wing.cos_Lambda ** 4
            == cm_sec * (1.0 + wing.lambda_ + wing.lambda_ ** 2)
               * wing.cos_Lambda ** 4,                          # [SP] SigEq
        # The same CM0 relation at the LANDING section moment. Identical
        # geometry, different c_m -- TASOPT re-runs surfcm per mission case
        # for exactly this reason, and the forward-CG tail sizing case is the
        # flaps-down one.
        3.0 * cmw_land * Kc_cm + cm_sec_land * (1.0 + wing.lambda_
                                                + wing.lambda_ ** 2)
            * eta_surf * wing.cos_Lambda ** 4
            == cm_sec_land * (1.0 + wing.lambda_ + wing.lambda_ ** 2)
               * wing.cos_Lambda ** 4,                          # [SP] SigEq
        # ---- surfcm CM1, the other half of the port ------------------------
        # CM0 above was ported; CM1 was left as a constant, and CM1 is the
        # term that carries the box-to-aerodynamic-centre offset. See the
        # note at the C_M_w1 declaration for what that cost.
        #
        # For an unbroken wing (etas = etao, lambdas = 1) surfcm collapses to
        #     Kc = etao + (1/2)(1+lt)(1-etao)              [Kc_cm, above]
        #     Ko = 1/(AR*Kc)
        #     Kp = etao(1+fLo) + (1/2)(gs+gt)(1-etao) - 2|fLt| gt lt/(AR Kc)
        #     C1 = (gs + (1/2)(gt + gs lt) + lt gt)(1-etao)
        #     C2 = (gs + 2 gt)(1-etao)^2
        # with gs = rcls (lambdas = 1) and gt = lt*rclt.
        #
        # CM1 is NEGATIVE, so CMw1 carries its magnitude and the dominant
        # sweep term -(tanL/Ko)C2/12 becomes the positive side. Note 1/Ko is
        # AR*Kc, so no reciprocal is needed. Every term below is positive,
        # which is why this is only a signomial equality and not worse.
        gam_t == wing.lambda_ * rclt_cm,
        C1_cm + (gam_s + 0.5 * (gam_t + gam_s * wing.lambda_)
                 + wing.lambda_ * gam_t) * eta_surf
            == (gam_s + 0.5 * (gam_t + gam_s * wing.lambda_)
                + wing.lambda_ * gam_t),                        # [SP] SigEq
        C2_cm + (gam_s + 2.0 * gam_t) * 2.0 * eta_surf
            == (gam_s + 2.0 * gam_t) * (1.0 + eta_surf ** 2),   # [SP] SigEq
        Kp_cm + 2.0 * fLt_a * gam_t * wing.lambda_
                / (wing.AR * Kc_cm)
                + 0.5 * (gam_s + gam_t) * eta_surf
            == eta_surf * (1.0 + fLo_cm)
               + 0.5 * (gam_s + gam_t),                         # [SP] SigEq
        CMw1 * Kp_cm
            + eta_surf * (1.0 + fLo_cm) * Xax_m
            + Xax_m * wing.cos_Lambda ** 2 * C1_cm / 3.0
            + fLt_a * wing.lambda_ * gam_t * wing.tan_Lambda
            == wing.tan_Lambda * wing.AR * Kc_cm * C2_cm / 12.0
               + 2.0 * fLt_a * wing.lambda_ ** 2 * gam_t * Xax_m
                 * wing.cos_Lambda ** 2 / (wing.AR * Kc_cm)
               + fLt_a * wing.lambda_ * gam_t * wing.tan_Lambda
                 * eta_surf,                                    # [SP] SigEq
    ]

    # Maximum wing C_L falls with sweep: the 2.15 is a section value and the
    # wing sees it through cos^2(Lambda). Pinned it was consistent with the
    # D8.2's 13.237 degrees only.
    cons += [CLwmax * wing.cos_Lambda ** 2 == 2.15]

    # ---- CG envelope from cabin loading (TASOPT cglpay) ----------------------
    # cglpay asks how much payload can be loaded from each end of the cabin
    # before the CG stops moving that way, and takes the two roots of the
    # resulting quadratic. That quadratic does not survive an SP cleanly, so
    # this uses the standard preliminary-design bracket instead: a fraction
    # r_pay_limit of the payload concentrated in the forward or aft part of
    # the cabin, against the empty aircraft. Same idea -- the envelope comes
    # out of cabin geometry and payload distribution rather than being a
    # number in feet -- with a simpler worst case.
    cons += [
        # Payload centroid at xcabin -/+ x_pay_off*lcabin, as cglpay places it.
        #
        # SENSE MATTERS, and both of these were backwards. A forward CG limit
        # demands MORE tail and an aft limit demands MORE stability, so the
        # optimiser wants the forward limit further aft and the aft limit
        # further forward -- i.e. it wants to shrink the envelope. Writing
        # x_CG_fwd with a >= let it do exactly that: the forward limit slid
        # aft until the trim row was satisfied with NO TAIL AT ALL (S_ht =
        # 0.00 m2, reported as converged).
        #
        # The forward limit must therefore be bounded ABOVE by the loading
        # calculation, and the aft limit BELOW. Then the envelope can only be
        # as wide as the cabin says, never narrower.
        # The EMPTY moment, which is what cglpay works from. x_CG is the
        # LOADED CG -- its buildup already carries payload at mid-cabin and
        # fuel at the wing box -- while W_dry excludes payload, so pairing the
        # two mixed a loaded centroid with an empty weight. The payload then
        # appeared twice: once buried at 0.5*l_fuse inside x_CG, and again as
        # the partial load at the cabin ends. That drags the forward limit aft
        # and under-sizes the tail, independently of anything else.
        xCGe * W_dry >= (xmisc * Wmisc + lg.x_CG_lg * lg.W_lg
                         + 0.5 * fu.W_fuse * fu.l_fuse
                         + ht.W_ht * ht.x_CG_ht + vt.W_vt * vt.x_CG_vt
                         + numeng * Wengsys * xeng
                         + wing.W_wing * (fu.x_wing + wing.dx_AC_wing)),
        # ---- CG envelope: TASOPT cglpay, solved rather than sampled --------
        # cglpay (balance.f:633) loads the cabin CONTIGUOUSLY FROM ONE END --
        # front for the forward extreme, rear for the aft one -- at zero fuel,
        # and solves for the payload fraction that maximises the excursion.
        # The worst case is INTERIOR: a full cabin has its centroid at the
        # middle and barely excurses, an almost-empty one can be bunched at an
        # extreme but weighs nothing against the empty aircraft, and the
        # maximum sits between. TASOPT takes the root of the quadratic that
        # falls out of d(xcg)/d(rpay) = 0.
        #
        # Two earlier attempts at this were both wrong. Freezing rpay at 0.457
        # cannot be a worst case, since the critical fraction depends on the
        # whole weight breakdown and moves every design iteration. Replacing
        # that with a SWEEP of fixed fractions gave only one-sided bounds, and
        # the bounds went slack: x_CG_fwd is also an unknown of the forward
        # trim equality (the htsize row), so with nothing pinning it from the
        # other side the optimiser co-solved it with S_ht and parked it at
        # 14.25 m -- 2.3 m ahead of ANY loading the sweep could produce, and
        # 1.06 m ahead of the front spar. That is the same defect class as the
        # rest: a quantity the optimiser can move for free.
        #
        # The quadratic does not need a radical. Writing the loaded CG as
        #     xcg(r) = [xpay(r)*Wpay*r + xWe] / [Wpay*r + We]
        # with the contiguous-loading centroid xpay(r) = xcabin + s*L*(1-r),
        # s = +/-1/2, the stationarity condition d(xcg)/dr = 0 reduces to
        #     xcg = xcabin + s*L*(1 - 2r)
        # which is LINEAR: the forward case is just xcg = x_shell1 + L*r and
        # the aft case xcg = x_shell2 - L*r. Paired with the CG definition
        # itself that is two signomial equalities and one new variable per
        # side, and it reproduces cglpay exactly. The quadratic's other root
        # is negative (-5.7 on the 737), so SP positivity discards it for free
        # and no root selection is needed.
        #
        # Hand-check against this model's own weights: rF = 0.4683 -> 16.583 m
        # and rB = 0.4508 -> 18.680 m, travel 2.10 m against TASOPT's 2.34.
        xCGfwd == fu.x_shell1 + fu.l_shell * rpayF,          # [SP] SigEq
        xCGfwd * (W_dry + rpayF * fu.W_payload)
        == xCGe * W_dry
           + fu.x_shell1 * fu.W_payload * rpayF
           + 0.5 * fu.l_shell * fu.W_payload * rpayF ** 2,   # [SP] SigEq
        # Aft: centroid measured forward from x_shell2, so the cross term
        # moves to the LHS to keep both sides posynomial.
        xCGaft + fu.l_shell * rpayB == fu.x_shell2,          # [SP] SigEq
        xCGaft * (W_dry + rpayB * fu.W_payload)
        + 0.5 * fu.l_shell * fu.W_payload * rpayB ** 2
        == xCGe * W_dry
           + fu.x_shell2 * fu.W_payload * rpayB,             # [SP] SigEq
        rpayF <= 1.0,
        rpayB <= 1.0,
        xCGaft >= xCGfwd,
    ]

    # ---- fuselage shape, freed ------------------------------------------------
    # These three were substitutions in subs/optimalD8.py, and each is
    # UNBOUNDED once the pin is removed, in a different direction:
    #
    #   l_nose      nose wetted area grows with length -> wants zero
    #   lambda_cone l_cone = R/lambda -> lambda wants infinity, cone -> zero
    #   h_floor     A_floor ~ 1/h_floor -> a deeper beam is always lighter,
    #               so it wants infinity
    #
    # Pinning them hid that. Freeing them means supplying the physics that
    # was doing the work, which is what these rows are. None of them is a
    # bound for the solver's benefit -- each is a real geometric limit.
    cons += [
        # NOSE LENGTH IS AN INPUT, the second of TASOPT's two nose numbers
        # (xshell1, deck line 294 = 17.0 ft = 5.182 m) alongside xlgnose.
        #
        # It used to be a floor standing in for a cockpit, backing up the
        # 1.2-calibre fineness rule. With that rule gone this was the only
        # thing left setting nose length, and it was a placeholder 4.0 m that
        # only the Citation had ever overridden -- so the 737 sized its nose
        # off a made-up number, came out 1.18 m short, and took the fuselage
        # with it (l_fuse 0.940 of the real aircraft, failing the check, while
        # the tailcone was if anything too LONG at 7.21 m against TASOPT's
        # 6.71). The whole deficit was in the nose.
        #
        # Equality rather than a floor for the same reason as x_lg_nose: this
        # is a specified station, not a limit the optimiser trades against.
        fu.l_nose == size_class.l_nose_min_m * units.m,       # [SP] SigEq
        # Nose fineness. Below about 1.2 calibre the flow separates and the
        # drag model here -- which has no separation term -- stops being
        # honest, so this keeps the optimiser inside the fits validity.
        # EQUALITY, not a floor. Nose length is a fixed geometric INPUT in
        # TASOPT (xshell1); here it was free above a 1.2-calibre floor, and
        # once the CG envelope was put on cglpay's footing the optimiser found
        # it: x_CG_fwd = x_shell1 + l_shell*r_F, so pushing the nose out moves
        # the forward loading extreme aft and relieves the trim demand. The
        # 737's nose went to 10.71 m against a 4.45 m floor, with the floor
        # left slack (g = -0.878) and my forward-CG row pulling it at a
        # shadow price of 2.32.
        #
        # It is NOT a harmless translation of the whole aeroplane: the cabin
        # and payload move aft by the full stretch while the fuselage's own
        # centroid term (0.5*W_fuse*l_fuse) moves only half as far, so the
        # optimiser buys a real shift of payload against wing and pays only
        # structure for it. Tying nose length to diameter removes the lever
        # and still scales by class -- 1.2 calibres puts the 737's cabin start
        # at 4.45 m, which is where it actually is.
        # REMOVED: `l_nose >= 1.2 * 2 * R_fuse`.
        #
        # It was a drag-fit validity fence wearing geometry's clothes -- the
        # nose polars carry no separation term, so a blunt nose leaves the
        # data they were fitted to. That is a reason to WATCH the fineness
        # ratio, not to let it set a length.
        #
        # Nose length is an INPUT in TASOPT (xlgnose), and it is set by what
        # the nose contains -- cockpit, avionics bay, radar, gear bay -- none
        # of which scales with fuselage radius. The per-class floor at
        # SizeClass.l_nose_min_m is the honest control and remains below.
        #
        # What actually drives the length is the gear: the 15% nose-load
        # maximum wants a long wheelbase, which wants the nose gear forward,
        # which through `x_n >= l_nose` wants a SHORT nose. So it runs down
        # to whatever floor exists.
        # Tailcone at least two calibres long, for rotation clearance. This
        # is a genuine trade rather than a formality: a shorter cone shortens
        # the tail arm, which the tail volume coefficients pay for in tail
        # area.
        fu.l_cone >= 2.0 * fu.R_fuse,
        # Floor beam depth cannot exceed the space under the floor.
        fu.h_floor <= 0.10 * fu.R_fuse,
    ]

    cons += _bound_constraints(f)
    f.ConstraintList(cons)
    _bound_variables(f)
    if seed == "reference":
        _seed_from_reference(f)
        if wing_model == "tasopt":
            _seed_cranked_wing(f)
    return f


#: Variables NOT to import from reference.json.
#:
#: The reference is the gpkit optimum of the model as it was, and its
#: horizontal tail is degenerate: tau_ht = 3.45e-10 (no thickness), I_cap and
#: M_r sitting on the 1e-30 box floor, and M_r_out at 2.66e11 -- the two sides
#: of the pi-tail moment subtraction annihilating each other, which is the
#: behaviour the pinned-support branch documents.
#:
#: That degeneracy has since been fixed (thickness floor, fixed supports, a
#: cantilever branch for conventional tails), so those values are now a
#: solution to a model that no longer exists. Importing them starts the solve
#: 10^24 outside the feasible set -- the tau floor alone is violated by
#: ln(0.08/3.45e-10) = 19.3 -- and no Phase I formulation recovers from that.
#: Every "phase 1 cannot find a feasible point" in this project traces here.
#:
#: The hand-written guesses in the component models are physical. Use them.
#: The whole horizontal tail. Skipping tau_ht and the box alone was not
#: enough: the degeneracy propagated into everything downstream of it. The
#: drag polar is CD0h^6.49 >= sum(... tau^0.912 ...), and the reference
#: evaluated it at tau = 3.45e-10, so its CD0h belongs to a tail with no
#: thickness. Restore a physical tau and the right-hand side jumps by
#: (0.12/3.45e-10)^0.912 = 6.2e7 -- ln of which is 17.9, matching the 18.2
#: violation that was left. Same story for the tail's lift, drag and
#: Reynolds number.
#:
#: The horizontal tail is also the component this project changed most
#: (cantilever branch, thickness floor, fixed supports, TASOPT trim and
#: neutral point), so its reference values are the least transferable in the
#: file. Take the component model's own guesses for all of it.
#: Fuse_w_db is architecture-dependent, not a degeneracy: the reference is a
#: double bubble with a 0.93 m web and a conventional hull pins it to ~0, so
#: importing it starts a round fuselage 9300x wide at the web -- ln(9300) =
#: 9.14, which was the worst remaining violation once the tail was clean.
#: Anything the ARCHITECTURE switches should come from the architecture, not
#: from a seed recorded for a different one.
_SEED_SKIP = ("HT_", "Fuse_w_db")


def _seed_from_reference(f):
    """Initialise every variable from the recorded gpkit optimum.

    Except the degenerate horizontal tail -- see ``_SEED_SKIP``.
    """
    from pyomo.core.base.var import IndexedVar
    from components.crosscheck import reference_point
    ref = reference_point(Path(__file__).parent / "components" / "reference.json")
    n = 0
    for v in f.get_variables():
        items = ([(f"{v.name}[{i}]", v[i]) for i in v.index_set()]
                 if isinstance(v, IndexedVar) else [(v.name, v)])
        for nm, vd in items:
            base = nm.split("[")[0]
            if any(base.startswith(sk) for sk in _SEED_SKIP):
                continue
            idx = int(nm.split("[")[1].rstrip("]")) if "[" in nm else None
            val = ref.get(base)
            if val is None:
                continue
            if isinstance(val, list):
                val = val[idx] if idx is not None and idx < len(val) else val[0]
            vd.set_value(float(val), skip_validation=True)
            n += 1
    return n


def _seed_cranked_wing(f):
    """Make the cranked planform consistent with the seeded point.

    reference.json records a converged SINGLE-taper wing -- 76 Wing_ entries,
    including S, b, c_root, lambda and mac -- and knows nothing about
    lambda_s, K_c, K_mac, c_break or the chord slopes, which therefore start
    at their declaration guesses. The two are not consistent: at the seed
    S == c_o*b*K_c is off by 4% and mac*S == b*c_o^2*K_mac by 7%, so several
    signomial equalities begin violated and phase 1 starts by trying to
    reconcile a planform that does not exist. It wandered to a rectangular
    20 m-span wing at 71 degrees of sweep.

    So derive them instead of guessing. Everything below follows from the
    seeded S, b, c_o, lambda and eta_o, with lambda_s chosen to satisfy the
    area relation exactly:

        K_c = S/(c_o*b)
        K_c = eta_o + (eta_s-eta_o)/2 + lam_t(1-eta_s)/2 + lam_s(1-eta_o)/2

    solved for lam_s. mac is then RE-seeded from the cranked integral rather
    than left at the single-taper value, because on this planform the recorded
    one is simply the wrong number.
    """
    from components.wing_tasopt import ETA_BREAK

    def g(name):
        v = getattr(f, name, None)
        try:
            return float(v.value) if v.value is not None else None
        except Exception:
            try:
                return float(v[0].value)
            except Exception:
                return None

    S, b, co = g("Wing_S"), g("Wing_b"), g("Wing_c_root")
    lam_t, eo = g("Wing_lambda"), g("Wing_eta_o")
    if None in (S, b, co, lam_t, eo):
        return 0
    es = ETA_BREAK
    Kc_req = S / (co * b)
    lam_s = ((Kc_req - eo - (es - eo) / 2.0 - lam_t * (1.0 - es) / 2.0)
             * 2.0 / (1.0 - eo))
    lam_s = min(max(lam_s, lam_t), 1.0)          # keep it a real planform
    Kc = (eo + (1 + lam_s) * (es - eo) / 2.0
          + (lam_s + lam_t) * (1 - es) / 2.0)
    Kmac = (eo + (es - eo) * (1 + lam_s + lam_s ** 2) / 3.0
            + (1 - es) * (lam_s ** 2 + lam_s * lam_t + lam_t ** 2) / 3.0)
    Dinn = (co - lam_s * co) / ((b / 2.0) * (es - eo))
    Dout = (lam_s * co - lam_t * co) / ((b / 2.0) * (1.0 - es))
    tanL = g("Wing_tan_Lambda") or 0.36
    vals = {"Wing_lambda_s": lam_s, "Wing_c_break": lam_s * co,
            "Wing_K_c": Kc, "Wing_K_o": 1.0 / Kc, "Wing_K_mac": Kmac,
            "Wing_mac": b * co ** 2 * Kmac / S,
            "Wing_dc_dy_inn": Dinn, "Wing_dc_dy_out": Dout,
            "Wing_tan_Lambda_LE": tanL + 0.25 * Dout,
            "Wing_tan_Lambda_qc_inn": tanL + 0.25 * Dout - 0.25 * Dinn}
    n = 0
    for nm, val in vals.items():
        v = getattr(f, nm, None)
        if v is None or val is None or val <= 0:
            continue
        try:
            v.set_value(float(val), skip_validation=True); n += 1
        except Exception:
            pass
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


def verify(tee: bool = True) -> dict:
    """Solve the LH2 D8.2 and return its headline weights, in lbf.

    ``presolve=False`` is not a preference. Adding the cryogenic tank to this
    model makes EDI's presolve overflow in ``restore_columns`` ->
    ``_solve_for`` -> ``_eval_terms`` (``math.exp`` of an accumulated log).
    The kerosene SPaircraft with the identical solver path presolves fine, and
    the trigger is not the near-zero ``f_wingfuel`` -- it reproduces at 1e-6,
    1e-4 and 1e-2 -- so it is one of the tank's fractional-power rows creating
    a column presolve eliminates and then cannot back-solve. Reported, not
    worked around silently.
    """
    import warnings

    import pyomo.environ as pyo
    from pyomo.environ import units as u

    from edi_compat import structure_detector
    from edi.solvers.ipopt.slcp_bridge import solve_sia
    from edi.solvers.ipopt.sia import SIAOptions
    from edi_compat import unit_corrector

    warnings.filterwarnings("ignore")
    fm = build(seed="reference")
    # unit_corrector RETURNS a corrected copy; the solved variables live on
    # it, not on `fm`. Reading `fm` instead gives a self-inconsistent answer
    # that still looks converged.
    cm = unit_corrector(fm)
    st = structure_detector(cm)
    res = solve_sia(st, options=SIAOptions(max_iterations=1500),
                    presolve=False)
    if not res.converged:
        raise RuntimeError(res.status)
    for v, val in zip(st['variables'], res.x):
        v.set_value(float(val))

    # The model legitimately mixes lbf (SPaircraft's own) and N (the tank and
    # wing box). Normalise everything to lbf on the way out.
    out = {}
    for v in cm.component_data_objects(pyo.Var):
        val = pyo.value(v)
        if str(u.get_units(v)) == "N":
            val *= 0.2248089431
        out[v.name] = val
    out["iterations"] = res.iterations
    if tee:
        for k in ("W_total", "W_dry", "W_f_total", "Wing_W_wing",
                  "Fuse_W_fuse", "Tank_W_tank", "Fuse_l_fuse", "Wing_S"):
            print(f"  {k:16s} {out[k]:12.1f}")
    return out