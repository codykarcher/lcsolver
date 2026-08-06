"""A hydrogen-electric transport aircraft as a signomial program.

What this is
------------
An SP built from the TASOPT v3 Python port in ``examples/tasopt/``, in the
spirit of York, Hoburg and Drela's turbofan SP and of ``SPaircraft``: derive
the constraints from a verified simulation rather than from a textbook, and
say which ones needed reformulating and why.

Objective: minimise maximum take-off weight.

The three couplings that make it a hydrogen aircraft
----------------------------------------------------
Take these away and it is a conventional tube-and-wing with odd numbers.

1. **The tank displaces fuselage.** LH2 will not go in a wing box, so the
   tank is a cylinder in the fuselage: its length stretches the fuselage,
   its wetted area buys drag. Kerosene costs none of this -- and because the
   wing carries no fuel, the wing box is sized with **no bending relief**,
   which is a constraint removed rather than added (see ``wing_h2.py``).
2. **The stack dominates the powertrain mass.** At the port's own
   ``f_outer = 4`` the structure and bipolar plates are four fifths of the
   stack, so the active layers the 1-D model computes so carefully are a
   fifth of what flies.
3. **The waste heat needs a radiator, and the radiator makes drag.** A ~60%
   efficient stack rejects two thirds of a watt per watt delivered at a
   temperature only ~180 K above ambient. Modelled as a drag power
   proportional to heat rejected -- the trade that couples current density
   (efficiency) to airframe drag.

Structure, audited
------------------
Every constraint here is GP-compatible except a handful, each of York's
"sum on the greater side" shape:

* the stack heat balance, ``n A j V_tn <= Q + P_e``  (one per segment);
* the tank volume closure, ``V_fuel <= l A + 2 V_head``;
* the momentum rows, ``F + mdot u_0 <= mdot u_j`` and the actuator disc
  ``mdot <= (rho/2) A_disc (u_0 + u_j)`` (see ``powertrain.py`` for why the
  disc row exists -- the solver diagnosed its absence);
* the weight decrement, ``W[i] <= W[i+1] + burn_i`` (see the comment at the
  constraint for why this direction and not the other).

An earlier draft also carried a third: ``R_req <= sum R_seg``, total range as
a free split across segments. That row cost more than its SP class: with
identical cruise segments the split is *arbitrary*, so the optimum is a flat
valley and SIA's iterate slid along the null direction at a constant creep --
3.5e-4 log-units per iteration, thousands of iterations, stationarity never
reached. Pinning each segment to ``R_req/N`` removes the null space and the
SP row in one stroke. A non-uniqueness in the model is a convergence bug in
the solver's lap.

The momentum-thrust and weight-decrement rows *look* signomial but are
posynomial-below-monomial, which divides through -- they stay in GP. The
solver's own structure detection confirms the split; ``verify()`` prints it.

Deliberately simplified
-----------------------
Fuselage and tail are area scalings; the mission is cruise-only with a 10%
fuel reserve; no gear, no systems weight, no field length, no climb. The
wing, tank, stack and powertrain are the derived parts.
"""
from __future__ import annotations

from lcsolver import Formulation

from .cryo_tank import add_cryo_tank
from .fuel_cell import add_fuel_cell
from .powertrain import add_powertrain
from .wing_h2 import add_wing_h2

N_CRUISE = 4
PI = 3.141592653589793


def build(N: int = N_CRUISE, *, W_pay: float = 1.8e5, R_req: float = 3.0e6,
          R_fuse: float = 1.9, l_cabin: float = 26.0) -> Formulation:
    """Build the aircraft. Returns an LCsolver ``Formulation``.

    The mission is parameterised so the same model can fly other aircraft
    classes: ``W_pay`` payload (N), ``R_req`` range (m), ``R_fuse`` fuselage
    radius (m), ``l_cabin`` cabin length (m). Defaults are the 180-pax,
    3000 km case the README documents.
    """
    f = Formulation()
    Vb = lambda n, g, u, d, bd: f.Variable(name=n, guess=g, units=u,
                                           description=d, bounds=bd)
    Vnb = lambda n, g, u, d, bd: f.Variable(name=n, guess=g, units=u,
                                            description=d, size=N, bounds=bd)
    C = lambda n, v, u, d: f.Constant(name=n, value=v, units=u, description=d)

    # Secondary structure per unit AREA, not as a multiple of box weight.
    #
    # The old default (f_nonstruct=1.2, k_area=0) was light twice over.
    # SPaircraft itemises secondary structure as flaps 0.2, slats 0.001,
    # ailerons 0.04, LE/TE 0.1, ribs 0.15, spoilers 0.02 and water 0.03 --
    # summing to 0.541 of box weight, so 1.2 was a guess and 22% short of it.
    #
    # But scaling it off the box is also the wrong shape, and add_wing_h2's
    # own docstring says why: the Hoburg box is weak in area (S^0.5), so with
    # span free the optimiser grows chord for free and aspect ratio collapses.
    # It did -- at f_nonstruct=1.541 with k_area=0 the wing came out AR 7.02,
    # which is not a transport wing. Skins, ribs and flaps scale with wetted
    # area, so charging them that way both fixes the magnitude and restores
    # the AR trade: this lands at AR 10.53 against SPaircraft's 10.39, and
    # 127.6 lbf/m2 of wing against its 124.8.
    #
    # 195 N/m2 is SPaircraft's own implied rate: 0.541 x its 80.9 lbf/m2 box.
    # Set at the call site, not in add_wing_h2, because model_lh2tf.py's
    # TASOPT replication is verified at the old default and must not move.
    wing, cons = add_wing_h2(f, f_nonstruct=1.0, k_area=195.0)
    tank, c = add_cryo_tank(f); cons += c
    fc, c = add_fuel_cell(f, N); cons += c
    pt, c = add_powertrain(f, N); cons += c

    # ---- aircraft-level -----------------------------------------------------
    # Bounds, not tuning: several relations are reciprocal in a design
    # variable, and an unbounded log-space iterate overflows before it can be
    # infeasible. All are generous engineering limits.
    W_MTO = Vb("W_MTO", 4.0e5, "N", "maximum take-off weight", (1e5, 3e6))
    W_dry = Vb("W_dry", 3.85e5, "N", "zero-fuel weight", (1e5, 3e6))
    W_fuse = Vb("W_fuse", 8.9e4, "N", "fuselage structural weight", (1e4, 1e6))
    W_tail = Vb("W_tail", 5.7e3, "N", "lumped empennage weight", (5e2, 2e5))
    W_lg = Vb("W_lg", 2.2e4, "N", "landing gear weight", (1e3, 5e5))
    W_hpesys = Vb("W_hpesys", 4.0e3, "N",
                  "hydraulic, pneumatic and electrical systems", (1e2, 1e5))
    W_padd = Vb("W_padd", 6.9e4, "N",
                "seats, galleys, furnishings and APU", (1e3, 1e6))
    l_fuse = Vb("l_fuse", 28.8, "m", "fuselage length", (10.0, 80.0))
    S_wet = Vb("S_wet", 480.0, "m^2", "total wetted area", (100.0, 3000.0))
    D_cool = Vb("D_cool", 2.3e3, "N", "cooling drag", (10.0, 1e5))

    W = Vnb("W", 4.0e5, "N", "aircraft weight at segment start", (1e5, 3e6))
    D = Vnb("D", 2.7e4, "N", "drag", (2e3, 3e5))
    C_L = Vnb("C_L", 0.58, "-", "lift coefficient", (0.10, 0.70))
    C_D = Vnb("C_D", 0.035, "-", "drag coefficient", (0.008, 0.30))
    t_seg = Vnb("t_seg", 3.24e3, "s", "segment duration", (100.0, 5e4))

    # ---- constants ----------------------------------------------------------
    g = C("g", 9.81, "m/s^2", "gravitational acceleration")
    rho = C("rho", 0.38, "kg/m^3", "air density at 11 km")
    V_inf = C("V_inf", 232.0, "m/s", "cruise speed, M 0.78 at 11 km")
    e_osw = C("e", 0.85, "-", "Oswald efficiency")
    C_f = C("C_f", 0.0032, "-", "equivalent skin friction coefficient")
    W_pay = C("W_pay", W_pay, "N", "payload weight")
    R_req = C("R_req", R_req, "m", "required range")
    R_fuse = C("R_fuse", R_fuse, "m", "fuselage radius")
    l_cabin = C("l_cabin", l_cabin, "m", "cabin length for the payload")
    k_fuse = C("k_fuse", 260.0, "N/m^2", "fuselage weight per wetted area")
    # TASOPT 2.16's own weight fractions. Its whole landing gear model is
    # flgnose + flgmain times MTOW -- the gear has no length, so nothing
    # connects it to the aircraft it holds up. v3 adds Raymer's correlations
    # instead, which are monomials in MTOW and would be GP-native, but they
    # need a gear *length* set by a max() of engine clearance against
    # tailstrike, both of which are differences. That is more machinery than
    # this 109-variable model earns, so 2.16's fractions are used.
    f_lg = C("f_lg", 0.011 + 0.044, "-",
             "landing gear, fraction of MTOW (flgnose + flgmain)")
    f_hpesys = C("f_hpesys", 0.01, "-", "systems, fraction of MTOW")
    # fpadd covers seats, galleys, lavatories and furnishings; fapu the APU.
    # k_fuse above is a structure-only calibration -- the solved fuselage sits
    # at exactly 260 N/m2, against TASOPT's ~450 N/m2 all-in -- so none of
    # this was previously counted anywhere.
    f_padd = C("f_padd", 0.35 + 0.035, "-",
               "furnishings and APU, fraction of payload (fpadd + fapu)")
    k_rad = C("k_rad", 0.12, "-", "cooling drag power over heat rejected")
    N_ult = C("N_ult", 3.0, "-", "ultimate load factor")
    f_res = C("f_res", 1.10, "-", "fuel reserve factor")
    eta_chain = C("eta_chain", 0.995 * 0.96 * 0.99, "-",
                  "inverter x motor x cable efficiency")

    cons += [
        # -- the wing carries the whole aircraft, dry (see wing_h2.py) ---------
        wing["L_max"] >= N_ult * W_MTO,

        # -- geometry -----------------------------------------------------------
        l_fuse >= l_cabin + tank["l_tank"],
        S_wet >= 2.0 * wing["S"] + 2.0 * PI * R_fuse * l_fuse
                 + pt["S_nace"],
        R_fuse >= tank["R_o"] + tank["t_insul"],

        # -- weights ------------------------------------------------------------
        W_fuse >= k_fuse * 2.0 * PI * R_fuse * l_fuse,
        W_tail >= 0.25 * wing["W_wing"],
        # Gear and systems scale with MTOW, so these close a loop: heavier
        # aircraft need heavier gear, which makes the aircraft heavier. The
        # fractions sum to 0.065, so the amplification is 1/(1-0.065) = 1.07
        # and the fixed point is well inside the GP.
        W_lg >= f_lg * W_MTO,
        W_hpesys >= f_hpesys * W_MTO,
        W_padd >= f_padd * W_pay,
        W_dry >= (wing["W_wing"] + W_fuse + W_tail + W_pay
                  + tank["W_tank"] + fc["W_stack"] + pt["W_pt"]
                  + W_lg + W_hpesys + W_padd),
        W_MTO >= W_dry + tank["W_fuel"],
        # The aircraft at the start of cruise weighs the full MTOW. The other
        # direction lets the lift equation see an arbitrarily light aircraft,
        # and the optimiser flies a 16 kPa wing loading (bug #3 in README).
        W[0] >= W_MTO,
    ]

    for i in range(N):
        cons += [
            # -- aerodynamics -----------------------------------------------------
            W[i] <= 0.5 * rho * V_inf ** 2 * wing["S"] * C_L[i],
            C_D[i] >= (C_f * S_wet / wing["S"]
                       + C_L[i] ** 2 / (PI * wing["AR"] * e_osw)),
            D[i] >= 0.5 * rho * V_inf ** 2 * wing["S"] * C_D[i] + D_cool,

            # -- propulsion balance ----------------------------------------------
            pt["F_net"][i] >= D[i],
            fc["P_e"][i] * eta_chain >= pt["P_shaft"][i],

            # -- cooling drag, sized by every segment's heat load ----------------
            D_cool >= k_rad * fc["Q_fc"][i] / V_inf,

            # -- segment kinematics ----------------------------------------------
            # Each segment covers exactly its share of the range. Monomial
            # above a constant, so GP -- and no arbitrary split to drift on.
            V_inf * t_seg[i] >= R_req / N,
        ]

    # Weight decrement -- and the DIRECTION is the whole constraint.
    # Written >= (as a first draft had it), the optimiser drops every later
    # segment's weight to its lower bound and flies a 10 t aircraft for
    # three quarters of the cruise: nothing stops weight falling FASTER than
    # fuel is burned. The physical statement is the opposite bound --
    #     W[i+1] >= W[i] - burn_i,
    # weight persists except for what was burned -- which in SP form is a
    # monomial below a posynomial: signomial, York's shape again. The wrong
    # direction was caught by a Breguet cross-check missing by 15%, not by
    # the solver, which satisfied it happily.
    for i in range(N - 1):
        cons += [W[i] <= W[i + 1]
                 + g * (fc["mdot_H2"][i] + tank["m_boil"]) * t_seg[i]]

    # The tank carries every segment's burn and boil-off, plus reserve.
    cons += [tank["W_fuel"] >= f_res * sum(
        g * (fc["mdot_H2"][i] + tank["m_boil"]) * t_seg[i] for i in range(N))]

    f.Objective(W_MTO)
    f.ConstraintList(cons)
    return f


def verify(max_iterations: int = 400, tee: bool = False,
           **mission) -> dict:
    """Build, solve under SIA defaults, and run the self-checks.

    Returns a dict of headline quantities. Raises if the solve does not
    certify KKT convergence, if any constraint is violated at the solution,
    or if the model disagrees with a continuous Breguet integration by more
    than 5% -- the check that caught the weight-decrement direction bug.
    """
    import math
    import warnings

    import pyomo.environ as pyo

    from lcsolver.presolve.structureDetector import structure_detector
    from lcsolver.solvers.ipopt.slcp_bridge import solve_sia, build_problem, \
        _apply_presolve
    from lcsolver.solvers.ipopt.sia import SIAOptions, classify
    from lcsolver.presolve.unitCorrector import unit_corrector

    fm = build(**mission)
    unit_corrector(fm)
    st = structure_detector(fm)

    # --- structure audit: how many rows are GP-exact vs conservative -------
    x0 = [float(pyo.value(v)) for v in st['variables']]
    st2, x02, _log, _n = _apply_presolve(st, x0)
    problem = build_problem(st2, sp_form=True)
    n_exact, n_cons, n_lin = classify(problem)

    res = solve_sia(st, options=SIAOptions(max_iterations=max_iterations))
    if not res.converged:
        raise RuntimeError(f"SIA did not certify convergence: {res.status}")

    # Load the solution back onto the formulation for value reads.
    for v, val in zip(st['variables'], res.x if len(res.x) >= len(
            st['variables']) else list(res.x) + [1.0] * (
            len(st['variables']) - len(res.x))):
        v.set_value(float(val))

    vals = {v.name: pyo.value(v) for v in fm.component_data_objects(pyo.Var)}
    g = 9.81

    # --- Breguet cross-check, INCLUDING cooling drag ------------------------
    # The first version of this check omitted D_cool and read 16% low, which
    # was almost exactly the size of a real bug it should have caught. The
    # effective L/D is W/D_total, nothing else.
    W0, D0 = vals["W[0]"], vals["D[0]"]
    eta_fc = vals["FC_P_e[0]"] / (vals["FC_mdot_H2[0]"] * 1.2e8)
    eta_chain = 0.995 * 0.96 * 0.99
    eta_prop = 2 * 232.0 / (232.0 + vals["PT_u_j[0]"])
    LoD_eff = W0 / D0
    R_m = mission.get("R_req", 3.0e6)
    z = R_m / (eta_fc * eta_chain * eta_prop * 1.2e8 / g * LoD_eff)
    breguet_kg = (1 - math.exp(-z)) * W0 / g
    burn_kg = sum((vals[f"FC_mdot_H2[{i}]"] + vals["Tank_m_boil"])
                  * vals[f"t_seg[{i}]"] for i in range(N_CRUISE))
    # Compare PROPULSION burn only: boil-off (~50 kg here, 3.6% of block
    # fuel) is a real hydrogen term the Breguet integral does not model --
    # the tank's thermal design leaking into the mission closure. Comparing
    # the total first tripped this check at 5.5%, which is exactly the kind
    # of near-threshold residual worth decomposing rather than loosening
    # the tolerance over.
    burn_prop_kg = sum(vals[f"FC_mdot_H2[{i}]"] * vals[f"t_seg[{i}]"]
                       for i in range(N_CRUISE))
    gap = abs(burn_prop_kg - breguet_kg) / breguet_kg
    if gap > 0.05:
        raise RuntimeError(
            f"model propulsion burn {burn_prop_kg:.0f} kg vs continuous "
            f"Breguet {breguet_kg:.0f} kg: {100 * gap:.1f}% apart")

    out = {
        "MTOW_t": vals["W_MTO"] / g / 1e3,
        "fuel_kg": vals["Tank_W_fuel"] / g,
        "burn_kg": burn_kg,
        "boiloff_kg": burn_kg - burn_prop_kg,
        "breguet_kg": breguet_kg,
        "LoD_eff": LoD_eff,
        "wing_loading_kgm2": vals["W_MTO"] / vals["Wing_S"] / g,
        "tank_gravimetric": vals["Tank_W_fuel"]
                            / (vals["Tank_W_fuel"] + vals["Tank_W_tank"]),
        "stack_kW_per_kg": vals["FC_P_e[0]"] / (vals["FC_W_stack"] / g) / 1e3,
        "D_fan_m": vals["PT_D_fan"],
        "eta_prop": eta_prop,
        "j_Am2": vals["FC_j[0]"],
        "iterations": res.iterations,
        "rows_exact": n_exact,
        "rows_conservative": n_cons,
    }
    if tee:
        for k, v in out.items():
            print(f"  {k:20s} {v:12.4g}")
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(verify(tee=True), indent=2))
