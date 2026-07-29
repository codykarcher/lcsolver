"""Hydrogen-electric aircraft with SPaircraft's real airframe.

Why a second model file
-----------------------
``model.py`` carries a deliberately crude airframe -- three hand-picked area
scalings -- so that the hydrogen physics has something to attach to. That was
honest scaffolding, not a model: the wing weight was a constant times area,
which the optimiser will happily exploit.

This version replaces it with the structural models from
``../convexengineering/spaircraft/``: a real flight state, a wing with a spar
box, and (as they are brought in) a fuselage, tails and gear. The hydrogen
modules are unchanged -- they are the part that was already derived.

Integration is incremental on purpose. SPaircraft's own docstring records that
its model "never solves bare" and needs a reference seed to converge, so
swapping in 400 constraints at once and hoping is not a plan. Each module is
added and the model re-solved, so that when convergence breaks it is obvious
which piece broke it.

Currently integrated
--------------------
* ``add_flight_state`` -- altitude, Mach, density, viscosity as variables
  rather than the fixed cruise constants ``model.py`` uses.
* ``add_wing`` -- planform plus the Hoburg spar-box structure, so wing weight
  now comes from bending loads instead of a constant.

Still crude here
----------------
Fuselage, tails and landing gear remain area scalings. The mission is still
cruise-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

from edi import Formulation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "convexengineering"))
from spaircraft.flight_state import add_flight_state   # noqa: E402
from spaircraft.wing import add_wing                   # noqa: E402

from .cryo_tank import add_cryo_tank                   # noqa: E402
from .fuel_cell import add_fuel_cell                   # noqa: E402
from .powertrain import add_powertrain                 # noqa: E402

N_CRUISE = 4
SWEEP_W = 13.237          # deg, quarter-chord, from SPaircraft's optimal D8


def build(N: int = N_CRUISE) -> Formulation:
    f = Formulation()
    Vb = lambda n, g, u, d, bd: f.Variable(name=n, guess=g, units=u,
                                           description=d, bounds=bd)
    Vnb = lambda n, g, u, d, bd: f.Variable(name=n, guess=g, units=u,
                                            description=d, size=N, bounds=bd)
    C = lambda n, v, u, d: f.Constant(name=n, value=v, units=u, description=d)

    st, cons = add_flight_state(f, N)
    wing, c = add_wing(f, N, st, sweep_deg=SWEEP_W); cons += c
    tank, c = add_cryo_tank(f); cons += c
    fc, c = add_fuel_cell(f, N); cons += c
    pt, c = add_powertrain(f, N); cons += c

    # ---- aircraft-level ----------------------------------------------------
    W_MTO = Vb("W_MTO", 7.0e5, "N", "maximum take-off weight", (1e5, 1e7))
    W_dry = Vb("W_dry", 6.79e5, "N", "zero-fuel weight", (1e5, 1e7))
    W_fuse = Vb("W_fuse", 1.05e5, "N", "fuselage weight", (1e3, 1e6))
    W_tail = Vb("W_tail", 1.6e4, "N", "lumped empennage weight", (1e2, 5e5))
    l_fuse = Vb("l_fuse", 33.0, "m", "fuselage length", (10.0, 90.0))
    S_wet = Vb("S_wet", 644.0, "m^2", "total wetted area", (50.0, 4000.0))
    D_cool = Vb("D_cool", 3.9e3, "N", "cooling drag", (1.0, 2e5))

    W = Vnb("W", 7.0e5, "N", "weight at segment start", (1e5, 1e7))
    D = Vnb("D", 4.4e4, "N", "drag", (1e3, 1e6))
    C_D = Vnb("C_D", 0.037, "-", "drag coefficient", (0.005, 0.5))
    t_seg = Vnb("t_seg", 3.25e3, "s", "segment duration", (10.0, 1e5))
    R_seg = Vnb("R_seg", 7.5e5, "m", "segment range", (1e3, 1e7))

    g = C("g", 9.81, "m/s^2", "gravitational acceleration")
    C_f = C("C_f", 0.0032, "-", "equivalent skin friction coefficient")
    W_pay = C("W_pay", 1.8e5, "N", "payload, 180 passengers")
    R_req = C("R_req", 3.0e6, "m", "required range")
    R_fuse = C("R_fuse", 1.9, "m", "fuselage radius")
    l_cabin = C("l_cabin", 26.0, "m", "cabin length")
    k_fuse = C("k_fuse", 260.0, "N/m^2", "fuselage weight per wetted area")
    k_rad = C("k_rad", 0.12, "-", "cooling drag power over heat rejected")
    N_lift = C("N_lift", 3.0, "-", "wing ultimate load factor")

    cons += [
        # -- the wing carries the aircraft --------------------------------------
        # SPaircraft's wing sizes its spar box from L_max, so the load case has
        # to come from here rather than from a constant.
        wing.L_max >= N_lift * W_MTO,
        # No fuel in the wing: that is the whole point of a hydrogen aircraft,
        # and it removes the bending relief a kerosene wing enjoys.
        wing.W_fuel_wing <= 1e-6 * W_MTO,

        # -- geometry -------------------------------------------------------------
        l_fuse >= l_cabin + tank["l_tank"],
        S_wet >= 2.0 * wing.S + 2.0 * 3.141592653589793 * R_fuse * l_fuse,
        R_fuse >= tank["R_o"] + tank["t_insul"],

        # -- weights ---------------------------------------------------------------
        W_fuse >= k_fuse * 2.0 * 3.141592653589793 * R_fuse * l_fuse,
        W_tail >= 0.25 * wing.W_wing,
        W_dry >= (wing.W_wing + W_fuse + W_tail + W_pay
                  + tank["W_tank"] + fc["W_stack"] + pt["W_pt"]),
        W_MTO >= W_dry + tank["W_fuel"],
        W[0] >= W_MTO,
    ]

    for i in range(N):
        cons += [
            # Lift, now against the flight state's own density and speed and
            # the wing's own lift coefficient.
            W[i] <= 0.5 * st.rho[i] * st.V[i] ** 2 * wing.S * wing.C_L[i],
            # Parasite plus induced, the latter from the wing's span
            # efficiency rather than a constant.
            C_D[i] >= (C_f * S_wet / wing.S
                       + wing.C_L[i] ** 2 / (3.141592653589793 * wing.AR * wing.e)),
            D[i] >= 0.5 * st.rho[i] * st.V[i] ** 2 * wing.S * C_D[i]
                    + D_cool,
            pt["F_net"][i] >= D[i],
            fc["P_e"][i] * 0.995 * 0.96 * 0.99 >= pt["P_shaft"][i],
            D_cool >= k_rad * fc["Q_fc"][i] / st.V[i],
            R_seg[i] <= st.V[i] * t_seg[i],
        ]

    cons += [sum(R_seg[i] for i in range(N)) >= R_req]
    for i in range(N - 1):
        cons += [W[i] >= W[i + 1]
                 + g * (fc["mdot_H2"][i] + tank["m_boil"]) * t_seg[i]]
    cons += [tank["W_fuel"] >= sum(
        g * (fc["mdot_H2"][i] + tank["m_boil"]) * t_seg[i] for i in range(N))]

    f.Objective(W_MTO)
    f.ConstraintList(cons)
    return f
