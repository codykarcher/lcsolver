"""SimPleAC — a simple aircraft sizing signomial program.

Source model
------------
``gpkitmodels/SP/SimPleAC/SimPleAC.py`` in
https://github.com/convexengineering/gplibrary

Paper
-----
The GP ancestor of this model is

    W. Hoburg and P. Abbeel, "Geometric Programming for Aircraft Design
    Optimization", AIAA Journal 52(11), 2014.

SimPleAC is the teaching descendant of that GP, extended into a signomial
program. It differs from the paper's model in three ways worth knowing:

* ``CDA0`` (fuselage drag area) is a *free variable* here, not the paper's
  fixed 0.035 m^2. It is what couples fuselage size to available fuel volume
  via ``V_f_fuse <= 10 m * CDA0``, so the model trades fuselage drag against
  fuel capacity.
* A fuel-volume model is added (``V_f_wing``, ``V_f_fuse``), and the
  ``V_f_avail <= V_f_wing + V_f_fuse`` constraint is what makes this an SP
  rather than a GP: a posynomial appears on the bounding side.
* The wing weight coefficient ``W_W_coeff1`` is 2e-5 1/m here; the source
  file's own comment records the original value as 12e-5 1/m.

Objective: minimize fuel weight ``W_f``.

Verification
------------
``reference.json`` is a snapshot of the gpkit model's ``localsolve``.
Regenerate with ``python reference.py`` (needs gpkit + a gplibrary checkout).
"""
from __future__ import annotations

import numpy as np
from pyomo.environ import units

from edi import Formulation


def build() -> Formulation:
    f = Formulation()

    # ---- environment constants ------------------------------------------
    g      = f.Constant(name="g",      value=9.81,     units="m/s^2",  description="gravitational acceleration")
    mu     = f.Constant(name="mu",     value=1.775e-5, units="kg/m/s", description="viscosity of air")
    rho    = f.Constant(name="rho",    value=1.23,     units="kg/m^3", description="density of air")
    rho_f  = f.Constant(name="rho_f",  value=817.0,    units="kg/m^3", description="density of fuel")

    # ---- non-dimensional constants --------------------------------------
    C_Lmax = f.Constant(name="C_Lmax", value=1.6,   units="-", description="max C_L with flaps down")
    e      = f.Constant(name="e",      value=0.92,  units="-", description="Oswald efficiency factor")
    k      = f.Constant(name="k",      value=1.17,  units="-", description="form factor")
    N_ult  = f.Constant(name="N_ult",  value=3.3,   units="-", description="ultimate load factor")
    S_wetratio = f.Constant(name="S_wetratio", value=2.075, units="-", description="wetted area ratio")
    tau    = f.Constant(name="tau",    value=0.12,  units="-", description="airfoil thickness to chord ratio")
    W_W_coeff1 = f.Constant(name="W_W_coeff1", value=2e-5, units="1/m", description="wing weight coefficient 1")
    W_W_coeff2 = f.Constant(name="W_W_coeff2", value=60.0, units="Pa",  description="wing weight coefficient 2")

    # ---- dimensional constants -------------------------------------------
    Range = f.Constant(name="Range", value=3000.0, units="km",   description="aircraft range")
    TSFC  = f.Constant(name="TSFC",  value=0.6,    units="1/hr", description="thrust specific fuel consumption")
    V_min = f.Constant(name="V_min", value=25.0,   units="m/s",  description="takeoff speed")
    W_0   = f.Constant(name="W_0",   value=6250.0, units="N",    description="aircraft weight excluding wing")

    # ---- free variables ---------------------------------------------------
    LoD   = f.Variable(name="LoD",   guess=20.0,     units="-",    description="lift-to-drag ratio")
    D     = f.Variable(name="D",     guess=500.0,    units="N",    description="total drag force")
    V     = f.Variable(name="V",     guess=50.0,     units="m/s",  description="cruising speed")
    W     = f.Variable(name="W",     guess=12000.0,  units="N",    description="total aircraft weight")
    Re    = f.Variable(name="Re",    guess=4.0e6,    units="-",    description="Reynolds number")
    CDA0  = f.Variable(name="CDA0",  guess=0.05,     units="m^2",  description="fuselage drag area")
    C_D   = f.Variable(name="C_D",   guess=0.015,    units="-",    description="drag coefficient")
    C_L   = f.Variable(name="C_L",   guess=0.3,      units="-",    description="lift coefficient of wing")
    C_f   = f.Variable(name="C_f",   guess=0.003,    units="-",    description="skin friction coefficient")
    W_f   = f.Variable(name="W_f",   guess=4000.0,   units="N",    description="fuel weight")
    V_f   = f.Variable(name="V_f",   guess=0.5,      units="m^3",  description="fuel volume")
    V_f_avail = f.Variable(name="V_f_avail", guess=0.5, units="m^3", description="fuel volume available")
    T_flight  = f.Variable(name="T_flight",  guess=16.0, units="hr", description="flight time")

    A     = f.Variable(name="A",     guess=12.0,   units="-",   description="aspect ratio")
    S     = f.Variable(name="S",     guess=20.0,   units="m^2", description="total wing area")
    W_w   = f.Variable(name="W_w",   guess=2500.0, units="N",   description="wing weight")
    W_w_strc = f.Variable(name="W_w_strc", guess=1200.0, units="N",   description="wing structural weight")
    W_w_surf = f.Variable(name="W_w_surf", guess=1300.0, units="N",   description="wing skin weight")
    V_f_wing = f.Variable(name="V_f_wing", guess=0.1,    units="m^3", description="fuel volume in the wing")
    V_f_fuse = f.Variable(name="V_f_fuse", guess=0.45,   units="m^3", description="fuel volume in the fuselage")

    f.Objective(W_f)

    pi = np.pi

    # ---- drag buildup -----------------------------------------------------
    C_D_fuse = CDA0 / S
    C_D_wpar = k * C_f * S_wetratio
    C_D_ind  = C_L**2 / (pi * A * e)

    f.ConstraintList([
        # weight and lift
        W >= W_0 + W_w + W_f,
        W_0 + W_w + 0.5 * W_f <= 0.5 * rho * S * C_L * V**2,
        W <= 0.5 * rho * S * C_Lmax * V_min**2,
        T_flight >= Range / V,
        LoD == C_L / C_D,

        # thrust and drag
        W_f >= TSFC * T_flight * D,
        D >= 0.5 * rho * S * C_D * V**2,
        C_D >= C_D_fuse + C_D_wpar + C_D_ind,
        # The 10 m coefficient carries units: it is a fuselage "length" scale
        # relating drag area to usable internal volume.
        V_f_fuse <= 10.0 * units.m * CDA0,
        Re <= (rho / mu) * V * (S / A)**0.5,
        C_f >= 0.074 / Re**0.2,

        # fuel volume  (V_f_avail <= V_f_wing + V_f_fuse is the SP constraint:
        # a posynomial on the bounding side)
        V_f == W_f / g / rho_f,
        V_f_wing**2 <= 0.0009 * S**3 / A * tau**2,
        V_f_avail <= V_f_wing + V_f_fuse,
        V_f_avail >= V_f,

        # wing weight
        W_w_surf >= W_W_coeff2 * S,
        W_w_strc**2 >= W_W_coeff1**2 / tau**2
                       * (N_ult**2 * A**3 * ((W_0 + V_f_fuse * g * rho_f) * W * S)),
        W_w >= W_w_surf + W_w_strc,
    ])
    return f


# Map reference (gpkit) names -> rebuilt names. gpkit uses LaTeX-ish keys.
ALIASES = {
    "(CDA0)": "CDA0",
    "L/D": "LoD",
    "T_{flight}": "T_flight",
    "V_{f_{avail}}": "V_f_avail",
}


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from harness import solve_edi, compare, load_reference, feasibility

    f = build()
    sol, obj, note = solve_edi(f)
    ref = load_reference(Path(__file__).with_name("reference.json"))
    rep = compare("SimPleAC", sol, ref, rtol=1e-3, aliases=ALIASES)
    nv, worst, where = feasibility(f)
    if note:
        rep.notes.append(note)
    rep.notes.append(f"objective W_f = {obj:.6g} N")
    rep.notes.append(f"feasibility: {nv} violated, worst rel {worst:.2e}"
                     + (f" at {where}" if where else ""))
    print(rep)
