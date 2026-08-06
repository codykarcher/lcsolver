"""Electric motor performance model (GP).

Source model
------------
``gpkitmodels/GP/aircraft/motor/motor.py`` in
https://github.com/convexengineering/gplibrary

The case rebuilt here is ``Motor_P_Test`` from ``motor_test.py``: a single
motor at a fixed operating torque, sized against its maximum torque, with

    cost = 1/etam + W/(100 lbf)

so the solve trades motor efficiency against motor weight. ``Qmax`` is
substituted at 100 N*m and the operating torque ``Q`` at 10 N*m, which pins
the weight through ``W >= Qstar*Qmax*g`` and leaves the voltage constant
``Kv`` as the real degree of freedom.

Angular velocity and the radian
-------------------------------
gpkit uses pint, which treats the radian as dimensionless, so ``Q*omega``
is a power directly. Pyomo (and therefore LCsolver) treats the radian as a real
dimension, so ``N*m*rpm`` will not convert to kW and the same expression
raises ``InconsistentUnitsError``.

Both constraints where a torque meets an angular rate therefore carry an
explicit ``/units.rad``:

    Pshaft == Q*omega/rad          (torque x angular velocity -> power)
    i      >= Q*Kv/rad + i0        (Kv is rpm/V, so Q*Kv is a power/voltage)

This is bookkeeping, not a model change: dividing by one radian is dividing
by one. It is called out because it is the single most likely place for a
gpkit model to fail to port cleanly to LCsolver.
"""
from __future__ import annotations

from pyomo.environ import units

from lcsolver import Formulation

G = 9.81  # m/s^2, gpkitmodels.g


def build() -> Formulation:
    f = Formulation()

    # ---- motor static constants (gplibrary Motor) ------------------------
    Qstar  = f.Constant(name="Qstar",  value=0.8,   units="kg/(N*m)", description="motor specific torque")
    V_max  = f.Constant(name="V_max",  value=300.0, units="V",        description="motor max voltage")
    Kv_min = f.Constant(name="Kv_min", value=1.0,   units="rpm/V",    description="min voltage constant")
    Kv_max = f.Constant(name="Kv_max", value=1000.0, units="rpm/V",   description="max voltage constant")
    i0     = f.Constant(name="i0",     value=4.5,   units="amp",      description="zero-load current")
    R      = f.Constant(name="R",      value=0.033, units="ohm",      description="internal resistance")
    g      = f.Constant(name="g",      value=G,     units="m/s^2",    description="gravitational acceleration")

    # ---- substituted in Motor_P_Test -------------------------------------
    Qmax = f.Constant(name="Qmax", value=100.0, units="N*m", description="motor max torque")
    Q    = f.Constant(name="Q",    value=10.0,  units="N*m", description="operating torque")

    # ---- free variables ---------------------------------------------------
    Pshaft = f.Variable(name="Pshaft", guess=50.0,   units="kW",  description="motor output shaft power")
    Pelec  = f.Variable(name="Pelec",  guess=55.0,   units="kW",  description="motor input electrical power")
    etam   = f.Variable(name="etam",   guess=0.9,    units="-",   description="motor efficiency")
    omega  = f.Variable(name="omega",  guess=50000.0, units="rpm", description="rotation rate")
    i      = f.Variable(name="i",      guess=200.0,  units="amp", description="current")
    v      = f.Variable(name="v",      guess=300.0,  units="V",   description="voltage")
    Kv     = f.Variable(name="Kv",     guess=200.0,  units="rpm/V", description="motor voltage constant")
    W      = f.Variable(name="W",      guess=176.0,  units="lbf", description="motor weight")

    # cost = 1/etam + W/(100 lbf)
    f.Objective(1.0 / etam + W / (100.0 * units.lbf))

    f.ConstraintList([
        # /units.rad: see module docstring — Pyomo keeps the radian as a real
        # dimension where pint drops it.
        Pshaft == Q * omega / units.rad,
        Pelec == v * i,
        etam == Pshaft / Pelec,
        Qmax >= Q,
        v <= V_max,
        i >= Q * Kv / units.rad + i0,
        v >= omega / Kv + i * R,
        # static motor sizing
        W >= Qstar * Qmax * g,
        Kv >= Kv_min,
        Kv <= Kv_max,
    ])
    return f


ALIASES: dict = {}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from harness import solve_edi, compare, load_reference, feasibility

    f = build()
    sol, obj, note = solve_edi(f)
    ref = load_reference(Path(__file__).with_name("reference.json"))
    rep = compare("Motor (Motor_P_Test)", sol, ref, rtol=1e-3, aliases=ALIASES)
    nv, worst, where = feasibility(f)
    if note:
        rep.notes.append(note)
    rep.notes.append(f"objective = {obj:.6g}   (gpkit cost = 2.8103277)")
    rep.notes.append(f"feasibility: {nv} violated, worst rel {worst:.2e}"
                     + (f" at {where}" if where else ""))
    print(rep)
