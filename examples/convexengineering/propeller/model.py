"""Actuator-disk propeller model (GP).

Source model
------------
``gpkitmodels/GP/aircraft/prop/propeller.py`` (``ActuatorProp`` +
``Propeller``) in https://github.com/convexengineering/gplibrary

The case rebuilt here is ``simpleprop_test`` from ``prop_test.py``: a
propeller at fixed flight speed, thrust, and rotation rate, with

    cost = 1/eta + W/(100 lbf) + Q/(100 N*m)

so the solve trades propulsive efficiency against prop weight and torque.
``rho``, ``V``, ``T`` and ``omega`` are substituted, leaving the radius as
the main degree of freedom.

The efficiency model is the classical actuator-disk result: the inviscid
efficiency ``etai`` satisfies

    etai*(z1 + sqrt(z2)/etaadd) <= 2,     z2 >= Tc + 1

where ``z1 = 2 - 1/etaadd`` is a *computed* constant (gpkit builds it from
``etaadd`` via a helper), and ``Tc`` is the thrust coefficient. Swirl and
viscous losses enter through the fixed factors ``etaadd`` and ``etav``.

Radians
-------
As in the motor model, every place an angular rate meets a length or a
torque needs an explicit ``/units.rad`` under Pyomo, which keeps the radian
as a real dimension where pint drops it. Here that is the tip speed
``omega*R``, the shaft power ``Q*omega``, and the power coefficient.
"""
from __future__ import annotations

from numpy import pi
from pyomo.environ import units

from edi import Formulation


def build() -> Formulation:
    f = Formulation()

    # ---- flight state (substituted in simpleprop_test) -------------------
    V   = f.Constant(name="V",   value=50.0,   units="m/s",    description="airspeed")
    rho = f.Constant(name="rho", value=1.225,  units="kg/m^3", description="air density")

    # ---- propeller constants ---------------------------------------------
    etaadd = f.Constant(name="etaadd", value=0.7,   units="-", description="swirl and nonuniformity losses")
    etav   = f.Constant(name="etav",   value=0.85,  units="-", description="viscous losses")
    # z1 is a computed constant in the source: z1 = 2 - 1/etaadd
    z1     = f.Constant(name="z1",     value=2.0 - 1.0 / 0.7, units="-", description="efficiency helper 1")
    T      = f.Constant(name="T",      value=100.0, units="lbf", description="thrust")
    omega  = f.Constant(name="omega",  value=1000.0, units="rpm", description="propeller rotation rate")
    omega_max = f.Constant(name="omega_max", value=10000.0, units="rpm", description="max rotation rate")
    M_tip  = f.Constant(name="M_tip",  value=0.5,   units="-",   description="tip Mach number")
    a      = f.Constant(name="a",      value=295.0, units="m/s", description="speed of sound at altitude")
    K      = f.Constant(name="K",      value=4e-4,  units="1/ft^2", description="prop weight scaling factor")

    # ---- free variables ---------------------------------------------------
    Tc   = f.Variable(name="Tc",   guess=0.05,  units="-", description="coefficient of thrust")
    etai = f.Variable(name="etai", guess=0.98,  units="-", description="inviscid efficiency")
    eta  = f.Variable(name="eta",  guess=0.83,  units="-", description="overall efficiency")
    z2   = f.Variable(name="z2",   guess=1.05,  units="-", description="efficiency helper 2")
    lam  = f.Variable(name="lam",  guess=0.36,  units="-", description="advance ratio")
    CT   = f.Variable(name="CT",   guess=0.0068, units="-", description="thrust coefficient")
    CP   = f.Variable(name="CP",   guess=0.0030, units="-", description="power coefficient")
    Q    = f.Variable(name="Q",    guess=250.0, units="N*m", description="torque")
    P_shaft = f.Variable(name="P_shaft", guess=26.0, units="kW", description="shaft power")
    R    = f.Variable(name="R",    guess=4.3,   units="ft",  description="prop radius")
    W    = f.Variable(name="W",    guess=0.75,  units="lbf", description="prop weight")
    T_m  = f.Variable(name="T_m",  guess=100.0, units="lbf", description="prop max static thrust")

    f.Objective(1.0 / eta + W / (100.0 * units.lbf) + Q / (100.0 * units.N * units.m))

    # tip speed: omega is an angular rate, so /rad makes it a velocity
    Vtip = omega * R / units.rad

    f.ConstraintList([
        eta <= etav * etai,
        Tc >= T / (0.5 * rho * V**2 * pi * R**2),
        z2 >= Tc + 1.0,
        etai * (z1 + z2**0.5 / etaadd) <= 2.0,
        lam >= V / Vtip,
        CT >= Tc * lam**2,
        CP <= Q * omega / units.rad / (0.5 * rho * Vtip**3 * pi * R**2),
        eta >= CT * lam / CP,
        omega <= omega_max,
        P_shaft == Q * omega / units.rad,
        (M_tip * a)**2 >= Vtip**2 + V**2,
        T_m >= T,
        # static prop sizing
        W >= K * T_m * R**2,
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
    rep = compare("Propeller (simpleprop_test)", sol, ref, rtol=1e-3, aliases=ALIASES)
    nv, worst, where = feasibility(f)
    if note:
        rep.notes.append(note)
    rep.notes.append(f"objective = {obj:.7g}   (gpkit cost = 3.7509322)")
    rep.notes.append(f"feasibility: {nv} violated, worst rel {worst:.2e}"
                     + (f" at {where}" if where else ""))
    print(rep)
