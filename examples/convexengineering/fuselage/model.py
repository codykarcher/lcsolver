"""Elliptical fuselage sizing and drag (GP).

Source model
------------
``gpkitmodels/GP/aircraft/fuselage/elliptical_fuselage.py`` in
https://github.com/convexengineering/gplibrary

The case rebuilt here is ``test_ellp`` from ``test_fuselage.py``: a prolate
spheroid fuselage of fixed internal volume, minimizing ``W * Cd`` — the
product of structural weight and drag coefficient, so the solve trades a
stubby heavy fuselage against a long slender one.

Two constraints are worth reading closely.

**Wetted area** uses Knud Thomsen's approximation for the surface area of an
ellipsoid, which is what makes the relation GP-compatible:

    3 (S/pi)^p >= 2 (2 l R)^p + (2R)^(2p),      p = 1.6075

For a prolate spheroid with semi-axes ``l/2, R, R`` Thomsen gives
``S ~= 4 pi ((2 a^p R^p + R^2p)/3)^(1/p)`` with ``a = l/2``; multiplying
through by ``4^p`` puts it in exactly the form above.

**Skin thickness** is driven by minimum gauge (``t >= nply * tmin``), not by
stress — this is a small composite airframe, so the skin is ply-count
limited. That makes ``t`` constant at 2 x 0.3048 mm = 0.024 in, and the
weight follows the wetted area alone.
"""
from __future__ import annotations

import numpy as np

from lcsolver import Formulation

G = 9.81  # m/s^2, gpkitmodels.g
THOMSEN_P = 1.6075


def build() -> Formulation:
    f = Formulation()

    # ---- flight state (gplibrary FlightState) ----------------------------
    V   = f.Constant(name="V",   value=50.0,    units="m/s",       description="airspeed")
    rho = f.Constant(name="rho", value=1.255,   units="kg/m^3",    description="air density")
    mu  = f.Constant(name="mu",  value=1.5e-5,  units="N*s/m^2",   description="air viscosity")

    # ---- CFRP fabric material (gplibrary GP/materials) -------------------
    rhocfrp = f.Constant(name="rhocfrp", value=1.6,    units="g/cm^3", description="density of CFRP")
    tmin    = f.Constant(name="tmin",    value=0.3048, units="mm",     description="minimum gauge thickness")

    # ---- fuselage constants ----------------------------------------------
    mfac  = f.Constant(name="mfac",  value=2.0, units="-", description="fuselage weight margin factor")
    mfacd = f.Constant(name="mfacd", value=1.0, units="-", description="fuselage drag margin")
    nply  = f.Constant(name="nply",  value=2.0, units="-", description="number of plys")
    g     = f.Constant(name="g",     value=G,   units="m/s^2", description="gravitational acceleration")
    Vol   = f.Constant(name="Vol",   value=1.33, units="ft^3", description="fuselage volume")

    # ---- free variables ---------------------------------------------------
    R  = f.Variable(name="R",  guess=0.31,  units="ft",   description="fuselage radius")
    l  = f.Variable(name="l",  guess=6.5,   units="ft",   description="fuselage length")
    S  = f.Variable(name="S",  guess=10.0,  units="ft^2", description="wetted fuselage area")
    W  = f.Variable(name="W",  guess=4.0,   units="lbf",  description="fuselage weight")
    fr = f.Variable(name="f",  guess=10.0,  units="-",    description="fineness ratio")
    k  = f.Variable(name="k",  guess=1.08,  units="-",    description="fuselage form factor")
    t  = f.Variable(name="t",  guess=0.024, units="in",   description="fuselage skin thickness")
    Cf = f.Variable(name="Cf", guess=0.0038, units="-",   description="skin friction coefficient")
    Cd = f.Variable(name="Cd", guess=0.0041, units="-",   description="fuselage drag coefficient")
    Re = f.Variable(name="Re", guess=8.3e6,  units="-",   description="fuselage Reynolds number")

    # cost = W * Cd
    f.Objective(W * Cd)

    f.ConstraintList([
        # geometry
        fr == l / R / 2.0,
        k >= 1.0 + 60.0 / fr**3 + fr / 400.0,
        # Knud Thomsen ellipsoid wetted area (see module docstring)
        3.0 * (S / np.pi)**THOMSEN_P
            >= 2.0 * (l * R * 2.0)**THOMSEN_P + (2.0 * R)**(2.0 * THOMSEN_P),
        Vol <= 4.0 * np.pi / 3.0 * (l / 2.0) * R**2,
        # structure: minimum-gauge skin
        W / mfac >= S * rhocfrp * t * g,
        t >= nply * tmin,
        # aerodynamics
        Re == V * rho * l / mu,
        Cf >= 0.455 / Re**0.3,
        Cd / mfacd >= Cf * k,
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
    rep = compare("Fuselage (test_ellp)", sol, ref, rtol=1e-3, aliases=ALIASES)
    nv, worst, where = feasibility(f)
    if note:
        rep.notes.append(note)
    rep.notes.append(f"objective W*Cd = {obj:.7g}   (gpkit cost = 0.016479883)")
    rep.notes.append(f"feasibility: {nv} violated, worst rel {worst:.2e}"
                     + (f" at {where}" if where else ""))
    print(rep)
