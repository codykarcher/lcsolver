"""Per-segment atmosphere and flight state.

Source model
------------
``stand_alone_simple_profile.py`` in
https://github.com/convexengineering/SPaircraft — classes ``FlightState``,
``Altitude`` and ``Atmosphere``.

Beware which ``stand_alone_simple_profile`` you get
---------------------------------------------------
The turbofan repo ships a *different* file of the same name, and SPaircraft's
``aircraft.py`` imports it unqualified. Picking up turbofan's version leaves
gravity a free variable and moves D8.2 fuel burn 2.2% with no error message.
See DISCREPANCIES.md §12.

The atmosphere
--------------
Standard troposphere, made GP-compatible in two places:

* ``T_sl == T_atm + L_atm*h`` is a ``SignomialEquality`` — the lapse relation
  is a sum, so it cannot be a monomial equality.
* Sutherland's law for viscosity, ``mu = C_1 T^1.5 / (T + T_s)``, is replaced
  by ``mu == C_1 T^1.5 / (6.64 K^0.28 * T_s^0.72)``, a monomial fit. The
  ``6.64 K^0.28`` is a dimensional fudge that makes the units resolve; it is
  not a physical constant, and the exponent 0.72 on ``T_s`` is what absorbs
  the temperature dependence the true denominator would have supplied.
"""
from __future__ import annotations


def add_flight_state(f, N, *, prefix="FS_"):
    """Add an N-segment flight state. Returns ``(vars, constraints)``."""
    fs = f.group("fs", prefix=prefix)
    C = fs.Constant
    Vn = lambda n, g, u, d: fs.Variable(n, g, u, d, size=N)

    h = Vn("h", 9000.0, "m", "segment altitude")
    hft = Vn("hft", 30000.0, "ft", "segment altitude in feet")
    V = Vn("V", 450.0, "kts", "aircraft flight speed")
    a = Vn("a", 300.0, "m/s", "speed of sound")
    M = Vn("M", 0.75, "-", "Mach number")
    rho = Vn("rho", 0.4, "kg/m^3", "density of air")
    p_atm = Vn("P_atm", 30000.0, "Pa", "air pressure")
    T_atm = Vn("T_atm", 230.0, "K", "air temperature")
    mu = Vn("mu", 1.5e-5, "kg/(m*s)", "dynamic viscosity")

    R = C("R", 287.0, "J/kg/K", "air specific heat")
    gamma = C("gamma", 1.4, "-", "air specific heat ratio")
    p_sl = C("p_sl", 101325.0, "Pa", "pressure at sea level")
    T_sl = C("T_sl", 288.15, "K", "temperature at sea level")
    L_atm = C("L_atm", 0.0065, "K/m", "temperature lapse rate")
    M_atm = C("M_atm", 0.0289644, "kg/mol", "molar mass of dry air")
    R_atm = C("R_atm", 8.31447, "J/mol/K", "universal gas constant")
    T_s = C("T_s", 110.4, "K", "Sutherland temperature")
    C_1 = C("C_1", 1.458e-6, "kg/(m*s*K^0.5)", "Sutherland coefficient")
    sfudge = C("sutherland_fudge", 6.64, "K^0.28",
               "dimensional factor in the monomial viscosity fit")

    out = dict(h=h, hft=hft, V=V, a=a, M=M, rho=rho, P_atm=p_atm,
               T_atm=T_atm, mu=mu, R=R, gamma=gamma)

    cons = []
    for i in range(N):
        cons += [
            h[i] == hft[i],
            a[i] == (gamma * R * T_atm[i]) ** 0.5,
            V[i] == M[i] * a[i],
            # Pressure-altitude relation. The 1/5.257 exponent is
            # (g M_atm)/(R_atm L_atm) evaluated once.
            (p_atm[i] / p_sl) ** (1 / 5.257) == T_atm[i] / T_sl,
            rho[i] == p_atm[i] / (R_atm / M_atm * T_atm[i]),
            T_sl == T_atm[i] + L_atm * h[i],                         # [SP] SigEq
            mu[i] == C_1 * T_atm[i] ** 1.5 / (sfudge * T_s ** 0.72),
        ]

    return out, cons
