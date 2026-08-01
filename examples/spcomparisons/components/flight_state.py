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
    """Add an N-segment flight state. Returns ``(group, constraints)``.

    The group is the namespace itself, so a caller reads ``st.T_atm`` rather
    than looking a name up in a dictionary that had to be kept in step with
    the declarations by hand.
    """
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
    ft_ref = C("ft_ref", 1.0, "ft", "unit scale for the dimensionless altitude")
    pa_ref = C("pa_ref", 1.0, "Pa", "unit scale for the dimensionless pressure")
    T_trop = C("T_trop", 216.65, "K", "tropopause / lower-stratosphere temperature")
    T_s = C("T_s", 110.4, "K", "Sutherland temperature")
    C_1 = C("C_1", 1.458e-6, "kg/(m*s*K^0.5)", "Sutherland coefficient")
    sfudge = C("sutherland_fudge", 6.64, "K^0.28",
               "dimensional factor in the monomial viscosity fit")

    # Every relation here holds segment by segment, so it is written once over
    # the whole vector rather than once per segment inside a loop.
    # ---- ISA, troposphere AND stratosphere ---------------------------------
    # The model this replaces was troposphere-only: `T_sl == T_atm + L*h` and
    # the 5.257 pressure exponent are both derived for a constant lapse rate
    # and stop at the tropopause, 11,000 m / 36,089 ft. Every case in this
    # study cruises above that -- 41,580 ft on the 737 -- where the lapse was
    # being extrapolated to 203 K against the true constant 216.65 K, a 6%
    # temperature error that grows with altitude and feeds Reynolds number,
    # viscosity and density.
    # The fit is numeric in feet and pascals, so the rows are written in
    # dimensionless h and p and the reference scales carry the units.
    Pcx = Vn("P_cvx", 140.0, "-", "convex part of the pressure fit")
    Qcc = Vn("P_ccv", 0.006, "-", "concave part of the pressure fit")
    hn = Vn("h_norm", 30000.0, "-", "altitude in feet, dimensionless")
    pn = Vn("P_norm", 30000.0, "-", "pressure in pascals, dimensionless")
    cons = [
        h == hft,
        a == (gamma * R * T_atm) ** 0.5,
        V == M * a,
        rho == p_atm / (R_atm / M_atm * T_atm),
        mu == C_1 * T_atm ** 1.5 / (sfudge * T_s ** 0.72),

        # TEMPERATURE: exactly max(T_sl - L*h, T_tropopause), which is already
        # the natural GP form -- two lower bounds, with the model pushing T
        # down of its own accord. Through Sutherland, Re ~ p/T^2, so cold air
        # is cheaper drag and T settles on whichever bound is higher. No fit,
        # no error.
        T_atm + L_atm * h >= T_sl,                                   # [SP] SigEq
        T_atm >= T_trop,

        # PRESSURE: a difference-of-softmax-affine (DSMA) surrogate in h.
        #
        # log p is CONCAVE in log h -- pressure decays faster than any power
        # law -- so a softmax-affine, which is convex in log space, cannot bend
        # that way: fitted directly it gave 10.5% RMS and drove its softness
        # parameter to 1e290. A DSMA is a difference of two convex functions,
        # hence signomial, hence legal here, and it fits the same data to
        # 0.053% RMS / 0.139% max over 15,000-55,000 ft, tropopause included.
        # (Karcher, "Data Fitting with Signomial Programming Compatible
        # Difference of Convex Functions", Optim Eng 2022.)
        #
        #     p = P/Q,  P**alpha == sum c_i h**e_i,  Q**beta == sum d_j h**f_j
        hn * ft_ref == hft,
        pn * pa_ref == p_atm,
        pn * Qcc == Pcx,                                             # [SP] SigEq
        Pcx ** 20.085536923187668
            == 5.4953651911e+74 * hn ** -6.91754298
             + 1.4556490268e+00 * hn ** -4.44786657,                 # [SP] SigEq
        Qcc ** 0.6138185640656072
            == 5.4341403568e-13 * hn ** 2.29830383
             + 2.6882789843e-02 * hn ** 0.00465927,                  # [SP] SigEq

        # The fit is valid over the band it was built on. Below 15,000 ft the
        # study only passes through in climb, where these segments are not
        # sized; above 55,000 ft nothing here flies.
        hn <= 55000.0,
    ]

    return fs, cons
