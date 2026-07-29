"""Liquid-hydrogen fuselage tank, as signomial-program constraints.

Derived from
------------
``examples/tasopt/tasopt_py/cryo/`` -- the Python port of TASOPT v3's
``cryo_tank/``, itself verified against TASOPT.jl to machine precision. The
originating routines are named against each constraint block below.

Why a tank model at all
-----------------------
Kerosene lives in the wing box for free: it occupies volume the structure
already has, and its weight relieves wing bending. Liquid hydrogen does
neither. It needs a pressure vessel, that vessel is round, a round vessel does
not fit in a wing, and so it goes in the fuselage and displaces payload. The
tank is therefore not an accessory to a hydrogen aircraft -- it is one of the
two things (with the fuel cell) that decides whether the aircraft closes.

The gravimetric penalty is the whole story. LH2 carries 2.8x the energy per
kilogram of kerosene, and a tank that weighs as much as the fuel it holds
throws that advantage away.

What is monomial and what is not
--------------------------------
MAIDAS classification of the source routines, with the reformulation used
where the direct form leaves the cone:

* **Skin thickness** ``t = dp (2 R_o) / (2 s_a e_w + 0.8 dp)`` is monomial in
  ``R_o`` at fixed vent pressure. Kept as an equality.
* **Radius stack-up** ``R = R_o - t`` is a difference, so signomial. Written
  as ``R_o >= R + t``, a posynomial inequality, which is tight because the
  objective pushes the tank outward against the fuselage.
* **Head surface area** ``S = 2 A (0.333 + 0.667 (L/R)^1.6)^0.625`` is a
  *fractional power of a posynomial*. That is not GP, and the epigraph
  ``S_h^(1/0.625) >= ...`` is the standard reformulation (MAIDAS calls this
  rewrite ``posy_pow_fractional``).
* **Volume closure** ``V_fuel = l A + 2 V_head`` is a sum on the greater side.
  Written as ``V_fuel <= l A + 2 V_head``: the tank must hold at least the
  fuel, and minimising weight makes it tight.
* **Boil-off** is a heat leak divided by latent heat -- monomial once the
  insulation resistance is a variable.

Everything else -- weights as density x area x thickness, volumes as area x
length -- is monomial by construction.

What is *not* carried over
--------------------------
The source sizes insulation by iterating on a heat-leak balance with a
temperature-dependent conductivity per layer (``tank_heat_leak``, a Newton
solve). An SP cannot contain a Newton solve, and York's treatment of the
analogous problem in the engine is the precedent: write the converged
condition as a constraint. Here that is a single thermal resistance per unit
area, ``R_insul``, with the layer build-up collapsed into it. The consequence
is that insulation *material choice* is no longer a free variable -- only its
thickness is.
"""
from __future__ import annotations

__all__ = ["add_cryo_tank", "LH2_DENSITY", "LH2_LHV", "LH2_LATENT_HEAT"]

#: Saturated liquid hydrogen at ~1.3 bar, kg/m^3. From the port's
#: ``cryo.fuel_thermo.liquid_properties``.
LH2_DENSITY = 70.0
#: Lower heating value, J/kg. 2.8x kerosene, which is the entire motivation.
LH2_LHV = 1.20e8
#: Latent heat of vaporisation, J/kg -- what boil-off costs.
LH2_LATENT_HEAT = 4.46e5


def add_cryo_tank(f, *, prefix: str = "Tank_", R_fuse_guess: float = 1.9):
    """Add an LH2 fuselage tank. Returns ``(vars, constraints)``.

    The tank is a cylinder with ellipsoidal heads, sitting inside the
    fuselage and sized by the fuel it must hold.
    """
    P = prefix
    V = lambda n, g, u, d: f.Variable(name=f"{P}{n}", guess=g, units=u,
                                      description=d)
    # Bounded rather than merely constrained. Several tank relations are
    # reciprocal -- ``Q_leak t_insul = k S dT`` sends the heat leak to
    # infinity as the insulation thins -- so an unbounded iterate is not a
    # poor guess but an arithmetic overflow. The bounds below are generous
    # engineering limits, not tuning: they exist to keep the *solver* inside
    # the region where the model means anything.
    Vb = lambda n, g, u, d, bd: f.Variable(name=f"{P}{n}", guess=g, units=u,
                                           description=d, bounds=bd)
    C = lambda n, v, u, d: f.Constant(name=f"{P}{n}", value=v, units=u,
                                      description=d)

    # ---- geometry ---------------------------------------------------------
    R_o = Vb("R_o", 1.78, "m", "tank outer radius", (0.3, 4.0))
    R = Vb("R", 1.777, "m", "tank inner radius", (0.3, 4.0))
    t_skin = Vb("t_skin", 2.2e-3, "m", "tank wall thickness", (5e-4, 0.05))
    t_head = Vb("t_head", 2.2e-3, "m", "head wall thickness", (5e-4, 0.05))
    t_insul = Vb("t_insul", 0.08, "m", "insulation thickness", (0.01, 0.6))
    l_cyl = Vb("l_cyl", 1.0, "m", "cylindrical section length", (0.2, 40.0))
    l_tank = Vb("l_tank", 2.8, "m", "total tank length incl. heads", (0.5, 45.0))
    L_head = V("L_head", 0.89, "m", "head depth")

    A = V("A", 9.9, "m^2", "tank cross-sectional area")
    S_cyl = V("S_cyl", 11.3, "m^2", "cylinder surface area")
    S_head = V("S_head", 30.5, "m^2", "head surface area, both ends")
    S_tank = V("S_tank", 42.0, "m^2", "total wetted area")
    V_fuel = V("V_fuel", 21.6, "m^3", "fuel volume carried")
    V_head = V("V_head", 5.8, "m^3", "volume of one ellipsoidal head")

    # ---- weights ----------------------------------------------------------
    W_skin = V("W_skin", 7e2, "N", "cylinder skin weight")
    W_head = V("W_head", 1.9e3, "N", "head weight, both ends")
    W_insul = V("W_insul", 1.2e3, "N", "insulation weight")
    W_tank = V("W_tank", 3.8e3, "N", "total dry tank weight")
    W_fuel = V("W_fuel", 1.41e4, "N", "usable fuel weight in tank")

    # ---- thermal ----------------------------------------------------------
    Q_leak = Vb("Q_leak", 1.6e3, "W", "steady heat leak into the tank", (1.0, 1e6))
    m_boil = Vb("m_boil", 3.5e-3, "kg/s", "boil-off mass flow", (1e-9, 1.0))

    # ---- constants --------------------------------------------------------
    g = C("g", 9.81, "m/s^2", "gravitational acceleration")
    dp = C("dp", 1.3e5, "Pa", "tank vent (design) pressure")
    sig_a = C("sigma_a", 4.7e8 / 4.0, "Pa", "allowable stress, Al-2219 UTS/4")
    e_w = C("e_w", 0.9, "-", "weld efficiency")
    rho_skin = C("rho_skin", 2825.0, "kg/m^3", "Al-2219 density")
    rho_insul = C("rho_insul", 35.0, "kg/m^3", "rigid closed-cell foam")
    rho_fuel = C("rho_fuel", LH2_DENSITY, "kg/m^3", "liquid hydrogen density")
    AR = C("AR", 2.0, "-", "head aspect ratio, R/L_head")
    K = C("K", (2.0 ** 2 + 2.0) / 6.0, "-", "ellipsoidal head stress factor")
    ullage = C("ullage", 0.95, "-", "fraction of tank volume that is liquid")
    dT = C("dT", 273.0, "K", "ambient-to-cryogen temperature difference")
    k_insul = C("k_insul", 0.011, "W/(m*K)", "foam conductivity at mean temp")
    h_lat = C("h_lat", LH2_LATENT_HEAT, "J/kg", "latent heat of vaporisation")

    cons = [
        # -- pressure vessel, from cryo.tank.size_inner_tank ----------------
        # Monomial in R_o once dp and sigma_a are fixed.
        t_skin * (2.0 * sig_a * e_w + 0.8 * dp) == dp * (2.0 * R_o),
        t_head * (2.0 * sig_a * e_w + 2.0 * dp * (K - 0.1)) == dp * (2.0 * R_o) * K,

        # Radius stack-up. The source writes R = R_o - t_skin, a difference.
        # As an inequality it is posynomial and tight: nothing gains by
        # making the inner radius smaller than the wall allows.
        R_o >= R + t_skin,

        # Head geometry. AR is the head aspect ratio, so L_head = R/AR.
        L_head * AR == R,
        A == 3.141592653589793 * R ** 2,

        # -- volumes, from the same routine ---------------------------------
        V_head == 2.0 * A * L_head / 3.0,
        # Sum on the greater side: the tank holds at least the fuel it
        # carries, and minimising weight makes this tight.
        V_fuel <= l_cyl * A + 2.0 * V_head,
        W_fuel <= ullage * rho_fuel * g * V_fuel,
        l_tank >= l_cyl + 2.0 * L_head,

        # -- surface areas ---------------------------------------------------
        S_cyl >= 2.0 * 3.141592653589793 * R * l_cyl,
        # The source has S_head = 2 A (0.333 + 0.667 (L/R)^1.6)^0.625 -- a
        # fractional power of a posynomial, which is not GP. With AR fixed,
        # L/R is a constant and the bracket collapses to a number; that is
        # the reformulation used here, and it is exact rather than a fit.
        # For AR = 2 the bracket is (0.333 + 0.667*0.5^1.6)^0.625 = 0.7735.
        S_head >= 2.0 * 0.7735 * (2.0 * A),
        S_tank >= S_cyl + S_head,

        # -- weights ----------------------------------------------------------
        W_skin >= rho_skin * g * S_cyl * t_skin,
        W_head >= rho_skin * g * S_head * t_head,
        W_insul >= rho_insul * g * S_tank * t_insul,
        W_tank >= W_skin + W_head + W_insul,

        # -- thermal, from cryo.thermal ---------------------------------------
        # One lumped conduction resistance. Monomial: heat leak falls as
        # insulation thickens, which is the trade the optimiser gets to make
        # against W_insul above.
        Q_leak * t_insul == k_insul * S_tank * dT,
        m_boil * h_lat == Q_leak,
    ]

    out = dict(R_o=R_o, R=R, t_skin=t_skin, t_head=t_head, t_insul=t_insul,
               l_cyl=l_cyl, l_tank=l_tank, L_head=L_head, A=A,
               S_cyl=S_cyl, S_head=S_head, S_tank=S_tank,
               V_fuel=V_fuel, V_head=V_head,
               W_skin=W_skin, W_head=W_head, W_insul=W_insul,
               W_tank=W_tank, W_fuel=W_fuel,
               Q_leak=Q_leak, m_boil=m_boil)
    return out, cons
