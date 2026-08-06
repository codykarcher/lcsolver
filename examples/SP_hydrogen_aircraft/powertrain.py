"""Electric powertrain -- inverter, motor, ducted fan -- as SP constraints.

Derived from
------------
``examples/tasopt/tasopt_py/propsys/`` (``electric.py``, ``motor.py``) and
``engine_v3/ducted_fan_cycle.py``, all verified against TASOPT.jl. TASOPT
2.16 has no electrical system whatsoever -- only shaft power offtakes that
vanish from the cycle without going anywhere -- so every constraint here
comes from v3.

Why this is mostly monomial
---------------------------
The drivetrain turns out to be the easiest part of a hydrogen aircraft to
put in a GP, which is the opposite of the usual expectation. Component masses
are specific powers (monomial by definition), and the loss chain is a product
of efficiencies (monomial). MAIDAS classifies the port's ``ohmic_loss`` and
``eddy_loss`` as monomial directly.

The free lunch this module used to serve
--------------------------------------
The first version priced the fan by *power* (a specific-power constant) and
charged nothing for its size. An ideal actuator disc then wants **infinite
bypass**: grow the mass flow, let the jet velocity fall toward free stream,
and propulsive efficiency climbs toward one with nothing pushing back. SIA
exposed this as a monotone creep -- ``mdot_a`` drifting up 0.4 log-units per
200 iterations, objective sliding, stationarity floored at a 1/k tail. The
solver was fine; the model had a null valley.

The cure is the fan-size charge TASOPT itself carries, derived from the
port's ``engine_v3/ducted_fan.py`` (``ductedfanweight.jl``, verified against
TASOPT.jl earlier in this project):

* blade mass  ``m_fan = KTECH * 135 D^2.7 / sqrt(AR) * (sigma/1.25)^0.3
  * (U_tip/350)^0.3``  -- monomial in ``D`` at fixed tip speed;
* nacelle wetted area ``S_nace = rSnace * (pi/4) D^2`` -- monomial, and it
  buys skin-friction drag through the aircraft's wetted-area row;
* the actuator disc itself, ``mdot <= (rho/2) A_disc (u_0 + u_j)`` -- a sum
  on the greater side, one more legitimately signomial row.

With those three, mass flow has a genuine optimum: bigger fans buy
propulsive efficiency and pay in blade mass and nacelle drag. That is the
bypass-ratio trade -- the thing TASOPT exists to sweep -- recovered here
because the *solver refused to converge without it*.

The two that are not monomial
-----------------------------
* **Motor sizing.** ``size_PMSM!`` is limit-driven: tip speed sets the
  radius, flux continuity sets the back-iron, saturation splits the tooth and
  slot annulus. Faithfully that is a chain of equalities with a square root of
  a difference in the middle, and MAIDAS calls the flux terms ``not_gp``. The
  useful fact from the port survives without it: **radius_gap = U_max / Omega
  exactly**, so a faster motor is a smaller motor, and mass falls as
  ``Omega^-1`` at fixed power. That is captured here as a monomial specific
  power with a speed exponent, which is a fit, and is labelled as one.
* **Fan thrust.** ``F = mdot (u_j - u_0)``, a difference, so signomial. It is
  written below as a sum on the greater side, exactly as York writes
  ``F <= F_6 + F_8``.

The thing worth noticing
------------------------
The motor is *not* the mass problem. At 19 kW/kg for the inverter and ~10
kW/kg for a high-speed PMSM, a 5 MW powertrain is around 800 kg of machinery
-- against a fuel cell stack that, at the port's own ``f_outer = 4``, is four
tonnes. The stack dominates, and the stack is sized by current density, which
is why ``j`` is the variable the optimiser cares most about.
"""
from __future__ import annotations

__all__ = ["add_powertrain"]


def add_powertrain(f, N, *, prefix: str = "PT_", n_fans: int = 2,
                   u_free: float = 232.0, rho_free: float = 0.38):
    """Add inverter + motor + ducted fan for ``n_fans`` propulsors."""
    P = prefix
    V = lambda n, g, u, d: f.Variable(name=f"{P}{n}", guess=g, units=u,
                                      description=d)
    Vn = lambda n, g, u, d: f.Variable(name=f"{P}{n}", guess=g, units=u,
                                       description=d, size=N)
    C = lambda n, v, u, d: f.Constant(name=f"{P}{n}", value=v, units=u,
                                      description=d)

    # ---- hardware, shared across segments ---------------------------------
    W_motor = V("W_motor", 6.9e3, "N", "all motors, installed")
    W_inv = V("W_inv", 3.7e3, "N", "all inverters")
    W_fan = V("W_fan", 1.5e4, "N", "fans, nacelles and pylons, installed")
    W_pt = V("W_pt", 2.6e4, "N", "total powertrain weight")
    P_shaft_max = V("P_shaft_max", 7.0e6, "W", "rated shaft power, all fans")
    D_fan = f.Variable(name=f"{P}D_fan", guess=1.85, units="m",
                       description="fan diameter, one fan",
                       bounds=(0.5, 4.5))
    S_nace = f.Variable(name=f"{P}S_nace", guess=32.0, units="m^2",
                        description="nacelle wetted area, all fans",
                        bounds=(1.0, 400.0))

    # ---- per-segment ------------------------------------------------------
    P_shaft = Vn("P_shaft", 6.9e6, "W", "shaft power, all fans")
    F_net = Vn("F_net", 2.75e4, "N", "net thrust, all fans")
    mdot_a = Vn("mdot_a", 700.0, "kg/s", "fan mass flow, all fans")
    u_j = Vn("u_j", 272.0, "m/s", "jet velocity")

    # ---- constants --------------------------------------------------------
    g = C("g", 9.81, "m/s^2", "gravitational acceleration")
    sp_inv = C("sp_inv", 19e3, "W/kg", "inverter specific power, SiC/GaN")
    sp_mot = C("sp_mot", 10e3, "W/kg", "motor specific power, high-speed PMSM")
    eta_inv = C("eta_inv", 0.995, "-", "inverter efficiency")
    eta_mot = C("eta_mot", 0.96, "-", "motor efficiency")
    eta_cbl = C("eta_cbl", 0.99, "-", "cable efficiency")
    u_0 = C("u_0", u_free, "m/s", "free-stream velocity")
    rho_0 = C("rho_0", rho_free, "kg/m^3", "free-stream density")
    # Fan-weight constants, straight from the port's ducted_fan.py.
    # Normalised by D_ref = 1 m so the fractional power acts on a
    # dimensionless ratio -- the same trick as j_ref in fuel_cell.py; a unit
    # of kg/m^2.7 does not survive the unit corrector.
    k_blade = C("k_blade", 0.5 * 135.0 / (3.0 ** 0.5)
                * (0.4 / 1.25) ** 0.3 * (300.0 / 350.0) ** 0.3,
                "kg", "KTECH 135/sqrt(AR) (sigma/1.25)^.3 (Utip/350)^.3")
    D_ref = C("D_ref", 1.0, "m", "reference diameter for the blade-mass fit")
    rSnace = C("rSnace", 6.0, "-", "nacelle wetted area over disc area")
    k_nace = C("k_nace", 220.0, "N/m^2", "nacelle areal weight, ~4.6 psf")
    fpylon = C("fpylon", 0.10, "-", "pylon fraction of fan+nacelle weight")
    n_f = C("n_fans", float(n_fans), "-", "number of fans")

    cons = [
        # Motor and inverter: specific power, monomial by definition.
        W_inv >= g * P_shaft_max / sp_inv,
        W_motor >= g * P_shaft_max / sp_mot,
        # Fan and nacelle: sized by DIAMETER, not power -- the charge that
        # closes the infinite-bypass free lunch. Blade mass D^2.7 and
        # nacelle area D^2, both from the port's ducted_fan_weight.
        S_nace >= n_f * rSnace * (3.141592653589793 / 4.0) * D_fan ** 2,
        W_fan >= (1.0 + fpylon) * (g * k_blade * n_f
                                   * (D_fan / D_ref) ** 2.7
                                   + 1.8 * k_nace * S_nace),
        W_pt >= W_inv + W_motor + W_fan,
    ]

    for i in range(N):
        cons += [
            P_shaft_max >= P_shaft[i],
            # Momentum thrust, F = mdot (u_j - u_0). A difference, so
            # signomial; written as a sum on the greater side exactly as York
            # writes F <= F_6 + F_8.
            F_net[i] + mdot_a[i] * u_0 <= mdot_a[i] * u_j[i],
            # The actuator disc: the flow the fans swallow is set by the disc
            # area and the velocity through it, (u_0 + u_j)/2. Sum on the
            # greater side -- signomial -- and the row that makes mass flow
            # cost something.
            mdot_a[i] <= 0.5 * rho_0 * n_f
                * (3.141592653589793 / 4.0) * D_fan ** 2 * (u_0 + u_j[i]),
            # Ideal fan power is the *excess* kinetic energy over free stream,
            # 0.5 mdot (u_j^2 - u_0^2). Using the absolute flux instead makes
            # the propulsor look about twice as bad as it is, because it
            # charges the fan for kinetic energy the air already had.
            P_shaft[i] + 0.5 * mdot_a[i] * u_0 ** 2
                >= 0.5 * mdot_a[i] * u_j[i] ** 2,
        ]

    out = dict(W_motor=W_motor, W_inv=W_inv, W_fan=W_fan, W_pt=W_pt,
               P_shaft_max=P_shaft_max, D_fan=D_fan, S_nace=S_nace,
               P_shaft=P_shaft,
               F_net=F_net, mdot_a=mdot_a, u_j=u_j,
               eta_chain=(eta_inv, eta_mot, eta_cbl))
    return out, cons
