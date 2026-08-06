"""Battery pack, sized by whichever of energy or power binds.

Source
------
No source model -- there is no battery in SPaircraft, TASOPT or the York
turbofan SP. This is written from the standard pack-level relations.

Why it is the easiest component here
------------------------------------
A battery does not get lighter as you fly it. That single fact removes the
weight-decrement chain (``W[i+1] == W[i] - g*mdot*t``) that every fuel-burning
architecture in this package needs, and with it the Breguet structure. The
mission closes on energy instead of mass.

The one interesting constraint
------------------------------
A pack has to satisfy an energy requirement AND a power requirement, and
which one binds is a design outcome, not an input:

    W_batt >= E_required / (e_spec * DoD)      -- energy-limited
    W_batt >= P_peak / p_spec                  -- power-limited

That is a max on the greater side, so it is two posynomial inequalities and
exactly GP-representable. Short-range designs come out power-limited (the
pack is sized by takeoff), long-range ones energy-limited (sized by cruise),
and the crossover falls out of the solve rather than being assumed.

Specific energy is quoted at PACK level, not cell level. The distinction is
about a third of the mass: cells alone reach 250-300 Wh/kg today, but a
flight pack carries module structure, busbars, contactors, cooling plates and
a BMS. ``e_spec`` here is the number after all of that.
"""
from __future__ import annotations


def add_battery(f, *, prefix="Batt_", e_spec=200.0, p_spec=1500.0,
                DoD=0.85, eta_pack=0.96, f_therm=0.05):
    """Add a battery pack. Returns ``(vars, constraints)``.

    Returns ``(group, constraints)`` -- read members as ``batt.W_batt``.

    ``e_spec``  pack specific energy, Wh/kg. 200 is a credible 2025 flight
                pack; 150 conservative; 350-500 is where the studies that
                claim regional viability tend to sit.
    ``p_spec``  pack specific power, W/kg. Rarely binds above ~1000 for
                transport duty cycles, but it is what makes short-range,
                high-thrust designs come out heavier than energy alone says.
    ``DoD``     usable depth of discharge. Never 1.0: cycle life collapses if
                the pack is run flat, and reserve energy has to come from
                somewhere that is not the usable band.
    ``eta_pack`` discharge efficiency, pack terminals to inverter input.
    ``f_therm`` waste heat as a fraction of delivered energy, for the cooling
                drag term the airframe charges.
    """
    batt = f.group("battery", prefix=prefix)
    V = lambda n, g, u, d, bd: batt.Variable(n, g, u, d, bounds=bd)
    C = batt.Constant

    W_batt = V("W_batt", 1.0e5, "N", "battery pack weight", (1e3, 1e7))
    E_batt = V("E_batt", 1.0e11, "J", "usable pack energy", (1e8, 1e13))
    E_req = V("E_req", 1.0e11, "J", "mission electrical energy", (1e8, 1e13))
    P_peak = V("P_peak", 5.0e6, "W", "peak electrical power", (1e4, 1e9))
    Q_waste = V("Q_waste", 2.0e5, "W", "pack waste heat at peak", (1e2, 1e8))

    g_ = C("g", 9.81, "m/s^2", "gravitational acceleration")
    e_s = C("e_spec", e_spec * 3600.0, "J/kg", "pack specific energy")
    p_s = C("p_spec", p_spec, "W/kg", "pack specific power")
    dod = C("DoD", DoD, "-", "usable depth of discharge")
    eta = C("eta_pack", eta_pack, "-", "pack discharge efficiency")
    f_th = C("f_therm", f_therm, "-", "waste heat fraction")

    cons = [
        # Energy: what the mission asks for, after pack losses and with only
        # DoD of the nameplate usable.
        E_batt * eta * dod >= E_req,
        # The two sizing rules. Whichever is tighter sets the pack; the model
        # is not told which.
        W_batt >= E_batt * g_ / e_s,
        W_batt >= P_peak * g_ / p_s,
        # Waste heat, for the airframe's cooling-drag charge.
        Q_waste >= f_th * P_peak,
    ]

    return batt, cons
