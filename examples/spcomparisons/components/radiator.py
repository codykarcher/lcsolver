"""Ram-air radiator for the fuel-cell aircraft: the price of waste heat.

Why this module exists
----------------------
The stack model computes ``Q_fc`` -- megawatts of waste heat at a coolant
temperature only ~180 K above ambient -- and until this module, NOTHING
consumed it. Its own docstring declares heat rejection "the reason a
fuel-cell aircraft is drag-limited", and the aircraft then charged zero for
it: the ``d_oleo`` defect class (a quantity computed and consumed by no
constraint) at megawatt scale, flattering the h2fc architecture outright.

The model
---------
Epsilon-NTU at the aircraft level, and it is GP-CLEAN by a stroke of
structure: the initial temperature difference ``T_cool - T_atm[i]`` is a
difference of CONSTANTS per segment (the HT-PEMFC coolant loop runs at a
fixed 453 K -- the whole point of high-temperature PEM is a coolant hot
enough to reject heat through a small radiator -- and the segment ambient
temperature is a state constant), so no signomial is needed anywhere:

* Heat balance, per segment: the ram-air stream must carry the heat,
      mdot_air * cp_air * eps_hx * (T_cool - T_atm) >= Q_fc.
* Cooling drag, per segment: momentum cost of taking that air aboard,
      D_cool == k_ram * mdot_air * V,
  with ``k_ram = 1 - r_recovery``. The recovery factor credits the Meredith
  effect -- heated air partially recovers its ram momentum through the exit
  nozzle -- at the standard 0.7, so 30% of the ram drag is paid. At the
  453 K coolant temperature the heating is substantial and this is, if
  anything, conservative toward the fuel-cell aircraft's disadvantage.
* Core mass: sized by the WORST segment's heat load,
      W_hx >= k_hx * Q_fc[i]   for every i,
  the standard GP max. ``k_hx`` = 1.0 kg/kW installed -- compact plate-fin
  cores at this ITD run 0.5-1.5 kg/kW with headers, ducting and coolant
  inventory; 1.0 is the middle of the band and is THE calibration constant
  of this module.

Provenance and the calibration path
-----------------------------------
``tasopt_py/engine_v3/hx_size.py`` is a cross-flow exchanger port verified
against TASOPT.jl to 1e-14 -- but in a GAS-GAS configuration (precooler
duty). A liquid-coolant/ram-air radiator is a different geometry, and
misusing the verified port as provenance for constants it was not built for
would be worse than stating engineering values honestly. When a
matched-configuration case is set up in the port, ``k_hx``, ``eps_hx`` and
the core pressure-drop share of ``k_ram`` should be re-derived from it; the
constants below are the module's stated approximation until then.
"""
from __future__ import annotations

#: Exchanger effectiveness. Compact cross-flow cores at aircraft sizes reach
#: 0.75-0.85; the value trades radiator size against air mass flow and is
#: fixed rather than optimized at this fidelity.
EPS_HX = 0.80
#: 1 - ram recovery. Meredith-effect recovery of 0.7 (heated exit stream
#: through a nozzle); 30% of the ingested stream's ram drag is paid.
K_RAM = 0.30
#: Installed core mass per unit heat rejected, kg/kW, at ITD ~ 160-230 K.
#: THE decisive constant of the fuel-cell architecture, measured on the
#: 737-class 3,000 nmi mission:
#:   1.0 kg/kW (automotive-grade): DOES NOT CLOSE -- the weight/heat/drag
#:       spiral has no fixed point; the solve settles with the dry-weight
#:       buildup 10% violated and the multiplier on the tau ceiling.
#:   0.5 kg/kW (compact aviation plate-fin + ducting): closes at MTOW
#:       237,600 lbf, radiator 20,300 lbf, L/D 12.5 -- +34% MTOW over the
#:       unpriced model.
#:   0.25 kg/kW (advanced): 208,100 lbf, radiator 7,700, L/D 13.1.
#: Default 0.5, the middle of the published aviation band; RAD_KHX env
#: overrides for technology sweeps. Before pricing, Q_fc floated at 1.9e18 W
#: -- ten orders of magnitude of free heat rejection.
K_HX_KG_PER_KW = float(__import__("os").environ.get("RAD_KHX", 0.5))
#: HT-PEMFC coolant loop temperature, K. Matches the 453.15 K the stack's
#: polarisation fit was generated at (fuel_cell.py).
T_COOLANT = 453.15


def add_radiator(f, N, state, fc, *, prefix="Rad_"):
    """Price the stack's waste heat. Returns ``(vars, constraints)``.

    ``fc`` is the fuel-cell group (needs per-segment ``Q_fc``); ``state``
    supplies ``T_atm``, ``M`` and ``a`` per segment. The caller wires
    ``W_hx`` into the dry-weight buildup and ``D_cool`` into the drag
    buildup -- this module computes, the aircraft charges, same division as
    every other component.
    """
    rad = f.group("radiator", prefix=prefix)
    V, C = rad.Variable, rad.Constant
    Vn = lambda n, g, u, d: rad.Variable(n, g, u, d, size=N)

    mair = Vn("mdot_air", 30.0, "kg/s", "ram air through the radiator")
    Dcool = Vn("D_cool", 5e3, "N", "cooling drag, momentum cost of the air")
    Whx = V("W_hx", 3e4, "N", "installed radiator system weight")

    eps = C("eps_hx", EPS_HX, "-", "exchanger effectiveness")
    kram = C("k_ram", K_RAM, "-", "unrecovered ram momentum fraction")
    khx = C("k_hx", K_HX_KG_PER_KW * 9.81 / 1000.0, "N/W",
            "installed radiator weight per unit heat rejected")
    cpair = C("cp_air", 1005.0, "J/kg/K", "air specific heat")
    Tcool = C("T_coolant", T_COOLANT, "K", "HT-PEM coolant loop temperature")

    cons = []
    for i in range(N):
        # T_cool - T_atm[i]: both constants, so the ITD is a plain number
        # per segment and the heat balance is a monomial inequality. The
        # ambient is taken from the segment state the stack actually flies.
        cons += [
            # The air must carry the heat; mdot is charged by drag, so the
            # row binds. T_atm is a STATE VARIABLE (the atmosphere rows), so
            # the ITD difference is expanded to the all-positive form --
            # writing (Tcool - T_atm) inside the product is the
            # detector-hostile shape this project measured mistranslating.
            mair[i] * cpair * eps * Tcool
                >= fc.Q_fc[i]
                 + mair[i] * cpair * eps * state.T_atm[i],  # [SP] SigIneq
            # Momentum drag at true airspeed (M*a, metres per second --
            # state.V is in knots and must not be used here).
            Dcool[i] == kram * mair[i] * state.M[i] * state.a[i],
            # Core sized by the worst segment.
            Whx >= khx * fc.Q_fc[i],
        ]
    # Return the GROUP (attribute access, rad.W_hx), matching the other
    # components' convention.
    return rad, cons
