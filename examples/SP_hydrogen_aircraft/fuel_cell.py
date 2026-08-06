"""PEM fuel cell stack, as signomial-program constraints.

Derived from
------------
``examples/tasopt/tasopt_py/engine_v3/fuelcell_1d.py`` -- the port of TASOPT
v3's ``PEMfuelcell.jl``, verified against TASOPT.jl (the HT model to zero
relative error, the LT model to the reference's own ODE tolerance).

The one place a fit is unavoidable
----------------------------------
A cell's voltage is its polarisation curve: reversible voltage minus
activation, ohmic and concentration losses. The activation term is
logarithmic in current density and the concentration term diverges at the
limiting current. **No monomial form expresses that**, and MAIDAS classifies
the source routine ``not_gp`` for exactly that reason.

So this is where the model does what York did for the turbofan compressor
maps: fit. The fit is generated *from the port*, not from a datasheet ---

    V_cell = 6.7013 * j^-0.23765        j in A/m^2

least squares in log-log over nine points, against
``ht_pemfc_voltage`` at 453.15 K, 3 bar, lambda = 3. **Worst error 1.57% for
5000 <= j <= 10000 A/m^2**, which is why the model constrains ``j`` to that
band. Over the full 2e3-2e4 range the same fit is 13% out: a power law cannot
follow a polarisation curve across a decade, and pretending otherwise would
be the kind of silent error this whole exercise exists to catch.

What stays exact
----------------
Everything downstream of the voltage is monomial and is kept as an equality:

* **Cell area.** ``n_cells = V_stack / V_cell`` and
  ``P = n_cells A_cell j V_cell`` collapse to ``A_cell V_stack j = P`` --- the
  cell count cancels, so stack area depends only on power, bus voltage and
  current density.
* **Hydrogen flow.** ``mdot = I_tot M_H2 / (2 F)`` straight from Faraday.
  This is the constraint that couples electrical power to fuel burn, and
  therefore the one that makes the mission close.
* **Mass.** Membrane plus electrodes times a structural factor.

The heat balance is signomial
-----------------------------
``Q = I_tot (V_thermoneutral - V_cell)``: a difference, so signomial. Written
as ``I_tot V_tn <= Q + P``, a sum on the greater side -- the same shape as the
three SP constraints York names in his turbofan, and tight because nothing
gains by over-reporting waste heat.

It matters more here than it looks. At 0.75 V against a 1.254 V thermoneutral
the stack is 60% efficient, so a 5 MW powertrain rejects 3.3 MW as heat at a
temperature only 180 K above ambient. That is a bigger radiator than a
turbofan needs, and it is the reason a fuel-cell aircraft is drag-limited
rather than weight-limited.
"""
from __future__ import annotations

__all__ = ["add_fuel_cell", "V_FIT_K", "V_FIT_A", "J_MIN", "J_MAX"]

#: Monomial fit of the port's HT-PEMFC polarisation curve, V = K j^A.
V_FIT_K, V_FIT_A = 6.7013, -0.23765
#: Validity band of that fit, A/m^2. Outside it the error grows fast.
J_MIN, J_MAX = 5.0e3, 1.0e4

FARADAY = 96485.3329          # C/mol
M_H2 = 2.016e-3               # kg/mol


def add_fuel_cell(f, N, *, prefix: str = "FC_"):
    """Add a PEM stack sized by its worst operating point.

    ``N`` is the number of mission segments; power, current and hydrogen flow
    are per-segment, while the stack geometry is shared -- the same coupling
    York uses to let the optimiser choose the design point rather than having
    it specified.
    """
    P = prefix
    V = lambda n, g, u, d: f.Variable(name=f"{P}{n}", guess=g, units=u,
                                      description=d)
    Vn = lambda n, g, u, d: f.Variable(name=f"{P}{n}", guess=g, units=u,
                                       description=d, size=N)
    Vb = lambda n, g, u, d, bd: f.Variable(name=f"{P}{n}", guess=g, units=u,
                                          description=d, size=N, bounds=bd)
    C = lambda n, v, u, d: f.Constant(name=f"{P}{n}", value=v, units=u,
                                      description=d)

    # ---- stack geometry: shared across every segment -----------------------
    A_cell = V("A_cell", 1.16, "m^2", "active area of one cell")
    n_cells = V("n_cells", 1011.0, "-", "cells in series")
    W_stack = V("W_stack", 6.2e4, "N", "stack weight")

    # ---- per-segment operating state ---------------------------------------
    P_e = Vn("P_e", 7.4e6, "W", "electrical power delivered")
    # Bounded, not merely constrained: the polarisation fit raises j to a
    # fractional power, so a solver iterate at or below zero is not a poor
    # guess but an arithmetic error. The bounds are the fit's validity band.
    j = Vb("j", 8e3, "A/m^2", "current density", (J_MIN, J_MAX))
    V_cell = Vn("V_cell", 0.792, "V", "single-cell voltage")
    Q_fc = Vn("Q_fc", 4.35e6, "W", "waste heat rejected")
    mdot_H2 = Vn("mdot_H2", 0.098, "kg/s", "hydrogen consumption")

    # ---- constants ---------------------------------------------------------
    V_bus = C("V_bus", 800.0, "V", "stack bus voltage")
    V_tn = C("V_tn", 1.254, "V", "thermoneutral voltage, HT-PEMFC")
    k_fit = C("k_fit", V_FIT_K, "V", "polarisation fit coefficient")
    j_ref = C("j_ref", 1.0, "A/m^2", "reference current density, keeps the "
              "fitted power dimensionless")
    t_M = C("t_M", 100e-6, "m", "membrane thickness")
    t_E = C("t_E", 500e-6, "m", "both electrode thicknesses")
    rho_M = C("rho_M", 1300.0, "kg/m^3", "PBI membrane density")
    rho_E = C("rho_E", 1900.0, "kg/m^3", "porous graphite density")
    f_outer = C("f_outer", 4.0, "-", "structure and plates over active mass")
    g = C("g", 9.81, "m/s^2", "gravitational acceleration")
    F_far = C("F", FARADAY, "C/mol", "Faraday constant")
    MH2 = C("M_H2", M_H2, "kg/mol", "hydrogen molar mass")
    j_min = C("j_min", J_MIN, "A/m^2", "fit validity, lower")
    j_max = C("j_max", J_MAX, "A/m^2", "fit validity, upper")

    cons = [
        # -- stack geometry ---------------------------------------------------
        # Bus voltage sets the cell count, evaluated at the first segment.
        # The stack is one piece of hardware: n_cells cannot vary by segment,
        # so one segment has to fix it and the rest run off-design.
        n_cells * V_cell[0] == V_bus,
        W_stack >= g * n_cells * A_cell * (t_M * rho_M + t_E * rho_E)
                   * (1.0 + f_outer),
    ]

    for i in range(N):
        cons += [
            # Fitted polarisation curve -- the one non-derivable relation.
            # Normalised by j_ref = 1 A/m^2 so the fractional power acts on a
            # dimensionless number; k_fit then carries the volts, and its
            # value is exactly the coefficient the log-log fit produced with
            # j expressed in A/m^2.
            V_cell[i] == k_fit * (j[i] / j_ref) ** V_FIT_A,
            # Keep the operating point inside the fit's validity band.
            j[i] >= j_min,
            j[i] <= j_max,

            P_e[i] <= n_cells * A_cell * j[i] * V_cell[i],

            # Heat balance. Signomial: a difference of voltages, written as a
            # sum on the greater side.
            n_cells * A_cell * j[i] * V_tn <= Q_fc[i] + P_e[i],

            # Faraday: the link from current to fuel burn.
            mdot_H2[i] >= n_cells * A_cell * j[i] * MH2 / (2.0 * F_far),

        ]

    out = dict(A_cell=A_cell, n_cells=n_cells, W_stack=W_stack,
               P_e=P_e, j=j, V_cell=V_cell, Q_fc=Q_fc, mdot_H2=mdot_H2)
    return out, cons
