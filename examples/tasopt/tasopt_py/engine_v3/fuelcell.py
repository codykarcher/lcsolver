"""PEM fuel cell -- ``PEMfuelcell.jl``.

The thing that makes a hydrogen-*electric* aircraft close: it turns the
hydrogen the cryogenic tank carries into electricity for the motors, without
a turbine. Nothing in TASOPT 2.16 corresponds to any of it.

What sets the voltage
---------------------
A cell's open-circuit voltage is thermodynamic -- about 1.23 V for liquid
water product -- and everything after that is loss. Three of them, and they
dominate in different places:

* **Activation.** The charge-transfer barrier at each electrode. Logarithmic
  in current, so it costs most at *low* current and is why a cell never
  reaches its reversible voltage.
* **Ohmic.** Membrane resistance, linear in current.
* **Concentration.** Reactant starvation as the limiting current is
  approached. ``c ln(j_L / (j_L - j))``, so it diverges at ``j_L``.

The sum gives the polarisation curve, and the **power** density ``j V`` has
a maximum in the middle of it -- which is why sizing a stack has two roots
for the same power and the reference is careful to take the smaller current.

The product phase matters
-------------------------
Above 383.15 K the water leaves as vapour and the reversible voltage is
1.185 V; below it the water is liquid and the voltage is 1.229 V. The
entropy change differs by nearly a factor of four between the two, so the
temperature coefficient does too. That switch is a real discontinuity in the
model, not a smooth transition.

Verified against TASOPT.jl; see ``tests/test_fuelcell.py``.
"""
from __future__ import annotations

import math

__all__ = ["R_GAS", "FARADAY", "cell_voltage_simple", "water_sat_pressure",
           "conductivity_nafion", "conductivity_pbi", "lambda_water",
           "porous_diffusion", "stack_size", "stack_operate",
           "stack_weight", "power_density", "V_HEAT", "T_VAP"]

R_GAS = 8.314           # J/(mol K)
FARADAY = 96485.3329    # C/mol
T0_REF = 298.15         # K
P0_REF = 1.01325e5      # Pa

#: Water leaves as vapour above this; below it as liquid. The reversible
#: voltage and the reaction entropy both jump here.
T_VAP = 383.15

#: Thermoneutral voltage -- the voltage at which a cell would generate no
#: heat. The difference from the operating voltage is what must be cooled.
V_HEAT = {"LT-PEMFC": 1.482, "HT-PEMFC": 1.254}

#: Membrane densities, kg/m^3, and the porous graphite plate density.
RHO_MEMBRANE = {"LT-PEMFC": 1970.0, "HT-PEMFC": 1300.0}
RHO_ELECTRODE = 1.9e3


def cell_voltage_simple(j: float, T: float, p_H2: float,
                        p_air: float) -> float:
    """Single-cell voltage, V, from the simple polarisation model.

    ``j`` current density (A/m^2), ``T`` temperature (K), ``p_H2`` and
    ``p_air`` the anode and cathode pressures (Pa).

    Every kinetic constant here is fixed in the source -- exchange current
    densities, transfer coefficients, area-specific resistance, leakage and
    limiting current. None is an input, so this is one particular cell
    rather than a family.
    """
    n = 2                       # electrons per half-reaction
    j0_H2, alpha_H2 = 1.0e3, 0.5
    j0_O2, alpha_O2 = 1.0, 0.3
    ASR = 1.0e-6                # ohm m^2
    j_leak = 1.0e2              # A/m^2
    j_L = 2.0e4                 # A/m^2, limiting current
    c = 0.1                     # V, concentration-loss scale

    p_O2 = 0.2095 * p_air
    a_H2 = p_H2 / P0_REF
    a_O2 = p_O2 / P0_REF

    if T > T_VAP:               # gaseous product
        ds0, E0 = -44.34, 1.185
    else:                       # liquid product
        ds0, E0 = -163.23, 1.229

    E_r = (E0 + ds0 / (n * FARADAY) * (T - T0_REF)
           - R_GAS * T / (n * FARADAY) * math.log(1.0 / (a_H2 * a_O2 ** 0.5)))

    a_A = -R_GAS * T / (alpha_H2 * n * FARADAY) * math.log(j0_H2)
    b_A = R_GAS * T / (alpha_H2 * n * FARADAY)
    a_C = -R_GAS * T / (alpha_O2 * n * FARADAY) * math.log(j0_O2)
    b_C = R_GAS * T / (alpha_O2 * n * FARADAY)

    if j + j_leak >= j_L:
        raise ValueError(
            f"current density {j:.1f} A/m^2 plus leakage reaches the "
            f"limiting current {j_L:.0f}; the concentration loss diverges "
            "there. The reference takes the logarithm regardless.")

    eta_act = a_A + b_A * math.log(j + j_leak) + a_C + b_C * math.log(
        j + j_leak)
    eta_ohm = j * ASR
    eta_conc = c * math.log(j_L / (j_L - (j + j_leak)))

    return E_r - eta_act - eta_ohm - eta_conc


def water_sat_pressure(T: float) -> float:
    """Saturation pressure of water, Pa.

    Two correlations spliced at 100 C -- Huang (2018) below, Jiao and Li
    (2010) above. They are **not** continuous there; see
    ``DISCREPANCIES.md`` §71.
    """
    t = T - 273.15
    if t < 100.0:
        return math.exp(34.494 - 4924.99 / (t + 237.1)) / (t + 105.0) ** 1.57
    return 0.68737 * T ** 3 - 732.39 * T ** 2 + 263390.0 * T - 31919000.0


def conductivity_nafion(T: float, lam: float) -> float:
    """Nafion membrane conductivity, S/m, from water content ``lam``.

    Linear in water content and Arrhenius in temperature. It goes
    **negative** below ``lam = 0.634``, which the reference does not guard.
    """
    sigma_30 = (0.005139 * lam - 0.00326) * 1.0e2
    return math.exp(1268.0 * (1.0 / 303.0 - 1.0 / T)) * sigma_30


def conductivity_pbi(T: float, DL: float, RH: float) -> float:
    """Phosphoric-acid-doped PBI conductivity, S/m -- Jiao and Li (2009).

    The high-temperature membrane. ``DL`` is the acid doping level and
    ``RH`` the relative humidity. The humidity correction is piecewise in
    temperature with three branches.
    """
    E_a = -619.6 * DL + 21750.0
    a = 168.0 * DL ** 3 - 6324.0 * DL ** 2 + 65750.0 * DL + 8460.0
    if T <= 413.15:
        b = 1.0 + (0.01704 * T - 4.767) * RH
    elif T <= 453.15:
        b = 1.0 + (0.1432 * T - 56.89) * RH
    else:
        b = 1.0 + (0.7 * T - 309.2) * RH
    return a * b / T * math.exp(-E_a / (R_GAS * T))


def lambda_water(a: float) -> float:
    """Membrane water content from water activity ``a``.

    A cubic below saturation and a line above it. The reference writes a
    third branch for ``a < 1`` after an ``a >= 1`` branch, which is
    unreachable -- see §72.
    """
    if a < 1.0:
        return 0.043 + 17.81 * a - 39.85 * a ** 2 + 36.0 * a ** 3
    return 14.0 + 1.4 * (a - 1.0)


def porous_diffusion(D: float, eps: float, tau: float) -> float:
    """Effective diffusivity through a porous electrode."""
    return D * eps / tau


def power_density(j: float, T: float, p_H2: float, p_air: float) -> float:
    """Areal power density ``j V``, W/m^2.

    Has a **maximum** in current density: activation and ohmic losses grow
    while the voltage falls, so the product peaks and then collapses toward
    the limiting current. That is why :func:`stack_operate` has two roots.
    """
    return j * cell_voltage_simple(j, T, p_H2, p_air)


def stack_size(P_des: float, V_des: float, j: float, T: float, p_H2: float,
               p_air: float, kind: str = "LT-PEMFC") -> tuple:
    """``(n_cells, A_cell, Q)`` for a stack at its design point.

    ``P_des`` design power (W), ``V_des`` design stack voltage (V), ``j``
    the chosen current density. ``Q`` is the heat to be rejected -- the gap
    between the thermoneutral voltage and the operating one, which for a
    cell running at 0.7 V is more heat than electricity.
    """
    P2A = power_density(j, T, p_H2, p_air)
    V_cell = P2A / j
    n_cells = V_des / V_cell
    A_cell = P_des / (n_cells * P2A)
    Q = n_cells * A_cell * (V_HEAT[kind] - V_cell) * j
    return n_cells, A_cell, Q


def stack_operate(P_stack: float, n_cells: float, A_cell: float, T: float,
                  p_H2: float, p_air: float,
                  kind: str = "LT-PEMFC") -> tuple:
    """``(V_stack, Q)`` for a stack off design.

    Solves for the current density that delivers ``P_stack``. **There are
    two roots** -- the power density peaks and falls -- and the physical one
    is the smaller current. The reference starts ``find_zero`` from 1 A/m^2
    with a comment saying so; this brackets below the peak instead, which
    cannot land on the wrong side.
    """
    P2A = P_stack / (n_cells * A_cell)

    # Locate the peak, then bracket strictly below it.
    j_peak = _peak_current(T, p_H2, p_air)
    if P2A > power_density(j_peak, T, p_H2, p_air):
        raise ValueError(
            f"stack power density {P2A:.1f} W/m^2 exceeds the cell's "
            f"maximum of {power_density(j_peak, T, p_H2, p_air):.1f} at "
            f"j = {j_peak:.0f} A/m^2; no current density delivers it")

    lo, hi = 1.0e-6, j_peak

    def f(j):
        return power_density(j, T, p_H2, p_air) - P2A

    flo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0 or (hi - lo) < 1.0e-12 * max(1.0, mid):
            break
        if flo * fm < 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    j = 0.5 * (lo + hi)

    V_cell = P2A / j
    return V_cell * n_cells, n_cells * A_cell * (V_HEAT[kind] - V_cell) * j


def _peak_current(T: float, p_H2: float, p_air: float) -> float:
    """The current density at which areal power peaks, A/m^2."""
    lo, hi = 1.0, 2.0e4 - 1.0e2 - 1.0
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - phi * (hi - lo)
    d = lo + phi * (hi - lo)
    fc, fd = (power_density(c, T, p_H2, p_air),
              power_density(d, T, p_H2, p_air))
    for _ in range(200):
        if abs(hi - lo) < 1.0e-9 * max(1.0, abs(lo)):
            break
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - phi * (hi - lo)
            fc = power_density(c, T, p_H2, p_air)
        else:
            lo, c, fc = c, d, fd
            d = lo + phi * (hi - lo)
            fd = power_density(d, T, p_H2, p_air)
    return 0.5 * (lo + hi)


def stack_weight(n_cells: float, A_cell: float, t_M: float, t_A: float,
                 t_C: float, fouter: float,
                 kind: str = "LT-PEMFC", gee: float = 9.81) -> float:
    """Stack weight, N.

    Membrane plus two electrode plates per cell, times a fraction for the
    housing. Nothing else -- no manifolds, no coolant, no compressor.
    """
    return (gee * n_cells * A_cell
            * ((t_A + t_C) * RHO_ELECTRODE + t_M * RHO_MEMBRANE[kind])
            * (1.0 + fouter))
