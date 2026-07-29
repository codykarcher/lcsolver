"""One-dimensional PEM fuel cell models -- ``PEMfuelcell.jl``.

The simple polarisation curve in :mod:`tasopt_py.engine_v3.fuelcell` fixes
every kinetic constant and knows nothing about what the gases are doing. This
is the model that does: it transports species through the porous electrodes,
tracks water through the membrane, and gets the membrane conductivity from
the local humidity rather than a constant.

Why it matters that the electrodes are modelled
-----------------------------------------------
A fuel cell's cathode consumes oxygen faster than diffusion can replace it,
so the oxygen fraction at the *catalyst* is lower than at the channel. Since
the cathode overvoltage is logarithmic in that fraction, getting it from the
channel value overstates the voltage. The electrode is where the loss is.

Two cells, two different problems
---------------------------------
* **HT-PEMFC** (PBI, ~450 K). The membrane is doped with phosphoric acid and
  its conductivity depends on humidity but not on a water content that has
  to be transported. So the electrodes are a linear ODE system with an
  analytic solution and there is nothing to iterate.
* **LT-PEMFC** (Nafion, ~350 K) is harder and lives in
  :func:`lt_pemfc_voltage`: Nafion conductivity depends on water content,
  water is dragged across the membrane by the protons *and* diffuses back
  against that drag, and the split is unknown. It needs a root find on the
  water flux ratio.

The electrode ODE
-----------------
Multicomponent (Stefan-Maxwell) diffusion through the cathode gives
``dx/dz = M x + B`` with constant 2x2 ``M``. The reference solves it exactly
by eigendecomposition rather than stepping it, which is worth preserving:
the answer is a matrix exponential, not a quadrature, and no step size
enters.

Verified against TASOPT.jl; see ``tests/test_fuelcell_1d.py``.
"""
from __future__ import annotations

import math

from .fuelcell import (R_GAS, FARADAY, water_sat_pressure, conductivity_pbi,
                       conductivity_nafion, lambda_water, V_HEAT,
                       RHO_MEMBRANE, RHO_ELECTRODE)

__all__ = ["PEMInputs", "binary_diffusion", "porous_diffusion",
           "cathode_j0", "solve_diffusion_ode", "ht_pemfc_voltage",
           "SPECIES", "eig2x2", "lt_pemfc_voltage", "water_balance",
           "nafion_diffusion", "dlambda_dz", "N_DRAG", "cell_voltage",
           "power_density_1d", "pem_size", "pem_operate",
           "pem_stack_weight"]

P0_REF = 101325.0
T0_REF = 298.15
X_ON = 0.21           # oxygen mole fraction in dry air

#: ``(Tc [K], pc [atm], M [g/mol])``. The H2O row also switches the
#: correlation's leading constants -- see :func:`binary_diffusion`.
SPECIES = {
    "H2":  (33.3, 12.8, 2.016),
    "O2":  (154.4, 49.7, 31.999),
    "N2":  (126.2, 33.5, 28.013),
    "H2O": (647.3, 217.5, 18.015),
}


class PEMInputs:
    """Operating point and cell construction."""

    def __init__(self, j=1.0e4, T=353.15, p_A=3.0e5, p_C=3.0e5,
                 x_H2O_A=0.1, x_H2O_C=0.1, lam_H2=3.0, lam_O2=3.0,
                 t_M=100e-6, t_A=250e-6, t_C=250e-6, kind="LT-PEMFC"):
        self.j = j                  # current density, A/m^2
        self.T = T                  # K
        self.p_A, self.p_C = p_A, p_C
        self.x_H2O_A, self.x_H2O_C = x_H2O_A, x_H2O_C
        self.lam_H2, self.lam_O2 = lam_H2, lam_O2   # stoichiometric ratios
        self.t_M, self.t_A, self.t_C = t_M, t_A, t_C
        self.kind = kind


def binary_diffusion(T: float, p: float, sps) -> float:
    """Binary diffusion coefficient, m^2/s -- Slattery and Bird.

    Note the correlation's two leading constants **switch if either species
    is water**: ``(a, b)`` goes from ``(2.745e-4, 1.823)`` to
    ``(3.64e-4, 2.334)``. Water is polar and does not follow the
    non-polar correlation, so this is physics rather than a fudge -- but it
    means a pair containing water is on a different curve entirely, not a
    perturbation of the same one.
    """
    a, b = 2.745e-4, 1.823
    Tc, pc, M = [], [], []
    for sp in sps:
        if sp == "H2O":
            a, b = 3.64e-4, 2.334
        try:
            tc, pcv, mv = SPECIES[sp]
        except KeyError:
            raise ValueError(
                f"no critical properties for {sp!r}; known species are "
                f"{sorted(SPECIES)}") from None
        Tc.append(tc)
        pc.append(pcv)
        M.append(mv)

    TcTc = Tc[0] * Tc[1]
    D = (a * (T / math.sqrt(TcTc)) ** b * (pc[0] * pc[1]) ** (1 / 3)
         * TcTc ** (5 / 12) * math.sqrt(1 / M[0] + 1 / M[1])
         / (p / 101325.0))
    return D * 1.0e-4


def porous_diffusion(D: float, eps: float, tau: float) -> float:
    """Bruggeman-type correction: ``eps**tau * D``.

    Note this is ``eps**tau``, not ``eps/tau``. The simple model in
    :mod:`~tasopt_py.engine_v3.fuelcell` uses the latter under the same
    name; at the defaults (eps = 0.4, tau = 1.5) they differ by a factor of
    four, so they are not interchangeable.
    """
    return eps ** tau * D


def cathode_j0(T: float, p: float, Aeff_ratio: float) -> float:
    """Cathode exchange current density, A/m^2, for ORR on platinum.

    ``Aeff_ratio`` is catalyst area over geometric area -- 1000 by default,
    which is what makes a workable cell out of a reference exchange current
    of 1e-5 A/m^2.
    """
    E_c = 66e3          # J/mol, activation energy for ORR on Pt
    p0 = 101250.0       # note: *not* 101325, unlike everywhere else here
    gamma = 0.75
    return (1.0e-5 * Aeff_ratio * (p / p0) ** gamma
            * math.exp(-E_c / (R_GAS * T) * (1.0 - T / T0_REF)))


def eig2x2(M):
    """Eigenvalues and eigenvectors of a real 2x2 matrix.

    Returned as ``(lam1, lam2, [[v1x, v2x], [v1y, v2y]])``. Only the real
    case is handled, which is all the diffusion system produces: ``M`` comes
    from a Stefan-Maxwell system whose eigenvalues are real and negative
    (species relax toward equilibrium, they do not oscillate).
    """
    a, b = M[0]
    c, d = M[1]
    tr, det = a + d, a * d - b * c
    disc = tr * tr / 4.0 - det
    if disc < 0.0:
        raise ValueError(
            f"complex eigenvalues in the electrode diffusion system "
            f"(discriminant {disc:.6g}); the Stefan-Maxwell matrix should "
            "be real-diagonalisable and is not")
    s = math.sqrt(disc)
    l1, l2 = tr / 2.0 + s, tr / 2.0 - s
    if b != 0.0:
        V = [[b, b], [l1 - a, l2 - a]]
    elif c != 0.0:
        V = [[l1 - d, l2 - d], [c, c]]
    else:
        V = [[1.0, 0.0], [0.0, 1.0]]
    return l1, l2, V


def solve_diffusion_ode(M, B, x0, d):
    """Solve ``dx/dz = M x + B`` from ``z = 0`` to ``z = d``, exactly.

    ``x0`` is the state at ``z = 0``. Diagonalising ``M`` decouples the
    system into two scalar linear ODEs, each with a closed-form solution --
    so this is a matrix exponential, not a quadrature, and no step size or
    tolerance enters the answer.

    Scaling of the eigenvectors cancels, so any valid eigenbasis gives the
    same result; this port's 2x2 closed form need not match the reference's
    LAPACK normalisation.
    """
    l1, l2, V = eig2x2(M)
    if l1 == 0.0 or l2 == 0.0:
        raise ValueError(
            "singular electrode diffusion system: an eigenvalue is zero, so "
            "the particular solution B/lambda is undefined")
    # Invert the 2x2 eigenvector matrix.
    det = V[0][0] * V[1][1] - V[0][1] * V[1][0]
    Vi = [[V[1][1] / det, -V[0][1] / det], [-V[1][0] / det, V[0][0] / det]]

    Bt = [Vi[0][0] * B[0] + Vi[0][1] * B[1],
          Vi[1][0] * B[0] + Vi[1][1] * B[1]]
    z0 = [Vi[0][0] * x0[0] + Vi[0][1] * x0[1],
          Vi[1][0] * x0[0] + Vi[1][1] * x0[1]]
    C = [z0[0] + Bt[0] / l1, z0[1] + Bt[1] / l2]
    ze = [C[0] * math.exp(l1 * d) - Bt[0] / l1,
          C[1] * math.exp(l2 * d) - Bt[1] / l2]
    return [V[0][0] * ze[0] + V[0][1] * ze[1],
            V[1][0] * ze[0] + V[1][1] * ze[1]]


def ht_pemfc_voltage(u: PEMInputs) -> float:
    """Single-cell voltage of a high-temperature (PBI) PEM cell, V.

    The simpler of the two 1-D models: PBI's conductivity depends on
    humidity but not on a transported water content, so nothing iterates.

    Three losses, and only two are charged. The **anode** overvoltage is
    omitted entirely -- hydrogen oxidation on platinum is fast enough that
    the reference neglects it -- so the cell pays a cathode activation loss
    and an ohmic loss and nothing else. There is no concentration loss term
    either; concentration effects enter through the oxygen fraction that the
    electrode ODE delivers to the catalyst.
    """
    n, n_C = 2, 4
    alpha = 0.3          # symmetry parameter
    eps, tau = 0.4, 1.5  # electrode porosity and tortuosity
    Aeff_ratio = 1000.0
    DL = 6.0             # membrane acid doping level

    T, j = u.T, u.j
    Iflux = j / (n * FARADAY)
    p_SAT = water_sat_pressure(T)

    x_H2_in = 1.0 - u.x_H2O_A
    x_O2_in = X_ON * (1.0 - u.x_H2O_C)
    p_O2 = x_O2_in * u.p_C
    p_H2 = x_H2_in * u.p_A

    ds0, E0 = -44.34, 1.1847      # water leaves as vapour at these temperatures
    E_r = (E0 + ds0 / (n * FARADAY) * (T - T0_REF)
           - R_GAS * T / (n * FARADAY)
           * math.log(1.0 / ((p_H2 / P0_REF) * (p_O2 / P0_REF) ** 0.5)))

    # Channel-outlet compositions from the stoichiometric ratios. These are
    # the boundary conditions at the electrode's outer face.
    den_C = u.lam_O2 + (1.0 - u.x_H2O_C) * X_ON
    x_H2O_4 = (u.x_H2O_C * u.lam_O2
               + 2.0 * (1.0 - u.x_H2O_C) * X_ON) / den_C
    x_O2_4 = ((u.lam_O2 - 1.0) * (1.0 - u.x_H2O_C) * X_ON) / den_C
    x_H2O_1 = (u.lam_H2 * u.x_H2O_A) / (u.x_H2O_A + u.lam_H2 - 1.0)

    De = {k: porous_diffusion(binary_diffusion(T, p, sps), eps, tau)
          for k, (p, sps) in {
              "H2H2O": (u.p_A, ["H2", "H2O"]),
              "O2N2": (u.p_C, ["O2", "N2"]),
              "N2H2O": (u.p_C, ["N2", "H2O"]),
              "O2H2O": (u.p_C, ["O2", "H2O"])}.items()}

    k = R_GAS * T * Iflux / u.p_C
    M = [[-(1.0 / De["O2H2O"] - 1.0 / De["O2N2"]) * k,
          -(0.5 / De["O2H2O"] - 1.0 / De["O2N2"]) * k],
         [(-1.0 / De["N2H2O"] + 1.0 / De["O2H2O"]) * k,
          (-1.0 / De["N2H2O"] + 0.5 / De["O2H2O"]) * k]]
    B = [-k / De["O2N2"], k / De["N2H2O"]]

    x_O2_3, x_H2O_3 = solve_diffusion_ode(M, B, [x_O2_4, x_H2O_4], u.t_C)
    x_H2O_2 = x_H2O_1 * math.exp(
        R_GAS * T * Iflux * u.t_A / (u.p_A * De["H2H2O"]))

    RH_avg = 0.5 * (x_H2O_2 * u.p_A / p_SAT + x_H2O_3 * u.p_C / p_SAT)
    sigma = conductivity_pbi(T, DL, RH_avg)
    eta_ohm = j * u.t_M / sigma

    if x_O2_3 < 0.0:
        # Diffusion cannot supply the oxygen this current demands: the
        # catalyst is starved. The reference reports zero volts rather than
        # taking the logarithm of a negative number.
        return 0.0

    j0 = cathode_j0(T, x_O2_3 * u.p_C, Aeff_ratio)
    eta_C = (R_GAS * T / (n_C * alpha * FARADAY)
             * math.log(j / (j0 * u.p_C / P0_REF * x_O2_3)))
    return max(E_r - eta_C - eta_ohm, 0.0)


# --------------------------------------------------------------------------
# Low-temperature (Nafion) cell -- Springer et al. (1991)
# --------------------------------------------------------------------------

#: Electro-osmotic drag coefficient in saturated Nafion: protons carry this
#: many water molecules each. It is why a low-temperature cell dries out at
#: the anode and floods at the cathode under load.
N_DRAG = 2.5
#: Nafion equivalent weight, kg/mol, and dry density, kg/m^3.
M_MEMBRANE = 1.0
RHO_DRY = 1970.0
#: ODE tolerances. The reference integrates at reltol 1e-6; this port
#: integrates far tighter, so any residual disagreement is the reference's
#: tolerance rather than the port's -- see ``DISCREPANCIES.md`` §81.
ODE_RTOL = 1.0e-12
ODE_ATOL = 1.0e-14


def nafion_diffusion(T: float, lam: float) -> float:
    """Water diffusivity in Nafion, m^2/s, from water content.

    A cubic in ``lam`` below 16.8, and above it what the reference calls
    "extrapolate with constant slope".

    It does not. The cubic's derivative at 16.8 is -0.01110912, and the
    source's stored constant ``-1.1109120000000084e-12`` is already that
    number times the ``1e-10`` the cubic branch applies -- then it is
    multiplied by ``1e-10`` a second time. The slope is therefore ten orders
    of magnitude too small and the branch extrapolates with *zero* slope:
    across lam from 16.8 to 1000 the diffusivity moves by 8e-8 percent.

    The port reproduces this exactly. The intended tangent would have fallen
    to zero at lam = 132.8, so the bug is arguably protective -- but
    saturated Nafion sits near lam = 14 to 22 and neither behaviour is
    reached in practice. See ``DISCREPANCIES.md`` §82.
    """
    arr = math.exp(2416.0 * (1.0 / 303.0 - 1.0 / T))
    if lam < 16.8:
        return arr * (2.563 - 0.33 * lam + 0.0264 * lam ** 2
                      - 0.000671 * lam ** 3) * 1e-10
    D0 = arr * 1.2885009279999995e-10
    slope = arr * (-1.1109120000000084e-12) * 1e-10
    return D0 + slope * (lam - 16.8)


def dlambda_dz(lam: float, T: float, Iflux: float,
               alpha_star: float) -> float:
    """Water-content gradient through the membrane, per metre.

    ``(n_drag lam/11 - alpha_star) Iflux M_m / (rho_dry D_lam)``.

    The bracket is the whole physics: electro-osmotic drag pulls water
    toward the cathode in proportion to the local content, and ``alpha_star``
    is the net water flux the cell actually carries. Where the two balance
    the profile is flat; on either side it climbs or falls. So
    ``alpha_star`` is exactly the value that makes the membrane's two ends
    consistent, and that is what the root find below solves for.
    """
    return ((N_DRAG * lam / 11.0 - alpha_star) * Iflux * M_MEMBRANE
            / (RHO_DRY * nafion_diffusion(T, lam)))


def _integrate(f, y0, span, rtol=None, atol=None):
    """Integrate ``dy/dz = f(z, y)`` across ``span``, tightly.

    Tolerances default to the module-level :data:`ODE_RTOL`/:data:`ODE_ATOL`
    at *call* time, not at definition time, so they can be varied to show
    that the answer is converged.
    """
    from scipy.integrate import solve_ivp
    rtol = ODE_RTOL if rtol is None else rtol
    atol = ODE_ATOL if atol is None else atol
    sol = solve_ivp(f, span, y0, method="DOP853", rtol=rtol, atol=atol,
                    dense_output=False)
    if not sol.success:
        raise ValueError(f"membrane ODE integration failed: {sol.message}")
    return [v[-1] for v in sol.y]


def _cathode_state(u: PEMInputs, alpha_star: float, De):
    """``(x_O2_3, x_H2O_3)`` at the cathode catalyst, and the channel-side
    boundary values it came from.

    ``alpha_star`` enters the transport matrix as well as the boundary
    conditions: every water molecule dragged through the membrane has to
    leave through the cathode, so a larger net flux steepens the profile.
    """
    k = R_GAS * u.T * (u.j / (2 * FARADAY)) / u.p_C
    a = 1.0 + alpha_star
    den = u.lam_O2 + (2.0 * alpha_star + 1.0) * (1.0 - u.x_H2O_C) * X_ON
    x_H2O_4 = (u.x_H2O_C * u.lam_O2
               + 2.0 * a * (1.0 - u.x_H2O_C) * X_ON) / den
    x_O2_4 = ((u.lam_O2 - 1.0) * (1.0 - u.x_H2O_C) * X_ON) / den

    M = [[-(a / De["O2H2O"] - 1.0 / De["O2N2"]) * k,
          -(0.5 / De["O2H2O"] - 1.0 / De["O2N2"]) * k],
         [(-a / De["N2H2O"] + a / De["O2H2O"]) * k,
          (-a / De["N2H2O"] + 0.5 / De["O2H2O"]) * k]]
    B = [-k / De["O2N2"], a / De["N2H2O"] * k]
    return solve_diffusion_ode(M, B, [x_O2_4, x_H2O_4], u.t_C)


def _anode_water(u: PEMInputs, alpha_star: float, De) -> tuple:
    """``(x_H2O_1, x_H2O_2)`` -- water at the anode channel and catalyst."""
    a = 1.0 + alpha_star
    num = u.lam_H2 * u.x_H2O_A - alpha_star * (1.0 - u.x_H2O_A)
    x_H2O_1 = num / (u.x_H2O_A - alpha_star * (1.0 - u.x_H2O_A)
                     + u.lam_H2 - 1.0)
    Iflux = u.j / (2 * FARADAY)
    x_H2O_2 = ((x_H2O_1 - alpha_star / a)
               * math.exp(a * R_GAS * u.T * Iflux * u.t_A
                          / (u.p_A * De["H2H2O"])) + alpha_star / a)
    return x_H2O_1, x_H2O_2


def _effective_diffusivities(u: PEMInputs, eps=0.4, tau=1.5) -> dict:
    return {k: porous_diffusion(binary_diffusion(u.T, p, sps), eps, tau)
            for k, (p, sps) in {
                "H2H2O": (u.p_A, ["H2", "H2O"]),
                "O2N2": (u.p_C, ["O2", "N2"]),
                "N2H2O": (u.p_C, ["N2", "H2O"]),
                "O2H2O": (u.p_C, ["O2", "H2O"])}.items()}


def water_balance(alpha_star: float, u: PEMInputs) -> float:
    """Residual on the net water flux ratio.

    The membrane's cathode-side water content can be reached two ways: by
    integrating the drag-diffusion ODE across from the anode, or by reading
    it off the cathode gas humidity. They agree only for the right
    ``alpha_star``, and the difference is the residual.

    This is the whole reason a low-temperature cell is harder than a
    high-temperature one. In PBI there is no water content to transport, so
    there is nothing to solve.
    """
    Iflux = u.j / (2 * FARADAY)
    p_SAT = water_sat_pressure(u.T)
    De = _effective_diffusivities(u)

    _, x_H2O_3 = _cathode_state(u, alpha_star, De)
    lam3_prime = lambda_water(x_H2O_3 * u.p_C / p_SAT)

    _, x_H2O_2 = _anode_water(u, alpha_star, De)
    lam2 = lambda_water(x_H2O_2 * u.p_A / p_SAT)

    lam3 = _integrate(lambda z, y: [dlambda_dz(y[0], u.T, Iflux,
                                               alpha_star)],
                      [lam2], (0.0, u.t_M))[0]
    return lam3 - lam3_prime


def lt_pemfc_voltage(u: PEMInputs, alpha_guess: float = 0.25) -> tuple:
    """``(V, alpha_star)`` for a low-temperature (Nafion) cell.

    Solves the water balance first, then the electrode transport and the
    membrane resistance at that solution. The membrane resistance is itself
    an integral -- ``1/sigma(lambda(z))`` across the thickness -- because
    the water content varies through the membrane and the conductivity
    follows it, so a single conductivity would be wrong.

    Returns zero volts in four unphysical cases, exactly as the reference:
    no net water flux, water fractions above one at either anode station, or
    a starved cathode.
    """
    from scipy.optimize import brentq

    n, n_C = 2, 4
    alpha = 0.3
    Aeff_ratio = 1000.0
    T, j = u.T, u.j
    Iflux = j / (n * FARADAY)
    p_SAT = water_sat_pressure(T)

    # Bracket rather than start from a guess: the residual is monotone in
    # alpha_star over the physical range, so a bracket cannot land on a
    # spurious root the way a secant from 0.25 can.
    lo, hi = 1.0e-8, 1.0
    f_lo = water_balance(lo, u)
    for _ in range(60):
        if f_lo * water_balance(hi, u) < 0.0:
            break
        hi *= 0.5
        if hi <= lo:
            raise ValueError(
                "no bracket for the water balance: the membrane water flux "
                f"residual does not change sign for alpha_star in "
                f"({lo:.3g}, 1.0] at j = {j:.4g} A/m^2, T = {T:.2f} K")
    alpha_star = brentq(lambda x: water_balance(x, u), lo, hi,
                        xtol=1e-14, rtol=8.9e-16)

    x_H2_in = 1.0 - u.x_H2O_A
    x_O2_in = X_ON * (1.0 - u.x_H2O_C)
    p_O2, p_H2 = x_O2_in * u.p_C, x_H2_in * u.p_A

    ds0, E0 = -163.23, 1.229       # liquid water product at these temperatures
    E_r = (E0 + ds0 / (n * FARADAY) * (T - T0_REF)
           - R_GAS * T / (n * FARADAY)
           * math.log(1.0 / ((p_H2 / P0_REF) * (p_O2 / P0_REF) ** 0.5)))

    De = _effective_diffusivities(u)
    x_O2_3, x_H2O_3 = _cathode_state(u, alpha_star, De)
    x_H2O_1, x_H2O_2 = _anode_water(u, alpha_star, De)

    # Water above saturation has condensed; it no longer occupies gas-phase
    # volume, so the remaining fractions are renormalised by (1 - xliq).
    xliq = x_H2O_3 - p_SAT / u.p_C
    lam2 = lambda_water(x_H2O_2 * u.p_A / p_SAT)

    def rhs(z, y):
        return [dlambda_dz(y[0], T, Iflux, alpha_star),
                1.0 / conductivity_nafion(T, y[0])]

    _, ASR = _integrate(rhs, [lam2, 0.0], (0.0, u.t_M))
    eta_ohm = j * ASR

    if alpha_star == 0.0 or x_H2O_1 > 1.0 or x_H2O_2 > 1.0 or x_O2_3 < 0.0:
        return 0.0, alpha_star

    p_eff = u.p_C * x_O2_3 / (1.0 - xliq)
    j0 = cathode_j0(T, p_eff, Aeff_ratio)
    eta_C = (R_GAS * T / (n_C * alpha * FARADAY)
             * math.log(j / (j0 * u.p_C / P0_REF * x_O2_3 / (1.0 - xliq))))
    return max(E_r - eta_C - eta_ohm, 0.0), alpha_star


# --------------------------------------------------------------------------
# Stack sizing on the 1-D voltage models
# --------------------------------------------------------------------------

def cell_voltage(u: PEMInputs) -> float:
    """Voltage of one cell from whichever 1-D model ``u.kind`` selects."""
    if u.kind == "LT-PEMFC":
        return lt_pemfc_voltage(u)[0]
    if u.kind == "HT-PEMFC":
        return ht_pemfc_voltage(u)
    raise ValueError(
        f"unknown cell type {u.kind!r}; expected 'LT-PEMFC' or 'HT-PEMFC'")


def power_density_1d(u: PEMInputs, j: float) -> float:
    """Areal power ``j V(j)``, W/m^2, at current density ``j``.

    The reference's ``P2Acalc`` writes ``j`` onto its input struct as a side
    effect, so the caller's operating point is silently changed. This
    restores it.
    """
    j_was = u.j
    try:
        u.j = j
        return j * cell_voltage(u)
    finally:
        u.j = j_was


def pem_size(P_des: float, V_des: float, u: PEMInputs) -> tuple:
    """``(n_cells, A_cell, Q)`` for a stack at its design point.

    Worth noticing: the cell **area does not depend on the voltage model**.
    ``n_cells = V_des/V_cell`` and ``A_cell = P_des/(n_cells j V_cell)``, so
    the ``V_cell`` cancels and ``A_cell = P_des/(V_des j)`` exactly. Only the
    cell count and the heat load care which model produced the voltage.

    ``Q`` is the heat to reject -- the gap between the thermoneutral voltage
    and the operating one -- and the membrane choice decides how bad it is.
    A high-temperature stack at 0.753 V against a 1.254 V thermoneutral is
    60% efficient and rejects 0.66 W per watt delivered; a low-temperature
    one at 0.710 V against 1.482 V is 48% efficient and rejects 1.09 W per
    watt, more heat than electricity. Either way it is what forces the heat
    exchangers in :mod:`tasopt_py.engine_v3.hx_size`.
    """
    P2A = power_density_1d(u, u.j)
    V_cell = P2A / u.j
    n_cells = V_des / V_cell
    A_cell = P_des / (n_cells * P2A)
    return n_cells, A_cell, n_cells * A_cell * (V_HEAT[u.kind]
                                                - V_cell) * u.j


def pem_operate(P_stack: float, n_cells: float, A_cell: float,
                u: PEMInputs) -> tuple:
    """``(V_stack, Q)`` for a stack off design.

    Solves for the current density that delivers ``P_stack``. **Two roots**
    -- areal power peaks and falls toward the limiting current -- and the
    physical one is the smaller. The reference starts a secant from
    ``j = 1`` with a comment warning about it; this brackets between 1 and
    the design current, which cannot land on the wrong side.
    """
    from scipy.optimize import brentq

    P2A = P_stack / (n_cells * A_cell)

    def f(j):
        return P2A - power_density_1d(u, j)

    lo, hi = 1.0, u.j
    while f(lo) * f(hi) > 0.0 and hi < 1.0e6:
        hi *= 1.5
    if f(lo) * f(hi) > 0.0:
        raise ValueError(
            f"no current density between 1 and {hi:.4g} A/m^2 delivers "
            f"{P2A:.4g} W/m^2; the cell's areal power peaks below it")
    j = brentq(f, lo, hi, xtol=1e-12, rtol=8.9e-16)

    V_cell = P2A / j
    return V_cell * n_cells, n_cells * A_cell * (V_HEAT[u.kind]
                                                 - V_cell) * j


def pem_stack_weight(gee: float, u: PEMInputs, n_cells: float,
                     A_cell: float, fouter: float) -> float:
    """Stack weight, N. Membrane plus both electrodes, times ``1 + fouter``.

    ``fouter`` carries the bipolar plates and structure, and at the
    reference's own value of 4.0 it is **four fifths of the stack** -- so
    the active layers this whole module models are a fifth of what flies.
    """
    return (gee * n_cells * A_cell
            * ((u.t_A + u.t_C) * RHO_ELECTRODE
               + u.t_M * RHO_MEMBRANE[u.kind]) * (1.0 + fouter))
