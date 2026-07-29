"""Inverter and cable models -- ``inverter.jl`` and ``cable.jl``.

The two pieces of an electric drivetrain that sit between a power source and
a motor. Nothing in TASOPT 2.16 corresponds to either: the Fortran has no
electrical system at all, only shaft power offtakes (``PofWpay``,
``PofWMTO``) that vanish from the cycle without going anywhere.

The inverter
------------
A three-point efficiency curve fitted to a quadratic-in-``1/x`` form, from
Faranda et al. (2015). Efficiency is quoted at 10%, 20% and 100% of design
power and a 3x3 linear system is solved for the coefficients of

    eta = k1 + k2 (P/Pdes) + k3 (P/Pdes)^-1

so it falls off at *both* ends -- the ``1/x`` term is what makes a lightly
loaded inverter inefficient, which a plain polynomial would miss. Mass is a
flat specific power, 19 kW/kg, from GE and Illinois SiC/GaN devices.

The switching frequency is the electrical frequency times a carrier ratio
``kcf`` of 20, and the peak efficiency falls linearly with it -- 2.5e-7 per
Hz, so a 1 kHz fundamental costs half a point of efficiency.

The cable
---------
Sized from the current it must carry and the voltage it must insulate. The
conductor area follows from a current-density limit, the insulation
thickness from a dielectric-strength limit, and the mass from both. It
returns an efficiency *function* of input power, because the loss is
ohmic and therefore quadratic -- so a cable sized at one power has a
different efficiency at another.

Verified against TASOPT.jl; see ``tests/test_electric.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..cryo.material_data import MATERIALS

__all__ = ["Inverter", "size_inverter", "operate_inverter",
           "Cable", "size_cable", "SizedCable", "GEE"]

GEE = 9.81


# --------------------------------------------------------------------------
# Inverter
# --------------------------------------------------------------------------

@dataclass
class Inverter:
    """A DC-AC inverter, or a rectifier run backwards."""
    P_design: float = 1.0e3
    P: float = 1.0e3
    P_input: float = 1.0e3
    mass: float = 0.0
    #: 19 kW/kg, from GE and Illinois SiC/GaN demonstrators.
    specific_power: float = 19.0e3
    #: Carrier frequency ratio -- switching frequency over electrical.
    kcf: float = 20.0
    eta: float = 0.0


def _efficiency_coefficients(f_switch: float) -> tuple:
    """``(k1, k2, k3)`` for ``eta = k1 + k2 x + k3 / x`` at ``x = P/Pdes``.

    Fitted through the efficiency at 10%, 20% and 100% of design power.
    Solved in closed form rather than by a 3x3 elimination -- the system is
    the same every time, so the inverse is a constant.
    """
    eta100 = -2.5e-7 * f_switch + 0.995
    eta20 = eta100 - 0.0017
    eta10 = eta100 - 0.0097

    # [1 0.1 10; 1 0.2 5; 1 1 1] x = [eta10; eta20; eta100]
    a = [[1.0, 0.1, 10.0], [1.0, 0.2, 5.0], [1.0, 1.0, 1.0]]
    b = [eta10, eta20, eta100]
    # Gaussian elimination with partial pivoting; the matrix is 3x3 and
    # well-conditioned, so this is exact enough to match the reference.
    M = [row[:] + [b[i]] for i, row in enumerate(a)]
    n = 3
    for k in range(n):
        piv = max(range(k, n), key=lambda i: abs(M[i][k]))
        M[k], M[piv] = M[piv], M[k]
        for i in range(k + 1, n):
            fac = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= fac * M[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j]
                              for j in range(i + 1, n))) / M[i][i]
    return tuple(x)


def operate_inverter(inv: Inverter, P: float, f: float) -> float:
    """Run an inverter off design. Returns the efficiency.

    ``f`` is the electrical frequency; the switching frequency is ``f *
    kcf``. Updates ``P``, ``P_input`` and ``eta`` on the object.
    """
    if inv.P_design <= 0.0:
        raise ValueError("inverter has no design power; size it first")
    k1, k2, k3 = _efficiency_coefficients(f * inv.kcf)
    x = P / inv.P_design
    if x <= 0.0:
        raise ValueError(
            f"inverter load fraction must be positive, got {x}; the "
            "efficiency fit has a 1/x term and diverges at zero")
    inv.P = P
    inv.eta = k1 + k2 * x + k3 / x
    inv.P_input = P / inv.eta
    return inv.eta


def size_inverter(inv: Inverter, P_design: float, f_design: float) -> None:
    """Size an inverter at its design point."""
    inv.P_design = P_design
    inv.P = P_design
    operate_inverter(inv, P_design, f_design)
    inv.mass = P_design / inv.specific_power


# --------------------------------------------------------------------------
# Cable
# --------------------------------------------------------------------------

@dataclass
class Cable:
    """A power cable's design limits and materials.

    Defaults are the reference's: a stranded copper conductor in polyamide,
    with the properties coming from the same material database the cryogenic
    tank uses.
    """
    conductor: str = "Cu"
    insulator: str = "polyamide"
    #: Conductor current density limit, A/m^2. 5 A/mm^2.
    Jmax: float = 5.0e6
    #: Packing factor -- conductor area over the circle that holds it.
    kpf: float = 0.91
    #: Conductor temperature, K. Defaults to 50 C, not room temperature.
    Tcon: float = 273.1 + 50.0

    @property
    def _cond(self):
        return MATERIALS[self.conductor]

    @property
    def _ins(self):
        return MATERIALS[self.insulator]

    @property
    def Emax(self) -> float:
        """Insulator dielectric strength, V/m."""
        return self._ins["dielectric_strength"]


@dataclass(frozen=True)
class SizedCable:
    mass: float          # kg
    W: float             # N
    R: float             # ohm
    V: float             # V
    length: float        # m
    A_con: float         # conductor area, m^2
    t_ins: float         # insulation thickness, m
    P_design: float      # W

    def efficiency(self, Pin: float) -> float:
        """Efficiency at input power ``Pin``, at the design voltage.

        The loss is ohmic, so it grows as the *square* of the current --
        which means a cable is most efficient near its design point and
        loses more, proportionally, above it.
        """
        Iin = Pin / self.V
        return (Pin - Iin ** 2 * self.R) / Pin

    def exceeds_design(self, Pin: float) -> bool:
        """Whether ``Pin`` puts the conductor past its current-density
        limit. The reference warns and carries on; this lets a caller ask."""
        return Pin > self.P_design


def resistivity(cable: Cable, T: float) -> float:
    """Conductor resistivity at temperature ``T``, ohm m."""
    c = cable._cond
    return c["resistivity"] * (1.0 + c["alpha"] * (T - c["T0"]))


def size_cable(cable: Cable, P: float, V: float, length: float) -> SizedCable:
    """Size a cable for ``P`` watts at ``V`` volts over ``length`` metres.

    Two independent limits set the geometry: the conductor area comes from
    the current-density limit, and the insulation thickness from the
    dielectric strength. So raising the voltage *shrinks* the conductor (less
    current for the same power) while *thickening* the insulation -- which is
    why there is an optimum voltage for cable mass rather than a monotone
    trend.
    """
    if V <= 0.0:
        raise ValueError(f"cable voltage must be positive, got {V}")
    I = P / V
    rho_con = resistivity(cable, cable.Tcon)
    t_ins = V / cable.Emax
    A_con = I / cable.Jmax
    ri = math.sqrt(A_con / (math.pi * cable.kpf))
    ro = ri + t_ins
    A_ins = math.pi * (ro ** 2 - ri ** 2)

    mass = length * (A_con * cable._cond["rho"] + A_ins * cable._ins["rho"])
    return SizedCable(mass=mass, W=mass * GEE,
                      R=rho_con * length / A_con, V=V, length=length,
                      A_con=A_con, t_ins=t_ins, P_design=P)
