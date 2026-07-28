"""Turbine blade cooling flow — a port of ``tfcool.f``.

Two routines, inverse to each other:

``mcool``
    Given the metal temperature each blade row can tolerate, how much cooling
    air does each row need?
``tmcalc``
    Given the cooling air actually supplied, how hot does the metal get?

Together they are what sets the cooling bleed in a cycle calculation, and
hence the ``alpha_c`` that the signomial engine model carries as a constant.

The model
---------
For each row, a cooling effectiveness

    theta = (Tg - Tmetal) / (Tg - Tt3)

is the fraction of the gas-to-coolant temperature difference the blade has to
survive. The required coolant fraction follows from a Stanton-number heat
balance,

    eps0 = StA * (theta (1 - efilm*tfilm) - tfilm (1 - efilm))
           / (efilm (1 - theta))

and the mass flow ratio is ``eps0 / (1 + eps0)`` — the difference between the
two being whether the coolant is counted in the flow it is added to.

Marching downstream
-------------------
Gas temperature falls through the turbine as ``Tt4 * Trrat^(i-1)``, except at
the *first* row, which instead sees ``Tt4 + dTstreak`` — the hot streak coming
straight off the burner, which is hotter than the mean. Once ``eps0`` goes
negative a row no longer needs cooling, and the march stops there: ``ncrow``
is the number of cooled rows, and the remaining entries stay zero.

Note ``tmcalc`` computes ``theta`` a second time at the end of its loop, from
the metal temperature it just set. It is dead — the value is never used and
the loop then moves on. Reproduced only in the sense that it is harmless.

Verified against the compiled Fortran; see ``tests/test_cooling.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CoolingFlow", "mcool", "tmcalc"]


@dataclass(frozen=True)
class CoolingFlow:
    ncrow: int        # number of blade rows that need cooling
    epsrow: list      # cooling mass flow ratio per row, m_c_row / m_air
    epsrow_Tt3: list  # d(epsrow)/d(Tt3)
    epsrow_Tt4: list  # d(epsrow)/d(Tt4)
    epsrow_Trr: list  # d(epsrow)/d(Trrat)


def mcool(Tmrow, Tt3: float, Tt4: float, dTstreak: float, Trrat: float,
          efilm: float, tfilm: float, StA: float) -> CoolingFlow:
    """Cooling mass flow required by each blade row.

    ``Tmrow`` is the design metal temperature of each row; its length sets the
    maximum number of rows considered. Sensitivities to ``Tt3``, ``Tt4`` and
    ``Trrat`` come back alongside, as the Fortran returns them.
    """
    ncrowx = len(Tmrow)
    epsrow = [0.0] * ncrowx
    epsrow_Tt3 = [0.0] * ncrowx
    epsrow_Tt4 = [0.0] * ncrowx
    epsrow_Trr = [0.0] * ncrowx

    ncrow = 0
    for i in range(ncrowx):
        if i == 0:
            # The first row sees the hot streak off the burner, not the mean.
            Tg = Tt4 + dTstreak
            Tg_Tt4 = 1.0
            Tg_Trr = 0.0
        else:
            Tg = Tt4 * Trrat ** i
            Tg_Tt4 = Trrat ** i
            Tg_Trr = i * Tg / Trrat

        theta = (Tg - Tmrow[i]) / (Tg - Tt3)
        theta_Tt3 = theta / (Tg - Tt3)
        theta_Tt4 = (1.0 - theta) / (Tg - Tt3) * Tg_Tt4
        theta_Trr = (1.0 - theta) / (Tg - Tt3) * Tg_Trr

        eps0 = (StA * (theta * (1.0 - efilm * tfilm) - tfilm * (1.0 - efilm))
                / (efilm * (1.0 - theta)))
        eps0_theta = (StA * (1.0 - efilm * tfilm) / (efilm * (1.0 - theta))
                      + eps0 / (1.0 - theta))

        # A negative requirement means this row runs cool enough uncooled, and
        # so does everything downstream of it: stop marching.
        if eps0 < 0.0:
            break

        ncrow = i + 1
        epsrow[i] = eps0 / (1.0 + eps0)
        epsrow_eps0 = (1.0 - epsrow[i]) / (1.0 + eps0)
        epsrow_Tt3[i] = epsrow_eps0 * eps0_theta * theta_Tt3
        epsrow_Tt4[i] = epsrow_eps0 * eps0_theta * theta_Tt4
        epsrow_Trr[i] = epsrow_eps0 * eps0_theta * theta_Trr

    return CoolingFlow(ncrow=ncrow, epsrow=epsrow, epsrow_Tt3=epsrow_Tt3,
                       epsrow_Tt4=epsrow_Tt4, epsrow_Trr=epsrow_Trr)


def tmcalc(ncrow: int, epsrow, Tt3: float, Tt4: float, dTstreak: float,
           Trrat: float, efilm: float, tfilm: float, StA: float) -> list:
    """Metal temperature reached by each cooled row, given the coolant supplied.

    The inverse of :func:`mcool`. Only the first ``ncrow`` rows are computed;
    the rest come back as zero, matching the Fortran, which leaves them
    untouched.
    """
    Tmrow = [0.0] * len(epsrow)
    for i in range(ncrow):
        Tg = (Tt4 + dTstreak) if i == 0 else Tt4 * Trrat ** i
        eps = epsrow[i]
        theta = ((eps * efilm + StA * tfilm * (1.0 - efilm) * (1.0 - eps))
                 / (eps * efilm + StA * (1.0 - efilm * tfilm) * (1.0 - eps)))
        Tmrow[i] = Tg - (Tg - Tt3) * theta
    return Tmrow
