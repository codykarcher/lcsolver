"""Transport properties -- ``gasPr`` in v3's ``gascalc.jl``.

TASOPT 2.16 needs no viscosity or conductivity: its engine is an ideal
cycle with prescribed efficiencies, and its only Reynolds numbers come from
the standard atmosphere. v3 adds heat exchangers and cryogenic tanks, which
need both.

Each gas carries a **Sutherland pair** -- a reference viscosity and
conductivity with their own reference temperatures and constants -- while
``cp`` and ``R`` come from the same polynomial tables the 2.16 cycle uses.
So this is a thin layer over :mod:`tasopt_py.gas.properties`, not a separate
thermodynamic model.

A wrinkle worth knowing: ``"air"`` and ``"air_simple"`` share Sutherland
constants but not ``cp``. ``"air"`` sums the five-species mixture at
temperature; ``"air_simple"`` returns a flat 1005 J/(kg K) and 287.1
J/(kg K). The reference uses the simple one wherever a Prandtl number is
wanted inside an iteration, which is a real 3-4% difference in ``cp`` at
800 K -- see the note in :func:`gas_Pr`.
"""
from __future__ import annotations

import math

from .properties import gasfun
from .mixture import gassum

__all__ = ["gas_Pr", "SUTHERLAND", "AIR_ALPHA"]

#: ``(mu0, S_mu, K0, S_k, T0, igas)``. ``igas`` is ``None`` for the two air
#: entries, which are handled separately.
SUTHERLAND = {
    "air":        (1.716e-5, 111.0, 0.0241, 194.0, 273.0, None),
    "air_simple": (1.716e-5, 111.0, 0.0241, 194.0, 273.0, None),
    "co2":        (1.370e-5, 222.0, 0.0146, 1800.0, 273.0, 3),
    "n2":         (1.663e-5, 107.0, 0.0242, 150.0, 273.0, 1),
    "o2":         (1.919e-5, 139.0, 0.0244, 240.0, 273.0, 2),
    "h2o":        (1.147e-5, 1010.0, 0.02133, 8331.0, 350.0, 4),
    "ch4":        (1.024e-5, 179.1, 0.02977, 2859.0, 273.0, 11),
    "h2":         (8.485e-6, 99.95, 0.1701, 161.1, 273.0, 40),
}

#: Dry air by mass fraction -- N2, O2, CO2, H2O, Ar.
AIR_ALPHA = [0.7532, 0.2315, 0.0006, 0.0020, 0.0127]


def gas_Pr(gas: str, T: float) -> tuple:
    """``(R, Pr, gamma, cp, mu, k)`` for ``gas`` at ``T`` kelvin.

    ``mu`` and ``k`` are Sutherland; ``cp`` and ``R`` come from the species
    tables, except for ``"air_simple"`` which uses constants.

    The Prandtl number is therefore *not* a fitted quantity -- it is
    ``cp mu / k`` from three independently sourced pieces, which is why it
    drifts with temperature rather than sitting at a textbook 0.71.
    """
    try:
        mu0, S_mu, K0, S_k, T0, igas = SUTHERLAND[gas]
    except KeyError:
        raise ValueError(
            f"no transport properties for {gas!r}; known gases are "
            f"{sorted(SUTHERLAND)}") from None

    if gas == "air_simple":
        R, cp = 287.1, 1005.0
    elif gas == "air":
        st = gassum(AIR_ALPHA, 5, T)
        R, cp = st.r, st.cp
    else:
        st = gasfun(igas, T)
        R, cp = st.r, st.cp

    mu = mu0 * (T / T0) ** 1.5 * ((T0 + S_mu) / (T + S_mu))
    k = K0 * (T / T0) ** 1.5 * ((T0 + S_k) / (T + S_k))
    return R, cp * mu / k, cp / (cp - R), cp, mu, k
