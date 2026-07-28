"""Combustion: fuel-air ratio and burnt-gas properties — a port of ``gasburn.f``.

Given the air temperature before the burner and the required temperature
after it, this solves for the fuel mass fraction that gets you there, and
returns the properties of the resulting gas mixture. It is the step that sits
between the compressor and the turbine in a cycle calculation, and a
prerequisite for ``tfsize``/``tfoper``.

Naming: the subroutine inside ``gasburn.f`` is called ``gasprop``, which is
also the name of a *different* file (``gasprop.f``). The one ported here is
the combustion routine.

Species and the reaction
------------------------
Five species are tracked, in this fixed order: N2, O2, CO2, H2O, fuel. The
fuel is described only by its atom counts, so any hydrocarbon works —
``nh=4, nc=1`` is methane, ``nh=8, nc=3`` propane, ``nh=2, nc=1`` a heavy
hydrocarbon. Only the *relative* numbers matter.

The stoichiometry is balanced from those counts into mass fractions per unit
fuel mass (``delta``), negative on the reactant side and positive on the
product side, and combined with the "fuel is consumed" vector ``beta`` to give
``gamma = beta + delta``, the net change in composition per unit fuel burnt.

The energy balance
------------------
Enthalpy is conserved across the burner, which gives the fuel fraction
directly:

    f = alpha.cp_cold (T4 - T3)
        / ( [beta.cp_cold (Tf - Ts) + beta.hs] - [gamma.cp_hot (T4 - Ts)
                                                  + gamma.hs] )

with ``Ts = 298 K`` the reference temperature for the enthalpies of
formation. Note the asymmetry, which is deliberate rather than an oversight:
the *cold* specific heats are used for the incoming air and fuel and the *hot*
ones for the products, since the two sit at very different temperatures.

Verified against the compiled Fortran; see ``tests/test_burn.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BurnResult", "SPECIES", "gasburn"]

# Fixed species order.
SPECIES = ("N2", "O2", "CO2", "H2O", "fuel")

# Gas constants, J/kg/K. The fuel entry is overwritten by the caller's value.
_R = [296.94, 259.82, 188.96, 461.91, 519.65]
# Specific heats at the cold end (reactants) and the hot end (products).
_CP_COLD = [1041.0, 920.0, 851.0, 1864.0, 2240.0]
_CP_HOT = [1219.0, 1125.0, 1300.0, 2524.0, 0.0]
# Enthalpies of formation, J/kg. Note the commented-out alternative in the
# source gives the fuel -5.681e6 rather than -4.675e6; the active value is
# used here.
_HS = [0.0, 0.0, -8.949e6, -13.444e6, -4.675e6]
# "One unit of fuel is consumed" — beta is nonzero only in the fuel slot.
_BETA = [0.0, 0.0, 0.0, 0.0, 1.0]

_TS = 298.0          # reference temperature for the enthalpies of formation

# Molar weights of H, C, N, O.
_WH, _WC, _WN, _WO = 1.00795, 12.01078, 14.00672, 15.99943


@dataclass(frozen=True)
class BurnResult:
    f: float          # fuel/air mass fraction
    cp3: float        # specific heat of the air before combustion
    gam3: float       # ratio of specific heats before combustion
    cp4: float        # specific heat of the combustion gas
    gam4: float       # ratio of specific heats after combustion
    alpha: list       # mass fractions after combustion, in SPECIES order


def _dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


def gasburn(ttf: float, tt3: float, tt4: float,
            rfuel: float, cpfuel: float, hsfuel: float,
            nhfuel: int, ncfuel: int, nnfuel: int, nofuel: int,
            alpha) -> BurnResult:
    """Fuel fraction and burnt-gas properties.

    Parameters
    ----------
    ttf, tt3, tt4
        Total temperature of the fuel, of the air before combustion, and the
        required temperature of the mixture after it. ``ttf`` has only a weak
        effect.
    rfuel, cpfuel, hsfuel
        Gas constant, specific heat and enthalpy of formation of the fuel
        vapour. ``hsfuel`` is negative.
    nhfuel, ncfuel, nnfuel, nofuel
        Atom counts in the fuel molecule; only their ratios matter.
    alpha
        Mass fractions of the incoming mixture, in ``SPECIES`` order.

    The Fortran mutates ``alpha`` in place; here the updated composition comes
    back on the result and the argument is left alone.
    """
    r = list(_R)
    cp_cold = list(_CP_COLD)
    hs = list(_HS)

    # Molar weight of the fuel from its atom counts.
    wfuel = (_WH * nhfuel + _WC * ncfuel + _WN * nnfuel + _WO * nofuel)

    # Molar weights of the products, balancing the reaction
    #     fuel + O2  ->  N2 + CO2 + H2O
    wn2 = (_WN * 2.0) * nnfuel * 0.5
    wco2 = (_WC + _WO * 2.0) * ncfuel
    wh2o = (_WO + _WH * 2.0) * nhfuel * 0.5
    wo2 = (_WO * 2.0) * (ncfuel + nhfuel * 0.25 - nofuel * 0.5)

    # Mass of each species per unit mass of fuel: negative for what is
    # consumed, positive for what is produced.
    delta = [wn2 / wfuel, -wo2 / wfuel, wco2 / wfuel, wh2o / wfuel, -1.0]

    # The caller's fuel properties go in the fuel slot.
    r[4] = rfuel
    cp_cold[4] = cpfuel
    hs[4] = hsfuel

    gamma = [b + d for b, d in zip(_BETA, delta)]

    r3 = _dot(alpha, r)
    cp3 = _dot(alpha, cp_cold)
    gam3 = cp3 / (cp3 - r3)

    # Enthalpy balance across the burner. Cold specific heats for the
    # reactants, hot ones for the products.
    dota = _dot(alpha, cp_cold) * (tt4 - tt3)
    dotb = _dot(_BETA, cp_cold) * (ttf - _TS) + _dot(_BETA, hs)
    dotc = _dot(gamma, _CP_HOT) * (tt4 - _TS) + _dot(gamma, hs)

    f = dota / (dotb - dotc)

    lam = [(a + f * g) / (1.0 + f) for a, g in zip(alpha, gamma)]

    cp4 = _dot(lam, _CP_HOT)
    r4 = _dot(lam, r)
    gam4 = cp4 / (cp4 - r4)

    return BurnResult(f=f, cp3=cp3, gam3=gam3, cp4=cp4, gam4=gam4, alpha=lam)
