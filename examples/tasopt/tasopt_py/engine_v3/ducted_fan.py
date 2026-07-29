"""Ducted fan weight -- ``ductedfanweight.jl``.

An electrically driven ducted fan: a fan and a nacelle, with no core. It is
what a fuel cell or a battery drives, and it has no counterpart in TASOPT
2.16, whose only engine is a turbofan.

The fan blade mass is a correlation in diameter, blade aspect ratio, solidity
and tip speed, scaled by a flat ``ktech = 0.5`` "technology factor" -- so
half the weight is an assumption rather than a calculation. The nacelle is
built from three areas (inlet, cowl, exhaust) at 40/20/40% of the wetted
area, each with its own areal density correlation written in pounds and feet.

The nacelle is counted about 1.9 times
--------------------------------------
Worth knowing before using any number from this. The reference computes::

    Wfan = (mfan*9.81 + Wnace*0.8)*(1 + fpylon)
    Weng = (Wfan + Wnace) * neng

so ``Wnace`` enters ``Weng`` once inside ``Wfan``, at ``0.8 (1 + fpylon)``,
and once again on its own -- a total of ``1 + 0.8(1 + fpylon)`` nacelles per
engine, or 1.88 at the default ``fpylon = 0.1``.

TASOPT 2.16's ``tfweight.f`` does not do this. There the nacelle appears once::

    Weng1 = (Wcore + Wfan + Wcomb + Wnace + Wnozz)/lb_N
    Weng  = (Weng1 + Wpylon)*neng

That is evidence, not proof -- ``Wfan`` here may be intended as "fan module
including its share of nacelle structure" -- so this port reproduces the
arithmetic exactly and pins it rather than correcting it. See
``DISCREPANCIES.md`` §70.

``Webare``, the "bare engine" weight, has the same issue from the other side:
it is ``Wfan * neng``, which *excludes* the nacelle proper but *includes*
0.8 of it. So it is neither bare nor consistent with ``Weng - Wnac``.

Verified against TASOPT.jl; see ``tests/test_ducted_fan.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["ducted_fan_weight", "DuctedFanWeight", "AR_FAN",
           "BLADE_SOLIDITY", "KTECH"]

GEE = 9.81

#: Fixed in the source, not inputs.
AR_FAN = 3.0             # blade aspect ratio
BLADE_SOLIDITY = 0.4     # chord over spacing
KTECH = 0.5              # flat technology factor on blade mass


@dataclass(frozen=True)
class DuctedFanWeight:
    Weng: float      # all engines, N -- includes ~1.9 nacelles each
    Wnac: float      # all nacelles, N
    Webare: float    # "bare" engines, N -- see the module docstring
    W_HX: float      # heat exchangers, N
    Snace1: float    # one nacelle's wetted area, m^2


def ducted_fan_weight(Dfan: float, Nmech: float, neng: float,
                      rSnace: float, fpylon: float,
                      W_HX: float = 0.0) -> DuctedFanWeight:
    """Weights for a set of electrically driven ducted fans.

    ``Dfan`` fan diameter (m), ``Nmech`` shaft speed (rpm), ``neng`` engine
    count, ``rSnace`` nacelle wetted area over fan area, ``fpylon`` pylon
    weight fraction. ``W_HX`` is any heat-exchanger weight, per engine set;
    the reference sums it from the engine's heat exchangers, which this port
    takes as an argument until :mod:`hxfun` is ported.
    """
    Utip = Dfan / 2.0 * (2.0 * math.pi * Nmech / 60.0)
    mfan = KTECH * (135.0 * Dfan ** 2.7 / math.sqrt(AR_FAN)
                    * (BLADE_SOLIDITY / 1.25) ** 0.3
                    * (Utip / 350.0) ** 0.3)

    Snace1 = rSnace * 0.25 * math.pi * Dfan ** 2
    Ainlet = 0.4 * Snace1
    Acowl = 0.2 * Snace1
    Aexh = 0.4 * Snace1

    # Written in pounds per square foot against a diameter in inches, then
    # converted -- which is why every term carries a unit factor.
    Wnace = (4.45 * (Ainlet / 0.3048 ** 2.0) * (2.5 + 0.0238 * Dfan / 0.0254)
             + 4.45 * (Acowl / 0.3048 ** 2.0) * 1.9
             + 4.45 * (Aexh / 0.3048 ** 2.0)
             * (2.5 + 0.0363 * Dfan / 0.0254))

    Wfan = (mfan * GEE + Wnace * 0.8) * (1.0 + fpylon)
    Weng = (Wfan + Wnace) * neng
    Webare = Wfan * neng
    Wnac = Wnace * neng

    return DuctedFanWeight(Weng=Weng + W_HX, Wnac=Wnac,
                           Webare=Webare + W_HX, W_HX=W_HX, Snace1=Snace1)
