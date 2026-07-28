"""Compressor and fan efficiency map — a port of ``ecmap`` from ``tfmap.f``.

Efficiency as a function of pressure ratio and corrected mass flow, relative
to the design point. ``tfsize`` calls it three times, once each for the fan,
the low-pressure compressor and the high-pressure compressor.

The shape
---------
Two normalised coordinates:

    p = (pi - 1) / (piD - 1)        pressure ratio, relative to design
    m = mb / mbD                    corrected mass flow, relative to design

and efficiency falls away from the design point along two independent
penalties:

    e1 = 1 - CK * |p/m^(a+da-1) - m|^c        off the working line
           - DK * |m/mo - 1|^d                 off the design mass flow

plus a linear trend in pressure ratio, ``effK*(pi - piK)``, which lets a
family of engines share one map shape while their peak efficiency drifts with
size.

The ``sign()`` factors
----------------------
``psgn`` and ``msgn`` carry the sign of each bracket *outward* so that the
fractional powers ``c`` and ``d`` act on a positive quantity. Without them
``(negative)**3.0`` would be fine but ``(negative)**2.5`` would not, and the
map constants make ``c`` and ``d`` freely choosable. The effect is an odd
symmetric penalty: falling below the design mass flow is penalised the same
way as exceeding it.

The penalties are switched off in TASOPT 2.16
---------------------------------------------
``tfmap.inc`` carries four generations of map constants, three commented out.
In the **active** set ``CK = DK = 0``, which kills both penalty terms
outright: ``e1`` is identically 1 and the efficiency collapses to

    eff = effo + effK * (pi - piK)

a straight line in pressure ratio, with no mass-flow dependence at all. The
elaborate map shape above is dead in the shipped configuration. That matters
for anyone reasoning about off-design behaviour from this routine, and it is
not visible without opening the include file.

The historical sets with nonzero ``CK``/``DK`` are kept here as
``*_PENALISED`` so the full expression can still be exercised, and the tests
use them for that.

A note on ``eo``
----------------
The source computes ``eo`` as a normalising factor and then immediately
overrides it with ``eo = 1.0``, leaving the original expression commented out
one line above. So the division by ``eo`` is a no-op. Kept here for symmetry
with the source rather than silently folded away.

Verified against the compiled Fortran; see ``tests/test_maps.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["MapEfficiency", "CMAPF", "CMAPLC", "CMAPHC",
           "CMAPF_PENALISED", "CMAPLC_PENALISED", "CMAPHC_PENALISED",
           "ecmap"]

# Map constants, in the order the routine unpacks them:
#     a, b, k, mo, da, c, d, CK, DK
#
# These are the ACTIVE values from tfmap.inc -- the ones not commented out.
# Note CK = DK = 0 in all three: see the module docstring, this switches the
# entire off-design penalty off.
CMAPF = (3.50, 0.80, 0.03, 0.95, -0.50, 3.0, 6.0, 0.0, 0.0)
CMAPLC = (1.90, 1.00, 0.03, 0.95, -0.20, 3.0, 5.5, 0.0, 0.0)
CMAPHC = (1.75, 2.00, 0.03, 0.95, -0.35, 3.0, 5.0, 0.0, 0.0)

# Historical sets, kept commented out in tfmap.inc above the active ones.
# These DO have nonzero CK/DK, so they exercise the penalty terms; the tests
# use them for exactly that reason.
CMAPF_PENALISED = (3.50, 0.80, 0.03, 0.75, -0.50, 3.0, 6.0, 2.5, 15.0)
CMAPLC_PENALISED = (2.50, 1.00, 0.03, 0.75, -0.20, 3.0, 5.5, 4.0, 6.0)
CMAPHC_PENALISED = (1.75, 2.00, 0.03, 0.75, -0.35, 3.0, 5.0, 10.5, 3.0)


@dataclass(frozen=True)
class MapEfficiency:
    eff: float       # efficiency
    eff_pi: float    # d(eff)/d(pressure ratio)
    eff_mb: float    # d(eff)/d(corrected mass flow)


def ecmap(pi: float, mb: float, piD: float, mbD: float, Cmap,
          effo: float, piK: float, effK: float) -> MapEfficiency:
    """Compressor or fan efficiency at ``(pi, mb)``.

    ``Cmap`` is the nine-element constant set; ``effo`` the peak efficiency;
    ``piK``/``effK`` the offset and slope of the linear pressure-ratio trend.
    Derivatives come back alongside, as the Fortran returns them.
    """
    a, b, k, mo, da, c, d, CK, DK = Cmap
    adm = a + da - 1.0

    p = (pi - 1.0) / (piD - 1.0)
    p_pi = 1.0 / (piD - 1.0)

    m = mb / mbD
    m_mb = 1.0 / mbD

    # Signs pulled out so the fractional powers act on positive quantities.
    psgn = math.copysign(1.0, p / m ** adm - m)
    msgn = math.copysign(1.0, m / mo - 1.0)

    eo = 1.0        # the source overrides its own expression with this

    e1 = (1.0 - CK * (psgn * (p / m ** adm - m)) ** c
          - DK * (msgn * (m / mo - 1.0)) ** d)
    e1_p = (-CK * (psgn * (p / m ** adm - m)) ** (c - 1.0)
            * c * psgn / m ** adm)
    e1_m = (-CK * (psgn * (p / m ** adm - m)) ** (c - 1.0)
            * c * psgn * (-adm * p / m ** adm / m - 1.0)
            - DK * (msgn * (m / mo - 1.0)) ** (d - 1.0)
            * d * msgn / mo)

    eff = effo * e1 / eo + effK * (pi - piK)
    eff_p = effo * e1_p / eo
    eff_m = effo * e1_m / eo

    return MapEfficiency(eff=eff,
                         eff_pi=eff_p * p_pi + effK,
                         eff_mb=eff_m * m_mb)
