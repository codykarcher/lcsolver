"""Engine and nacelle weight -- a port of ``tfweight`` from ``tfweight.f``.

Three weight models live behind one ``iengwgt`` switch:

=========  =========================================================
0          Drela's original: linear in OPR, a 1.2 power in BPR
1, 2       Fitzgerald's: ``a(BPR) * mdotc^b(BPR) * (OPR/40)^c(BPR)``,
           basic and advanced technology, geared and ungeared
3, 4       Pantalone's Gaussian-process surrogates
=========  =========================================================

Models 0-2 are ported here. **3 and 4 are not**: they are regression surrogates
whose coefficients live in ``crddc.inc`` and friends -- several thousand
tabulated training points per model, fitted offline, with a Gaussian-process
predictor (``gppre``) and four separate basis expansions. Porting them means
carrying that data, and none of the shipped cases use them; ``runs/737/737.tas``
selects ``iengwgt = 1``. Calling with 3 or 4 raises rather than silently
returning a different model's answer.

Geared versus ungeared
----------------------
The switch is ``|Gearf - 1| < 0.001`` -- a gear ratio of one *is* a direct
drive. A geared fan gets a different coefficient set entirely, not a
correction factor, because the fan and the low spool are then sized
independently.

What is in ``Weng``
-------------------
``Weng = Webare + Weadd + Wnac + Wpylon``, where the accessories are a
fraction ``feadd`` of the bare engine and the pylon a fraction ``fpylon`` of
*everything else including the nacelle*. So ``fpylon`` compounds -- raising
``feadd`` raises the pylon weight too.

The nacelle is from NASA CR 151970: four areas (inlet, cowl, exhaust, core)
each at their own weight per unit area, with the inlet and exhaust rates
growing with fan diameter. The unit conversions are written as they appear in
the source -- ``4.45`` N per lbf, ``0.3048`` m per ft, ``0.0254`` m per inch
-- rather than folded together, so the correlation's original form stays
visible.

Verified against the compiled Fortran; see ``tests/test_engine_weight.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["EngineWeight", "tfweight", "UnsupportedWeightModel"]

LB_N = 1.0 / 4.44822       # constants.inc: lbf per newton
FT_M = 0.3048              # metres per foot
IN_M = 0.0254              # metres per inch


class UnsupportedWeightModel(NotImplementedError):
    """``iengwgt`` selects one of the Gaussian-process surrogate models."""


@dataclass(frozen=True)
class EngineWeight:
    Weng: float      # all engines: bare + accessories + nacelles + pylons
    Wnac: float      # all nacelles
    Webare: float    # all bare engines
    Snace1: float    # one nacelle's wetted area


def _fitzgerald_coeffs(BPR: float, geared: bool, advanced: bool):
    """``(a, b, c)`` for ``We1 = a mdotc^b (OPR/40)^c``."""
    if not geared:
        if not advanced:
            return (18.09 * BPR ** 2 + 476.9 * BPR + 701.3,
                    0.001077 * BPR ** 2 - 0.03716 * BPR + 1.190,
                    -0.01058 * BPR + 0.3259)
        return (15.38 * BPR ** 2 + 401.1 * BPR + 631.5,
                0.001057 * BPR ** 2 - 0.03693 * BPR + 1.171,
                -0.01022 * BPR + 0.2321)
    if not advanced:
        return (-0.6590 * BPR ** 2 + 292.8 * BPR + 1915.0,
                0.00006784 * BPR ** 2 - 0.006488 * BPR + 1.061,
                -0.001969 * BPR + 0.07107)
    return (-0.6204 * BPR ** 2 + 237.3 * BPR + 1702.0,
            0.00005845 * BPR ** 2 - 0.005866 * BPR + 1.045,
            -0.001918 * BPR + 0.06765)


def tfweight(iengwgt: int, Gearf: float, OPR: float, BPR: float,
             mdotc: float, dfan: float, rSnace: float, dlcomp: float,
             neng: float, feadd: float, fpylon: float) -> EngineWeight:
    """Total installed engine weight.

    ``mdotc`` is the core mass flow in kg/s, ``dfan`` the fan diameter,
    ``rSnace`` the nacelle wetted area over the fan disc area, ``dlcomp`` the
    core cowl diameter, ``neng`` the engine count.
    """
    if iengwgt > 2:
        raise UnsupportedWeightModel(
            f"iengwgt = {iengwgt} selects a Gaussian-process surrogate "
            "(Pantalone); its training data lives in crddc.inc and is not "
            "ported. Use 0 (Drela), 1 or 2 (Fitzgerald).")

    if iengwgt == 0:
        # Drela: additive in OPR and BPR, referenced to a 45.35 kg/s core.
        We1 = ((mdotc / 45.35)
               * (1684.5 + 17.7 * (OPR / 30.0) + 1662.2 * (BPR / 5.0) ** 1.2)
               / LB_N)
    else:
        geared = abs(Gearf - 1.0) >= 0.001
        acon, bcon, ccon = _fitzgerald_coeffs(BPR, geared, iengwgt == 2)
        We1 = (acon / LB_N) * (mdotc / 45.35) ** bcon * (OPR / 40.0) ** ccon

    Webare = We1 * neng

    # Nacelle, NASA CR 151970. Areas split 40/20/40 of the nacelle wetted
    # area, plus a core cowl of length 3*dlcomp.
    Snace1 = rSnace * 0.25 * math.pi * dfan ** 2
    Ainlet = 0.4 * Snace1
    Acowl = 0.2 * Snace1
    Aexh = 0.4 * Snace1
    Acore = math.pi * dlcomp * (3.0 * dlcomp)
    Wnace1 = (4.45 * (Ainlet / FT_M ** 2) * (2.5 + 0.0238 * dfan / IN_M)
              + 4.45 * (Acowl / FT_M ** 2) * 1.9
              + 4.45 * (Aexh / FT_M ** 2) * (2.5 + 0.0363 * dfan / IN_M)
              + 4.45 * (Acore / FT_M ** 2) * 1.9)
    Wnac = Wnace1 * neng

    Weadd = Webare * feadd
    # The pylon fraction is taken on everything else, accessories included.
    Wpylon = (Webare + Weadd + Wnac) * fpylon
    Weng = Webare + Weadd + Wnac + Wpylon

    return EngineWeight(Weng=Weng, Wnac=Wnac, Webare=Webare, Snace1=Snace1)
