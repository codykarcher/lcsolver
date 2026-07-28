"""Wing pitching moment and tail planform sizing.

Ports of TASOPT ``src/surfcm.f`` and ``src/tailpo.f``.

``surfcm`` decomposes the wing pitching moment about the wing root axis into
a lift-independent and a lift-dependent part:

    CM = CM0 + CM1*(CL - CLhtail)

``CM0`` collects the airfoil section moments ``cmpo/cmps/cmpt`` at the three
spanwise stations, and ``CM1`` collects the geometric contributions — the
offset of the reference axis from the quarter chord, the sweep, and the tip
load correction ``fLt``.

The K-factors are planform integrals over the two-panel wing:

    Kc  chord-weighted area factor (so 1/(AR*Kc) is a mean chord ratio)
    Kp  load-weighted factor, including the centrebody carryover fLo and
        the tip correction fLt

``tailpo`` is the tail counterpart of the wing planform sizing: given area,
aspect ratio and taper it returns span, root chord and the root loading at
the never-exceed dynamic pressure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

PI = 3.1415926535897932384626


@dataclass(frozen=True)
class MomentResult:
    CM0: float
    CM1: float


def surfcm(*, b, bs, bo, sweep, Xaxis,
           lambdat, lambdas, gammat, gammas,
           AR, fLo, fLt, cmpo, cmps, cmpt) -> MomentResult:
    """Wing CM components about the root axis. Follows ``surfcm.f`` exactly."""
    cosL = math.cos(sweep * PI / 180.0)
    tanL = math.tan(sweep * PI / 180.0)

    etao = bo / b
    etas = bs / b

    Kc = (etao
          + 0.5 * (1.0 + lambdas) * (etas - etao)
          + 0.5 * (lambdas + lambdat) * (1.0 - etas))

    Ko = 1.0 / (AR * Kc)

    Kp = (etao
          + 0.5 * (1.0 + gammas) * (etas - etao)
          + 0.5 * (gammas + gammat) * (1.0 - etas)
          + fLo * etao + 2.0 * fLt * Ko * gammat * lambdat)

    C1 = ((1.0 + 0.5 * (lambdas + gammas) + lambdas * gammas) * (etas - etao)
          + (lambdas * gammas
             + 0.5 * (lambdas * gammat + gammas * lambdat)
             + lambdat * gammat) * (1.0 - etas))

    C2 = ((1.0 + 2.0 * gammas) * (etas - etao)**2
          + (gammas + 2.0 * gammat) * (1.0 - etas)**2
          + 3.0 * (gammas + gammat) * (etas - etao) * (1.0 - etas))

    C3 = ((cmpo * (3.0 + 2.0 * lambdas + lambdas**2)
           + cmps * (3.0 * lambdas**2 + 2.0 * lambdas + 1.0)) * (etas - etao)
          + (cmps * (3.0 * lambdas**2 + 2.0 * lambdas * lambdat + lambdat**2)
             + cmpt * (3.0 * lambdat**2 + 2.0 * lambdas * lambdat + lambdas**2))
          * (1.0 - etas))

    CM1 = ((1.0 / Kp)
           * (etao * (1.0 + fLo) * (Xaxis - 0.25)
              + (Xaxis - 0.25) * cosL**2 * C1 / 3.0
              - (tanL / Ko) * C2 / 12.0
              + 2.0 * fLt * lambdat * gammat
              * (Ko * lambdat * (Xaxis - 0.25) * cosL**2
                 - 0.5 * (1.0 - etao) * tanL)))

    CM0 = (cosL**4 / Kc) * C3 / 12.0

    return MomentResult(CM0=CM0, CM1=CM1)


@dataclass(frozen=True)
class TailPlanform:
    b: float    # span
    co: float   # root chord
    po: float   # root loading (force per unit span)


def tailpo(*, S, AR, lambda_, qne, CLmax) -> TailPlanform:
    """Tail span, root chord and root loading. Follows ``tailpo.f`` exactly.

    ``lambda`` is spelled ``lambda_`` because it is a Python keyword.
    """
    b = math.sqrt(S * AR)
    co = S / (0.5 * b * (1.0 + lambda_))
    po = qne * S * CLmax / b * 2.0 / (1.0 + lambda_)
    return TailPlanform(b=b, co=co, po=po)


__all__ = ["surfcm", "tailpo", "MomentResult", "TailPlanform"]
