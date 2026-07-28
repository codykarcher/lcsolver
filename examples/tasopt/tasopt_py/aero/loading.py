"""Wing root loading and section lift coefficients — a port of ``wingpo.f``.

``wingpo``
    The root loading ``po`` that balances the net load ``N*W - Lhtail``.
``wingcl``
    Section lift coefficients at the three planform stations: root, break
    and tip.

Both are built from the same three planform integrals that appear in
``surfw.f``, ``surfcm.f`` and ``surfcd.f``:

``Kc``
    the chord (area) integral, so ``Kc * b * co`` is the surface area;
``Kp``
    the load integral, plus the root and tip lift losses ``fLo`` and ``fLt``;
``Ko``
    ``1/(AR*Kc)``, which converts between the two.

The root loading then follows from load balance, ``po = (N*W - Lhtail)/(Kp*b)``,
and the section coefficients from the ratio ``Kc/Kp`` — area over load.

The tip-loss term ``2*fLt*Ko*gammat*lambdat`` carries ``Ko``, so it depends on
aspect ratio while the rest of ``Kp`` does not. That is not an inconsistency:
the tip loss is a fixed *lift* decrement, and dividing by ``AR*Kc`` is what
puts it on the same footing as the spanwise load integral.

``wingcl`` duplicates the section-coefficient calculation inside
``surfcd2``, which gives a free cross-check between two independently ported
routines; ``tests/test_loading.py`` asserts they agree.

Verified against the compiled Fortran; see ``tests/test_loading.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["SectionCl", "planform_integrals", "wingcl", "wingpo"]


@dataclass(frozen=True)
class SectionCl:
    clo: float   # root
    cls: float   # break
    clt: float   # tip


def planform_integrals(b, bs, bo, lambdat, lambdas, gammat, gammas,
                       AR, fLo, fLt):
    """``(Kc, Ko, Kp)`` — the chord, conversion and load integrals."""
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
    return Kc, Ko, Kp


def wingpo(b: float, bs: float, bo: float,
           lambdat: float, lambdas: float, gammat: float, gammas: float,
           AR: float, N: float, W: float, Lhtail: float,
           fLo: float, fLt: float) -> float:
    """Wing root loading balancing ``N*W - Lhtail``."""
    _, _, Kp = planform_integrals(b, bs, bo, lambdat, lambdas, gammat, gammas,
                                  AR, fLo, fLt)
    return (N * W - Lhtail) / (Kp * b)


def wingcl(b: float, bs: float, bo: float,
           lambdat: float, lambdas: float, gammat: float, gammas: float,
           sweep: float, AR: float, CL: float, CLhtail: float,
           fLo: float, fLt: float,
           duo: float, dus: float, dut: float) -> SectionCl:
    """Section lift coefficients at root, break and tip.

    ``duo``/``dus``/``dut`` are the local fractional overspeeds; a section
    running fast needs less lift coefficient for the same lift, hence the
    ``(1 + du)^-2``.
    """
    cosL = math.cos(sweep * math.pi / 180.0)
    Kc, _, Kp = planform_integrals(b, bs, bo, lambdat, lambdas, gammat,
                                   gammas, AR, fLo, fLt)
    cl1 = (CL - CLhtail) / cosL ** 2 * (Kc / Kp)
    return SectionCl(clo=cl1 / (1.0 + duo) ** 2,
                     cls=cl1 * gammas / lambdas / (1.0 + dus) ** 2,
                     clt=cl1 * gammat / lambdat / (1.0 + dut) ** 2)
