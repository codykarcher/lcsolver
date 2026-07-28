"""Planform sizing and centroid geometry.

Ports ``tailpo`` (``tailpo.f``), ``surfdx`` (``surfdx.f``) and
``wingsc``/``wingAc`` (``wingsc.f``) -- the small routines that turn a lift
requirement and an aspect ratio into an actual planform, and that locate its
area centroid once swept.

The planform integrals
----------------------
All of these are built from the same normalised chord integrals over a
three-panel trapezoidal planform (centre section to ``bo``, then to the break
at ``bs``, then to the tip):

``Kc``
    ``2 int c dy / (co b)``, i.e. ``S/(co b)`` -- area
``Kcx``
    ``2 int c (x - xo) dy / (co b^2)`` -- the swept first moment, which is
    what sets how far aft the area centroid sits
``Kcc``
    ``2 int c^2 dy / (co^2 b)`` -- gives the mean aerodynamic chord

``Kc`` is shared with :mod:`tasopt_py.aero.loading`, which builds it alongside
the load integral ``Kp``. The two are kept separate rather than shared because
the source does, and because ``loading`` needs the tip and root lift-loss
terms that these do not.

``wingsc`` versus ``wingAc``
----------------------------
Same three equations, different unknown. ``wingsc`` takes aspect ratio and
solves for span; ``wingAc`` takes span and solves for aspect ratio. Which one
the sizing loop uses is what decides whether a span constraint (a gate-box
limit, say) is active.

Verified against the compiled Fortran; see ``tests/test_planform.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["TailPlanform", "WingPlanform", "SurfaceCentroid",
           "tailpo", "surfdx", "wingsc", "wingAc", "chord_integrals"]


@dataclass(frozen=True)
class TailPlanform:
    b: float      # span
    co: float     # root chord
    po: float     # root loading at the never-exceed condition


@dataclass(frozen=True)
class WingPlanform:
    S: float      # reference area
    b: float      # span
    bs: float     # break span
    co: float     # root chord
    AR: float     # aspect ratio


@dataclass(frozen=True)
class SurfaceCentroid:
    dx: float      # area centroid offset aft of the root quarter chord
    macco: float   # mean aerodynamic chord, over the root chord


def chord_integrals(b, bs, bo, lambdat, lambdas):
    """``(Kc, Kcx, Kcc)`` for the three-panel planform."""
    etao = bo / b
    etas = bs / b

    Kc = (etao
          + 0.5 * (1.0 + lambdas) * (etas - etao)
          + 0.5 * (lambdas + lambdat) * (1.0 - etas))
    Kcx = ((1.0 + 2.0 * lambdas) * (etas - etao) ** 2 / 12.0
           + (lambdas + 2.0 * lambdat) * (1.0 - etas) ** 2 / 12.0
           + (lambdas + lambdat) * (1.0 - etas) * (etas - etao) / 4.0)
    Kcc = (etao
           + (1.0 + lambdas + lambdas ** 2) * (etas - etao) / 3.0
           + (lambdas ** 2 + lambdas * lambdat + lambdat ** 2)
           * (1.0 - etas) / 3.0)
    return Kc, Kcx, Kcc


def tailpo(S: float, AR: float, lambda_: float, qne: float,
           CLmax: float) -> TailPlanform:
    """Tail span, root chord and root loading from area and aspect ratio.

    A plain two-panel taper -- no break, no centre section -- so the chord
    integral collapses to ``0.5 b (1 + lambda)``. ``qne`` is the never-exceed
    dynamic pressure and ``CLmax`` the tail's design load, so ``po`` is a
    structural load, not a trim condition.
    """
    b = math.sqrt(S * AR)
    co = S / (0.5 * b * (1.0 + lambda_))
    po = qne * S * CLmax / b * 2.0 / (1.0 + lambda_)
    return TailPlanform(b=b, co=co, po=po)


def surfdx(b: float, bs: float, bo: float, lambdat: float, lambdas: float,
           sweep: float) -> SurfaceCentroid:
    """Area centroid offset from sweep, and the mean aerodynamic chord."""
    tanL = math.tan(sweep * math.pi / 180.0)
    Kc, Kcx, Kcc = chord_integrals(b, bs, bo, lambdat, lambdas)
    return SurfaceCentroid(dx=Kcx / Kc * b * tanL, macco=Kcc / Kc)


def wingsc(W: float, CL: float, qinf: float, AR: float, etasi: float,
           bo: float, lambdat: float, lambdas: float) -> WingPlanform:
    """Wing area, span, break span and root chord from weight, CL and AR.

    ``etasi`` is where the break is wanted as a fraction of span; the break is
    pushed out to ``bo`` if that would put it inside the fuselage.
    """
    S = W / (qinf * CL)
    b = math.sqrt(S * AR)
    bs = max(b * etasi, bo)
    Kc, _, _ = chord_integrals(b, bs, bo, lambdat, lambdas)
    return WingPlanform(S=S, b=b, bs=bs, co=S / (Kc * b), AR=AR)


def wingAc(W: float, CL: float, qinf: float, b: float, bs: float,
           bo: float, lambdat: float, lambdas: float) -> WingPlanform:
    """Wing area, aspect ratio and root chord from weight, CL and a *fixed span*.

    The alternative to :func:`wingsc` when span is constrained rather than
    aspect ratio. Note the source's ``wingAc`` takes ``AR`` and ``etasi`` in
    its argument list and uses neither -- ``bs`` comes in already set and
    ``AR`` is an output. The unused arguments are dropped here.
    """
    S = W / (qinf * CL)
    Kc, _, _ = chord_integrals(b, bs, bo, lambdat, lambdas)
    return WingPlanform(S=S, b=b, bs=bs, co=S / (Kc * b), AR=b ** 2 / S)
