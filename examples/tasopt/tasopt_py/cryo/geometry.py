"""Fuselage cross-sections, scaled -- ``layout.jl`` / ``fuselage_geometry.jl``.

A cryogenic tank sits inside the fuselage and takes its shape: on a
double-bubble the tank is a double bubble too, scaled down. So the tank sizing
needs the perimeter and enclosed area of a cross-section *at an arbitrary
radius*, not just at the fuselage's own.

This is not new physics -- TASOPT 2.16 computes exactly the same area inline
in three places (``fusew.f``, ``wsize.f``, and the ASWING export). The Fortran
writes it out each time::

      thetafb = asin(wfb/Rfuse)
      sin2t   = 2*sin(thetafb)*cos(thetafb)
      Afuse   = (pi + 2*thetafb + sin2t)*Rfuse**2 + 2*Rfuse*dRfuse

which is this module's :meth:`CrossSection.area` at ``n_webs = 1``. v3
factors it into an object so it can be scaled; the port follows, and the tests
check the two agree with 2.16's inline version.

The mapping between the two vocabularies:

===========================  =========================================
v3                           TASOPT 2.16
===========================  =========================================
``radius``                   ``Rfuse``
``bubble_lower_downward_shift``  ``dRfuse``
``bubble_center_y_offset``   ``wfb``
``n_webs``                   ``nfweb``
===========================  =========================================

A ``SingleBubble`` is just ``n_webs = 0`` with no offset, so one class covers
both of the reference's types.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["CrossSection", "scaled_cross_section"]


@dataclass(frozen=True)
class CrossSection:
    """A fuselage cross-section: circular, double-bubble, or multi-bubble.

    ``n_webs = 0`` is the reference's ``SingleBubble``; anything else is its
    ``MultiBubble``.
    """
    radius: float                          # Rfuse
    bubble_lower_downward_shift: float = 0.0   # dRfuse
    bubble_center_y_offset: float = 0.0    # wfb
    n_webs: int = 0

    def __post_init__(self):
        if self.radius <= 0.0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        if self.n_webs and abs(self.bubble_center_y_offset) > self.radius:
            raise ValueError(
                f"bubble centre offset {self.bubble_center_y_offset} exceeds "
                f"radius {self.radius}; asin would be undefined")

    # -- web geometry ------------------------------------------------------
    @property
    def theta_web(self) -> float:
        """Half-angle subtended by a web. Zero without webs."""
        if not self.n_webs:
            return 0.0
        return math.asin(self.bubble_center_y_offset / self.radius)

    @property
    def h_web(self) -> float:
        """Half-height of a web."""
        if not self.n_webs:
            return 0.0
        return math.sqrt(self.radius ** 2
                         - self.bubble_center_y_offset ** 2)

    @property
    def sin2theta(self) -> float:
        if not self.n_webs:
            return 0.0
        cos_t = self.h_web / self.radius
        sin_t = self.bubble_center_y_offset / self.radius
        return 2.0 * sin_t * cos_t

    # -- the two quantities the tank sizing wants ---------------------------
    @property
    def area(self) -> float:
        """Enclosed cross-sectional area, m^2."""
        R = self.radius
        dR = self.bubble_lower_downward_shift
        if not self.n_webs:
            return math.pi * R ** 2 + 2.0 * R * dR
        return ((math.pi + self.n_webs * (2.0 * self.theta_web
                                          + self.sin2theta)) * R ** 2
                + 2.0 * R * dR)

    @property
    def perimeter(self) -> float:
        """Outer perimeter, m."""
        R = self.radius
        dR = self.bubble_lower_downward_shift
        if not self.n_webs:
            return 2.0 * math.pi * R + 2.0 * dR
        return ((2.0 * math.pi + 4.0 * self.theta_web * self.n_webs) * R
                + 2.0 * dR)


def scaled_cross_section(cs: CrossSection, R: float) -> tuple:
    """``(perimeter, area)`` of a geometrically similar section at radius R.

    Every length scales by ``R / cs.radius`` -- radius, downward shift and
    bubble centre offset alike -- so the web half-angle
    ``asin(offset / radius)`` is preserved and the scaled section really is
    similar to the original. The area is therefore exactly quadratic in R and
    the perimeter exactly linear; the tests check both directly.
    """
    ratio = R / cs.radius
    scaled = CrossSection(
        radius=R,
        bubble_lower_downward_shift=(ratio
                                     * cs.bubble_lower_downward_shift),
        bubble_center_y_offset=ratio * cs.bubble_center_y_offset,
        n_webs=cs.n_webs,
    )
    return scaled.perimeter, scaled.area
