"""Landing gear sizing -- ``size_landing_gear.jl``.

TASOPT 2.16 sizes landing gear by **fixed mass fraction**: ``flgnose`` and
``flgmain`` in the ``.tas`` file, multiplied by MTOW, and that is the whole
model. The gear has no length, so nothing connects it to the aircraft it is
holding up -- change the engine diameter or the tailstrike angle and the gear
weight does not move.

v3 keeps that as one option and adds a second: Raymer (2012) historical
correlations, which need a gear *length*, which in turn has to come from
geometry. That closes a loop 2.16 does not have.

How the length is decided
-------------------------
The nose gear has to be long enough to satisfy **both**:

* **Engine ground clearance** -- the nacelle must clear the ground, allowing
  for the wing dihedral lifting the outboard engine.
* **Tailstrike** -- the tail must clear the ground at the rotation angle.

whichever is larger. The main gear is then that length plus whatever the
dihedral adds at its spanwise station. So a bigger fan or a longer tailcone
makes the gear longer, and a longer gear makes it heavier -- which is the
coupling the fraction model cannot express.

Verified against TASOPT.jl; see ``tests/test_landing_gear.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Gear", "LandingGear", "size_landing_gear",
           "MASS_FRACTIONS", "HISTORICAL_CORRELATIONS", "LOAD_FACTOR",
           "IN_TO_M"]

MASS_FRACTIONS = "mass_fractions"
HISTORICAL_CORRELATIONS = "historical_correlations"

#: Three-times gear load factor times a 1.5 ultimate factor. Fixed in the
#: source, not an input.
LOAD_FACTOR = 4.5

LB_N = 1.0 / 4.44822
#: v3's ``in_to_m``, **metres per inch**, and the correlations divide by it.
#: TASOPT 2.16 carries the reciprocal directly as ``in_m = 39.37``, a
#: truncated 39.37007874..., so the two differ in the sixth digit. The
#: reference's convention is used here; see ``DISCREPANCIES.md`` §62.
IN_TO_M = 0.0254
KTS_MPS = 0.51444


@dataclass
class Gear:
    """One gear unit -- nose or main."""
    overall_mass_fraction: float = 0.0
    number_struts: int = 1
    wheels_per_strut: int = 2
    #: Filled in by the correlation model, m.
    length: float = 0.0
    #: Main gear only: how far aft of the aft CG limit it sits, m.
    distance_CG_to_landing_gear: float = 0.0
    #: Main gear only: spanwise station as a fraction of the half span.
    y_offset_halfspan_fraction: float = 0.0
    #: Weight, N, and its station.
    weight: float = 0.0
    x: float = 0.0
    y: float = 0.0


@dataclass
class LandingGear:
    model: str = MASS_FRACTIONS
    nose_gear: Gear = None
    main_gear: Gear = None
    #: Rotation angle the tail must clear the ground at, rad.
    tailstrike_angle: float = 0.0
    #: Required clearance under the nacelle, m.
    engine_ground_clearance: float = 0.0
    wing_dihedral_angle: float = 0.0

    def __post_init__(self):
        if self.nose_gear is None:
            self.nose_gear = Gear()
        if self.main_gear is None:
            self.main_gear = Gear()
        if self.model not in (MASS_FRACTIONS, HISTORICAL_CORRELATIONS):
            raise ValueError(
                f"landing gear model must be {MASS_FRACTIONS!r} or "
                f"{HISTORICAL_CORRELATIONS!r}, not {self.model!r}")


def size_landing_gear(gear: LandingGear, WMTO: float, *, xCGaft: float = 0.0,
                      x_end: float = 0.0, Rfuse: float = 0.0,
                      span: float = 0.0, etas: float = 0.0,
                      dfan: float = 0.0, Vstall: float = 0.0) -> tuple:
    """``(Wlgnose, Wlgmain)`` in N, and the gear lengths filled in.

    With ``model = "mass_fractions"`` this is 2.16's model exactly and the
    geometry arguments are ignored. With ``"historical_correlations"`` the
    lengths are worked out first and Raymer's weight correlations applied.
    """
    if gear.model == MASS_FRACTIONS:
        return (WMTO * gear.nose_gear.overall_mass_fraction,
                WMTO * gear.main_gear.overall_mass_fraction)

    nose, main = gear.nose_gear, gear.main_gear

    # Main gear station, and the tail clearance it has to provide.
    x_lg = xCGaft + main.distance_CG_to_landing_gear
    l_tailstrike = ((x_end - x_lg) * math.tan(gear.tailstrike_angle)
                    - 2.0 * Rfuse)

    # Engine clearance, with the dihedral lifting the outboard engine.
    yeng = etas * span / 2.0
    l_clearance = (gear.engine_ground_clearance + dfan
                   - yeng * math.tan(gear.wing_dihedral_angle))

    lgnose_length = max(l_clearance, l_tailstrike)
    y_main = span / 2.0 * main.y_offset_halfspan_fraction
    lgmain_length = (lgnose_length
                     + y_main * math.tan(gear.wing_dihedral_angle))

    n_wheels_main = main.number_struts * main.wheels_per_strut
    n_wheels_nose = nose.number_struts * nose.wheels_per_strut

    # Raymer (2012). Written in pounds, inches and knots, then converted
    # back -- which is why every term carries a unit factor.
    Wlgmain = (0.0106 * (WMTO * LB_N) ** 0.888 * LOAD_FACTOR ** 0.25
               * (lgmain_length / IN_TO_M) ** 0.4 * n_wheels_main ** 0.321
               * main.number_struts ** -0.5
               * (Vstall / KTS_MPS) ** 0.1 / LB_N)
    Wlgnose = (0.032 * (WMTO * LB_N) ** 0.646 * LOAD_FACTOR ** 0.2
               * (lgnose_length / IN_TO_M) ** 0.5
               * n_wheels_nose ** 0.45 / LB_N)

    nose.length = lgnose_length
    main.length = lgmain_length
    nose.overall_mass_fraction = Wlgnose / WMTO
    main.overall_mass_fraction = Wlgmain / WMTO
    main.y = y_main
    main.x = x_lg
    nose.weight = Wlgnose
    main.weight = Wlgmain

    return Wlgnose, Wlgmain
