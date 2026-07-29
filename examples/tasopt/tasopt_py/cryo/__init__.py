"""Cryogenic fuel tanks -- ``TASOPT.jl/src/cryo_tank``.

Nothing here has a TASOPT 2.16 counterpart. The Fortran has no fuel-tank
model at all: fuel is a weight and a volume in the wing box, which is a fair
description of kerosene and a useless one for a cryogen that boils at 20 K,
has to be carried in a pressure vessel, and warms up all the way through the
mission.

Verified against a running TASOPT.jl; see ``julia_ref/`` for the drivers that
produce the reference values and ``tests/test_fuel_thermo.py`` onward.
"""
from .fuel_thermo import (SaturatedPhase, gas_properties,  # noqa: F401
                          liquid_properties)
from .geometry import CrossSection, scaled_cross_section  # noqa: F401
from .material_data import MATERIALS  # noqa: F401
from .tank import (FuselageTank, InnerTank, OuterTank,  # noqa: F401
                   material, size_inner_tank, size_outer_tank)
from .stiffeners import (find_K1_head, stiffener_weight,  # noqa: F401
                         stiffeners_bending_moment,
                         stiffeners_bending_moment_outer)

__all__ = ["gas_properties", "liquid_properties", "SaturatedPhase",
           "CrossSection", "scaled_cross_section",
           "stiffeners_bending_moment", "stiffeners_bending_moment_outer",
           "stiffener_weight", "find_K1_head",
           "FuselageTank", "InnerTank", "OuterTank", "material",
           "size_inner_tank", "size_outer_tank",
           "MATERIALS"]
