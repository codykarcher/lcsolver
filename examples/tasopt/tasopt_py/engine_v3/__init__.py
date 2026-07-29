"""v3 engine additions -- ``TASOPT.jl/src/engine``.

Kept separate from :mod:`tasopt_py.engine`, which is the verified port of
TASOPT 2.16's turbofan. Nothing here has a 2.16 counterpart.
"""
from .ducted_fan import ducted_fan_weight, DuctedFanWeight  # noqa: F401

__all__ = ["ducted_fan_weight", "DuctedFanWeight"]
