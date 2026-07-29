"""Electric propulsion -- ``TASOPT.jl/src/propsys``.

Nothing here has a TASOPT 2.16 counterpart. The Fortran has no electrical
system: it has shaft-power offtakes that vanish from the cycle without going
anywhere, which is a fair model of a bleed-driven accessory and useless for
an aircraft where electricity is the propulsion.

Verified against a running TASOPT.jl; see ``julia_ref/``.
"""
from .electric import (Cable, Inverter, SizedCable,  # noqa: F401
                       operate_inverter, resistivity, size_cable,
                       size_inverter)

__all__ = ["Inverter", "size_inverter", "operate_inverter",
           "Cable", "size_cable", "SizedCable", "resistivity"]
