"""Electric propulsion -- ``TASOPT.jl/src/propsys``.

Nothing here has a TASOPT 2.16 counterpart. The Fortran has no electrical
system: it has shaft-power offtakes that vanish from the cycle without going
anywhere, which is a fair model of a bleed-driven accessory and useless for
an aircraft where electricity is the propulsion.

Verified against a running TASOPT.jl; see ``julia_ref/``.
"""
from .motor import (MotorDesign, SizedMotor,  # noqa: F401
                    airgap_flux, core_loss, eddy_loss, size_motor,
                    hysteresis_loss, ohmic_loss, remanent_flux,
                    windage_loss)
from .electric import (Cable, Inverter, SizedCable,  # noqa: F401
                       operate_inverter, resistivity, size_cable,
                       size_inverter)

__all__ = ["Inverter", "size_inverter", "operate_inverter",
           "Cable", "size_cable", "SizedCable", "resistivity",
           "airgap_flux", "remanent_flux", "ohmic_loss", "hysteresis_loss",
           "eddy_loss", "core_loss", "windage_loss",
           "MotorDesign", "SizedMotor", "size_motor"]
