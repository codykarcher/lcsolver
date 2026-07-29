"""v3 engine additions -- ``TASOPT.jl/src/engine``.

Kept separate from :mod:`tasopt_py.engine`, which is the verified port of
TASOPT 2.16's turbofan. Nothing here has a 2.16 counterpart.
"""
from .ducted_fan import ducted_fan_weight, DuctedFanWeight  # noqa: F401
from .heat_exchanger import (colburn_j_pipe, hx_weight,  # noqa: F401
                             nusselt_staggered,
                             pressure_drop_staggered, tube_thickness)
from .fuelcell import (cell_voltage_simple, power_density,  # noqa: F401
                       stack_operate, stack_size, stack_weight)

__all__ = ["ducted_fan_weight", "DuctedFanWeight",
           "cell_voltage_simple", "power_density", "stack_size",
           "stack_operate", "stack_weight",
           "colburn_j_pipe", "nusselt_staggered",
           "pressure_drop_staggered", "tube_thickness", "hx_weight"]
from .hx_size import (HXGas, HXTubular, hx_size, hx_operate,  # noqa: F401,E501
                      hx_optimize, hx_objective, tube_geometry,
                      gas_tset_single)

__all__ += ["HXGas", "HXTubular", "hx_size", "hx_operate", "hx_optimize",
            "hx_objective", "tube_geometry", "gas_tset_single"]
from .maps import (CompressorMap, FAN_MAP, LPC_MAP, HPC_MAP,  # noqa: F401,E501
                   MAP_BY_NAME, find_NR_inverse,
                   compressor_speed_and_efficiency)
from .ducted_fan_cycle import (DuctedFanState, ducted_fan_size,  # noqa: F401,E501
                               ducted_fan_operate, V3_CMAPF)

__all__ += ["CompressorMap", "FAN_MAP", "LPC_MAP", "HPC_MAP", "MAP_BY_NAME",
            "find_NR_inverse", "compressor_speed_and_efficiency",
            "DuctedFanState", "ducted_fan_size", "ducted_fan_operate",
            "V3_CMAPF"]
