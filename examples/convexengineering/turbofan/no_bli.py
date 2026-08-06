"""The TASOPT D8.2 core (``eng == 3``) without boundary layer ingestion.

``model.py`` registers ``D82_SPaircraft``, which is the ``eng == 3`` engine on
the ``BLI`` branch -- the one ``optimalD8`` takes. Two other SPaircraft
configurations use the same turbomachinery with the inlet in clean flow:
``D8_no_BLI`` and ``optimal737``. They need a separate entry, because the
source picks *different numbers* for the same engine depending on the flag.

What actually changes
---------------------
Three things, and only three:

1. **The on-design mass-flow anchors.** ``engine_validation.py`` carries two
   parallel ``onDest`` blocks under ``if eng == 3``. Ingesting boundary layer
   lowers the stagnation pressure the compressor sees, so the sizing-point
   pressures drop across the board -- HPT 1598.32 -> 1433.49 kPa, LPT 835.585
   -> 706.84, LPC 80.237 -> 65.79434, HPC 399.58 -> 327.66, fan 50 -> 41 --
   and the LPT sizing temperature with them, 1142.6 -> 1121.85 K. Using the
   BLI numbers for a clean inlet leaves the LPC mass-flow bracket about 12%
   out.

2. **Heat of combustion**, 42.5 against 43.003 MJ/kg.

3. **Combustor Cp**, 1253.9 against 1257.9 J/kg/K.

(2) and (3) are not physical consequences of the inlet -- they are simply what
``subs/D8_no_BLI.py`` and ``subs/optimal737.py`` specify where
``subs/optimalD8.py`` specifies something else. They are reproduced because
the gpkit reference for ``D8_no_BLI`` depends on them.

What does *not* change
----------------------
``setvals()`` also sets ``fan_eta_reduct`` to 0.96 under BLI and 1.0 without,
which reads like a fourth difference. It is not: the value is written into the
constants dictionary and never read again, by ``engine_validation.py`` or by
``compressor.py``, ``maps.py``, ``combustor.py`` or ``turbine.py``. The fan
polytropic efficiency is 0.9300 either way. Nothing here applies it, and
nothing in the source does either.

The remaining BLI difference is not a constant at all: it is the pair of
stagnation-pressure and velocity loss factors in the thrust balance, which
``add_engine`` already handles from its own ``BLI`` argument. Pass
``BLI=False`` alongside this engine name.

Importing this module registers the entry. It is additive -- three new keys in
three tables -- so ``D82_SPaircraft`` and every other engine behave exactly as
before.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turbofan.model import ETAS, ONDESIGN, SUBS  # noqa: E402

__all__ = ["NAME"]

#: Engine name to pass to ``add_engine(..., engine=NAME, BLI=False)``.
NAME = "D82_SPaircraft_noBLI"

# Same turbomachinery: fan, LPC, HPC, HPT, LPT polytropic efficiencies.
ETAS.setdefault(NAME, ETAS["D82_SPaircraft"])

# The `else` branch of `if eng == 3` in engine_validation.py, as
# (Tt_HPT, Pt_HPT, Tt_LPT, Pt_LPT, Tt_lpc, Pt_lpc, Tt_hpc, Pt_hpc,
#  fan_lo, fan_hi, Pt_fan).
ONDESIGN.setdefault(NAME, (1400.0, 1598.32, 1142.6, 835.585,
                           289.77, 80.237, 481.386, 399.58,
                           0.7, 1.3, 50.0))

# Everything from the BLI entry except the fuel and the combustor Cp. The
# three design pressure ratios stay free (None), which is pRatOpt=True --
# what SPaircraft.test() does and what reference.py records against.
SUBS.setdefault(NAME, dict(SUBS["D82_SPaircraft"], hf=42.5, Cp_c=1253.9))
