"""Aircraft acquisition cost -- ``cost_est.jl`` and ``cost_val.jl``.

TASOPT 2.16 has no cost model at all. v3 has two, and **both are disclaimed
by their own authors**::

    !!! warning "Unused and Unvetted"
        This legacy function is not used elsewhere in the code but has been
        retained for reference ... it has not been vetted, and is not
        endorsed by the current dev team.

``CostVal``'s docstring adds, in full: *"Honestly, not a clue."*

Worse, they are in different states of disrepair:

* **``CostVal`` runs, but every input is hard-wired** to a 737 MAX9 --
  fuselage weight, tail weights, wing weight, engine weight, max speed,
  passenger count, all of them literals in the body. It is a function of the
  production quantity alone.
* **``CostEst`` cannot be called.** It indexes ``parpt[ipt_Ptshaft]`` and
  ``parpt[ipt_nTshaft]``, turboelectric parameters that have been removed
  from the model -- ``isdefined(TASOPT, :ipt_Ptshaft)`` is ``false``. Any
  call raises immediately.

What this module does about it
------------------------------
The underlying correlations are real and worth having: they are the DAPCA IV
model as published in Raymer, plus Birkler's gas-turbine model and a few
newer fits for electric and hydrogen components. So they are ported **with
their inputs as arguments** rather than as literals, which is what makes them
usable for an aircraft other than a 737 MAX9.

:func:`cost_val_baseline` reproduces the reference's hard-wired case exactly,
which is the only thing there is to verify against.

Nothing here should be read as endorsed. See ``DISCREPANCIES.md`` §69.
"""
from .dapca import (COST_WARNING, AirframeCost, cost_val_baseline,  # noqa
                    dapca_hours, engine_cost, airframe_cost)

__all__ = ["dapca_hours", "airframe_cost", "engine_cost",
           "cost_val_baseline", "AirframeCost", "COST_WARNING"]
