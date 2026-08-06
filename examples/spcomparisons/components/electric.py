"""Electric propulsion presented to the airframe as an engine.

Why a shim and not a rewrite
----------------------------
SPaircraft's airframe touches the engine in twelve places, and they split
cleanly in two:

SIX are real physical couplings, and an electric propulsor has all of them --
thrust, weight, fan diameter (nacelle area, pylon weight, gear clearance),
fan-face area (windmill drag after an engine-out), and a core cowl area.

SIX are turbofan CYCLE MATCHING -- ``M_2``, ``M_25``, ``hold_2``,
``hold_25``, ``c1``, ``alpha_max``. They tie the fan face to the LPC face to
the burner. An electric fan has no core, no burner and no bypass ratio, so
these are not "set to zero", they are absent: the rows do not exist.

This class supplies the first six from the powertrain and the fuel cell or
battery behind it, so the 1,200-variable airframe does not have to know which
kind of propulsor it is carrying. That is the whole point of the component
library -- the wing, fuselage, tails, gear and trim chain are identical
across all five architectures, and only the thing making thrust changes.

The one place the abstraction genuinely leaks
---------------------------------------------
``TSFC``. A turbofan burns fuel proportional to thrust; a fuel cell burns
hydrogen proportional to POWER, and a battery burns nothing at all and does
not get lighter. So ``TSFC`` is deliberately not provided, and the caller
must write the energy accounting itself. Faking a TSFC would silently
reintroduce a weight decrement that a battery aircraft does not have.
"""
from __future__ import annotations

PI = 3.141592653589793


class ElectricPropulsor:
    """Duck-types the six engine attributes the airframe actually reads."""

    def __init__(self, f, N, pt, *, n_eng, prefix="Elec_"):
        self._pt = pt
        V = lambda n, g, u, d: f.Variable(name=f"{prefix}{n}", guess=g,
                                          units=u, description=d)

        # Thrust PER ENGINE. add_powertrain multiplies mass flow by n_fans,
        # so F_net is already the total across all propulsors -- but the
        # airframe writes numeng * eng.F, so handing it the total would
        # double-count thrust and the aircraft would fly on half the
        # propulsion it paid for.
        self.F = f.Variable(name=f"{prefix}F", guess=5.0e4, units="N",
                            description="thrust per propulsor", size=N)
        self.d_f = pt.D_fan

        # Weight per engine, again because the airframe multiplies by n_eng.
        self.W_engine = V("W_engine", 1.0e4, "N", "propulsor weight per unit")
        self.A_2 = V("A_2", 2.0, "m^2", "fan face area")
        # No core. d_LPC feeds only the core-cowl area in the nacelle weight
        # buildup; an electric fan's "core" is a motor fairing, which is what
        # this small residual diameter represents rather than a compressor.
        self.d_LPC = V("d_LPC", 0.30, "m", "motor fairing diameter")

        self.cons = [
            self.F * n_eng <= pt.F_net,
            self.W_engine * n_eng >= pt.W_pt,
            self.A_2 >= (PI / 4.0) * pt.D_fan ** 2,
            self.d_LPC >= 0.25 * pt.D_fan,
        ]

    # TSFC is intentionally absent -- see the module docstring. Touching it
    # should be a loud failure, not a silent wrong answer.
    @property
    def TSFC(self):
        raise AttributeError(
            "ElectricPropulsor has no TSFC: a fuel cell burns hydrogen with "
            "POWER and a battery burns nothing. Write the energy accounting "
            "explicitly instead of reusing the turbofan burn row.")
