"""Technology level, as swappable data.

The model was being validated against a 1990s 737-800 while carrying 2020s
assumptions: spar caps 21% stronger than TASOPT's own 737 deck allows, and an
engine at OPR 45 / BPR 15 against the CFM56-7B's ~32 / ~5.1. Every one of
those makes the aeroplane lighter, which is the direction the 8% MTOW gap ran.

That is not a bug in the physics -- it is a category error in the comparison.
An advanced-technology model SHOULD come out lighter than the aluminium
aeroplane it is being checked against, and until the two are separated there
is no way to tell a modelling error from a technology delta.

So technology is a keyword now. ``CFM56_ERA`` reproduces TASOPT's 737 deck and
is what the validation case should use; ``MODERN`` is what the architecture
matrix should use, because comparing a hydrogen or fuel-cell aircraft against a
1990s baseline would flatter it for reasons that have nothing to do with its
energy source.

Sources
-------
Structural allowables are TASOPT's own 737 run deck
(``Tasopt2.16/runs/737/737.tas``), which lists them in psi::

    1.0     ! sigfac    convenient multiplier on all the stress values below
    15000.0 / 0.000145  ! sigskin   fuselage pressurization skin stress
    30000.0 / 0.000145  ! sigcap    wing,tail bending caps
    20000.0 / 0.000145  ! tauweb    wing,tail shear webs

Secondary-structure fractions come from the same deck, lines 219-225. Those are
NOT technology: a slat weighs what a slat weighs in any era, so they live in
wing.py as plain corrected values rather than as a knob here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Technology:
    """Structural allowables and engine cycle limits for one technology era."""
    name: str
    #: Spar cap allowable tensile stress, Pa. TASOPT `sigcap`.
    sigma_cap: float
    #: Shear web allowable, Pa. TASOPT `tauweb`.
    tau_web: float
    #: Box material density, kg/m^3.
    rho_cap: float
    #: Overall pressure ratio ceiling.
    opr_max: float
    #: Bypass ratio ceiling.
    bpr_max: float
    source: str = ''

    def material(self):
        """The dict the wingbox model wants."""
        return {"rho": self.rho_cap, "sigma": self.sigma_cap, "tau": self.tau_web}


#: TASOPT's 737 run deck exactly: 30 ksi caps, 20 ksi webs, aluminium, and a
#: CFM56-7B-class cycle. This is the level to validate against a real 737-800.
CFM56_ERA = Technology(
    name='cfm56_era',
    sigma_cap=30000.0 / 0.000145,      # 206.9 MPa
    tau_web=20000.0 / 0.000145,        # 137.9 MPa
    rho_cap=2700.0,
    opr_max=32.0,                      # CFM56-7B, ~32.7 at cruise
    bpr_max=5.5,                       # CFM56-7B, ~5.1
    source='Tasopt2.16/runs/737/737.tas; CFM56-7B public cycle data',
)

#: What the architecture matrix should run: current-generation allowables and a
#: geared-fan cycle. The stress values are the ones that were hard-coded in
#: wingbox.py as ALUMINIUM before technology was separated out.
MODERN = Technology(
    name='modern',
    sigma_cap=250e6,
    tau_web=167e6,
    rho_cap=2700.0,
    opr_max=45.0,
    bpr_max=15.0,
    source='wingbox.py ALUMINIUM as it stood; LEAP/GTF-class cycle limits',
)

#: Composite box, otherwise modern.
MODERN_COMPOSITE = Technology(
    name='modern_composite',
    sigma_cap=450e6,
    tau_web=300e6,
    rho_cap=1600.0,
    # 60: the GE9X generation's overall pressure ratio. 45 was BINDING on
    # the 787 -- the cycle wanted more, which is faithful to where that
    # engine generation actually went.
    opr_max=60.0,
    bpr_max=15.0,
    source='wingbox.py COMPOSITE as it stood',
)

#: The D8.2's technology, read off runs/D8/d82.tas. NOTE the structure is
#: UNCHANGED from the 737 -- sigcap, tauweb, Ecap and the densities are
#: identical in the two decks (rhocap 0.0975*27680.4 = 2698.8 vs 2700). The
#: D8's advantage is aerodynamic and propulsive, not materials, and giving it
#: composite allowables would flatter it for a reason its own deck disclaims.
D8_ERA = Technology(
    name='d8_era',
    sigma_cap=30000.0 / 0.000145,      # identical to CFM56_ERA
    tau_web=20000.0 / 0.000145,
    rho_cap=2700.0,
    opr_max=35.0,                      # d82.tas OPR 35
    # d82.tas BPR 6.9674, EXACTLY. This was 7.5 -- "small headroom" -- and the
    # headroom became a binding constraint: bypass ratio pinned to the ceiling
    # in every D8 solve, because higher bypass at fixed FPR shrinks the core
    # and the York/Hoburg/Drela weight fit scales with core mass flow, so the
    # engine got lighter the further BPR was allowed to run.
    #
    # TASOPT has no BPR coupling to port -- it is a plain deck input there, so
    # this ceiling IS the equivalent of TASOPT's input and should carry the
    # deck's value rather than a guess above it.
    bpr_max=6.9674,
    source='Tasopt2.16/runs/D8/d82.tas',
)

TECHNOLOGIES = {t.name: t for t in (CFM56_ERA, MODERN, MODERN_COMPOSITE, D8_ERA)}


def current():
    """The technology level in force, from ``TECH`` (default ``modern``)."""
    return TECHNOLOGIES[os.environ.get("TECH", "modern")]
