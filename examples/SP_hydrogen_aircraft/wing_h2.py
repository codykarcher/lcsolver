"""Structural wing for the hydrogen aircraft: planform + Hoburg wing box.

Why this module exists
----------------------
The first pass carried ``W_wing >= 350 N/m^2 * S`` -- a constant the optimiser
cannot argue with. The attempt to fix that by importing SPaircraft's
``add_wing`` failed for a reason recorded in the README: that module's wing is
closed by the fuselage, landing gear and trim block, so it cannot be taken
alone.

Its *wing box* can. ``spaircraft.wingbox.add_wingbox`` is self-contained given
the planform and a load case: nine posynomial constraints (Hoburg & Abbeel
2014), sizing spar caps against root bending and the shear web against the
ultimate load, with the taper factor entering through the posynomial fit

    nu^3.94 >= 0.86 p^-2.38 + 0.14 p^0.56

whose provenance this session established twice over: MAIDAS classifies the
fit posynomial, and classifies the *exact* ratio it replaces --
``(1+lam+lam^2)/(1+lam)^2``, straight out of the TASOPT port's ``surfw`` --
as not_gp (posynomial over posynomial). The fit is the standard cure, and the
box is the already-worked-out SP form of the same physics MAIDAS mapped in
``tasopt_py/structures/surface.py``.

The hydrogen difference
-----------------------
A kerosene wing carries fuel, and that fuel relieves bending: the load case
subtracts fuel weight from the lift the structure must react. A hydrogen wing
carries none -- the LH2 is in the fuselage tank -- so the box here is sized by
the full ultimate load with **no relief term**. That is a real structural
penalty of hydrogen, and dropping the relief is one constraint *removed*
rather than added.

Taper is fixed at lambda = 0.3 (p = 1.6, q = 1.3), as in the simple Hoburg
model; freeing it costs two variables and buys little at this fidelity.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "convexengineering"))
from spaircraft.wingbox import add_wingbox   # noqa: E402

__all__ = ["add_wing_h2"]

#: Taper substitutions for lambda = 0.3.
P_TAPER, Q_TAPER = 1.6, 1.3


def add_wing_h2(f, *, prefix: str = "Wing_"):
    """Planform + structural box. Returns ``(vars, constraints)``.

    The caller supplies the load case by constraining ``L_max`` (ultimate
    load, load factor already applied) -- typically ``L_max >= N_ult * W_MTO``.
    Everything else closes here.
    """
    wg = f.group("wing", prefix=prefix)
    V = wg.Variable

    # Bounds are generous engineering limits, for the same reason as
    # everywhere else in this model: several relations are reciprocal in the
    # design variables, and an unbounded log-space iterate overflows before it
    # can be infeasible.
    AR = V("AR", 10.0, "-", "aspect ratio", bounds=(6.0, 14.0))
    S = V("S", 68.0, "m^2", "reference area", bounds=(30.0, 250.0))
    b = V("b", 26.0, "m", "span", bounds=(12.0, 55.0))
    tau = V("tau", 0.13, "-", "thickness-to-chord ratio", bounds=(0.08, 0.15))
    Lmax = V("L_max", 1.2e6, "N", "ultimate load", bounds=(1e5, 2e7))
    Mr = V("M_r", 8.0e5, "N", "root moment per root chord",
           bounds=(1e4, 2e7))
    W_wing = V("W_wing", 2.3e4, "N", "wing weight incl. non-structural",
               bounds=(2e3, 5e5))

    box = add_wingbox("wing", AR=AR, b=b, S=S, p=P_TAPER, q=Q_TAPER,
                      tau=tau, Lmax=Lmax, Mr=Mr, tau_max=0.15,
                      group=wg.group("box", prefix=f"{prefix}box_"))
    box_vars, cons = box if isinstance(box, tuple) else (box, [])

    f_ns = wg.Constant("f_nonstruct", 1.2, "-",
                       "LE/TE devices, ribs, controls over box weight")

    cons += [
        # Root moment per root chord for triangular-ish spanwise loading --
        # the same form SPaircraft's vertical tail uses with its own L_max.
        # No fuel-relief subtraction: a hydrogen wing is dry.
        Mr >= Lmax * AR * P_TAPER / 24.0,
        W_wing >= f_ns * box_vars.W_struct,
    ]

    out = dict(AR=AR, S=S, b=b, tau=tau, L_max=Lmax, M_r=Mr,
               W_wing=W_wing, box=box_vars)
    return out, cons
