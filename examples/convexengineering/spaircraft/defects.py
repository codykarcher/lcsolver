"""Known defects in the source model, and what this port does about them.

A rebuild has to decide, for every mistake it finds upstream, whether to copy
it or correct it. Copying keeps the recorded gpkit reference meaningful --
that is the only way "does the transcription agree" is a question with an
answer. Correcting gives an aeroplane whose numbers you can use. You cannot
have both from one build, so the decks take a ``faithful`` flag: default off,
so what you get is the corrected model, and on for verification, so the diff
against gpkit is measuring transcription rather than these decisions.

Every defect below is *upstream*, in ``convexengineering/SPaircraft`` or
``convexengineering/turbofan``. Errors in this port are simply fixed, and are
not listed here -- there is no reason to keep one. The audit that looks for
them evaluates every constant in the decks against the source ``subs/`` dicts;
at the time of writing it reports one, ``V_mn`` in ``model.py``, since fixed.

Nothing here is a matter of taste. Each is a statement the source makes twice,
in two different ways, that disagree with itself.
"""
from __future__ import annotations

from numpy import cos, pi

from .airframe import set_constant

__all__ = ["cl_w_max", "correct_M_4a", "M_4A", "M_4A_SOURCE"]


# ---------------------------------------------------------------------------
# 1. Degrees fed to cos() as radians
# ---------------------------------------------------------------------------

def cl_w_max(coeff: float, sweep_deg: float, *, faithful: bool = False):
    """Maximum wing lift coefficient, ``coeff / cos(sweep)**2``.

    ``subs/D8_no_BLI.py`` line 76 reads::

        'C_{L_{w,max}}': 2.15/(cos(sweep)**2)

    where every other subs deck -- ``optimalD8``, ``optimal737``,
    ``optimal777``, ``M072_737`` -- reads ``cos(sweep * pi / 180.)``. The local
    ``sweep`` is 13.237 and is commented ``# [deg]``, so the missing conversion
    hands 13.237 *radians* to ``cos``. That is two full turns plus 0.671 rad,
    which lands at 0.784 instead of 0.973, and squares into a maximum lift
    coefficient of 3.503 where 2.269 was meant -- 54% high, in the limit that
    sizes the wing for low speed.

    ``faithful=True`` reproduces it, because the recorded gpkit solution for
    ``D8_no_BLI`` was produced by that file.
    """
    return coeff / cos(sweep_deg if faithful else sweep_deg * pi / 180) ** 2


# ---------------------------------------------------------------------------
# 2. M_4a inconsistent with the stagnation factor computed from it
# ---------------------------------------------------------------------------

#: The station-4a Mach number the subs decks intend.
M_4A = 0.2
#: What the model actually carries, being the Combustor's declared default.
M_4A_SOURCE = 0.1025


def correct_M_4a(p, *, faithful: bool = False):
    """Make ``M_4a`` agree with the ``hold_4a`` computed from it.

    ``hold_{4a}`` is by definition ``1 + (gamma-1)/2 * M_4a**2`` -- the
    stagnation correction at station 4a, and nothing else.
    ``subs/optimalD8.py`` opens with a local ``M4a = .2`` and sets::

        'hold_{4a}': 1.+.5*(1.313-1.)*M4a**2.

    and then never substitutes ``M_{4a}``, which therefore keeps the
    Combustor's declared default of 0.1025. So the model carries a stagnation
    factor derived from Mach 0.2 alongside a Mach number of 0.1025, and

        u_4a == M_4a * sqrt(1.313 R T_t4) / hold_4a

    reads the second. The cooling-flow velocity comes out at 0.51 of what the
    same file's own arithmetic says it should be.

    The two self-consistent readings are ``M_4a = 0.2`` and ``M_4a = 0.1025``;
    this takes 0.2, because that is the number the subs author wrote down and
    derived from. Set ``faithful=True`` to leave the inconsistency in place.

    Applied to the built engine rather than to ``turbofan/model.py``'s tables,
    so the engine module stays a transcription of its source and every deck
    that wants the correction asks for it in its own file.
    """
    if not faithful:
        set_constant(p.eng, "M_4a", M_4A)
    return []
