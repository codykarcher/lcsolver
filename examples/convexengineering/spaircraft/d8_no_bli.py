"""SPaircraft ``D8_no_BLI``: the D8.2 airframe with a clean inlet.

Same aeroplane as ``model.py`` -- rear engines, pi-tail, double-bubble
fuselage, 3000 nm with 180 passengers -- with boundary layer ingestion taken
away. It is the control case for the D8's headline claim: everything else
about the configuration is held fixed, so the fuel difference against
``optimalD8`` is what ingestion is worth.

Why this deck exists
--------------------
It has a gpkit reference. ``reference.json`` has carried a ``D8_no_BLI`` entry
all along -- 25033.1 lbf of fuel, 132949.7 takeoff, 69216.6 dry -- recorded
by ``reference.py``, which solves both configurations. Nothing built against
it. It is the second of the two SPaircraft configurations that converge in
gpkit (see ``reference.py`` and DISCREPANCIES.md), so it is the only other
case where "does the rebuild agree" is a question with an answer.

What changes, against ``model.py``
----------------------------------
Structurally, nothing except where the engines sit spanwise. With ingestion
the engines are buried against the fuselage and ``y_eng`` is exactly half the
fuselage width; without it they are podded, and have to clear the side of the
fuselage by their own radius. Everything else is a number:

=======================  ==========  ==========  ==========================
quantity                 optimalD8   D8_no_BLI   why
=======================  ==========  ==========  ==========================
``D_reduct``             0.98416     1.0         no wake ingestion, so no
                                                 drag credit
``f_pylon``              0.05        0.11        podded engines need real
                                                 pylons
``r_S_nacelle``          6.0         16.0        a podded nacelle is a
                                                 nacelle, not a fuselage
                                                 fairing
``r_v_nacelle``          0.925       1.0         inlet sees free stream
``h_f``                  43.003      42.5        (subs deck says so)
``C_p_c``                1257.9      1253.9      (subs deck says so)
``T_t41`` cruise cap     none        1125 K      (subs deck says so)
``C_L_w_max``            2.269       3.503       a bug -- see below
=======================  ==========  ==========  ==========================

plus the engine's on-design mass-flow anchors, which are the ``else`` branch
of ``if eng == 3`` -- see :mod:`turbofan.no_bli`.

The ``C_L_w_max`` bug
---------------------
``subs/D8_no_BLI.py`` line 76 reads ``2.15/(cos(sweep)**2)`` where every other
subs deck reads ``2.15/(cos(sweep * pi / 180.)**2)``. ``sweep`` is 13.237
*degrees*, so the missing conversion feeds 13.237 radians to ``cos`` -- which
wraps twice round and lands at 0.784 instead of 0.973. The maximum wing lift
coefficient comes out 3.503 rather than 2.269, a 54% error in a limit that
sizes the wing for low speed.

It is reproduced here, deliberately, because the recorded gpkit reference was
produced by that file. Correcting it would be a different aeroplane and the
reference would no longer apply. ``build(cl_max_bug=False)`` uses the intended
2.269 for anyone who wants to see what it costs.
"""
from __future__ import annotations

from pathlib import Path

from numpy import cos, pi, tan
from pyomo.environ import units

from edi import Formulation

from . import airframe, layouts
from .airframe import pin
# SPR (8 seats/row) and M_fuseD (0.72) are already the double-bubble values in
# fuselage.py, so unlike the 737 deck this one has no component constants to
# retune.
from .model import CHECKS, MAX_ITER, _bound_constraints, _bound_variables

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turbofan.no_bli import NAME as ENGINE  # noqa: E402  (registers the entry)

# Geometry, from subs/D8_no_BLI.py.
SWEEP_W, SWEEP_VT, SWEEP_HT = 13.237, 25.0, 8.0
NCLIMB, NCRUISE = 3, 2

#: Max cruise turbine inlet temperature. aircraft.py declares this at 3000 K
#: -- effectively no limit, which is why optimalD8, whose subs deck is silent,
#: never feels it -- and constrains T_t41 in the cruise segments against it.
#: subs/D8_no_BLI.py substitutes 1125 K, which does bind.
CRUISE_TT41_MAX = 1125.0


def build(Nclimb: int = NCLIMB, Ncruise: int = NCRUISE,
          pi_tail_supports: str = "pinned", cl_max_bug: bool = True,
          seed: str | None = None) -> Formulation:
    """Build the non-ingesting D8. Returns an EDI ``Formulation``."""
    N = Nclimb + Ncruise
    f = Formulation()

    p, cons = airframe.add_components(
        f, N, sweep_w=SWEEP_W, sweep_vt=SWEEP_VT, sweep_ht=SWEEP_HT,
        engine=ENGINE, BLI=False)

    K = airframe.add_constants(
        f,
        # --- shared with optimalD8 ---------------------------------------
        n_vt=2.0, n_aisle=2.0,
        SM_min=0.05, dx_CG=6.0, M_min=0.72,
        f_L_total_wing=1.195, f_wingfuel=1.0,
        b_max=140.0 * 0.3048,
        Fsafetyfac=1.0, MinCruiseAlt=38478.0, MaxClimbTime=16.0,
        rdot_req=0.1475, C_D_fuse=0.018081,
        C_L_w_max=(2.15 / cos(SWEEP_W) ** 2 if cl_max_bug
                   else 2.15 / cos(SWEEP_W * pi / 180) ** 2),
        # --- the clean-inlet numbers ---------------------------------------
        D_reduct=1.0,
        f_pylon=0.11,
        r_S_nacelle=16.0,
        r_v_nacelle=1.0,
    )

    v = airframe.add_variables(f, N)
    f.Objective(v.W_ftotal)

    cons += airframe.add_shared(f, N, Nclimb, p, v, K)

    # ---- structure: rear engines, double bubble, pi-tail --------------------
    cons += layouts.rear_engines(p, v, K)
    cons += layouts.double_bubble_floor(p)
    cons += layouts.pi_tail(f, p, v, K, supports=pi_tail_supports)

    # Podded engines beside the aft fuselage, not buried in it. optimalD8 has
    # y_eng == 0.5*w_fuse because a BLI engine sits on the centreline; here
    # the nacelle has to clear the fuselage side by its own radius, with a
    # foot of gap.
    cons += [v.y_eng >= p.fu.w_fuse + 0.5 * p.eng.d_f + 1.0 * units.ft]

    # Pi-tail trailing edge: limited by the fin sweep, not by the fuselage.
    cons += [
        p.ht.dx_trail_ht <= (p.vt.dx_lead_vt
                             + p.vt.b_vt / tan(SWEEP_VT * pi / 180)
                             + p.fu.w_fuse / tan(SWEEP_HT * pi / 180)
                             + p.ht.c_root_ht),
    ]

    # Cruise turbine inlet temperature limit.
    cons += [p.eng.T_t_41[Nclimb:] <= CRUISE_TT41_MAX * units.K]

    # ---- substitutions -------------------------------------------------------
    # Identical to optimalD8's: the subs decks differ only in the numbers
    # tabulated in the module docstring.
    for handle, name, value, unit in [
        (p.fu, "n_pass", 180.0, None),
        (p.fu, "W_cargo", 0.1, units.N),
        (p.fu, "l_nose", 29.0, units.ft),
        (p.fu, "h_floor", 5.12, units.inch),
        (p.fu, "w_db", 0.93, units.m),
        (p.fu, "lambda_cone", 0.3, None),
        (p.vt, "A_vt", 2.2, None),
        (p.vt, "V_1", 70.0, units.m / units.s),
        (p.vt, "c_l_vt_EO", 0.5, None),
        (p.vt, "e_vt", 0.8, None),
        (p.vt, "lambda_vt", 0.3, None),
        (p.vt, "rho_TO", 1.225, units.kg / units.m ** 3),
        (p.ht, "lambda_ht", 0.3, None),
        (p.ht, "C_L_ht_fCG", 0.85, None),
        (p.lg, "z_CG", 2.0, units.m),
        (p.lg, "z_wing", 0.5, units.m),
        (p.lg, "h_hold", 1.0, units.m),
        (p.lg, "t_nacelle", 0.15, units.m),
    ]:
        cons.append(pin(handle, name, value, unit))

    cons += _bound_constraints(f)
    f.ConstraintList(cons)
    _bound_variables(f)
    if seed == "reference":
        from .model import _seed_from_reference
        _seed_from_reference(f)
    return f


def verify(rtol=0.02, max_iter=MAX_ITER, **kw):
    """Solve and diff the headline quantities against the gpkit reference."""
    from harness import solve_edi, feasibility, load_reference, solution_dict

    fm = build(**kw)
    solve_edi(fm, solver="ipopt-convex", max_iter=max_iter)
    sol = solution_dict(fm)
    chk = load_reference(Path(__file__).with_name("reference.json"))
    chk = chk["D8_no_BLI"]["checked"]

    rows = []
    for key, name, scale in CHECKS:
        got = sol[name] * scale
        exp = chk[key]
        rows.append((key, got, exp, abs(got - exp) / abs(exp)))
    return rows, feasibility(fm)


if __name__ == "__main__":
    rows, (nv, worst, where) = verify()
    print("\nSPaircraft D8, no BLI")
    print(f"{'quantity':22} {'rebuilt':>12} {'gpkit':>12} {'rel':>9}")
    worst_rel = 0.0
    for key, got, exp, rel in rows:
        worst_rel = max(worst_rel, rel)
        print(f"{key:22} {got:12.6g} {exp:12.6g} {rel:9.2e}")
    print(f"worst relative difference vs gpkit: {worst_rel:.2e}")
    print(f"feasibility: {nv} violated, worst rel {worst:.2e}"
          + (f" at {where}" if where else ""))
