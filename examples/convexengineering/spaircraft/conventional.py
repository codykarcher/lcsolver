"""SPaircraft ``optimal737``: the conventional tube-and-wing baseline.

A TASOPT Boeing 737-800 -- single-bubble fuselage, six-abreast with one aisle,
two engines podded under a 26-degree swept wing, one fin, a conventional
horizontal tail on the tail cone -- flying the same mission as the D8.2: 3000
nm with 180 passengers. It is the aeroplane the D8 is measured against, and
the reason the D8 numbers mean anything.

Validation: read this first
---------------------------
**There is no gpkit reference for this configuration, and one cannot be
produced from the source.** ``optimal737`` diverges: the sequential-GP loop
runs to a degenerate near-zero-fuel point with variables pinned at gpkit's own
``Bounded`` limit of 1e30, while PCCP reports 4-5% slack on the signomial
constraints. Raising ``pccp_penalty``, warm-starting from the converged D8,
and every combination of ``fixedBPR``/``pRatOpt`` leave it unchanged.
``optimal777``, ``M072_737`` and ``D8_eng_wing`` do the same. SPaircraft's own
``TESTS`` manifest drives only ``optimalD8``, so no other configuration is
covered by the repository's CI either. See ``reference.py`` and
DISCREPANCIES.md.

So this deck is checked two ways instead:

1. **Feasibility.** Every constraint satisfied, the mission closed, nothing
   resting on the 1e-30/1e30 box. That is a real check -- it is precisely what
   the diverging gpkit run fails.
2. **TASOPT.** ``percent_diff.py`` carries the TASOPT 737-800 weights and
   areas that SPaircraft itself compares ``optimal737`` against, and
   :data:`TASOPT` reproduces them. This is a comparison against a *different
   tool*, not a reproduction of a recorded answer: agreement of a few percent
   is the most it can show, and a disagreement could be either model's.

Do not read a number out of this file as validated the way ``model.py``'s D8
is. It is a faithful transcription of a configuration whose reference
implementation does not converge.

Configuration
-------------
``geometryFlags.py`` gives ``optimal737`` just ``conventional = True``, which
in ``aircraft.py`` expands to ``wingengine = True; tube = True`` and engine 3
-- the same TASOPT D8.2 core as the D8, without boundary layer ingestion. The
three structural consequences are :func:`~.layouts.wing_engines`,
:func:`~.layouts.tube_floor` and :func:`~.layouts.conventional_tail`.

Everything else is the number list in :func:`build`, transcribed from
``subs/optimal737.py``.
"""
from __future__ import annotations

from pathlib import Path

from numpy import cos, pi
from pyomo.environ import units

from edi import Formulation

from . import airframe, layouts
from .airframe import pin, set_constant
from .model import MAX_ITER, _bound_constraints, _bound_variables

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turbofan.no_bli import NAME as ENGINE  # noqa: E402  (registers the entry)

# Geometry, from subs/optimal737.py.
SWEEP_W, SWEEP_VT, SWEEP_HT = 26.0, 25.0, 25.0
NCLIMB, NCRUISE = 3, 2

#: Max cruise turbine inlet temperature, substituted at 1125 K. aircraft.py
#: declares it at 3000 K, which is why optimalD8 -- whose subs deck is silent
#: -- never feels it.
CRUISE_TT41_MAX = 1125.0

#: Spanwise engine location, subs/optimal737.py ``y_{eng}``. On a rear-engined
#: aeroplane this is derived from the fuselage width; here it is simply given,
#: because where you hang a podded engine is a layout decision rather than a
#: consequence of anything else in the model.
Y_ENG = 4.8768                                          # m, = 16 ft

LBF_N = 4.44822161526
M2_FT2 = 10.7639104
M_FT = 3.28083990


def build(Nclimb: int = NCLIMB, Ncruise: int = NCRUISE,
          seed: str | None = None) -> Formulation:
    """Build the 737-800. Returns an EDI ``Formulation``."""
    N = Nclimb + Ncruise
    f = Formulation()

    p, cons = airframe.add_components(
        f, N, sweep_w=SWEEP_W, sweep_vt=SWEEP_VT, sweep_ht=SWEEP_HT,
        engine=ENGINE, BLI=False)

    K = airframe.add_constants(
        f,
        n_vt=1.0,                    # one fin
        n_aisle=1.0,                 # single aisle
        SM_min=0.15,
        dx_CG=7.68,                  # ft
        M_min=0.8,
        f_L_total_wing=1.127,
        f_wingfuel=0.5,
        r_S_nacelle=16.0,            # podded nacelle
        r_v_nacelle=1.02,
        f_pylon=0.1,
        D_reduct=1.0,                # no ingestion, so no drag credit
        b_max=117.5 * 0.3048,        # m, gate span limit
        C_L_w_max=2.25 / cos(SWEEP_W * pi / 180) ** 2,
        Fsafetyfac=1.8,
        MinCruiseAlt=35000.0,        # ft
        MaxClimbTime=14.0,           # min
        # 0.0001 s^-2 is effectively no yaw-rate requirement at flare. The D8
        # asks for 0.1475; the source's comment on this line reads "NOTE:
        # Constraint inactive", so the fin is sized by engine-out and by
        # V_vt_min instead.
        rdot_req=0.0001,
        C_D_fuse=0.01107365,         # tube, from the conventional branch
    )

    v = airframe.add_variables(f, N)
    f.Objective(v.W_ftotal)

    cons += airframe.add_shared(f, N, Nclimb, p, v, K)

    # ---- structure: wing engines, tube fuselage, conventional tail ----------
    cons += layouts.wing_engines(p, v, K, sweep_w=SWEEP_W)
    cons += layouts.tube_floor(p)
    cons += layouts.conventional_tail(p, v, K, sweep_vt=SWEEP_VT,
                                      sweep_ht=SWEEP_HT)

    # Cruise turbine inlet temperature limit.
    cons += [p.eng.T_t_41[Nclimb:] <= CRUISE_TT41_MAX * units.K]

    # ---- component constants declared for the D8 ---------------------------
    # fuselage.py and horizontal_tail.py hard-code these to double-bubble
    # values. They are mutable Pyomo Params, so a deck retunes them in place
    # rather than adding a constraint to fight the default.
    set_constant(p.fu, "SPR", 6.0)          # six abreast, not eight
    set_constant(p.fu, "M_fuseD", 0.80)     # tube drag reference Mach
    set_constant(p.ht, "eta_ht", 0.9)       # tail on the cone, in wing wake
    set_constant(p.wing, "tau_max_w", 0.1267)
    set_constant(p.wing, "f_slat", 0.1)     # a conventional wing has slats

    # ---- substitutions -------------------------------------------------------
    for handle, name, value, unit in [
        (p.fu, "n_pass", 180.0, None),
        (p.fu, "W_cargo", 0.1, units.N),
        (p.fu, "l_nose", 20.0, units.ft),
        (p.fu, "h_floor", 5.0, units.inch),
        (p.fu, "lambda_cone", 0.3, None),
        # A circular barrel. fuselage.py carries the double-bubble geometry
        # unconditionally -- theta_db == w_db/R_fuse, and the section areas
        # and bending inertias are written in terms of both -- so a tube is
        # made by collapsing the joining angle and the straight section to
        # zero rather than by leaving them out. 1e-4 rather than 0 because
        # every quantity in a geometric program is strictly positive; this is
        # what subs/optimal737.py does.
        (p.fu, "theta_db", 0.0001, None),
        (p.fu, "dR_fuse", 0.0001, units.m),
        (p.vt, "A_vt", 2.0, None),
        (p.vt, "V_1", 70.0, units.m / units.s),
        (p.vt, "c_l_vt_EO", 0.5, None),
        (p.vt, "e_vt", 0.8, None),
        (p.vt, "lambda_vt", 0.3, None),
        (p.vt, "rho_TO", 1.225, units.kg / units.m ** 3),
        (p.ht, "lambda_ht", 0.25, None),
        (p.ht, "C_L_ht_fCG", 0.7, None),
        (p.lg, "z_CG", 2.0, units.m),
        (p.lg, "z_wing", 0.5, units.m),
        (p.lg, "h_hold", 1.0, units.m),
        (p.lg, "t_nacelle", 0.15, units.m),
    ]:
        cons.append(pin(handle, name, value, unit))

    # Engines 16 ft out on each wing.
    cons += [v.y_eng == Y_ENG * units.m]

    cons += _bound_constraints(f)
    f.ConstraintList(cons)
    _bound_variables(f)
    return f


#: TASOPT 737-800, from ``percent_diff.py`` under ``if aircraft ==
#: 'optimal737'``. These are the numbers SPaircraft itself reports this
#: configuration against. ``key -> (rebuilt name, scale to TASOPT's unit,
#: TASOPT value)``; scale converts the declared unit to the reported one.
TASOPT = [
    ("fuel_lbf",        "W_f_total",  1.0,            45057.0),
    ("takeoff_wt_lbf",  "W_total",    1.0,           166502.0),
    ("payload_lbf",     "Fuse_W_payload", 1.0,        38715.5),
    ("fuse_wt_lbf",     "Fuse_W_fuse", 1.0,           35641.6),
    ("wing_wt_lbf",     "Wing_W_wing", 1 / LBF_N,     21691.8),
    ("vt_wt_lbf",       "VT_W_vt",     1 / LBF_N,      1477.2),
    ("ht_wt_lbf",       "HT_W_ht",     1 / LBF_N,      1463.8),
    ("span_ft",         "Wing_b",      M_FT,           113.586),
    ("wing_area_ft2",   "Wing_S",      M2_FT2,        1277.41),
    ("wing_AR",         "Wing_AR",     1.0,             10.1),
    ("ht_area_ft2",     "HT_S_ht",     M2_FT2,         313.86),
    ("ht_AR",           "HT_AR_ht",    1.0,              6.0),
    ("V_ht",            "HT_V_ht",     1.0,              1.450),
    ("vt_span_ft",      "VT_b_vt",     M_FT,            22.99),
    ("vt_area_ft2",     "VT_S_vt",     M2_FT2,         264.29),
    ("A_vt",            "VT_A_vt",     1.0,              2.0),
]


def verify(max_iter=MAX_ITER, **kw):
    """Solve, then report feasibility and the TASOPT comparison.

    Feasibility is the check that carries weight here; see the module
    docstring on why there is no gpkit column.
    """
    from harness import solve_edi, feasibility, solution_dict

    fm = build(**kw)
    solve_edi(fm, solver="ipopt-convex", max_iter=max_iter)
    sol = solution_dict(fm)

    # Engine system weight is per engine in the model and reported for the
    # pair by TASOPT, so it does not fit the table's one-name-one-scale shape.
    rows = [(key, sol[name] * scale, tas) for key, name, scale, tas in TASOPT]
    rows.append(("engsys_wt_lbf", 2.0 * sol["W_engsys"], 11632.6))
    rows = [(k, got, tas, abs(got - tas) / abs(tas)) for k, got, tas in rows]
    return rows, feasibility(fm)


if __name__ == "__main__":
    rows, (nv, worst, where) = verify()
    print("\nSPaircraft optimal737 (TASOPT 737-800)")
    print("no gpkit reference exists for this configuration -- see the module "
          "docstring")
    print(f"\n{'quantity':18} {'rebuilt':>12} {'TASOPT':>12} {'rel':>9}")
    for key, got, tas, rel in rows:
        print(f"{key:18} {got:12.6g} {tas:12.6g} {rel:9.2e}")
    print(f"\nfeasibility: {nv} violated, worst rel {worst:.2e}"
          + (f" at {where}" if where else ""))
