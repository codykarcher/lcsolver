"""Record the gpkit SPaircraft solution as reference.json.

Requires gpkit plus checkouts of https://github.com/convexengineering/SPaircraft,
.../turbofan (SPaircraft imports it for the engine) and .../gplibrary::

    SPAIRCRAFT=~/src/SPaircraft TURBOFAN=~/src/turbofan GPLIBRARY=~/src/gplibrary \
        python reference.py

Three things about this reference are worth knowing before trusting a number
out of it.

1. cvxopt cannot solve this model.  The D8 compiles to 1198 free variables in
   4698 posynomial inequalities; cvxopt forms a dense KKT system and fails,
   first with ``Rank(A) < p`` (its linear-equality block, built from gpkit's
   monomial equalities, is rank deficient) and then, with ``use_leqs=False``,
   by simply not converging.  This is not a surprise to the authors:
   SPaircraft's own ``TESTCONFIG`` reads ``skipsolvers : cvxopt``.  The
   published results came from MOSEK.  Lacking a MOSEK license, the GP
   subproblems here are solved with IPOPT via ``gpkit_ipopt`` -- same gpkit
   model, different numerics.

2. The shipped convergence tolerance is far too loose.  ``optimize_aircraft``
   calls ``localsolve(..., reltol=0.01)``, which stops the sequential-GP loop
   once two successive costs are within 1% of each other.  That happens well
   before convergence: repeated runs of the identical model land anywhere in
   21.6k-23.3k lbf and differ from *each other* by up to 6%, against a
   converged value of 20.86k.  At ``reltol <= 1e-4`` the loop settles and
   reproduces to ~1e-5, so that is what is recorded.

3. Only one configuration converges.  ``optimalD8`` (the paper's D8.2) solves
   cleanly, and its solution satisfies every constraint of the unmodified
   gpkit model to 4e-8.  ``D8_no_BLI`` also solves.  ``optimal737``,
   ``optimal777``, ``M072_737`` and ``D8_eng_wing`` all diverge to a
   degenerate near-zero-fuel point with variables pinned at gpkit's own
   ``Bounded`` limit of 1e30, and PCCP reports 4-5% slack on the signomial
   constraints.  Raising ``pccp_penalty``, warm-starting from the converged
   D8, and varying ``fixedBPR``/``pRatOpt`` all leave it unchanged.  Note that
   SPaircraft's ``TESTS`` manifest lists only ``SPaircraft.py``, whose
   ``test()`` drives ``optimalD8`` -- so no other configuration is covered by
   the repo's CI.

See DISCREPANCIES.md for how the converged D8.2 compares with Table 3 of
York et al., "Efficient Aircraft Multidisciplinary Design Optimization and
Sensitivity Analysis via Signomial Programming" (DOI 10.2514/1.J057020).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import save_reference  # noqa: E402

NCLIMB, NCRUISE, NMISSION = 3, 2, 1
PRAT_KEYS = ["\\pi_{f_D}", "\\pi_{lc_D}", "\\pi_{hc_D}"]

# Configurations that converge, with their Table 1 mission.
CONFIGS = {
    "optimalD8": (3000.0, 180.0),
    "D8_no_BLI": (3000.0, 180.0),
}

# York et al. Table 3, SP column, for the D8.2.
PAPER_SP_D82 = {"takeoff_weight_lbf": 143421.0, "fuel_lbf": 27529.0,
                "empty_weight_lbf": 77129.0, "span_ft": 140.0}


def main() -> None:
    checkouts = ("SPAIRCRAFT", "TURBOFAN", "GPLIBRARY")
    for var in checkouts:
        if not os.environ.get(var):
            raise SystemExit(f"set {var} to a checkout")
    # SPaircraft MUST precede turbofan on sys.path. Both ship a top-level
    # stand_alone_simple_profile.py, and aircraft.py does a bare
    # `from stand_alone_simple_profile import FlightState`. With turbofan
    # first you silently get *its* FlightState, whose Atmosphere leaves g a
    # free variable rather than a constant; the model still solves, and still
    # looks converged, but burns 20391.6 lbf instead of 20859.7 -- a 2.2%
    # error with no warning anywhere. (simple_ac_imports.py collides too.)
    # Running from inside the SPaircraft checkout hides this, because cwd
    # wins; running from anywhere else does not.
    for var in reversed(checkouts):
        sys.path.insert(0, os.environ[var])

    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import gpkit_ipopt
    from gpkit import Model, units
    from gpkit.constraints.bounded import Bounded
    from aircraft import Mission

    out = {}
    for config, (rng_nmi, n_pass) in CONFIGS.items():
        mod, fn = f"subs.{config}", f"get_{config}_subs"
        get_subs = getattr(__import__(mod, fromlist=[fn]), fn)

        m = Mission(NCLIMB, NCRUISE, config, NMISSION)
        m.cost = m["W_{f_{total}}"].sum()
        subs = get_subs()
        subs.update({"R_{req}": rng_nmi * units("nmi"), "n_{pass}": n_pass})
        for k in PRAT_KEYS:          # pRatOpt=True, as in SPaircraft.test()
            subs.pop(k, None)
        m.substitutions.update(subs)

        gpkit_ipopt.reset_cache()
        sol = Model(m.cost, Bounded(m), m.substitutions).localsolve(
            verbosity=0, iteration_limit=500, reltol=1e-6,
            solver=gpkit_ipopt.optimize)

        def scalar(name):
            try:
                v = sol(name)
                v = v.magnitude if hasattr(v, "magnitude") else v
                return float(np.atleast_1d(np.asarray(v, float)).ravel()[0])
            except Exception:
                return float("nan")

        checked = {
            "fuel_lbf": float(sol["cost"]),
            "takeoff_weight_lbf": scalar("W_{total}"),
            "dry_weight_lbf": scalar("W_{dry}"),
            "span_ft": scalar("b") * 3.28084,
        }

        values = {}
        for key, val in sol["variables"].items():
            try:
                values[str(key)] = float(val)
            except (TypeError, ValueError):
                try:
                    values[str(key)] = [float(x)
                                        for x in np.atleast_1d(val).ravel()]
                except Exception:
                    pass
        out[config] = {"checked": checked, "values": values}
        if config == "optimalD8":
            out[config]["paper_sp"] = PAPER_SP_D82
        print(f"{config}: fuel={checked['fuel_lbf']:.1f} lbf, "
              f"{len(values)} values")

    save_reference(Path(__file__).with_name("reference.json"), out,
                   source="convexengineering/SPaircraft @ master, "
                          "solved with gpkit_ipopt (see module docstring)",
                   meta={"reltol": 1e-6, "Nclimb": NCLIMB,
                         "Ncruise": NCRUISE, "Nmission": NMISSION,
                         "pRatOpt": True, "fixedBPR": False})


if __name__ == "__main__":
    main()
