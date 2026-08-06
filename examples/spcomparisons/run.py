"""Solve the 4 x 5 matrix, one process per case.

Why processes and not threads
-----------------------------
Each solve is a long IPOPT call holding its own Pyomo model. They share
nothing, so processes give real parallelism where threads would serialise on
the GIL and on IPOPT's non-reentrant internals.

Seeding
-------
Every case starts from ``reference.json``, the D8.2 gpkit optimum. That is a
good seed for the 737 classes and a progressively worse one as the geometry
moves away, which is the main reason a case fails to converge rather than
being genuinely infeasible. ``--continuation`` re-runs failures warm-started
from the nearest solved neighbour, which is what rescued the fuel-cell range
sweep earlier: a cold phase-1 failure is not evidence of infeasibility.
"""
from __future__ import annotations

import argparse, json, os, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def _solve_one(args):
    cls_key, arch_key, maxit = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "components"))
    sys.path.insert(0, "/Users/codykarcher/Dropbox/research/lcsolver")
    import pyomo.environ as pyo
    from pyomo.environ import units as u
    import classes, architectures, aircraft
    from lcsolver_compat import structure_detector
    from lcsolver.solvers.ipopt.slcp_bridge import solve_sia
    from lcsolver.solvers.ipopt.sia import SIAOptions
    from lcsolver_compat import unit_corrector

    LBF = 0.2248089431
    cl, ar = classes.CLASSES[cls_key], architectures.ARCHS[arch_key]
    rec = {"class": cls_key, "arch": arch_key, "converged": False}
    t0 = time.time()
    try:
        cm = unit_corrector(aircraft.build(cl, ar, seed="reference"))
        st = structure_detector(cm)
        rec["n_var"] = len(st["variables"])
        _o = SIAOptions(max_iterations=maxit)
        # 1e-06 is an unreasonable ask of this model, and not because the
        # design is unconverged. The fuselage bending block saturates: with the
        # shell carrying the loads unaided, x_hbend runs past the tail (39.4 m
        # on an aircraft whose tail is at 23.8 m) and the reinforcement areas
        # go to ~1e-04 m^2. That block is then flat, its gradient is noise, and
        # the last digits never settle -- on the E175 the objective is stable
        # to eight figures by iteration 75 while stationarity spends 325 more
        # iterations going from 2e-04 to 6e-06, all of it in twelve Fuse_*
        # variables. res.report names them, so a run that stops short says why.
        _o.stationarity_tolerance = 1e-5
        # The SUBPROBLEM tolerance, distinct from the outer KKT tolerances
        # above, and load-bearing: at ipopt's default 1e-12 the linearized
        # equality cluster around the free-sweep identity is declared
        # infeasible at points whose true residuals are 1e-9, and the
        # free-sweep D8 dies mid-run ("the SIA sub-problem failed:
        # infeasible"). At 1e-9 the same cold start converges to the same
        # optimum a warm start reaches, to 7 significant figures.
        _o.ipopt_options = dict(_o.ipopt_options, tol=1e-9,
                                constr_viol_tol=1e-9)
        r = solve_sia(st, options=_o, presolve=False)
        rec["status"] = str(r.status)[:120]
        rec["report"] = getattr(r, "report", None)
        rec["stationarity"] = float(getattr(r, "stationarity", float("nan")))
        rec["max_violation"] = float(getattr(r, "max_violation", float("nan")))
        rec["phase1_iterations"] = int(getattr(r, "phase1_iterations", 0))
        rec["phase1_mode"] = getattr(r, "phase1_mode", None)
        rec["iterations"] = int(getattr(r, "iterations", 0))
        if r.converged:
            for v, val in zip(st["variables"], r.x):
                v.set_value(float(val))
            vals = {}
            for v in cm.component_data_objects(pyo.Var):
                val = pyo.value(v); un = str(u.get_units(v))
                vals[v.name] = val * LBF if un == "N" else val
            rec["values"] = vals
            rec["converged"] = True
    except Exception as e:                      # a build or solver blow-up is
        rec["status"] = f"{type(e).__name__}: {e}"[:200]   # data, not a crash
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--maxit", type=int, default=2000)
    ap.add_argument("--only", default="", help="comma'd class:arch filters")
    a = ap.parse_args()

    sys.path.insert(0, str(HERE))
    import classes, architectures
    # ORDER_ALL, not ORDER: the Mach-locked twins live only in ORDER_LOCKED,
    # so building the job list from ORDER silently matched nothing and the
    # run reported "0 cases on 3 workers" rather than failing.
    jobs = [(c, r, a.maxit) for c in classes.ORDER
            for r in architectures.ORDER_ALL]
    if a.only:
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if f"{j[0]}:{j[1]}" in keep]

    OUT.mkdir(exist_ok=True)
    print(f"{len(jobs)} cases on {a.workers} workers\n", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_solve_one, j): j for j in jobs}
        for fut in as_completed(futs):
            rec = fut.result(); done += 1
            tag = f"{rec['class']}/{rec['arch']}"
            if rec["converged"]:
                v = rec["values"]
                mtow = v.get("W_total") or v.get("W_MTO")
                print(f"[{done:2d}/{len(jobs)}] {tag:22s} OK   MTOW {mtow:9,.0f} lb"
                      f"  fuel {v.get('W_f_total', float('nan')):8,.0f}"
                      f"  {rec['iterations']:4d} it  {rec['wall_s']:6.1f}s", flush=True)
            else:
                print(f"[{done:2d}/{len(jobs)}] {tag:22s} FAIL {rec['status'][:60]}"
                      f"  {rec['wall_s']:6.1f}s", flush=True)
            (OUT / f"{rec['class']}__{rec['arch']}.json").write_text(
                json.dumps(rec, indent=1))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
