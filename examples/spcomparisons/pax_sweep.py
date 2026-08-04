"""Sweep passenger count x architecture, for the market-comparison plot.

Runs every architecture at pax = 100..500 in steps of 100 and records MTOW
(and the energy quantity), to overlay the SP models on the published-fleet
passengers-vs-MTOW scatter. Parallelised with MPI, one case per rank slot:

    mpirun -n 6 python pax_sweep.py

Single-process (no mpirun) also works; it just runs the cases serially.
Results land in results_pax_sweep/{arch}_{pax}.json -- the standard flat
variable dump (weights in lbf) plus a "_case" header.

ROUGHNESS, declared up front (the point is a trend line, not 25 designs):

* Classes are SYNTHESISED from the calibrated four by nearest-template:
  <=150 pax on the E175 template (5-abreast, CFM56 deck, cfm56_era),
  <=250 on the 737-800 template (6-abreast, TASOPT_737800, cfm56_era),
  above that on the 787-8 template (9-abreast twin-aisle, GE90, composite,
  modern_composite tech, high-speed mses_e polar; 10-abreast at 450+).
  The tech jump between 250 and 300 pax is deliberate -- the real fleet has
  the same jump (the widebodies ARE the newer-technology aircraft here).
* Design range follows the market trend, so the conventional column should
  overlay the fleet: 2500 / 3200 / 7000 / 7600 / 8000 nmi at 100..500 pax.
* BATTERY flies 200 nmi at every size -- at fleet ranges it does not close
  at any pax count (physics, not numerics), so its curve answers "what does
  a battery aircraft of this capacity weigh on a mission it can actually
  fly", and is labelled accordingly on the plot.
* The D8 keeps the conventional cabin at each size (not the canonical
  8-abreast anchor cabin) so airframe is the only change within a column.
* Multi-mission corners, per-class tail/fuselage floors and the D8-era tech
  deck are all OFF -- one mission, one tech level per size.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import traceback

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
os.chdir(D)
# GEAR_BOX_FRAC is read at aircraft-import time; the rest are read per build.
os.environ.setdefault("GEAR_BOX_FRAC", "1.00")
os.environ.setdefault("V_HT_FLOOR", "0.01")

import architectures  # noqa: E402
import classes  # noqa: E402
import aircraft  # noqa: E402
from edi_compat import structure_detector, unit_corrector  # noqa: E402
from edi.solvers.ipopt.slcp_bridge import solve_sia  # noqa: E402
from edi.solvers.ipopt.sia import SIAOptions  # noqa: E402

PAX = [100, 200, 300, 400, 500]
ARCHS = ["conventional", "d8", "h2burn", "h2fc", "battery"]
RANGE_NMI = {100: 2500, 200: 3200, 300: 7000, 400: 7600, 500: 8000}
BATTERY_RANGE_NMI = 200.0
OUTDIR = os.path.join(D, "results_pax_sweep")


def size_class_for(pax: int):
    """Synthesise a SizeClass and (tech, polar) for a passenger count."""
    if pax <= 150:
        t, sa, na, tech, polar = classes.CLASSES["e175"], 5, 1, "cfm56_era", None
    elif pax <= 250:
        t, sa, na, tech, polar = classes.CLASSES["b737"], 6, 1, "cfm56_era", None
    else:
        t, sa, na, tech, polar = (classes.CLASSES["b787"], 9, 2,
                                  "modern_composite", "mses_e")
        if pax >= 400:
            sa = 10   # 777-9 is 10-abreast at 426; 9-abreast at 400 built a
                      # fuselage past what the shell model closes on
    cl = dataclasses.replace(
        t, n_pass=float(pax), seats_abreast=float(sa), n_aisles=float(na),
        range_nmi=float(RANGE_NMI[pax]),
        missions=None, v_ht_min=None, R_fuse_min=None,
        # FAR 25.121 gradient requirements exist for 2/3/4 engines only --
        # the 787 template's 6 electric fans have no certification row to
        # size against. Four is the most the ruleset (and the rule) knows.
        n_fans=min(int(t.n_fans), 4))
    return cl, tech, polar


def run_case(arch_key: str, pax: int) -> dict:
    cl, tech, polar = size_class_for(pax)
    if arch_key == "battery":
        cl = dataclasses.replace(cl, range_nmi=BATTERY_RANGE_NMI)
    os.environ["TECH"] = tech
    ar = dataclasses.replace(architectures.ARCHS[arch_key], lock_mach=True)
    kw = {"polar": polar} if polar else {}
    st = structure_detector(unit_corrector(
        aircraft.build(cl, ar, seed="reference", **kw)))
    o = SIAOptions(max_iterations=200)
    o.stationarity_tolerance = 1e-5
    o.condense_numerator = True
    o.ipopt_options = dict(o.ipopt_options, tol=1e-9, constr_viol_tol=1e-9)
    r = solve_sia(st, options=o, presolve=False)
    names = [v.name for v in st["variables"]]
    U = {v.name: (str(v.get_units()) if v.get_units() is not None else "")
         for v in st["variables"]}
    x = np.asarray(r.x, float)
    d = {"_case": {"arch": arch_key, "pax": pax, "tech": tech,
                   "range_nmi": cl.range_nmi,
                   "converged": "converged" in str(r.status)},
         "_status": str(r.status)}
    for k, val in zip(names, x):
        d[k] = float(val) / 4.448222 if U.get(k, "") == "N" else float(val)
    return d


def main():
    try:
        from mpi4py import MPI
        rank, size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    except ImportError:
        rank, size = 0, 1
    os.makedirs(OUTDIR, exist_ok=True)
    cases = [(a, p) for p in PAX for a in ARCHS]
    if len(sys.argv) > 1:   # e.g.  pax_sweep.py h2fc_300 battery_400
        want = set(sys.argv[1:])
        cases = [(a, p) for a, p in cases if f"{a}_{p}" in want]
    for arch_key, pax in cases[rank::size]:
        out = os.path.join(OUTDIR, f"{arch_key}_{pax}.json")
        tag = f"[rank {rank}] {arch_key} @ {pax} pax"
        try:
            d = run_case(arch_key, pax)
            json.dump(d, open(out, "w"))
            print(f"{tag}: {d['_status']}"
                  + (f"  MTOW {d.get('W_total_max', float('nan')):,.0f} lbf"
                     if d["_case"]["converged"] else ""), flush=True)
        except Exception:
            json.dump({"_case": {"arch": arch_key, "pax": pax,
                                 "converged": False},
                       "_status": "EXCEPTION",
                       "_traceback": traceback.format_exc()}, open(out, "w"))
            print(f"{tag}: EXCEPTION (see json)", flush=True)


if __name__ == "__main__":
    main()
