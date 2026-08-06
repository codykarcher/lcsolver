"""Where does a battery-electric aircraft actually close?

The matrix runs each class at its real design range, and every battery case
fails there -- correctly. A 200 Wh/kg pack carries roughly 150-260 nmi of
useful range by the electric range equation, and the shortest mission in the
matrix is the E175's 2,200 nmi.

Reporting "battery infeasible" from that is true but uninformative: it says
only that 2,200 nmi is not 250 nmi. The question worth answering is where the
crossover sits, which is what this sweeps for. Each class is re-solved at
decreasing range until it converges, and the last converged range is the
aircraft's battery-electric limit at that pack technology.
"""
from __future__ import annotations

import json, sys, time, warnings
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "components"))
sys.path.insert(0, "/Users/codykarcher/Dropbox/research/lcsolver")
import classes, architectures  # noqa: E402  -- needed at module scope too


def solve_at(cls_key, range_nmi, maxit=800, arch_key="battery", warm=None):
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
    cl = replace(classes.CLASSES[cls_key], range_nmi=range_nmi)
    cm = unit_corrector(aircraft.build(cl, architectures.ARCHS[arch_key],
                                       seed="reference"))
    if warm:
        for v in cm.component_data_objects(pyo.Var):
            if v.name in warm:
                val = warm[v.name]
                if str(u.get_units(v)) == "N":
                    val /= LBF
                v.set_value(float(val), skip_validation=True)
    st = structure_detector(cm)
    r = solve_sia(st, options=SIAOptions(max_iterations=maxit), presolve=False)
    if not r.converged:
        return None
    for v, val in zip(st["variables"], r.x):
        v.set_value(float(val))
    out = {}
    for v in cm.component_data_objects(pyo.Var):
        val = pyo.value(v)
        out[v.name] = val * LBF if str(u.get_units(v)) == "N" else val
    out["_it"] = r.iterations
    return out


# ---------------------------------------------------------------------------
# The sweep has to WALK DOWN in range, warm-starting each step from the last.
#
# The first version solved every range cold from reference.json and concluded
# that a 100 nmi Citation was infeasible. It is not: the same airframe and the
# same architecture converge at 3,000 nmi. reference.json is a 180-passenger
# 3,000 nmi D8.2, so a 100 nmi 8-seater is the largest extrapolation anywhere
# in this study, and Phase I was being asked to cross it cold. A short-range
# aircraft is not a harder DESIGN than a long-range one -- it is just further
# from this particular seed.
#
# So: start where the matrix already converged and step down, each solve
# seeded by the one before it. And because there is no converged battery case
# at any range to start from, the walk begins on the fuel-cell twin, which
# shares the airframe and the electric drivetrain and differs only in what
# supplies the energy.
if __name__ == "__main__":
    import json as _json
    results = {}
    for ck in ("citation", "e175"):
        # Step 1: get onto the battery architecture at a range that is known
        # to solve, seeded from the converged fuel-cell twin.
        seed = None
        fp = Path("results") / f"{ck}__h2fc.json"
        if fp.exists():
            rec = _json.loads(fp.read_text())
            if rec.get("converged"):
                seed = rec["values"]
        base = classes.CLASSES[ck].range_nmi
        t0 = time.time()
        cur = solve_at(ck, base, arch_key="battery", warm=seed)
        if cur is None:
            print(f"  {ck:10s} {base:5d} nmi   battery does not close even "
                  f"warm-started from h2fc  ({time.time()-t0:.0f}s)", flush=True)
            continue
        print(f"  {ck:10s} {base:5d} nmi   MTOW {cur['W_total']:9,.0f} lb"
              f"   pack {cur.get('Batt_W_batt',0):8,.0f} lb", flush=True)
        results[f"{ck}_{base}"] = {"MTOW": cur["W_total"],
                                   "pack": cur.get("Batt_W_batt", 0.0)}
        # Step 2: walk DOWN in range, each solve warm-started from the last.
        for rng in (1500, 1000, 700, 500, 350, 250, 150, 100):
            if rng >= base:
                continue
            t0 = time.time()
            o = solve_at(ck, rng, arch_key="battery", warm=cur)
            if o is None:
                print(f"  {ck:10s} {rng:5d} nmi   no start even warm"
                      f"   ({time.time()-t0:.0f}s)", flush=True)
                continue
            cur = o
            mt = o["W_total"]; pk = o.get("Batt_W_batt", 0.0)
            print(f"  {ck:10s} {rng:5d} nmi   MTOW {mt:9,.0f} lb"
                  f"   pack {pk:8,.0f} lb = {pk/mt*100:4.1f}% MTOW"
                  f"   ({time.time()-t0:.0f}s)", flush=True)
            results[f"{ck}_{rng}"] = {"MTOW": mt, "pack": pk}
    Path("battery_sweep.json").write_text(_json.dumps(results, indent=1))
