"""Rescue cold-start failures by walking the design space in small steps.

Why this exists
---------------
Every case cold-starts from ``reference.json``, the D8.2 gpkit optimum: 180
passengers, 8 abreast, double bubble, BLI, Jet-A. That is an excellent seed
for ``b737/d8`` and a progressively worse one the further a case sits from
it. A phase-1 failure from a bad seed looks exactly like genuine
infeasibility and is not the same thing -- the fuel-cell model earlier in
this project "could not fly 3000 nmi" right up until it was warm-started
from its own 2800 nmi solution, after which it converged in 30 iterations.

So a failure is only reported as INFEASIBLE after it has also failed from a
neighbour that did converge.

The walk
--------
Two hops, because the seed is wrong along two independent axes:

  size:          b737 -> e175 -> citation,  and  b737 -> b787
  architecture:  d8 -> conventional -> h2burn -> h2fc -> battery

Size first, because cabin geometry moves the most variables. Each solved
case becomes the seed for its neighbours, so the reference point propagates
outward by small steps instead of one large jump.
"""
from __future__ import annotations

import json, sys, time, warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

# Nearest already-solved neighbour to try, in order of preference.
SIZE_NEIGHBOUR = {"b737": [], "e175": ["b737"], "citation": ["e175", "b737"],
                  "b787": ["b737"]}
ARCH_NEIGHBOUR = {"d8": ["conventional"], "conventional": ["d8"],
                  "h2burn": ["conventional", "d8"],
                  "h2fc": ["h2burn", "conventional"],
                  "battery": ["h2fc", "conventional"]}


def _seeds_for(cls_key, arch_key):
    """Neighbours to try, nearest first: same size other arch, then same arch
    other size. Same-size hops move fewer variables than same-architecture
    ones, because cabin geometry touches the whole fuselage chain."""
    out = []
    for a in ARCH_NEIGHBOUR.get(arch_key, []):
        out.append((cls_key, a))
    for c in SIZE_NEIGHBOUR.get(cls_key, []):
        out.append((c, arch_key))
    for c in SIZE_NEIGHBOUR.get(cls_key, []):
        for a in ARCH_NEIGHBOUR.get(arch_key, []):
            out.append((c, a))
    return out


def solve_warm(cls_key, arch_key, warm_values, maxit=2000):
    """Solve one case with every matching variable pre-set from ``warm_values``."""
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "components"))
    sys.path.insert(0, "/Users/codykarcher/Dropbox/research/edi")
    import pyomo.environ as pyo
    from pyomo.environ import units as u
    import classes, architectures, aircraft
    from edi_compat import structure_detector
    from edi.solvers.ipopt.slcp_bridge import solve_sia
    from edi.solvers.ipopt.sia import SIAOptions
    from edi_compat import unit_corrector

    LBF = 0.2248089431
    cl, ar = classes.CLASSES[cls_key], architectures.ARCHS[arch_key]
    cm = unit_corrector(aircraft.build(cl, ar, seed="reference"))
    n = 0
    if warm_values:
        # Values are stored normalised to lbf; put newtons back before
        # seeding, or every force variable starts a factor of 4.45 out.
        for v in cm.component_data_objects(pyo.Var):
            if v.name in warm_values:
                val = warm_values[v.name]
                if str(u.get_units(v)) == "N":
                    val = val / LBF
                v.set_value(float(val), skip_validation=True); n += 1
    st = structure_detector(cm)
    t0 = time.time()
    r = solve_sia(st, options=SIAOptions(max_iterations=maxit), presolve=False)
    rec = {"class": cls_key, "arch": arch_key, "converged": bool(r.converged),
           "status": str(r.status)[:120], "seeded_vars": n,
           "iterations": int(getattr(r, "iterations", 0)),
           "wall_s": round(time.time() - t0, 1)}
    if r.converged:
        for v, val in zip(st["variables"], r.x):
            v.set_value(float(val))
        vals = {}
        for v in cm.component_data_objects(pyo.Var):
            val = pyo.value(v); un = str(u.get_units(v))
            vals[v.name] = val * LBF if un == "N" else val
        rec["values"] = vals
    return rec


def main():
    sys.path.insert(0, str(HERE))
    import classes, architectures
    solved = {}
    for fp in sorted(OUT.glob("*.json")):
        rec = json.loads(fp.read_text())
        if rec.get("converged"):
            solved[(rec["class"], rec["arch"])] = rec["values"]
    print(f"{len(solved)} cases already solved\n", flush=True)

    # Repeat until a full pass rescues nothing: each success creates new seeds.
    for rnd in range(1, 6):
        failures = []
        for c in classes.ORDER:
            for a in architectures.ORDER:
                if (c, a) not in solved:
                    failures.append((c, a))
        if not failures:
            break
        print(f"--- round {rnd}: {len(failures)} unsolved ---", flush=True)
        rescued = 0
        for c, a in failures:
            for sc, sa in _seeds_for(c, a):
                if (sc, sa) not in solved:
                    continue
                rec = solve_warm(c, a, solved[(sc, sa)])
                tag = f"{c}/{a}"
                if rec["converged"]:
                    solved[(c, a)] = rec["values"]
                    v = rec["values"]
                    mtow = v.get("W_total") or v.get("W_MTO")
                    print(f"  {tag:22s} RESCUED from {sc}/{sa}"
                          f"  MTOW {mtow:9,.0f} lb  {rec['iterations']:4d} it"
                          f"  {rec['wall_s']:6.1f}s", flush=True)
                    (OUT / f"{c}__{a}.json").write_text(json.dumps(rec, indent=1))
                    rescued += 1
                    break
                else:
                    print(f"  {tag:22s} still fails from {sc}/{sa}"
                          f"  ({rec['status'][:44]})", flush=True)
        if rescued == 0:
            print("\nnothing rescued this round -- remaining cases are "
                  "infeasible as posed, not merely badly seeded", flush=True)
            break
    print(f"\n{len(solved)}/20 converged")


if __name__ == "__main__":
    main()
