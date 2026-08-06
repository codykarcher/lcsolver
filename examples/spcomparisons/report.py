"""Read results/ and print the comparison tables.

Weights are normalised to lbf in run.py, so nothing here needs to know which
component declared newtons and which declared pounds. That mixing is real --
SPaircraft uses lbf, the hydrogen components use N -- and LCsolver's unit
corrector reconciles it inside the constraints.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import classes, architectures

R = {}
for fp in (HERE / "results").glob("*.json"):
    rec = json.loads(fp.read_text())
    R[(rec["class"], rec["arch"])] = rec


def g(rec, *keys):
    v = rec.get("values", {})
    for k in keys:
        if k in v:
            return v[k]


def cell(rec, keys, scale=1.0, p=0, w=13):
    if rec is None:
        return "-".rjust(w)
    if not rec.get("converged"):
        # Which failure this is matters, and the two are told apart by how
        # many Phase I iterations were spent. A handful means the sub-problem
        # itself was malformed -- the equality bug, now fixed. Spending the
        # whole 50 means Phase I genuinely searched and found nothing, which
        # is evidence about the DESIGN rather than the solver.
        #
        # rec["iterations"] is the PHASE II count and is 0 whenever Phase I
        # fails, so it cannot be used for this. The real number is in the
        # status string.
        import re
        m = re.search(r"after (\d+) iterations", rec.get("status", ""))
        it = int(m.group(1)) if m else -1
        if it < 0:
            return "fail".rjust(w)
        # "no start" is deliberately NOT "infeasible". Spending the Phase I
        # budget means the search gave up, and Phase I on a signomial program
        # is a LOCAL method working on a convex approximation about the
        # current point -- it can fail on a perfectly feasible problem. Only
        # a case that also resists a warm start from a converged neighbour
        # has earned the word infeasible, and the physics has to agree.
        return ("x(bug)" if it <= 5 else "no start").rjust(w)
    val = g(rec, *keys)
    return "?".rjust(w) if val is None else f"{val*scale:,.{p}f}".rjust(w)


def table(title, keys, order, scale=1.0, p=0):
    print(f"\n{title}")
    print("-" * (12 + 13 * len(order)))
    print(f"{'':12s}" + "".join(a.replace("conventional", "conv")[:12].rjust(13)
                                for a in order))
    for c in classes.ORDER:
        print(f"{c:12s}" + "".join(cell(R.get((c, a)), keys, scale, p)
                                   for a in order))


FREE = architectures.ORDER
LOCK = architectures.ORDER_LOCKED
have_lock = any((c, a) in R for c in classes.ORDER for a in LOCK)

n_ok = sum(1 for r in R.values() if r.get("converged"))
print("=" * 66)
print(f"  {len(R)} cases run, {n_ok} converged")
print("  x(bug)   = Phase I equality bug, pre-fix run -- says nothing about the design")
print("  no start = Phase I budget spent. NOT proof of infeasibility: Phase I is a")
print("             local method, so this needs a warm start before it means anything")
print("=" * 66)

for title, keys, sc, p in (("MTOW [lb]", ("W_total", "W_MTO"), 1.0, 0),
                           ("OEW [lb]", ("W_dry",), 1.0, 0),
                           ("fuel carried [lb]", ("W_f_total",), 1.0, 0),
                           ("wing area [m2]", ("Wing_S",), 1.0, 1),
                           ("aspect ratio", ("Wing_AR",), 1.0, 2),
                           ("cruise Mach", ("FS_M[4]",), 1.0, 3)):
    table(f"{title}  -- Mach FREE", keys, FREE, sc, p)
    if have_lock:
        table(f"{title}  -- Mach LOCKED to design", keys, LOCK, sc, p)

# --- what locking Mach costs ----------------------------------------------
if have_lock:
    print("\n\nWHAT CRUISE SPEED COSTS: free-Mach optimum vs the real design Mach")
    print("-" * 78)
    print(f"{'':11s}{'M free':>8s}{'M real':>8s}{'OEW free':>11s}{'OEW lock':>11s}"
          f"{'OEW real':>11s}{'ratio free':>12s}{'ratio lock':>12s}")
    for c in classes.ORDER:
        cl = classes.CLASSES[c]
        fr, lk = R.get((c, "conventional")), R.get((c, "conventional_M"))
        if not (fr and fr.get("converged")):
            continue
        o_f = g(fr, "W_dry")
        line = (f"{c:11s}{g(fr,'FS_M[4]'):8.3f}{cl.ref_mach:8.2f}{o_f:11,.0f}")
        if lk and lk.get("converged"):
            o_l = g(lk, "W_dry")
            line += (f"{o_l:11,.0f}{cl.ref_OEW_lb:11,.0f}"
                     f"{o_f/cl.ref_OEW_lb:12.2f}{o_l/cl.ref_OEW_lb:12.2f}")
        else:
            line += f"{'--':>11s}{cl.ref_OEW_lb:11,.0f}{o_f/cl.ref_OEW_lb:12.2f}{'--':>12s}"
        print(line)

# --- battery range sweep ---------------------------------------------------
sweep = HERE / "battery_sweep.json"
if sweep.exists():
    print("\n\nBATTERY-ELECTRIC RANGE SWEEP (200 Wh/kg pack)")
    print("-" * 60)
    data = json.loads(sweep.read_text())
    for k in sorted(data, key=lambda x: (x.split("_")[0], int(x.split("_")[1]))):
        cls, rng = k.rsplit("_", 1)
        d = data[k]
        print(f"  {cls:10s}{int(rng):6d} nmi   MTOW {d['MTOW']:9,.0f} lb"
              f"   pack {d['pack']:8,.0f} lb  ({d['pack']/d['MTOW']*100:4.1f}% MTOW)")
