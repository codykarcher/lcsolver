"""Thin regression runner for the rubber-engine project.

Solves the conventional 737 (default build: cranked wing, TASOPT_737800 deck,
TECH=cfm56_era, sea-level basis) and prints MTOW / fuel in lbf against the
gate numbers from HANDOFF_ENGINE.md. Environment knobs that other scratch
runners export are scrubbed first so class values are actually in force.
"""
from __future__ import annotations

import os, sys, time, warnings

# Scrub every knob the models read from the environment, then pin the case.
for _k in ("V_VT_FLOOR", "V_HT_FLOOR", "H_FIELD_FT", "T_FIELD_K", "SM_MIN",
           "GEAR_BOX_FRAC", "FREE_Y_ENG", "PUSH", "PUSH_XM", "EDI_FAST"):
    os.environ.pop(_k, None)
os.environ["TECH"] = "cfm56_era"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "components"))
sys.path.insert(0, "/Users/codykarcher/Dropbox/research/edi")

warnings.filterwarnings("ignore")

# Regression gates, lbf. HANDOFF_ENGINE.md's original numbers (737
# 166,406.8 / 47,162.5; D8 137,024.9 / 26,907.9) were superseded when the
# descent model landed (eebb46e "descent exists now") -- cruise no longer
# over-burns to cover a missing descent leg. The 737 gate below is
# bit-identical to the baseline documented in 502d139 (strut-braced wing:
# "b737 mission, conventional 165,495 / 46,170 bit-identical"); the D8 gate
# is this runner's own post-descent solve at the same commit, bit-stable
# across repeated runs.
# 737 gate updated again at 86a22af (underwing engine station at the wing
# LE): its message documents "MTOW 165,346 (-0.1%), fuel 46,385 (+0.47%)",
# bit-identical to this runner's solve at that commit.
# Updated at eb06bed (nose-gear bay reach): "conventional +0.70% fuel (NEW
# BASELINE ... 46,709 fuel)" -- fuel matches this runner to the digit; the
# D8 moved +2.4% fuel ("its 14.3 m gear was doing free CG work").
GATES = {
    "b737/conventional_M": {"W_total": 164263.7, "W_f_total": 45725.9},
    # Free-Mach d8: the double-bubble build pins cruise near M 0.72 itself;
    # d8_M would wrongly lock it to the 737 class's 0.785.
    # D8 re-baselined at 6ecf10d (rear-engine spacing walls): its message
    # documents "CANONICAL D8: MTOW ... 142,520 ... fuel 27,601",
    # bit-identical to this runner.
    "b737/d8": {"W_total": 140126.9, "W_f_total": 26976.0},
}

# Per-case environment: the D8 gate needs its own tech level and knobs.
CASE_ENV = {
    "b737/conventional_M": {"TECH": "cfm56_era"},
    "b737/d8": {"TECH": "d8_era", "V_VT_FLOOR": "0.001"},
}


# Class overrides for the D8 gate: 8-abreast 2-aisle double bubble, span
# limit 44.2 m, field 4,960 ft (all bind in d8_final.json).
CASE_CLASS = {
    "b737/d8": dict(seats_abreast=8, n_aisles=2, span_max_m=44.2,
                    field_length_ft=4960.0),
}


def solve_case(cls_key="b737", arch_key="conventional_M"):
    import importlib, dataclasses
    env = CASE_ENV.get(f"{cls_key}/{arch_key}", {"TECH": "cfm56_era"})
    for _k in ("V_VT_FLOOR", "V_HT_FLOOR", "H_FIELD_FT", "T_FIELD_K"):
        os.environ.pop(_k, None)
    os.environ.update(env)
    for mod in ("components.technology", "components.turbofan.model",
                "components.wing", "aircraft"):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    import pyomo.environ as pyo
    from pyomo.environ import units as u
    import classes, architectures, aircraft
    from edi_compat import structure_detector, unit_corrector
    from edi.solvers.ipopt.slcp_bridge import solve_sia
    from edi.solvers.ipopt.sia import SIAOptions

    LBF = 0.2248089431
    t0 = time.time()
    cl = classes.CLASSES[cls_key]
    over = CASE_CLASS.get(f"{cls_key}/{arch_key}")
    if over:
        cl = dataclasses.replace(cl, **over)
    cm = unit_corrector(aircraft.build(cl, architectures.ARCHS[arch_key],
                                       seed="reference"))
    st = structure_detector(cm)
    opts = SIAOptions(max_iterations=200)
    opts.stationarity_tolerance = 1e-5
    opts.condense_numerator = True
    opts.ipopt_options = dict(opts.ipopt_options, tol=1e-9,
                              constr_viol_tol=1e-9)
    res = solve_sia(st, options=opts, presolve=False)
    if not res.converged:
        print(f"{cls_key}/{arch_key}: FAILED  {str(res.status)[:160]}")
        return None
    for v, val in zip(st["variables"], res.x):
        v.set_value(float(val))
    vals = {}
    for v in cm.component_data_objects(pyo.Var):
        val = pyo.value(v)
        vals[v.name] = val * LBF if str(u.get_units(v)) == "N" else val
    tag = f"{cls_key}/{arch_key}"
    print(f"{tag}: converged in {res.iterations} it, {time.time()-t0:.1f}s  "
          f"stationarity {getattr(res, 'stationarity', float('nan')):.2e}")
    gate = GATES.get(tag, {})
    for name in ("W_total", "W_f_total"):
        got = vals.get(name, float("nan"))
        want = gate.get(name)
        mark = ""
        if want is not None:
            mark = "  OK" if abs(got - want) < 1.0 else f"  GATE={want:,.0f} MISMATCH"
        print(f"  {name:10s} {got:12,.1f} lbf{mark}")
    return vals


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=["b737/conventional_M",
                                                 "b737/d8"])
    a = ap.parse_args()
    for case in a.cases:
        cls_key, arch_key = case.split("/")
        solve_case(cls_key, arch_key)
