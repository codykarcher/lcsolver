"""The one-shot free coupled solve, standalone.

Build the SP-engined aircraft, seed from the case-reference artifact
(plus the engine's build-time self-seed), solve ONCE with free levers,
report with certificates. No stages, no ladder, no pins.

Env: SP_TECH (default cfm56_era), SIA_OPTS (comma k=v), CASE_REF path,
TRAJ (log per-iteration lever trajectory via solver verbose parse).
"""
import os, sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/components")
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
os.chdir(HERE)
os.environ["SP_ENGINE"] = os.environ.get("SP_TECH", "cfm56_era")

import pyomo.environ as pyo
import classes, architectures, aircraft, dataclasses
from edi_compat import structure_detector, unit_corrector
from edi.solvers.ipopt.slcp_bridge import solve_sia, build_problem
from edi.solvers.ipopt.sia import SIAOptions

ar = dataclasses.replace(architectures.ARCHS["conventional"],
                         lock_mach=True)
t0 = time.time()
cls_key = os.environ.get("CLASS", "b737")
_kw = ({"polar": os.environ["POLAR"]} if os.environ.get("POLAR") else {})
cm = unit_corrector(aircraft.build(classes.CLASSES[cls_key], ar,
                                   seed="reference", **_kw))
ref_path = os.environ.get("CASE_REF", "b737_case_reference.json")
ref = json.load(open(ref_path)) if os.path.exists(ref_path) else {}
_skip = tuple(s for s in os.environ.get("SEED_SKIP", "").split(",") if s)
n_set = 0
for v in cm.component_data_objects(pyo.Var):
    if _skip and v.name.startswith(_skip):
        continue
    val = ref.get(v.name)
    if val and val > 0:
        v.set_value(float(val)); n_set += 1
# PIN="pi_f_D=1.55,BPR_D=9.0,..." adds v == value ROWS (pins must be
# rows -- var.fix() and bounds are both ignored by the detector). Used
# to manufacture a pinned-cycle case reference for a new airframe class.
_pins = {}
for kv in os.environ.get("PIN", "").split(","):
    if "=" in kv:
        k, val = kv.split("=", 1)
        _pins[k.strip()] = float(val)
if _pins:
    cm.pin_rows = pyo.ConstraintList()
    for v in cm.component_data_objects(pyo.Var):
        for k, val in _pins.items():
            if v.name == k or v.name.endswith("_" + k):
                cm.pin_rows.add(v == val)
                v.set_value(val)
                print(f"pinned {v.name} == {val}", flush=True)
st = structure_detector(cm)
print(f"built+seeded ({n_set} from {ref_path}) in {time.time()-t0:.0f}s",
      flush=True)

opts = SIAOptions(max_iterations=int(os.environ.get("MAXIT", "400")))
opts.stationarity_tolerance = 1e-5
opts.condense_numerator = True
opts.kkt_min_norm = True
opts.ipopt_options = dict(opts.ipopt_options, tol=1e-9,
                          constr_viol_tol=1e-9, max_iter=6000)
for kv in os.environ.get("SIA_OPTS", "").split(","):
    if "=" in kv:
        k, val = kv.split("=", 1)
        setattr(opts, k, eval(val))
if os.environ.get("TRAJ_LOG"):
    opts.traj_log = os.environ["TRAJ_LOG"]
    opts.traj_vars = tuple(os.environ.get(
        "TRAJ_VARS", "pi_f_D,pi_hc_D,BPR_D,s_cool_D,mdotc_D").split(","))

t1 = time.time()
res = solve_sia(st, options=opts, presolve=False, split_equalities=False)
dt = time.time() - t1
print(f"converged={res.converged} it={res.iterations} ({dt:.0f}s)  "
      f"stat={float(getattr(res,'stationarity',float('nan'))):.3e}  "
      f"feas={float(getattr(res,'max_violation',float('nan'))):.3e}",
      flush=True)
rep = getattr(res, "report", None)
if rep and not res.converged:
    print(str(rep)[:1200])
for v, val in zip(st["variables"], res.x):
    v.set_value(float(val))
out = {v.name: float(pyo.value(v))
       for v in cm.component_data_objects(pyo.Var)}
for k in ("Eng_cyc_pi_f_D", "Eng_cyc_pi_lc_D", "Eng_cyc_pi_hc_D",
          "Eng_cyc_BPR_D", "Eng_cyc_s_cool_D", "Eng_cyc_eff_fan_D",
          "W_f_total", "W_total", "Eng_W_engine"):
    print(f"  {k} = {out.get(k)}")
out_path = os.environ.get("OUT", "b737_free_solution.json")
json.dump(out, open(out_path, "w"))
print(f"wrote {out_path}")
