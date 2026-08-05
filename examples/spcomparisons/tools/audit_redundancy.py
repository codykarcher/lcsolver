"""Static redundancy audit at the settled point: find EVERY near-duplicate
constraint pair and near-degenerate direction, the class of modeling error
the s3 duplicate belonged to."""
import os, sys, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = "/Users/codykarcher/Dropbox/research/edi/examples/spcomparisons"
sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/components")
sys.path.insert(0, "/Users/codykarcher/Dropbox/research/edi")
os.chdir(HERE)
os.environ["SP_ENGINE"] = "cfm56_hiopr"
import pyomo.environ as pyo
import classes, architectures, aircraft, dataclasses
from edi_compat import structure_detector, unit_corrector
from edi.solvers.ipopt.slcp_bridge import build_problem
from edi.solvers.ipopt.sia import _log_g

ar = dataclasses.replace(architectures.ARCHS["conventional"], lock_mach=True)
cm = unit_corrector(aircraft.build(classes.CLASSES["b737"], ar,
                                   seed="reference"))
ref = json.load(open("b737_case_reference.json"))
for v in cm.component_data_objects(pyo.Var):
    val = ref.get(v.name)
    if val and val > 0:
        v.set_value(float(val))
st = structure_detector(cm)
pb = build_problem(st, sp_form=True)
x = np.array([float(pyo.value(v)) for v in st["variables"]],
             dtype=float)[:pb.n]
x = np.where(x > 0, x, 1.0)
names = pb.names

# active rows: all equalities + inequalities within 1e-5 of tight
rows, ridx = [], []
for i, c in enumerate(pb.constraints):
    g = _log_g(c, x)
    if c.operator == '==' or g >= -1e-5:
        gr = np.asarray(c.body.log_grad(x), dtype=np.float64)
        n = np.linalg.norm(gr)
        if n > 1e-12:
            rows.append(gr / n); ridx.append(i)
G = np.array(rows, dtype=np.float32)
print(f"{len(ridx)} active rows of {len(pb.constraints)}, n={pb.n}",
      flush=True)

def rowvars(i, k=4):
    c = pb.constraints[i]
    terms = list(getattr(c.body, 'terms', None) or [])
    for s_ in ('p', 'q'):
        sub = getattr(c.body, s_, None)
        if sub is not None:
            terms.extend(getattr(sub, 'terms', None) or [])
    js = sorted({j for _c, a in terms for j, e in enumerate(a) if e != 0.0})
    return ",".join(names[j] for j in js[:k])

# near-parallel PAIRS (|cos| > 0.9995), excluding split-equality twins
C = G @ G.T
np.fill_diagonal(C, 0.0)
hits = np.argwhere(np.abs(C) > 0.9995)
seen = set(); n_dup = 0
print("--- near-duplicate active row pairs (|cos| > 0.9995) ---")
for a, b in hits:
    if a >= b or (a, b) in seen:
        continue
    seen.add((a, b))
    ia, ib = ridx[a], ridx[b]
    oa, ob = pb.constraints[ia].operator, pb.constraints[ib].operator
    # split twins are the SAME source equality; skip pairs whose var sets
    # and opposite orientation mark them as a split pair
    if abs(C[a, b] + 1.0) < 5e-4 and rowvars(ia) == rowvars(ib):
        continue
    n_dup += 1
    if n_dup <= 25:
        print(f"  cos={C[a,b]:+.4f} [{ia}{oa}] {rowvars(ia)}")
        print(f"            [{ib}{ob}] {rowvars(ib)}")
print(f"  total non-twin near-duplicate pairs: {n_dup}")

# smallest singular directions of the active-row matrix
k = min(G.shape) - 1
sv = np.linalg.svd(G, compute_uv=False)
small = sv[sv < 1e-4]
print(f"--- rank: {G.shape[0]} rows, smallest 6 singular values: "
      f"{[f'{s:.1e}' for s in sv[-6:]]}  (<1e-4: {len(small)})")
