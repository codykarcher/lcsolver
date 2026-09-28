"""Benchmark min_norm_multipliers on the spcomparisons 737 deck.

usage:
  POLAR=cair GROUND=1 ALTCR=1 PYTHONPATH=$HOME/lcsolver_s4 \
    python bench_min_norm.py [--tag NAME] [--act-tol 1e-5] [--dump data.npz]
          [--lcsolver PATH] [--tree DIR] [--artifact JSON] [--out DIR]

Builds the model exactly as ~/tasopt_logs/v1/verify_full.py does, loads the
saved point, then times ``min_norm_multipliers(prob, x, act_tol)`` and
reports the stationarity / complementarity residuals ``kkt_residuals``
computes from the returned multipliers. Saves multipliers + residuals to
<out>/<tag>.npz so two implementations can be compared afterwards.

--dump writes g0, log_g of every row, the operators and the log-gradients
of every equality row and every inequality row within 1e-3 of active as a
scipy.sparse CSC matrix, so the LSQ can be replayed without the model
(see replay_min_norm.py).
"""
import os, sys, json, time, argparse, warnings, dataclasses
warnings.filterwarnings("ignore")
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="run")
ap.add_argument("--act-tol", type=float, default=1e-5)
ap.add_argument("--dump", default=None)
ap.add_argument("--lcsolver", default=os.path.expanduser("~/lcsolver_s4"))
ap.add_argument("--tree", default=os.path.expanduser(
    "~/Dropbox/research/lcjetliner/lcjetliner/spcomparisons"))
ap.add_argument("--artifact", default=os.path.expanduser(
    "~/Dropbox/research/lcjetliner/lcjetliner/spcomparisons/b737_7seg_F22isa0.json"))
ap.add_argument("--out", default=os.path.expanduser("~/tasopt_logs/v1/minnorm_bench"))
ap.add_argument("--skip-solve", action="store_true")
args = ap.parse_args()

TREE, SOL = args.tree, args.artifact
sys.path.insert(0, TREE); sys.path.insert(0, TREE + "/components"); sys.path.insert(0, TREE + "/tools")
sys.path.insert(0, args.lcsolver)
os.chdir(TREE)
for k, v in (("TECH", "cfm56_era"), ("SP_ENGINE", "cfm56_era"), ("GEAR_BOX_FRAC", "1.00"), ("V_HT_FLOOR", "0.01")):
    os.environ.setdefault(k, v)
for k in ("H_FIELD_FT", "T_FIELD_K"):
    os.environ.pop(k, None)
import pyomo.environ as pyo
import classes, architectures, aircraft
from lcsolver_compat import structure_detector, unit_corrector
import lcsolver
from lcsolver.solvers.sequential.bridge import build_problem, fold_bound_rows
from lcsolver.solvers.sequential import sia
from lcsolver.solvers.sequential.sia import log_g as _log_g, kkt_residuals, min_norm_multipliers
print("lcsolver from", os.path.dirname(lcsolver.__file__), flush=True)

sol = json.load(open(SOL))
nseg = sum(1 for k in sol if k.startswith("FS_hft["))
ground = any(k.startswith("Eng_st_") or k.startswith("Eng_cyc_st_") for k in sol)
ndesc = nseg - 7
cl = dataclasses.replace(classes.CLASSES["b737"], ref_mach=0.80, field_length_ft=8000.0)
ar = dataclasses.replace(architectures.ARCHS["conventional"], lock_mach=True)
t0 = time.time()
cm = unit_corrector(aircraft.build(cl, ar, seed="reference", mission="tasopt", n_descent=ndesc, ground_points=ground))
st = structure_detector(cm)
LBF = 4.448222
RAW_N = float(sol.get("Eng_W_engine", 0.0)) > 8000.0
x, miss = [], []
for v in st["variables"]:
    val = sol.get(v.name)
    if not isinstance(val, (int, float)) or val <= 0:
        miss.append(v.name); val = float(pyo.value(v)) if v.value else 1.0
    elif str(v.get_units()) == "N" and not RAW_N:
        val *= LBF
    x.append(float(val))
prob = build_problem(fold_bound_rows(st), sp_form=True, split_equalities=False, pair_equalities=True)
n = prob.n; x = np.asarray(x[:n])
print("built %s: %d segs ground=%s, n=%d, rows=%d, %d unseeded vars (%.0fs)" % (
    TREE, nseg, ground, n, len(prob.constraints), len(miss), time.time() - t0), flush=True)

lg = np.array([_log_g(c, x) for c in prob.constraints])
ops = np.array([c.operator for c in prob.constraints])
eq = ops == "=="
print("equalities %d, inequalities %d, active ineq at %g: %d" % (
    eq.sum(), (~eq).sum(), args.act_tol, int(((~eq) & (lg >= -args.act_tol)).sum())), flush=True)

if args.dump:
    import scipy.sparse as sp
    g0 = np.asarray(prob.objective.log_grad(x), dtype=float)
    rows = [i for i, c in enumerate(prob.constraints) if ops[i] == "==" or lg[i] >= -1e-3]
    cols = [sp.csc_matrix(np.asarray(prob.constraints[i].body.log_grad(x), dtype=float).reshape(-1, 1)) for i in rows]
    A = sp.hstack(cols).tocsc()
    np.savez_compressed(args.dump, g0=g0, lg=lg, ops=ops, rows=np.array(rows), x=x,
                        A_data=A.data, A_indices=A.indices, A_indptr=A.indptr, A_shape=np.array(A.shape))
    print("dumped %s: A %s nnz %d" % (args.dump, A.shape, A.nnz), flush=True)

if args.skip_solve:
    sys.exit(0)

opts = sia.SIAOptions() if hasattr(sia, "SIAOptions") else None
x_min = getattr(opts, "x_min", 1e-9) if opts is not None else 1e-9
t1 = time.time()
m = min_norm_multipliers(prob, x, args.act_tol)
dt = time.time() - t1
stat, viol, comp = kkt_residuals(prob, x, m, x_min)
stat0, _, _ = kkt_residuals(prob, x, m, None)
print("%s: min_norm_multipliers %.1fs  stat(proj)=%.6e stat(raw)=%.6e viol=%.3e comp=%.3e  "
      "nnz(mults)=%d  |mults|_inf=%.3e" % (args.tag, dt, stat, stat0, viol, comp,
                                          int((m != 0).sum()), np.abs(m).max()), flush=True)
os.makedirs(args.out, exist_ok=True)
np.savez(os.path.join(args.out, args.tag + ".npz"), mults=m, stat=stat, stat_raw=stat0,
         viol=viol, comp=comp, time=dt, act_tol=args.act_tol)
