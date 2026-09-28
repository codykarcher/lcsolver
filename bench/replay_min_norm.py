"""Replay the min-norm LSQ on data dumped by bench_min_norm.py --dump.

usage: python replay_min_norm.py data.npz [--act-tol 1e-5] [--baseline] [--ref ref.npz]

--baseline also runs the original lsq_linear formulation (slow).
--ref compares against multipliers saved by bench_min_norm.py.
"""
import sys, time, argparse
import numpy as np
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--act-tol", type=float, default=1e-5)
ap.add_argument("--baseline", action="store_true")
ap.add_argument("--ref", default=None)
args = ap.parse_args()

d = np.load(args.data, allow_pickle=True)
g0, lg, ops, rows = d["g0"], d["lg"], d["ops"], d["rows"]
A_all = sp.csc_matrix((d["A_data"], d["A_indices"], d["A_indptr"]), shape=tuple(d["A_shape"]))
n = g0.size
sel = [k for k, i in enumerate(rows) if ops[i] == "==" or lg[i] >= -args.act_tol]
idx = rows[sel]
A = A_all[:, sel].toarray()
free = np.array([ops[i] == "==" for i in idx])
print("n=%d  cols=%d (eq %d, ineq %d)  nnz=%d" % (n, A.shape[1], free.sum(), (~free).sum(), A_all[:, sel].nnz))

def resid(lam):
    r = g0 + A @ lam
    return np.abs(r).max(), np.linalg.norm(r)

sys.path.insert(0, __import__("os").path.expanduser("~/lcsolver_s4"))
from lcsolver.solvers.sequential.sia import _min_norm_lsq

t = time.time()
lam_new = _min_norm_lsq(A, g0, free)
dt_new = time.time() - t
mi, l2 = resid(lam_new)
print("NEW : %.2fs  resid inf=%.6e l2=%.6e  min ineq mult=%.2e  |lam|inf=%.3e  nnz=%d" % (
    dt_new, mi, l2, lam_new[~free].min() if (~free).any() else 0, np.abs(lam_new).max(), (lam_new != 0).sum()))

if args.ref:
    r = np.load(args.ref)
    m = r["mults"][idx]
    mi_r, l2_r = resid(m)
    print("REF : %.1fs  resid inf=%.6e l2=%.6e  |lam|inf=%.3e   (stat recorded %.6e)" % (
        float(r["time"]), mi_r, l2_r, np.abs(m).max(), float(r["stat"])))
    print("      |lam_new - lam_ref|inf = %.3e, rel %.3e" % (np.abs(lam_new - m).max(),
          np.abs(lam_new - m).max() / max(np.abs(m).max(), 1e-300)))

if args.baseline:
    from scipy.optimize import lsq_linear
    scale = np.maximum(np.linalg.norm(A, axis=0), 1e-12)
    lo = np.where(free, -np.inf, 0.0)
    t = time.time()
    r = lsq_linear(A / scale, -g0, bounds=(lo, np.inf), tol=1e-12, max_iter=3000,
                   lsq_solver="lsmr", lsmr_tol=1e-12)
    lam_b = r.x / scale
    mi, l2 = resid(lam_b)
    print("BASE: %.1fs  status %d nit %d  resid inf=%.6e l2=%.6e  |lam|inf=%.3e" % (
        time.time() - t, r.status, r.nit, mi, l2, np.abs(lam_b).max()))
