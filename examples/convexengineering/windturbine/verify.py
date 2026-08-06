"""Verify the wind turbine GP against the two results the paper states.

This model has no published source code, so the paper is the only reference.
Two checks:

1. **Betz limit** (exact). With no viscous drag and no tip loss the rotor
   power coefficient must approach 16/27 = 0.5926 as the tip speed ratio
   grows. This is an analytic result, not a digitized one, so it is the
   stronger of the two checks.

2. **Figure 1** (digitized). Peak Cp and the tip speed ratio at which it
   occurs, for c_l/c_d in {5, 10, 20, 40}, with and without tip loss.
   Tolerances are loose because the reference values are read off a plot.

Run: ``python verify.py``
"""
from __future__ import annotations

from model import BETZ, cp_at

# Peak (lambda, Cp) read off Figure 1. Plot-digitized: +-0.02 in Cp and
# +-0.5 in lambda is as good as this reference gets.
FIG1_PEAKS = {
    # eps      no tip loss      with tip loss (B=3 assumed)
    0.200: ((1.5, 0.31), (2.0, 0.24)),
    0.100: ((2.0, 0.39), (3.0, 0.32)),
    0.050: ((2.75, 0.46), (4.0, 0.41)),
    0.025: ((3.75, 0.51), (5.5, 0.47)),
}

LAMS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]


def betz_check(tol: float = 5e-3) -> bool:
    print("1. Betz limit  (eps = 0, no tip loss)")
    print(f"   target Cp -> 16/27 = {BETZ:.4f}\n")
    cp = None
    for lam in (4.0, 8.0, 16.0, 32.0):
        cp = cp_at(N=10, lam=lam, eps=0.0, tip_loss=False)
        print(f"     lambda={lam:5.1f}   Cp={cp:.4f}   Cp/Betz={cp / BETZ:.4f}")
    gap = (BETZ - cp) / BETZ
    ok = 0 <= gap < tol
    print(f"\n   at lambda=32, {gap * 100:.2f}% below Betz  ->  "
          f"{'PASS' if ok else 'FAIL'}  (must approach from below, gap < {tol * 100:g}%)\n")
    return ok


def figure1_check(cp_tol: float = 0.03, lam_tol: float = 1.0) -> bool:
    print("2. Figure 1 peaks   (reference values digitized from the plot)")
    allok = True
    for tip_loss in (False, True):
        label = "with tip loss (B=3, GUESS)" if tip_loss else "no tip loss"
        print(f"\n   {label}")
        print(f"   {'c_l/c_d':>8}  {'lam*':>6} {'ref':>6}   {'Cp*':>6} {'ref':>6}   result")
        for eps, peaks in sorted(FIG1_PEAKS.items()):
            lam_ref, cp_ref = peaks[1 if tip_loss else 0]
            curve = [(lam, cp_at(N=10, lam=lam, eps=eps, B=3, tip_loss=tip_loss))
                     for lam in LAMS]
            lam_pk, cp_pk = max(curve, key=lambda t: t[1])
            ok = abs(cp_pk - cp_ref) <= cp_tol and abs(lam_pk - lam_ref) <= lam_tol
            allok &= ok
            print(f"   {int(1 / eps):>8}  {lam_pk:>6.2f} {lam_ref:>6.2f}   "
                  f"{cp_pk:>6.3f} {cp_ref:>6.3f}   {'ok' if ok else 'OFF'}")
    print()
    return allok


if __name__ == "__main__":
    a = betz_check()
    b = figure1_check()
    print("=" * 62)
    print(f"Betz limit: {'PASS' if a else 'FAIL'}    Figure 1: {'PASS' if b else 'FAIL'}")
    print("=" * 62)
