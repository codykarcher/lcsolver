"""Conservative fitting on the Hoburg UAV problem, paper Sections 7.4-7.6.

The three variants of ``slcp_cases.hoburg`` differ only in how much of the
profile-drag relation the solver can see: as an explicit posynomial (0), or
behind a black box for one segment (1) or all three (3). The optimum is the
same in every case, because the black box evaluates the same underlying fit.

This script asks a different question. Given only *samples* of the black box,
what happens if the constraint is refitted and handed back to the solver as a
posynomial -- moving it from the linearized class into the exact one, so the
trust region and step rejection are no longer needed?

Two fits are compared:

  least squares   accurate, but two-sided: it sits below the truth about half
                  the time, so the solver can walk into the gap
  conservative    constrained to lie above every sample, so a point feasible
                  for the surrogate is feasible for the truth

The measure that matters is not the fitting error but the violation of the
TRUE constraint at the returned optimum.
"""
import sys, os, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter('ignore')

from edi.fitting import fit_max_affine, evaluate_fit, fit_report
from edi.solvers.ipopt.slcp import Constraint, Problem
from edi.solvers.ipopt.sia import solve_sia, SIAOptions
import slcp_cases as C


def sample_blackbox(x0, names, n=12, spread=2.0, seed=0):
    """Sample the drag black box over a box around the starting point."""
    rng = np.random.default_rng(seed)
    icl = [names.index(f'C_L_{i}') for i in range(3)]
    ire = [names.index(f'Re_{i}') for i in range(3)]
    ita = names.index('tau')
    lo_cl, hi_cl = x0[icl].min() / spread, x0[icl].max() * spread
    lo_re, hi_re = x0[ire].min() / spread, x0[ire].max() * spread
    lo_ta, hi_ta = x0[ita] / 1.5, x0[ita] * 1.5

    grid = []
    for cl in np.geomspace(lo_cl, hi_cl, n):
        for re in np.geomspace(lo_re, hi_re, n):
            for ta in np.geomspace(lo_ta, hi_ta, 4):
                grid.append((cl, re, ta))
    X = np.array(grid)
    y = np.array([C._profile_drag_blackbox(a, b, c)[0] for a, b, c in X])
    return X, y


def fitted_constraints(fit, names, seg):
    """`C_Dp_seg >= fit(C_L_seg, Re_seg, tau)` as K monomial constraints."""
    icl, ire = names.index(f'C_L_{seg}'), names.index(f'Re_{seg}')
    ita, icd = names.index('tau'), names.index(f'C_Dp_{seg}')
    out = []
    for k in range(fit['K']):
        e = fit['e'][k]
        out.append(Constraint(C.posy(
            [(fit['c'][k], {f'C_L_{seg}': e[0], f'Re_{seg}': e[1],
                            'tau': e[2], f'C_Dp_{seg}': -1.0})], names), '<='))
    return out


def build_with_fit(fit, names):
    """The Hoburg problem with all three drag constraints supplied by `fit`."""
    obj = C.posy([(1.0, {'W_fuel_out': 1}), (1.0, {'W_fuel_ret': 1})], names)
    cons = C._hoburg_constraints(names)[:-C._HOB_SEG]
    for seg in range(C._HOB_SEG):
        cons += fitted_constraints(fit, names, seg)
    return Problem(len(names), obj, cons, names=names)


def true_violation(x, names):
    """Worst log violation of the REAL drag constraint at `x`."""
    worst = -np.inf
    for seg in range(3):
        cl = x[names.index(f'C_L_{seg}')]
        re = x[names.index(f'Re_{seg}')]
        ta = x[names.index('tau')]
        cdp = x[names.index(f'C_Dp_{seg}')]
        truth = C._profile_drag_blackbox(cl, re, ta)[0]
        worst = max(worst, np.log(truth / cdp))
    return worst


def run(label, problem, x0):
    opts = SIAOptions(verbose=False)
    r = solve_sia(problem, x0.copy(), opts)
    return dict(label=label, obj=float(r.objective), it=int(r.iterations),
                conv=bool(r.converged), x=np.asarray(r.x, dtype=float),
                stat=float(r.stationarity), viol=float(r.max_violation))


if __name__ == '__main__':
    names = list(C.HOBURG_NAMES)
    _, x0, _ = C.hoburg(0)

    print('=' * 78)
    print('Reference: the drag relation as an explicit posynomial (Sec 7.4)')
    print('=' * 78)
    p0, x0_, _ = C.hoburg(0)
    ref = run('explicit posynomial', p0, x0_)
    print(f"  objective {ref['obj']:.6f}   iterations {ref['it']}   "
          f"converged {ref['conv']}")
    print(f"  true drag-constraint violation at the optimum: "
          f"{true_violation(ref['x'], names):+.3e}")

    print()
    print('=' * 78)
    print('Black-boxed (Sec 7.6): three linearized constraints')
    print('=' * 78)
    p3, x3, _ = C.hoburg(3)
    bb = run('black box, linearized', p3, x3)
    print(f"  objective {bb['obj']:.6f}   iterations {bb['it']}   "
          f"converged {bb['conv']}")
    print(f"  true drag-constraint violation at the optimum: "
          f"{true_violation(bb['x'], names):+.3e}")

    print()
    print('=' * 78)
    print('Refitting the black box from samples')
    print('=' * 78)
    X, y = sample_blackbox(x0, names)
    print(f"  {len(y)} samples of the black box over the design box")
    print()

    for mode in (None, 'shift', 'upper'):
        fit = fit_max_affine(X, y, K=4, conservative=mode)
        tag = {None: 'least squares',
               'shift': 'conservative (fit, then shift clear of the data)',
               'upper': 'conservative (each piece constrained as fitted)'}[mode]
        print(f"  --- {tag} ---")
        for line in fit_report(fit).splitlines()[1:]:
            print('   ' + line)
        try:
            res = run(tag, build_with_fit(fit, names), x0.copy())
            tv = true_violation(res['x'], names)
            print(f"    objective {res['obj']:.6f}   iterations {res['it']}"
                  f"   converged {res['conv']}")
            print(f"    TRUE drag-constraint violation at the optimum: "
                  f"{tv:+.3e}"
                  + ("   <-- INFEASIBLE for the real problem" if tv > 1e-6
                     else "   <-- feasible for the real problem"))
            print(f"    objective vs. reference: "
                  f"{100*(res['obj']/ref['obj'] - 1):+.2f}%")
        except Exception as exc:
            print(f"    solve failed: {type(exc).__name__}: {exc}")
        print()
