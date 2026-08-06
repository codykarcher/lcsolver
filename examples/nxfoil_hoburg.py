"""Hoburg UAV with profile drag from NeuralFoil, and the same fitted.

``slcp_xfoil`` swaps the Hoburg drag fit for XFOIL. This does the same with
metafoil's ``nxfoil`` -- NeuralFoil reimplemented in PyTorch -- which changes
the economics enough to be worth its own file:

* a polar is a single batched forward pass, milliseconds rather than seconds;
* the network is differentiable, so ``dC_Dp/dC_L`` comes from autograd rather
  than from a finite difference across two more polars.

That makes it cheap enough to ask the question this example exists for. A black
box has to be *linearized*, which is a local model: it needs a trust region, it
can be optimistic, and every iteration costs a gradient. A posynomial does not.
So the same drag relation is supplied three ways and the cost compared:

1. the original five-term posynomial fit (the reference optimum);
2. nxfoil as a black box, linearized every iteration;
3. nxfoil sampled once offline and refitted as a posynomial -- least squares,
   and constrained to lie above the samples.

The measurement that matters for (3) is not the fitting error but the
violation of the *true* relation at the returned optimum, obtained by calling
nxfoil there. A least-squares fit sits below the truth about half the time, and
the optimizer walks into the gap.

Run:  python examples/nxfoil_hoburg.py
"""
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/Dropbox/research'))
sys.path.insert(0, os.path.expanduser(
    '~/Dropbox/publications/2026/05_helicopter_GP'))
warnings.simplefilter('ignore')

import slcp_cases as C                                          # noqa: E402
from lcsolver.fitting import fit_max_affine, fit_report               # noqa: E402
from lcsolver.solvers.ipopt.slcp import Constraint, Problem, Signomial  # noqa: E402
from lcsolver.solvers.ipopt.sia import SIAOptions, solve_sia          # noqa: E402

#: Stated explicitly rather than taken from a default. The two nxfoil entry
#: points disagree -- `get_aero_from_kulfan_parameters` defaults to 'xlarge',
#: `helicopter_nxfoil.nxfoil_cd` to 'xxxlarge' -- so leaving it implicit makes
#: this example and `nxfoil_naca4_hoburg` silently different models.
MODEL_SIZE = 'xxxlarge'

#: nxfoil builds its network in float32 by default. That is fine for
#: evaluation and fatal for optimization: a central difference in float32 is
#: usable only down to a step of about 1e-4 and returns *exactly zero* at 1e-7,
#: because the two evaluations quantise to the same number. A gradient that is
#: zero to seven digits cannot certify a KKT residual of 1e-6 -- the solve sits
#: at |d| ~ 1e-10 with stationarity bouncing between 1e-5 and 0.9, learning
#: curvature from rounding error until the sub-problem goes infeasible.
#:
#: In float64 the same difference is flat to seven digits from h = 1e-2 to
#: 1e-7. This is a precision setting, not a device one -- the network already
#: runs on the CPU.
PRECISION = 'float64'
_PRECISION_SET = [False]


def _ensure_precision():
    if _PRECISION_SET[0]:
        return
    from metafoil import nxfoil
    net = nxfoil.get_net(MODEL_SIZE, 'cpu')
    net.double() if PRECISION == 'float64' else net.float()
    _PRECISION_SET[0] = True

STATS = {'evals': 0, 'calls': 0}
_SECTION_CACHE = {}


def _section(tau):
    """NACA 24xx Kulfan section of the requested thickness-to-chord."""
    key = round(float(tau), 7)
    if key not in _SECTION_CACHE:
        from metafoil.core.kulfan import Kulfan
        afl = Kulfan().naca4_like(2, 4, float(tau) * 100.0)
        _SECTION_CACHE[key] = dict(
            upper_weights=np.asarray(afl.upperCoefficients, dtype=float),
            lower_weights=np.asarray(afl.lowerCoefficients, dtype=float),
            leading_edge_weight=0.0, TE_thickness=0.0)
    return _SECTION_CACHE[key]


def _chain(tau):
    """NACA 24xx of the requested thickness: Kulfan weights and dK/dtau."""
    from helicopter_naca4 import shape_K
    K, J = shape_K([2.0, 0.4, float(tau) * 100.0])
    return K, J[:, 2] * 100.0        # d/dtau, not d/d(thickness percent)


def cd_at(tau, Re, C_L):
    """Profile drag at a requested lift coefficient."""
    from helicopter_nxfoil import nxfoil_cd
    _ensure_precision()
    STATS['calls'] += 1
    K, _ = _chain(tau)
    return float(nxfoil_cd(K, float(C_L), float(Re), model_size=MODEL_SIZE)[0])


def profile_drag(C_L, Re, tau):
    """``C_Dp`` and its three first derivatives, all exact.

    No finite differences of the network and no angle-of-attack inversion.
    ``nxfoil_cd`` differentiates the network by autograd and takes the
    derivatives *at fixed lift* through the implicit function theorem, so the
    angle of attack moving to hold C_L is already accounted for. The only
    difference taken anywhere is dK/dtau across the Kulfan geometry map, which
    is a smooth analytic function and costs no network evaluations.

    The earlier version differenced `cd_at` three times over, which needed
    seven evaluations per gradient, each one a secant inversion for alpha --
    seventy network calls where this needs one -- and carried both truncation
    error and, in float32, quantisation error large enough to stop the solve
    certifying at all.
    """
    from helicopter_nxfoil import nxfoil_cd
    _ensure_precision()
    STATS['evals'] += 1
    STATS['calls'] += 1
    K, dK_dtau = _chain(tau)
    cd, dcd_dK, dcd_dcl, dcd_dRe = nxfoil_cd(K, float(C_L), float(Re),
                                             model_size=MODEL_SIZE)
    d_tau = float(np.asarray(dcd_dK, dtype=float) @ dK_dtau)
    return float(cd), float(dcd_dcl), float(dcd_dRe), d_tau


# ---------------------------------------------------------------------------
# the three ways of supplying the same relation
# ---------------------------------------------------------------------------
def _blackbox_constraint(names, seg):
    """``C_Dp_seg >= nxfoil(C_L_seg, Re_seg, tau)``, linearized each iterate."""
    icl, ire = names.index(f'C_L_{seg}'), names.index(f'Re_{seg}')
    ita, icd = names.index('tau'), names.index(f'C_Dp_{seg}')
    n = len(names)

    def fn(x):
        x = np.asarray(x, dtype=float)
        cdp, d_cl, d_re, d_tau = profile_drag(x[icl], x[ire], x[ita])
        g = np.zeros(n)
        g[icl] = d_cl / x[icd]
        g[ire] = d_re / x[icd]
        g[ita] = d_tau / x[icd]
        g[icd] = -cdp / x[icd] ** 2
        return cdp / x[icd], g

    return Constraint(Signomial(fn, n), '<=')


def _fitted_constraints(fit, names, seg):
    """The same relation as ``K`` monomial rows -- the exact class."""
    out = []
    for k in range(fit['K']):
        e = fit['e'][k]
        out.append(Constraint(C.posy(
            [(fit['c'][k], {f'C_L_{seg}': e[0], f'Re_{seg}': e[1],
                            'tau': e[2], f'C_Dp_{seg}': -1.0})], names), '<='))
    return out


def build(kind, fit=None, segments=(0, 1, 2)):
    names = list(C.HOBURG_NAMES)
    obj = C.posy([(1.0, {'W_fuel_out': 1}), (1.0, {'W_fuel_ret': 1})], names)
    base = C._hoburg_constraints(names)
    cons = base[:-C._HOB_SEG]
    for seg in range(C._HOB_SEG):
        if seg not in segments:
            cons.append(base[-C._HOB_SEG + seg])
        elif kind == 'blackbox':
            cons.append(_blackbox_constraint(names, seg))
        elif kind == 'fit':
            cons += _fitted_constraints(fit, names, seg)
        else:
            raise ValueError(kind)
    return Problem(len(names), obj, cons, names=names)


def sample(x0, names, n_cl=14, n_re=8, n_tau=6, spread=1.6):
    """Sample nxfoil over a box around the starting point."""
    icl = [names.index(f'C_L_{i}') for i in range(3)]
    ire = [names.index(f'Re_{i}') for i in range(3)]
    ita = names.index('tau')
    cls = np.geomspace(x0[icl].min() / spread, x0[icl].max() * spread, n_cl)
    res = np.geomspace(x0[ire].min() / spread, x0[ire].max() * spread, n_re)
    tas = np.geomspace(x0[ita] / 1.4, x0[ita] * 1.4, n_tau)
    X, y = [], []
    for ta in tas:
        for re in res:
            for cl in cls:
                X.append((cl, re, ta))
                y.append(cd_at(ta, re, cl))
    return np.array(X), np.array(y)


def true_violation(x, names):
    """Worst log violation of the real nxfoil relation at ``x``."""
    worst = -np.inf
    for seg in range(3):
        cdp = x[names.index(f'C_Dp_{seg}')]
        truth = cd_at(x[names.index('tau')], x[names.index(f'Re_{seg}')],
                      x[names.index(f'C_L_{seg}')])
        worst = max(worst, np.log(truth / cdp))
    return worst


def run(problem, x0):
    t = time.time()
    r = solve_sia(problem, x0.copy(), SIAOptions(verbose=False))
    return dict(t=time.time() - t, obj=float(r.objective),
                it=int(r.iterations), conv=bool(r.converged),
                x=np.asarray(r.x, dtype=float))


if __name__ == '__main__':
    names = list(C.HOBURG_NAMES)
    p_ref, x0, _ = C.hoburg(0)

    t0 = time.time()
    profile_drag(0.5, 3e6, 0.12)
    print(f'nxfoil: first evaluation {time.time() - t0:.2f}s (model load), '
          f'{PRECISION}')
    t0 = time.time()
    for i in range(50):
        profile_drag(0.4 + 0.004 * i, 3e6, 0.12)
    print(f'        {50 / (time.time() - t0):.0f} value+gradient/s thereafter, '
          f'exact by autograd')
    print()

    ref = run(p_ref, x0)
    print(f"{'reference: five-term posynomial fit':52s} "
          f"obj {ref['obj']:9.2f}  it {ref['it']:3d}  {ref['t']:6.2f}s")

    STATS.update(evals=0, calls=0)
    bb = run(build('blackbox'), x0)
    print(f"{'nxfoil as a black box, linearized':52s} "
          f"obj {bb['obj']:9.2f}  it {bb['it']:3d}  {bb['t']:6.2f}s"
          f"  converged={bb['conv']}")
    print(f"{'':52s} {STATS['evals']} black-box evaluations "
          f"({STATS['calls']} nxfoil calls)")
    print(f"{'':52s} true violation {true_violation(bb['x'], names):+.2e}")
    print()

    t0 = time.time()
    X, y = sample(x0, names)
    print(f'sampled nxfoil at {len(y)} points in {time.time() - t0:.2f}s')
    print()

    for mode in (None, 'shift'):
        fit = fit_max_affine(X, y, K=4, conservative=mode)
        tag = 'least squares' if mode is None else 'conservative (shift)'
        r = run(build('fit', fit), x0)
        tv = true_violation(r['x'], names)
        print(f"nxfoil refitted, {tag:22s} "
              f"obj {r['obj']:9.2f}  it {r['it']:3d}  {r['t']:6.2f}s"
              f"  converged={r['conv']}")
        print(f"{'':52s} vs reference {100 * (r['obj'] / ref['obj'] - 1):+6.2f}%"
              f"   true violation {tv:+.2e}"
              + ('  INFEASIBLE' if tv > 1e-6 else '  feasible'))
        for line in fit_report(fit).splitlines()[1:]:
            print(f"{'':52s} {line.strip()}")
        print()
