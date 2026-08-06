"""Hoburg UAV with the airfoil designed alongside the aircraft.

``nxfoil_hoburg`` fixes the section at a NACA 24xx and lets the optimizer
choose only its thickness. This adds the other two NACA 4-digit digits --
camber magnitude and camber position -- as design variables, so the airfoil is
sized by the same solve that sizes the aircraft:

    minimize   fuel
    subject to the UAV constraints, and
               C_Dp_i = nxfoil( NACA4(m, p, t), C_L_i, Re_i )

The framework follows ``helicopter_naca4.py`` from the helicopter paper, which
does the same thing for a rotor blade section, and reuses its two pieces:

``shape_K(z)``
    ``z = [m, p, t]`` to the sixteen Kulfan weights, with the geometry Jacobian
    by central differences. Geometry only -- no network evaluations.
``nxfoil_cd(K16, cl, Re)``
    drag and its exact derivatives from torch autograd, taken at fixed ``cl``
    through the implicit function theorem, so ``dC_D/dK`` already accounts for
    the angle of attack moving to hold lift.

Chaining the two gives ``dC_Dp/d(m, p, t)`` exactly, which is what makes the
shape variables usable: a finite difference through the whole geometry-plus-
network chain is both slower and noisier, and a black box whose gradient is
quietly wrong does not fail loudly -- it returns a confident, infeasible
answer.

The same three routes as ``nxfoil_hoburg`` are compared: the section behind a
black box and linearized, and the section sampled once and refitted as a
posynomial in five variables, least squares and conservative.

Run:
    python examples/nxfoil_naca4_hoburg.py
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

import slcp_cases as C                                            # noqa: E402
from lcsolver.fitting import fit_max_affine, fit_report                # noqa: E402
from lcsolver.solvers.ipopt.slcp import Constraint, Problem, Signomial   # noqa: E402
from lcsolver.solvers.ipopt.sia import SIAOptions, solve_sia            # noqa: E402

#: Explicit, for the reason given in `nxfoil_hoburg`: the two nxfoil entry
#: points have different defaults, and the examples must agree to be compared.
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

STATS = {'evals': 0}

#: Hoburg's variables plus the two extra NACA-4 digits. ``tau`` is already in
#: the Hoburg list as thickness-to-chord, so it serves as the third digit.
NAMES = list(C.HOBURG_NAMES) + ['m_c', 'p_c']
X0 = np.concatenate([C.HOBURG_X0, [2.0, 0.4]])   # a NACA 24xx to start from


def _chain(m, p, tau):
    from helicopter_naca4 import shape_K
    return shape_K([float(m), float(p), float(tau) * 100.0])


def profile_drag(C_L, Re, tau, m, p):
    """``C_Dp`` and its five first derivatives, all exact but the geometry."""
    from helicopter_nxfoil import nxfoil_cd
    _ensure_precision()
    STATS['evals'] += 1
    K, J = _chain(m, p, tau)
    cd, dcd_dK, dcd_dcl, dcd_dRe = nxfoil_cd(K, float(C_L), float(Re),
                                             model_size=MODEL_SIZE)
    dz = np.asarray(dcd_dK, dtype=float) @ J        # d/d(m, p, t_percent)
    return (float(cd), float(dcd_dcl), float(dcd_dRe),
            float(dz[2]) * 100.0,                   # d/dtau, not d/dt_percent
            float(dz[0]), float(dz[1]))


def cd_only(C_L, Re, tau, m, p):
    from helicopter_nxfoil import nxfoil_cd
    _ensure_precision()
    K, _ = _chain(m, p, tau)
    return float(nxfoil_cd(K, float(C_L), float(Re),
                           model_size=MODEL_SIZE)[0])


# ---------------------------------------------------------------------------
def _blackbox_constraint(seg):
    icl, ire = NAMES.index(f'C_L_{seg}'), NAMES.index(f'Re_{seg}')
    ita, icd = NAMES.index('tau'), NAMES.index(f'C_Dp_{seg}')
    imc, ipc = NAMES.index('m_c'), NAMES.index('p_c')
    n = len(NAMES)

    def fn(x):
        x = np.asarray(x, dtype=float)
        cdp, d_cl, d_re, d_tau, d_m, d_p = profile_drag(
            x[icl], x[ire], x[ita], x[imc], x[ipc])
        g = np.zeros(n)
        g[icl], g[ire], g[ita] = d_cl / x[icd], d_re / x[icd], d_tau / x[icd]
        g[imc], g[ipc] = d_m / x[icd], d_p / x[icd]
        g[icd] = -cdp / x[icd] ** 2
        return cdp / x[icd], g

    return Constraint(Signomial(fn, n), '<=')


def _fitted_constraints(fit, seg):
    """The five-variable fit as ``K`` monomial rows -- the exact class."""
    out = []
    for k in range(fit['K']):
        e = fit['e'][k]
        out.append(Constraint(C.posy(
            [(fit['c'][k], {f'C_L_{seg}': e[0], f'Re_{seg}': e[1],
                            'tau': e[2], 'm_c': e[3], 'p_c': e[4],
                            f'C_Dp_{seg}': -1.0})], NAMES), '<='))
    return out


def build(kind, fit=None):
    obj = C.posy([(1.0, {'W_fuel_out': 1}), (1.0, {'W_fuel_ret': 1})], NAMES)
    base = C._hoburg_constraints(NAMES)
    cons = base[:-C._HOB_SEG]
    for seg in range(C._HOB_SEG):
        cons += ([_blackbox_constraint(seg)] if kind == 'blackbox'
                 else _fitted_constraints(fit, seg))
    # Keep the section inside the range the NACA4 map and the network are
    # meaningful over. p -> 1 is singular: the camber line carries (1-p)^2 in
    # its denominator.
    for nm, lo, hi in (('m_c', 0.5, 8.0), ('p_c', 0.15, 0.75),
                       ('tau', 0.06, 0.20)):
        cons.append(Constraint(C.posy([(lo, {nm: -1.0})], NAMES), '<='))
        cons.append(Constraint(C.posy([(1.0 / hi, {nm: 1.0})], NAMES), '<='))
    return Problem(len(NAMES), obj, cons, names=NAMES)


def sample(n_cl=8, n_re=4, n_tau=4, n_m=5, n_p=4):
    icl = [NAMES.index(f'C_L_{i}') for i in range(3)]
    ire = [NAMES.index(f'Re_{i}') for i in range(3)]
    grid = [np.geomspace(X0[icl].min() / 1.5, X0[icl].max() * 1.5, n_cl),
            np.geomspace(X0[ire].min() / 1.5, X0[ire].max() * 1.5, n_re),
            np.geomspace(0.07, 0.18, n_tau),
            np.geomspace(0.8, 7.0, n_m),
            np.geomspace(0.2, 0.7, n_p)]
    X, y = [], []
    for ta in grid[2]:
        for m in grid[3]:
            for p in grid[4]:
                for re in grid[1]:
                    for cl in grid[0]:
                        X.append((cl, re, ta, m, p))
                        y.append(cd_only(cl, re, ta, m, p))
    return np.array(X), np.array(y)


def true_violation(x):
    worst = -np.inf
    for seg in range(3):
        cdp = x[NAMES.index(f'C_Dp_{seg}')]
        truth = cd_only(x[NAMES.index(f'C_L_{seg}')],
                        x[NAMES.index(f'Re_{seg}')], x[NAMES.index('tau')],
                        x[NAMES.index('m_c')], x[NAMES.index('p_c')])
        worst = max(worst, np.log(truth / cdp))
    return worst


def section(x):
    m, p, t = (x[NAMES.index('m_c')], x[NAMES.index('p_c')],
               x[NAMES.index('tau')] * 100.0)
    return f'NACA {m:.2f}/{p:.3f}/{t:.2f}  (~NACA {round(m):.0f}' \
           f'{round(p*10):.0f}{round(t):02.0f})'


def run(problem, x0, max_iterations=200):
    t = time.time()
    r = solve_sia(problem, x0.copy(),
                  SIAOptions(verbose=False, max_iterations=max_iterations))
    return dict(t=time.time() - t, obj=float(r.objective), it=int(r.iterations),
                conv=bool(r.converged), x=np.asarray(r.x, dtype=float))


if __name__ == '__main__':
    print(f'design variables: {len(NAMES)}  '
          f'(Hoburg {len(C.HOBURG_NAMES)} + camber m_c, camber position p_c)')
    print(f'starting section: {section(X0)}')
    print()

    STATS['evals'] = 0
    bb = run(build('blackbox'), X0)
    print(f"black box, linearized      obj {bb['obj']:9.2f}  it {bb['it']:3d}  "
          f"{bb['t']:7.2f}s  converged={bb['conv']}")
    print(f"                           {STATS['evals']} black-box evaluations")
    print(f"                           {section(bb['x'])}")
    print(f"                           true violation {true_violation(bb['x']):+.2e}")
    print()

    t0 = time.time()
    X, y = sample()
    print(f'sampled nxfoil at {len(y)} points in {time.time() - t0:.1f}s '
          f'(5-D: C_L, Re, tau, m, p)')
    print()

    for mode in (None, 'shift'):
        fit = fit_max_affine(X, y, K=6, conservative=mode)
        tag = 'least squares' if mode is None else 'conservative'
        try:
            r = run(build('fit', fit), X0)
            tv = true_violation(r['x'])
            print(f"refitted, {tag:14s}   obj {r['obj']:9.2f}  it {r['it']:3d}  "
                  f"{r['t']:7.2f}s  converged={r['conv']}")
            print(f"                           {section(r['x'])}")
            print(f"                           true violation {tv:+.2e}"
                  + ('  INFEASIBLE' if tv > 1e-6 else '  feasible'))
        except Exception as exc:
            print(f"refitted, {tag:14s}   solve failed: "
                  f"{type(exc).__name__}: {exc}")
        for line in fit_report(fit).splitlines()[1:]:
            print(f'                           {line.strip()}')
        print()
