#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""XFOIL tied directly into the Hoburg formulation, paper Section 7.6.

Once all three profile-drag constraints are black boxes, the posynomial surrogate
has been eliminated entirely and a higher-fidelity model can simply be swapped in.
This module is that swap: ``C_Dp = f(C_L, Re, tau)`` is evaluated by running
XFOIL on a NACA 24xx section of the requested thickness.

Unlike the implicit-fit black box, this **changes the answer**. The paper reports
the resulting shifts in Table 3 (``t:hoburg_x``) and notes that SLCP reached the
new optimum in 32 iterations starting from the GP solution.

Procedure, following ``XfoilDrag.py``:

1. Build a NACA 24xx Kulfan section with ``tau`` as the thickness.
2. Sweep alpha and collect the (cl, cd) polar.
3. Fit a Gaussian process to cd(cl) over the monotone-increasing branch, and
   evaluate it at the requested ``C_L``. The fit both smooths XFOIL's noise and
   supplies ``dC_Dp/dC_L`` analytically from the GP posterior mean.
4. ``dC_Dp/dRe`` and ``dC_Dp/dtau`` by central differences (0.5% and 5% steps),
   which is the finite-difference scheme the paper describes.

The reference implementation shells out to an XFOIL binary per evaluation. Here
metafoil's in-memory XFOIL is used instead, which is what makes the run tractable
-- one polar takes ~2 s, and a gradient needs five of them.
"""

import numpy as np

_SWEEP_CACHE = {}
_STATS = {'sweeps': 0, 'cache_hits': 0}


def _polar(thickness_pct, Re, alpha_min=-10.0, alpha_max=25.0, alpha_step=1.0):
    """XFOIL polar for a NACA 24xx of the given thickness, with caching.

    Cached on the rounded (thickness, Re) pair. The cache matters: within one
    SLCP iteration every segment shares ``tau``, and the line search revisits
    nearby points repeatedly.
    """
    key = (round(float(thickness_pct), 6), round(float(Re), 0))
    if key in _SWEEP_CACHE:
        _STATS['cache_hits'] += 1
        return _SWEEP_CACHE[key]

    from metafoil.core.kulfan import Kulfan
    import metafoil.xfoil as xf

    afl = Kulfan().naca4_like(2, 4, float(thickness_pct))
    out = xf.run_sweep(afl.upperCoefficients, afl.lowerCoefficients,
                       alpha_min, alpha_max, alpha_step,
                       Re=float(Re), max_iter=20)
    _STATS['sweeps'] += 1

    cl = np.asarray(out['cl'], dtype=float)
    cd = np.asarray(out['cd'], dtype=float)

    # Keep only the monotonically increasing-cl branch, as XfoilDrag.py does:
    # past stall the polar doubles back and cd(cl) stops being a function.
    keep = []
    prev = -np.inf
    for i in range(len(cl)):
        if np.isfinite(cl[i]) and np.isfinite(cd[i]) and cl[i] > prev:
            keep.append(i)
            prev = cl[i]
    cl, cd = cl[keep], cd[keep]
    if len(cl) < 5:
        raise RuntimeError(
            f'XFOIL returned only {len(cl)} usable polar points at '
            f'thickness={thickness_pct:.3f}%, Re={Re:.3e}')

    _SWEEP_CACHE[key] = (cl, cd)
    return cl, cd


_CDREF = 0.01          # normalization used by the reference implementation


def _fit(cl, cd):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    kernel = ConstantKernel() * Matern(nu=1.5) + WhiteKernel()
    return GaussianProcessRegressor(kernel=kernel).fit(
        cl.reshape(-1, 1), (cd / _CDREF).ravel())


def _predict(gp, value):
    """Scalar prediction. sklearn returns (n,) or (n,1) depending on the fit
    shape and version, so flatten rather than assuming either."""
    return float(np.asarray(gp.predict([[value]])).ravel()[0]) * _CDREF


def _cd_at(thickness_pct, Re, C_L):
    cl, cd = _polar(thickness_pct, Re)
    if not (cl.min() <= C_L <= cl.max()):
        raise ValueError(
            f'requested C_L={C_L:.4f} is outside the achievable range '
            f'[{cl.min():.4f}, {cl.max():.4f}] at thickness={thickness_pct:.3f}%, '
            f'Re={Re:.3e}')
    return _predict(_fit(cl, cd), C_L)


def xfoil_profile_drag(C_L, Re, tau):
    """``C_Dp = f(C_L, Re, tau)`` from XFOIL, with first derivatives.

    Returns ``(C_Dp, dC_Dp/dC_L, dC_Dp/dRe, dC_Dp/dtau)``. ``tau`` is a
    thickness-to-chord ratio; XFOIL wants percent, hence the factor of 100.
    """
    t_pct = float(tau) * 100.0
    cl, cd = _polar(t_pct, Re)
    gp = _fit(cl, cd)

    if not (cl.min() <= C_L <= cl.max()):
        raise ValueError(
            f'requested C_L={C_L:.4f} is outside the achievable range '
            f'[{cl.min():.4f}, {cl.max():.4f}] at tau={tau:.4f}, Re={Re:.3e}')

    C_Dp = _predict(gp, C_L)

    # dC_Dp/dC_L from the fitted surface -- no extra XFOIL runs needed.
    h = 1e-6 * max(abs(C_L), 1.0)
    up = _predict(gp, C_L + h)
    dn = _predict(gp, C_L - h)
    d_cl = (up - dn) / (2.0 * h)

    # dC_Dp/dRe and dC_Dp/dtau by central differences: two polars each.
    re_step = 0.005
    d_re = ((_cd_at(t_pct, Re * (1 + re_step), C_L)
             - _cd_at(t_pct, Re * (1 - re_step), C_L))
            / (2.0 * re_step * Re))

    t_step = 0.05
    d_tau = ((_cd_at(t_pct * (1 + t_step), Re, C_L)
              - _cd_at(t_pct * (1 - t_step), Re, C_L))
             / (2.0 * t_step * tau))

    return C_Dp, d_cl, d_re, d_tau


def hoburg_xfoil(segments=(0, 1, 2)):
    """Hoburg with XFOIL supplying profile drag on the named segments.

    Defaults to all three, matching the paper's demonstration.
    """
    from lcsolver.solvers.ipopt.slcp import Constraint, Problem, Signomial

    from examples.slcp_cases import (HOBURG_NAMES, HOBURG_X0, _HOB_SEG,
                                     _hoburg_constraints, posy)

    N = HOBURG_NAMES

    def make(seg):
        icl, ire = N.index(f'C_L_{seg}'), N.index(f'Re_{seg}')
        ita, icd = N.index('tau'), N.index(f'C_Dp_{seg}')
        n = len(N)

        def fn(x):
            x = np.asarray(x, dtype=float)
            cdp, d_cl, d_re, d_tau = xfoil_profile_drag(x[icl], x[ire], x[ita])
            g = np.zeros(n)
            g[icl] = d_cl / x[icd]
            g[ire] = d_re / x[icd]
            g[ita] = d_tau / x[icd]
            g[icd] = -cdp / x[icd] ** 2
            return cdp / x[icd], g

        return Constraint(Signomial(fn, n), '<=')

    base = _hoburg_constraints(N)
    constraints = base[:-_HOB_SEG] + [
        (make(i) if i in segments else base[-_HOB_SEG + i])
        for i in range(_HOB_SEG)
    ]
    objective = posy([(1.0, {'W_fuel_out': 1}), (1.0, {'W_fuel_ret': 1})], N)
    return Problem(len(N), objective, constraints, names=N), HOBURG_X0.copy(), None


def stats():
    """XFOIL call counts, for reporting the cost of a run."""
    return dict(_STATS, cached_polars=len(_SWEEP_CACHE))
