"""Fit pycycle's compressor/turbine maps as signomial surfaces.

Milestone 3 groundwork: the off-design SP rows need the SAME scaled-map
physics pycycle uses -- the maps are DATA (NPSS-heritage tables), and
pycycle interpolates them with structured 'slinear'. This generator samples
that exact interpolant (openmdao InterpND, method='slinear') on a fine grid
over the operating window, fits each output as a free-sign signomial in the
map coordinates, and writes ``components/turbofan/sp_maps.py``.

Map structure (see pycycle/elements/{compressor,turbine}.py):

* Compressors (FanMap/LPCMap/HPCMap): outputs (WcMap, PRmap, effMap) on a
  (NcMap, RlineMap) grid, alpha slice 0. Scaling: s_Nc = Nc/NcMap,
  s_PR = (PR-1)/(PRmap-1), s_eff = eff/effMap, s_Wc = Wc/WcMap, with the
  scalars set at the design point against the map read at its 'defaults'
  coordinates.
* Turbines (HPTMap/LPTMap): outputs (WpMap, effMap) on a (NpMap, PRmap)
  grid, alpha slice per defaults. Same scalar pattern (s_PR on PR-1).

Fit windows cover the four anchor off-design points (SLS takeoff, rolling
takeoff, top of climb, part-power cruise) with margin; the residual report
prints on-grid and off-grid (midpoint) errors. slinear's kinks bound how
well a smooth surface can track the table -- sub-1% is the target, and the
end-to-end OD validation is the real acceptance.

Run:  python -m components.turbofan.truth.fits_maps
"""
from __future__ import annotations

import numpy as np

import pycycle.api as pyc
from openmdao.components.interp_util.interp import InterpND

from .fits import fit_signomial_2d

sys_path_root = "/Users/codykarcher/Dropbox/research/edi"
import sys as _sys
if sys_path_root not in _sys.path:
    _sys.path.insert(0, sys_path_root)
from edi.fitting import fit_max_affine, evaluate_fit


def fit_sma(X, y, K=8, alphas=(1., 2., 4., 8., 16., 32.),
            force_direct=False):
    """SMA fit of a positive 2-D surface, trying both y and 1/y.

    An SMA equality enters the model as monomial == posynomial -- ZERO
    internal cancellation, the property the free-design solves need
    (free-sign signomial surfaces measured 10-30x cancellation; the SIA
    verified but could not NAVIGATE them). Pipeline: house max-affine
    planes seed a Gauss-Newton/LM refinement of the full SMA (softmax
    responsibilities give the analytic Jacobian) at EACH candidate
    softness; the winner is chosen on post-refinement residual. alpha=1
    is a plain posynomial. Returns (fit_dict, inv_flag, max_log_err).
    """
    import numpy as _np
    u_all = _np.log(_np.atleast_2d(X))
    m, dd = u_all.shape

    def lm(E, B, a, w):
        """SMA refinement via scipy least_squares (the hand-rolled damped
        GN collapsed distinct planes even on exact-answer problems; scipy
        recovers x^2+x^5 to 2e-15 from the same seed)."""
        from scipy.optimize import least_squares as _lsq
        Kp = E.shape[0]

        def unpack(th):
            return th[:Kp * dd].reshape(Kp, dd), th[Kp * dd:]

        def resid(th):
            E_, B_ = unpack(th)
            z = a * (u_all @ E_.T + B_)
            zmax = z.max(axis=1, keepdims=True)
            f = (zmax.ravel()
                 + _np.log(_np.exp(z - zmax).sum(axis=1))) / a
            return f - w

        def jac(th):
            E_, B_ = unpack(th)
            z = a * (u_all @ E_.T + B_)
            zmax = z.max(axis=1, keepdims=True)
            ez = _np.exp(z - zmax)
            resp = ez / ez.sum(axis=1)[:, None]
            J = _np.empty((m, Kp * (dd + 1)))
            for k in range(Kp):
                J[:, k * dd:(k + 1) * dd] = resp[:, [k]] * u_all
                J[:, Kp * dd + k] = resp[:, k]
            return J

        th0 = _np.concatenate([E.ravel(), B])
        r = _lsq(resid, th0, jac=jac, method='lm', max_nfev=4000)
        E_, B_ = unpack(r.x)
        return E_, B_, float(_np.max(_np.abs(r.fun)))

    def _tangent_seed(w, Kp):
        """K local-tangent planes at spread points: on nearly-straight
        log-log data the max-affine seed collapses to identical planes and
        LM cannot split them (measured: stuck at single-monomial quality).
        Local weighted least squares at quantiles of the first input gives
        distinct slopes by construction."""
        order = _np.argsort(u_all[:, 0])
        centers = [order[int((j + 0.5) * m / Kp)] for j in range(Kp)]
        scale = (u_all[:, 0].max() - u_all[:, 0].min()) / max(Kp, 2)
        E0 = _np.zeros((Kp, dd)); B0 = _np.zeros(Kp)
        for j, ci in enumerate(centers):
            r2 = _np.sum((u_all - u_all[ci])**2, axis=1)
            wt = _np.exp(-r2 / max(2*scale*scale, 1e-8))
            A = _np.hstack([u_all, _np.ones((m, 1))]) * wt[:, None]
            z = _np.linalg.lstsq(A, w * wt, rcond=None)[0]
            E0[j] = z[:-1]; B0[j] = z[-1]
        return E0, B0

    def sma_of(y_):
        w = _np.log(y_)
        E0, B0 = _tangent_seed(w, K)
        emax0 = float(_np.max(_np.abs(E0)))
        best = None
        for a in alphas:
            if a * max(emax0, 1.0) > 80.0:
                continue
            # center the seed: the softened max-affine start sits ln(K)/a
            # above the data on a rank-deficient responsibility plateau LM
            # cannot leave (psi fits measured stuck at exactly ln K)
            B0a = B0 - _np.log(max(len(B0), 1)) / a
            E, B, err = lm(E0.copy(), B0a.copy(), a, w)
            if float(_np.max(_np.abs(E))) * a > 80.0:
                continue      # refined slopes would overflow the rows
            if best is None or err < best[0]:
                best = (err, a, E, B)
        if best is None:
            a = max(1.0, 80.0 / max(emax0, 1.0))
            B0a = B0 - _np.log(max(len(B0), 1)) / a
            E, B, err = lm(E0.copy(), B0a.copy(), a, w)
            best = (err, a, E, B)
        err, a, E, B = best
        fit = {'ftype': 'SMA', 'K': E.shape[0], 'd': dd, 'a1': float(a),
               'c': [float(_np.exp(b)) for b in B],
               'e': [[float(v) for v in row] for row in E],
               'rms_err': err, 'max_err': err, 'conservative': None}
        return fit, err

    f1, e1 = sma_of(y)
    if force_direct:
        return f1, False, e1
    f2, e2 = sma_of(1.0 / y)
    if e1 <= e2:
        return f1, False, e1
    return f2, True, e2


#: (map, kind, window) per cycle component. Windows chosen to cover the
#: anchors' OD excursions: SLS static pushes corrected speed HIGH on the
#: fan (cold day would push higher; ISA SLS sits ~1.05) and part-power
#: cruise pushes it low.
# Windows sized from the anchors' MEASURED excursions (truth JSONs: fan Nc
# 0.945-1.024, lpc 0.960-1.029, hpc 0.973-0.989 across both engines and all
# four OD points), widened toward low corrected speed for the mission's
# part-power descent legs. Tight windows are what make sub-1% signomial
# fits of a kinked slinear table possible.
# R windows run to the table's own 3.0 ceiling: the anchors NEED the top --
# inverting the fitted Wc surfaces against truth puts the GEnx takeoff HPC
# at R = 2.77 and the CFM56 top-of-climb fan at 2.71, and a 2.70 window
# bound left phase-1 with a 2.9%-inconsistent row it could only slack.
# NOTE: extending windows past the table edges (R 3.3, Nc 1.15) was tried
# for the mission's high-corrected-speed climb states and REVERTED: the
# slinear eff extrapolation goes wild (878% fit error in the skirt) and
# the mid-climb stall it was meant to fix did not move -- the R=3.0 pinch
# was a symptom of a garbage attractor, not its cause.
COMPRESSORS = {
    'fan': (pyc.FanMap, dict(Nc=(0.80, 1.10), R=(1.40, 3.00))),
    'lpc': (pyc.LPCMap, dict(Nc=(0.80, 1.10), R=(1.40, 3.00))),
    'hpc': (pyc.HPCMap, dict(Nc=(0.88, 1.06), R=(1.40, 3.00))),
}
# Turbine windows are in MAP coordinates, not actual PRs: s_PR absorbs the
# difference, and the map coordinate stays glued near its design default
# (HPT PRmap_d 6.0 -> anchors sit 5.9-6.1; LPT ~3.9-4.05 computed through
# s_PR from the truth PRs). The first windows were sized on actual PRs
# (3-5), put the operating point OUTSIDE the fit, and the multipoint solve
# went infeasible against the window bounds.
# Widened again for the RUBBER engine: with the design anchor at
# part-power cruise (deck Tt4 1360 K, not the 1587 K rating) the map
# coordinate frame shifts, and the mission's high-thrust climb points need
# LPT PRmap below the old 5.0 floor (s0_lpt_PRmap sat ON the bound with a
# persistent 4.3e-3 row violation and 1e9 complementarity).
TURBINES = {
    'hpt': (pyc.HPTMap, dict(Np=(80.0, 120.0), PR=(4.0, 8.0))),
    'lpt': (pyc.LPTMap, dict(Np=(80.0, 120.0), PR=(3.0, 8.0))),
}


def _interp(map_data, field, alpha_idx=0):
    if field in ('WcMap', 'PRmap', 'effMap') and hasattr(map_data, 'RlineMap'):
        pts = (map_data.NcMap, map_data.RlineMap)
    else:
        pts = (map_data.NpMap, map_data.PRmap)
    vals = getattr(map_data, field)[alpha_idx]
    return InterpND(method='slinear', points=pts, values=vals,
                    extrapolate=True)


def _sample(itp, xs, ys):
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel()])
    v = itp.interpolate(pts, compute_derivative=False)
    return X.ravel(), Y.ravel(), np.asarray(v).ravel()


def _fit_surface(label, itp, w1, w2, exps1, exps2, n=33, x0=None, y0=None,
                 core=None):
    """Ridge-regularized signomial surface fit.

    Free-sign LSQ on these near-collinear bases balances enormous cancelling
    coefficients (measured: sum|c| up to 6e9 on surfaces whose values are
    O(100)) -- exactly the detector-hostile p-q shape house rule 3 bans, and
    the SIA solver overflowed an iterate into inf/nan on the first wild
    step. Tikhonov damping with the smallest lambda that keeps the
    cancellation ratio sum|c|/median|y| under ~30 costs a little residual
    and returns rows the solver can actually chew: coordinates are
    normalized to mid-window so the basis is as orthogonal as it gets.
    ``core`` points get 10x fit weight."""
    x0 = x0 if x0 is not None else 0.5 * (w1[0] + w1[1])
    y0 = y0 if y0 is not None else 0.5 * (w2[0] + w2[1])
    xs = np.linspace(w1[0], w1[1], n)
    ys = np.linspace(w2[0], w2[1], n)
    X, Y, V = _sample(itp, xs, ys)
    wgt = np.ones_like(V)
    if core is not None:
        (xl, xh), (yl, yh) = core
        wgt[(X >= xl) & (X <= xh) & (Y >= yl) & (Y <= yh)] = 10.0
    pairs = [(a, b) for a in exps1 for b in exps2]
    A = np.vstack([(X / x0)**a * (Y / y0)**b for a, b in pairs]).T
    Wm = wgt / V
    Aw, bw = A * Wm[:, None], wgt
    scale = float(np.median(np.abs(V)))
    budget = 30.0 * scale

    best = None
    for lam in (0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        if lam == 0.0:
            c, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        else:
            k = Aw.shape[1]
            c, *_ = np.linalg.lstsq(
                np.vstack([Aw, lam * np.eye(k)]),
                np.concatenate([bw, np.zeros(k)]), rcond=None)
        tot = float(np.sum(np.abs(c)))
        res = float(np.max(np.abs((A @ c - V) / V)))
        if tot <= budget:
            best = (c, res, tot)
            break
        best = (c, res, tot)     # fall through with the largest lambda
    c, _, tot = best
    tlist = [(float(ck), float(a / 1.0), float(b / 1.0))
             for ck, (a, b) in zip(c, pairs) if ck != 0.0]
    # store normalized-coordinate terms: y = sum c*(x/x0)^a*(y/y0)^b
    xm = 0.5 * (xs[:-1] + xs[1:])
    ym = 0.5 * (ys[:-1] + ys[1:])
    Xm, Ym, Vm = _sample(itp, xm, ym)
    fit = sum(ck * (Xm / x0)**a * (Ym / y0)**b for ck, a, b in tlist)
    resm = np.abs(fit / Vm - 1.0)
    res_off = float(np.max(resm))
    if core is not None:
        (xl, xh), (yl, yh) = core
        inc = (Xm >= xl) & (Xm <= xh) & (Ym >= yl) & (Ym <= yh)
        res_core = float(np.max(resm[inc])) if inc.any() else float('nan')
    else:
        res_core = res_off
    return tlist, res_core, res_off, tot / scale, (x0, y0)


def generate(path=None):
    import io, pathlib, pprint
    out, report = {}, []

    exps_n = [0.0, 1.0, 2.0, 3.0]
    exps_r = [0.0, 1.0, 2.0, 3.0]

    for name, (mp, w) in COMPRESSORS.items():
        d = {}
        for field, key in (('WcMap', 'Wc'), ('PRmap', 'PR'),
                           ('effMap', 'eff')):
            itp = _interp(mp, field, alpha_idx=0)
            # per-component core boxes from the measured anchor excursions
            core = {'fan': ((0.93, 1.06), (1.7, 2.9)),
                    'lpc': ((0.94, 1.06), (1.7, 2.9)),
                    'hpc': ((0.96, 1.02), (1.7, 2.9))}[name]
            terms, res, res_off, cr, (x0, y0) = _fit_surface(
                f"{name}.{key}", itp, w['Nc'], w['R'], exps_n, exps_r,
                core=core)
            d[f'terms_{key}'] = terms
            d['x0'], d['y0'] = x0, y0
            report.append(f"{name}.{key:3s} signomial({len(terms):2d}) "
                          f"core {res:.2e} full {res_off:.2e} "
                          f"cancel {cr:.1f}")
            # SMA refit on the same window, core samples replicated 4x for
            # weight (the MA fitter is unweighted)
            xs_ = np.linspace(w['Nc'][0], w['Nc'][1], 33)
            ys_ = np.linspace(w['R'][0], w['R'][1], 33)
            Xs, Ys, Vs = _sample(itp, xs_, ys_)
            (cl, ch), (rl, rh) = core
            inc = (Xs >= cl) & (Xs <= ch) & (Ys >= rl) & (Ys <= rh)
            Xw = np.concatenate([Xs] + [Xs[inc]] * 3)
            Yw = np.concatenate([Ys] + [Ys[inc]] * 3)
            Vw = np.concatenate([Vs] + [Vs[inc]] * 3)
            XY = np.column_stack([Xw / x0, Yw / y0])
            fitd, inv, err = fit_sma(XY, Vw, K=8)
            # core residual of the SMA
            XYc = np.column_stack([Xs[inc] / x0, Ys[inc] / y0])
            pred = evaluate_fit(fitd, XYc)
            if inv:
                pred = 1.0 / pred
            resc = float(np.max(np.abs(np.log(pred / Vs[inc]))))
            d[f'sma_{key}'] = {k_: fitd[k_] for k_ in
                               ('ftype', 'a1', 'K', 'd', 'c', 'e')}
            d[f'sma_{key}']['inv'] = bool(inv)
            report.append(f"{name}.{key:3s} SMA(K={fitd['K']},a={fitd['a1']:.0f}"
                          f"{',inv' if inv else ''}) core {resc:.2e} "
                          f"full {err:.2e}")
        dflt = mp.defaults
        d['NcMap_d'] = float(dflt['NcMap'])
        d['RlineMap_d'] = float(dflt['RlineMap'])
        at = {}
        for field, key in (('WcMap', 'Wc'), ('PRmap', 'PR'),
                           ('effMap', 'eff')):
            itp = _interp(mp, field, alpha_idx=0)
            v = itp.interpolate(np.array([[d['NcMap_d'],
                                           d['RlineMap_d']]]),
                                compute_derivative=False)
            at[key] = float(np.asarray(v).ravel()[0])
        d['map_at_defaults'] = at
        d['window'] = {'Nc': list(w['Nc']), 'R': list(w['R'])}
        out[name] = d

    exps_np = [0.0, 1.0, 2.0]
    exps_pr = [0.0, 1.0, 2.0, 3.0]
    for name, (mp, w) in TURBINES.items():
        d = {}
        alpha_idx = int(mp.defaults.get('alphaMap', 0))
        # alphaMap slice: defaults say 1.0 for these maps; index the slice
        # that matches (the alpha grid is [0, 1] for both turbine maps).
        for field, key in (('WpMap', 'Wp'), ('effMap', 'eff')):
            itp = _interp(mp, field, alpha_idx=alpha_idx)
            core = ((90.0, 112.0), (w['PR'][0] + 0.8, w['PR'][1] - 0.8))
            terms, res, res_off, cr, (x0t, y0t) = _fit_surface(
                f"{name}.{key}", itp, w['Np'], w['PR'],
                exps_np, exps_pr, core=core)
            d[f'terms_{key}'] = terms
            d['x0'], d['y0'] = x0t, y0t
            report.append(f"{name}.{key:3s} signomial({len(terms):2d}) "
                          f"core {res:.2e} full {res_off:.2e} "
                          f"cancel {cr:.1f}")
            xs_ = np.linspace(w['Np'][0], w['Np'][1], 33)
            ys_ = np.linspace(w['PR'][0], w['PR'][1], 33)
            Xs, Ys, Vs = _sample(itp, xs_, ys_)
            (cl, ch), (rl, rh) = core
            inc = (Xs >= cl) & (Xs <= ch) & (Ys >= rl) & (Ys <= rh)
            Xw = np.concatenate([Xs] + [Xs[inc]] * 3)
            Yw = np.concatenate([Ys] + [Ys[inc]] * 3)
            Vw = np.concatenate([Vs] + [Vs[inc]] * 3)
            XY = np.column_stack([Xw / x0t, Yw / y0t])
            fitd, inv, err = fit_sma(XY, Vw, K=6)
            XYc = np.column_stack([Xs[inc] / x0t, Ys[inc] / y0t])
            pred = evaluate_fit(fitd, XYc)
            if inv:
                pred = 1.0 / pred
            resc = float(np.max(np.abs(np.log(pred / Vs[inc]))))
            d[f'sma_{key}'] = {k_: fitd[k_] for k_ in
                               ('ftype', 'a1', 'K', 'd', 'c', 'e')}
            d[f'sma_{key}']['inv'] = bool(inv)
            report.append(f"{name}.{key:3s} SMA(K={fitd['K']},a={fitd['a1']:.0f}"
                          f"{',inv' if inv else ''}) core {resc:.2e} "
                          f"full {err:.2e}")
        d['NpMap_d'] = float(mp.defaults['NpMap'])
        d['PRmap_d'] = float(mp.defaults['PRmap'])
        at = {}
        for field, key in (('WpMap', 'Wp'), ('effMap', 'eff')):
            itp = _interp(mp, field, alpha_idx=alpha_idx)
            v = itp.interpolate(np.array([[d['NpMap_d'], d['PRmap_d']]]), compute_derivative=False)
            at[key] = float(np.asarray(v).ravel()[0])
        d['map_at_defaults'] = at
        d['window'] = {'Np': list(w['Np']), 'PR': list(w['PR'])}
        out[name] = d

    if path is None:
        path = (pathlib.Path(__file__).resolve().parents[1] / 'sp_maps.py')
    buf = io.StringIO()
    buf.write('"""Scaled-map signomial fits, GENERATED by '
              'components/turbofan/truth/fits_maps.py.\n\n'
              'Do not edit by hand; rerun the generator. Sampled from '
              'pycycle\'s own slinear\ninterpolant of the NPSS-heritage '
              'map tables. See fits_maps.py.\n"""\n\n')
    for key in out:
        buf.write(f"{key.upper()} = ")
        buf.write(pprint.pformat(out[key], width=78))
        buf.write("\n\n")
    pathlib.Path(path).write_text(buf.getvalue())
    report.append(f"wrote {path}")
    return out, report


if __name__ == "__main__":
    _, report = generate()
    print("\n".join(report))
