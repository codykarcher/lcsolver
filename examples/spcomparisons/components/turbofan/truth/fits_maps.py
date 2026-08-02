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

#: (map, kind, window) per cycle component. Windows chosen to cover the
#: anchors' OD excursions: SLS static pushes corrected speed HIGH on the
#: fan (cold day would push higher; ISA SLS sits ~1.05) and part-power
#: cruise pushes it low.
# Windows sized from the anchors' MEASURED excursions (truth JSONs: fan Nc
# 0.945-1.024, lpc 0.960-1.029, hpc 0.973-0.989 across both engines and all
# four OD points), widened toward low corrected speed for the mission's
# part-power descent legs. Tight windows are what make sub-1% signomial
# fits of a kinked slinear table possible.
COMPRESSORS = {
    'fan': (pyc.FanMap, dict(Nc=(0.80, 1.10), R=(1.40, 2.70))),
    'lpc': (pyc.LPCMap, dict(Nc=(0.80, 1.10), R=(1.40, 2.70))),
    'hpc': (pyc.HPCMap, dict(Nc=(0.88, 1.06), R=(1.40, 2.70))),
}
TURBINES = {
    'hpt': (pyc.HPTMap, dict(Np=(85.0, 115.0), PR=(3.0, 5.0))),
    'lpt': (pyc.LPTMap, dict(Np=(85.0, 115.0), PR=(3.5, 7.0))),
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


def _fit_surface(label, itp, w1, w2, exps1, exps2, n=33, x0=1.0, y0=1.0,
                 core=None):
    """``core`` = ((x_lo, x_hi), (y_lo, y_hi)): the region the anchors
    actually occupy. Points inside get 10x fit weight -- a smooth surface
    cannot track a kinked slinear table below ~1% everywhere, so spend the
    residual budget where the validation points live. Reports (core
    residual, full-window off-grid residual)."""
    xs = np.linspace(w1[0], w1[1], n)
    ys = np.linspace(w2[0], w2[1], n)
    X, Y, V = _sample(itp, xs, ys)
    wgt = np.ones_like(V)
    if core is not None:
        (xl, xh), (yl, yh) = core
        inside = (X >= xl) & (X <= xh) & (Y >= yl) & (Y <= yh)
        wgt[inside] = 10.0
    A = np.vstack([(X / x0)**a * (Y / y0)**b
                   for a in exps1 for b in exps2]).T
    Wm = wgt / V
    c, *_ = np.linalg.lstsq(A * Wm[:, None], wgt, rcond=None)
    tlist = [(float(ck), float(a), float(b))
             for ck, (a, b) in zip(c, [(a, b) for a in exps1 for b in exps2])
             if ck != 0.0]
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
    return tlist, res_core, res_off


def generate(path=None):
    import io, pathlib, pprint
    out, report = {}, []

    exps_n = [0.0, 0.75, 1.5, 2.25, 3.0, 3.75]
    exps_r = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

    for name, (mp, w) in COMPRESSORS.items():
        d = {}
        for field, key in (('WcMap', 'Wc'), ('PRmap', 'PR'),
                           ('effMap', 'eff')):
            itp = _interp(mp, field, alpha_idx=0)
            # per-component core boxes from the measured anchor excursions
            core = {'fan': ((0.93, 1.04), (1.7, 2.6)),
                    'lpc': ((0.94, 1.05), (1.7, 2.6)),
                    'hpc': ((0.96, 1.01), (1.7, 2.6))}[name]
            terms, res, res_off = _fit_surface(
                f"{name}.{key}", itp, w['Nc'], w['R'], exps_n, exps_r,
                core=core)
            d[f'terms_{key}'] = terms
            report.append(f"{name}.{key:3s} signomial({len(terms):2d}) "
                          f"core {res:.2e} full {res_off:.2e}")
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

    exps_np = [0.0, 0.5, 1.0, 1.5, 2.0]
    exps_pr = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for name, (mp, w) in TURBINES.items():
        d = {}
        alpha_idx = int(mp.defaults.get('alphaMap', 0))
        # alphaMap slice: defaults say 1.0 for these maps; index the slice
        # that matches (the alpha grid is [0, 1] for both turbine maps).
        for field, key in (('WpMap', 'Wp'), ('effMap', 'eff')):
            itp = _interp(mp, field, alpha_idx=alpha_idx)
            core = ((95.0, 110.0), (w['PR'][0] + 0.3, w['PR'][1] - 0.3))
            terms, res, res_off = _fit_surface(
                f"{name}.{key}", itp, w['Np'], w['PR'],
                exps_np, exps_pr, x0=100.0, y0=4.0, core=core)
            d[f'terms_{key}'] = terms
            report.append(f"{name}.{key:3s} signomial({len(terms):2d}) "
                          f"core {res:.2e} full {res_off:.2e}")
        d['NpMap_d'] = float(mp.defaults['NpMap'])
        d['PRmap_d'] = float(mp.defaults['PRmap'])
        at = {}
        for field, key in (('WpMap', 'Wp'), ('effMap', 'eff')):
            itp = _interp(mp, field, alpha_idx=alpha_idx)
            v = itp.interpolate(np.array([[d['NpMap_d'], d['PRmap_d']]]), compute_derivative=False)
            at[key] = float(np.asarray(v).ravel()[0])
        d['map_at_defaults'] = at
        d['window'] = {'Np': list(w['Np']), 'PR': list(w['PR'])}
        d['x0'], d['y0'] = 100.0, 4.0
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
