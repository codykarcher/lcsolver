"""Run an anchor engine: design + 4 off-design points, report, dump JSON.

Usage (from examples/spcomparisons)::

    python -m components.turbofan.truth.run_anchors [cfm56_class genx_class]

Continuation
------------
The off-design points are built parked AT the design condition and walked to
their targets over ``STEPS`` warm-started re-runs, interpolating (MN, alt,
T4 | PC) linearly. This is the same trick the pycycle example's envelope
sweep uses (MN 0.8 -> 0.001 in descending steps): a cold Newton start at
static sea level with takeoff T4 diverges, a warm walk down does not. All
points step together -- one ``run_model`` per stage solves every point.

Output
------
``data/<engine>.json``: per point, the flight condition, performance
figures, and the full station table (W, Tt, Pt, ht, s, MN, area per flow
station) plus component states (PR, eff, corrected flow, speeds) and burner
product mole fractions. This is the station-by-station truth the SP cycle
rows are validated against -- milestone 2 compares each station's
(Tt, Pt, ht) to these numbers.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np

from .engines import ENGINES
from .hbtf import build_problem

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

STEPS = 6

FS_NAMES = ['fc.Fl_O', 'inlet.Fl_O', 'fan.Fl_O', 'splitter.Fl_O1',
            'splitter.Fl_O2', 'duct4.Fl_O', 'lpc.Fl_O', 'duct6.Fl_O',
            'hpc.Fl_O', 'bld3.Fl_O', 'burner.Fl_O', 'hpt.Fl_O',
            'duct11.Fl_O', 'lpt.Fl_O', 'duct13.Fl_O', 'core_nozz.Fl_O',
            'byp_bld.Fl_O', 'duct15.Fl_O', 'byp_nozz.Fl_O']

#: (suffix, units) pulled for every flow station. Units chosen to match the
#: pycycle viewer tables so the JSON is directly comparable to them.
FS_VARS = [(':tot:T', 'degR'), (':tot:P', 'psi'), (':tot:h', 'Btu/lbm'),
           (':tot:S', 'Btu/(lbm*degR)'), (':stat:W', 'lbm/s'),
           (':stat:MN', None), (':stat:T', 'degR'), (':stat:P', 'psi'),
           (':stat:area', 'inch**2'), (':stat:V', 'ft/s')]


def _get(prob, name, units=None):
    try:
        v = prob.get_val(name, units=units)
        return float(np.asarray(v).ravel()[0])
    except Exception:
        return None


def _collect_point(prob, pt, design):
    out = {"condition": {
        "MN": _get(prob, f"{pt}.fc.Fl_O:stat:MN"),
        "alt_ft": _get(prob, f"{pt}.fc.alt", "ft"),
    }}
    perf = {
        "Fn_lbf": _get(prob, f"{pt}.perf.Fn", "lbf"),
        "Fg_lbf": _get(prob, f"{pt}.perf.Fg", "lbf"),
        "Fram_lbf": _get(prob, f"{pt}.inlet.F_ram", "lbf"),
        "TSFC": _get(prob, f"{pt}.perf.TSFC", "lbm/(lbf*h)"),
        "OPR": _get(prob, f"{pt}.perf.OPR"),
        "FAR": _get(prob, f"{pt}.balance.FAR"),
        "Wfuel_lbm_s": _get(prob, f"{pt}.burner.Wfuel", "lbm/s"),
        "W_lbm_s": _get(prob, f"{pt}.inlet.Fl_O:stat:W", "lbm/s"),
        "BPR": (_get(prob, f"{pt}.splitter.BPR") if design
                else _get(prob, f"{pt}.balance.BPR")),
        "T4_R": _get(prob, f"{pt}.burner.Fl_O:tot:T", "degR"),
        "LP_Nmech_rpm": _get(prob, f"{pt}.LP_Nmech", "rpm"),
        "HP_Nmech_rpm": _get(prob, f"{pt}.HP_Nmech", "rpm"),
    }
    out["performance"] = perf

    stations = {}
    for fs in FS_NAMES:
        row = {}
        for suffix, units in FS_VARS:
            v = _get(prob, f"{pt}.{fs}{suffix}", units)
            if v is not None:
                key = suffix.replace(':tot:', 'Tt_' if suffix == ':tot:T'
                                     else 'tot_').replace(':stat:', 'stat_')
                row[key.lstrip('_')] = v
        # friendlier names
        ren = {'Tt_': 'Tt_R', 'tot_P': 'Pt_psi', 'tot_h': 'ht_Btu_lbm',
               'tot_S': 's_Btu_lbmR', 'stat_W': 'W_lbm_s', 'stat_MN': 'MN',
               'stat_T': 'Ts_R', 'stat_P': 'Ps_psi',
               'stat_area': 'area_in2', 'stat_V': 'V_fps'}
        stations[fs] = {ren.get(k, k): v for k, v in row.items()}
    out["stations"] = stations

    comps = {}
    for c in ('fan', 'lpc', 'hpc'):
        comps[c] = {"PR": _get(prob, f"{pt}.{c}.PR"),
                    "eff": _get(prob, f"{pt}.{c}.eff"),
                    "eff_poly": _get(prob, f"{pt}.{c}.eff_poly"),
                    "Wc_lbm_s": _get(prob, f"{pt}.{c}.Wc", "lbm/s"),
                    "Nc": _get(prob, f"{pt}.{c}.map.NcMap")}
    for t in ('hpt', 'lpt'):
        comps[t] = {"PR": _get(prob, f"{pt}.{t}.PR"),
                    "eff": _get(prob, f"{pt}.{t}.eff"),
                    "eff_poly": _get(prob, f"{pt}.{t}.eff_poly"),
                    "Wp_lbm_s": _get(prob, f"{pt}.{t}.Wp", "lbm/s")}
    out["components"] = comps

    nozz = {}
    for n in ('core_nozz', 'byp_nozz'):
        nozz[n] = {"Fg_lbf": _get(prob, f"{pt}.{n}.Fg", "lbf"),
                   "PR": _get(prob, f"{pt}.{n}.PR"),
                   "throat_area_in2": _get(prob, f"{pt}.{n}.Throat:stat:area",
                                           "inch**2"),
                   "throat_MN": _get(prob, f"{pt}.{n}.Throat:stat:MN")}
    out["nozzles"] = nozz

    shafts = {}
    for s in ('hp_shaft', 'lp_shaft'):
        shafts[s] = {"pwr_in_hp": _get(prob, f"{pt}.{s}.pwr_in_real", "hp"),
                     "pwr_out_hp": _get(prob, f"{pt}.{s}.pwr_out_real", "hp")}
    out["shafts"] = shafts

    # Burner-exit product amounts (kmol per kg mixture), with species names
    # reconstructed from the same janaf air+fuel product set the Combustor
    # uses. This is the mass-action truth data for the SP burner rows.
    try:
        from pycycle.thermo.cea import species_data
        from pycycle.constants import CEA_AIR_FUEL_COMPOSITION
        n = prob.get_val(
            f"{pt}.burner.vitiated_flow.base_thermo.chem_eq.n")
        thermo = species_data.Properties(
            species_data.janaf, init_elements=CEA_AIR_FUEL_COMPOSITION)
        if len(thermo.products) == len(n):
            out["burner_products_kmol_per_kg"] = {
                sp: float(v) for sp, v in zip(thermo.products, n)}
    except Exception as e:
        out["burner_products_error"] = f"{type(e).__name__}: {e}"

    return out


def run_engine(spec, steps=STEPS):
    t0 = time.time()
    print(f"\n=== {spec.name} ===", flush=True)
    prob = build_problem(spec)

    # Stage 0: everything at the design condition.
    prob.run_model()
    print(f"  stage 0 (all points at design condition) "
          f"{time.time()-t0:.0f}s", flush=True)

    for k in range(1, steps + 1):
        f = k / steps
        for od in spec.od_points:
            prob.set_val(f"{od.name}.fc.MN",
                         spec.MN_des + f * (od.MN - spec.MN_des))
            prob.set_val(f"{od.name}.fc.alt",
                         spec.alt_des_ft + f * (od.alt_ft - spec.alt_des_ft),
                         units='ft')
            if od.mode == 'T4':
                prob.set_val(f"{od.name}.T4_MAX",
                             spec.T4_des_R + f * (od.T4_R - spec.T4_des_R),
                             units='degR')
            else:
                prob.set_val(f"{od.name}.PC", 1.0 + f * (od.PC - 1.0))
        prob.run_model()
        print(f"  stage {k}/{steps} {time.time()-t0:.0f}s", flush=True)

    # ---- report ----
    result = {"spec": {
        "name": spec.name, "MN_des": spec.MN_des,
        "alt_des_ft": spec.alt_des_ft, "Fn_des_lbf": spec.Fn_des_lbf,
        "T4_des_R": spec.T4_des_R, "BPR": spec.BPR, "FPR": spec.FPR,
        "LPC_PR": spec.LPC_PR, "HPC_PR": spec.HPC_PR,
        "eff_fan": spec.eff_fan, "eff_lpc": spec.eff_lpc,
        "eff_hpc": spec.eff_hpc, "eff_hpt": spec.eff_hpt,
        "eff_lpt": spec.eff_lpt, "notes": spec.notes},
        "points": {}}
    result["points"]["DESIGN"] = _collect_point(prob, "DESIGN", design=True)
    for od in spec.od_points:
        result["points"][od.name] = _collect_point(prob, od.name,
                                                   design=False)

    print(f"\n  {'point':8s} {'MN':>6s} {'alt':>7s} {'Fn lbf':>9s} "
          f"{'TSFC':>7s} {'OPR':>6s} {'BPR':>6s} {'T4 R':>7s} "
          f"{'W lbm/s':>8s}")
    for name, p in result["points"].items():
        c, pf = p["condition"], p["performance"]
        print(f"  {name:8s} {c['MN']:6.3f} {c['alt_ft']:7.0f} "
              f"{pf['Fn_lbf']:9.1f} {pf['TSFC']:7.4f} {pf['OPR']:6.2f} "
              f"{pf['BPR']:6.3f} {pf['T4_R']:7.1f} {pf['W_lbm_s']:8.1f}")

    print(f"\n  reference anchors ({spec.name}):")
    for key, ref in spec.refs.items():
        pt, _, var = key.partition('.')
        got = None
        p = result["points"].get(pt)
        if p:
            got = {"perf.Fn": p["performance"]["Fn_lbf"],
                   "perf.TSFC": p["performance"]["TSFC"],
                   "perf.OPR": p["performance"]["OPR"],
                   "splitter.BPR": p["performance"]["BPR"]}.get(var)
        mark = "~" if ref["approximate"] else " "
        if got is None:
            print(f"   {key:18s} model      --  ref {ref['value']:9.3f}{mark}")
        else:
            print(f"   {key:18s} model {got:9.3f}  ref {ref['value']:9.3f}"
                  f"{mark}  ratio {got/ref['value']:6.3f}   [{ref['source']}]")

    DATA.mkdir(exist_ok=True)
    out_path = DATA / f"{spec.name}.json"
    out_path.write_text(json.dumps(result, indent=1))
    print(f"\n  wrote {out_path}  ({time.time()-t0:.0f}s total)", flush=True)
    return result


if __name__ == "__main__":
    names = sys.argv[1:] or list(ENGINES)
    for name in names:
        run_engine(ENGINES[name])
