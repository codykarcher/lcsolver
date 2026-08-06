"""Validate the SP cycle against the pycycle truth anchors, station by
station.

Pins the SP on-design cycle (components/turbofan/sp_cycle.py) with exactly
the inputs the truth harness gave pycycle at the DESIGN point, solves it with
the house solver settings, and prints the ratio of every station's
(Tt, Pt, ht) and the headline performance numbers to the values recorded in
``truth/data/<engine>.json``. This is the milestone-2 acceptance: each
station within its stated tolerance, with any excursion explained rather
than filed away.

Run (from examples/spcomparisons):
    python -m components.turbofan.truth.validate_sp [cfm56_class genx_class]
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import warnings

HERE = pathlib.Path(__file__).resolve().parent

R2K = 1.0 / 1.8
PSI2PA = 6894.757293168
BTULBM2JKG = 2326.0

#: (truth flow station, SP var basenames for Tt/Pt/ht) per station.
STATIONS = {
    "21": ("fan.Fl_O", "fan_Tt", "fan_Pt", "fan_ht"),
    "25": ("lpc.Fl_O", "lpc_Tt", "lpc_Pt", "lpc_ht"),
    "3": ("hpc.Fl_O", "hpc_Tt", "hpc_Pt", "hpc_ht"),
    "45": ("hpt.Fl_O", "hpt_Ttout", "hpt_Ptout", "hpt_htout"),
    "49": ("lpt.Fl_O", "lpt_Ttout", "lpt_Ptout", "lpt_htout"),
}


def pins_from_truth(name):
    from components.turbofan.truth.engines import ENGINES
    from components.turbofan.sp_cycle import CyclePins, Bleed

    spec = ENGINES[name]
    d = json.loads((HERE / "data" / f"{name}.json").read_text())
    des = d["points"]["DESIGN"]
    fc = des["stations"]["fc.Fl_O"]
    noz = des["nozzles"]

    LB2N = 4.4482216152605
    pins = CyclePins(
        name=name,
        T0_K=fc["Ts_R"] * R2K,
        P0_Pa=fc["Ps_psi"] * PSI2PA,
        MN=des["condition"]["MN"],
        V0_m_s=fc["V_fps"] * 0.3048,
        Fn_N=spec.Fn_des_lbf * LB2N,
        T4_K=spec.T4_des_R * R2K,
        FPR=spec.FPR, LPC_PR=spec.LPC_PR, HPC_PR=spec.HPC_PR,
        BPR=spec.BPR,
        eff_fan=spec.eff_fan, eff_lpc=spec.eff_lpc, eff_hpc=spec.eff_hpc,
        eff_hpt=spec.eff_hpt, eff_lpt=spec.eff_lpt,
        HPX_W=dict(spec.params).get(("hp_shaft.HPX", "hp"), 250.0) * 745.699872,
        cust=Bleed(dict(spec.params).get("hpc.cust:frac_W", 0.0445),
                   0.5, 0.5),
        choked_core=abs(noz["core_nozz"]["throat_MN"] - 1.0) < 1e-3,
        choked_byp=abs(noz["byp_nozz"]["throat_MN"] - 1.0) < 1e-3,
    )
    return pins, des


def solve(f):
    from lcsolver_compat import structure_detector, unit_corrector
    from lcsolver.solvers.ipopt.slcp_bridge import solve_sia
    from lcsolver.solvers.ipopt.sia import SIAOptions

    cm = unit_corrector(f)
    st = structure_detector(cm)
    opts = SIAOptions(max_iterations=200)
    opts.stationarity_tolerance = 1e-5
    opts.condense_numerator = True
    opts.ipopt_options = dict(opts.ipopt_options, tol=1e-9,
                              constr_viol_tol=1e-9)
    res = solve_sia(st, options=opts, presolve=False)
    vals = {}
    if res.converged:
        import pyomo.environ as pyo
        for v, val in zip(st["variables"], res.x):
            v.set_value(float(val))
        vals = {v.name: pyo.value(v)
                for v in cm.component_data_objects(pyo.Var)}
    return res, vals


def compare(name):
    from components.turbofan import sp_cycle

    pins, des = pins_from_truth(name)
    print(f"\n=== {name}: SP cycle vs pycycle DESIGN ===")
    print(f"  choked: core={pins.choked_core} byp={pins.choked_byp}")
    t0 = time.time()
    f = sp_cycle.build(pins)
    res, vals = solve(f)
    print(f"  converged={res.converged} iterations={res.iterations} "
          f"({time.time()-t0:.1f}s)  "
          f"stationarity={getattr(res, 'stationarity', float('nan')):.2e}")
    if not res.converged:
        print("  status:", str(res.status)[:200])
        rep = getattr(res, "report", None)
        if rep:
            print(rep[:1500])
        return False

    H = sp_cycle.H_SHIFT
    rows = []

    def row(label, mine, truth):
        rows.append((label, mine, truth,
                     mine / truth if truth else float("nan")))

    st = des["stations"]
    for tag, (fs, vT, vP, vh) in STATIONS.items():
        s = st[fs]
        row(f"Tt{tag} K", vals[vT], s["Tt_T"] * R2K)
        row(f"Pt{tag} Pa", vals[vP], s["Pt_psi"] * PSI2PA)
        row(f"ht{tag} J/kg", vals[vh] - H + 0.0,
            s["ht_Btu_lbm"] * BTULBM2JKG)
    s = st["burner.Fl_O"]
    row("Tt4 K", vals["Tt4"], s["Tt_T"] * R2K)
    row("Pt4 Pa", vals["Pt4"], s["Pt_psi"] * PSI2PA)
    row("ht4 J/kg", vals["h4_mix"] - H, s["ht_Btu_lbm"] * BTULBM2JKG)

    perf = des["performance"]
    row("W kg/s", vals["W"], perf["W_lbm_s"] * 0.45359237)
    row("FAR", vals["far"], perf["FAR"])
    row("Wf kg/s", vals["Wf"], perf["Wfuel_lbm_s"] * 0.45359237)
    row("TSFC", vals["TSFC"], perf["TSFC"] * 2.8325e-5)
    row("Fg_core N", vals["Fg_core"],
        des["nozzles"]["core_nozz"]["Fg_lbf"] * 4.4482216)
    row("Fg_byp N", vals["Fg_byp"],
        des["nozzles"]["byp_nozz"]["Fg_lbf"] * 4.4482216)
    hpt_PR = vals["hpt_PR"]
    lpt_PR = vals["lpt_PR"]
    row("HPT PR", hpt_PR, des["components"]["hpt"]["PR"])
    row("LPT PR", lpt_PR, des["components"]["lpt"]["PR"])

    print(f"  {'quantity':14s} {'SP':>14s} {'pycycle':>14s} {'ratio':>9s}")
    worst = ("", 0.0)
    for label, mine, truth, r in rows:
        flag = ""
        if abs(r - 1) > 0.005:
            flag = "  <-- >0.5%"
        if abs(r - 1) > abs(worst[1]):
            worst = (label, r - 1)
        print(f"  {label:14s} {mine:14.4f} {truth:14.4f} {r:9.5f}{flag}")

    # burner species vs pycycle's chem_eq
    prods = des.get("burner_products_kmol_per_kg")
    if prods:
        print(f"  {'species':10s} {'SP':>12s} {'pycycle':>12s} {'ratio':>8s}")
        for sp in ("N2", "O2", "CO2", "H2O", "CO", "OH", "NO", "O"):
            pv = prods.get(sp)
            mv = vals.get(f"n_{sp}")
            if pv and mv and pv > 1e-12:
                print(f"  n_{sp:8s} {mv:12.5e} {pv:12.5e} {mv/pv:8.4f}")

    print(f"  worst station ratio: {worst[0]} {1+worst[1]:.5f}")
    return True


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    names = sys.argv[1:] or ["cfm56_class", "genx_class"]
    ok = True
    for n in names:
        ok = compare(n) and ok
    sys.exit(0 if ok else 1)
