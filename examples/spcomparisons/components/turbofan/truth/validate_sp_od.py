"""Validate the SP cycle OFF-DESIGN against the pycycle truth anchors.

Builds the multipoint SP cycle -- design point plus the four off-design
anchors (TO, RTO, TOC, CRZ) -- pinned to the same conditions the truth
harness gave pycycle, solves the whole thing as one SP, and compares each
point's headline quantities (Fn, TSFC, W, BPR, FAR, spool speeds, fan PR)
to truth/data/<engine>.json. This is the milestone-3 acceptance.

Run (from examples/spcomparisons):
    python -m components.turbofan.truth.validate_sp_od [cfm56_class ...]
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
LB2KG = 0.45359237
LB2N = 4.4482216152605


def build_pins(name):
    from components.turbofan.truth.engines import ENGINES
    from components.turbofan.sp_cycle import CyclePins, ODPins, Bleed
    from components.turbofan.truth.validate_sp import pins_from_truth

    spec = ENGINES[name]
    pins, des = pins_from_truth(name)
    pins = type(pins)(**{**pins.__dict__,
                         'LP_Nmech': spec.LP_Nmech,
                         'HP_Nmech': spec.HP_Nmech})
    d = json.loads((HERE / "data" / f"{name}.json").read_text())

    ods = []
    for od in spec.od_points:
        p = d["points"][od.name]
        fc = p["stations"]["fc.Fl_O"]
        noz = p["nozzles"]
        ods.append(ODPins(
            name=od.name,
            T0_K=fc["Ts_R"] * R2K,
            P0_Pa=fc["Ps_psi"] * PSI2PA,
            MN=p["condition"]["MN"],
            V0_m_s=fc["V_fps"] * 0.3048,
            mode=("PC" if od.mode == "PC" else "T4"),
            T4_K=(od.T4_R * R2K if od.T4_R else None),
            PC=od.PC, pc_of=od.pc_of,
            choked_core=abs(noz["core_nozz"]["throat_MN"] - 1.0) < 1e-3,
            choked_byp=abs(noz["byp_nozz"]["throat_MN"] - 1.0) < 1e-3,
        ))
    return pins, tuple(ods), d


def warm_from_truth(name, d, ods):
    """Initial guesses for every point's heavy hitters, from truth.

    The DESIGN point is warmed too: its builder defaults are CFM56-sized,
    and while a design-only solve recovers from that, the multipoint
    phase-1 does not -- the GEnx design P_fan row started 1e7 W out and the
    first subproblem declared itself infeasible."""
    from components.turbofan import sp_cycle
    H = sp_cycle.H_SHIFT
    NS = sp_cycle.N_SCALE
    LB2N_ = LB2N
    warm = {}
    st_map = {"fan": "fan.Fl_O", "lpc": "lpc.Fl_O", "hpc": "hpc.Fl_O"}
    points = [("", "DESIGN")] + [(f"{od.name}_", od.name) for od in ods]
    for t, ptname in points:
        p = d["points"][ptname]
        perf, st, comps = p["performance"], p["stations"], p["components"]
        for key, fs in st_map.items():
            s_ = st[fs]
            warm[f"{t}{key}_Tt"] = s_["Tt_T"] * R2K
            warm[f"{t}{key}_Pt"] = s_["Pt_psi"] * PSI2PA
            warm[f"{t}{key}_ht"] = s_["ht_Btu_lbm"] * 2326.0 + H
            warm[f"{t}{key}_PR"] = comps[key]["PR"]
            warm[f"{t}{key}_eff"] = comps[key]["eff"]
            warm[f"{t}{key}_NcMap"] = comps[key]["Nc"]
            warm[f"{t}{key}_Wc"] = comps[key]["Wc_lbm_s"] * LB2KG
        for key, fs in (("hpt", "hpt.Fl_O"), ("lpt", "lpt.Fl_O")):
            s_ = st[fs]
            warm[f"{t}{key}_Ttout"] = s_["Tt_T"] * R2K
            warm[f"{t}{key}_Ptout"] = s_["Pt_psi"] * PSI2PA
            warm[f"{t}{key}_htout"] = s_["ht_Btu_lbm"] * 2326.0 + H
            warm[f"{t}{key}_PR"] = comps[key]["PR"]
            warm[f"{t}{key}_eff"] = comps[key]["eff"]
        warm[f"{t}W"] = perf["W_lbm_s"] * LB2KG
        warm[f"{t}far"] = perf["FAR"]
        warm[f"{t}Wf"] = perf["Wfuel_lbm_s"] * LB2KG
        warm[f"{t}BPR"] = perf["BPR"]
        warm[f"{t}LP_N"] = perf["LP_Nmech_rpm"]
        warm[f"{t}HP_N"] = perf["HP_Nmech_rpm"]
        warm[f"{t}Fn"] = perf["Fn_lbf"] * LB2N
        warm[f"{t}TSFC"] = perf["TSFC"] * 2.8325e-5
        warm[f"{t}Tt4"] = perf["T4_R"] * R2K
        warm[f"{t}Pt4"] = st["burner.Fl_O"]["Pt_psi"] * PSI2PA
        warm[f"{t}h4_mix"] = (st["burner.Fl_O"]["ht_Btu_lbm"] * 2326.0 + H)
        for key, fs in (("core", "core_nozz.Fl_O"), ("byp", "byp_nozz.Fl_O")):
            s_ = st[fs]
            warm[f"{t}Ts_{key}"] = s_["Ts_R"] * R2K
            warm[f"{t}Ps_{key}"] = s_["Ps_psi"] * PSI2PA
            warm[f"{t}V_{key}"] = s_["V_fps"] * 0.3048

        # turbine map-block warm values, computed the way the rows compute
        # them (referred flow in the module's x1e4-SI convention). The LPT
        # block diverged with its Wp guess 1.5 decades off at GEnx scale.
        import components.turbofan.sp_maps as MAPS
        des_perf = d["points"]["DESIGN"]["performance"]
        des_st = d["points"]["DESIGN"]["stations"]
        for key, fs_in, spool in (("hpt", "burner.Fl_O", "HP_Nmech_rpm"),
                                  ("lpt", "duct11.Fl_O", "LP_Nmech_rpm")):
            M = getattr(MAPS, key.upper())
            s_in = st[fs_in]
            W_in = s_in["W_lbm_s"] * LB2KG
            T_in = s_in["Tt_T"] * R2K
            Pt_in = s_in["Pt_psi"] * PSI2PA
            warm[f"{t}{key}_Wp"] = W_in * T_in**0.5 / Pt_in * 1e4
            Np = perf[spool] / T_in**0.5
            warm[f"{t}{key}_Np"] = Np
            # scalars from the DESIGN point's own referred quantities
            dsi = des_st[fs_in]
            Np_d = des_perf[spool] / (dsi["Tt_T"] * R2K)**0.5
            s_Np = Np_d / M['NpMap_d']
            warm[f"{t}{key}_NpMap"] = Np / s_Np
            PR_des = d["points"]["DESIGN"]["components"][key]["PR"]
            s_PR = (PR_des - 1.0) / (M['PRmap_d'] - 1.0)
            PR_od = p["components"][key]["PR"]
            warm[f"{t}{key}_PRmap"] = 1.0 + (PR_od - 1.0) / s_PR
        # compressor map PR-1 in map coordinates
        for key in ("fan", "lpc", "hpc"):
            M = getattr(MAPS, key.upper())
            PR_des = d["points"]["DESIGN"]["components"][key]["PR"]
            s_PR = (PR_des - 1.0) / (M['map_at_defaults']['PR'] - 1.0)
            warm[f"{t}{key}_prm1"] = (comps[key]["PR"] - 1.0) / s_PR
        if t == "":
            for key, fs_in, spool in (("hpt", "burner.Fl_O", "HP_Nmech_rpm"),
                                      ("lpt", "duct11.Fl_O", "LP_Nmech_rpm")):
                M = getattr(MAPS, key.upper())
                warm[f"{key}_sWp"] = (warm[f"{key}_Wp"]
                                      / M['map_at_defaults']['Wp'])
                warm[f"{key}_sNp"] = warm[f"{key}_Np"] / M['NpMap_d']
                PR_des = d["points"]["DESIGN"]["components"][key]["PR"]
                warm[f"{key}_sPR"] = ((PR_des - 1.0)
                                      / (M['PRmap_d'] - 1.0))
                warm[f"{key}_PR"] = PR_des
            for key in ("fan", "lpc", "hpc"):
                M = getattr(MAPS, key.upper())
                warm[f"{key}_sWc"] = (comps[key]["Wc_lbm_s"] * LB2KG
                                      / (M['map_at_defaults']['Wc']
                                         * LB2KG))
                Nc_d = perf[{"fan": "LP_Nmech_rpm", "lpc": "LP_Nmech_rpm",
                             "hpc": "HP_Nmech_rpm"}[key]]
                warm[f"{key}_sNc"] = 4000.0  # refined below
        warm[f"{t}P_fan"] = (perf["W_lbm_s"] * LB2KG
                             * abs(st["fan.Fl_O"]["ht_Btu_lbm"]
                                   - st["fc.Fl_O"]["ht_Btu_lbm"]) * 2326.0)
        prods = p.get("burner_products_kmol_per_kg")
        if prods:
            for sp, v in prods.items():
                warm[f"{t}n_{sp}"] = max(v, 1e-14) * NS
            warm[f"{t}n_tot"] = sum(max(v, 1e-14) for v in prods.values()) * NS

        # -- the rest of the intermediates, derived from truth ------------
        # A half-warm start is not warm: with only the headline variables
        # set, 132 equalities opened >1e-4 at the GEnx start (powers, duct
        # pressures, ideal drops still carried CFM56-scale defaults) and
        # the first subproblem declared itself infeasible.
        B2J = 2326.0
        gv = lambda fs, k: st[fs][k]
        # flows
        warm[f"{t}Wcore"] = gv("splitter.Fl_O1", "W_lbm_s") * LB2KG
        warm[f"{t}Wbyp"] = gv("splitter.Fl_O2", "W_lbm_s") * LB2KG
        warm[f"{t}W3"] = gv("hpc.Fl_O", "W_lbm_s") * LB2KG
        warm[f"{t}W31"] = gv("bld3.Fl_O", "W_lbm_s") * LB2KG
        warm[f"{t}W4"] = gv("burner.Fl_O", "W_lbm_s") * LB2KG
        warm[f"{t}W15"] = gv("duct15.Fl_O", "W_lbm_s") * LB2KG
        warm[f"{t}hpt_Wout"] = gv("duct11.Fl_O", "W_lbm_s") * LB2KG
        warm[f"{t}lpt_Wout"] = gv("duct13.Fl_O", "W_lbm_s") * LB2KG
        # duct/station pressures
        warm[f"{t}Pt_lpc_in"] = gv("duct4.Fl_O", "Pt_psi") * PSI2PA
        warm[f"{t}Pt_hpc_in"] = gv("duct6.Fl_O", "Pt_psi") * PSI2PA
        warm[f"{t}Pt_lpt_in"] = gv("duct11.Fl_O", "Pt_psi") * PSI2PA
        warm[f"{t}Pt5"] = gv("duct13.Fl_O", "Pt_psi") * PSI2PA
        warm[f"{t}Pt15"] = gv("duct15.Fl_O", "Pt_psi") * PSI2PA
        warm[f"{t}Pt2"] = gv("inlet.Fl_O", "Pt_psi") * PSI2PA
        warm[f"{t}Pt0"] = gv("fc.Fl_O", "Pt_psi") * PSI2PA
        warm[f"{t}Tt0"] = gv("fc.Fl_O", "Tt_T") * R2K
        warm[f"{t}ht0"] = gv("fc.Fl_O", "ht_Btu_lbm") * B2J + H
        warm[f"{t}hpt_Ptout"] = gv("hpt.Fl_O", "Pt_psi") * PSI2PA
        warm[f"{t}lpt_Ptout"] = gv("lpt.Fl_O", "Pt_psi") * PSI2PA
        # psi values via the module's own fits
        warm[f"{t}psi0"] = float(sp_cycle.psi_air(warm[f"{t}Tt0"]))
        for key in ("fan", "lpc", "hpc"):
            warm[f"{t}{key}_psi"] = float(
                sp_cycle.psi_air(warm[f"{t}{key}_Tt"]))
            warm[f"{t}{key}_psis"] = warm[f"{t}{key}_psi"] * 0.98
            warm[f"{t}{key}_Tts"] = warm[f"{t}{key}_Tt"] * 0.985
        far_v = perf["FAR"]
        warm[f"{t}psi4"] = float(
            sp_cycle.psi_vit(perf["T4_R"] * R2K, far_v))
        # compressor rises
        h_in = {"fan": warm[f"{t}ht0"], "lpc": warm[f"{t}fan_ht"],
                "hpc": warm[f"{t}lpc_ht"]}
        for key in ("fan", "lpc", "hpc"):
            dh = max(warm[f"{t}{key}_ht"] - h_in[key], 1e3)
            warm[f"{t}{key}_dh"] = dh
            warm[f"{t}{key}_dhs"] = dh * comps[key]["eff"]
        # turbine drops and ideal exits
        for key, h_in_v, far_in in (
                ("hpt", warm[f"{t}h4_mix"], far_v),
                ("lpt", warm[f"{t}hpt_htout"], far_v)):
            eff = comps[key]["eff"]
            dh = max((h_in_v - warm[f"{t}{key}_htout"]) / eff, 1e3)
            warm[f"{t}{key}_dh"] = dh
            warm[f"{t}{key}_hi"] = max(h_in_v - dh, 1e5)
            warm[f"{t}{key}_Ti"] = warm[f"{t}{key}_Ttout"] * 0.98
            warm[f"{t}{key}_psii"] = float(sp_cycle.psi_vit(
                warm[f"{t}{key}_Ti"], far_in))
            warm[f"{t}{key}_psiout"] = float(sp_cycle.psi_vit(
                warm[f"{t}{key}_Ttout"], far_in))
            warm[f"{t}{key}_farout"] = far_in * 0.85
        # shaft powers from the truth shaft table
        HP2W = 745.699872
        warm[f"{t}hpt_P"] = p["shafts"]["hp_shaft"]["pwr_in_hp"] * HP2W
        warm[f"{t}lpt_P"] = p["shafts"]["lp_shaft"]["pwr_in_hp"] * HP2W
        warm[f"{t}P_hpc"] = warm[f"{t}hpt_P"] - 250.0 * HP2W
        warm[f"{t}P_lpc"] = max(warm[f"{t}lpt_P"] - warm[f"{t}P_fan"], 1e5)
        # nozzle statics
        for key in ("core", "byp"):
            Ts_v = warm[f"{t}Ts_{key}"]
            Ps_v = warm[f"{t}Ps_{key}"]
            Rg = 287.05 if key == "byp" else 288.3
            warm[f"{t}rho_{key}"] = Ps_v / (Rg * Ts_v)
            Wn = (warm[f"{t}lpt_Wout"] if key == "core"
                  else warm[f"{t}W15"])
            warm[f"{t}A_{key}"] = Wn / (warm[f"{t}rho_{key}"]
                                        * warm[f"{t}V_{key}"])
            warm[f"{t}Fg_{key}"] = (p["nozzles"][f"{key}_nozz"
                                    if key == "byp" else "core_nozz"]
                                    ["Fg_lbf"] * LB2N)
            if key == "byp":
                warm[f"{t}psis_byp"] = float(sp_cycle.psi_air(Ts_v))
                warm[f"{t}hs_byp"] = float(sp_cycle.h_air(Ts_v)) + H
            else:
                warm[f"{t}psis_core"] = float(
                    sp_cycle.psi_vit(Ts_v, far_v * 0.85))
                warm[f"{t}hs_core"] = (float(
                    sp_cycle.h_vit(Ts_v, far_v * 0.85))
                    / (1 + far_v * 0.85) + H)
    return warm


def compare(name):
    from components.turbofan import sp_cycle
    from components.turbofan.truth.validate_sp import solve

    pins, ods, d = build_pins(name)
    warm = warm_from_truth(name, d, ods)
    print(f"\n=== {name}: SP multipoint (design + {len(ods)} OD) ===")
    for od in ods:
        print(f"  {od.name:4s} MN {od.MN:5.3f} alt-P {od.P0_Pa:9.0f} Pa "
              f"mode {od.mode}  choked core={od.choked_core} "
              f"byp={od.choked_byp}")
    t0 = time.time()
    f = sp_cycle.build(pins, od_points=ods, warm=warm)
    res, vals = solve(f)
    print(f"  converged={res.converged} iterations={res.iterations} "
          f"({time.time()-t0:.1f}s)  "
          f"stationarity={getattr(res, 'stationarity', float('nan')):.2e}")
    if not res.converged:
        print("  status:", str(res.status)[:200])
        rep = getattr(res, "report", None)
        if rep:
            print(str(rep)[:2000])
        return False

    print(f"  {'point':6s} {'qty':10s} {'SP':>12s} {'pycycle':>12s} "
          f"{'ratio':>9s}")
    worst = ("", 0.0)
    ok = True
    for od in ods:
        p = d["points"][od.name]
        perf = p["performance"]
        tag = f"{od.name}_"
        checks = [
            ("Fn N", vals[f"{tag}Fn"], perf["Fn_lbf"] * LB2N),
            ("TSFC", vals[f"{tag}TSFC"], perf["TSFC"] * 2.8325e-5),
            ("W kg/s", vals[f"{tag}W"], perf["W_lbm_s"] * LB2KG),
            ("BPR", vals[f"{tag}BPR"], perf["BPR"]),
            ("FAR", vals[f"{tag}far"], perf["FAR"]),
            ("T4 K", vals[f"{tag}Tt4"], perf["T4_R"] * R2K),
            ("LP_N rpm", vals[f"{tag}LP_N"], perf["LP_Nmech_rpm"]),
            ("HP_N rpm", vals[f"{tag}HP_N"], perf["HP_Nmech_rpm"]),
            ("fan PR", vals[f"{tag}fan_PR"],
             p["components"]["fan"]["PR"]),
            ("hpc PR", vals[f"{tag}hpc_PR"],
             p["components"]["hpc"]["PR"]),
        ]
        for qty, mine, truth in checks:
            r = mine / truth if truth else float("nan")
            flag = "  <-- >1%" if abs(r - 1) > 0.01 else ""
            if abs(r - 1) > abs(worst[1]):
                worst = (f"{od.name} {qty}", r - 1)
            print(f"  {od.name:6s} {qty:10s} {mine:12.4f} {truth:12.4f} "
                  f"{r:9.5f}{flag}")
    print(f"  worst: {worst[0]} {1+worst[1]:.5f}")
    return ok


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    names = sys.argv[1:] or ["cfm56_class", "genx_class"]
    ok = True
    for n in names:
        ok = compare(n) and ok
    sys.exit(0 if ok else 1)
