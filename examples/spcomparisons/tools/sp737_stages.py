"""Three-stage warm solve of the SP-engined 737 (milestone 4/5 runner).

Run from anywhere; paths are absolute. Stage C is the OPEN item: it runs
without structural blowups but wanders to an unphysical basin by dragging
the free flight-state variables -- the next step is a coupling
continuation (pin FS_h/FS_T_atm to the stage-A profile for a first pass,
then release). See the 9ff2807 commit message for the full story.


A: solve the baseline (deck-engine) aircraft -- airframe + thrusts.
B: solve the rubber engine STANDALONE, multipoint, at stage-A conditions
   and thrusts, warm-started from the CFM56 truth anchors.
C: build the SP-engined aircraft, load stage-A values into every matching
   name and stage-B values into the Eng_cyc_* names, then solve.
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
HERE = "/Users/codykarcher/Dropbox/research/edi/examples/spcomparisons"
sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/components")
sys.path.insert(0, "/Users/codykarcher/Dropbox/research/edi")
os.chdir(HERE)
os.environ.pop("SP_ENGINE", None)

import regression_gates as RG
import pyomo.environ as pyo
from pyomo.environ import units as u

LBF = 0.2248089431
N_LBF = 1.0 / LBF

t_all = time.time()

# ---- stage A ------------------------------------------------------------
print("=== stage A: baseline deck-engine solve ===", flush=True)
vals_A = RG.solve_case("b737", "conventional_M")
assert vals_A, "stage A failed"
json.dump(vals_A, open("b737_stageA.json", "w"))

segs = []
for i in range(5):
    segs.append(dict(
        T0=vals_A[f"FS_T_atm[{i}]"],
        # solved values are unit-corrected to base SI: already Pa
        P0=vals_A[f"FS_P_atm[{i}]"],
        M=vals_A[f"FS_M[{i}]"],
        V=vals_A[f"FS_V[{i}]"],
        F=vals_A[f"Eng_F[{i}]"] * N_LBF,     # stored lbf -> N
    ))
for i, sg in enumerate(segs):
    print(f"  seg{i}: T {sg['T0']:.1f} K  P {sg['P0']:.0f} Pa  "
          f"M {sg['M']:.3f}  F {sg['F']:.0f} N")

# ---- stage B ------------------------------------------------------------
print("=== stage B: standalone rubber-engine warm solve ===", flush=True)
from components.turbofan import sp_cycle as SC
from components.turbofan.truth.validate_sp_od import (build_pins,
                                                      warm_from_truth)
from components.turbofan.truth.validate_sp import solve as solve_cycle

pins_t, ods_t, d_t = build_pins("cfm56_class")
warm_t = warm_from_truth("cfm56_class", d_t, ods_t)

i_des = 3
pins = SC.CyclePins(
    name="b737_mission", T0_K=segs[i_des]['T0'], P0_Pa=segs[i_des]['P0'],
    MN=segs[i_des]['M'], V0_m_s=segs[i_des]['V'],
    # design pinned at the DECK's SOLVED CRUISE OPERATING STATE -- its
    # segment-3 pressure ratios and Tt4, not its nameplate design values.
    # The deck engine is an oversized fixed machine flying throttled;
    # anchoring the map frame at its actual cruise state is what makes the
    # climb segments reachable (nameplate PRs at 1360 K degenerate: the
    # walk blew W 309 -> 838 kg/s before going infeasible).
    # the deck's lc/hc SPLIT is TASOPT's booster convention (pi_lc 7.8,
    # pi_hc 2.3) and does not port to the NASA-map architecture; only the
    # core OPR does. Keep the deck's FPR/OPR/BPR/T4, split the core the
    # map-natural way.
    # the VALIDATED anchor configuration at mission cruise: cruise is the
    # largest corrected-thrust demand, so a cruise-at-rating engine covers
    # the mission under its caps; the deck's own off-design state is not
    # physically consistent under real cycle physics and is not a target.
    Fn_N=segs[i_des]['F'], T4_K=1587.0,
    FPR=1.685, LPC_PR=1.935, HPC_PR=9.369, BPR=5.105,
    eff_fan=0.8948, eff_lpc=0.9243, eff_hpc=0.8707,
    eff_hpt=0.8888, eff_lpt=0.8996,
    choked_core=True, choked_byp=True)

ods = []
for i in range(5):
    if i == i_des:
        continue
    ods.append(SC.ODPins(
        name=f"s{i}", T0_K=segs[i]['T0'], P0_Pa=segs[i]['P0'],
        MN=segs[i]['M'], V0_m_s=segs[i]['V'],
        mode='F', F_N=segs[i]['F'],
        T4_cap_K=1833.0 if i == 0 else 1587.0,
        choked_core=(i >= 3), choked_byp=True))

# Accretive continuation with FORWARD-EVALUATED warm states: each mission
# point gets its own design_state() at its own conditions, thrust and a
# corrected-similarity Tt4 estimate. Pressure-scaling a cruise state into a
# hot high-power climb state kept landing in the same garbage basin
# (stalled objective byte-identical across three structural changes).
from components.turbofan.truth import warm_start as WSTART
import dataclasses as _dc

def seg_warm(i, tag):
    Tt0_des = segs[i_des]['T0'] * (1 + 0.2 * segs[i_des]['M']**2)
    Tt0_i = segs[i]['T0'] * (1 + 0.2 * segs[i]['M']**2)
    T4_i = min(1587.0 * Tt0_i / Tt0_des, 1833.0 if i < 3 else 1587.0)
    p_i = _dc.replace(pins, T0_K=segs[i]['T0'], P0_Pa=segs[i]['P0'],
                      MN=segs[i]['M'], V0_m_s=segs[i]['V'],
                      Fn_N=segs[i]['F'], T4_K=T4_i,
                      choked_core=(i >= 3), choked_byp=True)
    ws = WSTART.design_state(p_i)
    outw = {f"{tag}{k}": v for k, v in ws.items()
            if not k.startswith("_")}
    # off-design-only variables the design evaluator does not emit
    r = (Tt0_des / Tt0_i) ** 0.5
    outw[f"{tag}LP_N"] = pins.LP_Nmech / r * 1.0
    outw[f"{tag}HP_N"] = pins.HP_Nmech / r * 1.0
    outw[f"{tag}BPR"] = pins.BPR
    import components.turbofan.sp_maps as MM
    for key, M in (("fan", MM.FAN), ("lpc", MM.LPC), ("hpc", MM.HPC)):
        outw[f"{tag}{key}_NcMap"] = M['NcMap_d']
        outw[f"{tag}{key}_R"] = M['RlineMap_d']
        at_PR = float(SC.map2d(M['terms_PR'], M['NcMap_d'] / M['x0'],
                               M['RlineMap_d'] / M['y0']))
        outw[f"{tag}{key}_prm1"] = at_PR - 1.0
        outw[f"{tag}{key}_PR"] = {"fan": pins.FPR, "lpc": pins.LPC_PR,
                                  "hpc": pins.HPC_PR}[key]
        outw[f"{tag}{key}_eff"] = {"fan": pins.eff_fan,
                                   "lpc": pins.eff_lpc,
                                   "hpc": pins.eff_hpc}[key]
    for key, M in (("hpt", MM.HPT), ("lpt", MM.LPT)):
        outw[f"{tag}{key}_NpMap"] = M['NpMap_d']
        outw[f"{tag}{key}_PRmap"] = M['PRmap_d']
        outw[f"{tag}{key}_eff"] = {"hpt": pins.eff_hpt,
                                   "lpt": pins.eff_lpt}[key]
    return outw

warm = WSTART.design_state(pins)
order = [None, 4, 2, 1, 0]
active = []
vals_B = None
for step, add in enumerate(order):
    if add is not None:
        active.append(add)
    ods = [SC.ODPins(
        name=f"s{i}", T0_K=segs[i]['T0'], P0_Pa=segs[i]['P0'],
        MN=segs[i]['M'], V0_m_s=segs[i]['V'],
        mode='F', F_N=segs[i]['F'],
        T4_cap_K=1833.0 if i < 3 else 1587.0,
        choked_core=(i >= 3), choked_byp=True) for i in active]
    if vals_B is not None:
        warm = dict(vals_B)
        if add is not None:
            warm.update(seg_warm(add, f"s{add}_"))
    f_eng = SC.build(pins, od_points=tuple(ods), warm=warm)
    res_B, vals_B = solve_cycle(f_eng)
    label = "design" if add is None else f"+s{add}"
    print(f"  step {label:7s}: converged={res_B.converged} "
          f"it={res_B.iterations}", flush=True)
    if not res_B.converged:
        print("  status:", str(res_B.status)[:200])
        rep = getattr(res_B, "report", None)
        if rep:
            print(str(rep)[:1200])
        sys.exit(1)
print(f"  design TSFC {vals_B['TSFC']*35303.9:.4f} 1/hr  "
      f"far {vals_B['far']:.5f}  Tt4 {vals_B['Tt4']:.1f} K  "
      f"W {vals_B['W']:.1f} kg/s")
json.dump(vals_B, open("b737_stageB_engine.json", "w"))

# ---- stage C ------------------------------------------------------------
print("=== stage C: SP-engined aircraft, warm ===", flush=True)
os.environ["SP_ENGINE"] = "cfm56_era"
import importlib
for mod in ("components.technology", "components.turbofan.model",
            "components.wing", "aircraft"):
    if mod in sys.modules:
        importlib.reload(sys.modules[mod])
import classes, architectures, aircraft, dataclasses
from edi_compat import structure_detector, unit_corrector
from edi.solvers.ipopt.slcp_bridge import solve_sia
from edi.solvers.ipopt.sia import SIAOptions

ar = dataclasses.replace(architectures.ARCHS["conventional"],
                         lock_mach=True)
t0 = time.time()
cm = unit_corrector(aircraft.build(classes.CLASSES["b737"], ar,
                                   seed="reference"))

n_set_A = n_set_B = 0
for v in cm.component_data_objects(pyo.Var):
    n = v.name
    if n.startswith("Eng_cyc_") and n[len("Eng_cyc_"):] in vals_B:
        v.set_value(float(vals_B[n[len("Eng_cyc_"):]]))
        n_set_B += 1
    elif n in vals_A:
        val = vals_A[n]
        if str(u.get_units(v)) == "N":
            val = val * N_LBF
        if val and val > 0:
            v.set_value(float(val))
            n_set_A += 1
# interface TSFC in 1/hr from the standalone engine
for i in range(5):
    tag = "" if i == i_des else f"s{i}_"
    for v in cm.component_data_objects(pyo.Var):
        pass
tsfc_map = {i: vals_B[("" if i == i_des else f"s{i}_") + "TSFC"] * 35303.9
            for i in range(5)}
# the FREE design variables and unit-ful interface scalars must be
# consistent with the loaded cycle state -- the Pt chain embeds FPR 1.685,
# and leaving pi_f_D at its 1.65 tech guess blew the engine block apart on
# the first subproblem (lpc_Tt ran to 6e4 K).
import math as _math
free_vals = {
    "Eng_cyc_pi_f_D": 1.685, "Eng_cyc_pi_lc_D": 1.935,
    "Eng_cyc_pi_hc_D": 9.369, "Eng_cyc_BPR_D": 5.105,
    "Eng_cyc_eff_fan_D": 0.8948,
    "Eng_mbar_fan_D": vals_B["fan_Wc"],
    "Eng_A_5": vals_B["A_core"], "Eng_A_7": vals_B["A_byp"],
}
A2v = vals_B["fan_Wc"] / 199.7
A25v = vals_B["hpc_Wc"] / 191.2
free_vals["Eng_A_2"] = A2v
free_vals["Eng_A_25"] = A25v
free_vals["Eng_d_f"] = (4 * A2v / (_math.pi * (1 - 0.30**2))) ** 0.5
free_vals["Eng_d_LPC"] = (4 * A25v / (_math.pi * (1 - 0.60**2))) ** 0.5
for v in cm.component_data_objects(pyo.Var):
    n = v.name
    if n.startswith("Eng_TSFC["):
        i = int(n[len("Eng_TSFC["):-1])
        v.set_value(float(tsfc_map[i]))
    elif n in free_vals:
        v.set_value(float(free_vals[n]))

print(f"  built + warmed: A-values {n_set_A}, B-values {n_set_B} "
      f"({time.time()-t0:.0f}s)", flush=True)

st = structure_detector(cm)
opts = SIAOptions(max_iterations=400)
opts.stationarity_tolerance = 1e-5
opts.condense_numerator = True
opts.ipopt_options = dict(opts.ipopt_options, tol=1e-9,
                          constr_viol_tol=1e-9)
res = solve_sia(st, options=opts, presolve=False)
print(f"  converged={res.converged} it={res.iterations} "
      f"({time.time()-t0:.0f}s total {time.time()-t_all:.0f}s)")
print(f"  status: {str(res.status)[:200]}")
if not res.converged:
    rep = getattr(res, "report", None)
    if rep:
        print(str(rep)[:2500])
    sys.exit(1)

for v, val in zip(st["variables"], res.x):
    v.set_value(float(val))
vals = {}
for v in cm.component_data_objects(pyo.Var):
    val = pyo.value(v)
    vals[v.name] = val * LBF if str(u.get_units(v)) == "N" else val
print(f"\n  {'MTOW':16s} {vals['W_total']:12,.1f} lbf   "
      f"(baseline {vals_A['W_total']:,.1f})")
print(f"  {'fuel':16s} {vals['W_f_total']:12,.1f} lbf   "
      f"(baseline {vals_A['W_f_total']:,.1f})")
print(f"  {'W_engine':16s} {vals['Eng_W_engine']:12,.1f} lbf   "
      f"(baseline {vals_A['Eng_W_engine']:,.1f})")
print("\n  FREE CYCLE at the optimum (deck values in parens):")
print(f"    FPR    {vals['Eng_cyc_pi_f_D']:.4f}  (1.685)")
print(f"    LPC PR {vals['Eng_cyc_pi_lc_D']:.4f}  (1.935)")
print(f"    HPC PR {vals['Eng_cyc_pi_hc_D']:.4f}  (9.369)")
print(f"    OPR    {vals['Eng_cyc_pi_f_D']*vals['Eng_cyc_pi_lc_D']*vals['Eng_cyc_pi_hc_D']:.2f}  (30.55, cap 32)")
print(f"    BPR    {vals['Eng_cyc_BPR_D']:.4f}  (5.105)")
print(f"    fan Wc {vals['Eng_cyc_fan_Wc']:.1f} kg/s")
print(f"    eff_fan {vals['Eng_cyc_eff_fan_D']:.4f}  (0.8948 at FPRo)")
json.dump(vals, open("b737_sp_rubber.json", "w"))
print("wrote b737_sp_rubber.json")
