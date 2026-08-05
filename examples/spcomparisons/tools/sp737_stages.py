"""Three-stage warm solve of the SP-engined 737.

A: solve the baseline (deck-engine) aircraft -- airframe + thrusts.
B: solve the rubber engine STANDALONE, multipoint, at stage-A conditions
   and thrusts, warm-started from the CFM56 truth anchors.
C: build the SP-engined aircraft, load stage-A values into every matching
   name and stage-B values into the Eng_cyc_* names, then solve.
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import pathlib
HERE = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/components")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
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
if os.environ.get("REUSE_A") and os.path.exists("b737_stageA.json"):
    vals_A = json.load(open("b737_stageA.json"))
    print("  (reused b737_stageA.json)")
else:
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
        # FS_V is in KNOTS (house rule 8)
        V=vals_A[f"FS_V[{i}]"] * 0.514444,
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
        choked_core=True, choked_byp=True))

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
                      choked_core=True, choked_byp=True)
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
        choked_core=True, choked_byp=True) for i in active]
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
os.environ["SP_ENGINE"] = os.environ.get("SP_TECH", "cfm56_era")
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
_COLD = bool(os.environ.get("COLD"))
for v in cm.component_data_objects(pyo.Var):
    if _COLD:
        break
    n = v.name
    if n.startswith("Eng_cyc_s3_") and n[len("Eng_cyc_s3_"):] in vals_B:
        # the restructured engine adds an s3_ off-design point at cruise
        # conditions; stage-B's design-tagged state is the right seed
        v.set_value(float(vals_B[n[len("Eng_cyc_s3_"):]]))
        n_set_B += 1
    elif n.startswith("Eng_cyc_") and n[len("Eng_cyc_"):] in vals_B:
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
# per-segment interface variables must be B-CONSISTENT, not deck values:
# the deck's u6/u8/m_fan differ 10-23% from the SP cycle's and loading
# them seeded the interface rows violated.
seg_iface = {}
for i in range(5):
    t = "" if i == i_des else f"s{i}_"
    seg_iface[f"Eng_cyc_T0_{i}"] = segs[i]['T0']
    seg_iface[f"Eng_cyc_P0_{i}"] = segs[i]['P0']
    seg_iface[f"Eng_cyc_u0_{i}"] = segs[i]['V']
    seg_iface[f"Eng_u_6[{i}]"] = vals_B[t + "V_core"]
    seg_iface[f"Eng_u_8[{i}]"] = vals_B[t + "V_byp"]
    seg_iface[f"Eng_m_fan[{i}]"] = vals_B[t + "W"]
    seg_iface[f"Eng_T_t_4[{i}]"] = vals_B[t + "Tt4"]
for v in cm.component_data_objects(pyo.Var):
    if _COLD:
        break
    n = v.name
    if n.startswith("Eng_TSFC["):
        i = int(n[len("Eng_TSFC["):-1])
        v.set_value(float(tsfc_map[i]))
    elif n in seg_iface:
        v.set_value(float(seg_iface[n]))
    elif n in free_vals:
        v.set_value(float(free_vals[n]))

print(f"  built + warmed: A-values {n_set_A}, B-values {n_set_B} "
      f"({time.time()-t0:.0f}s)", flush=True)

# coupling continuation, staged: pin flight state + thrust profile, then
# release thrust, then release everything. Each pass rebuilds fresh (pin
# rows cannot be removed once added) and warm-starts from the previous
# pass. var.fix() and variable bounds are both ignored by the detector;
# ROWS are the only pinning mechanism every layer respects.
from pyomo.environ import units as _u

def build_warmed(source_vals):
    cm_ = unit_corrector(aircraft.build(classes.CLASSES["b737"], ar,
                                        seed="reference"))
    for v in cm_.component_data_objects(pyo.Var):
        n = v.name
        if n in source_vals and source_vals[n] and source_vals[n] > 0:
            v.set_value(float(source_vals[n]))
    return cm_

def snapshot(cm_, st_, x):
    for v, val in zip(st_["variables"], x):
        v.set_value(float(val))
    return {v.name: float(pyo.value(v))
            for v in cm_.component_data_objects(pyo.Var)}

def add_pins(cm_, pred):
    rows, n_ = [], 0
    for v in cm_.component_data_objects(pyo.Var):
        if pred(v.name) and v.value and v.value > 0:
            vu = _u.get_units(v)
            rows.append(v == float(pyo.value(v)) * (vu if vu is not None
                                                    else 1.0))
            n_ += 1
    cm_.ConstraintList(rows)
    return n_

opts = SIAOptions(max_iterations=400)
opts.verbose = bool(os.environ.get("SIA_VERBOSE"))
opts.stationarity_tolerance = 1e-5
opts.condense_numerator = True
# tight trust region: the fitted-surface rows' condensed landscape is
# treacherous away from the seed (cancellation ~10-30 inside single rows),
# and the default step sizes walk the engine off a consistent start.
opts.trust_radius = 0.1
opts.trust_max = 0.5
opts.phase2_restore = True
opts.ipopt_options = dict(opts.ipopt_options, tol=1e-9,
                          constr_viol_tol=1e-9, max_iter=6000)
for kv in os.environ.get("SIA_OPTS", "").split(","):
    if "=" in kv:
        k, v = kv.split("=", 1)
        setattr(opts, k, eval(v))

# assemble the initial warm source from A + B (already set on cm)
warm0 = {v.name: float(pyo.value(v))
         for v in cm.component_data_objects(pyo.Var)
         if pyo.value(v, exception=False) and pyo.value(v) > 0}

_DESIGN = {"Eng_cyc_pi_f_D", "Eng_cyc_pi_lc_D", "Eng_cyc_pi_hc_D",
           "Eng_cyc_BPR_D"}
# Pin only the INDEPENDENT mission coordinates (altitude and Mach) plus
# thrusts and design variables: pinning all 65 FS_* variables duplicated
# the flight-state block's own internal equality rows -- LICQ death, house
# rule 2 -- and the solver drifted off a fully-determined manifold.
_prof = lambda n: (n.startswith("FS_h[") or n.startswith("FS_M[")
                   or n.startswith("Eng_F["))
# The engine INTERFACE pins isolate the consistent stage-B engine seed
# while the airframe re-solves around the SP engine's bigger fan and
# different TSFCs: phase 1's max-violation repair otherwise trades the
# airframe's 1-25% seed violations for violation spread across the
# engine's stiff fitted-surface rows (measured: fan_dhs dragged 4e4 ->
# 1.4e6 and the pass died at feasibility 4.7).
_IFACE_PRE = ("Eng_TSFC[", "Eng_u_6[", "Eng_u_8[", "Eng_m_fan[",
              "Eng_T_t_4[")
_IFACE_SC = {"Eng_d_f", "Eng_d_LPC", "Eng_A_2", "Eng_A_25", "Eng_A_5",
             "Eng_A_7", "Eng_W_engine", "Eng_mbar_fan_D"}
_iface = lambda n: (any(n.startswith(p) for p in _IFACE_PRE)
                    or n in _IFACE_SC)
passes = [
    ("iface", lambda n: _prof(n) or n in _DESIGN or _iface(n)),
    ("prof+D", lambda n: _prof(n) or n in _DESIGN),
    ("prof", _prof),
    ("free", None),
]
if os.environ.get("SWEEP"):
    passes = passes[:2]     # settle the coupled pinned basin, then sweep
if os.environ.get("REF_CASE"):
    ref_case = json.load(open("b737_case_reference.json"))
    cm = build_warmed(ref_case)
    st = structure_detector(cm)
    res = solve_sia(st, options=opts, presolve=False,
                    split_equalities=not bool(os.environ.get("NOSPLIT")))
    feas = float(getattr(res, "max_violation", float("nan")))
    print(f"  REF_CASE free solve: converged={res.converged} "
          f"it={res.iterations}  feas={feas:.2e}", flush=True)
    rep = getattr(res, "report", None)
    if rep:
        print(str(rep)[:1500])
    try:
        sys.path.insert(0, "/private/tmp/claude-501/-Users-codykarcher"
                           "/30345fc2-ed63-4738-8b70-2861477928ba/scratchpad")
        from kkt_verify import verify_kkt
        from edi.solvers.ipopt.slcp_bridge import build_problem as _bpv
        _s3, _f3, _m3 = verify_kkt(_bpv(st, sp_form=True), res.x)
        print(f"  REF_CASE VERIFIER: stationarity {_s3:.3e} "
              f"feasibility {_f3:.3e}", flush=True)
    except Exception as _e:
        print(f"  (verifier failed: {_e})")
    snapR = snapshot(cm, st, res.x)
    json.dump(snapR, open("b737_stageC_ladder.json", "w"))
    for k in ("Eng_cyc_pi_f_D", "Eng_cyc_pi_lc_D", "Eng_cyc_pi_hc_D",
              "Eng_cyc_BPR_D", "W_f_total", "W_total"):
        print(f"    {k} = {snapR.get(k)}")
    sys.exit(0)

if os.environ.get("POLISH"):
    # load the settled raw snapshot and re-solve UNSPLIT from rest: the
    # split-equality representation is a travel device, and from a settled
    # point plain equalities should verify/converge without it
    snap0 = json.load(open("b737_stageC_ladder.json"))
    cm = build_warmed(snap0)
    st = structure_detector(cm)
    opts.max_iterations = 60
    res = solve_sia(st, options=opts, presolve=False,
                    split_equalities=False)
    print(f"  POLISH (unsplit, from rest): converged={res.converged} "
          f"it={res.iterations}  stat="
          f"{float(getattr(res,'stationarity',float('nan'))):.3e}  "
          f"feas={float(getattr(res,'max_violation',float('nan'))):.3e}",
          flush=True)
    try:
        sys.path.insert(0, "/private/tmp/claude-501/-Users-codykarcher"
                           "/30345fc2-ed63-4738-8b70-2861477928ba/scratchpad")
        from kkt_verify import verify_kkt
        from edi.solvers.ipopt.slcp_bridge import build_problem as _bpv
        _s2, _f2, _m2 = verify_kkt(_bpv(st, sp_form=True), res.x)
        print(f"  POLISH VERIFIER: stationarity {_s2:.3e} "
              f"feasibility {_f2:.3e}", flush=True)
    except Exception as _e:
        print(f"  (verifier failed: {_e})")
    sys.exit(0)

if os.environ.get("SINGLE"):
    passes = [("free", None)]   # ONE pass, no pins, no ladder
if os.environ.get("PIND"):
    # consistency check: same single pass, levers row-pinned at the deck
    # cycle (pi_lc follows through the split-ratio row -- pinning it too
    # would duplicate an equality)
    for v in cm.component_data_objects(pyo.Var):
        if v.name == "Eng_cyc_pi_f_D":
            v.set_value(1.685)
        elif v.name == "Eng_cyc_pi_hc_D":
            v.set_value(9.369)
        elif v.name == "Eng_cyc_BPR_D":
            v.set_value(5.105)
    _PINDVALS = {"Eng_cyc_pi_f_D": 1.685, "Eng_cyc_pi_hc_D": 9.369,
                 "Eng_cyc_BPR_D": 5.105}
    passes = [("pinD", lambda n: n in _PINDVALS)]
# the restructured engine's s3_ OD point has no stage-B source for its
# OD-only variables (map coordinates, PR, eff, prm1); seg_warm supplies
# forward-consistent values exactly as the accretive stage-B steps do
if not _COLD:
    warm0.update({f"Eng_cyc_{k}": v
                  for k, v in seg_warm(3, "s3_").items()})

# ---- SEED REPAIR: rows the A+B name-mapping leaves violated ----------
# (external stage-B info; the COLD path self-seeds inside sp_engine)
# tfcool block (sp_engine-only, no stage-B source): solve the three-row
# chain at the stage-B takeoff state with 2% margin.
_Tt3TO = vals_B["s0_hpc_Tt"] if not _COLD else None
_Trr = 1.0 / (1.0 + 0.5 * (1.313 - 1.0) * 1.0 ** 2)
_ef, _tf, _StA = 0.7, 0.30, 0.09
for _r in (1, 2, 3) if not _COLD else ():
    _Tg = (1833.0 + 200.0) if _r == 1 else 1833.0 * _Trr ** (_r - 1)
    _th = min(0.999, (_Tg - 1280.0) / (_Tg - _Tt3TO) * 1.02)
    _e0 = max(_StA * (_th * (1 - _ef * _tf) - _tf * (1 - _ef))
              / (_ef * (1 - _th)), 1e-4) * 1.02
    _ep = _e0 / (1.0 + _e0) * 1.02
    warm0[f"Eng_cyc_theta_cool_{_r}"] = _th
    warm0[f"Eng_cyc_eps0_cool_{_r}"] = _e0
    warm0[f"Eng_cyc_eps_cool_{_r}"] = _ep
if os.environ.get("AUDIT"):
    # evaluate every row at the seed; print the worst violations by name
    import math as _mm
    cm_a = build_warmed(warm0)
    rows = []
    for c in cm_a.component_data_objects(pyo.Constraint, active=True):
        try:
            b = pyo.value(c.body, exception=False)
            lo = pyo.value(c.lower, exception=False) if c.lower is not None else None
            up = pyo.value(c.upper, exception=False) if c.upper is not None else None
        except Exception:
            continue
        if b is None or b != b:
            rows.append((float("inf"), c.name, "NaN body")); continue
        v = 0.0
        if lo is not None and up is not None and lo == up:
            s = max(abs(lo), abs(b), 1e-30)
            v = abs(b - lo) / s
        else:
            if up is not None and b > up:
                v = (b - up) / max(abs(up), 1e-30)
            if lo is not None and b < lo:
                v = max(v, (lo - b) / max(abs(lo), 1e-30))
        if v > 1e-6:
            rows.append((v, c.name, f"body={b:.6g} lo={lo} up={up}"))
    rows.sort(reverse=True)
    print(f"  AUDIT: {len(rows)} rows violated > 1e-6 at the seed")
    byname = {c.name: c for c in cm_a.component_data_objects(
        pyo.Constraint, active=True)}
    for v, nm, d in rows[:25]:
        expr = str(byname[nm].expr) if nm in byname else "?"
        print(f"    {v:10.3e}  {nm}  {d}")
        print(f"        {expr[:220]}")
    sys.exit(0)

if os.environ.get("AUDIT_VAR"):
    _tgt = os.environ["AUDIT_VAR"]
    cm_v = build_warmed(warm0)
    _vals = {v.name: pyo.value(v, exception=False)
             for v in cm_v.component_data_objects(pyo.Var)}
    print(f"  AUDIT_VAR {_tgt}: value = {_vals.get(_tgt)}")
    for c in cm_v.component_data_objects(pyo.Constraint, active=True):
        es = str(c.expr)
        if _tgt in es:
            b = pyo.value(c.body, exception=False)
            print(f"    {c.name}: body={b}")
            print(f"      {es[:200]}")
    sys.exit(0)

if os.environ.get("AUDIT_GP"):
    import numpy as _np
    from edi.solvers.ipopt.slcp_bridge import build_problem as _bp
    cm_g = build_warmed(warm0)
    st_g = structure_detector(cm_g)
    pb = _bp(st_g, sp_form=True)
    x0 = _np.array([float(pyo.value(v)) for v in st_g["variables"]],
                   dtype=float)[:pb.n]
    x0 = _np.where(x0 > 0, x0, 1.0)
    from edi.solvers.ipopt.sia import _log_g
    scored = []
    for ci, c in enumerate(pb.constraints):
        try:
            g = _log_g(c, x0)
        except Exception:
            g = float("nan")
        scored.append((g if g == g else 1e9, ci, c))
    scored.sort(reverse=True)
    print("  AUDIT_GP: worst solver-view rows at the seed:")
    for g, ci, c in scored[:20]:
        terms = list(getattr(c.body, 'terms', None) or [])
        for side in ('p', 'q'):
            sub = getattr(c.body, side, None)
            if sub is not None:
                terms.extend(getattr(sub, 'terms', None) or [])
        involved = sorted({j for _c2, a in terms
                           for j, e in enumerate(a) if e != 0.0})
        nms = [pb.names[j] for j in involved][:6]
        print(f"    log g {g:+9.4f}  [{ci}] {c.operator}  "
              f"{', '.join(nms)}")
    sys.exit(0)

src = warm0
res = None
for label, pred in passes:
    cm = build_warmed(src)
    if pred is not None:
        n_p = add_pins(cm, pred)
    else:
        n_p = 0
    st = structure_detector(cm)
    res = solve_sia(st, options=opts, presolve=False,
                    split_equalities=not bool(os.environ.get("NOSPLIT")))
    feas = float(getattr(res, "max_violation", float("nan")))
    print(f"  pass {label:5s} ({n_p} pinned): converged={res.converged} "
          f"it={res.iterations}  feas={feas:.2e}", flush=True)
    if not res.converged:
        print("  status:", str(res.status)[:160])
        rep = getattr(res, "report", None)
        if rep:
            print(str(rep)[:1600])
        # A pinned pass may be UNABLE to close rows the pins hold open --
        # measured: with u_8[0] pinned, the takeoff noise chain (p2 ~
        # u_8^7.5) cannot close and the pass sticks at ~2e-2 feasibility
        # with the objective stable to 1e-6. The remaining violation is
        # the next (freer) pass's work; hand the warm state on.
        if feas == feas and feas < 0.05:
            print("  (stable, residual localized -- continuing ladder)")
        else:
            sys.exit(1)
    try:
        sys.path.insert(0, "/private/tmp/claude-501/-Users-codykarcher"
                           "/30345fc2-ed63-4738-8b70-2861477928ba/scratchpad")
        from kkt_verify import verify_kkt
        from edi.solvers.ipopt.slcp_bridge import build_problem as _bpv
        _stat, _feas, _m = verify_kkt(_bpv(st, sp_form=True), res.x)
        print(f"  VERIFIER (unsplit, least-squares duals): "
              f"stationarity {_stat:.3e}  feasibility {_feas:.3e}",
              flush=True)
    except Exception as _e:
        print(f"  (verifier failed: {_e})", flush=True)
    src = snapshot(cm, st, res.x)

json.dump(src if isinstance(src, dict) else {},
          open("b737_stageC_ladder.json", "w"))
if os.environ.get("PIND"):
    json.dump(src if isinstance(src, dict) else {},
              open("b737_case_reference.json", "w"))
    print("  wrote b737_case_reference.json (baseline design snapshot)")

if os.environ.get("SWEEP"):
    # milestone-5 continuation: coupled PINNED solves along the lever
    # line deck -> engine-level optimum, each warm from the previous
    import math as _mth
    line = [
        (1.685, 5.105, 30.55),
        (1.655, 5.60, 31.0),
        (1.625, 6.20, 31.5),
        (1.595, 6.90, 32.0),
        (1.565, 7.60, 32.0),
    ]
    results = []
    for (fpr_t, bpr_t, opr_t) in line:
        lc_t = 1.935
        hpc_t = opr_t / (fpr_t * lc_t)
        p_t = _dc.replace(pins, FPR=fpr_t, LPC_PR=lc_t, HPC_PR=hpc_t,
                          BPR=bpr_t,
                          eff_fan=0.8948 - 0.077 * (fpr_t - 1.685))
        try:
            wsD = WSTART.design_state(p_t)
        except Exception as e:
            print(f"  [sweep] fwd warm failed at {fpr_t}/{bpr_t}: {e}")
            continue
        eng_w = {}
        eng_w.update({f"Eng_cyc_{k}": v for k, v in wsD.items()})
        eng_w.update({f"Eng_cyc_s3_{k}": v for k, v in wsD.items()})
        for i in range(5):
            if i == i_des:
                continue
            Tt0_d = segs[i_des]['T0'] * (1 + 0.2 * segs[i_des]['M']**2)
            Tt0_i = segs[i]['T0'] * (1 + 0.2 * segs[i]['M']**2)
            T4_i = min(1587.0 * Tt0_i / Tt0_d, 1833.0 if i < 3 else 1587.0)
            p_i = _dc.replace(p_t, T0_K=segs[i]['T0'], P0_Pa=segs[i]['P0'],
                              MN=segs[i]['M'], V0_m_s=segs[i]['V'],
                              Fn_N=segs[i]['F'], T4_K=T4_i,
                              choked_core=True, choked_byp=True)
            try:
                ws_i = WSTART.design_state(p_i)
            except Exception:
                ws_i = wsD
            eng_w.update({f"Eng_cyc_s{i}_{k}": v for k, v in ws_i.items()})
            eng_w[f"Eng_TSFC[{i}]"] = ws_i.get("TSFC", 1.7e-5) * 35303.9
            eng_w[f"Eng_u_6[{i}]"] = ws_i.get("V_core", 400.0)
            eng_w[f"Eng_u_8[{i}]"] = ws_i.get("V_byp", 290.0)
            eng_w[f"Eng_m_fan[{i}]"] = ws_i.get("W", 150.0)
            eng_w[f"Eng_T_t_4[{i}]"] = ws_i.get("Tt4", 1500.0)
        eng_w["Eng_TSFC[3]"] = wsD.get("TSFC", 1.7e-5) * 35303.9
        eng_w["Eng_u_6[3]"] = wsD.get("V_core", 400.0)
        eng_w["Eng_u_8[3]"] = wsD.get("V_byp", 290.0)
        eng_w["Eng_m_fan[3]"] = wsD.get("W", 150.0)
        eng_w["Eng_T_t_4[3]"] = wsD.get("Tt4", 1587.0)
        A2_t = wsD["fan_Wc"] / 199.7
        A25_t = wsD["hpc_Wc"] / 191.2
        eng_w.update({
            "Eng_cyc_pi_f_D": fpr_t, "Eng_cyc_pi_lc_D": lc_t,
            "Eng_cyc_pi_hc_D": hpc_t, "Eng_cyc_BPR_D": bpr_t,
            "Eng_cyc_eff_fan_D": 0.8948 - 0.077 * (fpr_t - 1.685),
            "Eng_mbar_fan_D": wsD["fan_Wc"],
            "Eng_A_5": wsD.get("A_core", 0.35), "Eng_A_7": wsD.get("A_byp", 1.0),
            "Eng_A_2": A2_t, "Eng_A_25": A25_t,
            "Eng_d_f": (4 * A2_t / (_mth.pi * (1 - 0.30**2))) ** 0.5,
            "Eng_d_LPC": (4 * A25_t / (_mth.pi * (1 - 0.60**2))) ** 0.5,
        })
        src2 = dict(src)
        src2.update(eng_w)
        cm = build_warmed(src2)
        n_p = add_pins(cm, lambda n: _prof(n) or n in _DESIGN)
        st = structure_detector(cm)
        res = solve_sia(st, options=opts, presolve=False,
                        split_equalities=not bool(os.environ.get("NOSPLIT")))
        feas = float(getattr(res, "max_violation", float("nan")))
        snap = snapshot(cm, st, res.x)
        # snapshot magnitudes for these variables are ALREADY lbf
        wf = snap.get("W_f_total", 0.0)
        wt = snap.get("W_total", 0.0)
        we = snap.get("Eng_W_engine", 0.0)
        print(f"  [sweep] FPR {fpr_t:.3f} BPR {bpr_t:.2f} OPR {opr_t:.2f}"
              f"  conv={res.converged} it={res.iterations} feas={feas:.1e}"
              f"  W_f {wf:,.0f}  MTOW {wt:,.0f}  W_eng {we:,.0f} lbf",
              flush=True)
        results.append(dict(FPR=fpr_t, BPR=bpr_t, OPR=opr_t, Wf=wf,
                            MTOW=wt, Weng=we, feas=feas,
                            it=res.iterations))
        if feas == feas and feas < 0.05:
            src = snap          # continuation
    json.dump(results, open("b737_sweep_coupled.json", "w"))
    print("  [sweep] done ->  b737_sweep_coupled.json")
    sys.exit(0)

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
