# Running a single case in raw Python

The minimal harness for solving any one aircraft of the matrix by hand,
consolidated from the session runners that produced every number in the
comparison tables. `run.py` remains the batch driver; this is the
one-case-at-a-time form for debugging, sweeps, and feeding the geometry
visualizer.

```python
import warnings, os, dataclasses, json
import numpy as np
warnings.filterwarnings("ignore")

# ---- 1. Environment BEFORE importing aircraft (modules read these at import)
D = "<repo>/examples/spcomparisons"
import sys; sys.path.insert(0, D); os.chdir(D)
os.environ["TECH"] = "cfm56_era"            # per-case, see table
os.environ.setdefault("GEAR_BOX_FRAC", "1.00")
os.environ.setdefault("V_HT_FLOOR", "0.01")
# D8 only: os.environ["H_FIELD_FT"] = "0.0"; os.environ["T_FIELD_K"] = "288.15"

import classes, architectures, aircraft
from edi_compat import structure_detector, unit_corrector
from edi.solvers.ipopt.slcp_bridge import solve_sia
from edi.solvers.ipopt.sia import SIAOptions

# ---- 2. Class + architecture, with per-case overrides
cl = dataclasses.replace(classes.CLASSES["b737"])        # + overrides, see table
ar = dataclasses.replace(architectures.ARCHS["conventional"], lock_mach=True)

# ---- 3. Build and solve
st = structure_detector(unit_corrector(
        aircraft.build(cl, ar, seed="reference")))       # 787: polar="mses_e"
o = SIAOptions(max_iterations=200)                       # 200 = the hard cap
o.stationarity_tolerance = 1e-5
o.condense_numerator = True
o.ipopt_options = dict(o.ipopt_options, tol=1e-9, constr_viol_tol=1e-9)
r = solve_sia(st, options=o, presolve=False)
print(r.status)

# ---- 4. Dump (weights are N internally; the JSON convention converts to lbf)
names = [v.name for v in st["variables"]]
U = {v.name: (str(v.get_units()) if v.get_units() is not None else "")
     for v in st["variables"]}
x = np.asarray(r.x, float)
d = {"_status": str(r.status)}
for k, val in zip(names, x):
    d[k] = float(val)/4.448222 if U.get(k, "") == "N" else float(val)
json.dump(d, open("out.json", "w"))
```

## Per-case settings

Everything not listed is the defaults above.

| case | `TECH` env | class / `dataclasses.replace` overrides | arch key | notes |
|---|---|---|---|---|
| conventional 737 | `cfm56_era` | `b737`, none | `conventional` | the regression anchor |
| strut-braced 737 | `cfm56_era` | `b737` | `strut` | `STRUT_ETAS` env moves the attach (and the planform break with it) |
| **canonical D8** | `d8_era` + `H_FIELD_FT=0.0`, `T_FIELD_K=288.15` | `b737` + `span_max_m=44.20, field_length_ft=4960, ref_mach=0.72, seats_abreast=8, n_aisles=2` | `d8` | `POLAR=mses_c` env; this is the 2-4-2 twin-aisle. Running arch `d8` with plain class defaults gives a 3-3 single-aisle variant — NOT the anchor |
| h2burn | `cfm56_era` | `b737` | `h2burn` | |
| h2fc | `cfm56_era` | `b737` | `h2fc` | `RAD_KHX` env sweeps radiator kg/kW (default 0.5; does not close at 1.0) |
| battery | `cfm56_era` | `citation` + `range_nmi=200` | `battery` | design-range battery cases fail physically, not numerically |
| electric D8s | as battery / h2fc | same | `battery_d8`, `h2fc_d8` | h2fc_d8 does not close at the 36 m gate (span-hungry) — real, not a bug |
| 787 | `modern_composite` | `b787` (two-mission corners are in the class) | `conventional` | `build(..., polar="mses_e")` — the high-speed airfoil family |

## The four details that bite

1. **Env vars go before `import aircraft`** — the modules read them at import
   time; setting them after silently uses defaults.
2. **Inner ipopt `tol=1e-9` / `constr_viol_tol=1e-9` is required.** At the
   1e-12 default the SLCP subproblems falsely report infeasible.
3. **200 iterations is the cap, not a suggestion.** A case past 200 is stuck,
   not slow — go look for the defect (duplicate equalities, free levers,
   detector-hostile shapes; see the component docstrings) instead of raising it.
4. **`lock_mach=True` produced every published comparison number.** Mach-free,
   the optimizer flies slow and buys structure with speed the product could
   not actually give up.

## Reading the JSON

Flat `name -> value`. SI units except weights (converted N -> lbf);
per-segment variables indexed `name[0..4]` (3 climb + 2 cruise; descent is a
compact model, not segments). `VT_*` values are PER FIN; constants (`n_vt`,
seat counts, `tan_gamma` = 0.0875 dihedral) are NOT dumped — only variables.
Absolute geometry anchors for renderers: `x_eng`, `y_eng`, `x_vt_le`,
`x_ht_le`, `LG_x_n/x_m/l_n/l_m`, `LG_z_wing` (wing height above the belly
line; the z-datum is the fuselage base, ground sits `l_n` below it).
