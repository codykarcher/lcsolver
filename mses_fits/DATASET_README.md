# MSES polars — NC / NE / N3-T airfoil families

Generated 2026-07-31 for per-family surrogate fitting in `sfit`.

## What this is
MSES aero polars for the 17 airfoils in `metafoil/core/raw_airfoil_files/`:
- **NC**: NC090 NC100 NC110 NC120 NC130 NC140 NC145
- **NE**: NE090 NE100 NE110 NE120 NE130 NE140 NE145
- **N3-T**: N3-T100 N3-T120 N3-T140

## Sweep spec
- **Mach**: 0.10–0.68 @0.02, then 0.68–0.86 @0.005  → **66 Mach**
- **Reynolds**: 5e6, 1e7, 2e7, 4e7
- **Alpha**: −3° → +10° @ 0.5° (alpha-mode; CL is a solved OUTPUT). Sweeps auto-truncate
  at the buffet/CLmax wall, so high-Mach polars cover only the feasible CL range.
- **Transition**: FORCED, xtr = 0.03 (top) / 0.05 (bottom) — matches the tasopt reference set.
- **Ncrit**: 9.0
- Solver: MSES (mset+mses, ESP build), driven by `metafoil.mses.run_sweep_multiRe`
  (one shared grid per (airfoil,Mach); Reynolds warm-started from a low-alpha anchor).

## Contents
- `data/shard_r{rank}_{gfirst}_{glast}.npz` — 288 shards, schema identical to
  `metafoil/training/generate_data.py`:
  `upper(8) lower(8) te_gap alpha Re n_crit xtr_u xtr_l mach converged`
  + `cl cd cdp cdf cm cpmin xtr_top xtr_bot cdw` + `bl(192, NaN — not emitted by MSES path)`.
  One row per (airfoil, Mach, Re, alpha). `converged`=1/0; unconverged rows carry NaN outputs.
  Airfoil identity is by the stored Kulfan `upper`/`lower` (match to `Kulfan(airfoil=NAME)`).

## Stats
- **121,176 rows** = 1122 units × 4 Re × 27 alpha (100% coverage, all units written).
- **72,410 converged (59.8%)**. By Re: 53/63/62/61%. By band: subsonic 70%, M0.5–0.68 71%, transonic 51%.
- **11 dead units** (0 converged), all thick foils (NE130/140/145, N3-T120) at M0.85–0.86 —
  physically past drag-divergence/buffet, not a solver failure.
- Runtime: 96-way MPI, ~5.3 h wall.

## Validation
Cross-checked vs `tasopt/Tasopt2.16/polfit/data/polars` (same Re=2e7, same 3%/5% transition):
**2965 points, median CD ratio 1.000, 96% within 5%, 99% within 10%.**

## Load example
```python
import numpy as np, glob
rows = [np.load(f) for f in glob.glob('data/shard_*.npz')]
cl = np.concatenate([r['cl'] for r in rows])
cd = np.concatenate([r['cd'] for r in rows])
conv = np.concatenate([r['converged'] for r in rows]) > 0
# fit cd(cl, mach, Re) per family in sfit using the converged rows
```
