# Rebuild status

`verified` means the EDI rebuild reproduces the reference solution on every
compared variable to the stated tolerance. Run each model's self-check with
`python <dir>/model.py`.

| model | source | status | agreement |
|---|---|---|---|
| `simpleac` | gplibrary `SP/SimPleAC` | **verified** | 20/20 vars, max rel 1.8e-5 |
| `windturbine` | Hoburg & Abbeel 2012 (paper only) | **verified** | Betz to 0.13%; Fig 1 peaks within 0.03 |
| `motor` | gplibrary `GP/aircraft/motor` | **verified** | 8/8 vars, max rel 6.1e-7 |
| `propeller` | gplibrary `GP/aircraft/prop` | **verified** | 12/12 vars, max rel 2.3e-4 |
| `fuselage` | gplibrary `GP/aircraft/fuselage` | **verified** | 10/10 vars, max rel 4.0e-7 |
| `tail` | gplibrary `GP/aircraft/tail` | pending | |
| `wing` | gplibrary `GP/aircraft/wing` | pending | |
| `turbofan` | York/Hoburg/Drela 2018 | pending | |
| `spaircraft` | Kirschen et al 2018 | pending | |
| `gassolar` | Burton & Hoburg 2018 | blocked | reference model needs `numpy<1.24` |
| `jho` | Ozturk et al | pending | |

## Findings so far

Collected in [DISCREPANCIES.md](DISCREPANCIES.md). Briefly:

* **Wind turbine eq (14) is missing the tip speed ratio.** Confirmed three
  ways, including symbolically against the paper's own eq (18). Only the
  corrected form yields the Betz limit; as printed, Cp decays as 1/lambda.
* **Two EDI bugs found and fixed** while porting: a crash in
  `structure_detector` on models it cannot classify, and constant-only
  constraints silently defeating structure detection (which misroutes a GP to
  IPOPT with no diagnostic).

## Porting notes

The recurring gotcha when moving a gpkit model to EDI is the **radian**.
gpkit uses pint, which treats the radian as dimensionless, so `Q*omega` is a
power and `omega*R` is a velocity directly. Pyomo keeps the radian as a real
dimension, so those expressions raise `InconsistentUnitsError` and need an
explicit `/units.rad`. This affects every model with a rotation rate — motor,
propeller, and the turbofan spool speeds.

Nothing else has required a semantic change so far: the constraint sets
transcribe one-to-one.
