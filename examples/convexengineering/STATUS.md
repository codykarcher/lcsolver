# Rebuild status

`verified` means the EDI rebuild reproduces the reference solution on every
compared variable to the stated tolerance. Run each model's self-check with
`python <dir>/model.py` — this needs only EDI, not gpkit.

| model | source | status | agreement |
|---|---|---|---|
| `simpleac` | gplibrary `SP/SimPleAC` | **verified** | 20/20 vars, max rel 1.8e-5 |
| `windturbine` | Hoburg & Abbeel 2012 (paper only) | **verified** | Betz to 0.13%; Fig 1 peaks within 0.03 |
| `motor` | gplibrary `GP/aircraft/motor` | **verified** | 8/8 vars, max rel 6.1e-7 |
| `propeller` | gplibrary `GP/aircraft/prop` | **verified** | 12/12 vars, max rel 2.3e-4 |
| `fuselage` | gplibrary `GP/aircraft/fuselage` | **verified** | 10/10 vars, max rel 4.0e-7 |
| `tail` | gplibrary `GP/aircraft/tail` | not started | |
| `wing` | gplibrary `GP/aircraft/wing` | not started | vectorized beam model, larger port |
| `turbofan` | York/Hoburg/Drela 2018 | not started | |
| `spaircraft` | Kirschen et al 2018 | not started | partial precedent in `../Kirschen2sp.py` |
| `gassolar` / `solar` | Burton & Hoburg 2018 | **blocked** | see below |
| `jho` | Ozturk et al | not started | shares the `gassolar` blocker |

## The gassolar / solar blocker

These two are the only models blocked rather than merely unstarted. They use
`gpfit`-fitted constraints for wind and solar availability, and the fit
machinery builds numpy arrays from ragged lists of monomials — which numpy
made a hard error in 1.24. `numpy<2` is *not* sufficient; it needs
`numpy<1.24`, which does not build on Python 3.11+.

Reproducing their reference solutions therefore needs an older interpreter as
well as an older numpy. That is doable but was not worth the time against
rebuilding models that run today. The EDI rebuild itself would be
unaffected — only the ground-truth generation is blocked.

## Findings

Collected in [DISCREPANCIES.md](DISCREPANCIES.md). The substantive one:

**Wind turbine equation (14) is missing the tip speed ratio.** Confirmed
three independent ways, including symbolically against the paper's own
equation (18). Only the corrected form yields the Betz limit; as printed, Cp
decays as 1/lambda and never approaches it. The paper's Figure 1 reproduces
correctly with the correction, so the figure was produced with the right
equation and the error is confined to the printed one.

## EDI bugs found and fixed while porting

Three, all on this branch or its parent:

1. `structure_detector` **crashed** (`TypeError`) instead of returning
   `unstructured_dict()` on models it cannot classify as a GP — any model
   with a transcendental hit it.
2. **Constant-only constraints silently defeated structure detection.** A
   constraint like `Qmax >= Q` where both were substituted zeroes to a bare
   negative number, which reads as a subtraction and declares the whole model
   unstructured. A model that *is* a GP was then misrouted to IPOPT, which
   failed with `Error in step computation` and no diagnostic.
3. Six unit/typo defects in the pre-existing `examples/Kirschen2sp.py`,
   including a missing `V_TO**2` and a `rdot_req / I_z` that should have been
   a product.

## Porting notes

The recurring gotcha moving a gpkit model to EDI is the **radian**. gpkit
uses pint, which treats it as dimensionless, so `Q*omega` is a power and
`omega*R` a velocity directly. Pyomo keeps the radian as a real dimension, so
those raise `InconsistentUnitsError` and need an explicit `/units.rad`. This
affects every model with a rotation rate — motor, propeller, and the turbofan
spool speeds.

Nothing else has required a semantic change: the constraint sets transcribe
one-to-one.

## Reproducing the references

See [README.md](README.md) for the isolated-environment recipe and the four
environment landmines (broken PyPI gpkit 1.1.0, the numpy ragged-array
change, the `gpfit` API rename, and `PYTHONPATH` leaking the wrong numpy).
