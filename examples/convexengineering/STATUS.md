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
| `empennage` | gplibrary `GP/aircraft/tail` | **verified** | 20/20 vars within 1.1e-3; objective to 7e-6 |
| `wing` | gplibrary `GP/aircraft/wing` | **verified to ~1%** | Cd +0.7%, AR 19.92 vs 20.15, weights within 0.3% |
| `turbofan` | York/Hoburg/Drela 2018 | not started | reference runs (244 vars, 13 GP solves) |
| `spaircraft` | Kirschen et al 2018 | not started | partial precedent in `../Kirschen2sp.py` |
| `solar` | Burton & Hoburg 2018 | **verified to <1%** | lat20 -0.5%, lat10 -0.7%; AR/E/Poper within 1% |
| `jho` / `gassolar` | Burton & Hoburg 2018 | **verified to ~2%** | MTOW 110.8 vs 108.5; fuel +0.4% |

## The gassolar / solar blocker is lifted

These were blocked on the `gpfit` fit machinery building numpy arrays from
ragged lists of monomials, which numpy made a hard error in 1.24. Two things
were needed and both now exist:

1. a **python 3.10 + numpy 1.23** conda env (`gpkit-old`) — `numpy<2` alone is
   not enough, and `numpy<1.24` will not build on python 3.11+;
2. the **pre-rename gpfit** checked out as a git worktree at `07b6362~1`.
   The current gpfit renamed both the module and the fit-type keys, so
   `fit(..., "MA")` raises `KeyError` against modern gpfit.

With those, `solar` runs. Reference solutions are recorded in
`solar/reference.json` for four configurations:

| configuration | Wtotal | free vars |
|---|---|---|
| Npod=0, GP, lat 20 | 436.43 | 210 |
| Npod=0, SP, lat 20 | 579.86 | 222 |
| Npod=1, SP, lat 20 | 845.72 | 236 |
| Npod=0, GP, lat 10 | 297.56 | 210 |

Only the `Npod=3, SP` configuration still defeats cvxopt — the original
targets MOSEK. That is a solver limitation, not a model one.

`solar` depends on essentially the whole gplibrary aircraft tree (wing,
empennage, tail boom, fuselage, prop, motor), so the port is gated on the
wing above.

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


## Solver backend — read this before debugging a model

Backend choice is **structure dependent** and matters more than it sounds.
`harness.solve_edi` defaults to `convex_backend="ipopt"`:

* **GP** -> log-space IPOPT. cvxopt stalls with `status='unknown'` on the
  solar aircraft and on the wing at N=8, where IPOPT converges cleanly.
* **SP** -> PCCP (penalty convex-concave), whose every subproblem is a GP.
  EDI now supports PCCP with either inner solver.

gpkit's reference solutions came from MOSEK, stronger than cvxopt again.
**A cvxopt stall says almost nothing about the model.** Several models here
were diagnosed as under-bounded when the real problem was the backend; that
mistake cost hours. Try both before concluding the model is at fault.

## Recurring bug patterns

Four distinct bugs showed up more than once. Every one of them produces a
*self-consistent model with the wrong answer* — the solver is happy,
feasibility is clean, nothing looks wrong. That is the case worth being able
to detect.

**1. Guessed constants.** The wing was 12.8% off in aspect ratio because
`CFRPUD` was assumed (E = 190 GPa) rather than read (137 GPa). Read every
material property and fit coefficient from source.

**2. Dict-key collisions.** A spar dict carrying `"W"` merged into a surface
dict also carrying `"W"`, so the surface weight silently became the spar
weight and the real variable ran to 1e36 while the model stayed feasible.
Prefer `dict(**other)`, which *raises* on a duplicate key, over `.update()`,
which does not — that is the only reason the same bug was caught instantly in
`jho` having taken an hour in `solar`.

**3. Missing load cases.** Structural variables bounded only from above run
to zero, which is an unbounded direction in the log-transformed GP. Every
spar needs a beam; every boom needs a bending case. Cost: the empennage
collapsing to 3.6e-10 lbf, and solar's wing at AR 48.8 instead of 38.1.

**4. Posynomial on the greater side.** `sum(t_i) >= T` is not
GP-representable and was the single constraint of 397 that pushed `jho` out
of GP into SP. The sources avoid it by constraining each element
(`t_i >= T/N`) — the same device the wind turbine uses for equal-power
spanwise bins. maidas' `check_problem_form` finds these immediately.

## The debugging technique that works

**Cross-substitution.** Substitute your optimum into the *reference* model
and evaluate every one of its constraints; whichever it violates names the
defect directly, with no reading or reasoning. This is what found the wing's
material-property bug — six of 130 constraints came back violated, all the
same deflection recursion, and the only term in it not read from source was E.

Two traps, both of which silently return a clean bill of health:

* substituting **bare floats** makes gpkit read them in the wrong units, so
  even `t >= tmin` appears violated — substitute quantities *with* units;
* not substituting the model's **constants** leaves most constraints
  unevaluable, and a bare `except: continue` counts them as passing. Always
  report evaluated-vs-skipped.
