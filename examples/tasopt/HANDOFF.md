# TASOPT port — handoff

Context for continuing the TASOPT 2.16 → Python port. Everything below is
current as of the last commit on branch `convexengineering-rebuild`.

---

## The job

Port TASOPT 2.16 (Drela's transport aircraft sizing code, Fortran) to Python,
**module by module, each verified against the compiled Fortran to machine
precision**. The goal is a fully functional Python TASOPT that can size an
aircraft end to end.

## Where things are

| what | path |
|---|---|
| Fortran source | `/Users/codykarcher/Desktop/Tasopt2.16/src/` |
| airfoil databases | `/Users/codykarcher/Desktop/Tasopt2.16/air/` (`C.air` etc.) |
| run cases | `/Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.tas` |
| the port | `/Users/codykarcher/Dropbox/research/edi/examples/tasopt/` |
| branch | `convexengineering-rebuild` |

Inside the port:

```
tasopt_py/          the port, by subsystem (gas, aero, engine, structures,
                    sizing, model)
tests/              pytest suite, 202 tests
tests/data/         committed reference CSVs — the suite runs with no compiler
fortran_ref/        the Fortran drivers that regenerate those CSVs
tools/gen_indices.py  generates tasopt_py/model/indices.py from index.inc
STATUS.md           module table + findings about the source
DISCREPANCIES.md    numbered findings (continues the sequence in
                    ../convexengineering/DISCREPANCIES.md, which ends at §21)
```

**The reference program builds and runs.** `cd .../src && make tasopt` (the
`install` step fails, harmless — the binary is built). Then
`cd ../runs/737 && ../../src/tasopt 737` sizes the 737 in 18 iterations to
WTO = 174979 lbf. This is the ground truth for anything ambiguous, and it is
worth using rather than reasoning from the source alone (see "Mistakes I made"
below).

---

## Verification method — this is the important part

Every module is diffed against the compiled Fortran. Two patterns:

**1. Standalone driver (most modules).** Write `fortran_ref/drv_<name>.f`,
compile with the same flags TASOPT uses, dump a CSV of inputs and outputs over
a grid, commit the CSV, and assert the port matches it.

```
gfortran -fdefault-real-8 -O0 -I. -o /tmp/drv drv_<name>.f <name>.f <deps>.f
```

`-fdefault-real-8` is mandatory — TASOPT's `implicit real` is double
precision. Some files also need `-fdollar-ok` (see gotchas).

**2. Instrument the real program (`mission.f`, and the right approach for
`wsize.f`).** Patch the routine to dump its complete input state (`pari`,
`parg`, `parm`, and the whole `para`/`pare` matrices) on entry and its output
state on exit, run the real 737 case, and feed the port exactly those inputs.
Far stronger than a hand-built case — every value is a real converged 737.
See `fortran_ref/mission_instrumented.f` and `tests/test_mission.py`.
**Restore the source file afterwards** (`cp` a saved copy back and rebuild).

Tolerances achieved are 1e-13..1e-16 for closed-form modules. Two exceptions,
both understood and documented: `tfoper` at 9e-10 (numerical vs analytic
Jacobian) and `mission` at 5.7e-6 (an ill-conditioned fixed point at
`ipclimb1` amplifying a 1e-9 engine difference).

---

## Done — 25 modules, 202 tests

| module | source | agreement |
|---|---|---|
| `atmosphere` | `atmos.f` | 4.4e-16 |
| `gas.properties` | `gasfun.f` | 3.6e-14, 2046 values |
| `gas.mixture` | `gascalc.f` (incl. `gas_mass`) | 4.4e-16 |
| `gas.burn` | `gasburn.f` | 1e-13 |
| `structures.fuselage` | `fusew.f` | 3.5e-16 |
| `structures.surface` | `surfw.f` | 3.8e-16 |
| `structures.planform` | `tailpo.f`, `surfdx.f`, `wingsc.f` | 4.1e-16 |
| `aero.moment` | `surfcm.f` | 5.2e-16 |
| `aero.drag` | `surfcd.f` | 1e-13 |
| `aero.loading` | `wingpo.f` | 1e-13 |
| `aero.airfoil` | `airtable.f`, `airfun.f`, `spline.f` | 4.1e-16 |
| `aero.trefftz` | `trefftz1` | 1.3e-13 |
| `aero.cdsum` | `cdsum.f` | 4.4e-16 |
| `aero.axisol` | `axisol.f` | 4.4e-16 |
| `aero.blclosure` | closure routines in `blsys.f` | 4.9e-16 |
| `engine.cooling` | `tfcool.f` | 1e-13 |
| `engine.maps` | `tfmap.f` | 3.4e-16 |
| `engine.tfsize` | `tfsize.f` | 1e-15 |
| `engine.tfoper` | `tfoper.f` | **9e-10** |
| `engine.tfcalc` | `tfcalc.f` (sizing path) | 4.2e-15 |
| `engine.weight` | `tfweight.f` | 2.6e-16 |
| `sizing.balance` | `balance.f` (+`htsize`,`cglpay`) | 3.4e-16 |
| `sizing.takeoff` | `takeoff.f` | 2.8e-16 |
| `sizing.mission` | `mission.f` | **5.7e-6** |
| `model` | `index.inc` | 611 constants, generated |

---

## Left to do

In dependency order.

**1. `blsys.f` — the 3×3 station system (~250 lines of the file).**
`blvar` (values already covered by `aero.blclosure`) plus `blsys` itself,
which assembles the momentum/shape/lag equations for one station against the
previous one. The file is derivative-dominated like `tfoper` — port the
*residuals* and differentiate numerically.

**2. `blax.f` (632 lines) — the global BL Newton.**
An initial direct march downstream, then a Newton over 3n unknowns
(`th`, `ds`/`md`, `ue`) with viscous-inviscid coupling. Local coupling only
(station `i` touches `i-1`), so a banded numerical Jacobian is cheap.

**3. `fusebl.f` (151 lines) — ties `axisol` + `blax` together.**
Straightforward once the two above exist. Produces `DAfsurf`, `DAfwake`,
`KAfTE`, `PAfinf` into `para`. `cdsum` already reads `PAfinf`, and `tfcalc`
already reads `DAfsurf`/`KAfTE` for the BLI defects — so nothing downstream
needs changing.

**4. `wsize.f` (1727 lines) — the outer sizing loop.**
The last piece. Verify it the `mission.f` way: instrument it to dump state and
compare against the real 737 run. It also contains `Wupdate`/`Wupdate0`/
`Wupdate1` (weight-fraction update) and `cfturb` (already ported in
`aero.cdsum`).

**Not needed:** `noise.f` (not on the sizing path), `engwrt` (output
formatting only, lives in `output.f`).

**Then:** size a 737 end to end in Python and diff against `runs/737/737.out`.

---

## Conventions to keep

* **Indices are 1-based** so ported lines match their originals:
  `parg[IGB]` here is `parg(igb)` there. Out-of-range access raises.
  `tasopt_py/model/indices.py` is **generated** — regenerate with
  `python tools/gen_indices.py`, do not hand-edit.
* **Reproduce the source, don't silently fix it.** Where the source does
  something surprising, port it as written, say so in a comment, and add a
  test pinning the behaviour. Several of these are load-bearing.
* **Inject dependencies rather than stubbing.** `surfcd2` takes an `airfoil`
  callable in place of `airfun`, which is what lets a fitted surrogate be
  substituted.
* **Where the Fortran has a hand-written Jacobian, differentiate numerically
  instead** (as `tfoper` does) and say so in the docstring. It costs
  convergence *rate*, not the root, and it is what makes 1265-line routines
  tractable.
* Docstrings explain *why*, and record what was learned about the source.
  Commit messages likewise — they are the record of what was found.
* **No `Co-Authored-By: Claude` trailers.**

---

## Gotchas that have cost real time

**Commented-out constant stacks.** `tfmap.inc` carries four generations of
compressor map constants and `airfrac.inc` three air compositions, in both
cases with **the live one last** and the earlier ones commented out. Taking
the first shifts sized core mass flow 0.14% — close enough to read as a
tolerance problem. Always check which line is actually active.

**Value and derivative lines interleave.** In `blsys.f`, `tfoper.f`, `tfsize.f`
a continuation line often belongs to the *derivative* above, not the value.
Filtering with grep to strip derivatives will merge them and produce a
plausible wrong formula. **Read the raw source for anything with
continuations.** This bit me on `hsl` (7.4e-4 error) and nearly on `CFT`.

**Uppercase routine names.** `compare`, `HKIN`, `HSL`, `CFT` and friends are
declared `SUBROUTINE FOO` in uppercase. A lowercase `grep "subroutine foo"`
finds nothing and you conclude it doesn't exist. Use `grep -i`.

**`constants.inc` is a COMMON block filled at runtime by `tasopt.f`.** A
standalone driver that does not fill it gets `pi = 0`, which silently deletes
terms. Fill it in the driver:
```fortran
      include 'constants.inc'
      pi = 3.1415926535897932384626
      gee = 9.81
      cpSL = 1004.0
      gamSL = 1.4
```

**Include order matters.** `parameter`/`real` declarations must precede `data`
statements, so `include 'tfmap.inc'` must come *before* `include 'airfrac.inc'`.
Getting it wrong compiles cleanly and gives zeros.

**`tfoper.f` needs `-fdollar-ok`** (it declares `res$`/`a$`) **and `compare.f`**,
which is not a dependency of any numerical module. Link `compare.f` or supply a
stub.

**Fortran implicit typing in drivers.** `mb`, `Nb` start with M/N and default
to INTEGER. Declare everything explicitly in drivers.

**Never write `gfortran ... 2>&1 | head`.** The pipe masks compile failures and
you end up running a stale binary and debugging phantom results. Check the exit
status, or don't pipe.

**gfortran's `E26.18` drops the `E`** when the exponent needs three digits
(`0.898846567431157954+308`). The mission loader has a regex for it.

**IEEE vs Python on division by zero.** Several places divide by zero and rely
on getting `±Inf` (`surfcd2`'s `kSuns`, `tfsize`'s `Fsp` at zero airspeed).
Python raises. Write the branch out explicitly with a comment.

---

## Mistakes I made — worth not repeating

**I claimed `tfoper.f` was broken. It isn't.** I wrote that its Newton "fails
to converge for almost any off-design perturbation" and concluded every
TASOPT-sized aircraft rested on a defective solver. That came entirely from a
synthetic driver of my own, with an engine whose LPC/HPC pressure-ratio split
(1.935/9.369) was nothing like the 737's (4.75/3.75). **I never ran the
shipped program before writing it down.** Doing so takes two minutes and
refutes it: 552 converged `tfoper` calls in one 737 sizing, median residual
3.7e-13, zero failures.

The retraction and what actually survives — a genuine robustness limit off the
shipped envelope, where it can exit *reporting success* with 19% residuals
because its convergence test is on step size, not residual — is
`DISCREPANCIES.md` §22. Read it before forming any view about that routine.

**The lesson:** before claiming the reference implementation is wrong, run it.
It builds.

**I also claimed `compare` existed in no source file.** It's in `compare.f`;
my grep was case-sensitive.

---

## Findings about the source worth knowing

Fuller list in `STATUS.md`. The ones that change what results *mean*:

* The **active compressor maps have their off-design penalties switched off**
  (`CK = DK = 0`), so `ecmap` collapses to a straight line in pressure ratio
  with no mass-flow dependence. The elaborate map shape is dead as shipped.
* **Fuselage drag is not computed from the fuselage** in `cdsum` —
  `CDfuse = PAfinf/S`, read from the BL solve. `bodycd` is commented out and
  dead. A drag buildup without the BL solve silently reports zero fuselage drag.
* **`airfun`'s wave drag output is always zero** (`cdw = 0.` unconditionally).
* **BLI is credited, not modelled**, with a bare `CDwing * 0.15` wing wake
  fraction hard-coded separately in both `cdsum` and `tfcalc` — they must stay
  in step (there's a test).
* **`balance`'s `itrim = 2` updates the weight moment but not the weight**, so
  `xCG != xCP` in that mode alone.
* **`mission`'s descent has a predictor *and* a corrector**, both stepping with
  the range interval *behind* the current point; and `FFC` is computed from
  TSFC then unconditionally overwritten from fuel mass flow (the comment says
  "if F < 0" but there is no `if`). Missing either costs ~0.5% on fuel.
* **Dead code, not ported:** `tfani.f`; the second routine in `trefftz.f`
  (its only call site is commented out); `bodycd`.
* `tfweight`'s `iengwgt` 3 and 4 are Gaussian-process surrogates whose
  training data lives in `crddc.inc`; not ported, and calling them raises. No
  shipped case uses them (`737.tas` selects 1).

---

## Quick start

```bash
cd /Users/codykarcher/Dropbox/research/edi/examples/tasopt
python -m pytest tests/ -q            # 202 tests, no compiler needed

# build the reference program
cd /Users/codykarcher/Desktop/Tasopt2.16/src && make tasopt
cd ../runs/737 && ../../src/tasopt 737
```

Read `STATUS.md` first, then `DISCREPANCIES.md` §22.
