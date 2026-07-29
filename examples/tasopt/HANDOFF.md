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
tests/              pytest suite, 389 tests
tests/data/         committed reference CSVs — the suite runs with no compiler
fortran_ref/        the Fortran drivers that regenerate those CSVs
tools/gen_indices.py  generates tasopt_py/model/indices.py from index.inc
tools/gen_beam_indices.py  the same for INDEXB.INC (the ASWING variables)
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

Tolerances achieved are 1e-13..1e-16 for closed-form modules. The exceptions,
both understood and documented: `tfoper` at 9e-10 (numerical vs analytic
Jacobian), and the BL chain at 1e-14 (`blax`'s capped Newton amplifies its
inputs by ~3000; see `STATUS.md`). `mission` is at 9.8e-11 — an earlier
version of this file blamed its then-5.7e-6 on an ill-conditioned fixed point
at `ipclimb1`; that was wrong, it was three porting bugs, and the retraction
is under "Mistakes I made" below.

**Dump the driver's inputs alongside its outputs** when the module sits
downstream of another one. `drv_blax.f` does this, and feeding those exact
inputs back in is what separated "`blax` is exact" (3e-14) from "`axisol`
composed with `blax` drifts" (1.2e-12). Without it the two are indistinguishable
and you cannot tell which module to look at.

---

## Done — 44 modules, 389 tests

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
| `sizing.mission` | `mission.f` | 9.8e-11 |
| `aero.blsys` | `blvar`, `blsys` in `blsys.f` | 1e-14 |
| `aero.blax` | `blax.f` | 3e-14 |
| `aero.fusebl` | `fusebl.f` | 2.8e-14 (real 737 call: 1.4e-15) |
| `linalg` | `gaussn.f` | literal port |
| `sizing.wsize` | `wsize.f` | **real 737 sizing, 1.5e-9** |
| `sizing.woper` | `woper.f` | real 737 off-design run, 1.2e-10 |
| `tasfile` | `getparm.f`, `getval.f` | **737.tas read exactly** |
| `output` | `output.f` (report) | **737.out byte-identical** |
| `sizing.noise` | `noise.f` | real 737 run |
| `acoustics` | `tfnoise.f`, `freq.inc` | three dB values, exact |
| `savefile` | `getsave.f` | header and body, exact |
| `planview` | `airpic.f`, `pltwrt` | both .plt files, exact |
| `plot` | `picwrt`, `picidr` | picwrt's 12 polylines, exact |
| `enginedeck` | `eopwrt` in `tasopt.f` | **737.oute, 4322/4323 lines** |
| `aswing` | `aswout.f`, `BOUTPUT` | **737.asw byte-identical** |
| `optimise` | `fobj.f`, `simpop.f`, `hsort.f` | **18/18 objective calls** |
| `model` | `index.inc`, `INDEXB.INC` | 611 + 103 constants, generated |

---

## The job is done

```
$ python -m tasopt_py /Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.tas \
      --out /tmp/port.out --deck /tmp/port.oute --aswing /tmp/port.asw
diff /tmp/port.out /Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.out
...
737-800: Baseline technology (Aluminum, CFM56 engine)
  WTO   =  174979.1500 lbf   (converged in 18 iterations)
  Wfuel =   47487.2555 lbf
  PFEI  =     7.849124
```

The port reads the case file, sizes the aircraft and flies the off-design
mission, reproducing both convergence tables the shipped program prints. The
Fortran gets 174979.1499 lbf and `PFEI = 7.8491`.

Three tests hold that up: `test_tasfile.py` (the reader produces the exact
state the program hands to `wsize` — every array entry, no tolerance),
`test_wsize.py` (that state sizes to the same aircraft, 1.5e-9) and
`test_woper.py` (the off-design loop, 1.2e-10).

## What is left

**Nothing that TASOPT can be made to do.** Every routine in the Makefile's
link list is ported except the following, and none of them runs:

| source | lines | why not |
|---|---|---|
| `blfwrt2`, `trpwrt`, `trpwrt2` | 154 | Trefftz and BL plot *file syntaxes*; `tasopt_py.plot` draws the same data |
| `gradop.f` | 40 | an empty shell — `DISCREPANCIES.md` §34 |
| `gppre.f` | 47 | the GP surrogate behind `iengwgt` 3/4; no shipped case selects it |
| `tails.f` | 29 | in the link list, called from nowhere |
| `interp2`, `spln2d` | 206 | used only by `airfun1.f`/`airfun2.f`, neither linked |
| `seconds.f` | 25 | a wall-clock timer |
| `BINPUT` in `aswio.f` | 1265 | *reads* `.asw` decks; nothing in TASOPT calls it |

The output path is complete: `.out`, `.oute`, `.asw`, `.sav`, the three
Matlab `.plt` files and the gnuplot stick figure.

```bash
python -m tasopt_py runs/737/737.tas \
    --out port.out --deck port.oute --aswing port.asw
```

`port.out` and `port.asw` are byte-identical to the reference program's;
`port.oute` differs on one line of 4323, a printed rounding boundary.

### If you are looking for something to do

The port reproduces the reference program. The open questions are now about
*using* it, not finishing it:

* **The signomial baseline.** That was goal 3 in the README and nothing here
  has been pointed at it yet. `engine_deck()` returning operating points as
  data is the obvious surrogate-fitting input.
* **A case that is not the 737.** Every verification here is against one
  aircraft. `runs/` has others; sizing one would exercise paths the 737 never
  takes — a Pi-tail (§50), a strut-braced wing, `iengloc = 2`.
* **The five ASWING mistranslations** (§46–§50) are reproduced, not fixed.
  If anyone means to *use* a deck rather than diff it, §48 and §49 change the
  structure and should probably be corrected behind a flag.

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
  **But check first whether the iteration is allowed to run out.** `blax` is
  the counter-example: it caps its Newton at twenty passes, limits the step,
  and stops on *step size* rather than residual, so where it stops depends on
  the iterate path — and the iterate path depends on the Jacobian. There the
  derivatives were transcribed (`aero/blsys.py`) and `gaussn.f` ported
  literally (`linalg.py`), which is what takes that chain from "close" to
  1e-14. `tfoper` gets away with numerical differentiation because it
  converges properly on the shipped engine.
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

**I explained three porting bugs as ill-conditioning.** `mission` agreed with
the real 737 only to 5.7e-6, and I wrote that down as amplification through a
badly conditioned flight-path-angle fixed point at `ipclimb1`, with a
supporting test showing the engine reproduced the Fortran's thrust from the
Fortran's own state. The mechanism was real; it was not what was happening.
The 5.7e-6 was three omissions — `pare(ieM0)` never set in the climb loop, no
`tfcalc` call at end-of-cruise, no descent-CL interpolation — and fixing them
took `mission` to 9.8e-11.

**Why the test could not see them:** it hands the port a *converged* 737
state, so every value the port failed to compute was already sitting in the
array with the right number in it. All three appeared the moment `wsize` ran a
sizing from unset arrays.

**The lesson:** a module test that replays a converged state cannot
distinguish "computed it correctly" from "did not compute it". Where a routine
*writes* state, check that it writes it, not only that the state is right
afterwards.

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
* **`blvar` writes `cd_ue` where it means `cf_ue`** in its wake branch, so the
  last pre-wake `cf_ue` is reused through the whole wake. Jacobian only — no
  answer changes. `DISCREPANCIES.md` §25.
* **`blax` forms the mass defect with and without `rn`** in two places, and
  its step limiter with a third. All cancel at the fixed point. §26.
* **`blax` reads `cdi(1)` without ever writing it**, relying on a zeroed
  `COMMON` block, and zeroes `phi(n+1)` where it means `phi(1)`. §27.
* **Only *one* fuselage BL solve happens per sizing.** `fusebl` is called
  outside the iteration loop, so `PAfinf`/`DAfsurf`/`DAfwake`/`KAfTE` are
  fixed after the first pass. Confirmed by instrumenting the real 737.
* **Dead code, not ported:** `tfani.f`; the second routine in `trefftz.f`
  (its only call site is commented out); `bodycd`; `blax1.f`, `axisol1.f`
  (not in the Makefile). Dead but ported anyway, being XFOIL's: `dilw`,
  `dit`, `hct` — `blvar` computes both inline instead. §28.
* `tfweight`'s `iengwgt` 3 and 4 are Gaussian-process surrogates whose
  training data lives in `crddc.inc`; not ported, and calling them raises. No
  shipped case uses them (`737.tas` selects 1).

---

## Quick start

```bash
cd /Users/codykarcher/Dropbox/research/edi/examples/tasopt
python -m pytest tests/ -q            # 389 tests, ~4 min
TASOPT_SLOW=1 python -m pytest tests/  # + the 18-evaluation optimiser check

# run the port itself
python -m tasopt_py /Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.tas \
      --out /tmp/port.out --deck /tmp/port.oute --aswing /tmp/port.asw
diff /tmp/port.out /Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.out

# build the reference program
cd /Users/codykarcher/Desktop/Tasopt2.16/src && make tasopt
cd ../runs/737 && ../../src/tasopt 737
```

Read `STATUS.md` first, then `DISCREPANCIES.md` §22.

If you touch a file under `Tasopt2.16/src/` to instrument it, **copy it aside
first and copy it back afterwards**, then rebuild and confirm the 737 still
sizes to 174979.1499 lbf in 18 iterations. The instrumented copies live in
`fortran_ref/` (`mission_instrumented.f`, `fusebl_instrumented.f`).
