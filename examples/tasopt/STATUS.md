# TASOPT Python port

A module-by-module port of TASOPT 2.16, each verified against the compiled
Fortran. TASOPT is built with `-fdefault-real-8`, so `implicit real` is double
precision; the reference drivers use the same flag.

## Verified

| module | source | check |
|---|---|---|
| `atmosphere` | `atmos.f` | 4.4e-16 |
| `gas.properties` | `gasfun.f` | 2046 values, 3.6e-14 |
| `gas.mixture` | `gascalc.f` | 212 values + `gas_mass`, 4.4e-16 |
| `gas.burn` | `gasburn.f` | 60 values, 1e-13 |
| `structures.fuselage` | `fusew.f` | 66x3 configs, 3.5e-16 |
| `structures.surface` | `surfw.f` | 111x3 planforms, 3.8e-16 |
| `structures.planform` | `tailpo.f`, `surfdx.f`, `wingsc.f` | 52 values, 4.1e-16 |
| `aero.moment` | `surfcm.f` | 15x3, 5.2e-16 |
| `aero.drag` | `surfcd.f` | 30 values, 1e-13 |
| `aero.loading` | `wingpo.f` | 20 values, 1e-13 |
| `aero.airfoil` | `airtable.f`, `airfun.f`, `spline.f` | 328 values, 4.1e-16 |
| `aero.trefftz` | `trefftz.f` (`trefftz1`) | 90 values, 1.3e-13 |
| `aero.cdsum` | `cdsum.f` | 64 values, 4.4e-16 |
| `engine.cooling` | `tfcool.f` | 105 values, 1e-13 |
| `engine.maps` | `tfmap.f` | 170 values, 3.4e-16 |
| `engine.tfsize` | `tfsize.f` | 54 values x 3 cases, 1e-15 |
| `engine.tfoper` | `tfoper.f` | 124 values x 4 cases, **9e-10** |
| `engine.weight` | `tfweight.f` | 72 values, 2.6e-16 |
| `sizing.balance` | `balance.f` | 56 values, 3.4e-16 |
| `sizing.takeoff` | `takeoff.f` | 36 values, 2.8e-16 |
| `sizing.mission` | `mission.f` | real 737 state, 9.8e-11 |
| `aero.axisol` | `axisol.f` | 873 values, 4.4e-16 |
| `aero.blclosure` | closure routines in `blsys.f` | 360 values, 4.9e-16 |
| `aero.blsys` | `blvar`, `blsys` in `blsys.f` | 2200 values, 1e-14 |
| `aero.blax` | `blax.f` | 2350 values, 3e-14 |
| `aero.fusebl` | `fusebl.f` | 28 values, 2.8e-14 |
| `linalg` | `gaussn.f` | literal port |
| `sizing.wsize` | `wsize.f` | **real 737 sizing, 1.5e-9** |
| `sizing.woper` | `woper.f` | real 737 off-design run, 1.2e-10 |
| `tasfile` | `getparm.f`, `getval.f` | **737.tas read exactly** |
| `output` | `output.f` (summary sections) | **byte-exact vs 737.out** |
| `model` | `index.inc` | 611 constants, generated |

276 tests. Reference CSVs are committed, so the suite runs without a Fortran
compiler; the drivers in `fortran_ref/` regenerate them.

`tfoper` is the one module at 1e-10 rather than 1e-13: it differentiates
numerically where the source differentiates analytically, so the two Newtons
stop at slightly different points inside the same convergence ball.

The reference program itself builds and runs: `runs/737/737.tas` sizes in 18
iterations to WTO = 174979 lbf. That is the check every claim below is held
against.

## The port sizes a 737

```
  iterw     errW          WMTO        Wfuel        Wfuse        Wwing         Weng        span     area     HTarea   xwbox
    18 -0.0000000001  174979.1500   47487.2555   37025.2291   23717.2688   12280.0283  116.427  1342.101   456.859    53.68101
```

Against the shipped program's

```
    18 -0.0000000001  174979.1499   47487.2555   37025.2291   23717.2688   12280.0282  116.427  1342.101   456.859    53.68101
```

Same 18 iterations, same aircraft. `wsize.f` was instrumented to dump its
complete input and output state on the real `runs/737/737.tas` run; the port
is handed those inputs and sizes from them, running its own fuselage boundary
layer, structures, engine and mission the whole way. The converged aircraft
agrees to **1.5e-9** across `parg`, 1.0e-7 across `para` and 5.2e-8 across
`pare`. The floor is `tfoper`'s numerical Jacobian (see below).

## The port runs as a program

```
$ python -m tasopt_py /path/to/runs/737/737.tas
...
737-800: Baseline technology (Aluminum, CFM56 engine)
  WTO   =  174979.1500 lbf   (converged in 18 iterations)
  Wfuel =   47487.2555 lbf
  PFEI  =     7.849124
  mission 2: WTO =  174979.1500 lbf   (converged in 3 iterations)
```

against the shipped program's 174979.1499 lbf and the `PFEI = 7.8491` in
`737.out`. Both convergence tables are reproduced line for line.

`tasfile.read_tas` is checked hard: it reads `737.tas` and produces the
parameter state the shipped program handed to `wsize`, **entry for entry with
no tolerance** — all 27 `pari`, 254 `parg`, 17 `parm`, 51x17 `para` and
269x17 `pare` values, unset entries included.

So is the report. `python -m tasopt_py 737.tas --out port.out` writes the
summary sections of the `.out` file and they come out **byte-identical** to
the reference program's own `737.out` — 345 of 355 lines, every value at its
printed precision in its printed column. The ten that differ are the five per
mission that `noise.f` fills, and `noise.f` is not ported.

## Still to port

| source | lines | what it is |
|---|---|---|
| `fobj.f`, `gradop.f`, `simpop.f` | ~600 | the optimiser wrapper around `wsize` |
| `noise.f`, `tfnoise.f` | ~700 | noise estimate — five report lines depend on it |
| `output.f` (`airwrt`, `engwrt`) | ~620 | the per-point engine dump, 2100 lines of `737.out` |
| `getsave.f` | 150 | optimiser restart files |

Nothing that computes an aircraft remains. What is left is the optimiser that
wraps the sizing loop, the noise estimate, and output formatting.

## Accuracy of the boundary-layer chain

Worth stating together, because the numbers are not all the same and the
reason matters. `axisol` (potential flow) agrees to 4.4e-16 and `blax`
(the BL Newton) to 3e-14 when each is driven with the Fortran's own inputs.
Composed, the answer moves by up to 1.2e-12: `blax` runs a *limited* Newton —
twenty passes, step-limited, stopping on step size rather than residual — so
it amplifies an input perturbation by about three thousand. The amplification
lands on derived quantities (`ct`, `cf`); the state variables `ue` and `th`
still agree to 3e-15, and `fusebl`'s four outputs to 2.8e-14.

This is also why `blsys` and `blax` carry the Fortran's *analytic* Jacobian
rather than differentiating numerically the way `tfoper` does, and why
`gaussn.f` is ported literally rather than replaced by a library solve. An
iteration that is allowed to run out is not a fixed point, so the answer
depends on the iterate path, and the iterate path depends on the Jacobian and
the elimination.

The strongest check available: instrumenting `fusebl.f` and re-running the
shipped 737 shows the program performs exactly **one** fuselage BL solve per
sizing (the geometry does not change over the loop). This port reproduces that
solve to **1.4e-15**. `tests/test_fusebl.py` pins the four numbers.

## Things found in the source

Recorded because they change what the results mean, and none is visible from
a call site.

**Four bugs a converged-state test could not see.** Writing the end-to-end
`wsize` test found three omissions in `mission` — a missing `pare(ieM0)`
assignment in the climb loop, a missing `tfcalc` call at end-of-cruise, and a
missing descent-CL interpolation. A fourth turned up when the `.out` report
was ported: range, time, weight fraction and buoyancy are zeroed at four
ground points and this port only did one, which nothing but the report reads.
All four are invisible when the port is
handed a *converged* 737 state, because the value it should have computed is
already sitting in the array; they only appear when the arrays start unset, as
they do in a real sizing. `mission`'s agreement went from 5.7e-6 to 9.8e-11
once they were fixed, which retracts the "ill-conditioned fixed point at
`ipclimb1`" explanation this file used to carry for that 5.7e-6. There was no
ill-conditioning; there were bugs. See `tests/test_mission.py`.

**`tfoper.f` is not robust off the shipped engine envelope.** On the shipped
737 it is fine — 552 converged calls in one sizing, worst residual 1.8e-10,
zero failures. On an engine with an unusual LPC/HPC pressure-ratio split it
can fail to converge, and in one case *exits reporting success with 19%
residuals*, because its convergence test is on the Newton step size rather
than the residual. This port's numerical Jacobian solves those cases; on the
shipped engine the two agree to 9e-10. See `DISCREPANCIES.md` §22 — an earlier
and much stronger version of this claim was wrong and is retracted there.

**Constants hide in commented-out stacks.** `tfmap.inc` carries four
generations of compressor map constants and `airfrac.inc` three air
compositions, in both cases with the live one *last* and the others commented
out. Taking the first shifts sized core mass flow by 0.14% — small enough to
read as a tolerance problem rather than wrong constants. Both were caught only
by comparing against the Fortran.

**The active compressor maps have their off-design penalties switched off.**
`CK = DK = 0` in all three live sets, so `ecmap` collapses to a straight line
in pressure ratio with no mass-flow dependence at all. The elaborate map shape
in the source is dead in the shipped configuration.

**Fuselage drag is not computed from the fuselage.** `cdsum` reads
`CDfuse = PAfinf/S` out of `para`, put there by the BL solve. The wetted-area
route through `bodycd` is commented out, so `bodycd` is dead — and a drag
buildup run without the BL solve reports zero fuselage drag rather than
failing.

**BLI is credited, not modelled**, and the wing credit is scaled by a bare
`CDwing * 0.15`, "assume 15% of the wing dissipation is in wake".

**`airfun`'s wave drag output is always zero.** `cdw = 0.` unconditionally;
compressibility drag reaches the aircraft through `surfcd`'s own correlation.
Outside the tabulated `cl`/`tau` box there is no guard, only a quadratic
penalty folded into `cdp` — an optimiser fence, not physics.

**`balance`'s `itrim = 2` updates the weight moment but not the weight**, so
the reported `xCG` does not equal `xCP` in that mode alone.

**`tfoper.f` needs two things to link.** `-fdollar-ok`, for its `res$`/`a$`
debug declarations; and `compare.f`, which defines `compare(ss, aa, dd)` as
uppercase `SUBROUTINE COMPARE` and is not a dependency of any numerical
module. The call sits behind `if (iter .eq. -1)` so it never runs, but the
reference is emitted anyway. See `DISCREPANCIES.md` §23.

**`blvar` writes `cd_ue` where it means `cf_ue`.** In the wake branch of
`blsys.f`'s `blvar`, so `cf_ue` keeps the value from the last station before
the wake and is reused through it. Confirmed by presetting the variable and
running the compiled routine. It reaches only the Jacobian — `cf` itself is
zero there — so no answer changes; see `DISCREPANCIES.md` §25.

**`blax` builds the mass defect two different ways**, with and without the
lateral-divergence factor `rn`, and its step limiter inverts `md -> ds` with a
third. All cancel at the fixed point. §26.

**`blax` reads `cdi(1)` without ever writing it**, and relies on the caller's
array being a zeroed `COMMON` block. Its `phi` initialisation writes
`phi(n+1)` rather than `phi(1)`, an off-by-one on a leftover loop variable.
Both harmless in the shipped program, neither harmless in general. §27.

**`Wupdate0`'s weight-explosion guard cannot fire** — it hardwires `fsum = 0`
— and a **non-converged sizing is not an error**: `wsize.f` prints a warning
and carries on into the takeoff and balance calculations, its `return` being
commented out. §29, §30.

**Dead code, not ported:** `tfani.f` entirely; `trefftz` (the second routine
in `trefftz.f`, whose only call site is commented out); `bodycd`; `blax1.f`
and `axisol1.f` (neither is in the Makefile); `muair` in `wsize.f`, which nothing calls and which would disagree with `atmos` if it did (§31). **Dead but ported anyway,**
because they are XFOIL's and a reader will look for them: `dilw`, `dit` and
`hct` in `blsys.f` are called from nowhere — `blvar` computes the density
shape parameter and the turbulent dissipation inline instead. §28.

**`constants.inc` is a COMMON block filled at runtime by `tasopt.f`.** A
standalone driver that does not fill it gets `pi = 0`, which silently deletes
the engine inlet's contribution to the neutral point. Every driver in
`fortran_ref/` that touches it fills it.

## Conventions

Where a routine needs a table this port does not carry, the dependency is
injected rather than stubbed — `surfcd2` takes an `airfoil` callable in place
of `airfun`, which is what lets a fitted surrogate be substituted, and is
exactly what the signomial formulation does.

Index constants are **generated** from `index.inc` by `tools/gen_indices.py`
rather than transcribed: 611 hand-copied integers is 611 chances at a silent
off-by-one. They stay 1-based so ported lines match their originals, and
out-of-range access raises.

Where the source does something surprising, the port reproduces it and says
so in a comment, with a test pinning the behaviour. Corrections are not made
silently.
