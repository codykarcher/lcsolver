# TASOPT source notes

Findings about TASOPT 2.16 itself, made while porting it. Numbered to
continue the sequence in `../convexengineering/DISCREPANCIES.md`.

## §22 — `tfoper.f`'s off-design Newton is not robust outside the shipped engine envelope

**Retraction first.** The first version of this section claimed that
`tfoper.f` "fails to converge for almost any off-design perturbation" and
concluded that "any TASOPT result that depends on off-design engine
performance — which is every mission analysis and therefore every sized
aircraft — is computed by a solver that converges at the design point and
fails away from it."

**That conclusion is false and is withdrawn.** It rested entirely on a
synthetic driver of my own construction, and I wrote it down without first
checking it against the shipped program. Doing so takes about two minutes and
refutes it immediately.

### What the real program does

Building TASOPT unmodified and running the shipped `runs/737/737.tas`, with
`tfoper.f` instrumented only to report `max|rrel|` at each successful exit:

```
552 converged tfoper calls in one 737 sizing
  max relative residual at exit:  median 3.7e-13   90th 2.5e-11   max 1.8e-10
  calls with residual > 1e-6:  0
  convergence failures:        0
```

Every call lands on a genuine root, and the sizing converges in 18 outer
iterations to WTO = 174979 lbf, Wfuel = 47487 lbf. The hand-written analytic
Jacobian is fine. **`tfoper.f` works.**

### What is actually true

The iteration is not robust for engine parameters away from the shipped ones.
My driver used an overall pressure ratio of about 30 split as
`pilc = 1.935, pihc = 9.369` — a reasonable-looking OPR, but a split nothing
like the 737's `pilc ≈ 4.75, pihc = 3.75` — and `epfK = 0` where the real case
uses `-0.077`. On that engine, warm-started from its own converged design
point:

* changing `Tt4` in either direction fails to converge;
* changing `p0` **converges to a wrong answer**.

The second matters more, because the convergence test is on the size of the
Newton *step*, not on the residual. A poor Newton direction can produce tiny
steps while the equations stay badly violated, and the routine exits reporting
success. Evaluating the residuals at the state it returns:

```
                    max |relative residual|
  this port         2.2e-11
  tfoper.f          1.9e-01
```

A 19% violation, reported as converged. That case has an exact analytic
answer — changing `p0` with `M0` and `T0` held leaves every corrected quantity
invariant and scales only the physical mass flow — and this port reproduces
the scaling to 3e-12 while `tfoper.f` misses it by 7.9%.

### Why this port behaves differently

`tfoper.py` computes the Jacobian by central differences rather than porting
the source's ~2000 lines of hand-written chain rules. On the shipped engine
that buys nothing: both converge, and they agree to 9e-10 across 124 compared
values. Away from it, it is what lets this port solve cases the original
cannot. It is also why this module's agreement figure is 1e-10 rather than the
1e-13 the closed-form modules reach — the two runs stop at slightly different
points inside the same convergence ball.

### Practical consequence

Much narrower than first claimed. Results from the shipped cases are sound.
The caution is for anyone driving the engine model to a new design — an
unusual compressor split, or an optimiser that wanders into one — where
`tfoper.f` can report success on a state that does not satisfy its own
equations. A residual check at exit would catch that; the step-size test does
not.

## §23 — linking `tfoper.f`

Two things a minimal link needs:

* `-fdollar-ok`, for the `res$` / `a$` debug declarations.
* `compare.f`, which defines `compare(ss, aa, dd)` as uppercase
  `SUBROUTINE COMPARE`. It is not a dependency of any numerical module, so a
  link of `tfoper.f` plus its gas and map dependencies does not pull it in.
  The call sits behind `if (iter .eq. -1)` and never runs, but the reference is
  emitted anyway. Link `compare.f` or supply a stub;
  `fortran_ref/drv_tfoper.f` does the latter.

(An earlier version of this note claimed `compare` existed in no file in the
distribution. Also wrong — the grep behind it was case-sensitive.)

## §24 — `tfcalc.f` stops rather than returning on engine failure

When `tfoper` does fail, `tfcalc.f` responds with a bare `stop` for
`iTFspec = 1`, terminating the program rather than returning a flag. So a
driver that hits the §22 regime gets no output at all, and
`fortran_ref/drv_tfcalc.f` can only produce reference values for the sizing
path. This port raises `TFCalcError` instead, which a caller can catch.

## §25 — `blvar` writes `cd_ue` where it means `cf_ue`

In `blsys.f`, `blvar`'s wake branch is

```fortran
      if(wake) then
       cf    = 0.
       cf_th = 0.
       cf_ds = 0.
       cd_ue = 0.          ! <- cf_ue
```

Under `implicit real (a-h,m,o-z)` this silently declares a new local and
leaves the output argument `cf_ue` alone. Because it is a dummy argument
aliased to the caller's variable, the value **carried over from the last
station before the wake** is reused for every wake station, and reused again
by `di_ue` a few lines down, which is built from `cf_ue`.

Verified rather than inferred: presetting `cf_ue` to `0.1234567890123456`
before the call and running the compiled routine returns it unchanged
(`fortran_ref/drv_blsys.f`, case 3).

**It does not change any answer.** `cf` itself is zero in the wake, so the
residuals are untouched; only `aa(1,3)`, `aa(2,3)` and their `bb` counterparts
— the Jacobian — carry the stale number. A wrong Jacobian costs Newton
iterations, not the root, and `blax` converges anyway. The port reproduces it
by threading the previous `cf_ue` into `blvar`, so the iterate path matches
too.

## §26 — `blax` forms the mass defect two inconsistent ways

`blax.f` closes the initial direct march with

```fortran
        mdi(i) = ue*ds*(b + 2.0*pi*ds)          ! line 255, no rn
```

and the global Newton update with

```fortran
          mdi(i) = uei(i)*dsi(i)*(bi(i) + 2.0*pi*dsi(i)*rni(i))   ! line 596
```

Only the second is the inverse of the quadratic the Newton sweep solves at its
top (`0.5 rn ds^2 + (b/4pi) ds - md/(4 pi ue) = 0`). The march's mass defects
are therefore inconsistent with its own displacement thicknesses wherever
`rn ≠ 1` — that is, everywhere but the cylindrical barrel.

Separately, the Newton's step limiter inverts `md -> ds` **without** the `rn`
division the sweep uses (line 569 against line 394), so `ddsi` is not the
change in `dsi` implied by `dmdi`.

Both are path effects rather than errors in the answer: the march only
supplies a starting guess, and `dsi` feeds nothing but `mdi`, which the next
sweep re-inverts consistently. At the fixed point they cancel. Ported as
written, with the consequence that the port's iterate sequence matches.

## §27 — `blax` reads `cdi(1)` and `phi(1)` before anything writes them

`blax.f` never assigns `cdi(1)`, yet the running-dissipation integral reads it:

```fortran
        dibm = cdi(i-1)*rhi(i-1)*uei(i-1)**3 * (...)
```

at `i = 2`. In the shipped program the array is `cdbl` from the `fbl.inc`
`COMMON` block, so it is statically zero and nothing ever writes it — the
integral is seeded with zero, which is what it should be. But a caller passing
a stack array gets whatever is there.

The initialisation block above has a related slip:

```fortran
      if(xi(1) .eq. 0.0) then
       thi(1) = 0.
       dsi(1) = 0.
       mdi(1) = 0.
       phi(i) = 0.          ! <- phi(1)
```

`i` is the loop variable left over from the `do i = 1, n` loop just above, so
after normal termination this zeroes `phi(n+1)`, not `phi(1)`. Harmless twice
over: `phi(1)` is set to zero before the integral that is actually returned,
and `n < ndim` in every shipped case so the write lands inside the array. It
would run off the end if `n == ndim`.

This port initialises the output arrays to zero explicitly and says why,
rather than inheriting it from storage class.

## §28 — three closure routines in `blsys.f` are dead

`dilw` (laminar wake dissipation), `dit` (turbulent dissipation) and `hct`
(density shape parameter) are called from nowhere in the shipped program.
`blvar`'s `hct` call is commented out and replaced by an inline
`Hc = (gam-1)/2 M_e^2 H`, which is not the same function; the turbulent
dissipation is likewise assembled inline from an equilibrium shear-stress
estimate rather than through `dit`; and `dil` covers the laminar wake as well
as the attached case. The only other call site of `hct` is `blax1.f`, which
the Makefile does not compile.

They are XFOIL's, and correct — this port keeps and tests them — but nothing
in a TASOPT result depends on them.

## §29 — `Wupdate0`'s weight-explosion guard cannot fire

`wsize.f` calls its weight update and then tests the fraction sum:

```fortran
       call Wupdate0(parg,rlx,fsum)
       if(fsum .ge. 1.0) go to 110
```

but `Wupdate0` sets `fsum = 0.0` on entry and never touches it again — the
`ftotadd` it computes just above goes into the `WMTO` division, not into
`fsum`. So the guard after the *first* update in each iteration is dead. The
`Wupdate` at the bottom of the loop does maintain `fsum` properly, so a
diverging weight is still caught, one call later.

Ported as written, with the dead branch kept and commented, because removing
it would hide the asymmetry between the two updates.

## §30 — a non-converged sizing is not an error

At the bottom of the weight loop `wsize.f` prints

```
WSIZE: Weight iteration not converged.  dWrel = ...
```

and then falls through — its `return` is commented out (`cc      return`). So
the takeoff run, the CG limits and the neutral point are all computed from a
state that did not converge, and `Lconv` is the only signal that anything was
wrong. A caller that ignores `Lconv` gets numbers that look ordinary.

This port does the same, and returns `converged` on the result.

## §31 — three routines in `wsize.f` are not called from it

`Wupdate1` (an alternative weight update that splits the fuselage into its
parts) has its only call site, at the top of `Wupdate`, commented out.
`pralt` is called only from `woper.f`. `muair` is called from nowhere at all,
and would not agree with the atmosphere if it were: it takes its reference
temperature as 288.0 K where `atmos` uses 288.2, a 0.07% shift in viscosity.
`cfturb` also lives here and *is* used, by `cdsum`.

## §32 — `woper.f` has an unused relaxation factor and two dead blocks

```fortran
      rlx = 1.0
      if(iterw .gt. iterfmax-5) then
        rlx = 0.5
      endif
```

is the first thing in the weight loop, and `rlx` is never read again. Unlike
`wsize`, `woper` applies no under-relaxation at all — its weight update is
whatever `mission` returns.

Two blocks cannot have an effect:

* `pare` is copied from `pared` for every point and every index at the top of
  the routine, and then copied again inside `if(initeng.eq.0)`. The second
  copy can only reproduce the first, so the `initeng` branch is decorative
  here; `initeng` does still reach `mission` and `tfcalc`, where it means
  something.
* `para(iaCfnace)` is copied from the design mission over
  `ipstatic..ipdescentn`, and then overwritten with a flat `0.003` at *every*
  point about eighty lines later.

All three are ported as written, the last with a test pinning the order.

## §33 — dead input in the `.tas` format

Three things a `.tas` file appears to specify but does not.

**The tail sweep.** `getparm.f` reads it and throws it away one line later:

```fortran
      call getrval(lu,iline,parg(igsweeph))
      parg(igsweeph) = parg(igsweep)   ! ###
```

`737.tas` carries a `sweeph` value with an explanatory comment. Changing it
has no effect; the horizontal tail is always swept like the wing.

**The number of fuselage webs.** `parg(ignfweb)` is hard-wired to 1.0 and its
read is commented out, so a double-bubble section always gets one web.

**The mission-varying excrescence factors.** The three lines that would read
per-mission `fexcdw`/`fexcdt`/`fexcdf` are inside `if(.false.)`, matching a
commented-out block in `woper.f`. Every mission uses the single set read later
in the file.

Two smaller notes on the reader itself:

* The airfoil database index is set to 1, a loop reads that one file, and then
  the count is set to 2 and `airfile(2) = airfile(1)` — so the same file is
  read a second time into a second, identical table. `iairf` is then 1 for the
  design mission and 2 for off-design missions, selecting between two copies
  of the same data. This port reads it once.
* `getrkey`'s header comment says the keyword must be "at the beginning of
  line", but the implementation is `index(line(1:kend), key(1:nkey))`, which
  matches anywhere. Reproduced as implemented.

## §34 — `gradop.f` is an empty shell

`gradop.f` advertises itself as a BFGS minimiser. Its step loop is

```fortran
      do 100 istep = 1, nstepmax
        grms = 0.
        gmax = 0.
        do i = 1, n
          grel = funv(i)*dvopt(i)
          grms = grms + grel**2
        enddo
 100  continue  ! with next step
```

— a gradient norm accumulated and then discarded, with no search direction,
no line search and no Hessian update. It runs `nstepmax` times, changes
nothing, and exits reporting the step limit. Its call site in `tasopt.f` is
commented out beside the `simpop` call that is live, so nothing is lost.

Not ported. `gobj`, the objective-and-gradient wrapper it would have consumed,
*is* ported, since it is well-defined and is what a working gradient method
would need.

## §35 — the optimiser's constraints are penalties, not constraints

`fobj.f` adds `penfac * max(g, 0)**2` for each of the four optional
constraints — balanced field length, fuel volume, span, top-of-climb angle —
with `penfac` between 1 and 25 times the payload weight. Nothing is enforced,
so a converged design can sit slightly outside a constraint, by an amount that
depends on how steep the objective is there.

Three further *ad hoc* penalties are added before any of those, and they shape
the design space rather than describe the aircraft:

* negative sweep, "because the model depends on `cos(sweep)`, so negative
  sweep is invisible";
* inner-panel reverse taper, which "might otherwise look attractive to the
  optimizer for strut-wing cases, because it doesn't know about the download
  requirements";
* excessive tip taper, "to strongly discourage the optimizer from trying a
  negative tip chord as it samples the design space".

All three are the source's own words. They are real modelling decisions, not
numerical guards, and anyone reading an optimised TASOPT design should know
they are there.

## §36 — `voptset` clamps a design variable and tells the optimiser

```fortran
        if(io .eq. iolamt ) then
         parg(iglambdat) = vopt(iv)
         if(parg(iglambdat) .lt. 0.1) then
          parg(iglambdat) = 0.1
          vopt(iv) = 0.1
         endif
        endif
```

The tip taper ratio is floored at 0.1, and the floored value is written back
into the optimiser's own variable vector — so the simplex *vertex* moves, not
just the aircraft built from it. That is unusual: most clamps are invisible to
the search, and a search that cannot see one will keep pushing against it.
Here it can, which means a vertex can be silently merged with another.

## §37 — a swept `OPR` is not the OPR at every point

`tasopt.f`'s `i`/`j` sweep over overall pressure ratio computes

```fortran
        ip = ipcruise1
        km = 1
        pihc = OPR/pare(iepilc,ip,km)
        do km = 1, nmission
          do ip = 1, iptotal
            pare(iepihc,ip,km) = pihc
```

with the per-point form `pihc = OPR/pare(iepilc,ip,km)` commented out just
above it. So the HPC pressure ratio is worked out once, from the
start-of-cruise LPC ratio of the first mission, and written to every point of
every mission — and any point whose LPC ratio differs ends up at a different
overall pressure ratio than the one swept for. Reproduced, with a test.

## §38 — `noise.f` passes the angle of attack in two different units

```fortran
      alpha = 5.0             ! line 187, sideline
      alpha = 5.0 * pi/180.0  ! line 272, cutback
      alpha = 5.0 * pi/180.0  ! line 381, flyover
```

The sideline calculation is handed 5 **radians** — 286 degrees — where the
other two get 5 degrees. It reaches `jet_noise` as `cos(alpha)` in the
effective-velocity and convective-Mach-number terms, and `cos(5 rad) = 0.28`
against `cos(5 deg) = 0.996`, so the sideline jet is computed with a much
weaker flight-velocity correction than the two points either side of it.

Not a rounding matter: reproducing it is what makes all three certification
levels come out right. Ported with the two values named separately so the
asymmetry is visible rather than looking like a typo in the port.

## §39 — `fpfun` is always evaluated at 110 degrees

`tfnoise.f`'s jet spectrum shape takes the corrected directivity angle in
degrees — its own comment says so — and clamps it:

```fortran
      tl = max( 110.0, min( 250.0 , t ) )
```

Its only caller passes `thetap`, which is an angle in **radians** and so never
exceeds about 3.5. `min(250, 3.5)` is 3.5, and `max(110, 3.5)` is 110. The
clamp therefore fires on every call and the jet spectral *shape* is frozen at
the 110-degree column of the table `fpfun` represents, whichever way the
observer lies.

The overall jet level still varies with angle — that is `UOL`, computed
separately — so this affects the distribution of energy across the frequency
bands rather than the total by very much. Reproduced.

## §40 — a convective Mach number its author flagged

```fortran
cc    Mc = 0.62*(u8-u0*cos(alpha))/c0   !%% BUG???
      Mc = 0.62*(u6-u0*cos(alpha))/c0
```

Somebody suspected the *fan* jet should set the convection velocity rather
than the core jet, wrote the alternative down, commented it out, and left the
question mark in. The core-jet form is what ships and is what this port does.

## §41 — the tone series stops one harmonic late

`esdu98008discretetone_total` appends the next blade-passing harmonic *before*
testing whether the current one is past 10 kHz, so the series it returns
always ends above the limit. Whether that last harmonic counts is then an
accident of where it falls: the top third-octave band runs to 11220 Hz, so a
harmonic between 10 and 11.2 kHz is still added in and one above 11.2 kHz is
dropped.

If the blade-passing frequency were low enough for 23 harmonics to fit under
10 kHz the loop would fall out with `nf` one larger than the number of levels
computed, and the caller would read an uninitialised level. No engine of
interest gets near that — a 737 breaks out at the seventh harmonic — and this
port returns only the pairs it computed.

## §42 — the save file is closed before anything is written to it

`tasopt.f` opens the optimiser restart file, writes its two header lines with
`wrtsave0`, and closes it — *before* the optimisation that produces the data
has run:

```fortran
       open(lusav,file=fname,status='unknown')
       call wrtsave0(lusav, ... )
       close(lusav)
      endif
```

Every later `wrtsave1` therefore writes to a closed unit 8, which gfortran
silently redirects to a file called `fort.8` in the working directory. Running
the 737 with `Lopt = T` and `Lsavwrite = T` produces:

* `737.sav` — two header lines and nothing else;
* `fort.8` — every simplex, unnamed and unmentioned.

Reading the `.sav` back gets no grid points at all and fails `getsave`'s
i/j-count check, so a restart cannot work. The bug is in the driver, not in
the `getsave`/`wrtsave` routines, and this port does not reproduce it — it
writes both halves to the same file. Both halves are checked against the real
Fortran output: the header against the `.sav`, the body against the `fort.8`.

## §43 — `pltwrt`'s PFEI column depends on the report having been written

The Matlab parameter file's first column is `parg(igPFEI)`, and the only place
that is ever assigned outside the optimiser is inside `outwrt` — the routine
that writes the `.out` report. `outwrt` happens to be called before `pltwrt`
in the `i`,`j` loop, so the column is populated.

Turn `Loutwrite` off and leave `Lplot`-style output on, and the column carries
the `2**1023` "unset" fill value instead. Ported as written, with the ordering
made explicit in `tasopt_py.output.report`.

## §44 — the drawn planform is pinned differently from the modelled one

`airpic.f` places the spanwise axis at a hard-wired 40% chord
(`xax = 0.40`, `xaxh = 0.40`) for both wing and tail. `parg(igXaxis)` — what
`surfcm` and the structural sizing use — is never read. The 737 sets `Xaxis`
to 0.40 as well so the two coincide there, but on any case that does not, the
picture and the model disagree about where the surfaces are pinned.

`airpic` also computes `clp` and `cmp` from a hard-wired `CL = 0.70` and
`cm = -0.1` and then never uses either, and a block that would walk the tail
root back along the fuselage contour sits inside `if(.false.)`.
