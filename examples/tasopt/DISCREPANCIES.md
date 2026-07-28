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
