
## §22 — `tfoper.f`'s Newton iteration diverges for almost any off-design point

`tfoper` is the off-design engine solver: nine unknowns, nine residuals, a
hand-differentiated 9x9 Jacobian, Newton with step limiting, tolerance 1e-10
on the largest relative step, 50 iterations.

Driven from a sized CFM56-class engine (`fortran_ref/drv_tfoper.f`), warm-started
from its own converged design point — which is how `wsize.f` and `mission.f` drive
it — it fails to converge for nearly every perturbation:

| perturbation from design | shipped `tfoper.f` | this port |
|---|---|---|
| none (re-solve at design)      | converges  | converges, 1 iteration |
| `Tt4` 1450 -> 1400             | **fails**  | converges, 4 iterations |
| `Tt4` 1450 -> 1500             | **fails**  | converges, 5 iterations |
| `Tt4` 1450 -> 1350             | **fails**  | converges |
| `p0` 23842 -> 30000            | **fails**  | converges, 4 iterations |
| `p0` 23842 -> 24500            | **fails**  | converges |
| `M0` 0.80 -> 0.78              | **fails**  | converges, 4 iterations |
| `T0` 219.4 -> 225              | **fails**  | converges |
| `iTFspec=2`, `Feng` 25k -> 23k | converges  | converges, agrees to 2e-10 |
| `iTFspec=2`, `Feng` 25k -> 27k | **fails**  | converges |
| cooled + offtakes, `Tt4` 1500  | converges  | converges, agrees to 9e-10 |

### It is a slow divergence, not a blow-up

Turning on the routine's own `Lprint` shows the iteration sitting *next to* the
solution and walking away from it. For `p0 = 30000`, from iteration 10 onward:

```
iter 10   dmax = 2.9713e-05    R8 = -2.4113e-04    dPc = -2.0357
iter 11   dmax = 3.1609e-05    R8 = -2.5655e-04    dPc = -2.1661
iter 12   dmax = 3.3550e-05    R8 = -2.7284e-04    dPc = -2.3000
```

Every step is ~6% larger than the last, so it grows geometrically until iteration
50 and reports failure. `rlx` is 1.0 throughout — no step limit is active, so the
limiter is not the cause. A correct Jacobian at a residual of 2.4e-04 would take
one step to 1e-8 and stop. The design point converges only because the residuals
are already zero there and the loop exits before its first update.

### Independent evidence that the physics is right and the Jacobian is not

The `p0 = 30000` case has an exact analytic answer. Holding `M0` and `T0`, `pt0`
is proportional to `p0`, and so is every total pressure down the gas path. Every
unknown except `Pc` is a corrected quantity normalised by those pressures, so the
solution is *identical* and only `Pc` and the physical mass flow scale. This port
reproduces that to 1e-11:

```
mcore ratio  1.25828370106045     p0 ratio  1.25828370103179
pif   1.68500000000000 vs 1.68500000006646     rel 3.9e-11
pilc  1.93500000000000 vs 1.93500000008953     rel 4.6e-11
pihc  9.36900000000000 vs 9.36899999975989     rel 2.6e-11
TSFC  1.8731315702289e-04 vs 1.8731315701840e-04
```

So a solution exists, it is trivially reachable, and the residual functions agree
between the two implementations — on the cases the Fortran *does* converge on, the
two agree to 9e-10 across all 124 compared values. What differs is only the
Jacobian: hand-coded there, central differences here. That isolates the defect to
the derivative bookkeeping, which is roughly two thirds of the file's 3400 lines.

I have not located which of the several hundred chain-rule terms is wrong, and
did not try — the port does not need it.

### Consequence

Any TASOPT result that depends on off-design engine performance — which is every
mission analysis and therefore every sized aircraft — is computed by a solver
that converges at the design point and fails away from it. `mission.f` marches in
small steps, which is presumably why this is survivable in practice: over a short
enough step the iteration may reach tolerance before the drift takes over. It
also means results may be sensitive to segment count in a way that has nothing to
do with the physics.

### Two things that block linking `tfoper.f` at all

* It declares `res$` and `a$` for a debug block, so it needs `-fdollar-ok`.
* It calls `compare(ss, aa, dd)`, which exists in **no source file in the
  distribution**. The call sits behind `if (iter .eq. -1)` and so never runs, but
  the reference to it is emitted regardless and the link fails without a stub.
  `fortran_ref/drv_tfoper.f` supplies one.
